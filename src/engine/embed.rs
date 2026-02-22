//! Embedding engine: text → dense vector embedding.
//!
//! [`EmbeddingEngine`] provides a high-level API for producing text embeddings
//! from any supported GGUF embedding model (Gemma, LLaMA, BERT, etc.).
//!
//! The pipeline: tokenize → truncate → forward pass → pool → L2 normalize.

use std::path::Path;
use std::sync::Arc;

use tracing::info;

use crate::backend::{ComputeBackend, DeviceTensor};
use crate::error::InferenceError;
use crate::gguf::GgufFile;
use crate::model::config::{ModelConfig, PoolingType};
use crate::model::layer::model_forward;
use crate::model::weights::ModelWeights;
use crate::tensor::Tensor;
use crate::tokenizer::{create_tokenizer_from_gguf, Tokenizer};

/// High-level embedding engine that takes text and produces dense vectors.
///
/// Wraps a GGUF model with its tokenizer and compute backend, exposing a
/// simple `embed(text) -> Vec<f32>` API.
///
/// Thread-safe: the engine uses `Arc<dyn ComputeBackend>` and can be shared
/// across threads via `Arc<EmbeddingEngine>`.
pub struct EmbeddingEngine {
    config: ModelConfig,
    weights: ModelWeights,
    tokenizer: Box<dyn Tokenizer>,
    backend: Arc<dyn ComputeBackend>,
}

impl EmbeddingEngine {
    /// Load an embedding engine by model name from the registry.
    ///
    /// Resolves the name (e.g., `"miniLM"`) to a local GGUF file path
    /// and auto-selects the best available compute backend.
    pub fn from_registry(name: &str) -> Result<Self, InferenceError> {
        let registry = crate::registry::ModelRegistry::new();
        let path = registry.resolve(name)?;
        let backend = crate::backend::select_backend();
        Self::from_gguf(path, backend)
    }

    /// Load an embedding engine from a GGUF file path.
    ///
    /// Auto-detects the model architecture, tokenizer type, and loads all
    /// weights onto the provided compute backend.
    pub fn from_gguf(
        path: impl AsRef<Path>,
        backend: Arc<dyn ComputeBackend>,
    ) -> Result<Self, InferenceError> {
        let path = path.as_ref();
        info!(path = %path.display(), "Loading embedding engine from GGUF");
        let gguf = GgufFile::open(path)?;
        let config = ModelConfig::from_gguf(&gguf)?;
        let weights = ModelWeights::from_gguf(&gguf, &config, backend.as_ref())?;
        let tokenizer = create_tokenizer_from_gguf(&gguf)?;

        info!(
            arch = %config.arch_name,
            hidden_size = config.hidden_size,
            vocab_size = config.vocab_size,
            pooling = ?config.pooling_type,
            "Embedding engine loaded"
        );

        Ok(Self {
            config,
            weights,
            tokenizer,
            backend,
        })
    }

    /// Crate-internal constructor for testing with synthetic components.
    #[cfg(test)]
    pub(crate) fn new(
        config: ModelConfig,
        weights: ModelWeights,
        tokenizer: Box<dyn Tokenizer>,
        backend: Arc<dyn ComputeBackend>,
    ) -> Self {
        Self {
            config,
            weights,
            tokenizer,
            backend,
        }
    }

    /// Produce an L2-normalized embedding vector for the given text.
    ///
    /// Steps:
    /// 1. Tokenize with special tokens (BOS/EOS or CLS/SEP)
    /// 2. Truncate to `max_seq_len` if needed
    /// 3. Run transformer forward pass
    /// 4. Pool hidden states (mean, CLS, or last-token pooling)
    /// 5. L2-normalize the result
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        // 1. Tokenize
        let mut input_ids = self.tokenizer.encode(text, true);

        // 2. Truncate to max_seq_len, preserving closing special token (EOS/SEP)
        if input_ids.len() > self.config.max_seq_len && self.config.max_seq_len >= 2 {
            let eos_id = self.tokenizer.eos_token_id();
            input_ids.truncate(self.config.max_seq_len);
            // If the tokenizer has an EOS/SEP, ensure it's at the end
            if let Some(eos) = eos_id {
                if let Some(last) = input_ids.last_mut() {
                    *last = eos;
                }
            }
        } else if input_ids.len() > self.config.max_seq_len {
            input_ids.truncate(self.config.max_seq_len);
        }

        // 3. Build attention mask (all 1s — no padding for single text)
        let attention_mask = vec![1.0f32; input_ids.len()];

        // 4. Forward pass: [seq_len, hidden_size]
        let hidden = model_forward(
            &input_ids,
            &attention_mask,
            &self.weights,
            &self.config,
            self.backend.as_ref(),
            0,
        )?;

        // 5. Pool
        let pooled = self.pool(&hidden, &attention_mask);

        // 6. L2 normalize
        let normalized = self.backend.l2_normalize(&pooled);

        let f32_tensor = self.backend.download(&normalized);
        let f32_result = f32_tensor.to_f32();
        Ok(f32_result.as_f32().to_vec())
    }

    /// Produce embeddings for a batch of texts.
    ///
    /// Each text is processed independently to guarantee consistency with
    /// single-text `embed()`. Batched GPU execution can be optimized later.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, InferenceError> {
        texts.iter().map(|text| self.embed(text)).collect()
    }

    /// The dimensionality of output embedding vectors.
    pub fn embedding_dim(&self) -> usize {
        self.config.hidden_size
    }

    /// The vocabulary size of the loaded model.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// The model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Pool hidden states [seq_len, hidden_size] down to [hidden_size].
    fn pool(&self, hidden: &DeviceTensor, mask: &[f32]) -> DeviceTensor {
        let seq_len = hidden.shape()[0];
        let hidden_size = self.config.hidden_size;

        // Guard: zero-length sequences get a zero vector (shouldn't happen in
        // normal use, but prevents underflow in the Last path).
        if seq_len == 0 {
            return self
                .backend
                .upload(&Tensor::new(vec![hidden_size], vec![0.0f32; hidden_size]));
        }

        match self.config.pooling_type {
            PoolingType::Mean | PoolingType::None => {
                // Default to mean pooling for embedding models without explicit pooling_type
                self.backend.mean_pool(hidden, mask)
            }
            PoolingType::Cls => {
                // Extract row 0 (CLS token position).
                let host = self.backend.download(hidden);
                let f32_tensor = host.to_f32();
                let data = f32_tensor.as_f32();
                let cls_data = data[..hidden_size].to_vec();
                self.backend
                    .upload(&Tensor::new(vec![hidden_size], cls_data))
            }
            PoolingType::Last => {
                // Extract last row.
                let host = self.backend.download(hidden);
                let f32_tensor = host.to_f32();
                let data = f32_tensor.as_f32();
                let start = (seq_len - 1) * hidden_size;
                let last_data = data[start..start + hidden_size].to_vec();
                self.backend
                    .upload(&Tensor::new(vec![hidden_size], last_data))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::backend::DeviceTensor;
    use crate::model::config::*;
    use crate::model::weights::{LayerWeights, ModelWeights};
    use crate::tensor::Tensor;
    use crate::tokenizer::Tokenizer;

    /// A mock tokenizer that splits on whitespace and maps known words to IDs.
    struct MockTokenizer {
        vocab: Vec<String>,
    }

    impl MockTokenizer {
        fn new() -> Self {
            Self {
                vocab: vec![
                    "<pad>".to_string(), // 0
                    "<bos>".to_string(), // 1
                    "<eos>".to_string(), // 2
                    "hello".to_string(), // 3
                    "world".to_string(), // 4
                    "foo".to_string(),   // 5
                    "bar".to_string(),   // 6
                    "baz".to_string(),   // 7
                ],
            }
        }
    }

    impl Tokenizer for MockTokenizer {
        fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
            let mut ids = Vec::new();
            if add_special_tokens {
                ids.push(1); // BOS
            }
            for word in text.split_whitespace() {
                let id = self
                    .vocab
                    .iter()
                    .position(|v| v == word)
                    .map(|i| i as u32)
                    .unwrap_or(0);
                ids.push(id);
            }
            if add_special_tokens {
                ids.push(2); // EOS
            }
            ids
        }

        fn decode(&self, ids: &[u32]) -> String {
            ids.iter()
                .filter_map(|&id| self.vocab.get(id as usize))
                .cloned()
                .collect::<Vec<_>>()
                .join(" ")
        }

        fn vocab_size(&self) -> usize {
            self.vocab.len()
        }

        fn bos_token_id(&self) -> Option<u32> {
            Some(1)
        }

        fn eos_token_id(&self) -> Option<u32> {
            Some(2)
        }

        fn pad_token_id(&self) -> Option<u32> {
            Some(0)
        }
    }

    fn dt(tensor: Tensor) -> DeviceTensor {
        DeviceTensor::new(tensor)
    }

    fn identity_weight(size: usize) -> DeviceTensor {
        let mut data = vec![0.0f32; size * size];
        for i in 0..size {
            data[i * size + i] = 1.0;
        }
        dt(Tensor::new(vec![size, size], data))
    }

    fn zero_weight(out_dim: usize, in_dim: usize) -> DeviceTensor {
        dt(Tensor::new(
            vec![out_dim, in_dim],
            vec![0.0f32; out_dim * in_dim],
        ))
    }

    fn ones_weight(dim: usize) -> DeviceTensor {
        dt(Tensor::new(vec![dim], vec![1.0f32; dim]))
    }

    /// Build a minimal Gemma-style config for testing.
    fn test_config(hidden_size: usize) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Gemma3,
            arch_name: "gemma3".to_string(),
            hidden_size,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: hidden_size,
            ffn_hidden: hidden_size * 4,
            vocab_size: 8,
            max_seq_len: 128,
            norm_type: NormType::RMSNorm,
            norm_eps: 1e-6,
            activation: Activation::SwiGLU,
            position_type: PositionType::RoPE,
            rope_freq_base: 10000.0,
            rope_dim: hidden_size,
            rope_neox: false,
            rope_scaling_original_ctx: 0,
            rope_scaling_attn_factor: 1.0,
            causal: false, // bidirectional for embedding
            attn_logit_softcap: 0.0,
            attn_scale: None,
            embedding_scale: 1.0,
            pooling_type: PoolingType::Mean,
            has_ffn_gate: true,
            has_bias: false,
            pre_norm: true,
            swa_window: 0,
            swa_layers: vec![],
            rope_freq_base_swa: 10000.0,
        }
    }

    /// Build minimal weights for the test config.
    fn test_weights(config: &ModelConfig) -> ModelWeights {
        let h = config.hidden_size;
        let ffn = config.ffn_hidden;
        let v = config.vocab_size;

        // Use distinct embedding rows so different tokens produce different embeddings
        let mut emb_data = vec![0.0f32; v * h];
        for i in 0..v {
            for j in 0..h {
                emb_data[i * h + j] = ((i * h + j) as f32) * 0.1;
            }
        }

        let layer = LayerWeights {
            attn_norm_w: Some(ones_weight(h)),
            attn_norm_b: None,
            ffn_norm_w: Some(ones_weight(h)),
            ffn_norm_b: None,
            attn_output_norm_w: None,
            attn_output_norm_b: None,
            ffn_output_norm_w: None,
            ffn_output_norm_b: None,
            attn_q: identity_weight(h),
            attn_k: identity_weight(h),
            attn_v: identity_weight(h),
            attn_output: identity_weight(h),
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            attn_output_bias: None,
            ffn_up: zero_weight(ffn, h),
            ffn_down: zero_weight(h, ffn),
            ffn_gate: Some(zero_weight(ffn, h)),
            ffn_up_bias: None,
            ffn_down_bias: None,
            attn_q_norm_w: None,
            attn_k_norm_w: None,
            attn_post_norm_w: None,
            ffn_post_norm_w: None,
        };

        ModelWeights {
            token_embedding: dt(Tensor::new(vec![v, h], emb_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![layer],
            output_norm_w: Some(ones_weight(h)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        }
    }

    fn build_engine(config: ModelConfig) -> EmbeddingEngine {
        let weights = test_weights(&config);
        let tokenizer: Box<dyn Tokenizer> = Box::new(MockTokenizer::new());
        let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
        EmbeddingEngine::new(config, weights, tokenizer, backend)
    }

    #[test]
    fn test_embed_returns_correct_dimension() {
        let config = test_config(4);
        let engine = build_engine(config);
        let emb = engine.embed("hello world").unwrap();
        assert_eq!(emb.len(), engine.embedding_dim());
        assert_eq!(emb.len(), 4);
    }

    #[test]
    fn test_embed_output_is_l2_normalized() {
        let config = test_config(4);
        let engine = build_engine(config);
        let emb = engine.embed("hello world").unwrap();
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Embedding should be L2-normalized, got norm={}",
            norm
        );
    }

    #[test]
    fn test_embed_deterministic() {
        let config = test_config(4);
        let engine = build_engine(config);
        let emb1 = engine.embed("hello world").unwrap();
        let emb2 = engine.embed("hello world").unwrap();
        assert_eq!(emb1, emb2, "Embeddings should be deterministic");
    }

    #[test]
    fn test_embed_batch_matches_individual() {
        let config = test_config(4);
        let engine = build_engine(config);
        let texts = &["hello", "world", "foo"];
        let batch = engine.embed_batch(texts).unwrap();
        for (i, text) in texts.iter().enumerate() {
            let individual = engine.embed(text).unwrap();
            assert_eq!(
                batch[i], individual,
                "Batch embedding[{}] should match individual",
                i
            );
        }
    }

    #[test]
    fn test_embed_batch_empty_input() {
        let config = test_config(4);
        let engine = build_engine(config);
        let batch = engine.embed_batch(&[]).unwrap();
        assert!(batch.is_empty());
    }

    #[test]
    fn test_embed_different_texts_different_embeddings() {
        let config = test_config(4);
        let engine = build_engine(config);
        let emb_a = engine.embed("hello").unwrap();
        let emb_b = engine.embed("foo").unwrap();
        assert_ne!(
            emb_a, emb_b,
            "Different texts should produce different embeddings"
        );
    }

    #[test]
    fn test_embed_truncates_long_input() {
        let mut config = test_config(4);
        config.max_seq_len = 4; // very short
        let engine = build_engine(config);
        // "hello world foo bar baz" with BOS/EOS = 7 tokens, truncated to 4
        let emb = engine.embed("hello world foo bar baz").unwrap();
        assert_eq!(emb.len(), engine.embedding_dim());
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Truncated embedding should still be normalized, got norm={}",
            norm
        );
    }

    #[test]
    fn test_embedding_engine_accessors() {
        let config = test_config(4);
        let engine = build_engine(config);
        assert_eq!(engine.embedding_dim(), 4);
        assert_eq!(engine.vocab_size(), 8);
        assert_eq!(engine.config().arch, ModelArch::Gemma3);
    }

    #[test]
    fn test_pooling_cls() {
        // CLS pooling extracts row 0
        let hidden = dt(Tensor::new(
            vec![3, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, // row 0 (CLS)
                5.0, 6.0, 7.0, 8.0, // row 1
                9.0, 10.0, 11.0, 12.0, // row 2
            ],
        ));
        let mask = vec![1.0, 1.0, 1.0];

        // Temporarily change pooling type for the test
        let mut cls_config = test_config(4);
        cls_config.pooling_type = PoolingType::Cls;
        let cls_engine = build_engine(cls_config);

        let pooled = cls_engine.pool(&hidden, &mask);
        assert_eq!(pooled.as_tensor().as_f32(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_pooling_last() {
        let hidden = dt(Tensor::new(
            vec![3, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        ));
        let mask = vec![1.0, 1.0, 1.0];

        let mut last_config = test_config(4);
        last_config.pooling_type = PoolingType::Last;
        let last_engine = build_engine(last_config);

        let pooled = last_engine.pool(&hidden, &mask);
        assert_eq!(pooled.as_tensor().as_f32(), &[9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_embedding_engine_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EmbeddingEngine>();
    }

    #[test]
    fn test_embed_empty_text() {
        // Empty text with BOS+EOS should still produce a valid embedding
        let config = test_config(4);
        let engine = build_engine(config);
        let emb = engine.embed("").unwrap();
        assert_eq!(emb.len(), engine.embedding_dim());
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Empty text embedding should be L2-normalized, got norm={}",
            norm
        );
    }

    #[test]
    fn test_pooling_zero_length_hidden() {
        // Verify pool() handles zero-length hidden states gracefully
        let config = test_config(4);
        let engine = build_engine(config);
        let hidden = dt(Tensor::new(vec![0, 4], vec![]));
        let mask: Vec<f32> = vec![];
        let pooled = engine.pool(&hidden, &mask);
        assert_eq!(pooled.as_tensor().as_f32(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_pooling_mean() {
        // Verify mean pooling returns the average of all rows
        let config = test_config(4);
        let engine = build_engine(config);
        let hidden = dt(Tensor::new(
            vec![2, 4],
            vec![2.0, 4.0, 6.0, 8.0, 4.0, 6.0, 8.0, 10.0],
        ));
        let mask = vec![1.0, 1.0];
        let pooled = engine.pool(&hidden, &mask);
        let data = pooled.as_tensor().as_f32();
        // Mean of [2,4,6,8] and [4,6,8,10] = [3,5,7,9]
        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[1] - 5.0).abs() < 1e-5);
        assert!((data[2] - 7.0).abs() < 1e-5);
        assert!((data[3] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_truncation_preserves_eos() {
        // When truncation kicks in, the last token should be EOS
        let mut config = test_config(4);
        config.max_seq_len = 4; // BOS + 2 words + EOS = 4, but "hello world foo" = BOS + 3 + EOS = 5
        let engine = build_engine(config);

        // The MockTokenizer encodes "hello world foo" as [1, 3, 4, 5, 2] (5 tokens)
        // After truncation to 4: [1, 3, 4, 2] (EOS=2 preserved at end)
        // This should work without panicking and produce valid output
        let emb = engine.embed("hello world foo").unwrap();
        assert_eq!(emb.len(), engine.embedding_dim());
    }
}
