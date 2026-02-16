//! Generation engine: prompt → text via autoregressive decoding.
//!
//! [`GenerationEngine`] provides a high-level API for text generation from
//! any supported causal GGUF model (Gemma, LLaMA, etc.) using a KV cache
//! for efficient token-by-token decoding.

use std::path::Path;
use std::sync::Arc;

use tracing::{debug, info};

use crate::backend::{ComputeBackend, DeviceTensor};
use crate::error::InferenceError;
use crate::gguf::GgufFile;
use crate::model::cache::KvCache;
use crate::model::config::ModelConfig;
use crate::model::layer::{linear_forward, model_forward_step};
use crate::model::weights::ModelWeights;
use crate::tensor::Tensor;
use crate::tokenizer::{create_tokenizer_from_gguf, Tokenizer};

use super::sampler::{SamplingConfig, XorShiftRng, sample_token};

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Generated an end-of-sequence or explicit stop token.
    StopToken,
    /// Reached the `max_tokens` limit.
    MaxTokens,
    /// Filled the model's context window.
    ContextLength,
    /// Streaming callback returned `false`.
    Cancelled,
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopReason::StopToken => write!(f, "eos"),
            StopReason::MaxTokens => write!(f, "max_tokens"),
            StopReason::ContextLength => write!(f, "context_length"),
            StopReason::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Output from text generation, including metadata.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    /// Generated token IDs (not including the prompt).
    pub token_ids: Vec<u32>,
    /// Why generation stopped.
    pub stop_reason: StopReason,
    /// Number of prompt tokens (for timing calculations).
    pub prompt_tokens: usize,
}

/// Configuration for text generation.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate (excluding the prompt).
    pub max_tokens: usize,
    /// Stop generation when any of these token IDs are produced.
    pub stop_tokens: Vec<u32>,
    /// Sampling parameters (temperature, top-k, top-p).
    pub sampling: SamplingConfig,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            stop_tokens: Vec::new(),
            sampling: SamplingConfig::default(),
        }
    }
}

/// High-level generation engine for autoregressive text generation.
///
/// Wraps a GGUF model with its tokenizer, compute backend, and KV cache
/// to provide `generate(prompt) -> String` and streaming APIs.
pub struct GenerationEngine {
    config: ModelConfig,
    weights: ModelWeights,
    tokenizer: Box<dyn Tokenizer>,
    backend: Arc<dyn ComputeBackend>,
}

impl GenerationEngine {
    /// Load a generation engine from a GGUF file path.
    ///
    /// Validates that the model is causal (generation doesn't make sense
    /// for bidirectional models like BERT).
    pub fn from_gguf(
        path: impl AsRef<Path>,
        backend: Arc<dyn ComputeBackend>,
    ) -> Result<Self, InferenceError> {
        let path = path.as_ref();
        info!(path = %path.display(), "Loading generation engine from GGUF");
        let gguf = GgufFile::open(path)?;
        let config = ModelConfig::from_gguf(&gguf)?;

        if !config.causal {
            return Err(InferenceError::Generation(
                "generation requires a causal model (not bidirectional)".to_string(),
            ));
        }

        let weights = ModelWeights::from_gguf(&gguf, &config, backend.as_ref())?;
        let tokenizer = create_tokenizer_from_gguf(&gguf)?;

        info!(
            arch = %config.arch_name,
            hidden_size = config.hidden_size,
            vocab_size = config.vocab_size,
            max_seq_len = config.max_seq_len,
            "Generation engine loaded"
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

    /// Generate text from a prompt.
    ///
    /// Steps:
    /// 1. Tokenize the prompt
    /// 2. Validate prompt length
    /// 3. Create KV cache
    /// 4. Prefill: run all prompt tokens through the model
    /// 5. Sample first token from prefill output
    /// 6. Decode loop: generate tokens one at a time until stop condition
    /// 7. Decode generated token IDs to text
    pub fn generate(
        &self,
        prompt: &str,
        gen_config: &GenerationConfig,
    ) -> Result<String, InferenceError> {
        if !self.config.causal {
            return Err(InferenceError::Generation(
                "generation requires a causal model (not bidirectional)".to_string(),
            ));
        }

        let prompt_ids = self.tokenizer.encode(prompt, true);

        if prompt_ids.is_empty() {
            return Err(InferenceError::Generation(
                "prompt tokenized to empty sequence".to_string(),
            ));
        }

        if prompt_ids.len() >= self.config.max_seq_len {
            return Err(InferenceError::Generation(format!(
                "prompt length ({}) exceeds max_seq_len ({})",
                prompt_ids.len(),
                self.config.max_seq_len
            )));
        }

        let mut rng = XorShiftRng::new(gen_config.sampling.seed.unwrap_or(42));
        let mut cache = KvCache::new(&self.config);
        let mut generated_ids: Vec<u32> = Vec::new();

        // Collect all stop tokens (explicit + EOS)
        let mut stop_tokens = gen_config.stop_tokens.clone();
        if let Some(eos) = self.tokenizer.eos_token_id() {
            if !stop_tokens.contains(&eos) {
                stop_tokens.push(eos);
            }
        }

        // Prefill: process all prompt tokens at once
        let hidden = model_forward_step(
            &prompt_ids,
            &self.weights,
            &self.config,
            self.backend.as_ref(),
            &mut cache,
        )?;

        // Project last hidden state to logits and sample first token
        let logits = self.project_to_logits(&hidden, prompt_ids.len() - 1);
        let mut next_token = sample_token(&logits, &gen_config.sampling, &mut rng);

        debug!(first_token = next_token, "Prefill complete, starting decode");

        // Decode loop
        for step in 0..gen_config.max_tokens {
            // Check stop conditions
            if stop_tokens.contains(&next_token) {
                debug!(step, token = next_token, "Stop token encountered");
                break;
            }

            generated_ids.push(next_token);

            if cache.len() >= self.config.max_seq_len {
                debug!(step, "Context length reached");
                break;
            }

            // Forward pass with single token
            let hidden = model_forward_step(
                &[next_token],
                &self.weights,
                &self.config,
                self.backend.as_ref(),
                &mut cache,
            )?;

            // Project to logits and sample
            let logits = self.project_to_logits(&hidden, 0);
            next_token = sample_token(&logits, &gen_config.sampling, &mut rng);
        }

        // Decode generated token IDs to text
        Ok(self.tokenizer.decode(&generated_ids))
    }

    /// Generate text with full metadata (stop reason, prompt token count).
    ///
    /// Same as [`generate`] but returns a [`GenerationOutput`] with metadata
    /// useful for timing and reporting.
    pub fn generate_full(
        &self,
        prompt: &str,
        gen_config: &GenerationConfig,
    ) -> Result<GenerationOutput, InferenceError> {
        self.generate_stream_full(prompt, gen_config, |_| true)
    }

    /// Generate text with streaming: invokes callback for each generated token.
    ///
    /// The callback receives each token ID as it's generated. Return `false`
    /// from the callback to stop generation early.
    pub fn generate_stream(
        &self,
        prompt: &str,
        gen_config: &GenerationConfig,
        callback: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>, InferenceError> {
        let output = self.generate_stream_full(prompt, gen_config, callback)?;
        Ok(output.token_ids)
    }

    /// Generate with streaming and full metadata (stop reason, prompt token count).
    pub fn generate_stream_full(
        &self,
        prompt: &str,
        gen_config: &GenerationConfig,
        mut callback: impl FnMut(u32) -> bool,
    ) -> Result<GenerationOutput, InferenceError> {
        if !self.config.causal {
            return Err(InferenceError::Generation(
                "generation requires a causal model (not bidirectional)".to_string(),
            ));
        }

        let prompt_ids = self.tokenizer.encode(prompt, true);
        let prompt_tokens = prompt_ids.len();

        if prompt_ids.is_empty() {
            return Err(InferenceError::Generation(
                "prompt tokenized to empty sequence".to_string(),
            ));
        }

        if prompt_ids.len() >= self.config.max_seq_len {
            return Err(InferenceError::Generation(format!(
                "prompt length ({}) exceeds max_seq_len ({})",
                prompt_ids.len(),
                self.config.max_seq_len
            )));
        }

        let mut rng = XorShiftRng::new(gen_config.sampling.seed.unwrap_or(42));
        let mut cache = KvCache::new(&self.config);
        let mut generated_ids: Vec<u32> = Vec::new();

        let mut stop_tokens = gen_config.stop_tokens.clone();
        if let Some(eos) = self.tokenizer.eos_token_id() {
            if !stop_tokens.contains(&eos) {
                stop_tokens.push(eos);
            }
        }

        // Prefill
        let hidden = model_forward_step(
            &prompt_ids,
            &self.weights,
            &self.config,
            self.backend.as_ref(),
            &mut cache,
        )?;

        let logits = self.project_to_logits(&hidden, prompt_ids.len() - 1);
        let mut next_token = sample_token(&logits, &gen_config.sampling, &mut rng);

        // Decode loop — track why we stopped
        let mut stop_reason = StopReason::MaxTokens; // default if loop exhausts

        for _ in 0..gen_config.max_tokens {
            if stop_tokens.contains(&next_token) {
                stop_reason = StopReason::StopToken;
                break;
            }

            generated_ids.push(next_token);

            if !callback(next_token) {
                stop_reason = StopReason::Cancelled;
                break;
            }

            if cache.len() >= self.config.max_seq_len {
                stop_reason = StopReason::ContextLength;
                break;
            }

            let hidden = model_forward_step(
                &[next_token],
                &self.weights,
                &self.config,
                self.backend.as_ref(),
                &mut cache,
            )?;

            let logits = self.project_to_logits(&hidden, 0);
            next_token = sample_token(&logits, &gen_config.sampling, &mut rng);
        }

        Ok(GenerationOutput {
            token_ids: generated_ids,
            stop_reason,
            prompt_tokens,
        })
    }

    /// Project a hidden state row to vocabulary logits.
    ///
    /// Uses `output_projection` weight if available, otherwise falls back to
    /// tied embeddings (`token_embedding`).
    fn project_to_logits(&self, hidden: &DeviceTensor, row: usize) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let hidden_host = self.backend.download(hidden);
        let hidden_data = hidden_host.as_f32();
        let start = row * hidden_size;
        let row_data = &hidden_data[start..start + hidden_size];

        let row_tensor = self.backend.upload(
            &Tensor::new(vec![1, hidden_size], row_data.to_vec()),
        );

        // Use output_projection if available, otherwise tied embeddings
        let proj_weight = self.weights.output_projection.as_ref()
            .unwrap_or(&self.weights.token_embedding);

        let logits_tensor = linear_forward(&row_tensor, proj_weight, None, self.backend.as_ref());
        let logits_host = self.backend.download(&logits_tensor);
        logits_host.as_f32().to_vec()
    }

    /// The model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The vocabulary size of the loaded model.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids)
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

    /// A mock tokenizer for generation tests.
    ///
    /// Maps token IDs to single-character strings for easy verification.
    struct MockTokenizer {
        vocab: Vec<String>,
        eos_id: u32,
    }

    impl MockTokenizer {
        fn new() -> Self {
            Self {
                vocab: vec![
                    "<pad>".to_string(),  // 0
                    "<bos>".to_string(),  // 1
                    "<eos>".to_string(),  // 2
                    "hello".to_string(),  // 3
                    "world".to_string(),  // 4
                    "foo".to_string(),    // 5
                    "bar".to_string(),    // 6
                    "baz".to_string(),    // 7
                ],
                eos_id: 2,
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
            // Note: no EOS added for generation (model generates EOS)
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
            Some(self.eos_id)
        }

        fn pad_token_id(&self) -> Option<u32> {
            Some(0)
        }
    }

    /// A mock tokenizer with no EOS token, for testing max_tokens without
    /// EOS interference.
    struct NoEosTokenizer;

    impl NoEosTokenizer {
        fn new() -> Self {
            Self
        }
    }

    impl Tokenizer for NoEosTokenizer {
        fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
            let mut ids = Vec::new();
            if add_special_tokens {
                ids.push(1); // BOS
            }
            for word in text.split_whitespace() {
                // Simple hash to token ID (3-7 range, avoiding 0-2)
                let id = 3 + (word.len() as u32 % 5);
                ids.push(id);
            }
            ids
        }

        fn decode(&self, ids: &[u32]) -> String {
            ids.iter().map(|id| format!("t{}", id)).collect::<Vec<_>>().join(" ")
        }

        fn vocab_size(&self) -> usize { 8 }
        fn bos_token_id(&self) -> Option<u32> { Some(1) }
        fn eos_token_id(&self) -> Option<u32> { None } // No EOS
        fn pad_token_id(&self) -> Option<u32> { Some(0) }
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

    /// Build a causal Gemma-style config for generation tests.
    fn gen_config(hidden_size: usize) -> ModelConfig {
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
            max_seq_len: 64,
            norm_type: NormType::RMSNorm,
            norm_eps: 1e-6,
            activation: Activation::SwiGLU,
            position_type: PositionType::RoPE,
            rope_freq_base: 10000.0,
            rope_dim: hidden_size,
            rope_neox: false,
            causal: true,
            attn_logit_softcap: 0.0,
            attn_scale: None,
            embedding_scale: 1.0,
            pooling_type: PoolingType::None,
            has_ffn_gate: true,
            has_bias: false,
            pre_norm: true,
        }
    }

    fn gen_weights(config: &ModelConfig) -> ModelWeights {
        let h = config.hidden_size;
        let ffn = config.ffn_hidden;
        let v = config.vocab_size;

        // Use distinct embedding rows
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
            output_projection: None, // tied embeddings
        }
    }

    fn build_gen_engine(config: ModelConfig) -> GenerationEngine {
        let weights = gen_weights(&config);
        let tokenizer: Box<dyn Tokenizer> = Box::new(MockTokenizer::new());
        let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
        GenerationEngine::new(config, weights, tokenizer, backend)
    }

    #[test]
    fn test_generate_greedy_deterministic() {
        let config = gen_config(4);
        let engine = build_gen_engine(config);
        let gen_cfg = GenerationConfig {
            max_tokens: 5,
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let result1 = engine.generate("hello", &gen_cfg).unwrap();
        let result2 = engine.generate("hello", &gen_cfg).unwrap();
        assert_eq!(result1, result2, "Greedy generation should be deterministic");
    }

    #[test]
    fn test_generate_stops_at_max_tokens() {
        // Use a tokenizer with no EOS to isolate max_tokens behavior
        let config = gen_config(4);
        let weights = gen_weights(&config);
        let tokenizer: Box<dyn Tokenizer> = Box::new(NoEosTokenizer::new());
        let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
        let engine = GenerationEngine::new(config, weights, tokenizer, backend);

        let gen_cfg = GenerationConfig {
            max_tokens: 3,
            stop_tokens: vec![],
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
        };

        // Use generate_stream to count exact tokens produced
        let mut token_count = 0;
        let result = engine.generate_stream("hello", &gen_cfg, |_| {
            token_count += 1;
            true
        });
        assert!(result.is_ok());
        let generated = result.unwrap();
        assert!(
            generated.len() <= 3,
            "Expected at most 3 tokens, got {}",
            generated.len()
        );
        assert_eq!(generated.len(), token_count);
    }

    #[test]
    fn test_generate_stops_at_eos() {
        let config = gen_config(4);
        let engine = build_gen_engine(config);

        // Use a large max_tokens — if EOS stops generation, we'll get
        // far fewer tokens than the limit.
        let gen_cfg = GenerationConfig {
            max_tokens: 100,
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        // Count tokens via stream to verify EOS stops before max_tokens
        let mut token_count = 0;
        let result = engine.generate_stream("hello", &gen_cfg, |_| {
            token_count += 1;
            true
        });
        assert!(result.is_ok());
        let generated = result.unwrap();
        // With EOS in the vocabulary (token 2), generation should stop
        // well before 100 tokens. Verify it stopped early.
        assert!(
            generated.len() < 100,
            "Expected EOS to stop generation before 100 tokens, got {}",
            generated.len()
        );
    }

    #[test]
    fn test_generate_stream_callback() {
        let config = gen_config(4);
        let engine = build_gen_engine(config);
        let gen_cfg = GenerationConfig {
            max_tokens: 10,
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut received_tokens = Vec::new();
        let result = engine.generate_stream("hello", &gen_cfg, |token_id| {
            received_tokens.push(token_id);
            received_tokens.len() < 3 // stop after 3 tokens
        });

        assert!(result.is_ok());
        let generated = result.unwrap();
        // Should have stopped at 3 or fewer tokens
        assert!(generated.len() <= 3);
        assert_eq!(generated.len(), received_tokens.len());
        // Verify callback received exactly the same token IDs as the result
        assert_eq!(
            generated, received_tokens,
            "Stream callback tokens should match returned tokens"
        );
    }

    #[test]
    fn test_generate_rejects_non_causal_model() {
        let mut config = gen_config(4);
        config.causal = false;
        config.arch = ModelArch::Bert;

        let weights = gen_weights(&config);
        let tokenizer: Box<dyn Tokenizer> = Box::new(MockTokenizer::new());
        let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
        let engine = GenerationEngine::new(config, weights, tokenizer, backend);

        let gen_cfg = GenerationConfig::default();
        let result = engine.generate("hello", &gen_cfg);
        assert!(result.is_err(), "generate() should reject non-causal model");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("causal"), "Error should mention causal: {}", err_msg);

        // Also test generate_stream rejects non-causal
        let stream_result = engine.generate_stream("hello", &gen_cfg, |_| true);
        assert!(stream_result.is_err(), "generate_stream() should reject non-causal model");
    }

    #[test]
    fn test_generate_empty_prompt() {
        let config = gen_config(4);
        let engine = build_gen_engine(config);
        let gen_cfg = GenerationConfig::default();

        // Empty string still gets BOS from mock tokenizer, so it's 1 token
        let result = engine.generate("", &gen_cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_prompt_exceeds_context() {
        let mut config = gen_config(4);
        config.max_seq_len = 3; // very short
        let engine = build_gen_engine(config);
        let gen_cfg = GenerationConfig::default();

        // "hello world foo" with BOS = 4 tokens, exceeds max_seq_len=3
        let result = engine.generate("hello world foo", &gen_cfg);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("exceeds max_seq_len"), "Error: {}", err_msg);
    }

    #[test]
    fn test_generate_with_explicit_stop_tokens() {
        let config = gen_config(4);
        let engine = build_gen_engine(config);
        let gen_cfg = GenerationConfig {
            max_tokens: 100,
            stop_tokens: vec![3, 4, 5, 6, 7], // stop on any content token
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
        };

        let result = engine.generate("hello", &gen_cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_accessors() {
        let config = gen_config(4);
        let engine = build_gen_engine(config);
        assert_eq!(engine.vocab_size(), 8);
        assert_eq!(engine.config().arch, ModelArch::Gemma3);
        assert!(engine.config().causal);
    }

    #[test]
    fn test_generation_engine_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GenerationEngine>();
    }

    #[test]
    fn test_generate_max_tokens_zero() {
        let config = gen_config(4);
        let engine = build_gen_engine(config);
        let gen_cfg = GenerationConfig {
            max_tokens: 0,
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = engine.generate("hello", &gen_cfg).unwrap();
        // max_tokens=0 means no decode steps, so output should be empty
        assert!(result.is_empty(), "max_tokens=0 should produce empty output, got: {:?}", result);
    }

    #[test]
    fn test_generate_and_stream_consistency() {
        // generate() and generate_stream() should produce the same tokens
        // for the same input with the same seed.
        let config = gen_config(4);

        // Build two engines with identical state
        let engine1 = build_gen_engine(config.clone());
        let engine2 = build_gen_engine(config);

        let gen_cfg = GenerationConfig {
            max_tokens: 5,
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        // generate() returns decoded text
        let text_result = engine1.generate("hello", &gen_cfg).unwrap();

        // generate_stream() returns token IDs
        let stream_ids = engine2.generate_stream("hello", &gen_cfg, |_| true).unwrap();

        // Decode stream IDs using a MockTokenizer (same as used internally)
        let mock_tok = MockTokenizer::new();
        let stream_text = mock_tok.decode(&stream_ids);
        assert_eq!(
            text_result, stream_text,
            "generate() and generate_stream() should produce identical output"
        );
    }

    #[test]
    fn test_stop_reason_display() {
        assert_eq!(StopReason::StopToken.to_string(), "eos");
        assert_eq!(StopReason::MaxTokens.to_string(), "max_tokens");
        assert_eq!(StopReason::ContextLength.to_string(), "context_length");
        assert_eq!(StopReason::Cancelled.to_string(), "cancelled");
    }

    #[test]
    fn test_generate_full_returns_metadata() {
        let config = gen_config(4);
        let engine = build_gen_engine(config);
        let gen_cfg = GenerationConfig {
            max_tokens: 5,
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let output = engine.generate_full("hello", &gen_cfg).unwrap();
        // prompt "hello" with BOS = 2 tokens
        assert_eq!(output.prompt_tokens, 2);
        assert!(!output.token_ids.is_empty() || output.stop_reason == StopReason::StopToken);
    }

    #[test]
    fn test_generate_stream_full_cancelled() {
        let config = gen_config(4);
        let weights = gen_weights(&config);
        let tokenizer: Box<dyn Tokenizer> = Box::new(NoEosTokenizer::new());
        let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
        let engine = GenerationEngine::new(config, weights, tokenizer, backend);

        let gen_cfg = GenerationConfig {
            max_tokens: 100,
            stop_tokens: vec![],
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
        };

        let output = engine.generate_stream_full("hello", &gen_cfg, |_| false).unwrap();
        assert_eq!(output.stop_reason, StopReason::Cancelled);
        // Only 1 token should be generated before callback cancels
        assert_eq!(output.token_ids.len(), 1);
    }

    #[test]
    fn test_generate_full_max_tokens_stop_reason() {
        let config = gen_config(4);
        let weights = gen_weights(&config);
        let tokenizer: Box<dyn Tokenizer> = Box::new(NoEosTokenizer::new());
        let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
        let engine = GenerationEngine::new(config, weights, tokenizer, backend);

        let gen_cfg = GenerationConfig {
            max_tokens: 3,
            stop_tokens: vec![],
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
        };

        let output = engine.generate_full("hello", &gen_cfg).unwrap();
        assert_eq!(output.stop_reason, StopReason::MaxTokens);
        assert_eq!(output.token_ids.len(), 3);
    }

    #[test]
    fn test_generate_full_stop_token_via_explicit_stop() {
        let config = gen_config(4);
        let engine = build_gen_engine(config);

        // First, do a greedy run to find out what the first generated token is
        let probe = engine.generate_full("hello", &GenerationConfig {
            max_tokens: 1,
            stop_tokens: vec![],
            sampling: SamplingConfig { temperature: 0.0, ..Default::default() },
        }).unwrap();

        assert!(!probe.token_ids.is_empty(), "Need at least one token to test stop");
        let first_token = probe.token_ids[0];

        // Now run again with that token as a stop token — it should be produced
        // as the first decode token and trigger StopToken
        let output = engine.generate_full("hello", &GenerationConfig {
            max_tokens: 100,
            stop_tokens: vec![first_token],
            sampling: SamplingConfig { temperature: 0.0, ..Default::default() },
        }).unwrap();

        assert_eq!(output.stop_reason, StopReason::StopToken);
        // The stop token is NOT included in the output
        assert!(output.token_ids.is_empty());
    }

    #[test]
    fn test_generate_full_context_length_stop_reason() {
        let mut config = gen_config(4);
        config.max_seq_len = 5; // very short: prompt(2) + max 3 decode steps
        let weights = gen_weights(&config);
        let tokenizer: Box<dyn Tokenizer> = Box::new(NoEosTokenizer::new());
        let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
        let engine = GenerationEngine::new(config, weights, tokenizer, backend);

        let gen_cfg = GenerationConfig {
            max_tokens: 100,
            stop_tokens: vec![],
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
        };

        let output = engine.generate_full("hello", &gen_cfg).unwrap();
        assert_eq!(output.stop_reason, StopReason::ContextLength);
        // prompt=2 tokens, max_seq_len=5 → 3 forward passes possible,
        // but 4 tokens sampled (1 from prefill + 3 from decode steps).
        // The last token is sampled but can't be fed back — it should
        // still be included in the output (this was the token drop bug).
        assert_eq!(
            output.token_ids.len(),
            4,
            "Context-length stop should preserve the final sampled token"
        );
    }

    #[test]
    fn test_generate_context_length_preserves_last_token() {
        // Verify the simple generate() API also preserves the last token
        // when hitting context length (same bug as generate_full).
        let mut config = gen_config(4);
        config.max_seq_len = 5; // prompt(2) + 3 decode forward passes
        let weights = gen_weights(&config);
        let tokenizer: Box<dyn Tokenizer> = Box::new(NoEosTokenizer::new());
        let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
        let engine = GenerationEngine::new(config, weights, tokenizer, backend);

        let gen_cfg = GenerationConfig {
            max_tokens: 100,
            stop_tokens: vec![],
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
        };

        // generate() returns decoded text; verify it's non-empty
        let text = engine.generate("hello", &gen_cfg).unwrap();
        assert!(!text.is_empty(), "Should produce output at context limit");
    }

    #[test]
    fn test_generate_stream_context_length_fires_callback() {
        // Verify the streaming API calls the callback for the final token
        // even when that token triggers the context-length stop.
        let mut config = gen_config(4);
        config.max_seq_len = 5;
        let weights = gen_weights(&config);
        let tokenizer: Box<dyn Tokenizer> = Box::new(NoEosTokenizer::new());
        let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
        let engine = GenerationEngine::new(config, weights, tokenizer, backend);

        let gen_cfg = GenerationConfig {
            max_tokens: 100,
            stop_tokens: vec![],
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
        };

        let mut streamed_tokens = Vec::new();
        let output = engine
            .generate_stream_full("hello", &gen_cfg, |tok| {
                streamed_tokens.push(tok);
                true
            })
            .unwrap();

        assert_eq!(output.stop_reason, StopReason::ContextLength);
        assert_eq!(output.token_ids.len(), 4);
        // Every token in token_ids should have been streamed via callback
        assert_eq!(
            output.token_ids, streamed_tokens,
            "Streamed tokens should match token_ids exactly"
        );
    }

    #[test]
    fn test_generate_stream_full_tokens_match_stream() {
        let config = gen_config(4);
        let engine1 = build_gen_engine(config.clone());
        let engine2 = build_gen_engine(config);
        let gen_cfg = GenerationConfig {
            max_tokens: 5,
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let stream_ids = engine1.generate_stream("hello", &gen_cfg, |_| true).unwrap();
        let full_output = engine2.generate_stream_full("hello", &gen_cfg, |_| true).unwrap();
        assert_eq!(stream_ids, full_output.token_ids);
    }

    #[test]
    fn test_decode_method() {
        let config = gen_config(4);
        let engine = build_gen_engine(config);
        assert_eq!(engine.decode(&[3, 4]), "hello world");
        assert_eq!(engine.decode(&[]), "");
    }
}
