// M4.1: Model configuration extracted from GGUF metadata.

use crate::error::InferenceError;
use crate::gguf::{GgufFile, GgufValue};

use tracing::info;

/// Supported model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    Gemma3,
    Gemma2,
    GemmaEmbedding,
    LLaMA,
    Bert,
    GPT2,
    Qwen3,
    Mistral3,
    Phi2,
    Phi3,
}

/// Normalization type used by the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    LayerNorm,
    RMSNorm,
}

/// Activation function in the FFN block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    GELU,
    SwiGLU,
    GeGLU,
}

/// Position embedding type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionType {
    Learned,
    RoPE,
}

/// Pooling strategy for embedding models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingType {
    None,
    Mean,
    Cls,
    Last,
}

/// Full model configuration extracted from GGUF metadata.
///
/// All dimension parameters and architectural choices are determined
/// from the GGUF key-value pairs at load time — no hardcoded values.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    // Architecture
    pub arch: ModelArch,
    pub arch_name: String,

    // Dimensions
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_hidden: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,

    // Normalization
    pub norm_type: NormType,
    pub norm_eps: f32,

    // Activation
    pub activation: Activation,

    // Position encoding
    pub position_type: PositionType,
    pub rope_freq_base: f32,
    pub rope_dim: usize,
    /// NeoX-style RoPE (pairs offset by n_rot/2) vs standard (consecutive pairs).
    /// Gemma, Qwen, Falcon, etc. use NeoX; LLaMA uses standard.
    pub rope_neox: bool,
    /// Original context length for RoPE scaling (LongRoPE / YaRN).
    /// When > 0, the model was trained with per-dimension frequency factors
    /// stored as `rope_factors_short.weight` and `rope_factors_long.weight`.
    /// Use short factors when seq_len < this value, long factors otherwise.
    pub rope_scaling_original_ctx: usize,
    /// LongRoPE magnitude scaling factor.
    /// Loaded from `{arch}.rope.scaling.attn_factor`, default 1.0.
    /// Passed directly as mscale to the RoPE kernel (scales cos/sin of both Q and K).
    pub rope_scaling_attn_factor: f32,

    // Attention
    pub causal: bool,
    pub attn_logit_softcap: f32,
    pub attn_scale: Option<f32>,

    // Embedding
    pub embedding_scale: f32,

    // Pooling
    pub pooling_type: PoolingType,

    // Architecture-specific flags
    pub has_ffn_gate: bool,
    pub has_bias: bool,

    // Norm ordering: true = pre-norm (Gemma/LLaMA), false = post-norm (BERT)
    pub pre_norm: bool,
}

impl ModelConfig {
    /// Extract a full model configuration from GGUF metadata.
    ///
    /// Reads `general.architecture` to determine the model family, then
    /// loads all dimension and configuration keys using the `{arch}.*`
    /// key pattern. Architecture-specific defaults (normalization type,
    /// activation function, position encoding, etc.) are applied based
    /// on the detected architecture.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, InferenceError> {
        // ---- Architecture detection ----
        let arch_str = gguf.require_str("general.architecture")?;
        let (arch, arch_name) = match arch_str {
            "gemma3" => (ModelArch::Gemma3, "gemma3"),
            "gemma2" => (ModelArch::Gemma2, "gemma2"),
            "llama" => (ModelArch::LLaMA, "llama"),
            "bert" => (ModelArch::Bert, "bert"),
            "gpt2" => (ModelArch::GPT2, "gpt2"),
            "qwen3" => (ModelArch::Qwen3, "qwen3"),
            "gemma-embedding" => (ModelArch::GemmaEmbedding, "gemma-embedding"),
            "mistral3" => (ModelArch::Mistral3, "mistral3"),
            "phi2" => (ModelArch::Phi2, "phi2"),
            "phi3" => (ModelArch::Phi3, "phi3"),
            other => {
                return Err(InferenceError::UnsupportedArchitecture(other.to_string()));
            }
        };
        let arch_name = arch_name.to_string();

        // ---- Required dimension keys ----
        let hidden_size = gguf
            .require_u32(&format!("{}.embedding_length", arch_name))?
            as usize;
        let num_layers = gguf
            .require_u32(&format!("{}.block_count", arch_name))?
            as usize;
        let num_heads = gguf
            .require_u32(&format!("{}.attention.head_count", arch_name))?
            as usize;
        let ffn_hidden = gguf
            .require_u32(&format!("{}.feed_forward_length", arch_name))?
            as usize;

        // ---- Optional dimension keys (with defaults) ----
        let num_kv_heads = gguf
            .get_u32(&format!("{}.attention.head_count_kv", arch_name))
            .map(|v| v as usize)
            .unwrap_or(num_heads);

        let default_head_dim = hidden_size / num_heads;
        let head_dim = gguf
            .get_u32(&format!("{}.attention.key_length", arch_name))
            .map(|v| v as usize)
            .unwrap_or(default_head_dim);

        let norm_eps = gguf
            .get_f32(&format!(
                "{}.attention.layer_norm_rms_epsilon",
                arch_name
            ))
            .or_else(|| {
                gguf.get_f32(&format!("{}.attention.layer_norm_epsilon", arch_name))
            })
            .unwrap_or(1e-6);

        let rope_freq_base = gguf
            .get_f32(&format!("{}.rope.freq_base", arch_name))
            .unwrap_or(10000.0);

        let rope_dim = gguf
            .get_u32(&format!("{}.rope.dimension_count", arch_name))
            .map(|v| v as usize)
            .unwrap_or(head_dim);

        let rope_scaling_original_ctx = gguf
            .get_u32(&format!("{}.rope.scaling.original_context_length", arch_name))
            .map(|v| v as usize)
            .unwrap_or(0);

        let rope_scaling_attn_factor = gguf
            .get_f32(&format!("{}.rope.scaling.attn_factor", arch_name))
            .unwrap_or(1.0);

        let max_seq_len = gguf
            .get_u32(&format!("{}.context_length", arch_name))
            .map(|v| v as usize)
            .unwrap_or(2048);

        let attn_logit_softcap = gguf
            .get_f32(&format!("{}.attn_logit_softcapping", arch_name))
            .unwrap_or(0.0);

        let attn_scale = gguf
            .get_f32(&format!("{}.attention.scale", arch_name));

        // ---- Vocab size (with fallback to tokenizer.ggml.tokens array length) ----
        let vocab_size = gguf
            .get_u32(&format!("{}.vocab_size", arch_name))
            .map(|v| v as usize)
            .or_else(|| {
                // Fallback: count tokens in the tokenizer vocabulary array
                match gguf.get("tokenizer.ggml.tokens") {
                    Some(GgufValue::Array(arr)) => Some(arr.len()),
                    _ => None,
                }
            })
            .unwrap_or(0);

        // ---- Pooling type ----
        let pooling_type = match gguf.get_u32(&format!("{}.pooling_type", arch_name)) {
            Some(0) => PoolingType::None,
            Some(1) => PoolingType::Mean,
            Some(2) => PoolingType::Cls,
            Some(3) => PoolingType::Last,
            _ => PoolingType::None,
        };

        // ---- Architecture-specific defaults ----
        let (norm_type, activation, position_type, has_ffn_gate, has_bias, embedding_scale, default_causal, pre_norm, rope_neox) =
            match arch {
                ModelArch::Gemma3 | ModelArch::Gemma2 => (
                    NormType::RMSNorm,
                    Activation::GeGLU,
                    PositionType::RoPE,
                    true,  // has_ffn_gate
                    false, // has_bias
                    (hidden_size as f32).sqrt(), // embedding_scale = sqrt(hidden_size)
                    true,  // default causal (overridden by metadata if present)
                    true,  // pre_norm (norm-before)
                    true,  // rope_neox (Gemma uses NeoX-style RoPE)
                ),
                ModelArch::LLaMA => (
                    NormType::RMSNorm,
                    Activation::SwiGLU,
                    PositionType::RoPE,
                    true,  // has_ffn_gate
                    false, // has_bias
                    1.0,   // embedding_scale
                    true,  // default causal
                    true,  // pre_norm
                    false, // rope_neox (LLaMA uses standard/normal RoPE)
                ),
                ModelArch::Qwen3 => (
                    NormType::RMSNorm,
                    Activation::SwiGLU,
                    PositionType::RoPE,
                    true,  // has_ffn_gate
                    false, // has_bias
                    1.0,   // embedding_scale
                    true,  // default causal
                    true,  // pre_norm
                    true,  // rope_neox (Qwen3 uses NeoX-style RoPE)
                ),
                ModelArch::GemmaEmbedding => (
                    NormType::RMSNorm,
                    Activation::GeGLU,
                    PositionType::RoPE,
                    true,  // has_ffn_gate
                    false, // has_bias
                    (hidden_size as f32).sqrt(), // embedding_scale = sqrt(hidden_size)
                    false, // default causal (bidirectional)
                    true,  // pre_norm (norm-before)
                    true,  // rope_neox (Gemma uses NeoX-style RoPE)
                ),
                ModelArch::Bert => (
                    NormType::LayerNorm,
                    Activation::GELU,
                    PositionType::Learned,
                    false, // has_ffn_gate
                    true,  // has_bias
                    1.0,   // embedding_scale
                    false, // default causal
                    false, // post_norm (norm-after)
                    false, // rope_neox (BERT uses learned position, no RoPE)
                ),
                ModelArch::GPT2 => (
                    NormType::LayerNorm,
                    Activation::GELU,
                    PositionType::Learned,
                    false, // has_ffn_gate (no SwiGLU gate)
                    true,  // has_bias (all projections + norms have bias)
                    1.0,   // embedding_scale
                    true,  // default causal (autoregressive)
                    true,  // pre_norm (norm BEFORE sublayers)
                    false, // rope_neox (no RoPE, uses learned positions)
                ),
                ModelArch::Mistral3 => (
                    NormType::RMSNorm,
                    Activation::SwiGLU,
                    PositionType::RoPE,
                    true,  // has_ffn_gate
                    true,  // has_bias (optional, loaded via load_tensor_optional)
                    1.0,   // embedding_scale
                    true,  // default causal
                    true,  // pre_norm
                    false, // rope_neox (Mistral3 uses standard/NORM RoPE, like LLaMA)
                ),
                ModelArch::Phi2 => (
                    NormType::LayerNorm,
                    Activation::GELU,
                    PositionType::RoPE,
                    false, // has_ffn_gate (no SwiGLU gate)
                    true,  // has_bias (all projections have bias)
                    1.0,   // embedding_scale
                    true,  // default causal
                    true,  // pre_norm
                    true,  // rope_neox (Phi2 uses NeoX-style RoPE)
                ),
                ModelArch::Phi3 => (
                    NormType::RMSNorm,
                    Activation::SwiGLU,
                    PositionType::RoPE,
                    true,  // has_ffn_gate
                    false, // has_bias (Phi-3 uses RMSNorm, no biases)
                    1.0,   // embedding_scale
                    true,  // default causal
                    true,  // pre_norm
                    true,  // rope_neox (Phi3 uses NeoX-style RoPE)
                ),
            };

        // ---- Causal attention (metadata overrides default) ----
        let causal = gguf
            .get_bool(&format!("{}.attention.causal", arch_name))
            .unwrap_or(default_causal);

        // ---- Validation ----
        // head_dim must divide evenly (unless explicit key_length was provided)
        if gguf
            .get_u32(&format!("{}.attention.key_length", arch_name))
            .is_none()
            && hidden_size % num_heads != 0
        {
            return Err(InferenceError::Model(format!(
                "hidden_size ({}) is not divisible by num_heads ({})",
                hidden_size, num_heads
            )));
        }

        // GQA group size must divide evenly
        if num_heads % num_kv_heads != 0 {
            return Err(InferenceError::Model(format!(
                "num_heads ({}) is not divisible by num_kv_heads ({})",
                num_heads, num_kv_heads
            )));
        }

        let config = ModelConfig {
            arch,
            arch_name,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            ffn_hidden,
            vocab_size,
            max_seq_len,
            norm_type,
            norm_eps,
            activation,
            position_type,
            rope_freq_base,
            rope_dim,
            rope_neox,
            rope_scaling_original_ctx,
            rope_scaling_attn_factor,
            causal,
            attn_logit_softcap,
            attn_scale,
            embedding_scale,
            pooling_type,
            has_ffn_gate,
            has_bias,
            pre_norm,
        };

        info!(
            arch = %config.arch_name,
            hidden_size = config.hidden_size,
            num_layers = config.num_layers,
            num_heads = config.num_heads,
            num_kv_heads = config.num_kv_heads,
            head_dim = config.head_dim,
            ffn_hidden = config.ffn_hidden,
            vocab_size = config.vocab_size,
            max_seq_len = config.max_seq_len,
            norm_type = ?config.norm_type,
            norm_eps = config.norm_eps,
            activation = ?config.activation,
            position_type = ?config.position_type,
            rope_freq_base = config.rope_freq_base,
            rope_dim = config.rope_dim,
            causal = config.causal,
            pooling_type = ?config.pooling_type,
            embedding_scale = config.embedding_scale,
            has_ffn_gate = config.has_ffn_gate,
            has_bias = config.has_bias,
            "model config loaded from GGUF"
        );

        Ok(config)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ====================================================================
    // Test GGUF builder helpers
    //
    // These mirror the helpers in src/gguf/mod.rs tests, building
    // synthetic GGUF v3 files that can be opened with GgufFile::open().
    // ====================================================================

    const GGUF_MAGIC: u32 = 0x4655_4747;

    /// Round `offset` up to the next multiple of `alignment`.
    fn align_offset(offset: u64, alignment: u64) -> u64 {
        let remainder = offset % alignment;
        if remainder == 0 {
            offset
        } else {
            offset + (alignment - remainder)
        }
    }

    /// Build a minimal valid GGUF v3 file in memory with the given KV pairs.
    fn build_gguf_bytes(kv_pairs: &[(&str, &[u8])]) -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version 3
        buf.extend_from_slice(&3u32.to_le_bytes());
        // n_tensors = 0
        buf.extend_from_slice(&0u64.to_le_bytes());
        // n_kv
        buf.extend_from_slice(&(kv_pairs.len() as u64).to_le_bytes());

        // KV pairs: each is (key_string, type_id+value already serialized)
        for &(key, value_bytes) in kv_pairs {
            // Key string: u64 length + raw bytes
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            // Value (type_id + data, already serialized)
            buf.extend_from_slice(value_bytes);
        }

        // Pad to 32-byte alignment
        let pos = buf.len();
        let aligned = align_offset(pos as u64, 32) as usize;
        buf.resize(aligned, 0);

        buf
    }

    /// Serialize a GGUF U32 KV value (type_id=4 + u32 value).
    fn kv_u32(val: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&4u32.to_le_bytes()); // GGUF_TYPE_UINT32
        buf.extend_from_slice(&val.to_le_bytes());
        buf
    }

    /// Serialize a GGUF String KV value (type_id=8 + string).
    fn kv_string(val: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&8u32.to_le_bytes()); // GGUF_TYPE_STRING
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val.as_bytes());
        buf
    }

    /// Serialize a GGUF F32 KV value (type_id=6 + f32 value).
    fn kv_f32(val: f32) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&6u32.to_le_bytes()); // GGUF_TYPE_FLOAT32
        buf.extend_from_slice(&val.to_le_bytes());
        buf
    }

    /// Serialize a GGUF Bool KV value (type_id=7 + i8 value).
    fn kv_bool(val: bool) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&7u32.to_le_bytes()); // GGUF_TYPE_BOOL
        buf.push(if val { 1 } else { 0 });
        buf
    }

    /// Serialize a GGUF String Array KV value (type_id=9 + elem_type=8 + count + strings).
    fn kv_str_array(vals: &[&str]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&9u32.to_le_bytes()); // GGUF_TYPE_ARRAY
        buf.extend_from_slice(&8u32.to_le_bytes()); // element type: String
        buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for s in vals {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        buf
    }

    /// Write bytes to a named temp file and return the file handle (keeps it alive).
    fn write_temp_gguf(bytes: &[u8]) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("failed to create temp file");
        file.write_all(bytes).expect("failed to write temp file");
        file.flush().expect("failed to flush temp file");
        file
    }

    /// Build standard Gemma3-style GGUF metadata KV pairs.
    fn gemma3_kv_pairs() -> Vec<(&'static str, Vec<u8>)> {
        vec![
            ("general.architecture", kv_string("gemma3")),
            ("gemma3.embedding_length", kv_u32(768)),
            ("gemma3.block_count", kv_u32(24)),
            ("gemma3.attention.head_count", kv_u32(16)),
            ("gemma3.feed_forward_length", kv_u32(3072)),
        ]
    }

    /// Build standard BERT-style GGUF metadata KV pairs.
    fn bert_kv_pairs() -> Vec<(&'static str, Vec<u8>)> {
        vec![
            ("general.architecture", kv_string("bert")),
            ("bert.embedding_length", kv_u32(384)),
            ("bert.block_count", kv_u32(6)),
            ("bert.attention.head_count", kv_u32(12)),
            ("bert.feed_forward_length", kv_u32(1536)),
        ]
    }

    /// Helper: build a GGUF file from owned KV pairs and open it as a GgufFile.
    fn open_gguf_from_kv(kv_pairs: &[(&str, Vec<u8>)]) -> (GgufFile, NamedTempFile) {
        // Convert owned Vec<u8> references to &[u8] for build_gguf_bytes
        let refs: Vec<(&str, &[u8])> = kv_pairs
            .iter()
            .map(|(k, v)| (*k, v.as_slice()))
            .collect();
        let bytes = build_gguf_bytes(&refs);
        let file = write_temp_gguf(&bytes);
        let gguf = GgufFile::open(file.path()).expect("failed to open temp GGUF");
        (gguf, file)
    }

    // ====================================================================
    // Tests
    // ====================================================================

    #[test]
    fn test_gemma3_config() {
        let kv = gemma3_kv_pairs();
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();

        assert_eq!(config.arch, ModelArch::Gemma3);
        assert_eq!(config.arch_name, "gemma3");
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.ffn_hidden, 3072);

        // Gemma3-specific defaults
        assert_eq!(config.norm_type, NormType::RMSNorm);
        assert_eq!(config.activation, Activation::GeGLU);
        assert_eq!(config.position_type, PositionType::RoPE);
        assert!(config.has_ffn_gate);
        assert!(!config.has_bias);
        assert!(config.causal); // default for Gemma
        assert!(config.pre_norm); // Gemma uses pre-norm
    }

    #[test]
    fn test_gemma2_config() {
        let kv = vec![
            ("general.architecture", kv_string("gemma2")),
            ("gemma2.embedding_length", kv_u32(512)),
            ("gemma2.block_count", kv_u32(12)),
            ("gemma2.attention.head_count", kv_u32(8)),
            ("gemma2.feed_forward_length", kv_u32(2048)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();

        assert_eq!(config.arch, ModelArch::Gemma2);
        assert_eq!(config.arch_name, "gemma2");
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.norm_type, NormType::RMSNorm);
        assert_eq!(config.activation, Activation::GeGLU);
        assert_eq!(config.position_type, PositionType::RoPE);
        assert!(config.has_ffn_gate);
        assert!(!config.has_bias);
        assert!(config.pre_norm); // Gemma2 uses pre-norm
        // embedding_scale = sqrt(512) for Gemma2
        assert!((config.embedding_scale - (512.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_bert_config() {
        let kv = bert_kv_pairs();
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();

        assert_eq!(config.arch, ModelArch::Bert);
        assert_eq!(config.arch_name, "bert");
        assert_eq!(config.hidden_size, 384);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.ffn_hidden, 1536);

        // BERT-specific defaults
        assert_eq!(config.norm_type, NormType::LayerNorm);
        assert_eq!(config.activation, Activation::GELU);
        assert_eq!(config.position_type, PositionType::Learned);
        assert!(!config.has_ffn_gate);
        assert!(config.has_bias);
        assert!(!config.causal); // BERT is bidirectional
        assert!(!config.pre_norm); // BERT uses post-norm
    }

    #[test]
    fn test_llama_config() {
        let kv = vec![
            ("general.architecture", kv_string("llama")),
            ("llama.embedding_length", kv_u32(4096)),
            ("llama.block_count", kv_u32(32)),
            ("llama.attention.head_count", kv_u32(32)),
            ("llama.feed_forward_length", kv_u32(11008)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();

        assert_eq!(config.arch, ModelArch::LLaMA);
        assert_eq!(config.arch_name, "llama");
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.head_dim, 128); // 4096 / 32
        assert_eq!(config.ffn_hidden, 11008);

        // LLaMA defaults
        assert_eq!(config.norm_type, NormType::RMSNorm);
        assert_eq!(config.activation, Activation::SwiGLU);
        assert_eq!(config.position_type, PositionType::RoPE);
        assert!(config.has_ffn_gate);
        assert!(!config.has_bias);
        assert!(config.causal); // default for LLaMA
        assert!(config.pre_norm); // LLaMA uses pre-norm
        assert!((config.embedding_scale - 1.0).abs() < 1e-6); // LLaMA scale = 1.0
    }

    #[test]
    fn test_unsupported_architecture() {
        let kv = vec![
            ("general.architecture", kv_string("mamba")),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let result = ModelConfig::from_gguf(&gguf);
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::UnsupportedArchitecture(name) => {
                assert_eq!(name, "mamba");
            }
            e => panic!("expected UnsupportedArchitecture, got {:?}", e),
        }
    }

    #[test]
    fn test_missing_architecture_key() {
        // No general.architecture key at all
        let kv: Vec<(&str, Vec<u8>)> = vec![
            ("some.other.key", kv_u32(42)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let result = ModelConfig::from_gguf(&gguf);
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::MissingKey(key) => {
                assert_eq!(key, "general.architecture");
            }
            e => panic!("expected MissingKey, got {:?}", e),
        }
    }

    #[test]
    fn test_missing_required_key_embedding_length() {
        // Has architecture but missing embedding_length
        let kv = vec![
            ("general.architecture", kv_string("gemma3")),
            // Missing: gemma3.embedding_length
            ("gemma3.block_count", kv_u32(24)),
            ("gemma3.attention.head_count", kv_u32(16)),
            ("gemma3.feed_forward_length", kv_u32(3072)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let result = ModelConfig::from_gguf(&gguf);
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::MissingKey(key) => {
                assert_eq!(key, "gemma3.embedding_length");
            }
            e => panic!("expected MissingKey for embedding_length, got {:?}", e),
        }
    }

    #[test]
    fn test_missing_required_key_block_count() {
        let kv = vec![
            ("general.architecture", kv_string("gemma3")),
            ("gemma3.embedding_length", kv_u32(768)),
            // Missing: gemma3.block_count
            ("gemma3.attention.head_count", kv_u32(16)),
            ("gemma3.feed_forward_length", kv_u32(3072)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let result = ModelConfig::from_gguf(&gguf);
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::MissingKey(key) => {
                assert_eq!(key, "gemma3.block_count");
            }
            e => panic!("expected MissingKey for block_count, got {:?}", e),
        }
    }

    #[test]
    fn test_missing_required_key_head_count() {
        let kv = vec![
            ("general.architecture", kv_string("gemma3")),
            ("gemma3.embedding_length", kv_u32(768)),
            ("gemma3.block_count", kv_u32(24)),
            // Missing: gemma3.attention.head_count
            ("gemma3.feed_forward_length", kv_u32(3072)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let result = ModelConfig::from_gguf(&gguf);
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::MissingKey(key) => {
                assert_eq!(key, "gemma3.attention.head_count");
            }
            e => panic!("expected MissingKey for head_count, got {:?}", e),
        }
    }

    #[test]
    fn test_missing_required_key_feed_forward_length() {
        let kv = vec![
            ("general.architecture", kv_string("gemma3")),
            ("gemma3.embedding_length", kv_u32(768)),
            ("gemma3.block_count", kv_u32(24)),
            ("gemma3.attention.head_count", kv_u32(16)),
            // Missing: gemma3.feed_forward_length
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let result = ModelConfig::from_gguf(&gguf);
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::MissingKey(key) => {
                assert_eq!(key, "gemma3.feed_forward_length");
            }
            e => panic!("expected MissingKey for feed_forward_length, got {:?}", e),
        }
    }

    #[test]
    fn test_default_values() {
        // Only required keys, all optional keys should get defaults
        let kv = gemma3_kv_pairs();
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();

        // num_kv_heads defaults to num_heads
        assert_eq!(config.num_kv_heads, config.num_heads);
        assert_eq!(config.num_kv_heads, 16);

        // head_dim defaults to hidden_size / num_heads
        assert_eq!(config.head_dim, 768 / 16);
        assert_eq!(config.head_dim, 48);

        // norm_eps defaults to 1e-6
        assert!((config.norm_eps - 1e-6).abs() < 1e-10);

        // rope_freq_base defaults to 10000.0
        assert!((config.rope_freq_base - 10000.0).abs() < 1e-6);

        // rope_dim defaults to head_dim
        assert_eq!(config.rope_dim, 48);

        // max_seq_len defaults to 2048
        assert_eq!(config.max_seq_len, 2048);

        // attn_logit_softcap defaults to 0.0
        assert!((config.attn_logit_softcap - 0.0).abs() < 1e-10);

        // attn_scale defaults to None
        assert!(config.attn_scale.is_none());

        // vocab_size defaults to 0 when no key and no tokenizer tokens
        assert_eq!(config.vocab_size, 0);

        // pooling_type defaults to None
        assert_eq!(config.pooling_type, PoolingType::None);
    }

    #[test]
    fn test_optional_keys_override_defaults() {
        let kv = vec![
            ("general.architecture", kv_string("gemma3")),
            ("gemma3.embedding_length", kv_u32(768)),
            ("gemma3.block_count", kv_u32(24)),
            ("gemma3.attention.head_count", kv_u32(16)),
            ("gemma3.feed_forward_length", kv_u32(3072)),
            // Optional overrides:
            ("gemma3.attention.head_count_kv", kv_u32(4)),
            ("gemma3.attention.key_length", kv_u32(64)),
            ("gemma3.attention.layer_norm_rms_epsilon", kv_f32(1e-5)),
            ("gemma3.rope.freq_base", kv_f32(500000.0)),
            ("gemma3.rope.dimension_count", kv_u32(32)),
            ("gemma3.context_length", kv_u32(8192)),
            ("gemma3.attn_logit_softcapping", kv_f32(50.0)),
            ("gemma3.attention.scale", kv_f32(0.125)),
            ("gemma3.vocab_size", kv_u32(256000)),
            ("gemma3.pooling_type", kv_u32(1)), // Mean
            ("gemma3.attention.causal", kv_bool(false)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();

        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.head_dim, 64);
        assert!((config.norm_eps - 1e-5).abs() < 1e-10);
        assert!((config.rope_freq_base - 500000.0).abs() < 1e-6);
        assert_eq!(config.rope_dim, 32);
        assert_eq!(config.max_seq_len, 8192);
        assert!((config.attn_logit_softcap - 50.0).abs() < 1e-6);
        assert_eq!(config.attn_scale, Some(0.125));
        assert_eq!(config.vocab_size, 256000);
        assert_eq!(config.pooling_type, PoolingType::Mean);
        assert!(!config.causal); // overridden to false
    }

    #[test]
    fn test_pooling_type_parsing() {
        // Test all pooling type values
        let test_cases = [
            (0u32, PoolingType::None),
            (1, PoolingType::Mean),
            (2, PoolingType::Cls),
            (3, PoolingType::Last),
        ];

        for (pooling_id, expected) in test_cases {
            let kv = vec![
                ("general.architecture", kv_string("bert")),
                ("bert.embedding_length", kv_u32(384)),
                ("bert.block_count", kv_u32(6)),
                ("bert.attention.head_count", kv_u32(12)),
                ("bert.feed_forward_length", kv_u32(1536)),
                ("bert.pooling_type", kv_u32(pooling_id)),
            ];
            let (gguf, _tmp) = open_gguf_from_kv(&kv);

            let config = ModelConfig::from_gguf(&gguf).unwrap();
            assert_eq!(
                config.pooling_type, expected,
                "pooling_type {} should map to {:?}",
                pooling_id, expected
            );
        }
    }

    #[test]
    fn test_pooling_type_unknown_defaults_to_none() {
        let kv = vec![
            ("general.architecture", kv_string("bert")),
            ("bert.embedding_length", kv_u32(384)),
            ("bert.block_count", kv_u32(6)),
            ("bert.attention.head_count", kv_u32(12)),
            ("bert.feed_forward_length", kv_u32(1536)),
            ("bert.pooling_type", kv_u32(99)), // unknown
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();
        assert_eq!(config.pooling_type, PoolingType::None);
    }

    #[test]
    fn test_gqa_head_counts() {
        // LLaMA with GQA: 32 heads, 8 KV heads (4 groups)
        let kv = vec![
            ("general.architecture", kv_string("llama")),
            ("llama.embedding_length", kv_u32(4096)),
            ("llama.block_count", kv_u32(32)),
            ("llama.attention.head_count", kv_u32(32)),
            ("llama.attention.head_count_kv", kv_u32(8)),
            ("llama.feed_forward_length", kv_u32(11008)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();

        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert!(config.num_heads > config.num_kv_heads);
        assert_eq!(config.num_heads % config.num_kv_heads, 0);
    }

    #[test]
    fn test_gqa_invalid_group_size() {
        // num_heads=32, num_kv_heads=5 -- 32 % 5 != 0, should error
        let kv = vec![
            ("general.architecture", kv_string("llama")),
            ("llama.embedding_length", kv_u32(4096)),
            ("llama.block_count", kv_u32(32)),
            ("llama.attention.head_count", kv_u32(32)),
            ("llama.attention.head_count_kv", kv_u32(5)),
            ("llama.feed_forward_length", kv_u32(11008)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let result = ModelConfig::from_gguf(&gguf);
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::Model(msg) => {
                assert!(msg.contains("num_heads"), "error should mention num_heads: {}", msg);
                assert!(msg.contains("num_kv_heads"), "error should mention num_kv_heads: {}", msg);
            }
            e => panic!("expected Model error, got {:?}", e),
        }
    }

    #[test]
    fn test_hidden_size_not_divisible_by_heads() {
        // hidden_size=768, num_heads=5 -- 768 % 5 != 0 and no explicit key_length
        let kv = vec![
            ("general.architecture", kv_string("gemma3")),
            ("gemma3.embedding_length", kv_u32(768)),
            ("gemma3.block_count", kv_u32(24)),
            ("gemma3.attention.head_count", kv_u32(5)),
            ("gemma3.feed_forward_length", kv_u32(3072)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let result = ModelConfig::from_gguf(&gguf);
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::Model(msg) => {
                assert!(msg.contains("hidden_size"), "error should mention hidden_size: {}", msg);
                assert!(msg.contains("num_heads"), "error should mention num_heads: {}", msg);
            }
            e => panic!("expected Model error, got {:?}", e),
        }
    }

    #[test]
    fn test_explicit_key_length_bypasses_divisibility_check() {
        // hidden_size=768, num_heads=5 -- normally invalid, but explicit key_length
        // should bypass the divisibility check
        let kv = vec![
            ("general.architecture", kv_string("gemma3")),
            ("gemma3.embedding_length", kv_u32(768)),
            ("gemma3.block_count", kv_u32(24)),
            ("gemma3.attention.head_count", kv_u32(5)),
            ("gemma3.attention.head_count_kv", kv_u32(5)),
            ("gemma3.attention.key_length", kv_u32(64)),
            ("gemma3.feed_forward_length", kv_u32(3072)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.num_heads, 5);
    }

    #[test]
    fn test_embedding_scale_gemma_vs_bert() {
        // Gemma3: embedding_scale = sqrt(768) ~= 27.71
        let kv_gemma = gemma3_kv_pairs();
        let (gguf_gemma, _tmp1) = open_gguf_from_kv(&kv_gemma);
        let gemma_config = ModelConfig::from_gguf(&gguf_gemma).unwrap();
        assert!((gemma_config.embedding_scale - (768.0f32).sqrt()).abs() < 1e-4);

        // BERT: embedding_scale = 1.0
        let kv_bert = bert_kv_pairs();
        let (gguf_bert, _tmp2) = open_gguf_from_kv(&kv_bert);
        let bert_config = ModelConfig::from_gguf(&gguf_bert).unwrap();
        assert!((bert_config.embedding_scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_scale_llama() {
        let kv = vec![
            ("general.architecture", kv_string("llama")),
            ("llama.embedding_length", kv_u32(4096)),
            ("llama.block_count", kv_u32(32)),
            ("llama.attention.head_count", kv_u32(32)),
            ("llama.feed_forward_length", kv_u32(11008)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();
        // LLaMA: embedding_scale = 1.0
        assert!((config.embedding_scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vocab_size_from_metadata() {
        let kv = vec![
            ("general.architecture", kv_string("gemma3")),
            ("gemma3.embedding_length", kv_u32(768)),
            ("gemma3.block_count", kv_u32(24)),
            ("gemma3.attention.head_count", kv_u32(16)),
            ("gemma3.feed_forward_length", kv_u32(3072)),
            ("gemma3.vocab_size", kv_u32(256000)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();
        assert_eq!(config.vocab_size, 256000);
    }

    #[test]
    fn test_vocab_size_from_tokenizer_tokens() {
        // No explicit vocab_size key, but tokenizer.ggml.tokens array present
        let kv = vec![
            ("general.architecture", kv_string("gemma3")),
            ("gemma3.embedding_length", kv_u32(768)),
            ("gemma3.block_count", kv_u32(24)),
            ("gemma3.attention.head_count", kv_u32(16)),
            ("gemma3.feed_forward_length", kv_u32(3072)),
            ("tokenizer.ggml.tokens", kv_str_array(&["hello", "world", "foo"])),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();
        assert_eq!(config.vocab_size, 3);
    }

    #[test]
    fn test_vocab_size_explicit_overrides_tokens() {
        // Both vocab_size key and tokenizer.ggml.tokens present; explicit key wins
        let kv = vec![
            ("general.architecture", kv_string("gemma3")),
            ("gemma3.embedding_length", kv_u32(768)),
            ("gemma3.block_count", kv_u32(24)),
            ("gemma3.attention.head_count", kv_u32(16)),
            ("gemma3.feed_forward_length", kv_u32(3072)),
            ("gemma3.vocab_size", kv_u32(256000)),
            ("tokenizer.ggml.tokens", kv_str_array(&["a", "b"])),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();
        assert_eq!(config.vocab_size, 256000); // explicit key takes precedence
    }

    #[test]
    fn test_causal_metadata_overrides_default() {
        // Gemma3 defaults to causal=true, but metadata says false
        let kv = vec![
            ("general.architecture", kv_string("gemma3")),
            ("gemma3.embedding_length", kv_u32(768)),
            ("gemma3.block_count", kv_u32(24)),
            ("gemma3.attention.head_count", kv_u32(16)),
            ("gemma3.feed_forward_length", kv_u32(3072)),
            ("gemma3.attention.causal", kv_bool(false)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();
        assert!(!config.causal);
    }

    #[test]
    fn test_bert_causal_metadata_overrides_default() {
        // BERT defaults to causal=false, but metadata says true
        let kv = vec![
            ("general.architecture", kv_string("bert")),
            ("bert.embedding_length", kv_u32(384)),
            ("bert.block_count", kv_u32(6)),
            ("bert.attention.head_count", kv_u32(12)),
            ("bert.feed_forward_length", kv_u32(1536)),
            ("bert.attention.causal", kv_bool(true)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();
        assert!(config.causal);
    }

    #[test]
    fn test_head_dim_computed_from_hidden_size() {
        let kv = gemma3_kv_pairs();
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();
        // hidden_size=768, num_heads=16 => head_dim=48
        assert_eq!(config.head_dim, 48);
    }

    #[test]
    fn test_rope_dim_defaults_to_head_dim() {
        let kv = gemma3_kv_pairs();
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();
        assert_eq!(config.rope_dim, config.head_dim);
    }

    #[test]
    fn test_debug_format() {
        let kv = gemma3_kv_pairs();
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("Gemma3"));
        assert!(debug_str.contains("768"));
        assert!(debug_str.contains("RMSNorm"));
    }

    #[test]
    fn test_clone() {
        let kv = gemma3_kv_pairs();
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();
        let cloned = config.clone();
        assert_eq!(cloned.arch, config.arch);
        assert_eq!(cloned.hidden_size, config.hidden_size);
        assert_eq!(cloned.arch_name, config.arch_name);
    }

    #[test]
    fn test_gemma3_full_config_with_all_optional_keys() {
        // A comprehensive Gemma3 config with every optional key set
        let kv = vec![
            ("general.architecture", kv_string("gemma3")),
            ("gemma3.embedding_length", kv_u32(768)),
            ("gemma3.block_count", kv_u32(24)),
            ("gemma3.attention.head_count", kv_u32(16)),
            ("gemma3.attention.head_count_kv", kv_u32(4)),
            ("gemma3.attention.key_length", kv_u32(48)),
            ("gemma3.feed_forward_length", kv_u32(3072)),
            ("gemma3.attention.layer_norm_rms_epsilon", kv_f32(1e-6)),
            ("gemma3.rope.freq_base", kv_f32(10000.0)),
            ("gemma3.rope.dimension_count", kv_u32(48)),
            ("gemma3.context_length", kv_u32(8192)),
            ("gemma3.attn_logit_softcapping", kv_f32(50.0)),
            ("gemma3.attention.scale", kv_f32(0.144)),
            ("gemma3.vocab_size", kv_u32(256128)),
            ("gemma3.pooling_type", kv_u32(1)),
            ("gemma3.attention.causal", kv_bool(false)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();

        assert_eq!(config.arch, ModelArch::Gemma3);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.head_dim, 48);
        assert_eq!(config.ffn_hidden, 3072);
        assert_eq!(config.vocab_size, 256128);
        assert_eq!(config.max_seq_len, 8192);
        assert!((config.norm_eps - 1e-6).abs() < 1e-10);
        assert!((config.rope_freq_base - 10000.0).abs() < 1e-6);
        assert_eq!(config.rope_dim, 48);
        assert!((config.attn_logit_softcap - 50.0).abs() < 1e-6);
        assert_eq!(config.attn_scale, Some(0.144));
        assert_eq!(config.pooling_type, PoolingType::Mean);
        assert!(!config.causal);
        assert_eq!(config.norm_type, NormType::RMSNorm);
        assert_eq!(config.activation, Activation::GeGLU);
        assert_eq!(config.position_type, PositionType::RoPE);
        assert!(config.has_ffn_gate);
        assert!(!config.has_bias);
        assert!((config.embedding_scale - (768.0f32).sqrt()).abs() < 1e-4);
    }

    #[test]
    fn test_gpt2_config() {
        let kv = vec![
            ("general.architecture", kv_string("gpt2")),
            ("gpt2.embedding_length", kv_u32(768)),
            ("gpt2.block_count", kv_u32(12)),
            ("gpt2.attention.head_count", kv_u32(12)),
            ("gpt2.feed_forward_length", kv_u32(3072)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();

        assert_eq!(config.arch, ModelArch::GPT2);
        assert_eq!(config.arch_name, "gpt2");
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.head_dim, 64); // 768 / 12
        assert_eq!(config.ffn_hidden, 3072);

        // GPT-2 defaults
        assert_eq!(config.norm_type, NormType::LayerNorm);
        assert_eq!(config.activation, Activation::GELU);
        assert_eq!(config.position_type, PositionType::Learned);
        assert!(!config.has_ffn_gate);
        assert!(config.has_bias);
        assert!(config.causal);
        assert!(config.pre_norm);
        assert!((config.embedding_scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gemma_embedding_config() {
        let kv = vec![
            ("general.architecture", kv_string("gemma-embedding")),
            ("gemma-embedding.embedding_length", kv_u32(768)),
            ("gemma-embedding.block_count", kv_u32(24)),
            ("gemma-embedding.attention.head_count", kv_u32(3)),
            ("gemma-embedding.attention.head_count_kv", kv_u32(1)),
            ("gemma-embedding.attention.key_length", kv_u32(256)),
            ("gemma-embedding.feed_forward_length", kv_u32(1152)),
            ("gemma-embedding.pooling_type", kv_u32(1)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();

        assert_eq!(config.arch, ModelArch::GemmaEmbedding);
        assert_eq!(config.arch_name, "gemma-embedding");
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_heads, 3);
        assert_eq!(config.num_kv_heads, 1);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.ffn_hidden, 1152);

        // GemmaEmbedding-specific defaults
        assert_eq!(config.norm_type, NormType::RMSNorm);
        assert_eq!(config.activation, Activation::GeGLU);
        assert_eq!(config.position_type, PositionType::RoPE);
        assert!(config.has_ffn_gate);
        assert!(!config.has_bias);
        assert!(!config.causal); // bidirectional
        assert!(config.pre_norm);
        assert_eq!(config.pooling_type, PoolingType::Mean);
        assert!((config.embedding_scale - (768.0f32).sqrt()).abs() < 1e-4);
    }

    #[test]
    fn test_mistral3_config() {
        let kv = vec![
            ("general.architecture", kv_string("mistral3")),
            ("mistral3.embedding_length", kv_u32(3072)),
            ("mistral3.block_count", kv_u32(24)),
            ("mistral3.attention.head_count", kv_u32(24)),
            ("mistral3.attention.head_count_kv", kv_u32(8)),
            ("mistral3.feed_forward_length", kv_u32(8192)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();

        assert_eq!(config.arch, ModelArch::Mistral3);
        assert_eq!(config.arch_name, "mistral3");
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_heads, 24);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128); // 3072 / 24
        assert_eq!(config.ffn_hidden, 8192);

        // Mistral3 defaults
        assert_eq!(config.norm_type, NormType::RMSNorm);
        assert_eq!(config.activation, Activation::SwiGLU);
        assert_eq!(config.position_type, PositionType::RoPE);
        assert!(config.has_ffn_gate);
        assert!(config.has_bias);
        assert!(config.causal);
        assert!(config.pre_norm);
        assert!(!config.rope_neox); // Mistral3 uses standard/NORM RoPE (like LLaMA)
        assert!((config.embedding_scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_phi2_config() {
        let kv = vec![
            ("general.architecture", kv_string("phi2")),
            ("phi2.embedding_length", kv_u32(2560)),
            ("phi2.block_count", kv_u32(32)),
            ("phi2.attention.head_count", kv_u32(32)),
            ("phi2.feed_forward_length", kv_u32(10240)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();

        assert_eq!(config.arch, ModelArch::Phi2);
        assert_eq!(config.arch_name, "phi2");
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.head_dim, 80); // 2560 / 32
        assert_eq!(config.ffn_hidden, 10240);

        // Phi2 defaults
        assert_eq!(config.norm_type, NormType::LayerNorm);
        assert_eq!(config.activation, Activation::GELU);
        assert_eq!(config.position_type, PositionType::RoPE);
        assert!(!config.has_ffn_gate);
        assert!(config.has_bias);
        assert!(config.causal);
        assert!(config.pre_norm);
        assert!(config.rope_neox); // Phi2 uses NeoX-style RoPE
        assert!((config.embedding_scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_phi3_config() {
        let kv = vec![
            ("general.architecture", kv_string("phi3")),
            ("phi3.embedding_length", kv_u32(3072)),
            ("phi3.block_count", kv_u32(32)),
            ("phi3.attention.head_count", kv_u32(32)),
            ("phi3.feed_forward_length", kv_u32(8192)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();

        assert_eq!(config.arch, ModelArch::Phi3);
        assert_eq!(config.arch_name, "phi3");
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 32); // MHA (no head_count_kv → defaults to head_count)
        assert_eq!(config.head_dim, 96); // 3072 / 32
        assert_eq!(config.ffn_hidden, 8192);

        // Phi3 defaults
        assert_eq!(config.norm_type, NormType::RMSNorm);
        assert_eq!(config.activation, Activation::SwiGLU);
        assert_eq!(config.position_type, PositionType::RoPE);
        assert!(config.has_ffn_gate);
        assert!(!config.has_bias); // Phi-3 uses RMSNorm, no biases
        assert!(config.causal);
        assert!(config.pre_norm);
        assert!(config.rope_neox);
        assert!((config.embedding_scale - 1.0).abs() < 1e-6);
        // Default attn_factor when not in GGUF
        assert!((config.rope_scaling_attn_factor - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_qwen3_config() {
        let kv = vec![
            ("general.architecture", kv_string("qwen3")),
            ("qwen3.embedding_length", kv_u32(2048)),
            ("qwen3.block_count", kv_u32(24)),
            ("qwen3.attention.head_count", kv_u32(16)),
            ("qwen3.attention.head_count_kv", kv_u32(4)),
            ("qwen3.feed_forward_length", kv_u32(5632)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let config = ModelConfig::from_gguf(&gguf).unwrap();

        assert_eq!(config.arch, ModelArch::Qwen3);
        assert_eq!(config.arch_name, "qwen3");
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.head_dim, 128); // 2048 / 16
        assert_eq!(config.ffn_hidden, 5632);

        // Qwen3 defaults
        assert_eq!(config.norm_type, NormType::RMSNorm);
        assert_eq!(config.activation, Activation::SwiGLU);
        assert_eq!(config.position_type, PositionType::RoPE);
        assert!(config.has_ffn_gate);
        assert!(!config.has_bias);
        assert!(config.causal);
        assert!(config.pre_norm);
        assert!(config.rope_neox);
        assert!((config.embedding_scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_phi35_longrope_config() {
        // Phi-3.5 includes LongRoPE scaling metadata
        let kv = vec![
            ("general.architecture", kv_string("phi3")),
            ("phi3.embedding_length", kv_u32(3072)),
            ("phi3.block_count", kv_u32(32)),
            ("phi3.attention.head_count", kv_u32(32)),
            ("phi3.feed_forward_length", kv_u32(8192)),
            ("phi3.context_length", kv_u32(131072)),
            ("phi3.rope.scaling.original_context_length", kv_u32(4096)),
            ("phi3.rope.scaling.attn_factor", kv_f32(1.190238)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);
        let config = ModelConfig::from_gguf(&gguf).unwrap();

        assert_eq!(config.arch, ModelArch::Phi3);
        assert_eq!(config.rope_scaling_original_ctx, 4096);
        assert!((config.rope_scaling_attn_factor - 1.190238).abs() < 1e-5);
        assert_eq!(config.max_seq_len, 131072);
        assert!(config.rope_neox);
    }
}
