// M4.2: Model weight loading from GGUF tensors.

use tracing::{debug, info};

use crate::backend::{ComputeBackend, DeviceTensor};
use crate::error::InferenceError;
use crate::gguf::GgufFile;
use crate::gguf::quant::GgufTensorType;
use crate::gguf::tensor::load_tensor_by_name;
use crate::tensor::{Tensor, TensorDtype};
use super::config::ModelConfig;
use super::config::PositionType;

/// Weights for a single transformer layer.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    // Pre-norm weights (Gemma/LLaMA only, None for BERT)
    pub attn_norm_w: Option<DeviceTensor>,
    pub attn_norm_b: Option<DeviceTensor>,
    pub ffn_norm_w: Option<DeviceTensor>,
    pub ffn_norm_b: Option<DeviceTensor>,

    // Post-norm weights (BERT only, None for Gemma/LLaMA)
    pub attn_output_norm_w: Option<DeviceTensor>,
    pub attn_output_norm_b: Option<DeviceTensor>,
    pub ffn_output_norm_w: Option<DeviceTensor>,
    pub ffn_output_norm_b: Option<DeviceTensor>,

    // Attention projections (always required)
    pub attn_q: DeviceTensor,
    pub attn_k: DeviceTensor,
    pub attn_v: DeviceTensor,
    pub attn_output: DeviceTensor,

    // Attention biases (BERT only)
    pub attn_q_bias: Option<DeviceTensor>,
    pub attn_k_bias: Option<DeviceTensor>,
    pub attn_v_bias: Option<DeviceTensor>,
    pub attn_output_bias: Option<DeviceTensor>,

    // FFN projections (always required)
    pub ffn_up: DeviceTensor,
    pub ffn_down: DeviceTensor,
    pub ffn_gate: Option<DeviceTensor>,

    // FFN biases (BERT only)
    pub ffn_up_bias: Option<DeviceTensor>,
    pub ffn_down_bias: Option<DeviceTensor>,
}

/// All model weights loaded from a GGUF file.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Token embedding table: [vocab_size, hidden_size]
    pub token_embedding: DeviceTensor,

    /// Position embedding table (BERT only): [max_seq_len, hidden_size]
    pub position_embedding: Option<DeviceTensor>,

    /// Embedding normalization (BERT only)
    pub embedding_norm_w: Option<DeviceTensor>,
    pub embedding_norm_b: Option<DeviceTensor>,

    /// Per-layer transformer weights
    pub layers: Vec<LayerWeights>,

    /// Final output normalization weight: [hidden_size]
    pub output_norm_w: DeviceTensor,
    pub output_norm_b: Option<DeviceTensor>,
}

// ---------------------------------------------------------------------------
// GgufTensorType -> TensorDtype mapping
// ---------------------------------------------------------------------------

/// Map a GGUF tensor type to our internal TensorDtype.
///
/// Only F32, F16, Q8_0, and Q4_0 are supported for inference. Other types
/// produce an error.
fn gguf_dtype_to_tensor_dtype(dtype: GgufTensorType) -> Result<TensorDtype, InferenceError> {
    match dtype {
        GgufTensorType::F32 => Ok(TensorDtype::F32),
        GgufTensorType::F16 => Ok(TensorDtype::F16),
        GgufTensorType::Q8_0 => Ok(TensorDtype::Q8_0),
        GgufTensorType::Q4_0 => Ok(TensorDtype::Q4_0),
        other => Err(InferenceError::Model(format!(
            "unsupported tensor dtype for inference: {}",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tensor loading helpers
// ---------------------------------------------------------------------------

/// Load a GGUF tensor by name, convert it to a [`Tensor`], and upload to the device.
///
/// The tensor data is kept in its original format:
/// - F32 data is stored as `TensorStorage::F32`
/// - F16 data is stored as `TensorStorage::F16` (raw u16 bits)
/// - Q8_0 / Q4_0 data is stored as `TensorStorage::Quantized` (raw bytes)
///
/// Quantized weights are NOT dequantized here; the fused `quantized_matmul`
/// handles dequantization during computation.
fn load_tensor(
    gguf: &GgufFile,
    name: &str,
    backend: &dyn ComputeBackend,
) -> Result<DeviceTensor, InferenceError> {
    let gguf_tensor = load_tensor_by_name(gguf, name)?;

    let tensor_dtype = gguf_dtype_to_tensor_dtype(gguf_tensor.dtype)?;

    // Convert GGUF dims (u64) to Tensor shape (usize). GGUF stores dims with
    // the innermost (fastest-varying) dimension first. For a 2D weight matrix
    // with GGUF dims [cols, rows], our row-major Tensor shape is [rows, cols].
    let shape: Vec<usize> = gguf_tensor.shape.iter().rev().map(|&d| d as usize).collect();

    debug!(
        tensor = name,
        ?shape,
        dtype = ?tensor_dtype,
        bytes = gguf_tensor.data.len(),
        "Loading tensor"
    );

    let tensor = match tensor_dtype {
        TensorDtype::F32 => {
            let f32_data = gguf_tensor.to_f32()?;
            Tensor::new(shape, f32_data)
        }
        TensorDtype::F16 => {
            let n = gguf_tensor.n_elements() as usize;
            let mut u16_data = Vec::with_capacity(n);
            for chunk in gguf_tensor.data.chunks_exact(2) {
                u16_data.push(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
            Tensor::from_f16(shape, u16_data)
        }
        TensorDtype::Q8_0 | TensorDtype::Q4_0 => {
            Tensor::from_quantized(shape, tensor_dtype, gguf_tensor.data.to_vec())
        }
    };

    Ok(backend.upload(&tensor))
}

/// Load an optional tensor by name. Returns `Ok(None)` if the tensor is not
/// found in the GGUF file, and propagates other errors.
fn load_tensor_optional(
    gguf: &GgufFile,
    name: &str,
    backend: &dyn ComputeBackend,
) -> Result<Option<DeviceTensor>, InferenceError> {
    match gguf.find_tensor(name) {
        Some(_) => load_tensor(gguf, name, backend).map(Some),
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// ModelWeights loading
// ---------------------------------------------------------------------------

impl ModelWeights {
    /// Load all model weights from a parsed GGUF file onto a compute device.
    ///
    /// The tensor name patterns follow the GGUF naming convention:
    /// - `token_embd.weight` for token embeddings
    /// - `blk.{i}.*` for per-layer weights
    /// - `output_norm.weight` for the final normalization
    ///
    /// Quantized tensors (Q8_0, Q4_0) are kept in quantized format and NOT
    /// dequantized to f32. The backend's `quantized_matmul` operation handles
    /// dequantization during computation (fused dequant+dot product).
    pub fn from_gguf(
        gguf: &GgufFile,
        config: &ModelConfig,
        backend: &dyn ComputeBackend,
    ) -> Result<ModelWeights, InferenceError> {
        info!(
            arch = ?config.arch,
            num_layers = config.num_layers,
            hidden_size = config.hidden_size,
            "Loading model weights from GGUF"
        );

        // -- Shared weights --

        let token_embedding = load_tensor(gguf, "token_embd.weight", backend)?;

        // Position embeddings (BERT with learned positions only)
        let position_embedding = if config.position_type == PositionType::Learned {
            Some(load_tensor(gguf, "position_embd.weight", backend)?)
        } else {
            None
        };

        // Embedding normalization (BERT only)
        let embedding_norm_w = if config.has_bias {
            load_tensor_optional(gguf, "token_embd_norm.weight", backend)?
        } else {
            None
        };
        let embedding_norm_b = if config.has_bias {
            load_tensor_optional(gguf, "token_embd_norm.bias", backend)?
        } else {
            None
        };

        // -- Per-layer weights --

        let mut layers = Vec::with_capacity(config.num_layers);

        for i in 0..config.num_layers {
            info!("Loading layer {}/{}...", i + 1, config.num_layers);

            let prefix = format!("blk.{}", i);

            // Pre-norm weights (Gemma/LLaMA: required, BERT: None)
            let (attn_norm_w, attn_norm_b, ffn_norm_w, ffn_norm_b) = if config.pre_norm {
                let w = Some(load_tensor(
                    gguf,
                    &format!("{}.attn_norm.weight", prefix),
                    backend,
                )?);
                let b = None; // RMSNorm has no bias
                let fw = Some(load_tensor(
                    gguf,
                    &format!("{}.ffn_norm.weight", prefix),
                    backend,
                )?);
                let fb = None;
                (w, b, fw, fb)
            } else {
                // BERT (post-norm): no pre-attention or pre-FFN norms
                (None, None, None, None)
            };

            // Attention projections
            let attn_q = load_tensor(
                gguf,
                &format!("{}.attn_q.weight", prefix),
                backend,
            )?;
            let attn_k = load_tensor(
                gguf,
                &format!("{}.attn_k.weight", prefix),
                backend,
            )?;
            let attn_v = load_tensor(
                gguf,
                &format!("{}.attn_v.weight", prefix),
                backend,
            )?;
            let attn_output = load_tensor(
                gguf,
                &format!("{}.attn_output.weight", prefix),
                backend,
            )?;

            // Attention biases (BERT only)
            let attn_q_bias = if config.has_bias {
                load_tensor_optional(
                    gguf,
                    &format!("{}.attn_q.bias", prefix),
                    backend,
                )?
            } else {
                None
            };
            let attn_k_bias = if config.has_bias {
                load_tensor_optional(
                    gguf,
                    &format!("{}.attn_k.bias", prefix),
                    backend,
                )?
            } else {
                None
            };
            let attn_v_bias = if config.has_bias {
                load_tensor_optional(
                    gguf,
                    &format!("{}.attn_v.bias", prefix),
                    backend,
                )?
            } else {
                None
            };
            let attn_output_bias = if config.has_bias {
                load_tensor_optional(
                    gguf,
                    &format!("{}.attn_output.bias", prefix),
                    backend,
                )?
            } else {
                None
            };

            // Post-norm weights (BERT only, None for Gemma/LLaMA)
            let (attn_output_norm_w, attn_output_norm_b, ffn_output_norm_w, ffn_output_norm_b) =
                if !config.pre_norm {
                    let aonw = load_tensor_optional(
                        gguf,
                        &format!("{}.attn_output_norm.weight", prefix),
                        backend,
                    )?;
                    let aonb = load_tensor_optional(
                        gguf,
                        &format!("{}.attn_output_norm.bias", prefix),
                        backend,
                    )?;
                    let fonw = load_tensor_optional(
                        gguf,
                        &format!("{}.layer_output_norm.weight", prefix),
                        backend,
                    )?;
                    let fonb = load_tensor_optional(
                        gguf,
                        &format!("{}.layer_output_norm.bias", prefix),
                        backend,
                    )?;
                    (aonw, aonb, fonw, fonb)
                } else {
                    (None, None, None, None)
                };

            // FFN projections
            let ffn_up = load_tensor(
                gguf,
                &format!("{}.ffn_up.weight", prefix),
                backend,
            )?;
            let ffn_down = load_tensor(
                gguf,
                &format!("{}.ffn_down.weight", prefix),
                backend,
            )?;

            // FFN gate (SwiGLU architectures only: Gemma, LLaMA)
            let ffn_gate = if config.has_ffn_gate {
                Some(load_tensor(
                    gguf,
                    &format!("{}.ffn_gate.weight", prefix),
                    backend,
                )?)
            } else {
                None
            };

            // FFN biases (BERT only)
            let ffn_up_bias = if config.has_bias {
                load_tensor_optional(
                    gguf,
                    &format!("{}.ffn_up.bias", prefix),
                    backend,
                )?
            } else {
                None
            };
            let ffn_down_bias = if config.has_bias {
                load_tensor_optional(
                    gguf,
                    &format!("{}.ffn_down.bias", prefix),
                    backend,
                )?
            } else {
                None
            };

            layers.push(LayerWeights {
                attn_norm_w,
                attn_norm_b,
                ffn_norm_w,
                ffn_norm_b,
                attn_output_norm_w,
                attn_output_norm_b,
                ffn_output_norm_w,
                ffn_output_norm_b,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                attn_q_bias,
                attn_k_bias,
                attn_v_bias,
                attn_output_bias,
                ffn_up,
                ffn_down,
                ffn_gate,
                ffn_up_bias,
                ffn_down_bias,
            });
        }

        // -- Output normalization --

        let output_norm_w = load_tensor(gguf, "output_norm.weight", backend)?;
        let output_norm_b = if config.has_bias {
            load_tensor_optional(gguf, "output_norm.bias", backend)?
        } else {
            None
        };

        info!(
            num_layers = layers.len(),
            "Model weights loaded successfully"
        );

        Ok(ModelWeights {
            token_embedding,
            position_embedding,
            embedding_norm_w,
            embedding_norm_b,
            layers,
            output_norm_w,
            output_norm_b,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::gguf::quant::f32_to_f16;
    use crate::model::config::{
        Activation, ModelArch, NormType, PoolingType,
    };

    // ====================================================================
    // GGUF builder helpers for tests
    // ====================================================================

    const GGUF_MAGIC: u32 = 0x4655_4747;

    /// Round `offset` up to the next multiple of `alignment`.
    fn align_up(offset: usize, alignment: usize) -> usize {
        let remainder = offset % alignment;
        if remainder == 0 {
            offset
        } else {
            offset + (alignment - remainder)
        }
    }

    /// Serialize a GGUF string KV value (type_id + string).
    fn kv_string(val: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&8u32.to_le_bytes()); // GGUF_TYPE_STRING
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val.as_bytes());
        buf
    }

    /// Build a GGUF file in memory with the given tensors.
    ///
    /// Each tensor is specified as (name, dims, dtype_id, tensor_data_bytes).
    /// The function computes correct offsets and alignment automatically.
    fn build_gguf_with_tensors(
        tensors: &[(&str, &[u64], u32, &[u8])],
    ) -> Vec<u8> {
        let alignment: u32 = 32;
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes()); // n_tensors
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_kv = 1

        // KV: general.architecture = "test"
        let key = "general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&kv_string("test"));

        // Compute tensor data offsets relative to data section start.
        // Tensors are laid out sequentially, each aligned to the boundary.
        let mut data_offsets = Vec::with_capacity(tensors.len());
        let mut current_offset: u64 = 0;
        for (i, &(_, _, _, data)) in tensors.iter().enumerate() {
            if i > 0 {
                // Align the current offset to the alignment boundary.
                current_offset = align_up(current_offset as usize, alignment as usize) as u64;
            }
            data_offsets.push(current_offset);
            current_offset += data.len() as u64;
        }

        // Tensor info entries
        for (i, &(name, dims, dtype_id, _)) in tensors.iter().enumerate() {
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&dtype_id.to_le_bytes());
            buf.extend_from_slice(&data_offsets[i].to_le_bytes());
        }

        // Pad header to alignment boundary
        let aligned_header = align_up(buf.len(), alignment as usize);
        buf.resize(aligned_header, 0);

        // Write tensor data with alignment padding between tensors
        for (i, &(_, _, _, data)) in tensors.iter().enumerate() {
            if i > 0 {
                let aligned = align_up(buf.len() - aligned_header, alignment as usize)
                    + aligned_header;
                buf.resize(aligned, 0);
            }
            buf.extend_from_slice(data);
        }

        // Pad to final alignment
        let final_aligned = align_up(buf.len(), alignment as usize);
        buf.resize(final_aligned, 0);

        buf
    }

    /// Write a GGUF byte buffer to a temp file and return the path.
    fn write_temp_gguf(name: &str, data: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("test_weights_{}.gguf", name));
        std::fs::write(&path, data).unwrap();
        path
    }

    /// Create F32 tensor data from a flat slice of f32 values.
    fn f32_tensor_data(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Create a Q8_0 tensor data from element count (fills with known pattern).
    /// Each block has scale=1.0 and values [0, 1, 2, ..., 31].
    fn q8_0_tensor_data(n_elements: usize) -> Vec<u8> {
        assert_eq!(n_elements % 32, 0, "Q8_0 requires multiple of 32 elements");
        let n_blocks = n_elements / 32;
        let mut data = Vec::with_capacity(n_blocks * 34);
        let scale_bits = f32_to_f16(1.0);
        for _ in 0..n_blocks {
            data.extend_from_slice(&scale_bits.to_le_bytes());
            for i in 0..32u8 {
                data.push(i);
            }
        }
        data
    }

    /// Build a minimal Gemma-style ModelConfig for testing.
    fn gemma_config(num_layers: usize, hidden_size: usize) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Gemma3,
            arch_name: "gemma3".to_string(),
            hidden_size,
            num_layers,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: hidden_size / 4,
            ffn_hidden: hidden_size * 4,
            vocab_size: 64,
            max_seq_len: 128,
            norm_type: NormType::RMSNorm,
            norm_eps: 1e-6,
            activation: Activation::SwiGLU,
            position_type: PositionType::RoPE,
            rope_freq_base: 10000.0,
            rope_dim: hidden_size / 4,
            causal: true,
            attn_logit_softcap: 0.0,
            attn_scale: None,
            embedding_scale: 1.0,
            pooling_type: PoolingType::Mean,
            has_ffn_gate: true,
            has_bias: false,
            pre_norm: true,
        }
    }

    /// Build a minimal BERT-style ModelConfig for testing.
    fn bert_config(num_layers: usize, hidden_size: usize) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Bert,
            arch_name: "bert".to_string(),
            hidden_size,
            num_layers,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: hidden_size / 4,
            ffn_hidden: hidden_size * 4,
            vocab_size: 64,
            max_seq_len: 128,
            norm_type: NormType::LayerNorm,
            norm_eps: 1e-12,
            activation: Activation::GELU,
            position_type: PositionType::Learned,
            rope_freq_base: 0.0,
            rope_dim: 0,
            causal: false,
            attn_logit_softcap: 0.0,
            attn_scale: None,
            embedding_scale: 1.0,
            pooling_type: PoolingType::Mean,
            has_ffn_gate: false,
            has_bias: true,
            pre_norm: false,
        }
    }

    /// Build the set of required Gemma-style tensors for a given config.
    ///
    /// Returns a list of (name, dims, dtype_id, data) tuples suitable for
    /// `build_gguf_with_tensors()`.
    fn build_gemma_tensors(config: &ModelConfig) -> Vec<(String, Vec<u64>, u32, Vec<u8>)> {
        let h = config.hidden_size;
        let ffn = config.ffn_hidden;
        let v = config.vocab_size;

        let mut tensors = Vec::new();

        // Token embedding: GGUF dims [hidden_size, vocab_size] (col-major order)
        tensors.push((
            "token_embd.weight".to_string(),
            vec![h as u64, v as u64],
            0, // F32
            f32_tensor_data(&vec![0.1f32; v * h]),
        ));

        // Output norm
        tensors.push((
            "output_norm.weight".to_string(),
            vec![h as u64],
            0,
            f32_tensor_data(&vec![1.0f32; h]),
        ));

        // Per-layer tensors
        for i in 0..config.num_layers {
            // attn_norm.weight [h]
            tensors.push((
                format!("blk.{}.attn_norm.weight", i),
                vec![h as u64],
                0,
                f32_tensor_data(&vec![1.0f32; h]),
            ));
            // attn_q.weight [h, h] -> GGUF dims [h, h]
            tensors.push((
                format!("blk.{}.attn_q.weight", i),
                vec![h as u64, h as u64],
                0,
                f32_tensor_data(&vec![0.01f32; h * h]),
            ));
            // attn_k.weight
            tensors.push((
                format!("blk.{}.attn_k.weight", i),
                vec![h as u64, h as u64],
                0,
                f32_tensor_data(&vec![0.01f32; h * h]),
            ));
            // attn_v.weight
            tensors.push((
                format!("blk.{}.attn_v.weight", i),
                vec![h as u64, h as u64],
                0,
                f32_tensor_data(&vec![0.01f32; h * h]),
            ));
            // attn_output.weight
            tensors.push((
                format!("blk.{}.attn_output.weight", i),
                vec![h as u64, h as u64],
                0,
                f32_tensor_data(&vec![0.01f32; h * h]),
            ));
            // ffn_norm.weight [h]
            tensors.push((
                format!("blk.{}.ffn_norm.weight", i),
                vec![h as u64],
                0,
                f32_tensor_data(&vec![1.0f32; h]),
            ));
            // ffn_up.weight [h, ffn]
            tensors.push((
                format!("blk.{}.ffn_up.weight", i),
                vec![h as u64, ffn as u64],
                0,
                f32_tensor_data(&vec![0.01f32; h * ffn]),
            ));
            // ffn_down.weight [ffn, h]
            tensors.push((
                format!("blk.{}.ffn_down.weight", i),
                vec![ffn as u64, h as u64],
                0,
                f32_tensor_data(&vec![0.01f32; h * ffn]),
            ));
            // ffn_gate.weight [h, ffn] (Gemma has SwiGLU gate)
            tensors.push((
                format!("blk.{}.ffn_gate.weight", i),
                vec![h as u64, ffn as u64],
                0,
                f32_tensor_data(&vec![0.01f32; h * ffn]),
            ));
        }

        tensors
    }

    // ====================================================================
    // Tests
    // ====================================================================

    #[test]
    fn test_gguf_dtype_to_tensor_dtype() {
        assert_eq!(gguf_dtype_to_tensor_dtype(GgufTensorType::F32).unwrap(), TensorDtype::F32);
        assert_eq!(gguf_dtype_to_tensor_dtype(GgufTensorType::F16).unwrap(), TensorDtype::F16);
        assert_eq!(gguf_dtype_to_tensor_dtype(GgufTensorType::Q8_0).unwrap(), TensorDtype::Q8_0);
        assert_eq!(gguf_dtype_to_tensor_dtype(GgufTensorType::Q4_0).unwrap(), TensorDtype::Q4_0);
        // Unsupported types should error
        assert!(gguf_dtype_to_tensor_dtype(GgufTensorType::Q4_1).is_err());
        assert!(gguf_dtype_to_tensor_dtype(GgufTensorType::BF16).is_err());
    }

    #[test]
    fn test_load_gemma_style_weights() {
        let config = gemma_config(1, 32);
        let tensor_specs = build_gemma_tensors(&config);

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensor_specs
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("gemma_1layer", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        // Verify structure
        assert_eq!(weights.layers.len(), 1);
        assert!(weights.position_embedding.is_none()); // Gemma uses RoPE
        assert!(weights.embedding_norm_w.is_none());
        assert!(weights.embedding_norm_b.is_none());
        assert!(weights.output_norm_b.is_none());

        // Layer should have gate (SwiGLU) but no biases
        let layer = &weights.layers[0];
        assert!(layer.ffn_gate.is_some());
        assert!(layer.attn_norm_b.is_none());
        assert!(layer.attn_q_bias.is_none());
        assert!(layer.attn_k_bias.is_none());
        assert!(layer.attn_v_bias.is_none());
        assert!(layer.attn_output_bias.is_none());
        assert!(layer.attn_output_norm_w.is_none());
        assert!(layer.attn_output_norm_b.is_none());
        assert!(layer.ffn_norm_b.is_none());
        assert!(layer.ffn_up_bias.is_none());
        assert!(layer.ffn_down_bias.is_none());
        assert!(layer.ffn_output_norm_w.is_none());
        assert!(layer.ffn_output_norm_b.is_none());

        // Verify token embedding shape: [vocab_size, hidden_size]
        assert_eq!(weights.token_embedding.shape(), &[64, 32]);

        // Verify output norm shape: [hidden_size]
        assert_eq!(weights.output_norm_w.shape(), &[32]);

        // Verify attention weight shapes: [hidden_size, hidden_size]
        assert_eq!(layer.attn_q.shape(), &[32, 32]);
        assert_eq!(layer.attn_k.shape(), &[32, 32]);
        assert_eq!(layer.attn_v.shape(), &[32, 32]);
        assert_eq!(layer.attn_output.shape(), &[32, 32]);

        // Verify norm weight shapes: [hidden_size] (pre-norm, so Some)
        assert_eq!(layer.attn_norm_w.as_ref().unwrap().shape(), &[32]);
        assert_eq!(layer.ffn_norm_w.as_ref().unwrap().shape(), &[32]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_multiple_layers() {
        let config = gemma_config(3, 32);
        let tensor_specs = build_gemma_tensors(&config);

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensor_specs
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("gemma_3layers", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        assert_eq!(weights.layers.len(), 3);
        for layer in &weights.layers {
            assert!(layer.ffn_gate.is_some());
            assert!(layer.attn_q_bias.is_none());
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_missing_required_tensor_returns_error() {
        // Build a GGUF file missing the attn_q.weight for layer 0.
        let config = gemma_config(1, 32);
        let tensor_specs = build_gemma_tensors(&config);

        // Filter out the attn_q.weight tensor
        let filtered: Vec<_> = tensor_specs
            .iter()
            .filter(|(name, _, _, _)| name != "blk.0.attn_q.weight")
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&filtered);
        let path = write_temp_gguf("missing_tensor", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let result = ModelWeights::from_gguf(&gguf, &config, &backend);

        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::TensorNotFound(name) => {
                assert_eq!(name, "blk.0.attn_q.weight");
            }
            e => panic!("expected TensorNotFound, got {:?}", e),
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_missing_token_embedding_returns_error() {
        // A GGUF with no tensors at all
        let gguf_bytes = build_gguf_with_tensors(&[]);
        let path = write_temp_gguf("no_tensors", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let config = gemma_config(0, 32);
        let result = ModelWeights::from_gguf(&gguf, &config, &backend);

        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::TensorNotFound(name) => {
                assert_eq!(name, "token_embd.weight");
            }
            e => panic!("expected TensorNotFound for token_embd.weight, got {:?}", e),
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_quantized_tensors_stay_quantized() {
        // Build a GGUF with Q8_0 attention weights to verify they stay quantized.
        let config = gemma_config(1, 32);
        let h = config.hidden_size;
        let ffn = config.ffn_hidden;
        let v = config.vocab_size;

        // Use Q8_0 for the attention weights (Q8_0 dtype_id = 8)
        let q8_weight_data = q8_0_tensor_data(h * h);

        let mut tensor_specs: Vec<(&str, &[u64], u32, Vec<u8>)> = Vec::new();

        // F32 token embedding
        let embd_dims = vec![h as u64, v as u64];
        let embd_data = f32_tensor_data(&vec![0.1f32; v * h]);
        tensor_specs.push(("token_embd.weight", &embd_dims, 0, embd_data));

        // F32 output norm
        let norm_dims = vec![h as u64];
        let norm_data = f32_tensor_data(&vec![1.0f32; h]);
        tensor_specs.push(("output_norm.weight", &norm_dims, 0, norm_data.clone()));

        // Layer 0 norms (F32)
        let h_dims = vec![h as u64];
        tensor_specs.push(("blk.0.attn_norm.weight", &h_dims, 0, norm_data.clone()));
        tensor_specs.push(("blk.0.ffn_norm.weight", &h_dims, 0, norm_data.clone()));

        // Layer 0 attention weights: Q8_0
        let hh_dims = vec![h as u64, h as u64];
        tensor_specs.push(("blk.0.attn_q.weight", &hh_dims, 8, q8_weight_data.clone()));
        tensor_specs.push(("blk.0.attn_k.weight", &hh_dims, 8, q8_weight_data.clone()));
        tensor_specs.push(("blk.0.attn_v.weight", &hh_dims, 8, q8_weight_data.clone()));
        tensor_specs.push(("blk.0.attn_output.weight", &hh_dims, 8, q8_weight_data.clone()));

        // Layer 0 FFN weights (F32)
        let ffn_up_dims = vec![h as u64, ffn as u64];
        let ffn_down_dims = vec![ffn as u64, h as u64];
        let ffn_up_data = f32_tensor_data(&vec![0.01f32; h * ffn]);
        let ffn_down_data = f32_tensor_data(&vec![0.01f32; h * ffn]);
        let ffn_gate_data = f32_tensor_data(&vec![0.01f32; h * ffn]);
        tensor_specs.push(("blk.0.ffn_up.weight", &ffn_up_dims, 0, ffn_up_data));
        tensor_specs.push(("blk.0.ffn_down.weight", &ffn_down_dims, 0, ffn_down_data));
        tensor_specs.push(("blk.0.ffn_gate.weight", &ffn_up_dims, 0, ffn_gate_data));

        // Build the tensor refs with correct lifetime handling
        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensor_specs
            .iter()
            .map(|(name, dims, dtype, data)| (*name, *dims, *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("quantized_stay", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        // The Q8_0 weights should remain quantized
        let layer = &weights.layers[0];
        assert_eq!(layer.attn_q.dtype(), TensorDtype::Q8_0);
        assert_eq!(layer.attn_k.dtype(), TensorDtype::Q8_0);
        assert_eq!(layer.attn_v.dtype(), TensorDtype::Q8_0);
        assert_eq!(layer.attn_output.dtype(), TensorDtype::Q8_0);

        // F32 weights should remain F32
        assert_eq!(layer.attn_norm_w.as_ref().unwrap().dtype(), TensorDtype::F32);
        assert_eq!(layer.ffn_norm_w.as_ref().unwrap().dtype(), TensorDtype::F32);
        assert_eq!(weights.token_embedding.dtype(), TensorDtype::F32);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_optional_tensor_loading_with_bias() {
        // Build a BERT-style model with biases.
        let config = bert_config(1, 32);
        let h = config.hidden_size;
        let ffn = config.ffn_hidden;
        let v = config.vocab_size;
        let max_seq = config.max_seq_len;

        let mut tensor_specs: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();

        // Token embedding
        tensor_specs.push((
            "token_embd.weight".to_string(),
            vec![h as u64, v as u64],
            0,
            f32_tensor_data(&vec![0.1f32; v * h]),
        ));

        // Position embedding (BERT uses learned positions)
        tensor_specs.push((
            "position_embd.weight".to_string(),
            vec![h as u64, max_seq as u64],
            0,
            f32_tensor_data(&vec![0.01f32; max_seq * h]),
        ));

        // Embedding norms
        tensor_specs.push((
            "token_embd_norm.weight".to_string(),
            vec![h as u64],
            0,
            f32_tensor_data(&vec![1.0f32; h]),
        ));
        tensor_specs.push((
            "token_embd_norm.bias".to_string(),
            vec![h as u64],
            0,
            f32_tensor_data(&vec![0.0f32; h]),
        ));

        // Output norm
        tensor_specs.push((
            "output_norm.weight".to_string(),
            vec![h as u64],
            0,
            f32_tensor_data(&vec![1.0f32; h]),
        ));
        tensor_specs.push((
            "output_norm.bias".to_string(),
            vec![h as u64],
            0,
            f32_tensor_data(&vec![0.0f32; h]),
        ));

        // Layer 0: all weights + biases + extra norms
        let prefix = "blk.0";

        // attn_norm
        tensor_specs.push((format!("{}.attn_norm.weight", prefix), vec![h as u64], 0, f32_tensor_data(&vec![1.0f32; h])));
        tensor_specs.push((format!("{}.attn_norm.bias", prefix), vec![h as u64], 0, f32_tensor_data(&vec![0.0f32; h])));

        // attn weights + biases
        for name in &["attn_q", "attn_k", "attn_v", "attn_output"] {
            tensor_specs.push((format!("{}.{}.weight", prefix, name), vec![h as u64, h as u64], 0, f32_tensor_data(&vec![0.01f32; h * h])));
            tensor_specs.push((format!("{}.{}.bias", prefix, name), vec![h as u64], 0, f32_tensor_data(&vec![0.0f32; h])));
        }

        // attn_output_norm
        tensor_specs.push((format!("{}.attn_output_norm.weight", prefix), vec![h as u64], 0, f32_tensor_data(&vec![1.0f32; h])));
        tensor_specs.push((format!("{}.attn_output_norm.bias", prefix), vec![h as u64], 0, f32_tensor_data(&vec![0.0f32; h])));

        // ffn_norm
        tensor_specs.push((format!("{}.ffn_norm.weight", prefix), vec![h as u64], 0, f32_tensor_data(&vec![1.0f32; h])));
        tensor_specs.push((format!("{}.ffn_norm.bias", prefix), vec![h as u64], 0, f32_tensor_data(&vec![0.0f32; h])));

        // ffn_up + ffn_down (no gate for BERT)
        tensor_specs.push((format!("{}.ffn_up.weight", prefix), vec![h as u64, ffn as u64], 0, f32_tensor_data(&vec![0.01f32; h * ffn])));
        tensor_specs.push((format!("{}.ffn_up.bias", prefix), vec![ffn as u64], 0, f32_tensor_data(&vec![0.0f32; ffn])));
        tensor_specs.push((format!("{}.ffn_down.weight", prefix), vec![ffn as u64, h as u64], 0, f32_tensor_data(&vec![0.01f32; h * ffn])));
        tensor_specs.push((format!("{}.ffn_down.bias", prefix), vec![h as u64], 0, f32_tensor_data(&vec![0.0f32; h])));

        // layer_output_norm
        tensor_specs.push((format!("{}.layer_output_norm.weight", prefix), vec![h as u64], 0, f32_tensor_data(&vec![1.0f32; h])));
        tensor_specs.push((format!("{}.layer_output_norm.bias", prefix), vec![h as u64], 0, f32_tensor_data(&vec![0.0f32; h])));

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensor_specs
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("bert_with_bias", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        // BERT-specific checks
        assert!(weights.position_embedding.is_some());
        assert!(weights.embedding_norm_w.is_some());
        assert!(weights.embedding_norm_b.is_some());
        assert!(weights.output_norm_b.is_some());

        let layer = &weights.layers[0];
        // BERT has pre_norm=false, so pre-norm weights should be None
        assert!(layer.attn_norm_w.is_none());
        assert!(layer.attn_norm_b.is_none());
        assert!(layer.ffn_norm_w.is_none());
        assert!(layer.ffn_norm_b.is_none());

        // BERT attention biases
        assert!(layer.attn_q_bias.is_some());
        assert!(layer.attn_k_bias.is_some());
        assert!(layer.attn_v_bias.is_some());
        assert!(layer.attn_output_bias.is_some());

        // BERT post-norm weights
        assert!(layer.attn_output_norm_w.is_some());
        assert!(layer.attn_output_norm_b.is_some());
        assert!(layer.ffn_output_norm_w.is_some());
        assert!(layer.ffn_output_norm_b.is_some());

        // BERT FFN biases
        assert!(layer.ffn_up_bias.is_some());
        assert!(layer.ffn_down_bias.is_some());

        // BERT should NOT have ffn_gate
        assert!(layer.ffn_gate.is_none());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_optional_bias_absent_for_gemma() {
        // Gemma config has has_bias=false, so all optional bias tensors should be None
        // even if the GGUF file happens to contain them (we skip the lookup entirely).
        let config = gemma_config(1, 32);
        let tensor_specs = build_gemma_tensors(&config);

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensor_specs
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("gemma_no_bias", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        let layer = &weights.layers[0];
        // Gemma has pre_norm=true, so pre-norm weights should be present
        assert!(layer.attn_norm_w.is_some());
        assert!(layer.ffn_norm_w.is_some());
        // But no bias (RMSNorm)
        assert!(layer.attn_norm_b.is_none());
        assert!(layer.ffn_norm_b.is_none());
        // No attention biases
        assert!(layer.attn_q_bias.is_none());
        // No post-norm weights
        assert!(layer.ffn_up_bias.is_none());
        assert!(layer.ffn_down_bias.is_none());
        assert!(layer.ffn_output_norm_w.is_none());
        assert!(layer.ffn_output_norm_b.is_none());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_f16_tensor_weights() {
        // Build a minimal model with F16 (dtype_id=1) norm weights.
        let config = gemma_config(1, 32);
        let h = config.hidden_size;
        let ffn = config.ffn_hidden;
        let v = config.vocab_size;

        let mut tensor_specs: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();

        // Token embedding (F32)
        tensor_specs.push((
            "token_embd.weight".to_string(),
            vec![h as u64, v as u64],
            0,
            f32_tensor_data(&vec![0.1f32; v * h]),
        ));

        // Output norm (F16: dtype_id = 1)
        let f16_norm: Vec<u8> = (0..h)
            .flat_map(|_| f32_to_f16(1.0).to_le_bytes().to_vec())
            .collect();
        tensor_specs.push((
            "output_norm.weight".to_string(),
            vec![h as u64],
            1,
            f16_norm.clone(),
        ));

        // Layer 0
        tensor_specs.push((
            "blk.0.attn_norm.weight".to_string(),
            vec![h as u64],
            1,
            f16_norm.clone(),
        ));
        tensor_specs.push((
            "blk.0.ffn_norm.weight".to_string(),
            vec![h as u64],
            1,
            f16_norm.clone(),
        ));

        // Attention + FFN weights (F32)
        for name in &["attn_q", "attn_k", "attn_v", "attn_output"] {
            tensor_specs.push((
                format!("blk.0.{}.weight", name),
                vec![h as u64, h as u64],
                0,
                f32_tensor_data(&vec![0.01f32; h * h]),
            ));
        }
        tensor_specs.push((
            "blk.0.ffn_up.weight".to_string(),
            vec![h as u64, ffn as u64],
            0,
            f32_tensor_data(&vec![0.01f32; h * ffn]),
        ));
        tensor_specs.push((
            "blk.0.ffn_down.weight".to_string(),
            vec![ffn as u64, h as u64],
            0,
            f32_tensor_data(&vec![0.01f32; h * ffn]),
        ));
        tensor_specs.push((
            "blk.0.ffn_gate.weight".to_string(),
            vec![h as u64, ffn as u64],
            0,
            f32_tensor_data(&vec![0.01f32; h * ffn]),
        ));

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensor_specs
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("f16_norms", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        // F16 tensors should be loaded as F16
        assert_eq!(weights.output_norm_w.dtype(), TensorDtype::F16);
        assert_eq!(weights.layers[0].attn_norm_w.as_ref().unwrap().dtype(), TensorDtype::F16);
        assert_eq!(weights.layers[0].ffn_norm_w.as_ref().unwrap().dtype(), TensorDtype::F16);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_missing_ffn_gate_for_non_swiglu_is_ok() {
        // A config with has_ffn_gate=false should NOT look for ffn_gate.weight.
        // Use a Gemma-like config (RoPE, no bias) but with has_ffn_gate=false
        // so that we don't need position_embd.weight and don't need ffn_gate.
        let mut config = gemma_config(0, 32);
        config.has_ffn_gate = false;
        let h = config.hidden_size;
        let v = config.vocab_size;

        let tensor_specs: Vec<(String, Vec<u64>, u32, Vec<u8>)> = vec![
            (
                "token_embd.weight".to_string(),
                vec![h as u64, v as u64],
                0,
                f32_tensor_data(&vec![0.1f32; v * h]),
            ),
            (
                "output_norm.weight".to_string(),
                vec![h as u64],
                0,
                f32_tensor_data(&vec![1.0f32; h]),
            ),
        ];

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensor_specs
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("no_gate_ok", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        // 0 layers, so we only need token_embd + output_norm
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        assert_eq!(weights.layers.len(), 0);
        assert!(weights.position_embedding.is_none()); // RoPE, no learned positions
        assert!(weights.embedding_norm_w.is_none());
        assert!(weights.output_norm_b.is_none());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_missing_ffn_gate_for_swiglu_returns_error() {
        // A config with has_ffn_gate=true but missing ffn_gate.weight should error.
        let config = gemma_config(1, 32);
        let tensor_specs = build_gemma_tensors(&config);

        // Filter out the ffn_gate.weight tensor
        let filtered: Vec<_> = tensor_specs
            .iter()
            .filter(|(name, _, _, _)| !name.contains("ffn_gate"))
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&filtered);
        let path = write_temp_gguf("missing_gate", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let result = ModelWeights::from_gguf(&gguf, &config, &backend);

        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::TensorNotFound(name) => {
                assert!(name.contains("ffn_gate"), "expected ffn_gate in error, got: {}", name);
            }
            e => panic!("expected TensorNotFound, got {:?}", e),
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_tensor_helper_f32() {
        let data = f32_tensor_data(&[1.0, 2.0, 3.0, 4.0]);
        let gguf_bytes = build_gguf_with_tensors(&[
            ("test.weight", &[4], 0, &data),
        ]);
        let path = write_temp_gguf("helper_f32", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let dt = load_tensor(&gguf, "test.weight", &backend).unwrap();

        assert_eq!(dt.dtype(), TensorDtype::F32);
        assert_eq!(dt.shape(), &[4]);
        assert_eq!(dt.tensor.as_f32(), &[1.0, 2.0, 3.0, 4.0]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_tensor_helper_q8_0() {
        let data = q8_0_tensor_data(32);
        let gguf_bytes = build_gguf_with_tensors(&[
            ("q8.weight", &[32], 8, &data),
        ]);
        let path = write_temp_gguf("helper_q8", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let dt = load_tensor(&gguf, "q8.weight", &backend).unwrap();

        assert_eq!(dt.dtype(), TensorDtype::Q8_0);
        assert_eq!(dt.shape(), &[32]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_tensor_optional_present() {
        let data = f32_tensor_data(&[1.0, 2.0]);
        let gguf_bytes = build_gguf_with_tensors(&[
            ("present.bias", &[2], 0, &data),
        ]);
        let path = write_temp_gguf("opt_present", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let result = load_tensor_optional(&gguf, "present.bias", &backend).unwrap();

        assert!(result.is_some());
        assert_eq!(result.unwrap().shape(), &[2]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_tensor_optional_absent() {
        let data = f32_tensor_data(&[1.0, 2.0]);
        let gguf_bytes = build_gguf_with_tensors(&[
            ("some.weight", &[2], 0, &data),
        ]);
        let path = write_temp_gguf("opt_absent", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let result = load_tensor_optional(&gguf, "missing.bias", &backend).unwrap();

        assert!(result.is_none());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_tensor_not_found() {
        let gguf_bytes = build_gguf_with_tensors(&[]);
        let path = write_temp_gguf("load_not_found", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let result = load_tensor(&gguf, "nonexistent", &backend);

        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::TensorNotFound(name) => {
                assert_eq!(name, "nonexistent");
            }
            e => panic!("expected TensorNotFound, got {:?}", e),
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_2d_tensor_shape_reversal() {
        // GGUF stores dims as [cols, rows] (innermost first).
        // Our Tensor shape should be [rows, cols] (row-major).
        let data = f32_tensor_data(&vec![0.0f32; 6]); // 2x3 = 6 elements
        let gguf_bytes = build_gguf_with_tensors(&[
            ("matrix", &[3, 2], 0, &data), // GGUF: [cols=3, rows=2]
        ]);
        let path = write_temp_gguf("shape_reversal", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let dt = load_tensor(&gguf, "matrix", &backend).unwrap();

        // Our shape should be reversed: [rows=2, cols=3]
        assert_eq!(dt.shape(), &[2, 3]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unsupported_gguf_dtype_returns_error() {
        // Q4_1 (dtype_id=3) is not supported for inference.
        // Q4_1 block: 20 bytes per 32 elements
        let data = vec![0u8; 20]; // 1 Q4_1 block
        let gguf_bytes = build_gguf_with_tensors(&[
            ("q4_1.weight", &[32], 3, &data),
        ]);
        let path = write_temp_gguf("unsupported_dtype", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let result = load_tensor(&gguf, "q4_1.weight", &backend);

        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("unsupported"), "Error: {}", err_msg);

        std::fs::remove_file(&path).ok();
    }
}
