// M4.2: Model weight loading from GGUF tensors.

use tracing::{debug, info};

use crate::backend::{ComputeBackend, DeviceTensor};
use crate::error::InferenceError;
use crate::gguf::GgufFile;
use crate::gguf::quant::GgufTensorType;
use crate::gguf::tensor::load_tensor_by_name;
use crate::tensor::{Tensor, TensorDtype};
use super::config::{ModelArch, ModelConfig, PositionType};

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

    // Per-head Q/K normalization (GemmaEmbedding, Qwen3)
    pub attn_q_norm_w: Option<DeviceTensor>,
    pub attn_k_norm_w: Option<DeviceTensor>,

    // Post-projection norms applied BEFORE residual (GemmaEmbedding only)
    pub attn_post_norm_w: Option<DeviceTensor>,
    pub ffn_post_norm_w: Option<DeviceTensor>,
}

/// All model weights loaded from a GGUF file.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Token embedding table: [vocab_size, hidden_size]
    pub token_embedding: DeviceTensor,

    /// Position embedding table (BERT only): [max_seq_len, hidden_size]
    pub position_embedding: Option<DeviceTensor>,

    /// Token type embedding table (BERT only): [num_types, hidden_size]
    /// Row 0 is added to all embeddings (hardcoded token_type=0, "Sentence A").
    pub token_type_embedding: Option<DeviceTensor>,

    /// Embedding normalization (BERT only)
    pub embedding_norm_w: Option<DeviceTensor>,
    pub embedding_norm_b: Option<DeviceTensor>,

    /// Per-layer transformer weights
    pub layers: Vec<LayerWeights>,

    /// Final output normalization weight: [hidden_size]
    /// `None` for BERT (which does per-layer post-norm instead of a global final norm).
    pub output_norm_w: Option<DeviceTensor>,
    pub output_norm_b: Option<DeviceTensor>,

    /// Output projection (lm_head): [vocab_size, hidden_size]
    /// When None, generation falls back to tied embeddings (token_embedding).
    pub output_projection: Option<DeviceTensor>,

    /// LongRoPE short frequency factors: [rope_dim/2], F32.
    /// Used when sequence length < rope_scaling_original_ctx.
    pub rope_factors_short: Option<DeviceTensor>,

    /// LongRoPE long frequency factors: [rope_dim/2], F32.
    /// Used when sequence length >= rope_scaling_original_ctx.
    pub rope_factors_long: Option<DeviceTensor>,
}

// ---------------------------------------------------------------------------
// GgufTensorType -> TensorDtype mapping
// ---------------------------------------------------------------------------

/// Map a GGUF tensor type to our internal TensorDtype.
fn gguf_dtype_to_tensor_dtype(dtype: GgufTensorType) -> Result<TensorDtype, InferenceError> {
    match dtype {
        GgufTensorType::F32 => Ok(TensorDtype::F32),
        GgufTensorType::F16 => Ok(TensorDtype::F16),
        GgufTensorType::Q8_0 => Ok(TensorDtype::Q8_0),
        GgufTensorType::Q4_0 => Ok(TensorDtype::Q4_0),
        GgufTensorType::Q4_1 => Ok(TensorDtype::Q4_1),
        GgufTensorType::Q5_0 => Ok(TensorDtype::Q5_0),
        GgufTensorType::Q5_1 => Ok(TensorDtype::Q5_1),
        GgufTensorType::Q4K => Ok(TensorDtype::Q4_K),
        GgufTensorType::Q5K => Ok(TensorDtype::Q5_K),
        GgufTensorType::Q6K => Ok(TensorDtype::Q6_K),
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
        _ => {
            // All remaining types are quantized (Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K)
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
// Fused QKV splitting helpers
// ---------------------------------------------------------------------------

/// Compute the byte size of `n_rows` in a quantized weight matrix with `k` columns.
///
/// For quantized formats, each row consists of `ceil(k / block_size)` blocks,
/// each of `block_byte_size` bytes. Rows are independently quantized.
fn quantized_row_bytes(dtype: TensorDtype, k: usize) -> usize {
    let block_size = dtype.block_size();
    let block_byte_size = dtype.block_byte_size();
    let blocks_per_row = (k + block_size - 1) / block_size;
    blocks_per_row * block_byte_size
}

/// Split a fused QKV weight tensor into separate Q, K, V tensors.
///
/// The fused tensor has shape `[n_embd + 2*n_kv, n_embd]` (after GGUF dim
/// reversal) where `n_kv = num_kv_heads * head_dim`. For standard MHA (GPT-2),
/// `n_kv == n_embd` so the shape is `[3*n_embd, n_embd]`.
///
/// For quantized tensors, we split at the raw byte level to keep the data
/// quantized. This avoids a massive F32 memory blowup for large models.
/// For F32/F16 tensors, we split the data directly.
fn split_qkv(
    qkv: &DeviceTensor,
    config: &ModelConfig,
    backend: &dyn ComputeBackend,
) -> (DeviceTensor, DeviceTensor, DeviceTensor) {
    let tensor = backend.download(qkv);
    let dtype = tensor.dtype();

    let n_embd = config.hidden_size;
    let n_kv = config.num_kv_heads * config.head_dim;

    if dtype.is_quantized() {
        // Split at the raw byte level, preserving quantization
        let raw = tensor.quantized_data();
        let k = n_embd; // input features (columns)
        let row_bytes = quantized_row_bytes(dtype, k);

        let q_rows = n_embd;
        let k_rows = n_kv;
        let v_rows = n_kv;
        let total_rows = q_rows + k_rows + v_rows;
        assert_eq!(
            raw.len(), total_rows * row_bytes,
            "fused QKV quantized data: {} bytes, expected {} = {} rows * {} bytes/row",
            raw.len(), total_rows * row_bytes, total_rows, row_bytes,
        );

        let q_end = q_rows * row_bytes;
        let k_end = (q_rows + k_rows) * row_bytes;

        let q_data = raw[..q_end].to_vec();
        let k_data = raw[q_end..k_end].to_vec();
        let v_data = raw[k_end..].to_vec();

        (
            backend.upload(&Tensor::from_quantized(vec![q_rows, k], dtype, q_data)),
            backend.upload(&Tensor::from_quantized(vec![k_rows, k], dtype, k_data)),
            backend.upload(&Tensor::from_quantized(vec![v_rows, k], dtype, v_data)),
        )
    } else {
        // F32/F16 path — dequantize and split as before
        let f32_tensor = tensor.to_f32();
        let f32_data = f32_tensor.as_f32();

        let expected = (n_embd + 2 * n_kv) * n_embd;
        assert_eq!(
            f32_data.len(), expected,
            "fused QKV weight has {} elements, expected {} = ({} + 2*{}) * {}",
            f32_data.len(), expected, n_embd, n_kv, n_embd,
        );

        let q_data = f32_data[..n_embd * n_embd].to_vec();
        let k_data = f32_data[n_embd * n_embd..(n_embd + n_kv) * n_embd].to_vec();
        let v_data = f32_data[(n_embd + n_kv) * n_embd..].to_vec();

        (
            backend.upload(&Tensor::new(vec![n_embd, n_embd], q_data)),
            backend.upload(&Tensor::new(vec![n_kv, n_embd], k_data)),
            backend.upload(&Tensor::new(vec![n_kv, n_embd], v_data)),
        )
    }
}

/// Split a fused QKV bias vector into separate Q, K, V biases.
///
/// The fused bias has shape `[n_embd + 2*n_kv]`.
fn split_qkv_bias(
    qkv_bias: &DeviceTensor,
    config: &ModelConfig,
    backend: &dyn ComputeBackend,
) -> (DeviceTensor, DeviceTensor, DeviceTensor) {
    let tensor = backend.download(qkv_bias);
    let f32_tensor = tensor.to_f32();
    let f32_data = f32_tensor.as_f32();

    let n_embd = config.hidden_size;
    let n_kv = config.num_kv_heads * config.head_dim;
    let expected = n_embd + 2 * n_kv;
    assert_eq!(
        f32_data.len(),
        expected,
        "fused QKV bias has {} elements, expected {} = {} + 2*{}",
        f32_data.len(), expected, n_embd, n_kv,
    );

    let q_data = f32_data[..n_embd].to_vec();
    let k_data = f32_data[n_embd..n_embd + n_kv].to_vec();
    let v_data = f32_data[n_embd + n_kv..].to_vec();

    (
        backend.upload(&Tensor::new(vec![n_embd], q_data)),
        backend.upload(&Tensor::new(vec![n_kv], k_data)),
        backend.upload(&Tensor::new(vec![n_kv], v_data)),
    )
}

/// Split a fused gate+up FFN weight tensor into separate gate and up tensors.
///
/// Some architectures (Phi-3) store `ffn_gate` and `ffn_up` as a single tensor
/// with shape `[2*ffn_hidden, hidden_size]`. The first half is the gate
/// projection, the second half is the up projection.
///
/// For quantized tensors, we split at the raw byte level to preserve
/// quantization and avoid massive F32 memory blowup.
fn split_gate_up(
    fused: &DeviceTensor,
    config: &ModelConfig,
    backend: &dyn ComputeBackend,
) -> (DeviceTensor, DeviceTensor) {
    let tensor = backend.download(fused);
    let dtype = tensor.dtype();

    let n_ff = config.ffn_hidden;
    let n_embd = config.hidden_size;

    if dtype.is_quantized() {
        // Split at the raw byte level, preserving quantization
        let raw = tensor.quantized_data();
        let k = n_embd; // input features (columns)
        let row_bytes = quantized_row_bytes(dtype, k);

        let total_rows = 2 * n_ff;
        assert_eq!(
            raw.len(), total_rows * row_bytes,
            "fused gate+up quantized data: {} bytes, expected {} = {} rows * {} bytes/row",
            raw.len(), total_rows * row_bytes, total_rows, row_bytes,
        );

        let gate_end = n_ff * row_bytes;
        let gate_data = raw[..gate_end].to_vec();
        let up_data = raw[gate_end..].to_vec();

        (
            backend.upload(&Tensor::from_quantized(vec![n_ff, n_embd], dtype, gate_data)),
            backend.upload(&Tensor::from_quantized(vec![n_ff, n_embd], dtype, up_data)),
        )
    } else {
        // F32/F16 path
        let f32_tensor = tensor.to_f32();
        let f32_data = f32_tensor.as_f32();

        let expected = 2 * n_ff * n_embd;
        assert_eq!(
            f32_data.len(), expected,
            "fused gate+up weight has {} elements, expected {} = 2 * {} * {}",
            f32_data.len(), expected, n_ff, n_embd,
        );

        let gate_data = f32_data[..n_ff * n_embd].to_vec();
        let up_data = f32_data[n_ff * n_embd..].to_vec();

        (
            backend.upload(&Tensor::new(vec![n_ff, n_embd], gate_data)),
            backend.upload(&Tensor::new(vec![n_ff, n_embd], up_data)),
        )
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

        // Token type embeddings (BERT only): [num_types, hidden_size]
        let token_type_embedding = load_tensor_optional(gguf, "token_types.weight", backend)?;

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
                let b = if config.has_bias {
                    load_tensor_optional(
                        gguf,
                        &format!("{}.attn_norm.bias", prefix),
                        backend,
                    )?
                } else {
                    None
                };
                let fw = Some(load_tensor(
                    gguf,
                    &format!("{}.ffn_norm.weight", prefix),
                    backend,
                )?);
                let fb = if config.has_bias {
                    load_tensor_optional(
                        gguf,
                        &format!("{}.ffn_norm.bias", prefix),
                        backend,
                    )?
                } else {
                    None
                };
                (w, b, fw, fb)
            } else {
                // BERT (post-norm): no pre-attention or pre-FFN norms
                (None, None, None, None)
            };

            // Attention projections — try fused QKV first (GPT-2, StarCoder, etc.)
            let (attn_q, attn_k, attn_v) =
                if gguf.find_tensor(&format!("{}.attn_qkv.weight", prefix)).is_some() {
                    let qkv = load_tensor(
                        gguf,
                        &format!("{}.attn_qkv.weight", prefix),
                        backend,
                    )?;
                    split_qkv(&qkv, config, backend)
                } else {
                    let q = load_tensor(
                        gguf,
                        &format!("{}.attn_q.weight", prefix),
                        backend,
                    )?;
                    let k = load_tensor(
                        gguf,
                        &format!("{}.attn_k.weight", prefix),
                        backend,
                    )?;
                    let v = load_tensor(
                        gguf,
                        &format!("{}.attn_v.weight", prefix),
                        backend,
                    )?;
                    (q, k, v)
                };
            let attn_output = load_tensor(
                gguf,
                &format!("{}.attn_output.weight", prefix),
                backend,
            )?;

            // Attention biases — try fused QKV bias first, then separate
            let (attn_q_bias, attn_k_bias, attn_v_bias) = if config.has_bias {
                if gguf.find_tensor(&format!("{}.attn_qkv.bias", prefix)).is_some() {
                    let qkv_bias = load_tensor(
                        gguf,
                        &format!("{}.attn_qkv.bias", prefix),
                        backend,
                    )?;
                    let (q, k, v) = split_qkv_bias(&qkv_bias, config, backend);
                    (Some(q), Some(k), Some(v))
                } else {
                    let q = load_tensor_optional(
                        gguf,
                        &format!("{}.attn_q.bias", prefix),
                        backend,
                    )?;
                    let k = load_tensor_optional(
                        gguf,
                        &format!("{}.attn_k.bias", prefix),
                        backend,
                    )?;
                    let v = load_tensor_optional(
                        gguf,
                        &format!("{}.attn_v.bias", prefix),
                        backend,
                    )?;
                    (q, k, v)
                }
            } else {
                (None, None, None)
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
            let ffn_up_raw = load_tensor(
                gguf,
                &format!("{}.ffn_up.weight", prefix),
                backend,
            )?;
            let ffn_down = load_tensor(
                gguf,
                &format!("{}.ffn_down.weight", prefix),
                backend,
            )?;

            // FFN gate (SwiGLU architectures only: Gemma, LLaMA, Phi-3, Mistral3)
            // Some models (Phi-3) fuse gate+up into a single ffn_up tensor.
            let (ffn_up, ffn_gate) = if config.has_ffn_gate {
                if gguf.find_tensor(&format!("{}.ffn_gate.weight", prefix)).is_some() {
                    // Separate gate tensor — load normally
                    let gate = load_tensor(
                        gguf,
                        &format!("{}.ffn_gate.weight", prefix),
                        backend,
                    )?;
                    (ffn_up_raw, Some(gate))
                } else if ffn_up_raw.shape()[0] == 2 * config.ffn_hidden {
                    // No separate gate — ffn_up has doubled width (fused [gate, up])
                    debug!(
                        prefix = prefix,
                        "Splitting fused gate+up from ffn_up.weight"
                    );
                    let (gate, up) = split_gate_up(&ffn_up_raw, config, backend);
                    (up, Some(gate))
                } else {
                    // Neither separate gate nor fused — missing tensor
                    return Err(InferenceError::TensorNotFound(
                        format!("{}.ffn_gate.weight", prefix),
                    ));
                }
            } else {
                (ffn_up_raw, None)
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

            // Per-head Q/K normalization (Gemma3, GemmaEmbedding, Qwen3)
            let (attn_q_norm_w, attn_k_norm_w) =
                if matches!(config.arch, ModelArch::Gemma3 | ModelArch::Gemma2 | ModelArch::GemmaEmbedding | ModelArch::Qwen3) {
                    let qn = load_tensor_optional(
                        gguf,
                        &format!("{}.attn_q_norm.weight", prefix),
                        backend,
                    )?;
                    let kn = load_tensor_optional(
                        gguf,
                        &format!("{}.attn_k_norm.weight", prefix),
                        backend,
                    )?;
                    (qn, kn)
                } else {
                    (None, None)
                };

            // Post-projection norms (Gemma3, Gemma2, GemmaEmbedding)
            let (attn_post_norm_w, ffn_post_norm_w) =
                if matches!(config.arch, ModelArch::Gemma3 | ModelArch::Gemma2 | ModelArch::GemmaEmbedding) {
                    let apn = load_tensor_optional(
                        gguf,
                        &format!("{}.post_attention_norm.weight", prefix),
                        backend,
                    )?;
                    let fpn = load_tensor_optional(
                        gguf,
                        &format!("{}.post_ffw_norm.weight", prefix),
                        backend,
                    )?;
                    (apn, fpn)
                } else {
                    (None, None)
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
                attn_q_norm_w,
                attn_k_norm_w,
                attn_post_norm_w,
                ffn_post_norm_w,
            });
        }

        // -- Output normalization --

        let output_norm_w = load_tensor_optional(gguf, "output_norm.weight", backend)?;
        let output_norm_b = if config.has_bias {
            load_tensor_optional(gguf, "output_norm.bias", backend)?
        } else {
            None
        };

        // -- Output projection (lm_head) --
        let output_projection = load_tensor_optional(gguf, "output.weight", backend)?;

        // -- LongRoPE frequency factors (Phi-3.5, etc.) --
        let rope_factors_short = load_tensor_optional(gguf, "rope_factors_short.weight", backend)?;
        let rope_factors_long = load_tensor_optional(gguf, "rope_factors_long.weight", backend)?;

        if rope_factors_short.is_some() || rope_factors_long.is_some() {
            info!(
                has_short = rope_factors_short.is_some(),
                has_long = rope_factors_long.is_some(),
                "Loaded LongRoPE frequency factors"
            );
        }

        // Validate rope_factors shapes
        if let Some(ref factors) = rope_factors_short {
            let expected = config.rope_dim / 2;
            let actual = factors.shape()[0];
            if actual != expected {
                return Err(InferenceError::Model(format!(
                    "rope_factors_short has {} elements, expected {} (rope_dim/2)",
                    actual, expected
                )));
            }
        }
        if let Some(ref factors) = rope_factors_long {
            let expected = config.rope_dim / 2;
            let actual = factors.shape()[0];
            if actual != expected {
                return Err(InferenceError::Model(format!(
                    "rope_factors_long has {} elements, expected {} (rope_dim/2)",
                    actual, expected
                )));
            }
        }

        info!(
            num_layers = layers.len(),
            has_output_projection = output_projection.is_some(),
            "Model weights loaded successfully"
        );

        Ok(ModelWeights {
            token_embedding,
            position_embedding,
            token_type_embedding,
            embedding_norm_w,
            embedding_norm_b,
            layers,
            output_norm_w,
            output_norm_b,
            output_projection,
            rope_factors_short,
            rope_factors_long,
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
            rope_neox: false,
            rope_scaling_original_ctx: 0,
            rope_scaling_attn_factor: 1.0,
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
            rope_neox: false,
            rope_scaling_original_ctx: 0,
            rope_scaling_attn_factor: 1.0,
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
        // New K-quant and non-K types should be supported
        assert_eq!(gguf_dtype_to_tensor_dtype(GgufTensorType::Q4_1).unwrap(), TensorDtype::Q4_1);
        assert_eq!(gguf_dtype_to_tensor_dtype(GgufTensorType::Q5_0).unwrap(), TensorDtype::Q5_0);
        assert_eq!(gguf_dtype_to_tensor_dtype(GgufTensorType::Q5_1).unwrap(), TensorDtype::Q5_1);
        assert_eq!(gguf_dtype_to_tensor_dtype(GgufTensorType::Q4K).unwrap(), TensorDtype::Q4_K);
        assert_eq!(gguf_dtype_to_tensor_dtype(GgufTensorType::Q5K).unwrap(), TensorDtype::Q5_K);
        assert_eq!(gguf_dtype_to_tensor_dtype(GgufTensorType::Q6K).unwrap(), TensorDtype::Q6_K);
        // Truly unsupported types should error
        assert!(gguf_dtype_to_tensor_dtype(GgufTensorType::BF16).is_err());
        assert!(gguf_dtype_to_tensor_dtype(GgufTensorType::Q2K).is_err());
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
        assert_eq!(weights.output_norm_w.as_ref().unwrap().shape(), &[32]);

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
        assert_eq!(weights.output_norm_w.as_ref().unwrap().dtype(), TensorDtype::F16);
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
        assert_eq!(dt.as_tensor().as_f32(), &[1.0, 2.0, 3.0, 4.0]);

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
        // BF16 (dtype_id=30) is not supported for inference.
        // BF16: 2 bytes per element, block_size=1
        let data = vec![0u8; 64]; // 32 BF16 elements
        let gguf_bytes = build_gguf_with_tensors(&[
            ("bf16.weight", &[32], 30, &data),
        ]);
        let path = write_temp_gguf("unsupported_dtype", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let result = load_tensor(&gguf, "bf16.weight", &backend);

        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("unsupported"), "Error: {}", err_msg);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_q4_1_tensor() {
        // Q4_1 (dtype_id=3) is now supported. 20 bytes per block of 32 elements.
        let data = vec![0u8; 20]; // 1 Q4_1 block
        let gguf_bytes = build_gguf_with_tensors(&[
            ("q4_1.weight", &[32], 3, &data),
        ]);
        let path = write_temp_gguf("q4_1_load", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let dt = load_tensor(&gguf, "q4_1.weight", &backend).unwrap();

        assert_eq!(dt.dtype(), TensorDtype::Q4_1);
        assert_eq!(dt.shape(), &[32]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_bert_output_norm_w_is_none_when_absent() {
        // Build a BERT model where output_norm.weight is NOT in the GGUF.
        // Since output_norm_w is now optional, this should succeed with None.
        let config = bert_config(1, 32);
        let h = config.hidden_size;
        let ffn = config.ffn_hidden;
        let v = config.vocab_size;
        let max_seq = config.max_seq_len;

        let mut tensor_specs: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();

        tensor_specs.push(("token_embd.weight".to_string(), vec![h as u64, v as u64], 0, f32_tensor_data(&vec![0.1f32; v * h])));
        tensor_specs.push(("position_embd.weight".to_string(), vec![h as u64, max_seq as u64], 0, f32_tensor_data(&vec![0.01f32; max_seq * h])));
        // NO output_norm.weight

        let prefix = "blk.0";
        for name in &["attn_q", "attn_k", "attn_v", "attn_output"] {
            tensor_specs.push((format!("{}.{}.weight", prefix, name), vec![h as u64, h as u64], 0, f32_tensor_data(&vec![0.01f32; h * h])));
        }
        tensor_specs.push((format!("{}.attn_output_norm.weight", prefix), vec![h as u64], 0, f32_tensor_data(&vec![1.0f32; h])));
        tensor_specs.push((format!("{}.attn_output_norm.bias", prefix), vec![h as u64], 0, f32_tensor_data(&vec![0.0f32; h])));
        tensor_specs.push((format!("{}.ffn_up.weight", prefix), vec![h as u64, ffn as u64], 0, f32_tensor_data(&vec![0.01f32; h * ffn])));
        tensor_specs.push((format!("{}.ffn_down.weight", prefix), vec![ffn as u64, h as u64], 0, f32_tensor_data(&vec![0.01f32; h * ffn])));
        tensor_specs.push((format!("{}.layer_output_norm.weight", prefix), vec![h as u64], 0, f32_tensor_data(&vec![1.0f32; h])));
        tensor_specs.push((format!("{}.layer_output_norm.bias", prefix), vec![h as u64], 0, f32_tensor_data(&vec![0.0f32; h])));

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensor_specs
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("bert_no_output_norm", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        // output_norm_w should be None for BERT when the tensor is absent
        assert!(weights.output_norm_w.is_none());
        assert!(weights.output_norm_b.is_none());

        std::fs::remove_file(&path).ok();
    }

    // ====================================================================
    // GPT-2 / fused QKV tests
    // ====================================================================

    /// Build a minimal GPT-2-style ModelConfig for testing.
    fn gpt2_config(num_layers: usize, hidden_size: usize) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::GPT2,
            arch_name: "gpt2".to_string(),
            hidden_size,
            num_layers,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: hidden_size / 4,
            ffn_hidden: hidden_size * 4,
            vocab_size: 64,
            max_seq_len: 128,
            norm_type: NormType::LayerNorm,
            norm_eps: 1e-5,
            activation: Activation::GELU,
            position_type: PositionType::Learned,
            rope_freq_base: 0.0,
            rope_dim: 0,
            rope_neox: false,
            rope_scaling_original_ctx: 0,
            rope_scaling_attn_factor: 1.0,
            causal: true,
            attn_logit_softcap: 0.0,
            attn_scale: None,
            embedding_scale: 1.0,
            pooling_type: PoolingType::None,
            has_ffn_gate: false,
            has_bias: true,
            pre_norm: true,
        }
    }

    /// Build all GPT-2-style tensors with fused QKV.
    ///
    /// Uses known data patterns so split results can be verified:
    /// - Q rows filled with 0.1
    /// - K rows filled with 0.2
    /// - V rows filled with 0.3
    fn build_gpt2_tensors(config: &ModelConfig) -> Vec<(String, Vec<u64>, u32, Vec<u8>)> {
        let h = config.hidden_size;
        let ffn = config.ffn_hidden;
        let v = config.vocab_size;
        let max_seq = config.max_seq_len;

        let mut tensors = Vec::new();

        // Token embedding: GGUF dims [hidden_size, vocab_size]
        tensors.push((
            "token_embd.weight".to_string(),
            vec![h as u64, v as u64],
            0,
            f32_tensor_data(&vec![0.1f32; v * h]),
        ));

        // Position embedding: GGUF dims [hidden_size, max_seq_len]
        tensors.push((
            "position_embd.weight".to_string(),
            vec![h as u64, max_seq as u64],
            0,
            f32_tensor_data(&vec![0.01f32; max_seq * h]),
        ));

        // Output norm + bias
        tensors.push((
            "output_norm.weight".to_string(),
            vec![h as u64],
            0,
            f32_tensor_data(&vec![1.0f32; h]),
        ));
        tensors.push((
            "output_norm.bias".to_string(),
            vec![h as u64],
            0,
            f32_tensor_data(&vec![0.0f32; h]),
        ));

        for i in 0..config.num_layers {
            let prefix = format!("blk.{}", i);

            // Pre-norm weights + biases (GPT-2 uses LayerNorm before sublayers)
            tensors.push((format!("{}.attn_norm.weight", prefix), vec![h as u64], 0, f32_tensor_data(&vec![1.0f32; h])));
            tensors.push((format!("{}.attn_norm.bias", prefix), vec![h as u64], 0, f32_tensor_data(&vec![0.0f32; h])));
            tensors.push((format!("{}.ffn_norm.weight", prefix), vec![h as u64], 0, f32_tensor_data(&vec![1.0f32; h])));
            tensors.push((format!("{}.ffn_norm.bias", prefix), vec![h as u64], 0, f32_tensor_data(&vec![0.0f32; h])));

            // Fused QKV weight: GGUF dims [hidden_size, 3*hidden_size]
            // (reversed to [3*h, h] in our row-major layout)
            // Fill Q region with 0.1, K region with 0.2, V region with 0.3
            let mut qkv_data = Vec::with_capacity(3 * h * h);
            qkv_data.extend(std::iter::repeat(0.1f32).take(h * h)); // Q
            qkv_data.extend(std::iter::repeat(0.2f32).take(h * h)); // K
            qkv_data.extend(std::iter::repeat(0.3f32).take(h * h)); // V
            tensors.push((
                format!("{}.attn_qkv.weight", prefix),
                vec![h as u64, (3 * h) as u64],
                0,
                f32_tensor_data(&qkv_data),
            ));

            // Fused QKV bias: [3*h]
            let mut qkv_bias = Vec::with_capacity(3 * h);
            qkv_bias.extend(std::iter::repeat(1.0f32).take(h)); // Q bias
            qkv_bias.extend(std::iter::repeat(2.0f32).take(h)); // K bias
            qkv_bias.extend(std::iter::repeat(3.0f32).take(h)); // V bias
            tensors.push((
                format!("{}.attn_qkv.bias", prefix),
                vec![(3 * h) as u64],
                0,
                f32_tensor_data(&qkv_bias),
            ));

            // attn_output + bias
            tensors.push((format!("{}.attn_output.weight", prefix), vec![h as u64, h as u64], 0, f32_tensor_data(&vec![0.01f32; h * h])));
            tensors.push((format!("{}.attn_output.bias", prefix), vec![h as u64], 0, f32_tensor_data(&vec![0.0f32; h])));

            // FFN up/down + biases (no gate for GPT-2)
            tensors.push((format!("{}.ffn_up.weight", prefix), vec![h as u64, ffn as u64], 0, f32_tensor_data(&vec![0.01f32; h * ffn])));
            tensors.push((format!("{}.ffn_up.bias", prefix), vec![ffn as u64], 0, f32_tensor_data(&vec![0.0f32; ffn])));
            tensors.push((format!("{}.ffn_down.weight", prefix), vec![ffn as u64, h as u64], 0, f32_tensor_data(&vec![0.01f32; h * ffn])));
            tensors.push((format!("{}.ffn_down.bias", prefix), vec![h as u64], 0, f32_tensor_data(&vec![0.0f32; h])));
        }

        tensors
    }

    #[test]
    fn test_load_gpt2_style_weights_with_fused_qkv() {
        // Full GPT-2 weight loading: fused QKV, pre-norm with bias, learned positions
        let config = gpt2_config(1, 32);
        let tensor_specs = build_gpt2_tensors(&config);

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensor_specs
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("gpt2_1layer", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        // GPT-2 structural checks
        assert_eq!(weights.layers.len(), 1);
        assert!(weights.position_embedding.is_some()); // Learned positions
        assert!(weights.embedding_norm_w.is_none()); // GPT-2 has no embedding norm
        assert!(weights.embedding_norm_b.is_none());
        assert!(weights.output_norm_w.is_some());
        assert!(weights.output_norm_b.is_some()); // LayerNorm bias
        assert!(weights.output_projection.is_none()); // Tied embeddings

        let layer = &weights.layers[0];

        // Pre-norm weights AND biases should be present (LayerNorm, not RMSNorm)
        assert!(layer.attn_norm_w.is_some());
        assert!(layer.attn_norm_b.is_some(), "GPT-2 pre-norm must have attn_norm bias");
        assert!(layer.ffn_norm_w.is_some());
        assert!(layer.ffn_norm_b.is_some(), "GPT-2 pre-norm must have ffn_norm bias");

        // Fused QKV should have been split into separate Q, K, V
        assert_eq!(layer.attn_q.shape(), &[32, 32]);
        assert_eq!(layer.attn_k.shape(), &[32, 32]);
        assert_eq!(layer.attn_v.shape(), &[32, 32]);
        assert_eq!(layer.attn_output.shape(), &[32, 32]);

        // Attention biases from fused QKV bias
        assert!(layer.attn_q_bias.is_some());
        assert!(layer.attn_k_bias.is_some());
        assert!(layer.attn_v_bias.is_some());
        assert!(layer.attn_output_bias.is_some());

        // No FFN gate (GPT-2 uses GELU, not SwiGLU)
        assert!(layer.ffn_gate.is_none());

        // FFN biases should be present
        assert!(layer.ffn_up_bias.is_some());
        assert!(layer.ffn_down_bias.is_some());

        // Post-norm weights should be absent (GPT-2 uses pre-norm)
        assert!(layer.attn_output_norm_w.is_none());
        assert!(layer.attn_output_norm_b.is_none());
        assert!(layer.ffn_output_norm_w.is_none());
        assert!(layer.ffn_output_norm_b.is_none());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_fused_qkv_split_produces_correct_data() {
        // Verify that the Q/K/V data values are correct after splitting.
        // Q region filled with 0.1, K with 0.2, V with 0.3.
        let config = gpt2_config(1, 32);
        let tensor_specs = build_gpt2_tensors(&config);

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensor_specs
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("gpt2_qkv_data", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        let layer = &weights.layers[0];
        let h = config.hidden_size;

        // Download and verify Q data (should be all 0.1)
        let q_tensor = backend.download(&layer.attn_q);
        let q_data = q_tensor.as_f32();
        assert_eq!(q_data.len(), h * h);
        for (i, &val) in q_data.iter().enumerate() {
            assert!(
                (val - 0.1).abs() < 1e-6,
                "Q[{}] = {}, expected 0.1", i, val,
            );
        }

        // Download and verify K data (should be all 0.2)
        let k_tensor = backend.download(&layer.attn_k);
        let k_data = k_tensor.as_f32();
        assert_eq!(k_data.len(), h * h);
        for (i, &val) in k_data.iter().enumerate() {
            assert!(
                (val - 0.2).abs() < 1e-6,
                "K[{}] = {}, expected 0.2", i, val,
            );
        }

        // Download and verify V data (should be all 0.3)
        let v_tensor = backend.download(&layer.attn_v);
        let v_data = v_tensor.as_f32();
        assert_eq!(v_data.len(), h * h);
        for (i, &val) in v_data.iter().enumerate() {
            assert!(
                (val - 0.3).abs() < 1e-6,
                "V[{}] = {}, expected 0.3", i, val,
            );
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_fused_qkv_bias_split_produces_correct_data() {
        // Verify bias values: Q bias = 1.0, K bias = 2.0, V bias = 3.0
        let config = gpt2_config(1, 32);
        let tensor_specs = build_gpt2_tensors(&config);

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensor_specs
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("gpt2_qkv_bias_data", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        let layer = &weights.layers[0];
        let h = config.hidden_size;

        // Q bias = 1.0
        let q_bias = backend.download(layer.attn_q_bias.as_ref().unwrap());
        let q_bias_data = q_bias.as_f32();
        assert_eq!(q_bias_data.len(), h);
        assert!((q_bias_data[0] - 1.0).abs() < 1e-6, "Q bias[0] = {}", q_bias_data[0]);
        assert!((q_bias_data[h - 1] - 1.0).abs() < 1e-6);

        // K bias = 2.0
        let k_bias = backend.download(layer.attn_k_bias.as_ref().unwrap());
        let k_bias_data = k_bias.as_f32();
        assert_eq!(k_bias_data.len(), h);
        assert!((k_bias_data[0] - 2.0).abs() < 1e-6, "K bias[0] = {}", k_bias_data[0]);

        // V bias = 3.0
        let v_bias = backend.download(layer.attn_v_bias.as_ref().unwrap());
        let v_bias_data = v_bias.as_f32();
        assert_eq!(v_bias_data.len(), h);
        assert!((v_bias_data[0] - 3.0).abs() < 1e-6, "V bias[0] = {}", v_bias_data[0]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_fused_qkv_q8_0_dequantizes_and_splits() {
        // GPT-2 with Q8_0 fused QKV: verify dequant + split produces correct shapes
        // and the result is F32 (since we dequantize before splitting).
        let config = gpt2_config(1, 32);
        let h = config.hidden_size;
        let ffn = config.ffn_hidden;
        let v = config.vocab_size;
        let max_seq = config.max_seq_len;

        let mut tensor_specs: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();

        // Token + position embeddings (F32)
        tensor_specs.push(("token_embd.weight".to_string(), vec![h as u64, v as u64], 0, f32_tensor_data(&vec![0.1f32; v * h])));
        tensor_specs.push(("position_embd.weight".to_string(), vec![h as u64, max_seq as u64], 0, f32_tensor_data(&vec![0.01f32; max_seq * h])));
        tensor_specs.push(("output_norm.weight".to_string(), vec![h as u64], 0, f32_tensor_data(&vec![1.0f32; h])));
        tensor_specs.push(("output_norm.bias".to_string(), vec![h as u64], 0, f32_tensor_data(&vec![0.0f32; h])));

        // Layer 0 norms
        tensor_specs.push(("blk.0.attn_norm.weight".to_string(), vec![h as u64], 0, f32_tensor_data(&vec![1.0f32; h])));
        tensor_specs.push(("blk.0.attn_norm.bias".to_string(), vec![h as u64], 0, f32_tensor_data(&vec![0.0f32; h])));
        tensor_specs.push(("blk.0.ffn_norm.weight".to_string(), vec![h as u64], 0, f32_tensor_data(&vec![1.0f32; h])));
        tensor_specs.push(("blk.0.ffn_norm.bias".to_string(), vec![h as u64], 0, f32_tensor_data(&vec![0.0f32; h])));

        // Fused QKV as Q8_0 (dtype_id = 8)
        let qkv_elements = 3 * h * h;
        let q8_data = q8_0_tensor_data(qkv_elements);
        tensor_specs.push((
            "blk.0.attn_qkv.weight".to_string(),
            vec![h as u64, (3 * h) as u64],
            8, // Q8_0
            q8_data,
        ));

        // Fused QKV bias (F32)
        let qkv_bias_data: Vec<f32> = (0..3 * h).map(|i| i as f32 * 0.01).collect();
        tensor_specs.push((
            "blk.0.attn_qkv.bias".to_string(),
            vec![(3 * h) as u64],
            0,
            f32_tensor_data(&qkv_bias_data),
        ));

        // attn_output + bias
        tensor_specs.push(("blk.0.attn_output.weight".to_string(), vec![h as u64, h as u64], 0, f32_tensor_data(&vec![0.01f32; h * h])));
        tensor_specs.push(("blk.0.attn_output.bias".to_string(), vec![h as u64], 0, f32_tensor_data(&vec![0.0f32; h])));

        // FFN
        tensor_specs.push(("blk.0.ffn_up.weight".to_string(), vec![h as u64, ffn as u64], 0, f32_tensor_data(&vec![0.01f32; h * ffn])));
        tensor_specs.push(("blk.0.ffn_up.bias".to_string(), vec![ffn as u64], 0, f32_tensor_data(&vec![0.0f32; ffn])));
        tensor_specs.push(("blk.0.ffn_down.weight".to_string(), vec![ffn as u64, h as u64], 0, f32_tensor_data(&vec![0.01f32; h * ffn])));
        tensor_specs.push(("blk.0.ffn_down.bias".to_string(), vec![h as u64], 0, f32_tensor_data(&vec![0.0f32; h])));

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensor_specs
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("gpt2_q8_qkv", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        let layer = &weights.layers[0];

        // After splitting Q8_0 fused QKV, results stay Q8_0 (quantized byte-level split)
        assert_eq!(layer.attn_q.dtype(), TensorDtype::Q8_0);
        assert_eq!(layer.attn_k.dtype(), TensorDtype::Q8_0);
        assert_eq!(layer.attn_v.dtype(), TensorDtype::Q8_0);

        // Shapes should be correct
        assert_eq!(layer.attn_q.shape(), &[h, h]);
        assert_eq!(layer.attn_k.shape(), &[h, h]);
        assert_eq!(layer.attn_v.shape(), &[h, h]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gpt2_multiple_layers() {
        let config = gpt2_config(3, 32);
        let tensor_specs = build_gpt2_tensors(&config);

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensor_specs
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("gpt2_3layers", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        assert_eq!(weights.layers.len(), 3);
        for (i, layer) in weights.layers.iter().enumerate() {
            // Every layer should have pre-norm biases (LayerNorm)
            assert!(layer.attn_norm_b.is_some(), "layer {} missing attn_norm_b", i);
            assert!(layer.ffn_norm_b.is_some(), "layer {} missing ffn_norm_b", i);

            // Every layer should have split QKV
            assert_eq!(layer.attn_q.shape(), &[32, 32], "layer {} Q shape wrong", i);
            assert_eq!(layer.attn_k.shape(), &[32, 32], "layer {} K shape wrong", i);
            assert_eq!(layer.attn_v.shape(), &[32, 32], "layer {} V shape wrong", i);

            // Biases from fused QKV
            assert!(layer.attn_q_bias.is_some(), "layer {} missing Q bias", i);
            assert!(layer.attn_k_bias.is_some(), "layer {} missing K bias", i);
            assert!(layer.attn_v_bias.is_some(), "layer {} missing V bias", i);

            // No gate
            assert!(layer.ffn_gate.is_none(), "layer {} has unexpected ffn_gate", i);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_separate_qkv_still_works_when_fused_absent() {
        // Verify that models without fused QKV (LLaMA/Gemma) still load correctly
        // after the fused-QKV fallback logic was added.
        let config = gemma_config(1, 32);
        let tensor_specs = build_gemma_tensors(&config);

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensor_specs
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("gemma_after_qkv_change", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        let layer = &weights.layers[0];
        assert_eq!(layer.attn_q.shape(), &[32, 32]);
        assert_eq!(layer.attn_k.shape(), &[32, 32]);
        assert_eq!(layer.attn_v.shape(), &[32, 32]);
        // Gemma: no bias
        assert!(layer.attn_q_bias.is_none());
        assert!(layer.attn_norm_b.is_none());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_pre_norm_with_bias_vs_without() {
        // Verify that LLaMA (pre_norm=true, has_bias=false) gets None for norm bias
        // while GPT-2 (pre_norm=true, has_bias=true) gets Some for norm bias.
        let config_llama = gemma_config(1, 32); // has_bias=false, pre_norm=true
        let tensors_llama = build_gemma_tensors(&config_llama);
        let refs_llama: Vec<(&str, &[u64], u32, &[u8])> = tensors_llama
            .iter()
            .map(|(n, d, t, data)| (n.as_str(), d.as_slice(), *t, data.as_slice()))
            .collect();
        let bytes_llama = build_gguf_with_tensors(&refs_llama);
        let path_llama = write_temp_gguf("prenorm_no_bias", &bytes_llama);

        let gguf_llama = GgufFile::open(&path_llama).unwrap();
        let backend = CpuBackend::new();
        let weights_llama = ModelWeights::from_gguf(&gguf_llama, &config_llama, &backend).unwrap();

        let config_gpt2 = gpt2_config(1, 32); // has_bias=true, pre_norm=true
        let tensors_gpt2 = build_gpt2_tensors(&config_gpt2);
        let refs_gpt2: Vec<(&str, &[u64], u32, &[u8])> = tensors_gpt2
            .iter()
            .map(|(n, d, t, data)| (n.as_str(), d.as_slice(), *t, data.as_slice()))
            .collect();
        let bytes_gpt2 = build_gguf_with_tensors(&refs_gpt2);
        let path_gpt2 = write_temp_gguf("prenorm_with_bias", &bytes_gpt2);

        let gguf_gpt2 = GgufFile::open(&path_gpt2).unwrap();
        let weights_gpt2 = ModelWeights::from_gguf(&gguf_gpt2, &config_gpt2, &backend).unwrap();

        // LLaMA: pre-norm weights present, biases absent
        assert!(weights_llama.layers[0].attn_norm_w.is_some());
        assert!(weights_llama.layers[0].attn_norm_b.is_none());
        assert!(weights_llama.layers[0].ffn_norm_w.is_some());
        assert!(weights_llama.layers[0].ffn_norm_b.is_none());

        // GPT-2: pre-norm weights AND biases both present
        assert!(weights_gpt2.layers[0].attn_norm_w.is_some());
        assert!(weights_gpt2.layers[0].attn_norm_b.is_some());
        assert!(weights_gpt2.layers[0].ffn_norm_w.is_some());
        assert!(weights_gpt2.layers[0].ffn_norm_b.is_some());

        std::fs::remove_file(&path_llama).ok();
        std::fs::remove_file(&path_gpt2).ok();
    }

    // -----------------------------------------------------------------------
    // Fused gate+up FFN tests (Phi-3 style)
    // -----------------------------------------------------------------------

    /// Build a Phi3-style config: RMSNorm, SwiGLU, RoPE, fused QKV, has_ffn_gate=true, has_bias=false.
    fn phi3_config(num_layers: usize, hidden_size: usize) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Phi3,
            arch_name: "phi3".to_string(),
            hidden_size,
            num_layers,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: hidden_size / 4,
            ffn_hidden: hidden_size * 2, // e.g., h=32 → ffn=64
            vocab_size: 64,
            max_seq_len: 128,
            norm_type: NormType::RMSNorm,
            norm_eps: 1e-5,
            activation: Activation::SwiGLU,
            position_type: PositionType::RoPE,
            rope_freq_base: 10000.0,
            rope_dim: hidden_size / 4,
            rope_neox: true,
            rope_scaling_original_ctx: 0,
            rope_scaling_attn_factor: 1.0,
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

    /// Build Phi3-style tensors with fused QKV and fused gate+up FFN.
    fn build_phi3_tensors(config: &ModelConfig) -> Vec<(String, Vec<u64>, u32, Vec<u8>)> {
        let h = config.hidden_size;
        let ffn = config.ffn_hidden;
        let v = config.vocab_size;
        let n_kv = config.num_kv_heads * config.head_dim;

        let mut tensors = Vec::new();

        // Token embedding
        tensors.push((
            "token_embd.weight".to_string(),
            vec![h as u64, v as u64],
            0,
            f32_tensor_data(&vec![0.1f32; v * h]),
        ));

        // Output norm
        tensors.push((
            "output_norm.weight".to_string(),
            vec![h as u64],
            0,
            f32_tensor_data(&vec![1.0f32; h]),
        ));

        for i in 0..config.num_layers {
            let prefix = format!("blk.{}", i);

            // Pre-norm weights
            tensors.push((
                format!("{}.attn_norm.weight", prefix),
                vec![h as u64],
                0,
                f32_tensor_data(&vec![1.0f32; h]),
            ));
            tensors.push((
                format!("{}.ffn_norm.weight", prefix),
                vec![h as u64],
                0,
                f32_tensor_data(&vec![1.0f32; h]),
            ));

            // Fused QKV: [h, h + 2*n_kv] in GGUF dims
            let qkv_cols = h + 2 * n_kv;
            tensors.push((
                format!("{}.attn_qkv.weight", prefix),
                vec![h as u64, qkv_cols as u64],
                0,
                f32_tensor_data(&vec![0.01f32; qkv_cols * h]),
            ));

            // Attention output
            tensors.push((
                format!("{}.attn_output.weight", prefix),
                vec![h as u64, h as u64],
                0,
                f32_tensor_data(&vec![0.01f32; h * h]),
            ));

            // Fused gate+up: [h, 2*ffn] in GGUF dims — NO separate ffn_gate
            tensors.push((
                format!("{}.ffn_up.weight", prefix),
                vec![h as u64, (2 * ffn) as u64],
                0,
                {
                    // Fill gate half with 1.0 and up half with 2.0 for verification
                    let mut data = vec![1.0f32; ffn * h];
                    data.extend(vec![2.0f32; ffn * h]);
                    f32_tensor_data(&data)
                },
            ));

            // FFN down
            tensors.push((
                format!("{}.ffn_down.weight", prefix),
                vec![ffn as u64, h as u64],
                0,
                f32_tensor_data(&vec![0.01f32; h * ffn]),
            ));
        }

        tensors
    }

    #[test]
    fn test_fused_gate_up_split_produces_correct_shapes() {
        // Phi3-style: ffn_up.weight has shape [2*ffn, h] — no separate ffn_gate
        let config = phi3_config(1, 32);
        let tensor_specs = build_phi3_tensors(&config);

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensor_specs
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("phi3_fused_gate_up", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        let h = config.hidden_size;   // 32
        let ffn = config.ffn_hidden;  // 64

        let layer = &weights.layers[0];

        // ffn_gate should exist (split from fused tensor)
        assert!(layer.ffn_gate.is_some(), "ffn_gate should exist after fused split");

        // Shapes: gate=[ffn, h], up=[ffn, h], down=[h, ffn] (GGUF dims reversed)
        assert_eq!(layer.ffn_gate.as_ref().unwrap().shape(), &[ffn, h]);
        assert_eq!(layer.ffn_up.shape(), &[ffn, h]);
        assert_eq!(layer.ffn_down.shape(), &[h, ffn]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_fused_gate_up_split_produces_correct_data() {
        // Verify the split puts gate data in ffn_gate and up data in ffn_up
        let config = phi3_config(1, 32);
        let tensor_specs = build_phi3_tensors(&config);

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensor_specs
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("phi3_fused_data", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        let layer = &weights.layers[0];

        // Gate was filled with 1.0, up was filled with 2.0 in build_phi3_tensors
        let gate_data = layer.ffn_gate.as_ref().unwrap().as_tensor().as_f32();
        let up_data = layer.ffn_up.as_tensor().as_f32();

        assert!(gate_data.iter().all(|&v| (v - 1.0).abs() < 1e-6),
            "gate data should be all 1.0");
        assert!(up_data.iter().all(|&v| (v - 2.0).abs() < 1e-6),
            "up data should be all 2.0");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_separate_gate_still_works_when_fused_fallback_exists() {
        // Verify that models with separate ffn_gate (Gemma/LLaMA) still load
        // correctly after the fused gate+up fallback was added.
        let config = gemma_config(1, 32);
        let tensor_specs = build_gemma_tensors(&config);

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensor_specs
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("gemma_after_fused_change", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        let h = config.hidden_size;   // 32
        let ffn = config.ffn_hidden;  // 128

        let layer = &weights.layers[0];
        assert!(layer.ffn_gate.is_some());
        assert_eq!(layer.ffn_gate.as_ref().unwrap().shape(), &[ffn, h]);
        assert_eq!(layer.ffn_up.shape(), &[ffn, h]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_fused_gate_up_q8_0_split() {
        // Verify that fused gate+up splitting works correctly for quantized Q8_0 tensors.
        // Each block of 32 values should be independently preserved after splitting.
        let config = phi3_config(1, 32);
        let h = config.hidden_size; // 32
        let ffn = config.ffn_hidden; // 64

        // Build Q8_0 data for the fused gate+up tensor [2*ffn, h] = [128, 32]
        // Gate half: scale=1.0, quants=[0,1,...,31]
        // Up half: scale=2.0, quants=[0,1,...,31]
        let gate_rows = ffn; // 64 rows for gate
        let up_rows = ffn;   // 64 rows for up
        let gate_data = q8_0_tensor_data(gate_rows * h); // 64*32 = 2048 elements
        let up_data = {
            // Custom Q8_0 data with scale=2.0 to distinguish from gate
            let n_elements = up_rows * h;
            let n_blocks = n_elements / 32;
            let mut data = Vec::with_capacity(n_blocks * 34);
            let scale_bits = f32_to_f16(2.0);
            for _ in 0..n_blocks {
                data.extend_from_slice(&scale_bits.to_le_bytes());
                for i in 0..32u8 {
                    data.push(i);
                }
            }
            data
        };

        // Fused tensor data: gate rows followed by up rows
        let mut fused_data = gate_data.clone();
        fused_data.extend_from_slice(&up_data);

        let mut tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();

        // Token embedding
        tensors.push((
            "token_embd.weight".to_string(),
            vec![h as u64, 64],
            0, // F32
            f32_tensor_data(&vec![0.1f32; 64 * h]),
        ));

        // Output norm
        tensors.push((
            "output_norm.weight".to_string(),
            vec![h as u64],
            0,
            f32_tensor_data(&vec![1.0f32; h]),
        ));

        // Layer tensors
        let prefix = "blk.0";
        tensors.push((format!("{}.attn_norm.weight", prefix), vec![h as u64], 0, f32_tensor_data(&vec![1.0; h])));
        tensors.push((format!("{}.ffn_norm.weight", prefix), vec![h as u64], 0, f32_tensor_data(&vec![1.0; h])));

        // Fused QKV
        let n_kv = config.num_kv_heads * config.head_dim;
        let qkv_cols = h + 2 * n_kv;
        tensors.push((
            format!("{}.attn_qkv.weight", prefix),
            vec![h as u64, qkv_cols as u64],
            0,
            f32_tensor_data(&vec![0.01; qkv_cols * h]),
        ));

        // Attention output
        tensors.push((format!("{}.attn_output.weight", prefix), vec![h as u64, h as u64], 0, f32_tensor_data(&vec![0.01; h * h])));

        // Fused gate+up as Q8_0: GGUF dtype=8 (Q8_0), dims [h, 2*ffn]
        tensors.push((
            format!("{}.ffn_up.weight", prefix),
            vec![h as u64, (2 * ffn) as u64],
            8, // Q8_0
            fused_data,
        ));

        // FFN down
        tensors.push((format!("{}.ffn_down.weight", prefix), vec![ffn as u64, h as u64], 0, f32_tensor_data(&vec![0.01; h * ffn])));

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensors
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("phi3_q8_0_fused", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        let layer = &weights.layers[0];

        // Verify shapes
        assert!(layer.ffn_gate.is_some(), "ffn_gate should exist after fused split");
        assert_eq!(layer.ffn_gate.as_ref().unwrap().shape(), &[ffn, h]);
        assert_eq!(layer.ffn_up.shape(), &[ffn, h]);

        // Verify dtypes are preserved as Q8_0
        assert_eq!(layer.ffn_gate.as_ref().unwrap().dtype(), TensorDtype::Q8_0);
        assert_eq!(layer.ffn_up.dtype(), TensorDtype::Q8_0);

        // Verify data is correct by dequantizing
        let gate_deq = layer.ffn_gate.as_ref().unwrap().as_tensor().to_f32();
        let gate_f32 = gate_deq.as_f32();
        let up_deq = layer.ffn_up.as_tensor().to_f32();
        let up_f32 = up_deq.as_f32();

        // Gate: scale=1.0, values [0,1,...,31] repeated → first 32 values are 0.0, 1.0, ..., 31.0
        assert!((gate_f32[0] - 0.0).abs() < 1e-3, "gate[0] should be 0.0, got {}", gate_f32[0]);
        assert!((gate_f32[1] - 1.0).abs() < 1e-3, "gate[1] should be 1.0, got {}", gate_f32[1]);
        assert!((gate_f32[31] - 31.0).abs() < 1e-3, "gate[31] should be 31.0, got {}", gate_f32[31]);

        // Up: scale=2.0, values [0,1,...,31] repeated → first 32 values are 0.0, 2.0, ..., 62.0
        assert!((up_f32[0] - 0.0).abs() < 1e-3, "up[0] should be 0.0, got {}", up_f32[0]);
        assert!((up_f32[1] - 2.0).abs() < 1e-3, "up[1] should be 2.0, got {}", up_f32[1]);
        assert!((up_f32[31] - 62.0).abs() < 1e-3, "up[31] should be 62.0, got {}", up_f32[31]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_fused_qkv_split_with_gqa() {
        // GQA: num_kv_heads < num_heads. Q is [n_embd, n_embd], K/V are [n_kv, n_embd].
        // This is the Phi-3.5-mini case (32 Q heads, 8 KV heads).
        let mut config = phi3_config(1, 32);
        config.num_heads = 8;
        config.num_kv_heads = 2; // GQA: 4 Q heads per KV group
        config.head_dim = 4; // hidden_size / num_heads = 32 / 8 = 4
        config.rope_dim = 4;

        let h = config.hidden_size; // 32
        let n_kv = config.num_kv_heads * config.head_dim; // 2 * 4 = 8
        let ffn = config.ffn_hidden;
        let v = config.vocab_size;

        let mut tensors = Vec::new();

        // Token embedding
        tensors.push(("token_embd.weight".to_string(), vec![h as u64, v as u64], 0, f32_tensor_data(&vec![0.1f32; v * h])));
        // Output norm
        tensors.push(("output_norm.weight".to_string(), vec![h as u64], 0, f32_tensor_data(&vec![1.0f32; h])));
        // Layer norms
        tensors.push(("blk.0.attn_norm.weight".to_string(), vec![h as u64], 0, f32_tensor_data(&vec![1.0f32; h])));
        tensors.push(("blk.0.ffn_norm.weight".to_string(), vec![h as u64], 0, f32_tensor_data(&vec![1.0f32; h])));

        // Fused QKV: [h, h + 2*n_kv] in GGUF dims → [h + 2*n_kv, h] rows
        // Fill Q region with 0.1, K with 0.2, V with 0.3
        let qkv_cols = h + 2 * n_kv; // 32 + 2*8 = 48
        let mut qkv_data = Vec::with_capacity(qkv_cols * h);
        qkv_data.extend(std::iter::repeat(0.1f32).take(h * h));      // Q: h rows
        qkv_data.extend(std::iter::repeat(0.2f32).take(n_kv * h));   // K: n_kv rows
        qkv_data.extend(std::iter::repeat(0.3f32).take(n_kv * h));   // V: n_kv rows
        tensors.push((
            "blk.0.attn_qkv.weight".to_string(),
            vec![h as u64, qkv_cols as u64],
            0,
            f32_tensor_data(&qkv_data),
        ));

        // Attention output
        tensors.push(("blk.0.attn_output.weight".to_string(), vec![h as u64, h as u64], 0, f32_tensor_data(&vec![0.01f32; h * h])));
        // Fused gate+up
        tensors.push(("blk.0.ffn_up.weight".to_string(), vec![h as u64, (2 * ffn) as u64], 0, f32_tensor_data(&vec![0.01f32; 2 * ffn * h])));
        // FFN down
        tensors.push(("blk.0.ffn_down.weight".to_string(), vec![ffn as u64, h as u64], 0, f32_tensor_data(&vec![0.01f32; h * ffn])));

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensors
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("phi3_gqa_fused_qkv", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        let layer = &weights.layers[0];

        // Q should be [h, h] = [32, 32], K and V should be [n_kv, h] = [8, 32]
        assert_eq!(layer.attn_q.shape(), &[h, h], "Q shape with GQA");
        assert_eq!(layer.attn_k.shape(), &[n_kv, h], "K shape with GQA");
        assert_eq!(layer.attn_v.shape(), &[n_kv, h], "V shape with GQA");

        // Verify data correctness
        let q_data = backend.download(&layer.attn_q);
        let k_data = backend.download(&layer.attn_k);
        let v_data = backend.download(&layer.attn_v);

        for (i, &val) in q_data.as_f32().iter().enumerate() {
            assert!((val - 0.1).abs() < 1e-6, "Q[{}] = {}, expected 0.1", i, val);
        }
        for (i, &val) in k_data.as_f32().iter().enumerate() {
            assert!((val - 0.2).abs() < 1e-6, "K[{}] = {}, expected 0.2", i, val);
        }
        for (i, &val) in v_data.as_f32().iter().enumerate() {
            assert!((val - 0.3).abs() < 1e-6, "V[{}] = {}, expected 0.3", i, val);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_fused_qkv_q8_0_split_with_gqa() {
        // GQA with quantized (Q8_0) fused QKV — verify byte-level split preserves
        // quantization format and produces correct asymmetric shapes.
        let mut config = phi3_config(1, 32);
        config.num_heads = 8;
        config.num_kv_heads = 2;
        config.head_dim = 4;
        config.rope_dim = 4;

        let h = config.hidden_size; // 32
        let n_kv = config.num_kv_heads * config.head_dim; // 8
        let ffn = config.ffn_hidden;
        let v = config.vocab_size;
        let qkv_cols = h + 2 * n_kv; // 48

        let mut tensors = Vec::new();

        // Token embedding, norms (F32)
        tensors.push(("token_embd.weight".to_string(), vec![h as u64, v as u64], 0, f32_tensor_data(&vec![0.1f32; v * h])));
        tensors.push(("output_norm.weight".to_string(), vec![h as u64], 0, f32_tensor_data(&vec![1.0f32; h])));
        tensors.push(("blk.0.attn_norm.weight".to_string(), vec![h as u64], 0, f32_tensor_data(&vec![1.0f32; h])));
        tensors.push(("blk.0.ffn_norm.weight".to_string(), vec![h as u64], 0, f32_tensor_data(&vec![1.0f32; h])));

        // Fused QKV as Q8_0: rows = qkv_cols = 48, cols = h = 32
        // Q8_0: block_size=32, so h=32 means 1 block per row, 34 bytes per row
        let q8_qkv = q8_0_tensor_data(qkv_cols * h);
        tensors.push((
            "blk.0.attn_qkv.weight".to_string(),
            vec![h as u64, qkv_cols as u64],
            8, // Q8_0
            q8_qkv,
        ));

        // Attention output (F32)
        tensors.push(("blk.0.attn_output.weight".to_string(), vec![h as u64, h as u64], 0, f32_tensor_data(&vec![0.01f32; h * h])));
        // Fused gate+up (F32)
        tensors.push(("blk.0.ffn_up.weight".to_string(), vec![h as u64, (2 * ffn) as u64], 0, f32_tensor_data(&vec![0.01f32; 2 * ffn * h])));
        // FFN down (F32)
        tensors.push(("blk.0.ffn_down.weight".to_string(), vec![ffn as u64, h as u64], 0, f32_tensor_data(&vec![0.01f32; h * ffn])));

        let tensor_refs: Vec<(&str, &[u64], u32, &[u8])> = tensors
            .iter()
            .map(|(name, dims, dtype, data)| (name.as_str(), dims.as_slice(), *dtype, data.as_slice()))
            .collect();

        let gguf_bytes = build_gguf_with_tensors(&tensor_refs);
        let path = write_temp_gguf("phi3_gqa_q8_qkv", &gguf_bytes);

        let gguf = GgufFile::open(&path).unwrap();
        let backend = CpuBackend::new();
        let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();

        let layer = &weights.layers[0];

        // Quantized byte-level split should preserve Q8_0 format
        assert_eq!(layer.attn_q.dtype(), TensorDtype::Q8_0, "Q dtype preserved");
        assert_eq!(layer.attn_k.dtype(), TensorDtype::Q8_0, "K dtype preserved");
        assert_eq!(layer.attn_v.dtype(), TensorDtype::Q8_0, "V dtype preserved");

        // Asymmetric shapes: Q=[h,h], K=[n_kv,h], V=[n_kv,h]
        assert_eq!(layer.attn_q.shape(), &[h, h], "Q shape with GQA Q8_0");
        assert_eq!(layer.attn_k.shape(), &[n_kv, h], "K shape with GQA Q8_0");
        assert_eq!(layer.attn_v.shape(), &[n_kv, h], "V shape with GQA Q8_0");

        std::fs::remove_file(&path).ok();
    }
}
