//! Compute backend trait and device tensor abstraction.
//!
//! Defines the [`ComputeBackend`] trait with all operations needed for transformer inference,
//! and [`DeviceTensor`] as a wrapper for tensors that live on a compute device.

pub mod cpu;

use crate::tensor::{Tensor, TensorDtype};

/// A tensor that lives on a compute device (CPU, GPU, etc.).
///
/// For the CPU backend, this is a simple wrapper around [`Tensor`].
/// GPU backends will extend this with device-specific buffer handles.
#[derive(Debug, Clone)]
pub struct DeviceTensor {
    pub tensor: Tensor,
}

impl DeviceTensor {
    /// Create a new device tensor wrapping the given tensor.
    pub fn new(tensor: Tensor) -> Self {
        Self { tensor }
    }

    /// Returns the shape of the underlying tensor.
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    /// Returns the data type of the underlying tensor.
    pub fn dtype(&self) -> TensorDtype {
        self.tensor.dtype()
    }
}

/// Compute backend trait -- all operations needed for transformer inference.
///
/// Implementations exist for CPU, Metal (macOS), and CUDA (NVIDIA).
/// All backends operate on [`DeviceTensor`] values, which wrap tensors
/// stored on the relevant device.
pub trait ComputeBackend: Send + Sync {
    /// Upload a host tensor to the device.
    fn upload(&self, tensor: &Tensor) -> DeviceTensor;

    /// Download a device tensor back to the host.
    fn download(&self, tensor: &DeviceTensor) -> Tensor;

    /// Matrix multiplication: [M, K] x [K, N] -> [M, N]
    fn matmul(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor;

    /// Matrix multiply with transposed B: [M, K] x [N, K]^T -> [M, N]
    fn matmul_transpose(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor;

    /// Quantized matmul: quantized weights x f32 input -> f32 output.
    ///
    /// Weights [N, K] in Q8_0 or Q4_0 format, input [M, K] in f32.
    /// Produces output [M, N] in f32.
    /// Dequantizes on-the-fly during the dot product (fused dequant).
    fn quantized_matmul(&self, weights: &DeviceTensor, input: &DeviceTensor) -> DeviceTensor;

    /// Element-wise addition: a + b (shapes must match).
    fn add(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor;

    /// Add bias to each row: [M, N] + [N] -> [M, N]
    fn add_bias(&self, a: &DeviceTensor, bias: &DeviceTensor) -> DeviceTensor;

    /// GELU activation (approximate form).
    fn gelu(&self, t: &DeviceTensor) -> DeviceTensor;

    /// SiLU activation: x * sigmoid(x)
    fn silu(&self, t: &DeviceTensor) -> DeviceTensor;

    /// SwiGLU: silu(gate) * up (element-wise)
    fn swiglu(&self, gate: &DeviceTensor, up: &DeviceTensor) -> DeviceTensor;

    /// LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
    fn layer_norm(
        &self,
        t: &DeviceTensor,
        weight: &DeviceTensor,
        bias: &DeviceTensor,
        eps: f32,
    ) -> DeviceTensor;

    /// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
    fn rms_norm(&self, t: &DeviceTensor, weight: &DeviceTensor, eps: f32) -> DeviceTensor;

    /// Softmax over last dimension (per row for 2D).
    fn softmax(&self, t: &DeviceTensor) -> DeviceTensor;

    /// Scale every element of the tensor by a scalar factor.
    fn scale(&self, t: &DeviceTensor, factor: f32) -> DeviceTensor;

    /// Apply causal attention mask: set positions where col > row to -inf.
    fn apply_causal_mask(&self, scores: &DeviceTensor, seq_len: usize) -> DeviceTensor;

    /// Apply Rotary Position Embeddings (RoPE) to Q and K tensors.
    ///
    /// q, k: [seq_len, n_heads * head_dim]
    /// `rope_dim` specifies how many dimensions per head to rotate (the rest pass through).
    /// When `rope_dim == head_dim`, all dimensions are rotated (the common case).
    /// Returns (q_rotated, k_rotated) with the same shapes.
    fn rope(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        pos_offset: usize,
        freq_base: f32,
        head_dim: usize,
        rope_dim: usize,
    ) -> (DeviceTensor, DeviceTensor);

    /// Mean pooling with attention mask.
    ///
    /// hidden: [seq_len, hidden_size], mask: [seq_len]
    /// Returns: [hidden_size] (1D vector)
    fn mean_pool(&self, hidden: &DeviceTensor, mask: &[f32]) -> DeviceTensor;

    /// L2 normalize a vector.
    fn l2_normalize(&self, t: &DeviceTensor) -> DeviceTensor;

    /// Embedding lookup: select rows from embedding table by token IDs.
    ///
    /// table: [vocab_size, hidden_size], ids: token IDs
    /// Returns: [len(ids), hidden_size]
    fn embedding_lookup(&self, table: &DeviceTensor, ids: &[u32]) -> DeviceTensor;
}
