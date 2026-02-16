//! Compute backend trait and device tensor abstraction.
//!
//! Defines the [`ComputeBackend`] trait with all operations needed for transformer inference,
//! and [`DeviceTensor`] as a wrapper for tensors that live on a compute device.

pub mod cpu;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
#[cfg(feature = "cuda")]
pub mod cuda;
mod dl;

use std::any::Any;
use std::sync::Arc;

use tracing::info;

use crate::tensor::{Tensor, TensorDtype};

/// Storage for device tensor data.
///
/// CPU storage wraps a [`Tensor`] directly. GPU backends store an opaque
/// buffer handle via `Box<dyn Any + Send + Sync>`.
#[allow(dead_code)]
enum DeviceStorage {
    Cpu(Tensor),
    Gpu(Box<dyn Any + Send + Sync>),
}

impl std::fmt::Debug for DeviceStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceStorage::Cpu(t) => write!(f, "Cpu({:?})", t),
            DeviceStorage::Gpu(_) => write!(f, "Gpu(...)"),
        }
    }
}

impl Clone for DeviceStorage {
    fn clone(&self) -> Self {
        match self {
            DeviceStorage::Cpu(t) => DeviceStorage::Cpu(t.clone()),
            DeviceStorage::Gpu(_) => panic!("GPU DeviceTensor cannot be cloned; download first"),
        }
    }
}

/// A tensor that lives on a compute device (CPU, GPU, etc.).
///
/// For the CPU backend, wraps a [`Tensor`] directly.
/// GPU backends store an opaque buffer handle alongside shape/dtype metadata.
#[derive(Debug, Clone)]
pub struct DeviceTensor {
    shape: Vec<usize>,
    dtype: TensorDtype,
    storage: DeviceStorage,
}

impl DeviceTensor {
    /// Create a CPU-resident device tensor wrapping the given tensor.
    pub fn new(tensor: Tensor) -> Self {
        let shape = tensor.shape().to_vec();
        let dtype = tensor.dtype();
        Self {
            shape,
            dtype,
            storage: DeviceStorage::Cpu(tensor),
        }
    }

    /// Create a GPU-resident device tensor with an opaque buffer.
    #[allow(dead_code)]
    pub(crate) fn from_gpu(
        shape: Vec<usize>,
        dtype: TensorDtype,
        inner: Box<dyn Any + Send + Sync>,
    ) -> Self {
        Self {
            shape,
            dtype,
            storage: DeviceStorage::Gpu(inner),
        }
    }

    /// Returns the shape of this tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the data type of this tensor.
    pub fn dtype(&self) -> TensorDtype {
        self.dtype
    }

    /// Try to get the underlying CPU tensor (returns `None` for GPU tensors).
    pub fn try_as_tensor(&self) -> Option<&Tensor> {
        match &self.storage {
            DeviceStorage::Cpu(t) => Some(t),
            DeviceStorage::Gpu(_) => None,
        }
    }

    /// Get the underlying CPU tensor. Panics if this is a GPU tensor.
    pub fn as_tensor(&self) -> &Tensor {
        self.try_as_tensor()
            .expect("DeviceTensor::as_tensor called on GPU-resident tensor; download first")
    }

    /// Downcast the GPU-resident inner buffer to a concrete type.
    #[allow(dead_code)]
    pub(crate) fn gpu_inner<T: 'static>(&self) -> Option<&T> {
        match &self.storage {
            DeviceStorage::Gpu(inner) => inner.downcast_ref::<T>(),
            DeviceStorage::Cpu(_) => None,
        }
    }
}

// Backwards compatibility: allow tests and code that used `.tensor` to keep working
// via the `as_tensor()` method. The `pub tensor` field is replaced by storage.
// Direct accesses like `dt.tensor.as_f32()` should use `dt.as_tensor().as_f32()`.

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

    /// Element-wise multiply with broadcast: [M, N] * [N] -> [M, N] or same-shape.
    fn mul(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor;

    /// Tanh activation (element-wise).
    fn tanh(&self, t: &DeviceTensor) -> DeviceTensor;

    /// GELU-gated linear unit: gelu(gate) * up (element-wise).
    ///
    /// Used by Gemma models whose FFN uses GELU gating instead of SiLU.
    fn geglu(&self, gate: &DeviceTensor, up: &DeviceTensor) -> DeviceTensor;

    /// RoPE with NeoX-style half-split dimension pairing.
    ///
    /// Pairs `(x[i], x[i + n_dims/2])` instead of `(x[2i], x[2i+1])`.
    /// Used by Gemma3 and most modern models.
    fn rope_neox(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        pos_offset: usize,
        freq_base: f32,
        head_dim: usize,
        rope_dim: usize,
    ) -> (DeviceTensor, DeviceTensor);
}

/// Auto-detect and return the best available compute backend.
///
/// Priority: CUDA > Metal > CPU.
pub fn select_backend() -> Arc<dyn ComputeBackend> {
    // Try CUDA first (if feature enabled)
    #[cfg(feature = "cuda")]
    {
        match cuda::CudaBackend::try_new() {
            Ok(backend) => {
                info!("Selected CUDA backend");
                return Arc::new(backend);
            }
            Err(e) => {
                info!(error = %e, "CUDA not available, trying next backend");
            }
        }
    }

    // Try Metal (macOS only, if feature enabled)
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        match metal::MetalBackend::try_new() {
            Ok(backend) => {
                info!("Selected Metal backend");
                return Arc::new(backend);
            }
            Err(e) => {
                info!(error = %e, "Metal not available, falling back to CPU");
            }
        }
    }

    // CPU fallback (always available)
    info!("Selected CPU backend");
    Arc::new(cpu::CpuBackend::new())
}
