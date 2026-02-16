//! CUDA compute backend for GPU-accelerated tensor operations.
//!
//! Loads the CUDA driver at runtime (no link-time dependency) and dispatches
//! all tensor operations to GPU kernels written in PTX. Falls back gracefully
//! if CUDA is not available — `CudaBackend::try_new()` returns `Err` and the
//! backend selector moves on to the next option.
//!
//! Key differences from strata-core's CUDA backend:
//! - 21 ComputeBackend trait methods + new operations (quantized_matmul, rms_norm,
//!   silu, swiglu, geglu, rope, rope_neox, causal_mask, mul, tanh, l2_normalize,
//!   embedding_lookup)
//! - Uses `DeviceTensor::from_gpu()` with `CudaBuffer` inner type
//! - No batched ops, no head transpose/untranspose, no slice/scatter
//! - Error type: `InferenceError` for `try_new()` return

use std::os::raw::c_void;
use std::sync::Arc;

use tracing::{debug, trace};

use crate::error::InferenceError;
use crate::tensor::{Tensor, TensorDtype, TensorStorage};

use super::{ComputeBackend, DeviceTensor};
use ffi::{CUdeviceptr, CUfunction, CUmodule, CUstream, CublasApi, CudaApi};

pub mod ffi;
pub mod kernels;

// ---------------------------------------------------------------------------
// CudaBuffer — RAII wrapper for a device allocation
// ---------------------------------------------------------------------------

/// A buffer allocated on the CUDA device.
///
/// Automatically freed when dropped. Prefers `cuMemFreeAsync` (stream-ordered,
/// <1us) when available, falls back to synchronous `cuMemFree`.
struct CudaBuffer {
    ptr: CUdeviceptr,
    #[allow(dead_code)]
    len: usize, // in bytes — retained for debugging / future introspection
    api: Arc<CudaApi>,
    stream: CUstream,
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // Try async free first (stream-ordered, fast). Fall back to sync
            // free if the stream has been destroyed or async API is unavailable.
            if self.api.mem_free_async(self.ptr, self.stream).is_err() {
                if let Err(e) = self.api.mem_free(self.ptr) {
                    tracing::warn!(error = %e, "CUDA: failed to free device memory");
                }
            }
        }
    }
}

// SAFETY: CudaBuffer holds a u64 device pointer and an Arc to thread-safe API.
// The device pointer is just an integer handle; it does not reference host memory.
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

// ---------------------------------------------------------------------------
// CudaBackend
// ---------------------------------------------------------------------------

/// CUDA compute backend.
///
/// Manages a CUDA context, stream, loaded PTX module, and pre-resolved kernel
/// function handles. All tensor operations are dispatched as asynchronous kernel
/// launches on the backend's stream, with synchronization at download boundaries.
pub struct CudaBackend {
    api: Arc<CudaApi>,
    stream: CUstream,
    module: CUmodule,
    cublas: Option<CublasApi>,

    // Pre-loaded kernel function handles (22 total)
    fn_gemm: CUfunction,
    fn_gemm_transpose: CUfunction,
    fn_gelu: CUfunction,
    fn_add_tensor: CUfunction,
    fn_add_bias: CUfunction,
    fn_scale: CUfunction,
    fn_layer_norm: CUfunction,
    fn_softmax_rows: CUfunction,
    fn_mean_pool: CUfunction,
    fn_quantized_matmul_q8_0: CUfunction,
    fn_quantized_matmul_q4_0: CUfunction,
    fn_rms_norm: CUfunction,
    fn_silu: CUfunction,
    fn_swiglu: CUfunction,
    fn_geglu: CUfunction,
    fn_rope_norm: CUfunction,
    fn_rope_neox: CUfunction,
    fn_causal_mask: CUfunction,
    fn_mul_elementwise: CUfunction,
    fn_tanh_kernel: CUfunction,
    fn_l2_normalize: CUfunction,
    fn_embedding_lookup: CUfunction,
}

// SAFETY: All CUDA function handles are process-global, and the Driver API is
// documented as thread-safe. The stream is used behind &self which Rust's
// borrow checker already serialises for &mut operations.
unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl CudaBackend {
    /// Attempt to create a new CUDA backend.
    ///
    /// This will:
    /// 1. Load the CUDA driver library and initialise it.
    /// 2. Create a context on device 0.
    /// 3. Create a compute stream.
    /// 4. Optionally load cuBLAS for accelerated GEMM.
    /// 5. Load the PTX module containing all kernels.
    /// 6. Resolve every kernel function handle.
    ///
    /// Returns `Err` if any step fails (no CUDA driver, no GPU, PTX load error, etc.).
    pub fn try_new() -> Result<Self, InferenceError> {
        let api = Arc::new(
            CudaApi::load().map_err(|e| InferenceError::Backend(format!("CUDA load: {}", e)))?,
        );

        let stream = api
            .stream_create()
            .map_err(|e| InferenceError::Backend(format!("CUDA stream: {}", e)))?;

        // Try to load cuBLAS for accelerated GEMM. Falls back to PTX if unavailable.
        let cublas = match CublasApi::load(stream) {
            Ok(api) => {
                tracing::info!("cuBLAS loaded for accelerated GEMM");
                Some(api)
            }
            Err(e) => {
                tracing::info!(error = %e, "cuBLAS not available, using PTX GEMM kernels");
                None
            }
        };

        // Load the PTX module. cuModuleLoadData expects a null-terminated string.
        let ptx = kernels::PTX_MODULE;
        let module = api
            .module_load_data(ptx.as_ptr() as *const c_void)
            .map_err(|e| InferenceError::Backend(format!("CUDA PTX load: {}", e)))?;

        // Resolve all kernel functions.
        macro_rules! get_fn {
            ($name:expr) => {{
                let cname = concat!($name, "\0");
                let cstr =
                    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(cname.as_bytes()) };
                api.module_get_function(module, cstr)
                    .map_err(|e| InferenceError::Backend(format!("CUDA kernel '{}': {}", $name, e)))?
            }};
        }

        let fn_gemm = get_fn!("gemm");
        let fn_gemm_transpose = get_fn!("gemm_transpose");
        let fn_gelu = get_fn!("gelu");
        let fn_add_tensor = get_fn!("add_tensor");
        let fn_add_bias = get_fn!("add_bias");
        let fn_scale = get_fn!("scale");
        let fn_layer_norm = get_fn!("layer_norm");
        let fn_softmax_rows = get_fn!("softmax_rows");
        let fn_mean_pool = get_fn!("mean_pool");
        let fn_quantized_matmul_q8_0 = get_fn!("quantized_matmul_q8_0");
        let fn_quantized_matmul_q4_0 = get_fn!("quantized_matmul_q4_0");
        let fn_rms_norm = get_fn!("rms_norm");
        let fn_silu = get_fn!("silu");
        let fn_swiglu = get_fn!("swiglu");
        let fn_geglu = get_fn!("geglu");
        let fn_rope_norm = get_fn!("rope_norm");
        let fn_rope_neox = get_fn!("rope_neox");
        let fn_causal_mask = get_fn!("causal_mask");
        let fn_mul_elementwise = get_fn!("mul_elementwise");
        let fn_tanh_kernel = get_fn!("tanh_kernel");
        let fn_l2_normalize = get_fn!("l2_normalize");
        let fn_embedding_lookup = get_fn!("embedding_lookup");

        debug!("CudaBackend initialized with 22 kernel functions");

        Ok(Self {
            api,
            stream,
            module,
            cublas,
            fn_gemm,
            fn_gemm_transpose,
            fn_gelu,
            fn_add_tensor,
            fn_add_bias,
            fn_scale,
            fn_layer_norm,
            fn_softmax_rows,
            fn_mean_pool,
            fn_quantized_matmul_q8_0,
            fn_quantized_matmul_q4_0,
            fn_rms_norm,
            fn_silu,
            fn_swiglu,
            fn_geglu,
            fn_rope_norm,
            fn_rope_neox,
            fn_causal_mask,
            fn_mul_elementwise,
            fn_tanh_kernel,
            fn_l2_normalize,
            fn_embedding_lookup,
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Allocate device memory and copy host f32 data to it.
    fn upload_f32(&self, data: &[f32]) -> Result<CudaBuffer, String> {
        let bytesize = data.len() * std::mem::size_of::<f32>();
        let ptr = self.api.mem_alloc_async(bytesize, self.stream)?;
        self.api
            .memcpy_h_to_d(ptr, data.as_ptr() as *const c_void, bytesize)?;
        Ok(CudaBuffer {
            ptr,
            len: bytesize,
            api: Arc::clone(&self.api),
            stream: self.stream,
        })
    }

    /// Allocate device memory and copy raw bytes to it.
    fn upload_raw(&self, data: &[u8]) -> Result<CudaBuffer, String> {
        let bytesize = data.len();
        let ptr = self.api.mem_alloc_async(bytesize, self.stream)?;
        self.api
            .memcpy_h_to_d(ptr, data.as_ptr() as *const c_void, bytesize)?;
        Ok(CudaBuffer {
            ptr,
            len: bytesize,
            api: Arc::clone(&self.api),
            stream: self.stream,
        })
    }

    /// Allocate zeroed device memory for `n` f32 elements.
    fn alloc_zeros_f32(&self, n: usize) -> Result<CudaBuffer, String> {
        let bytesize = n * std::mem::size_of::<f32>();
        let ptr = self.api.mem_alloc_async(bytesize, self.stream)?;
        // cuMemsetD32Async sets each 32-bit word on the compute stream;
        // 0u32 corresponds to 0.0f32.
        self.api.memset_d32_async(ptr, 0, n, self.stream)?;
        Ok(CudaBuffer {
            ptr,
            len: bytesize,
            api: Arc::clone(&self.api),
            stream: self.stream,
        })
    }

    /// Upload u32 data to the device.
    fn upload_u32(&self, data: &[u32]) -> Result<CudaBuffer, String> {
        let bytesize = data.len() * std::mem::size_of::<u32>();
        let ptr = self.api.mem_alloc_async(bytesize, self.stream)?;
        self.api
            .memcpy_h_to_d(ptr, data.as_ptr() as *const c_void, bytesize)?;
        Ok(CudaBuffer {
            ptr,
            len: bytesize,
            api: Arc::clone(&self.api),
            stream: self.stream,
        })
    }

    /// Download `n` f32 elements from device to host.
    fn download_f32(&self, ptr: CUdeviceptr, n: usize) -> Result<Vec<f32>, String> {
        let mut host = vec![0.0f32; n];
        let bytesize = n * std::mem::size_of::<f32>();
        self.api
            .memcpy_d_to_h(host.as_mut_ptr() as *mut c_void, ptr, bytesize)?;
        Ok(host)
    }

    /// Download raw bytes from device to host.
    fn download_raw(&self, ptr: CUdeviceptr, n_bytes: usize) -> Result<Vec<u8>, String> {
        let mut host = vec![0u8; n_bytes];
        self.api
            .memcpy_d_to_h(host.as_mut_ptr() as *mut c_void, ptr, n_bytes)?;
        Ok(host)
    }

    /// Get the `CudaBuffer` from a GPU-resident `DeviceTensor`.
    ///
    /// Panics if the tensor is CPU-resident (should have been uploaded first).
    fn get_buf(dt: &DeviceTensor) -> &CudaBuffer {
        dt.gpu_inner::<CudaBuffer>()
            .expect("CudaBackend: expected GPU-resident CudaBuffer in DeviceTensor")
    }

    /// Wrap a `CudaBuffer` into a `DeviceTensor` with the given shape and dtype.
    fn wrap(buf: CudaBuffer, shape: Vec<usize>, dtype: TensorDtype) -> DeviceTensor {
        DeviceTensor::from_gpu(shape, dtype, Box::new(buf))
    }

    /// Synchronize the compute stream.
    fn sync(&self) {
        if let Err(e) = self.api.stream_synchronize(self.stream) {
            tracing::warn!(error = %e, "CUDA: stream synchronize failed");
        }
    }

    /// Launch a kernel with the given grid/block configuration and parameters.
    ///
    /// # Safety
    ///
    /// `params` must be a correctly constructed parameter array matching the
    /// kernel signature.
    unsafe fn launch(
        &self,
        func: CUfunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        params: &mut [*mut c_void],
    ) {
        if let Err(e) = self.api.launch_kernel(
            func,
            grid,
            block,
            shared_mem,
            self.stream,
            params.as_mut_ptr(),
        ) {
            tracing::warn!(error = %e, "CUDA: kernel launch failed");
        }
    }

    /// Integer ceiling division.
    fn div_ceil(a: u32, b: u32) -> u32 {
        (a + b - 1) / b
    }

    /// Get 2D shape (rows, cols) from a DeviceTensor, flattening all but the
    /// last dimension into rows.
    fn shape_2d(dt: &DeviceTensor) -> (usize, usize) {
        let shape = dt.shape();
        match shape.len() {
            1 => (1, shape[0]),
            2 => (shape[0], shape[1]),
            _ => {
                let cols = *shape.last().unwrap();
                let rows: usize = shape.iter().take(shape.len() - 1).product();
                (rows, cols)
            }
        }
    }

    /// Copy device memory from src to dst buffer (same size).
    fn copy_device_to_device(
        &self,
        dst: CUdeviceptr,
        src: CUdeviceptr,
        bytesize: usize,
    ) -> Result<(), String> {
        // Use cuMemcpyDtoD via H2D trick: read src to host, write to dst.
        // For simplicity, we use memcpy_d_to_h + memcpy_h_to_d.
        // A proper implementation would use cuMemcpy, but our FFI doesn't expose it.
        let mut temp = vec![0u8; bytesize];
        self.api
            .memcpy_d_to_h(temp.as_mut_ptr() as *mut c_void, src, bytesize)?;
        self.api
            .memcpy_h_to_d(dst, temp.as_ptr() as *const c_void, bytesize)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RoPE dispatch helper — shared by rope() and rope_neox()
    // -----------------------------------------------------------------------

    /// Dispatch a RoPE kernel (rope_norm or rope_neox) on a single tensor.
    ///
    /// Input layout: [seq_len, n_heads * head_dim] (2D flat).
    /// Returns a new DeviceTensor with the same shape.
    unsafe fn dispatch_rope(
        &self,
        func: CUfunction,
        input_dt: &DeviceTensor,
        pos_offset: usize,
        freq_base: f32,
        head_dim: usize,
        rope_dim: usize,
    ) -> DeviceTensor {
        let shape = input_dt.shape();
        let seq_len = shape[0];
        let total_cols = shape[1];
        let n_heads = total_cols / head_dim;
        let half_rope = rope_dim / 2;

        let count = seq_len * total_cols;
        let out_buf = match self.alloc_zeros_f32(count) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: rope alloc failed");
                panic!("CUDA: failed to allocate rope output: {}", e);
            }
        };

        let in_buf = Self::get_buf(input_dt);

        // If rope_dim < head_dim, copy input to output first so non-rotated dims
        // are preserved (the kernel only writes the rotated portion).
        if rope_dim < head_dim {
            let bytesize = count * std::mem::size_of::<f32>();
            let _ = self.copy_device_to_device(out_buf.ptr, in_buf.ptr, bytesize);
        }

        let mut p_in = in_buf.ptr;
        let mut p_out = out_buf.ptr;
        let mut p_pos_offset = pos_offset as u32;
        let mut p_freq_base = freq_base;
        let mut p_head_dim = head_dim as u32;
        let mut p_rope_dim = rope_dim as u32;
        let mut p_n_heads = n_heads as u32;
        let mut p_seq_len = seq_len as u32;

        let mut params: [*mut c_void; 8] = [
            &mut p_in as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut p_pos_offset as *mut _ as *mut c_void,
            &mut p_freq_base as *mut _ as *mut c_void,
            &mut p_head_dim as *mut _ as *mut c_void,
            &mut p_rope_dim as *mut _ as *mut c_void,
            &mut p_n_heads as *mut _ as *mut c_void,
            &mut p_seq_len as *mut _ as *mut c_void,
        ];

        // 3D grid: (ceil(half_rope/16), ceil(n_heads/16), seq_len)
        let gx = Self::div_ceil(half_rope as u32, 16);
        let gy = Self::div_ceil(n_heads as u32, 16);
        let gz = seq_len as u32;

        self.launch(func, (gx, gy, gz), (16, 16, 1), 0, &mut params);

        Self::wrap(out_buf, vec![seq_len, total_cols], TensorDtype::F32)
    }
}

// ---------------------------------------------------------------------------
// ComputeBackend implementation
// ---------------------------------------------------------------------------

impl ComputeBackend for CudaBackend {
    fn upload(&self, tensor: &Tensor) -> DeviceTensor {
        let shape = tensor.shape().to_vec();
        let dtype = tensor.dtype();

        let buf = match tensor.storage() {
            TensorStorage::F32(data) => {
                trace!(shape = ?shape, dtype = ?dtype, "CUDA upload F32");
                match self.upload_f32(data) {
                    Ok(buf) => buf,
                    Err(e) => {
                        tracing::warn!(error = %e, "CUDA: upload F32 failed");
                        panic!("CUDA: failed to upload F32 tensor: {}", e);
                    }
                }
            }
            TensorStorage::F16(data) => {
                // Upload raw f16 bits as bytes
                trace!(shape = ?shape, dtype = ?dtype, "CUDA upload F16");
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        data.len() * 2,
                    )
                };
                match self.upload_raw(bytes) {
                    Ok(buf) => buf,
                    Err(e) => {
                        tracing::warn!(error = %e, "CUDA: upload F16 failed");
                        panic!("CUDA: failed to upload F16 tensor: {}", e);
                    }
                }
            }
            TensorStorage::Quantized(data) => {
                // Upload raw quantized block data as bytes
                trace!(shape = ?shape, dtype = ?dtype, "CUDA upload Quantized");
                match self.upload_raw(data) {
                    Ok(buf) => buf,
                    Err(e) => {
                        tracing::warn!(error = %e, "CUDA: upload Quantized failed");
                        panic!("CUDA: failed to upload Quantized tensor: {}", e);
                    }
                }
            }
        };

        Self::wrap(buf, shape, dtype)
    }

    fn download(&self, dt: &DeviceTensor) -> Tensor {
        // If the tensor is CPU-resident, just clone it.
        if let Some(tensor) = dt.try_as_tensor() {
            trace!(shape = ?dt.shape(), "CUDA download (CPU tensor, clone)");
            return tensor.clone();
        }

        // GPU tensor — sync the stream and read back.
        self.sync();

        let shape = dt.shape().to_vec();
        let dtype = dt.dtype();
        let buf = Self::get_buf(dt);

        match dtype {
            TensorDtype::F32 => {
                let n_elements: usize = shape.iter().product();
                match self.download_f32(buf.ptr, n_elements) {
                    Ok(data) => {
                        trace!(shape = ?shape, "CUDA download F32");
                        Tensor::new(shape, data)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "CUDA: download F32 failed, returning zeros");
                        Tensor::zeros(&shape)
                    }
                }
            }
            TensorDtype::F16 => {
                let n_elements: usize = shape.iter().product();
                let n_bytes = n_elements * 2;
                match self.download_raw(buf.ptr, n_bytes) {
                    Ok(raw) => {
                        // Reinterpret bytes as u16
                        let data: Vec<u16> = raw
                            .chunks_exact(2)
                            .map(|c| u16::from_le_bytes([c[0], c[1]]))
                            .collect();
                        trace!(shape = ?shape, "CUDA download F16");
                        Tensor::from_f16(shape, data)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "CUDA: download F16 failed, returning zeros");
                        Tensor::zeros(&shape)
                    }
                }
            }
            TensorDtype::Q8_0 | TensorDtype::Q4_0 => {
                let n_elements: usize = shape.iter().product();
                let block_size = dtype.block_size();
                let block_byte_size = dtype.block_byte_size();
                let n_blocks = (n_elements + block_size - 1) / block_size;
                let total_bytes = n_blocks * block_byte_size;
                match self.download_raw(buf.ptr, total_bytes) {
                    Ok(data) => {
                        trace!(shape = ?shape, dtype = ?dtype, "CUDA download quantized");
                        Tensor::from_quantized(shape, dtype, data)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "CUDA: download quantized failed, returning zeros");
                        Tensor::zeros(&shape)
                    }
                }
            }
        }
    }

    fn matmul(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor {
        let (m, k) = Self::shape_2d(a);
        let (k2, n) = Self::shape_2d(b);
        assert_eq!(k, k2, "matmul dimension mismatch: a cols {} != b rows {}", k, k2);

        let out = match self.alloc_zeros_f32(m * n) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: matmul alloc failed");
                panic!("CUDA: failed to allocate matmul output: {}", e);
            }
        };

        let a_buf = Self::get_buf(a);
        let b_buf = Self::get_buf(b);

        // Try cuBLAS first. Row-major C(M,N) = A(M,K) * B(K,N):
        // cuBLAS is column-major, so we compute C^T = B^T * A^T, i.e.:
        // cublasSgemm(OP_N, OP_N, N, M, K, 1, B, N, A, K, 0, C, N)
        if let Some(ref cublas) = self.cublas {
            if cublas
                .sgemm(
                    ffi::CUBLAS_OP_N,
                    ffi::CUBLAS_OP_N,
                    n as i32,
                    m as i32,
                    k as i32,
                    1.0,
                    b_buf.ptr,
                    n as i32,
                    a_buf.ptr,
                    k as i32,
                    0.0,
                    out.ptr,
                    n as i32,
                )
                .is_ok()
            {
                return Self::wrap(out, vec![m, n], TensorDtype::F32);
            }
        }

        // Fallback: PTX GEMM kernel
        let mut p_a = a_buf.ptr;
        let mut p_b = b_buf.ptr;
        let mut p_c = out.ptr;
        let mut p_m = m as u32;
        let mut p_k = k as u32;
        let mut p_n = n as u32;

        let mut params: [*mut c_void; 6] = [
            &mut p_a as *mut _ as *mut c_void,
            &mut p_b as *mut _ as *mut c_void,
            &mut p_c as *mut _ as *mut c_void,
            &mut p_m as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
        ];

        let grid = (
            Self::div_ceil(n as u32, 16),
            Self::div_ceil(m as u32, 16),
            1,
        );
        let block = (16, 16, 1);
        unsafe {
            self.launch(self.fn_gemm, grid, block, 0, &mut params);
        }

        Self::wrap(out, vec![m, n], TensorDtype::F32)
    }

    fn matmul_transpose(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor {
        let (m, k) = Self::shape_2d(a);
        let (n, k2) = Self::shape_2d(b); // B is (N, K) and we treat it as transposed
        assert_eq!(
            k, k2,
            "matmul_transpose dimension mismatch: a cols {} != b cols {}",
            k, k2
        );

        let out = match self.alloc_zeros_f32(m * n) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: matmul_transpose alloc failed");
                panic!("CUDA: failed to allocate matmul_transpose output: {}", e);
            }
        };

        let a_buf = Self::get_buf(a);
        let b_buf = Self::get_buf(b);

        // Try cuBLAS first. Row-major C(M,N) = A(M,K) * B(N,K)^T:
        // cuBLAS column-major: C^T = B * A^T, i.e.:
        // cublasSgemm(OP_T, OP_N, N, M, K, 1, B, K, A, K, 0, C, N)
        if let Some(ref cublas) = self.cublas {
            if cublas
                .sgemm(
                    ffi::CUBLAS_OP_T,
                    ffi::CUBLAS_OP_N,
                    n as i32,
                    m as i32,
                    k as i32,
                    1.0,
                    b_buf.ptr,
                    k as i32,
                    a_buf.ptr,
                    k as i32,
                    0.0,
                    out.ptr,
                    n as i32,
                )
                .is_ok()
            {
                return Self::wrap(out, vec![m, n], TensorDtype::F32);
            }
        }

        // Fallback: PTX GEMM transpose kernel
        let mut p_a = a_buf.ptr;
        let mut p_b = b_buf.ptr;
        let mut p_c = out.ptr;
        let mut p_m = m as u32;
        let mut p_k = k as u32;
        let mut p_n = n as u32;

        let mut params: [*mut c_void; 6] = [
            &mut p_a as *mut _ as *mut c_void,
            &mut p_b as *mut _ as *mut c_void,
            &mut p_c as *mut _ as *mut c_void,
            &mut p_m as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
        ];

        let grid = (
            Self::div_ceil(n as u32, 16),
            Self::div_ceil(m as u32, 16),
            1,
        );
        let block = (16, 16, 1);
        unsafe {
            self.launch(self.fn_gemm_transpose, grid, block, 0, &mut params);
        }

        Self::wrap(out, vec![m, n], TensorDtype::F32)
    }

    fn quantized_matmul(&self, weights: &DeviceTensor, input: &DeviceTensor) -> DeviceTensor {
        // Weights: [N, K] in Q8_0 or Q4_0, Input: [M, K] in F32 → Output: [M, N] in F32
        let input_shape = input.shape();
        let weight_shape = weights.shape();
        let dtype = weights.dtype();

        let m = if input_shape.len() == 1 { 1 } else { input_shape[0] };
        let k = *input_shape.last().unwrap();
        let n = weight_shape[0];

        assert_eq!(
            k,
            weight_shape[1],
            "quantized_matmul: input cols ({}) must match weight cols ({})",
            k,
            weight_shape[1]
        );

        let out = match self.alloc_zeros_f32(m * n) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: quantized_matmul alloc failed");
                panic!("CUDA: failed to allocate quantized_matmul output: {}", e);
            }
        };

        let w_buf = Self::get_buf(weights);
        let in_buf = Self::get_buf(input);

        let func = match dtype {
            TensorDtype::Q8_0 => {
                trace!(m, k, n, "CUDA quantized_matmul Q8_0");
                self.fn_quantized_matmul_q8_0
            }
            TensorDtype::Q4_0 => {
                trace!(m, k, n, "CUDA quantized_matmul Q4_0");
                self.fn_quantized_matmul_q4_0
            }
            _ if dtype.is_quantized() => {
                // K-quant and other types without native CUDA kernels: dequant fallback.
                tracing::debug!(
                    ?dtype,
                    "CUDA quantized_matmul: no native kernel for {:?}, using dequant fallback",
                    dtype
                );
                let weights_f32 = weights.as_tensor().to_f32();
                let weights_f32_dev = self.upload(&weights_f32);
                return self.matmul_transpose(input, &weights_f32_dev);
            }
            _ => panic!("quantized_matmul: unsupported dtype {:?}", dtype),
        };

        // Kernel params: weights_ptr, input_ptr, output_ptr, N, K, M
        let mut p_w = w_buf.ptr;
        let mut p_in = in_buf.ptr;
        let mut p_out = out.ptr;
        let mut p_n = n as u32;
        let mut p_k = k as u32;
        let mut p_m = m as u32;

        let mut params: [*mut c_void; 6] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_in as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_m as *mut _ as *mut c_void,
        ];

        // Grid: (N, M, 1), Block: (256, 1, 1)
        let grid = (n as u32, m as u32, 1);
        let block = (256, 1, 1);
        unsafe {
            self.launch(func, grid, block, 0, &mut params);
        }

        Self::wrap(out, vec![m, n], TensorDtype::F32)
    }

    fn add(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor {
        let n: usize = a.shape().iter().product();

        let out = match self.alloc_zeros_f32(n) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: add alloc failed");
                panic!("CUDA: failed to allocate add output: {}", e);
            }
        };

        let a_buf = Self::get_buf(a);
        let b_buf = Self::get_buf(b);

        let mut p_a = a_buf.ptr;
        let mut p_b = b_buf.ptr;
        let mut p_c = out.ptr;
        let mut p_n = n as u32;

        let mut params: [*mut c_void; 4] = [
            &mut p_a as *mut _ as *mut c_void,
            &mut p_b as *mut _ as *mut c_void,
            &mut p_c as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
        ];

        let grid = (Self::div_ceil(n as u32, 256), 1, 1);
        let block = (256, 1, 1);
        unsafe {
            self.launch(self.fn_add_tensor, grid, block, 0, &mut params);
        }

        Self::wrap(out, a.shape().to_vec(), TensorDtype::F32)
    }

    fn add_bias(&self, a: &DeviceTensor, bias: &DeviceTensor) -> DeviceTensor {
        // [M, N] + [N] -> [M, N]
        let a_shape = a.shape();
        let rows = if a_shape.len() == 1 { 1 } else { a_shape[0] };
        let cols = *a_shape.last().unwrap();

        let n_total = rows * cols;
        let out = match self.alloc_zeros_f32(n_total) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: add_bias alloc failed");
                panic!("CUDA: failed to allocate add_bias output: {}", e);
            }
        };

        let a_buf = Self::get_buf(a);
        let bias_buf = Self::get_buf(bias);

        // Copy a to output first, then add bias in-place
        // Actually, our add_bias kernel does: out[row * cols + col] = a[row * cols + col] + bias[col]
        // So we dispatch directly.
        let mut p_a = a_buf.ptr;
        let mut p_bias = bias_buf.ptr;
        let mut p_out = out.ptr;
        let mut p_rows = rows as u32;
        let mut p_cols = cols as u32;

        let mut params: [*mut c_void; 5] = [
            &mut p_a as *mut _ as *mut c_void,
            &mut p_bias as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut p_rows as *mut _ as *mut c_void,
            &mut p_cols as *mut _ as *mut c_void,
        ];

        let grid = (rows as u32, Self::div_ceil(cols as u32, 256), 1);
        let block = (256, 1, 1);
        unsafe {
            self.launch(self.fn_add_bias, grid, block, 0, &mut params);
        }

        Self::wrap(out, a_shape.to_vec(), TensorDtype::F32)
    }

    fn gelu(&self, t: &DeviceTensor) -> DeviceTensor {
        let n: usize = t.shape().iter().product();

        let out = match self.alloc_zeros_f32(n) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: gelu alloc failed");
                panic!("CUDA: failed to allocate gelu output: {}", e);
            }
        };

        let t_buf = Self::get_buf(t);

        let mut p_in = t_buf.ptr;
        let mut p_out = out.ptr;
        let mut p_n = n as u32;

        let mut params: [*mut c_void; 3] = [
            &mut p_in as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
        ];

        let grid = (Self::div_ceil(n as u32, 256), 1, 1);
        let block = (256, 1, 1);
        unsafe {
            self.launch(self.fn_gelu, grid, block, 0, &mut params);
        }

        Self::wrap(out, t.shape().to_vec(), TensorDtype::F32)
    }

    fn silu(&self, t: &DeviceTensor) -> DeviceTensor {
        let n: usize = t.shape().iter().product();

        let out = match self.alloc_zeros_f32(n) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: silu alloc failed");
                panic!("CUDA: failed to allocate silu output: {}", e);
            }
        };

        let t_buf = Self::get_buf(t);

        let mut p_in = t_buf.ptr;
        let mut p_out = out.ptr;
        let mut p_n = n as u32;

        let mut params: [*mut c_void; 3] = [
            &mut p_in as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
        ];

        let grid = (Self::div_ceil(n as u32, 256), 1, 1);
        let block = (256, 1, 1);
        unsafe {
            self.launch(self.fn_silu, grid, block, 0, &mut params);
        }

        Self::wrap(out, t.shape().to_vec(), TensorDtype::F32)
    }

    fn swiglu(&self, gate: &DeviceTensor, up: &DeviceTensor) -> DeviceTensor {
        let n: usize = gate.shape().iter().product();

        let out = match self.alloc_zeros_f32(n) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: swiglu alloc failed");
                panic!("CUDA: failed to allocate swiglu output: {}", e);
            }
        };

        let gate_buf = Self::get_buf(gate);
        let up_buf = Self::get_buf(up);

        let mut p_gate = gate_buf.ptr;
        let mut p_up = up_buf.ptr;
        let mut p_out = out.ptr;
        let mut p_n = n as u32;

        let mut params: [*mut c_void; 4] = [
            &mut p_gate as *mut _ as *mut c_void,
            &mut p_up as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
        ];

        let grid = (Self::div_ceil(n as u32, 256), 1, 1);
        let block = (256, 1, 1);
        unsafe {
            self.launch(self.fn_swiglu, grid, block, 0, &mut params);
        }

        Self::wrap(out, gate.shape().to_vec(), TensorDtype::F32)
    }

    fn layer_norm(
        &self,
        t: &DeviceTensor,
        weight: &DeviceTensor,
        bias: &DeviceTensor,
        eps: f32,
    ) -> DeviceTensor {
        let (rows, cols) = Self::shape_2d(t);

        let out = match self.alloc_zeros_f32(rows * cols) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: layer_norm alloc failed");
                panic!("CUDA: failed to allocate layer_norm output: {}", e);
            }
        };

        let t_buf = Self::get_buf(t);
        let w_buf = Self::get_buf(weight);
        let b_buf = Self::get_buf(bias);

        let mut p_in = t_buf.ptr;
        let mut p_out = out.ptr;
        let mut p_w = w_buf.ptr;
        let mut p_b = b_buf.ptr;
        let mut p_rows = rows as u32;
        let mut p_cols = cols as u32;
        let mut p_eps = eps;

        let mut params: [*mut c_void; 7] = [
            &mut p_in as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut p_w as *mut _ as *mut c_void,
            &mut p_b as *mut _ as *mut c_void,
            &mut p_rows as *mut _ as *mut c_void,
            &mut p_cols as *mut _ as *mut c_void,
            &mut p_eps as *mut _ as *mut c_void,
        ];

        // One block per row, reduction within the block
        let block_size = std::cmp::min(256, cols.next_power_of_two()) as u32;
        let grid = (rows as u32, 1, 1);
        let block = (block_size, 1, 1);
        unsafe {
            self.launch(self.fn_layer_norm, grid, block, 0, &mut params);
        }

        Self::wrap(out, t.shape().to_vec(), TensorDtype::F32)
    }

    fn rms_norm(&self, t: &DeviceTensor, weight: &DeviceTensor, eps: f32) -> DeviceTensor {
        let (rows, cols) = Self::shape_2d(t);

        let out = match self.alloc_zeros_f32(rows * cols) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: rms_norm alloc failed");
                panic!("CUDA: failed to allocate rms_norm output: {}", e);
            }
        };

        let t_buf = Self::get_buf(t);
        let w_buf = Self::get_buf(weight);

        let mut p_in = t_buf.ptr;
        let mut p_out = out.ptr;
        let mut p_w = w_buf.ptr;
        let mut p_rows = rows as u32;
        let mut p_cols = cols as u32;
        let mut p_eps = eps;

        let mut params: [*mut c_void; 6] = [
            &mut p_in as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut p_w as *mut _ as *mut c_void,
            &mut p_rows as *mut _ as *mut c_void,
            &mut p_cols as *mut _ as *mut c_void,
            &mut p_eps as *mut _ as *mut c_void,
        ];

        // One block per row, reduction within the block
        let block_size = std::cmp::min(256, cols.next_power_of_two()) as u32;
        let grid = (rows as u32, 1, 1);
        let block = (block_size, 1, 1);
        unsafe {
            self.launch(self.fn_rms_norm, grid, block, 0, &mut params);
        }

        Self::wrap(out, t.shape().to_vec(), TensorDtype::F32)
    }

    fn softmax(&self, t: &DeviceTensor) -> DeviceTensor {
        let (rows, cols) = Self::shape_2d(t);

        let out = match self.alloc_zeros_f32(rows * cols) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: softmax alloc failed");
                panic!("CUDA: failed to allocate softmax output: {}", e);
            }
        };

        let t_buf = Self::get_buf(t);

        // Copy input to output, then softmax in-place on the output buffer
        let bytesize = rows * cols * std::mem::size_of::<f32>();
        if let Err(e) = self.copy_device_to_device(out.ptr, t_buf.ptr, bytesize) {
            tracing::warn!(error = %e, "CUDA: softmax copy failed");
        }

        let mut p_data = out.ptr;
        let mut p_rows = rows as u32;
        let mut p_cols = cols as u32;

        let mut params: [*mut c_void; 3] = [
            &mut p_data as *mut _ as *mut c_void,
            &mut p_rows as *mut _ as *mut c_void,
            &mut p_cols as *mut _ as *mut c_void,
        ];

        // One block per row
        let block_size = std::cmp::min(256, cols.next_power_of_two()) as u32;
        let grid = (rows as u32, 1, 1);
        let block = (block_size, 1, 1);
        unsafe {
            self.launch(self.fn_softmax_rows, grid, block, 0, &mut params);
        }

        Self::wrap(out, t.shape().to_vec(), TensorDtype::F32)
    }

    fn scale(&self, t: &DeviceTensor, factor: f32) -> DeviceTensor {
        let n: usize = t.shape().iter().product();

        let out = match self.alloc_zeros_f32(n) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: scale alloc failed");
                panic!("CUDA: failed to allocate scale output: {}", e);
            }
        };

        let t_buf = Self::get_buf(t);

        // Copy input to output, then scale in-place
        let bytesize = n * std::mem::size_of::<f32>();
        if let Err(e) = self.copy_device_to_device(out.ptr, t_buf.ptr, bytesize) {
            tracing::warn!(error = %e, "CUDA: scale copy failed");
        }

        let mut p_t = out.ptr;
        let mut p_factor = factor;
        let mut p_n = n as u32;

        let mut params: [*mut c_void; 3] = [
            &mut p_t as *mut _ as *mut c_void,
            &mut p_factor as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
        ];

        let grid = (Self::div_ceil(n as u32, 256), 1, 1);
        let block = (256, 1, 1);
        unsafe {
            self.launch(self.fn_scale, grid, block, 0, &mut params);
        }

        Self::wrap(out, t.shape().to_vec(), TensorDtype::F32)
    }

    fn apply_causal_mask(&self, scores: &DeviceTensor, seq_len: usize) -> DeviceTensor {
        let (rows, cols) = Self::shape_2d(scores);

        let out = match self.alloc_zeros_f32(rows * cols) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: causal_mask alloc failed");
                panic!("CUDA: failed to allocate causal_mask output: {}", e);
            }
        };

        let scores_buf = Self::get_buf(scores);

        // Copy input to output, then apply causal mask in-place
        let bytesize = rows * cols * std::mem::size_of::<f32>();
        if let Err(e) = self.copy_device_to_device(out.ptr, scores_buf.ptr, bytesize) {
            tracing::warn!(error = %e, "CUDA: causal_mask copy failed");
        }

        // offset = cols - seq_len for KV-cache scenarios where cols > seq_len.
        // For simple causal mask (square matrix), offset = 0.
        let offset = if cols > seq_len { cols - seq_len } else { 0 };

        let mut p_data = out.ptr;
        let mut p_rows = rows as u32;
        let mut p_cols = cols as u32;
        let mut p_offset = offset as u32;

        let mut params: [*mut c_void; 4] = [
            &mut p_data as *mut _ as *mut c_void,
            &mut p_rows as *mut _ as *mut c_void,
            &mut p_cols as *mut _ as *mut c_void,
            &mut p_offset as *mut _ as *mut c_void,
        ];

        let grid = (
            Self::div_ceil(cols as u32, 16),
            Self::div_ceil(rows as u32, 16),
            1,
        );
        let block = (16, 16, 1);
        unsafe {
            self.launch(self.fn_causal_mask, grid, block, 0, &mut params);
        }

        Self::wrap(out, scores.shape().to_vec(), TensorDtype::F32)
    }

    fn rope(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        pos_offset: usize,
        freq_base: f32,
        head_dim: usize,
        rope_dim: usize,
    ) -> (DeviceTensor, DeviceTensor) {
        assert!(head_dim > 0, "rope: head_dim must be > 0");
        assert!(
            rope_dim <= head_dim,
            "rope: rope_dim ({}) must be <= head_dim ({})",
            rope_dim,
            head_dim
        );
        assert!(rope_dim % 2 == 0, "rope: rope_dim ({}) must be even", rope_dim);

        trace!(
            q_shape = ?q.shape(),
            k_shape = ?k.shape(),
            head_dim,
            rope_dim,
            pos_offset,
            freq_base,
            "CUDA rope"
        );

        let q_out = unsafe {
            self.dispatch_rope(self.fn_rope_norm, q, pos_offset, freq_base, head_dim, rope_dim)
        };
        let k_out = unsafe {
            self.dispatch_rope(self.fn_rope_norm, k, pos_offset, freq_base, head_dim, rope_dim)
        };

        (q_out, k_out)
    }

    fn mean_pool(&self, hidden: &DeviceTensor, mask: &[f32]) -> DeviceTensor {
        let shape = hidden.shape();
        assert_eq!(shape.len(), 2, "mean_pool: hidden must be 2D");

        let rows = shape[0];
        let cols = shape[1];

        assert_eq!(
            mask.len(),
            rows,
            "mean_pool: mask length ({}) must match seq_len ({})",
            mask.len(),
            rows
        );

        // Upload mask as f32 to device
        let mask_buf = match self.upload_f32(mask) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: mean_pool mask upload failed");
                panic!("CUDA: failed to upload mask: {}", e);
            }
        };

        let out = match self.alloc_zeros_f32(cols) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: mean_pool alloc failed");
                panic!("CUDA: failed to allocate mean_pool output: {}", e);
            }
        };

        let hidden_buf = Self::get_buf(hidden);

        let mut p_hidden = hidden_buf.ptr;
        let mut p_mask = mask_buf.ptr;
        let mut p_out = out.ptr;
        let mut p_rows = rows as u32;
        let mut p_cols = cols as u32;

        let mut params: [*mut c_void; 5] = [
            &mut p_hidden as *mut _ as *mut c_void,
            &mut p_mask as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut p_rows as *mut _ as *mut c_void,
            &mut p_cols as *mut _ as *mut c_void,
        ];

        // One block, reduction over rows for each column element
        let grid = (1, 1, 1);
        let block = (256, 1, 1);
        unsafe {
            self.launch(self.fn_mean_pool, grid, block, 0, &mut params);
        }

        Self::wrap(out, vec![cols], TensorDtype::F32)
    }

    fn l2_normalize(&self, t: &DeviceTensor) -> DeviceTensor {
        let n: usize = t.shape().iter().product();

        let out = match self.alloc_zeros_f32(n) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: l2_normalize alloc failed");
                panic!("CUDA: failed to allocate l2_normalize output: {}", e);
            }
        };

        let t_buf = Self::get_buf(t);

        let mut p_in = t_buf.ptr;
        let mut p_out = out.ptr;
        let mut p_n = n as u32;

        let mut params: [*mut c_void; 3] = [
            &mut p_in as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
        ];

        // One block, reduction to compute L2 norm then normalize
        let block_size = std::cmp::min(256, n.next_power_of_two()) as u32;
        let grid = (1, 1, 1);
        let block = (block_size, 1, 1);
        unsafe {
            self.launch(self.fn_l2_normalize, grid, block, 0, &mut params);
        }

        Self::wrap(out, t.shape().to_vec(), TensorDtype::F32)
    }

    fn embedding_lookup(&self, table: &DeviceTensor, ids: &[u32]) -> DeviceTensor {
        let shape = table.shape();
        assert_eq!(shape.len(), 2, "embedding_lookup: table must be 2D");

        let vocab_size = shape[0];
        let hidden_size = shape[1];
        let n_tokens = ids.len();

        // Bounds check: all token IDs must be within vocab range
        for (i, &id) in ids.iter().enumerate() {
            assert!(
                (id as usize) < vocab_size,
                "embedding_lookup: token_id {} at index {} is out of range (vocab_size = {})",
                id, i, vocab_size
            );
        }

        trace!(vocab_size, hidden_size, n_tokens, "CUDA embedding_lookup");

        // For quantized embedding tables, dequantize to F32 first and re-upload.
        // The GPU kernel expects F32 input for simplicity.
        let table_dt;
        let effective_table = if table.dtype() != TensorDtype::F32 {
            // Download, dequantize, re-upload
            let tensor = self.download(table);
            let f32_tensor = tensor.to_f32();
            table_dt = self.upload(&f32_tensor);
            &table_dt
        } else {
            table
        };

        // Upload token IDs
        let ids_buf = match self.upload_u32(ids) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: embedding_lookup ids upload failed");
                panic!("CUDA: failed to upload token IDs: {}", e);
            }
        };

        let out = match self.alloc_zeros_f32(n_tokens * hidden_size) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: embedding_lookup alloc failed");
                panic!("CUDA: failed to allocate embedding_lookup output: {}", e);
            }
        };

        let table_buf = Self::get_buf(effective_table);

        let mut p_table = table_buf.ptr;
        let mut p_ids = ids_buf.ptr;
        let mut p_out = out.ptr;
        let mut p_n_tokens = n_tokens as u32;
        let mut p_hidden = hidden_size as u32;

        let mut params: [*mut c_void; 5] = [
            &mut p_table as *mut _ as *mut c_void,
            &mut p_ids as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut p_n_tokens as *mut _ as *mut c_void,
            &mut p_hidden as *mut _ as *mut c_void,
        ];

        // Grid: (ceil(hidden/256), num_tokens, 1) — kernel uses blockIdx.y as token index
        let grid = (
            Self::div_ceil(hidden_size as u32, 256),
            n_tokens as u32,
            1,
        );
        let block = (256, 1, 1);
        unsafe {
            self.launch(self.fn_embedding_lookup, grid, block, 0, &mut params);
        }

        Self::wrap(out, vec![n_tokens, hidden_size], TensorDtype::F32)
    }

    fn mul(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape == b_shape {
            // Same-shape element-wise multiply
            let n: usize = a_shape.iter().product();

            let out = match self.alloc_zeros_f32(n) {
                Ok(buf) => buf,
                Err(e) => {
                    tracing::warn!(error = %e, "CUDA: mul alloc failed");
                    panic!("CUDA: failed to allocate mul output: {}", e);
                }
            };

            let a_buf = Self::get_buf(a);
            let b_buf = Self::get_buf(b);

            let mut p_a = a_buf.ptr;
            let mut p_b = b_buf.ptr;
            let mut p_out = out.ptr;
            let mut p_n = n as u32;

            let mut params: [*mut c_void; 4] = [
                &mut p_a as *mut _ as *mut c_void,
                &mut p_b as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
                &mut p_n as *mut _ as *mut c_void,
            ];

            let grid = (Self::div_ceil(n as u32, 256), 1, 1);
            let block = (256, 1, 1);
            unsafe {
                self.launch(self.fn_mul_elementwise, grid, block, 0, &mut params);
            }

            Self::wrap(out, a_shape.to_vec(), TensorDtype::F32)
        } else if b_shape.len() == 1 && a_shape.len() == 2 && a_shape[1] == b_shape[0] {
            // Broadcast: [M, N] * [N] -> [M, N]
            // For the broadcast case, we use the same mul_elementwise kernel
            // but need to handle the broadcast on the host side by expanding b.
            // For now, fall back to CPU for broadcast mul.
            let a_tensor = self.download(a);
            let b_tensor = self.download(b);
            let a_data = a_tensor.as_f32();
            let b_data = b_tensor.as_f32();
            let m = a_shape[0];
            let n = a_shape[1];
            let mut result = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    result[i * n + j] = a_data[i * n + j] * b_data[j];
                }
            }
            let out_tensor = Tensor::new(vec![m, n], result);
            self.upload(&out_tensor)
        } else {
            panic!("mul: unsupported shapes {:?} and {:?}", a_shape, b_shape);
        }
    }

    fn tanh(&self, t: &DeviceTensor) -> DeviceTensor {
        let n: usize = t.shape().iter().product();

        let out = match self.alloc_zeros_f32(n) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: tanh alloc failed");
                panic!("CUDA: failed to allocate tanh output: {}", e);
            }
        };

        let t_buf = Self::get_buf(t);

        let mut p_in = t_buf.ptr;
        let mut p_out = out.ptr;
        let mut p_n = n as u32;

        let mut params: [*mut c_void; 3] = [
            &mut p_in as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
        ];

        let grid = (Self::div_ceil(n as u32, 256), 1, 1);
        let block = (256, 1, 1);
        unsafe {
            self.launch(self.fn_tanh_kernel, grid, block, 0, &mut params);
        }

        Self::wrap(out, t.shape().to_vec(), TensorDtype::F32)
    }

    fn geglu(&self, gate: &DeviceTensor, up: &DeviceTensor) -> DeviceTensor {
        let n: usize = gate.shape().iter().product();

        let out = match self.alloc_zeros_f32(n) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!(error = %e, "CUDA: geglu alloc failed");
                panic!("CUDA: failed to allocate geglu output: {}", e);
            }
        };

        let gate_buf = Self::get_buf(gate);
        let up_buf = Self::get_buf(up);

        let mut p_gate = gate_buf.ptr;
        let mut p_up = up_buf.ptr;
        let mut p_out = out.ptr;
        let mut p_n = n as u32;

        let mut params: [*mut c_void; 4] = [
            &mut p_gate as *mut _ as *mut c_void,
            &mut p_up as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
        ];

        let grid = (Self::div_ceil(n as u32, 256), 1, 1);
        let block = (256, 1, 1);
        unsafe {
            self.launch(self.fn_geglu, grid, block, 0, &mut params);
        }

        Self::wrap(out, gate.shape().to_vec(), TensorDtype::F32)
    }

    fn rope_neox(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        pos_offset: usize,
        freq_base: f32,
        head_dim: usize,
        rope_dim: usize,
    ) -> (DeviceTensor, DeviceTensor) {
        assert!(head_dim > 0, "rope_neox: head_dim must be > 0");
        assert!(
            rope_dim <= head_dim,
            "rope_neox: rope_dim ({}) must be <= head_dim ({})",
            rope_dim,
            head_dim
        );
        assert!(
            rope_dim % 2 == 0,
            "rope_neox: rope_dim ({}) must be even",
            rope_dim
        );

        trace!(
            q_shape = ?q.shape(),
            k_shape = ?k.shape(),
            head_dim,
            rope_dim,
            pos_offset,
            freq_base,
            "CUDA rope_neox"
        );

        let q_out = unsafe {
            self.dispatch_rope(
                self.fn_rope_neox,
                q,
                pos_offset,
                freq_base,
                head_dim,
                rope_dim,
            )
        };
        let k_out = unsafe {
            self.dispatch_rope(
                self.fn_rope_neox,
                k,
                pos_offset,
                freq_base,
                head_dim,
                rope_dim,
            )
        };

        (q_out, k_out)
    }
}

// ---------------------------------------------------------------------------
// Drop implementation
// ---------------------------------------------------------------------------

impl Drop for CudaBackend {
    fn drop(&mut self) {
        // Synchronize before cleanup to ensure all work is complete.
        let _ = self.api.stream_synchronize(self.stream);

        // Drop cuBLAS BEFORE destroying the stream it's bound to.
        // cublasDestroy may synchronize on the stream internally.
        drop(self.cublas.take());

        if let Err(e) = self.api.module_unload(self.module) {
            tracing::warn!(error = %e, "CUDA: failed to unload module");
        }
        if let Err(e) = self.api.stream_destroy(self.stream) {
            tracing::warn!(error = %e, "CUDA: failed to destroy stream");
        }
        // Context destruction is handled by CudaApi::drop (which destroys self.api.ctx).
        // We do NOT destroy the context here because the Arc<CudaApi> may still be
        // held by CudaBuffer instances that need to call cuMemFree.
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn try_cuda() -> Option<CudaBackend> {
        CudaBackend::try_new().ok()
    }

    #[test]
    fn test_cuda_init() {
        match CudaBackend::try_new() {
            Ok(_) => eprintln!("CUDA backend initialized successfully"),
            Err(e) => eprintln!("CUDA not available: {}", e),
        }
    }

    #[test]
    fn test_upload_download_roundtrip() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::new(vec![2, 3], data.clone());
        let dt = cuda.upload(&t);
        let back = cuda.download(&dt);
        assert_eq!(back.as_f32(), &data);
        assert_eq!(back.shape(), &[2, 3]);
    }

    #[test]
    fn test_matmul_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        // A(4, 8) * B(8, 6) = C(4, 6)
        let a_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let b_data: Vec<f32> = (0..48).map(|i| (i as f32) * 0.05 - 1.0).collect();

        let a = Tensor::new(vec![4, 8], a_data);
        let b = Tensor::new(vec![8, 6], b_data);

        let c_cpu = cpu.download(&cpu.matmul(&cpu.upload(&a), &cpu.upload(&b)));
        let c_cuda = cuda.download(&cuda.matmul(&cuda.upload(&a), &cuda.upload(&b)));

        let diff = max_abs_diff(c_cpu.as_f32(), c_cuda.as_f32());
        eprintln!("matmul max diff: {diff}");
        assert!(diff < 1e-3, "matmul: max abs diff = {diff}");
    }

    #[test]
    fn test_add_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        let a_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..12).map(|i| i as f32 * -0.2 + 1.0).collect();
        let a = Tensor::new(vec![3, 4], a_data);
        let b = Tensor::new(vec![3, 4], b_data);

        let cpu_r = cpu.download(&cpu.add(&cpu.upload(&a), &cpu.upload(&b)));
        let cuda_r = cuda.download(&cuda.add(&cuda.upload(&a), &cuda.upload(&b)));

        let diff = max_abs_diff(cpu_r.as_f32(), cuda_r.as_f32());
        eprintln!("add max diff: {diff}");
        assert!(diff < 1e-5, "add: max abs diff = {diff}");
    }

    #[test]
    fn test_gelu_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        let data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 4.0).collect();
        let t = Tensor::new(vec![4, 4], data);

        let cpu_r = cpu.download(&cpu.gelu(&cpu.upload(&t)));
        let cuda_r = cuda.download(&cuda.gelu(&cuda.upload(&t)));

        let diff = max_abs_diff(cpu_r.as_f32(), cuda_r.as_f32());
        eprintln!("gelu max diff: {diff}");
        assert!(diff < 1e-3, "gelu: max abs diff = {diff}");
    }

    #[test]
    fn test_rms_norm_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let w_data = vec![1.0f32; 6];

        let t = Tensor::new(vec![4, 6], data);
        let w = Tensor::new(vec![6], w_data);

        let cpu_r = cpu.download(&cpu.rms_norm(&cpu.upload(&t), &cpu.upload(&w), 1e-5));
        let cuda_r = cuda.download(&cuda.rms_norm(&cuda.upload(&t), &cuda.upload(&w), 1e-5));

        let diff = max_abs_diff(cpu_r.as_f32(), cuda_r.as_f32());
        eprintln!("rms_norm max diff: {diff}");
        eprintln!("CPU first 6: {:?}", &cpu_r.as_f32()[..6]);
        eprintln!("CUDA first 6: {:?}", &cuda_r.as_f32()[..6]);
        assert!(diff < 1e-3, "rms_norm: max abs diff = {diff}");
    }

    #[test]
    fn test_silu_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        let data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 4.0).collect();
        let t = Tensor::new(vec![4, 4], data);

        let cpu_r = cpu.download(&cpu.silu(&cpu.upload(&t)));
        let cuda_r = cuda.download(&cuda.silu(&cuda.upload(&t)));

        let diff = max_abs_diff(cpu_r.as_f32(), cuda_r.as_f32());
        eprintln!("silu max diff: {diff}");
        assert!(diff < 1e-3, "silu: max abs diff = {diff}");
    }

    #[test]
    fn test_softmax_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        let data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 4.0).collect();
        let t = Tensor::new(vec![4, 4], data);

        let cpu_r = cpu.download(&cpu.softmax(&cpu.upload(&t)));
        let cuda_r = cuda.download(&cuda.softmax(&cuda.upload(&t)));

        let diff = max_abs_diff(cpu_r.as_f32(), cuda_r.as_f32());
        eprintln!("softmax max diff: {diff}");
        assert!(diff < 1e-3, "softmax: max abs diff = {diff}");
    }

    #[test]
    fn test_layer_norm_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let w_data = vec![1.0f32; 6];
        let b_data = vec![0.0f32; 6];

        let t = Tensor::new(vec![4, 6], data);
        let w = Tensor::new(vec![6], w_data);
        let b = Tensor::new(vec![6], b_data);

        let cpu_r = cpu.download(&cpu.layer_norm(
            &cpu.upload(&t),
            &cpu.upload(&w),
            &cpu.upload(&b),
            1e-5,
        ));
        let cuda_r = cuda.download(&cuda.layer_norm(
            &cuda.upload(&t),
            &cuda.upload(&w),
            &cuda.upload(&b),
            1e-5,
        ));

        let diff = max_abs_diff(cpu_r.as_f32(), cuda_r.as_f32());
        eprintln!("layer_norm max diff: {diff}");
        assert!(diff < 1e-3, "layer_norm: max abs diff = {diff}");
    }

    #[test]
    fn test_scale_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.3).collect();
        let t = Tensor::new(vec![3, 4], data);

        let cpu_r = cpu.download(&cpu.scale(&cpu.upload(&t), 0.5));
        let cuda_r = cuda.download(&cuda.scale(&cuda.upload(&t), 0.5));

        let diff = max_abs_diff(cpu_r.as_f32(), cuda_r.as_f32());
        eprintln!("scale max diff: {diff}");
        assert!(diff < 1e-5, "scale: max abs diff = {diff}");
    }

    #[test]
    fn test_tanh_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        let data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 4.0).collect();
        let t = Tensor::new(vec![4, 4], data);

        let cpu_r = cpu.download(&cpu.tanh(&cpu.upload(&t)));
        let cuda_r = cuda.download(&cuda.tanh(&cuda.upload(&t)));

        let diff = max_abs_diff(cpu_r.as_f32(), cuda_r.as_f32());
        eprintln!("tanh max diff: {diff}");
        assert!(diff < 1e-3, "tanh: max abs diff = {diff}");
    }

    #[test]
    fn test_swiglu_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        let gate_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.3 - 1.5).collect();
        let up_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.2 + 0.5).collect();
        let gate = Tensor::new(vec![3, 4], gate_data);
        let up = Tensor::new(vec![3, 4], up_data);

        let cpu_r = cpu.download(&cpu.swiglu(&cpu.upload(&gate), &cpu.upload(&up)));
        let cuda_r = cuda.download(&cuda.swiglu(&cuda.upload(&gate), &cuda.upload(&up)));

        let diff = max_abs_diff(cpu_r.as_f32(), cuda_r.as_f32());
        eprintln!("swiglu max diff: {diff}");
        assert!(diff < 1e-3, "swiglu: max abs diff = {diff}");
    }

    #[test]
    fn test_embedding_lookup_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        // Vocab of 10 words, hidden dim 4
        let table_data: Vec<f32> = (0..40).map(|i| i as f32 * 0.1).collect();
        let table = Tensor::new(vec![10, 4], table_data);
        let ids = vec![0u32, 3, 7, 1];

        let cpu_r = cpu.download(&cpu.embedding_lookup(&cpu.upload(&table), &ids));
        let cuda_r = cuda.download(&cuda.embedding_lookup(&cuda.upload(&table), &ids));

        let diff = max_abs_diff(cpu_r.as_f32(), cuda_r.as_f32());
        eprintln!("embedding_lookup max diff: {diff}");
        assert!(diff < 1e-5, "embedding_lookup: max abs diff = {diff}");
    }

    #[test]
    fn test_l2_normalize_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        let data: Vec<f32> = vec![3.0, 4.0, 0.0, 5.0];
        let t = Tensor::new(vec![4], data);

        let cpu_r = cpu.download(&cpu.l2_normalize(&cpu.upload(&t)));
        let cuda_r = cuda.download(&cuda.l2_normalize(&cuda.upload(&t)));

        let diff = max_abs_diff(cpu_r.as_f32(), cuda_r.as_f32());
        eprintln!("l2_normalize max diff: {diff}");
        assert!(diff < 1e-4, "l2_normalize: max abs diff = {diff}");
    }

    #[test]
    fn test_matmul_transpose_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        // A(4, 8) * B(6, 8)^T = C(4, 6)
        let a_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let b_data: Vec<f32> = (0..48).map(|i| (i as f32) * 0.05 - 1.0).collect();

        let a = Tensor::new(vec![4, 8], a_data);
        let b = Tensor::new(vec![6, 8], b_data);

        let c_cpu = cpu.download(&cpu.matmul_transpose(&cpu.upload(&a), &cpu.upload(&b)));
        let c_cuda = cuda.download(&cuda.matmul_transpose(&cuda.upload(&a), &cuda.upload(&b)));

        let diff = max_abs_diff(c_cpu.as_f32(), c_cuda.as_f32());
        eprintln!("matmul_transpose max diff: {diff}");
        assert!(diff < 1e-3, "matmul_transpose: max abs diff = {diff}");
    }

    #[test]
    fn test_causal_mask_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let t = Tensor::new(vec![4, 4], data);

        let cpu_r = cpu.download(&cpu.apply_causal_mask(&cpu.upload(&t), 4));
        let cuda_r = cuda.download(&cuda.apply_causal_mask(&cuda.upload(&t), 4));

        let diff = max_abs_diff(cpu_r.as_f32(), cuda_r.as_f32());
        eprintln!("causal_mask max diff: {diff}");
        assert!(diff < 1e-5, "causal_mask: max abs diff = {diff}");
    }

    #[test]
    fn test_geglu_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        let gate_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.3 - 1.5).collect();
        let up_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.2 + 0.5).collect();
        let gate = Tensor::new(vec![3, 4], gate_data);
        let up = Tensor::new(vec![3, 4], up_data);

        let cpu_r = cpu.download(&cpu.geglu(&cpu.upload(&gate), &cpu.upload(&up)));
        let cuda_r = cuda.download(&cuda.geglu(&cuda.upload(&gate), &cuda.upload(&up)));

        let diff = max_abs_diff(cpu_r.as_f32(), cuda_r.as_f32());
        eprintln!("geglu max diff: {diff}");
        assert!(diff < 1e-3, "geglu: max abs diff = {diff}");
    }

    #[test]
    fn test_add_bias_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        let data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1).collect();
        let bias: Vec<f32> = vec![1.0, -1.0, 0.5, 0.25];

        let t = Tensor::new(vec![3, 4], data);
        let b = Tensor::new(vec![4], bias);

        let cpu_r = cpu.download(&cpu.add_bias(&cpu.upload(&t), &cpu.upload(&b)));
        let cuda_r = cuda.download(&cuda.add_bias(&cuda.upload(&t), &cuda.upload(&b)));

        let diff = max_abs_diff(cpu_r.as_f32(), cuda_r.as_f32());
        eprintln!("add_bias max diff: {diff}");
        assert!(diff < 1e-5, "add_bias: max abs diff = {diff}");
    }

    #[test]
    fn test_mean_pool_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        // 3 tokens, 4 dims. Mask: [1, 1, 0] (only first 2 active)
        let data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1).collect();
        let t = Tensor::new(vec![3, 4], data);
        let mask = vec![1.0f32, 1.0, 0.0];

        let cpu_r = cpu.download(&cpu.mean_pool(&cpu.upload(&t), &mask));
        let cuda_r = cuda.download(&cuda.mean_pool(&cuda.upload(&t), &mask));

        let diff = max_abs_diff(cpu_r.as_f32(), cuda_r.as_f32());
        eprintln!("mean_pool max diff: {diff}");
        eprintln!("CPU: {:?}", cpu_r.as_f32());
        eprintln!("CUDA: {:?}", cuda_r.as_f32());
        assert!(diff < 1e-4, "mean_pool: max abs diff = {diff}");
    }

    #[test]
    fn test_rope_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        // seq_len=2, 2 heads, head_dim=4, rope_dim=4
        let q_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let k_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 + 0.5).collect();
        let q = Tensor::new(vec![2, 8], q_data);
        let k = Tensor::new(vec![2, 8], k_data);

        let (cpu_q, cpu_k) = cpu.rope(&cpu.upload(&q), &cpu.upload(&k), 0, 10000.0, 4, 4);
        let (cuda_q, cuda_k) = cuda.rope(&cuda.upload(&q), &cuda.upload(&k), 0, 10000.0, 4, 4);

        let cpu_q_d = cpu.download(&cpu_q);
        let cuda_q_d = cuda.download(&cuda_q);
        let cpu_k_d = cpu.download(&cpu_k);
        let cuda_k_d = cuda.download(&cuda_k);

        let q_diff = max_abs_diff(cpu_q_d.as_f32(), cuda_q_d.as_f32());
        let k_diff = max_abs_diff(cpu_k_d.as_f32(), cuda_k_d.as_f32());
        eprintln!("rope Q max diff: {q_diff}");
        eprintln!("rope K max diff: {k_diff}");
        assert!(q_diff < 1e-3, "rope Q: max abs diff = {q_diff}");
        assert!(k_diff < 1e-3, "rope K: max abs diff = {k_diff}");
    }

    #[test]
    fn test_rope_neox_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        // seq_len=2, 2 heads, head_dim=4, rope_dim=4
        let q_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let k_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 + 0.5).collect();
        let q = Tensor::new(vec![2, 8], q_data);
        let k = Tensor::new(vec![2, 8], k_data);

        let (cpu_q, cpu_k) = cpu.rope_neox(&cpu.upload(&q), &cpu.upload(&k), 3, 10000.0, 4, 4);
        let (cuda_q, cuda_k) = cuda.rope_neox(&cuda.upload(&q), &cuda.upload(&k), 3, 10000.0, 4, 4);

        let cpu_q_d = cpu.download(&cpu_q);
        let cuda_q_d = cuda.download(&cuda_q);
        let cpu_k_d = cpu.download(&cpu_k);
        let cuda_k_d = cuda.download(&cuda_k);

        let q_diff = max_abs_diff(cpu_q_d.as_f32(), cuda_q_d.as_f32());
        let k_diff = max_abs_diff(cpu_k_d.as_f32(), cuda_k_d.as_f32());
        eprintln!("rope_neox Q max diff: {q_diff}");
        eprintln!("rope_neox K max diff: {k_diff}");
        assert!(q_diff < 1e-3, "rope_neox Q: max abs diff = {q_diff}");
        assert!(k_diff < 1e-3, "rope_neox K: max abs diff = {k_diff}");
    }

    #[test]
    fn test_quantized_matmul_q8_0_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        // Create a Q8_0 weight tensor: 2 rows x 32 cols
        let scale = half::f16::from_f32(0.1);
        let mut raw = Vec::new();
        for _ in 0..2 {
            raw.extend_from_slice(&scale.to_bits().to_le_bytes());
            for i in 0..32u8 {
                raw.push(i);
            }
        }
        let weights = Tensor::from_quantized(vec![2, 32], TensorDtype::Q8_0, raw);

        // Input: 1x32 f32
        let input_data: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
        let input = Tensor::new(vec![1, 32], input_data);

        let cpu_out = cpu.download(&cpu.quantized_matmul(&cpu.upload(&weights), &cpu.upload(&input)));
        let cuda_out = cuda.download(&cuda.quantized_matmul(&cuda.upload(&weights), &cuda.upload(&input)));

        assert_eq!(cpu_out.shape(), &[1, 2]);
        assert_eq!(cuda_out.shape(), &[1, 2]);
        let diff = max_abs_diff(cpu_out.as_f32(), cuda_out.as_f32());
        eprintln!("quantized_matmul_q8_0 max diff: {diff}");
        assert!(diff < 1e-2, "quantized_matmul_q8_0: max abs diff = {diff}");
    }

    #[test]
    fn test_mul_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![2.0, 3.0, 4.0, 5.0];
        let a = Tensor::new(vec![4], a_data);
        let b = Tensor::new(vec![4], b_data);

        let cpu_r = cpu.download(&cpu.mul(&cpu.upload(&a), &cpu.upload(&b)));
        let cuda_r = cuda.download(&cuda.mul(&cuda.upload(&a), &cuda.upload(&b)));

        let diff = max_abs_diff(cpu_r.as_f32(), cuda_r.as_f32());
        assert!(diff < 1e-6, "mul: max abs diff = {diff}");
    }

    #[test]
    fn test_quantized_matmul_q4_0_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        // Create Q4_0 weights: 2 rows x 32 cols
        let scale = half::f16::from_f32(0.5);
        let mut raw = Vec::new();
        for _ in 0..2 {
            raw.extend_from_slice(&scale.to_bits().to_le_bytes());
            // All nibbles = 12 -> after -8 = 4, dequant = 0.5 * 4 = 2.0
            for _ in 0..16 {
                raw.push(0xCC);
            }
        }
        let weights = Tensor::from_quantized(vec![2, 32], TensorDtype::Q4_0, raw);
        let input = Tensor::new(vec![1, 32], vec![1.0f32; 32]);

        let cpu_out = cpu.download(&cpu.quantized_matmul(&cpu.upload(&weights), &cpu.upload(&input)));
        let cuda_out = cuda.download(&cuda.quantized_matmul(&cuda.upload(&weights), &cuda.upload(&input)));

        assert_eq!(cpu_out.shape(), &[1, 2]);
        assert_eq!(cuda_out.shape(), &[1, 2]);
        let diff = max_abs_diff(cpu_out.as_f32(), cuda_out.as_f32());
        eprintln!("quantized_matmul_q4_0 max diff: {diff}");
        assert!(diff < 1.0, "quantized_matmul_q4_0: max abs diff = {diff}");
    }

    #[test]
    fn test_upload_download_roundtrip_f32() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };

        let data = vec![1.0f32, -2.5, 3.14, 0.0, 100.0];
        let t = Tensor::new(vec![5], data.clone());
        let dt = cuda.upload(&t);
        let result = cuda.download(&dt);

        assert_eq!(result.shape(), &[5]);
        let diff = max_abs_diff(result.as_f32(), &data);
        assert!(diff < 1e-6, "roundtrip_f32: max abs diff = {diff}");
    }

    #[test]
    fn test_upload_download_roundtrip_q8_0() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };

        // Create Q8_0 tensor: 1 row x 32 cols
        let scale = half::f16::from_f32(0.1);
        let mut raw = Vec::new();
        raw.extend_from_slice(&scale.to_bits().to_le_bytes());
        for i in 0..32i8 {
            raw.push(i as u8);
        }
        let t = Tensor::from_quantized(vec![1, 32], TensorDtype::Q8_0, raw);

        // Compare CPU dequant vs upload+download+dequant roundtrip
        let cpu_f32 = t.to_f32();
        let dt = cuda.upload(&t);
        let result = cuda.download(&dt);
        let result_f32 = result.to_f32();

        assert_eq!(cpu_f32.shape(), result_f32.shape());
        let diff = max_abs_diff(result_f32.as_f32(), cpu_f32.as_f32());
        assert!(diff < 1e-6, "roundtrip_q8_0: max abs diff = {diff}");
    }

    #[test]
    fn test_rope_partial_rotation_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        // seq_len=1, 2 heads, head_dim=8, rope_dim=4 (only first 4 dims rotated)
        let q_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let k_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 + 0.5).collect();
        let q = Tensor::new(vec![1, 16], q_data);
        let k = Tensor::new(vec![1, 16], k_data);

        let (cpu_q, cpu_k) = cpu.rope(&cpu.upload(&q), &cpu.upload(&k), 1, 10000.0, 8, 4);
        let (cuda_q, cuda_k) = cuda.rope(&cuda.upload(&q), &cuda.upload(&k), 1, 10000.0, 8, 4);

        let cpu_q_d = cpu.download(&cpu_q);
        let cuda_q_d = cuda.download(&cuda_q);
        let cpu_k_d = cpu.download(&cpu_k);
        let cuda_k_d = cuda.download(&cuda_k);

        let q_diff = max_abs_diff(cpu_q_d.as_f32(), cuda_q_d.as_f32());
        let k_diff = max_abs_diff(cpu_k_d.as_f32(), cuda_k_d.as_f32());
        eprintln!("rope partial Q max diff: {q_diff}");
        eprintln!("rope partial K max diff: {k_diff}");
        assert!(q_diff < 1e-3, "rope partial Q: max abs diff = {q_diff}");
        assert!(k_diff < 1e-3, "rope partial K: max abs diff = {k_diff}");
    }

    #[test]
    fn test_rope_neox_partial_rotation_vs_cpu() {
        let cuda = match try_cuda() {
            Some(b) => b,
            None => {
                eprintln!("CUDA not available, skipping");
                return;
            }
        };
        let cpu = CpuBackend::new();

        // seq_len=1, 2 heads, head_dim=8, rope_dim=4
        let q_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let k_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 + 0.5).collect();
        let q = Tensor::new(vec![1, 16], q_data);
        let k = Tensor::new(vec![1, 16], k_data);

        let (cpu_q, cpu_k) = cpu.rope_neox(&cpu.upload(&q), &cpu.upload(&k), 1, 10000.0, 8, 4);
        let (cuda_q, cuda_k) = cuda.rope_neox(&cuda.upload(&q), &cuda.upload(&k), 1, 10000.0, 8, 4);

        let cpu_q_d = cpu.download(&cpu_q);
        let cuda_q_d = cuda.download(&cuda_q);
        let cpu_k_d = cpu.download(&cpu_k);
        let cuda_k_d = cuda.download(&cuda_k);

        let q_diff = max_abs_diff(cpu_q_d.as_f32(), cuda_q_d.as_f32());
        let k_diff = max_abs_diff(cpu_k_d.as_f32(), cuda_k_d.as_f32());
        eprintln!("rope_neox partial Q max diff: {q_diff}");
        eprintln!("rope_neox partial K max diff: {k_diff}");
        assert!(q_diff < 1e-3, "rope_neox partial Q: max abs diff = {q_diff}");
        assert!(k_diff < 1e-3, "rope_neox partial K: max abs diff = {k_diff}");
    }
}
