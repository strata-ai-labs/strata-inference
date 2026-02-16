//! Metal compute backend for macOS GPU acceleration.
//!
//! Provides `MetalBackend`, which implements `ComputeBackend` by dispatching
//! MSL compute kernels through the Metal framework via raw Objective-C FFI.
//! No external Objective-C or Metal crate dependencies are required.
//!
//! Key differences from strata-core's Metal backend:
//! - F32 only — no f16 kernel variants, no f32↔f16 conversion
//! - New kernels: quantized matmul (Q8_0/Q4_0), RMS norm, SiLU, SwiGLU,
//!   GeGLU, RoPE (normal + NeoX), causal mask, mul, tanh, L2 normalize,
//!   embedding lookup
//! - Uses `DeviceTensor::from_gpu()` with `MetalBuffer` inner type
//! - No batched ops, no head transpose/untranspose, no slice/scatter

use std::sync::Mutex;

use tracing::{debug, trace};

use crate::error::InferenceError;
use crate::tensor::{Tensor, TensorDtype, TensorStorage};

use super::{ComputeBackend, DeviceTensor};
use ffi::*;

pub mod ffi;
pub mod kernels;

/// `MTLBarrierScopeBuffers` — ensures all buffer writes from prior dispatches
/// are visible to subsequent dispatches within the same encoder.
const MTL_BARRIER_SCOPE_BUFFERS: NSUInteger = 1;

// ---------------------------------------------------------------------------
// MetalBuffer — a reference-counted MTLBuffer wrapper
// ---------------------------------------------------------------------------

/// A buffer allocated on the Metal device with `StorageModeShared`.
struct MetalBuffer {
    /// Raw `MTLBuffer` Objective-C object pointer.
    buffer: Id,
    /// Size in bytes — retained for debugging / future introspection.
    #[allow(dead_code)]
    len: usize,
}

// Metal shared-mode buffers can be read/written from any thread.
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        if self.buffer != NIL {
            unsafe {
                msg_send_void(self.buffer, sel_registerName(b"release\0".as_ptr() as _));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MetalBackend
// ---------------------------------------------------------------------------

/// Metal compute backend for macOS GPU acceleration.
///
/// Holds the Metal device, command queue, selector cache, and pre-compiled
/// pipeline state objects (PSOs) for every kernel in the MSL source.
///
/// Uses deferred command buffer batching: multiple GPU dispatches are encoded
/// into a single command buffer separated by memory barriers. The buffer is
/// only committed (`flush()`) when the CPU needs to read results (download,
/// mean_pool). This reduces `waitUntilCompleted` stalls to ~1-2 per inference.
pub struct MetalBackend {
    device: Id,
    command_queue: Id,
    sels: Selectors,
    // Pipeline state objects — f32 kernels only (22 total)
    pso_gemm: Id,
    pso_gemm_transpose: Id,
    pso_gelu: Id,
    pso_add_tensor: Id,
    pso_add_bias: Id,
    pso_scale: Id,
    pso_layer_norm: Id,
    pso_softmax_rows: Id,
    pso_mean_pool: Id,
    pso_quantized_matmul_q8_0: Id,
    pso_quantized_matmul_q4_0: Id,
    pso_rms_norm: Id,
    pso_silu: Id,
    pso_swiglu: Id,
    pso_geglu: Id,
    pso_rope_norm: Id,
    pso_rope_neox: Id,
    pso_causal_mask: Id,
    pso_mul_elementwise: Id,
    pso_tanh_kernel: Id,
    pso_l2_normalize: Id,
    pso_embedding_lookup: Id,
    /// Active command buffer and compute encoder for deferred dispatch.
    /// `None` means no open command buffer; dispatches create one lazily.
    active_cmd: Mutex<Option<(Id, Id)>>,
}

// We flush (waitUntilCompleted) before any CPU read, so the backend can
// safely be shared across threads.
unsafe impl Send for MetalBackend {}
unsafe impl Sync for MetalBackend {}

impl Drop for MetalBackend {
    fn drop(&mut self) {
        unsafe {
            // Flush any pending GPU work.
            self.flush();

            let rel = self.sels.release;
            // Release all PSOs
            for pso in [
                self.pso_gemm,
                self.pso_gemm_transpose,
                self.pso_gelu,
                self.pso_add_tensor,
                self.pso_add_bias,
                self.pso_scale,
                self.pso_layer_norm,
                self.pso_softmax_rows,
                self.pso_mean_pool,
                self.pso_quantized_matmul_q8_0,
                self.pso_quantized_matmul_q4_0,
                self.pso_rms_norm,
                self.pso_silu,
                self.pso_swiglu,
                self.pso_geglu,
                self.pso_rope_norm,
                self.pso_rope_neox,
                self.pso_causal_mask,
                self.pso_mul_elementwise,
                self.pso_tanh_kernel,
                self.pso_l2_normalize,
                self.pso_embedding_lookup,
            ] {
                if pso != NIL {
                    msg_send_void(pso, rel);
                }
            }
            if self.command_queue != NIL {
                msg_send_void(self.command_queue, rel);
            }
            if self.device != NIL {
                msg_send_void(self.device, rel);
            }
        }
    }
}

impl MetalBackend {
    /// Try to create a Metal compute backend.
    ///
    /// Returns `Err` if no Metal device is available or if the MSL kernels
    /// fail to compile.
    pub fn try_new() -> Result<Self, InferenceError> {
        unsafe {
            // 1. Get the default Metal device.
            let device = MTLCreateSystemDefaultDevice();
            if device == NIL {
                return Err(InferenceError::Backend(
                    "No Metal device available".into(),
                ));
            }
            // Retain the device so our Drop can release it.
            msg_send_void(device, sel_registerName(b"retain\0".as_ptr() as _));

            // 2. Pre-register all selectors.
            let sels = Selectors::new();

            // 3. Create command queue.
            let command_queue = msg_send_id(device, sels.new_command_queue);
            if command_queue == NIL {
                msg_send_void(device, sels.release);
                return Err(InferenceError::Backend(
                    "Failed to create Metal command queue".into(),
                ));
            }

            // 4. Compile the MSL library from source.
            let source = ns_string(kernels::MSL_SOURCE);
            let mut error: Id = NIL;
            let library = msg_send_id_id_id_id(
                device,
                sels.new_library_with_source,
                source,
                NIL, // default compile options
                &mut error as *mut Id as Id,
            );
            if library == NIL {
                let desc = obj_description(error);
                msg_send_void(command_queue, sels.release);
                msg_send_void(device, sels.release);
                return Err(InferenceError::Backend(format!(
                    "Metal MSL compile error: {}",
                    desc
                )));
            }

            // 5. Create pipeline state objects for each kernel.
            let kernel_names = [
                "gemm",
                "gemm_transpose",
                "gelu",
                "add_tensor",
                "add_bias",
                "scale_kernel",
                "layer_norm",
                "softmax_rows",
                "mean_pool",
                "quantized_matmul_q8_0",
                "quantized_matmul_q4_0",
                "rms_norm",
                "silu",
                "swiglu",
                "geglu",
                "rope_norm",
                "rope_neox",
                "causal_mask",
                "mul_elementwise",
                "tanh_kernel",
                "l2_normalize",
                "embedding_lookup",
            ];

            let mut psos = [NIL; 22];
            for (i, name) in kernel_names.iter().enumerate() {
                let ns_name = ns_string(name);
                let func =
                    msg_send_id_id(library, sels.new_function_with_name, ns_name);
                if func == NIL {
                    // Cleanup already-created PSOs.
                    for p in &psos[..i] {
                        if *p != NIL {
                            msg_send_void(*p, sels.release);
                        }
                    }
                    msg_send_void(library, sels.release);
                    msg_send_void(command_queue, sels.release);
                    msg_send_void(device, sels.release);
                    return Err(InferenceError::Backend(format!(
                        "Metal kernel function '{}' not found",
                        name
                    )));
                }

                let mut pso_error: Id = NIL;
                let pso = msg_send_id_id_err(
                    device,
                    sels.new_compute_pipeline,
                    func,
                    &mut pso_error,
                );
                msg_send_void(func, sels.release);

                if pso == NIL {
                    let desc = obj_description(pso_error);
                    for p in &psos[..i] {
                        if *p != NIL {
                            msg_send_void(*p, sels.release);
                        }
                    }
                    msg_send_void(library, sels.release);
                    msg_send_void(command_queue, sels.release);
                    msg_send_void(device, sels.release);
                    return Err(InferenceError::Backend(format!(
                        "Metal PSO creation failed for '{}': {}",
                        name, desc
                    )));
                }
                psos[i] = pso;
            }

            msg_send_void(library, sels.release);

            debug!("MetalBackend initialized with 22 PSOs");

            Ok(Self {
                device,
                command_queue,
                sels,
                pso_gemm: psos[0],
                pso_gemm_transpose: psos[1],
                pso_gelu: psos[2],
                pso_add_tensor: psos[3],
                pso_add_bias: psos[4],
                pso_scale: psos[5],
                pso_layer_norm: psos[6],
                pso_softmax_rows: psos[7],
                pso_mean_pool: psos[8],
                pso_quantized_matmul_q8_0: psos[9],
                pso_quantized_matmul_q4_0: psos[10],
                pso_rms_norm: psos[11],
                pso_silu: psos[12],
                pso_swiglu: psos[13],
                pso_geglu: psos[14],
                pso_rope_norm: psos[15],
                pso_rope_neox: psos[16],
                pso_causal_mask: psos[17],
                pso_mul_elementwise: psos[18],
                pso_tanh_kernel: psos[19],
                pso_l2_normalize: psos[20],
                pso_embedding_lookup: psos[21],
                active_cmd: Mutex::new(None),
            })
        }
    }

    // -------------------------------------------------------------------
    // Buffer helpers
    // -------------------------------------------------------------------

    /// Create a Metal buffer from a byte slice.
    unsafe fn create_buffer(&self, data: &[u8]) -> MetalBuffer {
        let buf = msg_send_new_buffer(
            self.device,
            self.sels.new_buffer_with_bytes,
            data.as_ptr(),
            data.len(),
            MTL_RESOURCE_STORAGE_MODE_SHARED,
        );
        MetalBuffer {
            buffer: buf,
            len: data.len(),
        }
    }

    /// Create a Metal buffer from a `&[f32]` slice.
    unsafe fn create_buffer_f32(&self, data: &[f32]) -> MetalBuffer {
        let bytes = std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        );
        self.create_buffer(bytes)
    }

    /// Create a Metal buffer from a `&[u32]` slice.
    unsafe fn create_buffer_u32(&self, data: &[u32]) -> MetalBuffer {
        let bytes = std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<u32>(),
        );
        self.create_buffer(bytes)
    }

    /// Create an uninitialised Metal buffer of `len` bytes.
    unsafe fn create_buffer_empty(&self, len: usize) -> MetalBuffer {
        let buf = msg_send_new_buffer_length(
            self.device,
            self.sels.new_buffer_with_length,
            len,
            MTL_RESOURCE_STORAGE_MODE_SHARED,
        );
        MetalBuffer {
            buffer: buf,
            len,
        }
    }

    /// Read back the contents of a Metal buffer as a `Vec<f32>`.
    unsafe fn read_buffer_f32(&self, buffer: Id, count: usize) -> Vec<f32> {
        let ptr = msg_send_ptr(buffer, self.sels.contents) as *const f32;
        let slice = std::slice::from_raw_parts(ptr, count);
        slice.to_vec()
    }

    // -------------------------------------------------------------------
    // Extract the raw MTLBuffer Id from a DeviceTensor
    // -------------------------------------------------------------------

    /// Get the `MetalBuffer` reference from a GPU-resident `DeviceTensor`.
    ///
    /// Panics if the tensor is CPU-resident (should have been uploaded first).
    fn metal_buffer(dt: &DeviceTensor) -> &MetalBuffer {
        dt.gpu_inner::<MetalBuffer>()
            .expect("MetalBackend: expected GPU-resident MetalBuffer in DeviceTensor")
    }

    /// Get the raw `MTLBuffer` pointer from a `DeviceTensor`.
    fn buf_id(dt: &DeviceTensor) -> Id {
        Self::metal_buffer(dt).buffer
    }

    /// Wrap a MetalBuffer into a DeviceTensor with the given shape and dtype.
    fn wrap(buffer: MetalBuffer, shape: Vec<usize>, dtype: TensorDtype) -> DeviceTensor {
        DeviceTensor::from_gpu(shape, dtype, Box::new(buffer))
    }

    // -------------------------------------------------------------------
    // Deferred command buffer batching
    // -------------------------------------------------------------------

    /// Get or create the active command buffer and compute encoder, then set
    /// the pipeline state. Inserts a memory barrier before the dispatch so
    /// prior buffer writes are visible (cheap GPU-side fence, no CPU stall).
    ///
    /// Returns the encoder to bind buffers/bytes and dispatch on.
    unsafe fn ensure_encoder(&self, pso: Id) -> Id {
        let mut guard = self.active_cmd.lock().unwrap();
        let (_, enc) = guard.get_or_insert_with(|| {
            let cmd = msg_send_id(self.command_queue, self.sels.command_buffer);
            assert!(!cmd.is_null(), "command_buffer returned nil");
            let enc = msg_send_id(cmd, self.sels.compute_command_encoder);
            assert!(!enc.is_null(), "compute_command_encoder returned nil");
            (cmd, enc)
        });
        let enc = *enc;
        // Memory barrier: ensure all buffer writes from prior dispatches are
        // visible before this dispatch reads them.
        msg_send_void_nsuint(
            enc,
            self.sels.memory_barrier_with_scope,
            MTL_BARRIER_SCOPE_BUFFERS,
        );
        // Set the pipeline for this dispatch.
        msg_send_void_id(enc, self.sels.set_compute_pipeline, pso);
        enc
    }

    /// Commit the active command buffer, wait for completion, and clear.
    /// This is the only place `waitUntilCompleted` is called.
    /// No-op if no command buffer is open.
    unsafe fn flush(&self) {
        let mut guard = self.active_cmd.lock().unwrap();
        if let Some((cmd, enc)) = guard.take() {
            msg_send_void(enc, self.sels.end_encoding);
            msg_send_void(cmd, self.sels.commit);
            msg_send_void(cmd, self.sels.wait_until_completed);
        }
    }

    // -------------------------------------------------------------------
    // Bind helpers
    // -------------------------------------------------------------------

    /// Bind a MTLBuffer at `index`.
    unsafe fn set_buffer(&self, enc: Id, buf: Id, index: usize) {
        msg_send_set_buffer(enc, self.sels.set_buffer, buf, 0, index);
    }

    /// Bind small constant data (e.g. a single u32 or f32) at `index`.
    unsafe fn set_bytes(&self, enc: Id, data: &[u8], index: usize) {
        msg_send_set_bytes(
            enc,
            self.sels.set_bytes,
            data.as_ptr(),
            data.len(),
            index,
        );
    }

    /// Bind a `u32` parameter at `index`.
    unsafe fn set_u32(&self, enc: Id, val: u32, index: usize) {
        self.set_bytes(enc, &val.to_ne_bytes(), index);
    }

    /// Bind a `f32` parameter at `index`.
    unsafe fn set_f32(&self, enc: Id, val: f32, index: usize) {
        self.set_bytes(enc, &val.to_ne_bytes(), index);
    }

    // -------------------------------------------------------------------
    // Dispatch helpers (deferred — no waitUntilCompleted)
    // -------------------------------------------------------------------

    /// Dispatch a 1D grid: `ceil(n / threads)` threadgroups, each `threads` wide.
    unsafe fn dispatch_1d(&self, enc: Id, n: usize) {
        let threads = 256usize;
        let groups = (n + threads - 1) / threads;
        msg_send_dispatch(
            enc,
            self.sels.dispatch_threadgroups,
            groups, 1, 1, // threadgroups
            threads, 1, 1, // threads per threadgroup
        );
    }

    /// Dispatch a 2D grid: ceil(width/tx) x ceil(height/ty) threadgroups.
    unsafe fn dispatch_2d(
        &self,
        enc: Id,
        width: usize,
        height: usize,
        tx: usize,
        ty: usize,
    ) {
        let gx = (width + tx - 1) / tx;
        let gy = (height + ty - 1) / ty;
        msg_send_dispatch(
            enc,
            self.sels.dispatch_threadgroups,
            gx, gy, 1, // threadgroups
            tx, ty, 1, // threads per threadgroup
        );
    }

    /// Dispatch a 3D grid.
    unsafe fn dispatch_3d(
        &self,
        enc: Id,
        gx: usize,
        gy: usize,
        gz: usize,
        tx: usize,
        ty: usize,
        tz: usize,
    ) {
        msg_send_dispatch(
            enc,
            self.sels.dispatch_threadgroups,
            gx, gy, gz,
            tx, ty, tz,
        );
    }

    /// Dispatch with one threadgroup per row, `threads_per_group` threads each.
    /// Used by reduction kernels (layer_norm, softmax_rows, rms_norm).
    unsafe fn dispatch_rows(
        &self,
        enc: Id,
        num_rows: usize,
        threads_per_group: usize,
    ) {
        msg_send_dispatch(
            enc,
            self.sels.dispatch_threadgroups,
            num_rows, 1, 1,          // threadgroups
            threads_per_group, 1, 1, // threads per threadgroup
        );
    }

    // -------------------------------------------------------------------
    // Rope dispatch helper — shared by rope() and rope_neox()
    // -------------------------------------------------------------------

    /// Dispatch a RoPE kernel (rope_norm or rope_neox) on a single tensor.
    ///
    /// Input layout: [seq_len, n_heads * head_dim] (2D flat).
    /// The kernel interprets this as [seq_len, n_heads, head_dim] (3D).
    unsafe fn dispatch_rope(
        &self,
        pso: Id,
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

        let count = seq_len * total_cols;
        let out_bytes = count * std::mem::size_of::<f32>();
        let out_buf = self.create_buffer_empty(out_bytes);

        let enc = self.ensure_encoder(pso);

        self.set_buffer(enc, Self::buf_id(input_dt), 0);
        self.set_buffer(enc, out_buf.buffer, 1);
        self.set_u32(enc, pos_offset as u32, 2);
        self.set_f32(enc, freq_base, 3);
        self.set_u32(enc, head_dim as u32, 4);
        self.set_u32(enc, rope_dim as u32, 5);
        self.set_u32(enc, n_heads as u32, 6);
        self.set_u32(enc, seq_len as u32, 7);

        // 3D grid: (rope_dim/2, n_heads, seq_len)
        let half_rope = rope_dim / 2;
        let gx = (half_rope + 15) / 16;
        let gy = (n_heads + 15) / 16;
        let gz = seq_len;

        // If rope_dim < head_dim, copy input to output first so non-rotated
        // dims are preserved. The kernel reads from input and only writes
        // the rotated pairs to output, so the copy is safe.
        if rope_dim < head_dim {
            let src_ptr = msg_send_ptr(Self::buf_id(input_dt), self.sels.contents);
            let dst_ptr = msg_send_ptr(out_buf.buffer, self.sels.contents);
            std::ptr::copy_nonoverlapping(
                src_ptr as *const u8,
                dst_ptr as *mut u8,
                out_bytes,
            );
        }

        self.dispatch_3d(enc, gx, gy, gz, 16, 16, 1);

        Self::wrap(out_buf, vec![seq_len, total_cols], TensorDtype::F32)
    }

    /// Helper to get 2D shape (rows, cols) from a DeviceTensor.
    /// Flattens all dimensions > 2 into rows.
    fn shape_2d(dt: &DeviceTensor) -> (usize, usize) {
        let shape = dt.shape();
        match shape.len() {
            1 => (1, shape[0]),
            2 => (shape[0], shape[1]),
            _ => {
                // Flatten all but last dim into rows.
                let cols = *shape.last().unwrap();
                let rows = shape.iter().take(shape.len() - 1).product::<usize>();
                (rows, cols)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ComputeBackend implementation
// ---------------------------------------------------------------------------

impl ComputeBackend for MetalBackend {
    fn upload(&self, tensor: &Tensor) -> DeviceTensor {
        let shape = tensor.shape().to_vec();
        let dtype = tensor.dtype();

        let buf = match tensor.storage() {
            TensorStorage::F32(data) => {
                trace!(shape = ?shape, dtype = ?dtype, "Metal upload F32");
                unsafe { self.create_buffer_f32(data) }
            }
            TensorStorage::F16(data) => {
                // Upload raw f16 bits as bytes
                trace!(shape = ?shape, dtype = ?dtype, "Metal upload F16");
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        data.len() * 2,
                    )
                };
                unsafe { self.create_buffer(bytes) }
            }
            TensorStorage::Quantized(data) => {
                // Upload raw quantized block data as bytes
                trace!(shape = ?shape, dtype = ?dtype, "Metal upload Quantized");
                unsafe { self.create_buffer(data) }
            }
        };

        Self::wrap(buf, shape, dtype)
    }

    fn download(&self, dt: &DeviceTensor) -> Tensor {
        // If the tensor is CPU-resident, just clone it.
        if let Some(tensor) = dt.try_as_tensor() {
            trace!(shape = ?dt.shape(), "Metal download (CPU tensor, clone)");
            return tensor.clone();
        }

        // GPU tensor — flush pending work and read back.
        unsafe { self.flush() };

        let shape = dt.shape().to_vec();
        let dtype = dt.dtype();
        let mb = Self::metal_buffer(dt);

        match dtype {
            TensorDtype::F32 => {
                let n_elements: usize = shape.iter().product();
                let data = unsafe { self.read_buffer_f32(mb.buffer, n_elements) };
                trace!(shape = ?shape, "Metal download F32");
                Tensor::new(shape, data)
            }
            TensorDtype::F16 => {
                let n_elements: usize = shape.iter().product();
                let ptr =
                    unsafe { msg_send_ptr(mb.buffer, self.sels.contents) } as *const u16;
                let data =
                    unsafe { std::slice::from_raw_parts(ptr, n_elements) }.to_vec();
                trace!(shape = ?shape, "Metal download F16");
                Tensor::from_f16(shape, data)
            }
            TensorDtype::Q8_0 | TensorDtype::Q4_0 => {
                let n_elements: usize = shape.iter().product();
                let block_size = dtype.block_size();
                let block_byte_size = dtype.block_byte_size();
                let n_blocks = (n_elements + block_size - 1) / block_size;
                let total_bytes = n_blocks * block_byte_size;
                let ptr =
                    unsafe { msg_send_ptr(mb.buffer, self.sels.contents) } as *const u8;
                let data =
                    unsafe { std::slice::from_raw_parts(ptr, total_bytes) }.to_vec();
                trace!(shape = ?shape, dtype = ?dtype, "Metal download quantized");
                Tensor::from_quantized(shape, dtype, data)
            }
        }
    }

    fn matmul(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor {
        let (m, k) = Self::shape_2d(a);
        let (k2, n) = Self::shape_2d(b);
        assert_eq!(k, k2, "matmul dimension mismatch: a cols {} != b rows {}", k, k2);

        let out_count = m * n;
        let out_bytes = out_count * std::mem::size_of::<f32>();

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_gemm);

            self.set_buffer(enc, Self::buf_id(a), 0);
            self.set_buffer(enc, Self::buf_id(b), 1);
            self.set_buffer(enc, out_buf.buffer, 2);
            self.set_u32(enc, m as u32, 3);
            self.set_u32(enc, k as u32, 4);
            self.set_u32(enc, n as u32, 5);

            // simdgroup_matrix: 32x32 tiles, 128 threads (4 simdgroups)
            let gx = (n + 31) / 32;
            let gy = (m + 31) / 32;
            msg_send_dispatch(
                enc,
                self.sels.dispatch_threadgroups,
                gx, gy, 1,
                128, 1, 1,
            );

            Self::wrap(out_buf, vec![m, n], TensorDtype::F32)
        }
    }

    fn matmul_transpose(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor {
        let (m, k) = Self::shape_2d(a);
        let (n, k2) = Self::shape_2d(b); // B is (N, K), transposed
        assert_eq!(
            k, k2,
            "matmul_transpose dimension mismatch: a cols {} != b cols {}",
            k, k2
        );

        let out_count = m * n;
        let out_bytes = out_count * std::mem::size_of::<f32>();

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_gemm_transpose);

            self.set_buffer(enc, Self::buf_id(a), 0);
            self.set_buffer(enc, Self::buf_id(b), 1);
            self.set_buffer(enc, out_buf.buffer, 2);
            self.set_u32(enc, m as u32, 3);
            self.set_u32(enc, k as u32, 4);
            self.set_u32(enc, n as u32, 5);

            // simdgroup_matrix: 32x32 tiles, 128 threads (4 simdgroups)
            let gx = (n + 31) / 32;
            let gy = (m + 31) / 32;
            msg_send_dispatch(
                enc,
                self.sels.dispatch_threadgroups,
                gx, gy, 1,
                128, 1, 1,
            );

            Self::wrap(out_buf, vec![m, n], TensorDtype::F32)
        }
    }

    fn quantized_matmul(&self, weights: &DeviceTensor, input: &DeviceTensor) -> DeviceTensor {
        // Weights: [N, K] in Q8_0 or Q4_0, Input: [M, K] in F32 → Output: [M, N] in F32
        // The GPU kernel handles one input row at a time (matrix-vector).
        // For M > 1, we dispatch per row.
        let input_shape = input.shape();
        let weight_shape = weights.shape();
        let dtype = weights.dtype();

        let m = if input_shape.len() == 1 { 1 } else { input_shape[0] };
        let k = *input_shape.last().unwrap();
        let n = weight_shape[0];

        assert_eq!(
            k, weight_shape[1],
            "quantized_matmul: input cols ({}) must match weight cols ({})",
            k, weight_shape[1]
        );

        let out_count = m * n;
        let out_bytes = out_count * std::mem::size_of::<f32>();

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let weight_buf_id = Self::buf_id(weights);

            // For each row of input, dispatch the quantized kernel.
            // The kernel computes output[row_offset + i] = dot(dequant(weights[i]), input_row)
            // for all i in 0..N.
            //
            // Since the kernel is a matrix-vector multiply (one input row at a time),
            // we need to handle M > 1 by dispatching M times with different input offsets.
            // However, the kernel expects the full input pointer, so we need to create
            // temporary single-row buffers or adjust the kernel to accept an offset.
            //
            // For simplicity, if M == 1 we dispatch directly. For M > 1, we use a loop.
            // Each iteration dispatches the kernel for one row of input.

            for row in 0..m {
                // Create a temporary buffer for this input row
                let input_row_data = if let Some(tensor) = input.try_as_tensor() {
                    let f32_data = tensor.as_f32();
                    &f32_data[row * k..(row + 1) * k]
                } else {
                    // GPU tensor — read back the full data to extract a row.
                    // This is suboptimal for multi-row GPU inputs but keeps the kernel simple.
                    // In practice, quantized_matmul is called with M=1 (single token) or
                    // with CPU-uploaded input.
                    self.flush();
                    let mb = Self::metal_buffer(input);
                    let ptr = msg_send_ptr(mb.buffer, self.sels.contents) as *const f32;
                    std::slice::from_raw_parts(ptr.add(row * k), k)
                };

                let input_row_buf = self.create_buffer_f32(input_row_data);
                let row_out_offset = row * n;

                match dtype {
                    TensorDtype::Q8_0 => {
                        let enc = self.ensure_encoder(self.pso_quantized_matmul_q8_0);

                        self.set_buffer(enc, weight_buf_id, 0);
                        self.set_buffer(enc, input_row_buf.buffer, 1);
                        // We need to write to the correct offset in the output buffer.
                        // Since the kernel always writes to output[0..N], we create a
                        // temporary output buffer per row and copy later.
                        // OR, we can offset the output pointer. Metal doesn't support
                        // buffer offsets in setBuffer at dispatch time easily, but
                        // we can set the offset parameter.
                        // Actually, setBuffer:offset:atIndex: supports byte offset.
                        // So we can offset into the output buffer.
                        msg_send_set_buffer(
                            enc,
                            self.sels.set_buffer,
                            out_buf.buffer,
                            row_out_offset * std::mem::size_of::<f32>(),
                            2,
                        );
                        self.set_u32(enc, n as u32, 3);
                        self.set_u32(enc, k as u32, 4);

                        // N_R0_Q8=2, N_SG_Q8=4 → 8 rows per threadgroup, 128 threads
                        let threadgroups = (n + 7) / 8; // ceil(N / (N_R0*N_SG))
                        msg_send_dispatch(
                            enc,
                            self.sels.dispatch_threadgroups,
                            threadgroups, 1, 1,
                            128, 1, 1,
                        );
                    }
                    TensorDtype::Q4_0 => {
                        let enc = self.ensure_encoder(self.pso_quantized_matmul_q4_0);

                        self.set_buffer(enc, weight_buf_id, 0);
                        self.set_buffer(enc, input_row_buf.buffer, 1);
                        msg_send_set_buffer(
                            enc,
                            self.sels.set_buffer,
                            out_buf.buffer,
                            row_out_offset * std::mem::size_of::<f32>(),
                            2,
                        );
                        self.set_u32(enc, n as u32, 3);
                        self.set_u32(enc, k as u32, 4);

                        // N_R0_Q4=4, N_SG_Q4=2 → 8 rows per threadgroup, 64 threads
                        let threadgroups = (n + 7) / 8;
                        msg_send_dispatch(
                            enc,
                            self.sels.dispatch_threadgroups,
                            threadgroups, 1, 1,
                            64, 1, 1,
                        );
                    }
                    _ => panic!(
                        "quantized_matmul: weights must be Q8_0 or Q4_0, got {:?}",
                        dtype
                    ),
                }

                // Drop the temporary input row buffer
                drop(input_row_buf);
            }

            Self::wrap(out_buf, vec![m, n], TensorDtype::F32)
        }
    }

    fn add(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor {
        let count: usize = a.shape().iter().product();
        let out_bytes = count * std::mem::size_of::<f32>();

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_add_tensor);

            self.set_buffer(enc, Self::buf_id(a), 0);
            self.set_buffer(enc, Self::buf_id(b), 1);
            self.set_buffer(enc, out_buf.buffer, 2);
            self.set_u32(enc, count as u32, 3);

            self.dispatch_1d(enc, count);

            Self::wrap(out_buf, a.shape().to_vec(), TensorDtype::F32)
        }
    }

    fn add_bias(&self, a: &DeviceTensor, bias: &DeviceTensor) -> DeviceTensor {
        let (rows, cols) = Self::shape_2d(a);
        let out_count = rows * cols;
        let out_bytes = out_count * std::mem::size_of::<f32>();

        unsafe {
            // add_bias kernel works in-place on the first buffer, so we need to
            // copy input to output first, then run in-place on the output.
            let out_buf = self.create_buffer_empty(out_bytes);

            // Copy input to output
            let src_ptr = msg_send_ptr(Self::buf_id(a), self.sels.contents);
            let dst_ptr = msg_send_ptr(out_buf.buffer, self.sels.contents);
            std::ptr::copy_nonoverlapping(
                src_ptr as *const u8,
                dst_ptr as *mut u8,
                out_bytes,
            );

            let enc = self.ensure_encoder(self.pso_add_bias);

            self.set_buffer(enc, out_buf.buffer, 0);
            self.set_buffer(enc, Self::buf_id(bias), 1);
            self.set_u32(enc, rows as u32, 2);
            self.set_u32(enc, cols as u32, 3);

            self.dispatch_2d(enc, cols, rows, 16, 16);

            Self::wrap(out_buf, a.shape().to_vec(), TensorDtype::F32)
        }
    }

    fn gelu(&self, t: &DeviceTensor) -> DeviceTensor {
        let count: usize = t.shape().iter().product();
        let out_bytes = count * std::mem::size_of::<f32>();

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_gelu);

            self.set_buffer(enc, Self::buf_id(t), 0);
            self.set_buffer(enc, out_buf.buffer, 1);
            self.set_u32(enc, count as u32, 2);

            self.dispatch_1d(enc, count);

            Self::wrap(out_buf, t.shape().to_vec(), TensorDtype::F32)
        }
    }

    fn silu(&self, t: &DeviceTensor) -> DeviceTensor {
        let count: usize = t.shape().iter().product();
        let out_bytes = count * std::mem::size_of::<f32>();

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_silu);

            self.set_buffer(enc, Self::buf_id(t), 0);
            self.set_buffer(enc, out_buf.buffer, 1);
            self.set_u32(enc, count as u32, 2);

            self.dispatch_1d(enc, count);

            Self::wrap(out_buf, t.shape().to_vec(), TensorDtype::F32)
        }
    }

    fn swiglu(&self, gate: &DeviceTensor, up: &DeviceTensor) -> DeviceTensor {
        let count: usize = gate.shape().iter().product();
        let out_bytes = count * std::mem::size_of::<f32>();

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_swiglu);

            self.set_buffer(enc, Self::buf_id(gate), 0);
            self.set_buffer(enc, Self::buf_id(up), 1);
            self.set_buffer(enc, out_buf.buffer, 2);
            self.set_u32(enc, count as u32, 3);

            self.dispatch_1d(enc, count);

            Self::wrap(out_buf, gate.shape().to_vec(), TensorDtype::F32)
        }
    }

    fn layer_norm(
        &self,
        t: &DeviceTensor,
        weight: &DeviceTensor,
        bias: &DeviceTensor,
        eps: f32,
    ) -> DeviceTensor {
        let (rows, cols) = Self::shape_2d(t);
        let out_bytes = rows * cols * std::mem::size_of::<f32>();

        // Choose threads per threadgroup: min(256, next_power_of_2(cols))
        let threads_per_group = (cols.next_power_of_two()).min(256);

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_layer_norm);

            self.set_buffer(enc, Self::buf_id(t), 0);
            self.set_buffer(enc, Self::buf_id(weight), 1);
            self.set_buffer(enc, Self::buf_id(bias), 2);
            self.set_buffer(enc, out_buf.buffer, 3);
            self.set_u32(enc, rows as u32, 4);
            self.set_u32(enc, cols as u32, 5);
            self.set_f32(enc, eps, 6);

            self.dispatch_rows(enc, rows, threads_per_group);

            Self::wrap(out_buf, t.shape().to_vec(), TensorDtype::F32)
        }
    }

    fn rms_norm(&self, t: &DeviceTensor, weight: &DeviceTensor, eps: f32) -> DeviceTensor {
        let (rows, cols) = Self::shape_2d(t);
        let out_bytes = rows * cols * std::mem::size_of::<f32>();

        // Choose threads per threadgroup: min(256, next_power_of_2(cols))
        let threads_per_group = (cols.next_power_of_two()).min(256);

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_rms_norm);

            self.set_buffer(enc, Self::buf_id(t), 0);
            self.set_buffer(enc, Self::buf_id(weight), 1);
            self.set_buffer(enc, out_buf.buffer, 2);
            self.set_u32(enc, rows as u32, 3);
            self.set_u32(enc, cols as u32, 4);
            self.set_f32(enc, eps, 5);

            self.dispatch_rows(enc, rows, threads_per_group);

            Self::wrap(out_buf, t.shape().to_vec(), TensorDtype::F32)
        }
    }

    fn softmax(&self, t: &DeviceTensor) -> DeviceTensor {
        let (rows, cols) = Self::shape_2d(t);
        let count = rows * cols;
        let out_bytes = count * std::mem::size_of::<f32>();

        let threads_per_group = (cols.next_power_of_two()).min(256);

        unsafe {
            // softmax_rows operates in-place, so copy input to output first.
            let out_buf = self.create_buffer_empty(out_bytes);

            let src_ptr = msg_send_ptr(Self::buf_id(t), self.sels.contents);
            let dst_ptr = msg_send_ptr(out_buf.buffer, self.sels.contents);
            std::ptr::copy_nonoverlapping(
                src_ptr as *const u8,
                dst_ptr as *mut u8,
                out_bytes,
            );

            let enc = self.ensure_encoder(self.pso_softmax_rows);

            self.set_buffer(enc, out_buf.buffer, 0);
            self.set_u32(enc, rows as u32, 1);
            self.set_u32(enc, cols as u32, 2);

            self.dispatch_rows(enc, rows, threads_per_group);

            Self::wrap(out_buf, t.shape().to_vec(), TensorDtype::F32)
        }
    }

    fn scale(&self, t: &DeviceTensor, factor: f32) -> DeviceTensor {
        let count: usize = t.shape().iter().product();
        let out_bytes = count * std::mem::size_of::<f32>();

        unsafe {
            // scale_kernel operates in-place, so copy input to output first.
            let out_buf = self.create_buffer_empty(out_bytes);

            let src_ptr = msg_send_ptr(Self::buf_id(t), self.sels.contents);
            let dst_ptr = msg_send_ptr(out_buf.buffer, self.sels.contents);
            std::ptr::copy_nonoverlapping(
                src_ptr as *const u8,
                dst_ptr as *mut u8,
                out_bytes,
            );

            let enc = self.ensure_encoder(self.pso_scale);

            self.set_buffer(enc, out_buf.buffer, 0);
            self.set_f32(enc, factor, 1);
            self.set_u32(enc, count as u32, 2);

            self.dispatch_1d(enc, count);

            Self::wrap(out_buf, t.shape().to_vec(), TensorDtype::F32)
        }
    }

    fn apply_causal_mask(&self, scores: &DeviceTensor, seq_len: usize) -> DeviceTensor {
        let (rows, cols) = Self::shape_2d(scores);
        let count = rows * cols;
        let out_bytes = count * std::mem::size_of::<f32>();

        unsafe {
            // causal_mask operates in-place, so copy input to output first.
            let out_buf = self.create_buffer_empty(out_bytes);

            let src_ptr = msg_send_ptr(Self::buf_id(scores), self.sels.contents);
            let dst_ptr = msg_send_ptr(out_buf.buffer, self.sels.contents);
            std::ptr::copy_nonoverlapping(
                src_ptr as *const u8,
                dst_ptr as *mut u8,
                out_bytes,
            );

            let enc = self.ensure_encoder(self.pso_causal_mask);

            self.set_buffer(enc, out_buf.buffer, 0);
            self.set_u32(enc, rows as u32, 1);
            self.set_u32(enc, cols as u32, 2);
            // offset = cols - seq_len, so that for a KV-cache scenario
            // the diagonal is shifted. For simple causal mask, offset = 0.
            let offset = if cols > seq_len { cols - seq_len } else { 0 };
            self.set_u32(enc, offset as u32, 3);

            self.dispatch_2d(enc, cols, rows, 16, 16);

            Self::wrap(out_buf, scores.shape().to_vec(), TensorDtype::F32)
        }
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
        let q_out =
            unsafe { self.dispatch_rope(self.pso_rope_norm, q, pos_offset, freq_base, head_dim, rope_dim) };
        let k_out =
            unsafe { self.dispatch_rope(self.pso_rope_norm, k, pos_offset, freq_base, head_dim, rope_dim) };
        (q_out, k_out)
    }

    fn mean_pool(&self, hidden: &DeviceTensor, mask: &[f32]) -> DeviceTensor {
        let (rows, cols) = Self::shape_2d(hidden);
        let out_bytes = cols * std::mem::size_of::<f32>();

        // Convert f32 mask to u32 (1.0 → 1u32, 0.0 → 0u32) for the kernel
        let mask_u32: Vec<u32> = mask.iter().map(|&v| if v > 0.5 { 1u32 } else { 0u32 }).collect();

        unsafe {
            let mask_buf = self.create_buffer_u32(&mask_u32);
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_mean_pool);

            self.set_buffer(enc, Self::buf_id(hidden), 0);
            self.set_buffer(enc, mask_buf.buffer, 1);
            self.set_buffer(enc, out_buf.buffer, 2);
            self.set_u32(enc, rows as u32, 3);
            self.set_u32(enc, cols as u32, 4);

            self.dispatch_1d(enc, cols);

            // CPU needs the result — flush all pending GPU work.
            self.flush();

            // mask_buf is dropped automatically
            Self::wrap(out_buf, vec![cols], TensorDtype::F32)
        }
    }

    fn l2_normalize(&self, t: &DeviceTensor) -> DeviceTensor {
        let count: usize = t.shape().iter().product();
        let out_bytes = count * std::mem::size_of::<f32>();

        // l2_normalize uses tree reduction — need power-of-2 thread count
        let threads_per_group = (count.next_power_of_two()).min(256);

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_l2_normalize);

            self.set_buffer(enc, Self::buf_id(t), 0);
            self.set_buffer(enc, out_buf.buffer, 1);
            self.set_u32(enc, count as u32, 2);

            // Single threadgroup processes entire vector
            self.dispatch_rows(enc, 1, threads_per_group);

            Self::wrap(out_buf, t.shape().to_vec(), TensorDtype::F32)
        }
    }

    fn embedding_lookup(&self, table: &DeviceTensor, ids: &[u32]) -> DeviceTensor {
        let table_shape = table.shape();
        let vocab_size = table_shape[0];
        let hidden = *table_shape.last().unwrap();
        let num_tokens = ids.len();

        // Bounds check: all token IDs must be within vocab range
        for (i, &id) in ids.iter().enumerate() {
            assert!(
                (id as usize) < vocab_size,
                "embedding_lookup: token_id {} at index {} is out of range (vocab_size = {})",
                id, i, vocab_size
            );
        }

        let out_count = num_tokens * hidden;
        let out_bytes = out_count * std::mem::size_of::<f32>();

        unsafe {
            let ids_buf = self.create_buffer_u32(ids);
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_embedding_lookup);

            self.set_buffer(enc, Self::buf_id(table), 0);
            self.set_buffer(enc, ids_buf.buffer, 1);
            self.set_buffer(enc, out_buf.buffer, 2);
            self.set_u32(enc, hidden as u32, 3);
            self.set_u32(enc, num_tokens as u32, 4);

            self.dispatch_2d(enc, hidden, num_tokens, 16, 16);

            // ids_buf is dropped automatically
            Self::wrap(out_buf, vec![num_tokens, hidden], TensorDtype::F32)
        }
    }

    fn mul(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor {
        let a_count: usize = a.shape().iter().product();
        let b_count: usize = b.shape().iter().product();
        let out_bytes = a_count * std::mem::size_of::<f32>();

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_mul_elementwise);

            self.set_buffer(enc, Self::buf_id(a), 0);
            self.set_buffer(enc, Self::buf_id(b), 1);
            self.set_buffer(enc, out_buf.buffer, 2);
            self.set_u32(enc, a_count as u32, 3);
            self.set_u32(enc, b_count as u32, 4);

            self.dispatch_1d(enc, a_count);

            Self::wrap(out_buf, a.shape().to_vec(), TensorDtype::F32)
        }
    }

    fn tanh(&self, t: &DeviceTensor) -> DeviceTensor {
        let count: usize = t.shape().iter().product();
        let out_bytes = count * std::mem::size_of::<f32>();

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_tanh_kernel);

            self.set_buffer(enc, Self::buf_id(t), 0);
            self.set_buffer(enc, out_buf.buffer, 1);
            self.set_u32(enc, count as u32, 2);

            self.dispatch_1d(enc, count);

            Self::wrap(out_buf, t.shape().to_vec(), TensorDtype::F32)
        }
    }

    fn geglu(&self, gate: &DeviceTensor, up: &DeviceTensor) -> DeviceTensor {
        let count: usize = gate.shape().iter().product();
        let out_bytes = count * std::mem::size_of::<f32>();

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_geglu);

            self.set_buffer(enc, Self::buf_id(gate), 0);
            self.set_buffer(enc, Self::buf_id(up), 1);
            self.set_buffer(enc, out_buf.buffer, 2);
            self.set_u32(enc, count as u32, 3);

            self.dispatch_1d(enc, count);

            Self::wrap(out_buf, gate.shape().to_vec(), TensorDtype::F32)
        }
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
        let q_out =
            unsafe { self.dispatch_rope(self.pso_rope_neox, q, pos_offset, freq_base, head_dim, rope_dim) };
        let k_out =
            unsafe { self.dispatch_rope(self.pso_rope_neox, k, pos_offset, freq_base, head_dim, rope_dim) };
        (q_out, k_out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::tensor::Tensor;

    fn backend() -> MetalBackend {
        MetalBackend::try_new().expect("Metal device required for these tests")
    }

    fn cpu() -> CpuBackend {
        CpuBackend::new()
    }

    fn assert_vecs_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(
            a.len(),
            b.len(),
            "{}: length mismatch {} vs {}",
            label,
            a.len(),
            b.len()
        );
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "{}: mismatch at index {}: {} vs {} (diff {})",
                label,
                i,
                x,
                y,
                (x - y).abs()
            );
        }
    }

    #[test]
    fn test_metal_init() {
        let _b = backend();
        // If we get here, try_new() succeeded.
    }

    #[test]
    fn test_upload_download_roundtrip() {
        let b = backend();
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::new(vec![2, 3], data.clone());
        let dt = b.upload(&t);
        let downloaded = b.download(&dt);
        assert_eq!(downloaded.shape(), &[2, 3]);
        assert_eq!(downloaded.as_f32(), &data[..]);
    }

    #[test]
    fn test_matmul_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        // A: 2x3, B: 3x4 -> C: 2x4
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ];

        let a_t = Tensor::new(vec![2, 3], a_data);
        let b_t = Tensor::new(vec![3, 4], b_data);

        let a_cpu = cpu_be.upload(&a_t);
        let b_cpu = cpu_be.upload(&b_t);
        let c_cpu = cpu_be.download(&cpu_be.matmul(&a_cpu, &b_cpu));

        let a_gpu = metal.upload(&a_t);
        let b_gpu = metal.upload(&b_t);
        let c_gpu = metal.download(&metal.matmul(&a_gpu, &b_gpu));

        assert_vecs_close(c_gpu.as_f32(), c_cpu.as_f32(), 1e-3, "matmul");
    }

    #[test]
    fn test_add_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![10.0, 20.0, 30.0, 40.0];
        let a_t = Tensor::new(vec![2, 2], a_data);
        let b_t = Tensor::new(vec![2, 2], b_data);

        let a_cpu = cpu_be.upload(&a_t);
        let b_cpu = cpu_be.upload(&b_t);
        let c_cpu = cpu_be.download(&cpu_be.add(&a_cpu, &b_cpu));

        let a_gpu = metal.upload(&a_t);
        let b_gpu = metal.upload(&b_t);
        let c_gpu = metal.download(&metal.add(&a_gpu, &b_gpu));

        assert_vecs_close(c_gpu.as_f32(), c_cpu.as_f32(), 1e-6, "add");
    }

    #[test]
    fn test_gelu_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        let data = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0];
        let t = Tensor::new(vec![6], data);

        let dt_cpu = cpu_be.upload(&t);
        let out_cpu = cpu_be.download(&cpu_be.gelu(&dt_cpu));

        let dt_gpu = metal.upload(&t);
        let out_gpu = metal.download(&metal.gelu(&dt_gpu));

        assert_vecs_close(out_gpu.as_f32(), out_cpu.as_f32(), 1e-4, "gelu");
    }

    #[test]
    fn test_rms_norm_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        // 2 rows, 4 cols
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-5;

        let t = Tensor::new(vec![2, 4], data);
        let w = Tensor::new(vec![4], weight);

        let dt_cpu = cpu_be.upload(&t);
        let w_cpu = cpu_be.upload(&w);
        let out_cpu = cpu_be.download(&cpu_be.rms_norm(&dt_cpu, &w_cpu, eps));

        let dt_gpu = metal.upload(&t);
        let w_gpu = metal.upload(&w);
        let out_gpu = metal.download(&metal.rms_norm(&dt_gpu, &w_gpu, eps));

        assert_vecs_close(out_gpu.as_f32(), out_cpu.as_f32(), 1e-4, "rms_norm");
    }

    #[test]
    fn test_quantized_matmul_q8_0_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        // Create a Q8_0 weight tensor: 2 rows x 32 cols
        // Each row is 1 block of 34 bytes
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

        // CPU reference
        let w_cpu = cpu_be.upload(&weights);
        let i_cpu = cpu_be.upload(&input);
        let out_cpu = cpu_be.download(&cpu_be.quantized_matmul(&w_cpu, &i_cpu));

        // Metal
        let w_gpu = metal.upload(&weights);
        let i_gpu = metal.upload(&input);
        let out_gpu = metal.download(&metal.quantized_matmul(&w_gpu, &i_gpu));

        assert_eq!(out_cpu.shape(), &[1, 2]);
        assert_eq!(out_gpu.shape(), &[1, 2]);
        assert_vecs_close(
            out_gpu.as_f32(),
            out_cpu.as_f32(),
            1e-3,
            "quantized_matmul_q8_0",
        );
    }

    #[test]
    fn test_silu_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        let data = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0];
        let t = Tensor::new(vec![6], data);

        let dt_cpu = cpu_be.upload(&t);
        let out_cpu = cpu_be.download(&cpu_be.silu(&dt_cpu));

        let dt_gpu = metal.upload(&t);
        let out_gpu = metal.download(&metal.silu(&dt_gpu));

        assert_vecs_close(out_gpu.as_f32(), out_cpu.as_f32(), 1e-4, "silu");
    }

    #[test]
    fn test_softmax_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        let data = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0];
        let t = Tensor::new(vec![2, 3], data);

        let dt_cpu = cpu_be.upload(&t);
        let out_cpu = cpu_be.download(&cpu_be.softmax(&dt_cpu));

        let dt_gpu = metal.upload(&t);
        let out_gpu = metal.download(&metal.softmax(&dt_gpu));

        assert_vecs_close(out_gpu.as_f32(), out_cpu.as_f32(), 1e-4, "softmax");
    }

    #[test]
    fn test_layer_norm_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let bias = vec![0.0, 0.0, 0.0, 0.0];
        let eps = 1e-5;

        let t = Tensor::new(vec![2, 4], data);
        let w = Tensor::new(vec![4], weight);
        let b = Tensor::new(vec![4], bias);

        let dt_cpu = cpu_be.upload(&t);
        let w_cpu = cpu_be.upload(&w);
        let b_cpu = cpu_be.upload(&b);
        let out_cpu = cpu_be.download(&cpu_be.layer_norm(&dt_cpu, &w_cpu, &b_cpu, eps));

        let dt_gpu = metal.upload(&t);
        let w_gpu = metal.upload(&w);
        let b_gpu = metal.upload(&b);
        let out_gpu = metal.download(&metal.layer_norm(&dt_gpu, &w_gpu, &b_gpu, eps));

        assert_vecs_close(out_gpu.as_f32(), out_cpu.as_f32(), 1e-4, "layer_norm");
    }

    #[test]
    fn test_scale_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let t = Tensor::new(vec![4], data);

        let dt_cpu = cpu_be.upload(&t);
        let out_cpu = cpu_be.download(&cpu_be.scale(&dt_cpu, 0.5));

        let dt_gpu = metal.upload(&t);
        let out_gpu = metal.download(&metal.scale(&dt_gpu, 0.5));

        assert_vecs_close(out_gpu.as_f32(), out_cpu.as_f32(), 1e-6, "scale");
    }

    #[test]
    fn test_embedding_lookup() {
        let metal = backend();

        // Table: 4 tokens, 3-dim embeddings
        let table_data = vec![
            0.1, 0.2, 0.3, // token 0
            0.4, 0.5, 0.6, // token 1
            0.7, 0.8, 0.9, // token 2
            1.0, 1.1, 1.2, // token 3
        ];
        let table = Tensor::new(vec![4, 3], table_data);
        let table_gpu = metal.upload(&table);

        let ids = vec![2u32, 0, 3];
        let result = metal.download(&metal.embedding_lookup(&table_gpu, &ids));

        assert_eq!(result.shape(), &[3, 3]);
        let data = result.as_f32();
        // Token 2: [0.7, 0.8, 0.9]
        assert_vecs_close(&data[0..3], &[0.7, 0.8, 0.9], 1e-6, "embed token 2");
        // Token 0: [0.1, 0.2, 0.3]
        assert_vecs_close(&data[3..6], &[0.1, 0.2, 0.3], 1e-6, "embed token 0");
        // Token 3: [1.0, 1.1, 1.2]
        assert_vecs_close(&data[6..9], &[1.0, 1.1, 1.2], 1e-6, "embed token 3");
    }

    #[test]
    fn test_mul_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![2.0, 3.0, 4.0, 5.0];
        let a = Tensor::new(vec![4], a_data);
        let b = Tensor::new(vec![4], b_data);

        let a_cpu = cpu_be.upload(&a);
        let b_cpu = cpu_be.upload(&b);
        let out_cpu = cpu_be.download(&cpu_be.mul(&a_cpu, &b_cpu));

        let a_gpu = metal.upload(&a);
        let b_gpu = metal.upload(&b);
        let out_gpu = metal.download(&metal.mul(&a_gpu, &b_gpu));

        assert_vecs_close(out_gpu.as_f32(), out_cpu.as_f32(), 1e-6, "mul");
    }

    #[test]
    fn test_tanh_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        let data = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0];
        let t = Tensor::new(vec![6], data);

        let dt_cpu = cpu_be.upload(&t);
        let out_cpu = cpu_be.download(&cpu_be.tanh(&dt_cpu));

        let dt_gpu = metal.upload(&t);
        let out_gpu = metal.download(&metal.tanh(&dt_gpu));

        assert_vecs_close(out_gpu.as_f32(), out_cpu.as_f32(), 1e-4, "tanh");
    }

    #[test]
    fn test_l2_normalize() {
        let metal = backend();

        let data = vec![3.0, 4.0]; // norm = 5
        let t = Tensor::new(vec![2], data);
        let dt = metal.upload(&t);
        let result = metal.download(&metal.l2_normalize(&dt));

        let expected = vec![3.0 / 5.0, 4.0 / 5.0];
        assert_vecs_close(result.as_f32(), &expected, 1e-4, "l2_normalize");
    }

    #[test]
    fn test_swiglu_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        let gate_data = vec![-1.0, 0.0, 1.0, 2.0];
        let up_data = vec![1.0, 2.0, 3.0, 4.0];
        let gate = Tensor::new(vec![4], gate_data);
        let up = Tensor::new(vec![4], up_data);

        let g_cpu = cpu_be.upload(&gate);
        let u_cpu = cpu_be.upload(&up);
        let out_cpu = cpu_be.download(&cpu_be.swiglu(&g_cpu, &u_cpu));

        let g_gpu = metal.upload(&gate);
        let u_gpu = metal.upload(&up);
        let out_gpu = metal.download(&metal.swiglu(&g_gpu, &u_gpu));

        assert_vecs_close(out_gpu.as_f32(), out_cpu.as_f32(), 1e-4, "swiglu");
    }

    #[test]
    fn test_matmul_transpose_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        // A: 2x3, B: 4x3 (transposed) -> C: 2x4
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let a = Tensor::new(vec![2, 3], a_data);
        let b = Tensor::new(vec![4, 3], b_data);

        let out_cpu = cpu_be.download(&cpu_be.matmul_transpose(&cpu_be.upload(&a), &cpu_be.upload(&b)));
        let out_gpu = metal.download(&metal.matmul_transpose(&metal.upload(&a), &metal.upload(&b)));

        assert_vecs_close(out_gpu.as_f32(), out_cpu.as_f32(), 1e-3, "matmul_transpose");
    }

    #[test]
    fn test_add_bias_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias_data = vec![10.0, 20.0, 30.0];
        let a = Tensor::new(vec![2, 3], a_data);
        let bias = Tensor::new(vec![3], bias_data);

        let out_cpu = cpu_be.download(&cpu_be.add_bias(&cpu_be.upload(&a), &cpu_be.upload(&bias)));
        let out_gpu = metal.download(&metal.add_bias(&metal.upload(&a), &metal.upload(&bias)));

        assert_vecs_close(out_gpu.as_f32(), out_cpu.as_f32(), 1e-6, "add_bias");
    }

    #[test]
    fn test_causal_mask_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let t = Tensor::new(vec![4, 4], data);

        let out_cpu = cpu_be.download(&cpu_be.apply_causal_mask(&cpu_be.upload(&t), 4));
        let out_gpu = metal.download(&metal.apply_causal_mask(&metal.upload(&t), 4));

        let cpu_d = out_cpu.as_f32();
        let gpu_d = out_gpu.as_f32();
        for i in 0..16 {
            if cpu_d[i].is_finite() && gpu_d[i].is_finite() {
                assert!((cpu_d[i] - gpu_d[i]).abs() < 1e-6,
                    "causal_mask mismatch at {}: {} vs {}", i, cpu_d[i], gpu_d[i]);
            } else {
                assert!(cpu_d[i] == gpu_d[i] || (cpu_d[i].is_infinite() && gpu_d[i].is_infinite()),
                    "causal_mask mismatch at {}: {} vs {}", i, cpu_d[i], gpu_d[i]);
            }
        }
    }

    #[test]
    fn test_rope_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        // seq_len=2, 2 heads, head_dim=4, rope_dim=4
        let q_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let k_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 + 0.5).collect();
        let q = Tensor::new(vec![2, 8], q_data);
        let k = Tensor::new(vec![2, 8], k_data);

        let (cpu_q, cpu_k) = cpu_be.rope(&cpu_be.upload(&q), &cpu_be.upload(&k), 0, 10000.0, 4, 4);
        let (gpu_q, gpu_k) = metal.rope(&metal.upload(&q), &metal.upload(&k), 0, 10000.0, 4, 4);

        let cpu_q_d = cpu_be.download(&cpu_q);
        let gpu_q_d = metal.download(&gpu_q);
        let cpu_k_d = cpu_be.download(&cpu_k);
        let gpu_k_d = metal.download(&gpu_k);

        assert_vecs_close(gpu_q_d.as_f32(), cpu_q_d.as_f32(), 1e-3, "rope Q");
        assert_vecs_close(gpu_k_d.as_f32(), cpu_k_d.as_f32(), 1e-3, "rope K");
    }

    #[test]
    fn test_rope_neox_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        // seq_len=2, 2 heads, head_dim=4, rope_dim=4
        let q_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let k_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 + 0.5).collect();
        let q = Tensor::new(vec![2, 8], q_data);
        let k = Tensor::new(vec![2, 8], k_data);

        let (cpu_q, cpu_k) = cpu_be.rope_neox(&cpu_be.upload(&q), &cpu_be.upload(&k), 3, 10000.0, 4, 4);
        let (gpu_q, gpu_k) = metal.rope_neox(&metal.upload(&q), &metal.upload(&k), 3, 10000.0, 4, 4);

        let cpu_q_d = cpu_be.download(&cpu_q);
        let gpu_q_d = metal.download(&gpu_q);
        let cpu_k_d = cpu_be.download(&cpu_k);
        let gpu_k_d = metal.download(&gpu_k);

        assert_vecs_close(gpu_q_d.as_f32(), cpu_q_d.as_f32(), 1e-3, "rope_neox Q");
        assert_vecs_close(gpu_k_d.as_f32(), cpu_k_d.as_f32(), 1e-3, "rope_neox K");
    }

    #[test]
    fn test_geglu_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        let gate_data = vec![-1.0, 0.0, 1.0, 2.0];
        let up_data = vec![1.0, 2.0, 3.0, 4.0];
        let gate = Tensor::new(vec![4], gate_data);
        let up = Tensor::new(vec![4], up_data);

        let out_cpu = cpu_be.download(&cpu_be.geglu(&cpu_be.upload(&gate), &cpu_be.upload(&up)));
        let out_gpu = metal.download(&metal.geglu(&metal.upload(&gate), &metal.upload(&up)));

        assert_vecs_close(out_gpu.as_f32(), out_cpu.as_f32(), 1e-4, "geglu");
    }

    #[test]
    fn test_mean_pool_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        // 3 tokens, 4 dims. Mask: [1, 1, 0] (only first 2 active)
        let data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1).collect();
        let t = Tensor::new(vec![3, 4], data);
        let mask = vec![1.0f32, 1.0, 0.0];

        let out_cpu = cpu_be.download(&cpu_be.mean_pool(&cpu_be.upload(&t), &mask));
        let out_gpu = metal.download(&metal.mean_pool(&metal.upload(&t), &mask));

        assert_vecs_close(out_gpu.as_f32(), out_cpu.as_f32(), 1e-4, "mean_pool");
    }

    #[test]
    fn test_quantized_matmul_q4_0_vs_cpu() {
        let metal = backend();
        let cpu_be = cpu();

        // Create Q4_0 weights: 2 rows x 32 cols
        // Q4_0 block: 2 bytes f16 scale + 16 bytes (32 nibbles)
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

        let out_cpu = cpu_be.download(&cpu_be.quantized_matmul(&cpu_be.upload(&weights), &cpu_be.upload(&input)));
        let out_gpu = metal.download(&metal.quantized_matmul(&metal.upload(&weights), &metal.upload(&input)));

        assert_eq!(out_cpu.shape(), &[1, 2]);
        assert_eq!(out_gpu.shape(), &[1, 2]);
        assert_vecs_close(out_gpu.as_f32(), out_cpu.as_f32(), 1.0, "quantized_matmul_q4_0");
    }

    #[test]
    fn test_upload_download_roundtrip_f32() {
        let metal = backend();

        let data = vec![1.0f32, -2.5, 3.14, 0.0, 100.0];
        let t = Tensor::new(vec![5], data.clone());
        let dt = metal.upload(&t);
        let result = metal.download(&dt);

        assert_eq!(result.shape(), &[5]);
        assert_vecs_close(result.as_f32(), &data, 1e-6, "roundtrip_f32");
    }

    #[test]
    fn test_upload_download_roundtrip_q8_0() {
        let metal = backend();

        // Create Q8_0 tensor: 1 row x 32 cols
        let scale = half::f16::from_f32(0.1);
        let mut raw = Vec::new();
        raw.extend_from_slice(&scale.to_bits().to_le_bytes());
        for i in 0..32i8 {
            raw.push(i as u8);
        }
        let t = Tensor::from_quantized(vec![1, 32], TensorDtype::Q8_0, raw);

        // Upload to GPU, download, and compare against CPU dequantization
        let cpu_f32 = t.to_f32();
        let dt = metal.upload(&t);
        let result = metal.download(&dt);
        let result_f32 = result.to_f32();

        assert_eq!(cpu_f32.shape(), result_f32.shape());
        assert_vecs_close(result_f32.as_f32(), cpu_f32.as_f32(), 1e-6, "roundtrip_q8_0");
    }

    #[test]
    fn test_rope_partial_rotation_vs_cpu() {
        // Test with rope_dim < head_dim to verify non-rotated dims are preserved
        let metal = backend();
        let cpu_be = cpu();

        // seq_len=1, 2 heads, head_dim=8, rope_dim=4 (only first 4 dims rotated)
        let q_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let k_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 + 0.5).collect();
        let q = Tensor::new(vec![1, 16], q_data);
        let k = Tensor::new(vec![1, 16], k_data);

        let (cpu_q, cpu_k) = cpu_be.rope(&cpu_be.upload(&q), &cpu_be.upload(&k), 1, 10000.0, 8, 4);
        let (gpu_q, gpu_k) = metal.rope(&metal.upload(&q), &metal.upload(&k), 1, 10000.0, 8, 4);

        let cpu_q_d = cpu_be.download(&cpu_q);
        let gpu_q_d = metal.download(&gpu_q);
        let cpu_k_d = cpu_be.download(&cpu_k);
        let gpu_k_d = metal.download(&gpu_k);

        assert_vecs_close(gpu_q_d.as_f32(), cpu_q_d.as_f32(), 1e-3, "rope partial Q");
        assert_vecs_close(gpu_k_d.as_f32(), cpu_k_d.as_f32(), 1e-3, "rope partial K");
    }

    #[test]
    fn test_rope_neox_partial_rotation_vs_cpu() {
        // Test with rope_dim < head_dim for NeoX variant
        let metal = backend();
        let cpu_be = cpu();

        // seq_len=1, 2 heads, head_dim=8, rope_dim=4
        let q_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let k_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 + 0.5).collect();
        let q = Tensor::new(vec![1, 16], q_data);
        let k = Tensor::new(vec![1, 16], k_data);

        let (cpu_q, cpu_k) = cpu_be.rope_neox(&cpu_be.upload(&q), &cpu_be.upload(&k), 1, 10000.0, 8, 4);
        let (gpu_q, gpu_k) = metal.rope_neox(&metal.upload(&q), &metal.upload(&k), 1, 10000.0, 8, 4);

        let cpu_q_d = cpu_be.download(&cpu_q);
        let gpu_q_d = metal.download(&gpu_q);
        let cpu_k_d = cpu_be.download(&cpu_k);
        let gpu_k_d = metal.download(&gpu_k);

        assert_vecs_close(gpu_q_d.as_f32(), cpu_q_d.as_f32(), 1e-3, "rope_neox partial Q");
        assert_vecs_close(gpu_k_d.as_f32(), cpu_k_d.as_f32(), 1e-3, "rope_neox partial K");
    }
}
