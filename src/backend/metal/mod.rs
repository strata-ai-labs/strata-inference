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
use std::time::Instant;

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
pub(crate) struct MetalBuffer {
    /// Raw `MTLBuffer` Objective-C object pointer.
    pub(crate) buffer: Id,
    /// Size in bytes — retained for debugging / future introspection.
    #[allow(dead_code)]
    pub(crate) len: usize,
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
// ProfileCounters — lightweight instrumentation for diagnosing overhead
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ProfileCounters {
    flush_count: u64,
    flush_time_us: u64,
    alloc_count: u64,
    alloc_bytes: u64,
    dispatch_count: u64,
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
    // Pipeline state objects
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
    pso_grouped_attn_decode: Id,
    pso_copy_buffer: Id,
    pso_quantized_matmul_q4_k: Id,
    pso_quantized_matmul_q5_k: Id,
    pso_quantized_matmul_q6_k: Id,
    pso_batched_causal_attention: Id,
    pso_copy_f32_to_f16: Id,
    pso_grouped_attn_decode_f16: Id,
    /// Active command buffer and compute encoder for deferred dispatch.
    /// `None` means no open command buffer; dispatches create one lazily.
    active_cmd: Mutex<Option<(Id, Id)>>,
    /// Profile counters for diagnosing per-token overhead.
    profile: Mutex<ProfileCounters>,
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
                self.pso_grouped_attn_decode,
                self.pso_copy_buffer,
                self.pso_quantized_matmul_q4_k,
                self.pso_quantized_matmul_q5_k,
                self.pso_quantized_matmul_q6_k,
                self.pso_batched_causal_attention,
                self.pso_copy_f32_to_f16,
                self.pso_grouped_attn_decode_f16,
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
                "grouped_attn_decode",
                "copy_buffer",
                "quantized_matmul_q4_k",
                "quantized_matmul_q5_k",
                "quantized_matmul_q6_k",
                "batched_causal_attention",
                "copy_f32_to_f16",
                "grouped_attn_decode_f16",
            ];

            let mut psos = [NIL; 30];
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

            debug!("MetalBackend initialized with 30 PSOs");

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
                pso_grouped_attn_decode: psos[22],
                pso_copy_buffer: psos[23],
                pso_quantized_matmul_q4_k: psos[24],
                pso_quantized_matmul_q5_k: psos[25],
                pso_quantized_matmul_q6_k: psos[26],
                pso_batched_causal_attention: psos[27],
                pso_copy_f32_to_f16: psos[28],
                pso_grouped_attn_decode_f16: psos[29],
                active_cmd: Mutex::new(None),
                profile: Mutex::new(ProfileCounters::default()),
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
        {
            let mut p = self.profile.lock().unwrap();
            p.alloc_count += 1;
            p.alloc_bytes += len as u64;
        }
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
        {
            let mut p = self.profile.lock().unwrap();
            p.dispatch_count += 1;
        }
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
            let t0 = Instant::now();
            msg_send_void(cmd, self.sels.wait_until_completed);
            let elapsed_us = t0.elapsed().as_micros() as u64;
            let mut p = self.profile.lock().unwrap();
            p.flush_count += 1;
            p.flush_time_us += elapsed_us;
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

        // Non-rotated dimensions (rope_dim .. head_dim) are copied by the
        // kernel itself, so no CPU-side memcpy is needed.

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
    fn is_gpu(&self) -> bool {
        true
    }

    fn reset_profile(&self) {
        let mut p = self.profile.lock().unwrap();
        *p = ProfileCounters::default();
    }

    fn profile_summary(&self) -> String {
        let p = self.profile.lock().unwrap();
        let flush_ms = p.flush_time_us as f64 / 1000.0;
        let alloc_mb = p.alloc_bytes as f64 / (1024.0 * 1024.0);
        format!(
            "flushes={} flush_time={:.1}ms allocs={} alloc_bytes={:.1}MB dispatches={}",
            p.flush_count, flush_ms, p.alloc_count, alloc_mb, p.dispatch_count,
        )
    }

    fn create_buffer_empty(
        &self,
        byte_size: usize,
        shape: Vec<usize>,
        dtype: TensorDtype,
    ) -> DeviceTensor {
        let buf = unsafe { self.create_buffer_empty(byte_size) };
        DeviceTensor::from_gpu(shape, dtype, Box::new(buf))
    }

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
            TensorDtype::Q8_0 | TensorDtype::Q4_0
            | TensorDtype::Q4_1 | TensorDtype::Q5_0 | TensorDtype::Q5_1
            | TensorDtype::Q4_K | TensorDtype::Q5_K | TensorDtype::Q6_K => {
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
        // Weights: [N, K] in quantized format, Input: [M, K] in F32 → Output: [M, N] in F32
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

        // For types without native GPU kernels, fall back to
        // dequantizing on CPU and using regular F32 matmul_transpose.
        if !matches!(dtype, TensorDtype::Q8_0 | TensorDtype::Q4_0 | TensorDtype::Q4_K | TensorDtype::Q5_K | TensorDtype::Q6_K) {
            tracing::debug!(
                ?dtype,
                "Metal quantized_matmul: no native kernel for {:?}, using dequant fallback",
                dtype
            );
            let weights_f32 = self.download(weights).to_f32();
            let weights_f32_dev = self.upload(&weights_f32);
            return self.matmul_transpose(input, &weights_f32_dev);
        }

        let out_count = m * n;
        let out_bytes = out_count * std::mem::size_of::<f32>();

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let weight_buf_id = Self::buf_id(weights);

            // Fast path: M=1 with GPU-resident input — bind the buffer directly,
            // no flush or copy needed. The quantized kernels expect a flat f32
            // input vector, which is exactly what a [1, K] GPU tensor is.
            if m == 1 && input.gpu_inner::<MetalBuffer>().is_some() {
                let input_buf_id = Self::buf_id(input);
                match dtype {
                    TensorDtype::Q8_0 => {
                        let enc = self.ensure_encoder(self.pso_quantized_matmul_q8_0);
                        self.set_buffer(enc, weight_buf_id, 0);
                        self.set_buffer(enc, input_buf_id, 1);
                        self.set_buffer(enc, out_buf.buffer, 2);
                        self.set_u32(enc, n as u32, 3);
                        self.set_u32(enc, k as u32, 4);
                        let threadgroups = (n + 7) / 8;
                        msg_send_dispatch(
                            enc, self.sels.dispatch_threadgroups,
                            threadgroups, 1, 1, 128, 1, 1,
                        );
                    }
                    TensorDtype::Q4_0 => {
                        let enc = self.ensure_encoder(self.pso_quantized_matmul_q4_0);
                        self.set_buffer(enc, weight_buf_id, 0);
                        self.set_buffer(enc, input_buf_id, 1);
                        self.set_buffer(enc, out_buf.buffer, 2);
                        self.set_u32(enc, n as u32, 3);
                        self.set_u32(enc, k as u32, 4);
                        let threadgroups = (n + 7) / 8;
                        msg_send_dispatch(
                            enc, self.sels.dispatch_threadgroups,
                            threadgroups, 1, 1, 64, 1, 1,
                        );
                    }
                    TensorDtype::Q4_K => {
                        let enc = self.ensure_encoder(self.pso_quantized_matmul_q4_k);
                        self.set_buffer(enc, weight_buf_id, 0);
                        self.set_buffer(enc, input_buf_id, 1);
                        self.set_buffer(enc, out_buf.buffer, 2);
                        self.set_u32(enc, n as u32, 3);
                        self.set_u32(enc, k as u32, 4);
                        let threadgroups = (n + 3) / 4;
                        msg_send_dispatch(
                            enc, self.sels.dispatch_threadgroups,
                            threadgroups, 1, 1, 64, 1, 1,
                        );
                    }
                    TensorDtype::Q5_K => {
                        let enc = self.ensure_encoder(self.pso_quantized_matmul_q5_k);
                        self.set_buffer(enc, weight_buf_id, 0);
                        self.set_buffer(enc, input_buf_id, 1);
                        self.set_buffer(enc, out_buf.buffer, 2);
                        self.set_u32(enc, n as u32, 3);
                        self.set_u32(enc, k as u32, 4);
                        let threadgroups = (n + 3) / 4;
                        msg_send_dispatch(
                            enc, self.sels.dispatch_threadgroups,
                            threadgroups, 1, 1, 64, 1, 1,
                        );
                    }
                    TensorDtype::Q6_K => {
                        let enc = self.ensure_encoder(self.pso_quantized_matmul_q6_k);
                        self.set_buffer(enc, weight_buf_id, 0);
                        self.set_buffer(enc, input_buf_id, 1);
                        self.set_buffer(enc, out_buf.buffer, 2);
                        self.set_u32(enc, n as u32, 3);
                        self.set_u32(enc, k as u32, 4);
                        let threadgroups = (n + 3) / 4;
                        msg_send_dispatch(
                            enc, self.sels.dispatch_threadgroups,
                            threadgroups, 1, 1, 64, 1, 1,
                        );
                    }
                    _ => unreachable!("handled by fallback above"),
                }
                return Self::wrap(out_buf, vec![1, n], TensorDtype::F32);
            }

            // General path: per-row dispatch
            for row in 0..m {
                let input_row_data = if let Some(tensor) = input.try_as_tensor() {
                    let f32_data = tensor.as_f32();
                    &f32_data[row * k..(row + 1) * k]
                } else {
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
                        msg_send_set_buffer(
                            enc,
                            self.sels.set_buffer,
                            out_buf.buffer,
                            row_out_offset * std::mem::size_of::<f32>(),
                            2,
                        );
                        self.set_u32(enc, n as u32, 3);
                        self.set_u32(enc, k as u32, 4);

                        let threadgroups = (n + 7) / 8;
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

                        let threadgroups = (n + 7) / 8;
                        msg_send_dispatch(
                            enc,
                            self.sels.dispatch_threadgroups,
                            threadgroups, 1, 1,
                            64, 1, 1,
                        );
                    }
                    TensorDtype::Q4_K | TensorDtype::Q5_K | TensorDtype::Q6_K => {
                        let pso = match dtype {
                            TensorDtype::Q4_K => self.pso_quantized_matmul_q4_k,
                            TensorDtype::Q5_K => self.pso_quantized_matmul_q5_k,
                            TensorDtype::Q6_K => self.pso_quantized_matmul_q6_k,
                            _ => unreachable!(),
                        };
                        let enc = self.ensure_encoder(pso);

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

                        let threadgroups = (n + 3) / 4;
                        msg_send_dispatch(
                            enc,
                            self.sels.dispatch_threadgroups,
                            threadgroups, 1, 1,
                            64, 1, 1,
                        );
                    }
                    _ => unreachable!("handled by fallback above"),
                }

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
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_add_bias);

            self.set_buffer(enc, Self::buf_id(a), 0);
            self.set_buffer(enc, out_buf.buffer, 1);
            self.set_buffer(enc, Self::buf_id(bias), 2);
            self.set_u32(enc, rows as u32, 3);
            self.set_u32(enc, cols as u32, 4);

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
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_softmax_rows);

            self.set_buffer(enc, Self::buf_id(t), 0);
            self.set_buffer(enc, out_buf.buffer, 1);
            self.set_u32(enc, rows as u32, 2);
            self.set_u32(enc, cols as u32, 3);

            self.dispatch_rows(enc, rows, threads_per_group);

            Self::wrap(out_buf, t.shape().to_vec(), TensorDtype::F32)
        }
    }

    fn scale(&self, t: &DeviceTensor, factor: f32) -> DeviceTensor {
        let count: usize = t.shape().iter().product();
        let out_bytes = count * std::mem::size_of::<f32>();

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_scale);

            self.set_buffer(enc, Self::buf_id(t), 0);
            self.set_buffer(enc, out_buf.buffer, 1);
            self.set_f32(enc, factor, 2);
            self.set_u32(enc, count as u32, 3);

            self.dispatch_1d(enc, count);

            Self::wrap(out_buf, t.shape().to_vec(), TensorDtype::F32)
        }
    }

    fn apply_causal_mask(&self, scores: &DeviceTensor, seq_len: usize) -> DeviceTensor {
        let (rows, cols) = Self::shape_2d(scores);
        let count = rows * cols;
        let out_bytes = count * std::mem::size_of::<f32>();

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_causal_mask);

            self.set_buffer(enc, Self::buf_id(scores), 0);
            self.set_buffer(enc, out_buf.buffer, 1);
            self.set_u32(enc, rows as u32, 2);
            self.set_u32(enc, cols as u32, 3);
            // offset = cols - seq_len, so that for a KV-cache scenario
            // the diagonal is shifted. For simple causal mask, offset = 0.
            let offset = if cols > seq_len { cols - seq_len } else { 0 };
            self.set_u32(enc, offset as u32, 4);

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
        // Convert non-F32 embedding tables — the GPU kernel expects F32.
        let table_f32;
        let effective_table = if table.dtype() != TensorDtype::F32 {
            let tensor = self.download(table);
            let f32_tensor = tensor.to_f32();
            table_f32 = self.upload(&f32_tensor);
            &table_f32
        } else {
            table
        };

        let table_shape = effective_table.shape();
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

            self.set_buffer(enc, Self::buf_id(effective_table), 0);
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

    fn copy_rows_into(
        &self,
        dest: &DeviceTensor,
        src: &DeviceTensor,
        dest_row_offset: usize,
    ) {
        debug_assert_eq!(src.dtype(), TensorDtype::F32, "copy_rows_into: src must be F32");
        debug_assert_eq!(dest.dtype(), TensorDtype::F32, "copy_rows_into: dest must be F32");
        let cols = dest.shape().last().copied().unwrap_or(0);
        let n_rows = src.shape()[0];
        let count = n_rows * cols; // number of f32 elements to copy
        let dest_offset = dest_row_offset * cols; // f32 element offset into dest

        if count == 0 {
            return;
        }

        unsafe {
            // GPU-side copy using the copy_buffer kernel.
            // The memory barrier in ensure_encoder() ensures prior kernel
            // writes to src are visible — no flush needed.
            let enc = self.ensure_encoder(self.pso_copy_buffer);
            self.set_buffer(enc, Self::buf_id(src), 0);
            self.set_buffer(enc, Self::buf_id(dest), 1);
            self.set_u32(enc, count as u32, 2);
            self.set_u32(enc, dest_offset as u32, 3);
            self.dispatch_1d(enc, count);
        }
    }

    fn grouped_attention_decode(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        total_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        attn_scale: f32,
        softcap: f32,
    ) -> DeviceTensor {
        assert!(
            head_dim <= 256,
            "grouped_attention_decode: head_dim {} exceeds max threadgroup size 256",
            head_dim
        );
        let total_dim = num_heads * head_dim;
        let out_bytes = total_dim * std::mem::size_of::<f32>();

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_grouped_attn_decode);

            self.set_buffer(enc, Self::buf_id(q), 0);
            self.set_buffer(enc, Self::buf_id(k), 1);
            self.set_buffer(enc, Self::buf_id(v), 2);
            self.set_buffer(enc, out_buf.buffer, 3);
            self.set_u32(enc, num_heads as u32, 4);
            self.set_u32(enc, num_kv_heads as u32, 5);
            self.set_u32(enc, head_dim as u32, 6);
            self.set_u32(enc, total_len as u32, 7);
            self.set_f32(enc, attn_scale, 8);
            self.set_f32(enc, softcap, 9);

            // One threadgroup per Q head, 256 threads each
            msg_send_dispatch(
                enc,
                self.sels.dispatch_threadgroups,
                num_heads, 1, 1,
                256, 1, 1,
            );

            Self::wrap(out_buf, vec![1, total_dim], TensorDtype::F32)
        }
    }

    fn batched_causal_attention(
        &self,
        q: &DeviceTensor,
        k_cache: &DeviceTensor,
        v_cache: &DeviceTensor,
        n_tokens: usize,
        total_len: usize,
        pos_offset: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        attn_scale: f32,
        softcap: f32,
    ) -> DeviceTensor {
        assert!(
            head_dim <= 256,
            "batched_causal_attention: head_dim {} exceeds max threadgroup size 256",
            head_dim
        );
        let total_dim = num_heads * head_dim;
        let out_bytes = n_tokens * total_dim * std::mem::size_of::<f32>();

        unsafe {
            let out_buf = self.create_buffer_empty(out_bytes);
            let enc = self.ensure_encoder(self.pso_batched_causal_attention);

            self.set_buffer(enc, Self::buf_id(q), 0);
            self.set_buffer(enc, Self::buf_id(k_cache), 1);
            self.set_buffer(enc, Self::buf_id(v_cache), 2);
            self.set_buffer(enc, out_buf.buffer, 3);
            self.set_u32(enc, num_heads as u32, 4);
            self.set_u32(enc, num_kv_heads as u32, 5);
            self.set_u32(enc, head_dim as u32, 6);
            self.set_u32(enc, n_tokens as u32, 7);
            self.set_u32(enc, total_len as u32, 8);
            self.set_u32(enc, pos_offset as u32, 9);
            self.set_f32(enc, attn_scale, 10);
            self.set_f32(enc, softcap, 11);

            // One threadgroup per (head, query_token) pair, 256 threads each
            // Flattened 1D: gid = q_idx * num_heads + h
            msg_send_dispatch(
                enc,
                self.sels.dispatch_threadgroups,
                num_heads * n_tokens, 1, 1,
                256, 1, 1,
            );

            Self::wrap(out_buf, vec![n_tokens, total_dim], TensorDtype::F32)
        }
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
    fn test_copy_rows_into_gpu() {
        let metal = backend();

        // Dest: [4, 3] zeros
        let dest_data = vec![0.0f32; 12];
        let dest_t = Tensor::new(vec![4, 3], dest_data);
        let dest_gpu = metal.upload(&dest_t);

        // Src: [2, 3] with known data
        let src_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let src_t = Tensor::new(vec![2, 3], src_data.clone());
        let src_gpu = metal.upload(&src_t);

        // Copy src into dest at row offset 1 (rows 1-2)
        metal.copy_rows_into(&dest_gpu, &src_gpu, 1);

        let result = metal.download(&dest_gpu);
        let data = result.as_f32();

        // Row 0 should remain zero
        assert_vecs_close(&data[0..3], &[0.0, 0.0, 0.0], 1e-6, "copy_rows row 0");
        // Rows 1-2 should match src
        assert_vecs_close(&data[3..6], &src_data[0..3], 1e-6, "copy_rows row 1");
        assert_vecs_close(&data[6..9], &src_data[3..6], 1e-6, "copy_rows row 2");
        // Row 3 should remain zero
        assert_vecs_close(&data[9..12], &[0.0, 0.0, 0.0], 1e-6, "copy_rows row 3");
    }

    #[test]
    fn test_quantized_matmul_q4_k_vs_cpu() {
        use crate::gguf::quant::{BlockQ4K, f32_to_f16};

        let metal = backend();
        let cpu_be = cpu();

        // Asymmetric nibbles (low=0xA, high=0x3), non-zero dmin, varied scales
        let mut block = BlockQ4K {
            d: f32_to_f16(0.75),
            dmin: f32_to_f16(0.25),
            scales: [0; 12],
            qs: [0; 128],
        };
        // Vary qs bytes: alternate 0x3A (low=0xA, high=3), 0x7C (low=0xC, high=7)
        for i in 0..128 {
            block.qs[i] = if i % 2 == 0 { 0x3A } else { 0x7C };
        }
        // Set all 8 sub-block scales and mins (first 4 direct, last 4 packed)
        block.scales[0] = 5;   // scale for sub-block 0
        block.scales[1] = 10;  // scale for sub-block 1
        block.scales[2] = 15;  // scale for sub-block 2
        block.scales[3] = 20;  // scale for sub-block 3
        block.scales[4] = 3;   // min for sub-block 0
        block.scales[5] = 7;   // min for sub-block 1
        block.scales[6] = 2;   // min for sub-block 2
        block.scales[7] = 9;   // min for sub-block 3
        // Packed scales for sub-blocks 4-7 (indices 8-11)
        block.scales[8] = 0x42;  // scale4=2, min4=4 (low/high nibbles)
        block.scales[9] = 0x53;  // scale5=3, min5=5
        block.scales[10] = 0x86; // scale6=6, min6=8
        block.scales[11] = 0xA7; // scale7=7, min7=10

        let mut raw = Vec::new();
        raw.extend_from_slice(&block.d.to_le_bytes());
        raw.extend_from_slice(&block.dmin.to_le_bytes());
        raw.extend_from_slice(&block.scales);
        raw.extend_from_slice(&block.qs);
        assert_eq!(raw.len(), 144);

        let weights = Tensor::from_quantized(vec![1, 256], TensorDtype::Q4_K, raw);
        // Diverse input: positive and negative, varied magnitudes
        let input_data: Vec<f32> = (0..256).map(|i| {
            let x = i as f32 * 0.02 - 2.56;  // range [-2.56, +2.54]
            x
        }).collect();
        let input = Tensor::new(vec![1, 256], input_data);

        let w_cpu = cpu_be.upload(&weights);
        let i_cpu = cpu_be.upload(&input);
        let out_cpu = cpu_be.download(&cpu_be.quantized_matmul(&w_cpu, &i_cpu));

        let w_gpu = metal.upload(&weights);
        let i_gpu = metal.upload(&input);
        let out_gpu = metal.download(&metal.quantized_matmul(&w_gpu, &i_gpu));

        assert_eq!(out_cpu.shape(), &[1, 1]);
        assert_eq!(out_gpu.shape(), &[1, 1]);
        assert_vecs_close(
            out_gpu.as_f32(),
            out_cpu.as_f32(),
            0.1,
            "quantized_matmul_q4_k",
        );
    }

    #[test]
    fn test_quantized_matmul_q5_k_vs_cpu() {
        use crate::gguf::quant::{BlockQ5K, f32_to_f16};

        let metal = backend();
        let cpu_be = cpu();

        // Non-zero qh (tests 5th bit), non-zero dmin, asymmetric nibbles, varied scales
        let mut qs = [0u8; 128];
        for i in 0..128 {
            // Vary: low nibbles 0x0..0xF, high nibbles different
            qs[i] = ((i as u8).wrapping_mul(3) & 0xF) | (((i as u8).wrapping_mul(7).wrapping_add(5) & 0xF) << 4);
        }
        let mut qh = [0u8; 32];
        // Set alternating high-bit patterns: 0xAA = bits 1,3,5,7 set
        for i in 0..32 {
            qh[i] = if i % 2 == 0 { 0xAA } else { 0x55 };
        }
        let block = BlockQ5K {
            d: f32_to_f16(0.5),
            dmin: f32_to_f16(0.125),
            scales: [4, 8, 12, 16, 3, 6, 9, 12, 0x31, 0x42, 0x53, 0x64],
            qh,
            qs,
        };

        let mut raw = Vec::new();
        raw.extend_from_slice(&block.d.to_le_bytes());
        raw.extend_from_slice(&block.dmin.to_le_bytes());
        raw.extend_from_slice(&block.scales);
        raw.extend_from_slice(&block.qh);
        raw.extend_from_slice(&block.qs);
        assert_eq!(raw.len(), 176);

        let weights = Tensor::from_quantized(vec![1, 256], TensorDtype::Q5_K, raw);
        let input_data: Vec<f32> = (0..256).map(|i| {
            let x = (i as f32) * 0.03 - 3.84;  // range [-3.84, +3.81]
            x
        }).collect();
        let input = Tensor::new(vec![1, 256], input_data);

        let w_cpu = cpu_be.upload(&weights);
        let i_cpu = cpu_be.upload(&input);
        let out_cpu = cpu_be.download(&cpu_be.quantized_matmul(&w_cpu, &i_cpu));

        let w_gpu = metal.upload(&weights);
        let i_gpu = metal.upload(&input);
        let out_gpu = metal.download(&metal.quantized_matmul(&w_gpu, &i_gpu));

        assert_eq!(out_cpu.shape(), &[1, 1]);
        assert_eq!(out_gpu.shape(), &[1, 1]);
        assert_vecs_close(
            out_gpu.as_f32(),
            out_cpu.as_f32(),
            0.1,
            "quantized_matmul_q5_k",
        );
    }

    #[test]
    fn test_quantized_matmul_q4_k_multi_row_vs_cpu() {
        use crate::gguf::quant::{BlockQ4K, f32_to_f16};

        let metal = backend();
        let cpu_be = cpu();

        // 5 rows x 256 cols — tests N=5 (not multiple of 4, exercises first_row < N guard)
        let mut raw = Vec::new();
        for row_idx in 0..5u8 {
            let mut block = BlockQ4K {
                d: f32_to_f16(0.5 + row_idx as f32 * 0.2),
                dmin: f32_to_f16(0.1 * (row_idx + 1) as f32),
                scales: [0; 12],
                qs: [0; 128],
            };
            for i in 0..128 {
                block.qs[i] = ((i as u8).wrapping_add(row_idx * 37)) | (((i as u8).wrapping_add(row_idx * 13)) << 4);
            }
            for s in 0..8 {
                block.scales[s] = (s as u8 + 1) * (row_idx + 1);
            }
            raw.extend_from_slice(&block.d.to_le_bytes());
            raw.extend_from_slice(&block.dmin.to_le_bytes());
            raw.extend_from_slice(&block.scales);
            raw.extend_from_slice(&block.qs);
        }

        let weights = Tensor::from_quantized(vec![5, 256], TensorDtype::Q4_K, raw);
        let input_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.02 - 2.56).collect();
        let input = Tensor::new(vec![1, 256], input_data);

        let w_cpu = cpu_be.upload(&weights);
        let i_cpu = cpu_be.upload(&input);
        let out_cpu = cpu_be.download(&cpu_be.quantized_matmul(&w_cpu, &i_cpu));

        let w_gpu = metal.upload(&weights);
        let i_gpu = metal.upload(&input);
        let out_gpu = metal.download(&metal.quantized_matmul(&w_gpu, &i_gpu));

        assert_eq!(out_cpu.shape(), &[1, 5]);
        assert_eq!(out_gpu.shape(), &[1, 5]);
        assert_vecs_close(
            out_gpu.as_f32(),
            out_cpu.as_f32(),
            0.5,
            "quantized_matmul_q4_k_multi_row",
        );
    }

    #[test]
    fn test_quantized_matmul_q5_k_multi_row_vs_cpu() {
        use crate::gguf::quant::{BlockQ5K, f32_to_f16};

        let metal = backend();
        let cpu_be = cpu();

        // 5 rows — tests first_row < N guard with N not a multiple of 4
        let mut raw = Vec::new();
        for row_idx in 0..5u8 {
            let mut qs = [0u8; 128];
            let mut qh = [0u8; 32];
            for i in 0..128 {
                qs[i] = (i as u8).wrapping_add(row_idx * 41);
            }
            for i in 0..32 {
                qh[i] = 0xCC_u8.wrapping_add(row_idx * 17 + i as u8);
            }
            let block = BlockQ5K {
                d: f32_to_f16(0.4 + row_idx as f32 * 0.15),
                dmin: f32_to_f16(0.05 * (row_idx + 1) as f32),
                scales: [3, 6, 9, 12, 2, 5, 8, 11, 0x21, 0x43, 0x65, 0x87],
                qh,
                qs,
            };
            raw.extend_from_slice(&block.d.to_le_bytes());
            raw.extend_from_slice(&block.dmin.to_le_bytes());
            raw.extend_from_slice(&block.scales);
            raw.extend_from_slice(&block.qh);
            raw.extend_from_slice(&block.qs);
        }

        let weights = Tensor::from_quantized(vec![5, 256], TensorDtype::Q5_K, raw);
        let input_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.025 - 3.2).collect();
        let input = Tensor::new(vec![1, 256], input_data);

        let w_cpu = cpu_be.upload(&weights);
        let i_cpu = cpu_be.upload(&input);
        let out_cpu = cpu_be.download(&cpu_be.quantized_matmul(&w_cpu, &i_cpu));

        let w_gpu = metal.upload(&weights);
        let i_gpu = metal.upload(&input);
        let out_gpu = metal.download(&metal.quantized_matmul(&w_gpu, &i_gpu));

        assert_eq!(out_cpu.shape(), &[1, 5]);
        assert_eq!(out_gpu.shape(), &[1, 5]);
        assert_vecs_close(
            out_gpu.as_f32(),
            out_cpu.as_f32(),
            0.5,
            "quantized_matmul_q5_k_multi_row",
        );
    }

    #[test]
    fn test_quantized_matmul_q6_k_vs_cpu() {
        use crate::gguf::quant::f32_to_f16;

        let metal = backend();
        let cpu_be = cpu();

        // Non-zero qh (tests high 2 bits), varied ql nibbles, varied scales
        let mut ql = [0u8; 128];
        for i in 0..128 {
            // Asymmetric nibbles: low varies, high varies differently
            ql[i] = ((i as u8).wrapping_mul(5).wrapping_add(3) & 0xF) | (((i as u8).wrapping_mul(11).wrapping_add(7) & 0xF) << 4);
        }
        let mut qh = [0u8; 64];
        for i in 0..64 {
            // Varied 2-bit patterns across all 4 shifts (bits 0-1, 2-3, 4-5, 6-7)
            qh[i] = (i as u8).wrapping_mul(0x37).wrapping_add(0x29);
        }
        let scales: [i8; 16] = [3, -2, 5, -4, 7, -1, 6, -3, 2, -5, 4, -6, 8, -7, 1, -8];

        let mut raw = Vec::new();
        raw.extend_from_slice(&ql);
        raw.extend_from_slice(&qh);
        raw.extend_from_slice(unsafe {
            std::slice::from_raw_parts(scales.as_ptr() as *const u8, 16)
        });
        raw.extend_from_slice(&f32_to_f16(0.6).to_le_bytes());
        assert_eq!(raw.len(), 210);

        let weights = Tensor::from_quantized(vec![1, 256], TensorDtype::Q6_K, raw);
        let input_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.025 - 3.2).collect();
        let input = Tensor::new(vec![1, 256], input_data);

        let w_cpu = cpu_be.upload(&weights);
        let i_cpu = cpu_be.upload(&input);
        let out_cpu = cpu_be.download(&cpu_be.quantized_matmul(&w_cpu, &i_cpu));

        let w_gpu = metal.upload(&weights);
        let i_gpu = metal.upload(&input);
        let out_gpu = metal.download(&metal.quantized_matmul(&w_gpu, &i_gpu));

        assert_eq!(out_cpu.shape(), &[1, 1]);
        assert_eq!(out_gpu.shape(), &[1, 1]);
        assert_vecs_close(
            out_gpu.as_f32(),
            out_cpu.as_f32(),
            0.1,
            "quantized_matmul_q6_k",
        );
    }

    #[test]
    fn test_quantized_matmul_q6_k_multi_row_vs_cpu() {
        use crate::gguf::quant::f32_to_f16;

        let metal = backend();
        let cpu_be = cpu();

        // 5 rows — tests first_row < N guard with N not a multiple of 4
        let mut raw = Vec::new();
        for row_idx in 0..5u8 {
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            for i in 0..128 { ql[i] = (i as u8).wrapping_add(row_idx * 53); }
            for i in 0..64 { qh[i] = (i as u8).wrapping_mul(0x2B).wrapping_add(row_idx * 19); }
            let scales: [i8; 16] = [
                (row_idx as i8 + 1), -(row_idx as i8 + 2), (row_idx as i8 + 3), -(row_idx as i8 + 1),
                (row_idx as i8 + 4), -(row_idx as i8 + 3), (row_idx as i8 + 2), -(row_idx as i8 + 4),
                (row_idx as i8 + 5), -(row_idx as i8 + 1), (row_idx as i8 + 6), -(row_idx as i8 + 2),
                (row_idx as i8 + 7), -(row_idx as i8 + 5), (row_idx as i8 + 3), -(row_idx as i8 + 6),
            ];
            raw.extend_from_slice(&ql);
            raw.extend_from_slice(&qh);
            raw.extend_from_slice(unsafe {
                std::slice::from_raw_parts(scales.as_ptr() as *const u8, 16)
            });
            raw.extend_from_slice(&f32_to_f16(0.5 + row_idx as f32 * 0.1).to_le_bytes());
        }

        let weights = Tensor::from_quantized(vec![5, 256], TensorDtype::Q6_K, raw);
        let input_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.025 - 3.2).collect();
        let input = Tensor::new(vec![1, 256], input_data);

        let w_cpu = cpu_be.upload(&weights);
        let i_cpu = cpu_be.upload(&input);
        let out_cpu = cpu_be.download(&cpu_be.quantized_matmul(&w_cpu, &i_cpu));

        let w_gpu = metal.upload(&weights);
        let i_gpu = metal.upload(&input);
        let out_gpu = metal.download(&metal.quantized_matmul(&w_gpu, &i_gpu));

        assert_eq!(out_cpu.shape(), &[1, 5]);
        assert_eq!(out_gpu.shape(), &[1, 5]);
        assert_vecs_close(
            out_gpu.as_f32(),
            out_cpu.as_f32(),
            0.5,
            "quantized_matmul_q6_k_multi_row",
        );
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

    #[test]
    fn test_batched_causal_attention_vs_cpu_ref() {
        // Simple test: 3 query tokens, 2 heads, 4-dim head, no GQA, no softcap
        let metal = backend();
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let n_tokens = 3;
        let total_len = 3; // all from this batch
        let pos_offset = 0;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();

        // Q: [3, 8]  (3 tokens, 2 heads * 4 head_dim)
        let q_data: Vec<f32> = (0..n_tokens * num_heads * head_dim)
            .map(|i| (i as f32) * 0.1)
            .collect();
        // K: [3, 8]  (3 tokens in cache)
        let k_data: Vec<f32> = (0..total_len * num_kv_heads * head_dim)
            .map(|i| (i as f32) * 0.05 + 0.1)
            .collect();
        // V: [3, 8]
        let v_data: Vec<f32> = (0..total_len * num_kv_heads * head_dim)
            .map(|i| (i as f32) * 0.02)
            .collect();

        // CPU reference: per-head, per-query causal attention
        let total_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mut expected = vec![0.0f32; n_tokens * total_dim];

        for q_idx in 0..n_tokens {
            let max_attend = pos_offset + q_idx + 1;
            for h in 0..num_heads {
                let kv_head = h; // no GQA
                let q_off = q_idx * total_dim + h * head_dim;
                let kv_off = kv_head * head_dim;

                // Compute scores
                let mut scores: Vec<f32> = Vec::new();
                for j in 0..max_attend {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_data[q_off + d] * k_data[j * kv_dim + kv_off + d];
                    }
                    scores.push(dot * attn_scale);
                }

                // Softmax
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
                let sum_exp: f32 = exps.iter().sum();
                let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

                // Weighted sum of V
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for j in 0..max_attend {
                        acc += probs[j] * v_data[j * kv_dim + kv_off + d];
                    }
                    expected[q_idx * total_dim + h * head_dim + d] = acc;
                }
            }
        }

        // Run on Metal
        let q_gpu = metal.upload(&Tensor::new(vec![n_tokens, total_dim], q_data));
        let k_gpu = metal.upload(&Tensor::new(vec![total_len, kv_dim], k_data));
        let v_gpu = metal.upload(&Tensor::new(vec![total_len, kv_dim], v_data));

        let result = metal.batched_causal_attention(
            &q_gpu, &k_gpu, &v_gpu,
            n_tokens, total_len, pos_offset,
            num_heads, num_kv_heads, head_dim,
            attn_scale, 0.0,
        );

        let result_data = metal.download(&result);
        assert_vecs_close(result_data.as_f32(), &expected, 1e-4, "batched_causal_attention");
    }

    #[test]
    fn test_batched_causal_attention_gqa() {
        // GQA test: 4 Q heads, 2 KV heads (groups of 2)
        let metal = backend();
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let n_tokens = 2;
        let total_len = 2;
        let pos_offset = 0;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();

        let total_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q_data: Vec<f32> = (0..n_tokens * total_dim)
            .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
            .collect();
        let k_data: Vec<f32> = (0..total_len * kv_dim)
            .map(|i| ((i % 5) as f32) * 0.08)
            .collect();
        let v_data: Vec<f32> = (0..total_len * kv_dim)
            .map(|i| ((i % 3) as f32) * 0.15)
            .collect();

        // CPU reference with GQA
        let mut expected = vec![0.0f32; n_tokens * total_dim];
        let heads_per_kv = num_heads / num_kv_heads;

        for q_idx in 0..n_tokens {
            let max_attend = pos_offset + q_idx + 1;
            for h in 0..num_heads {
                let kv_head = h / heads_per_kv;
                let q_off = q_idx * total_dim + h * head_dim;
                let kv_off = kv_head * head_dim;

                let mut scores: Vec<f32> = Vec::new();
                for j in 0..max_attend {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_data[q_off + d] * k_data[j * kv_dim + kv_off + d];
                    }
                    scores.push(dot * attn_scale);
                }

                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
                let sum_exp: f32 = exps.iter().sum();
                let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for j in 0..max_attend {
                        acc += probs[j] * v_data[j * kv_dim + kv_off + d];
                    }
                    expected[q_idx * total_dim + h * head_dim + d] = acc;
                }
            }
        }

        let q_gpu = metal.upload(&Tensor::new(vec![n_tokens, total_dim], q_data));
        let k_gpu = metal.upload(&Tensor::new(vec![total_len, kv_dim], k_data));
        let v_gpu = metal.upload(&Tensor::new(vec![total_len, kv_dim], v_data));

        let result = metal.batched_causal_attention(
            &q_gpu, &k_gpu, &v_gpu,
            n_tokens, total_len, pos_offset,
            num_heads, num_kv_heads, head_dim,
            attn_scale, 0.0,
        );

        let result_data = metal.download(&result);
        assert_vecs_close(result_data.as_f32(), &expected, 1e-4, "batched_causal_attention_gqa");
    }

    #[test]
    fn test_batched_causal_attention_softcap() {
        // Test softcap > 0: scores are clamped via tanh before softmax
        let metal = backend();
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let n_tokens = 2;
        let total_len = 2;
        let pos_offset = 0;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();
        let softcap = 5.0f32;

        let total_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Use larger values to make softcap meaningful
        let q_data: Vec<f32> = (0..n_tokens * total_dim)
            .map(|i| (i as f32) * 0.5 - 1.0)
            .collect();
        let k_data: Vec<f32> = (0..total_len * kv_dim)
            .map(|i| (i as f32) * 0.3 + 0.5)
            .collect();
        let v_data: Vec<f32> = (0..total_len * kv_dim)
            .map(|i| (i as f32) * 0.1)
            .collect();

        // CPU reference with softcap
        let mut expected = vec![0.0f32; n_tokens * total_dim];

        for q_idx in 0..n_tokens {
            let max_attend = pos_offset + q_idx + 1;
            for h in 0..num_heads {
                let kv_head = h * num_kv_heads / num_heads;
                let q_off = q_idx * total_dim + h * head_dim;
                let kv_off = kv_head * head_dim;

                let mut scores: Vec<f32> = Vec::new();
                for j in 0..max_attend {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_data[q_off + d] * k_data[j * kv_dim + kv_off + d];
                    }
                    let mut s = dot * attn_scale;
                    // Apply softcap: softcap * tanh(s / softcap)
                    s = softcap * (s / softcap).tanh();
                    scores.push(s);
                }

                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
                let sum_exp: f32 = exps.iter().sum();
                let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for j in 0..max_attend {
                        acc += probs[j] * v_data[j * kv_dim + kv_off + d];
                    }
                    expected[q_idx * total_dim + h * head_dim + d] = acc;
                }
            }
        }

        let q_gpu = metal.upload(&Tensor::new(vec![n_tokens, total_dim], q_data));
        let k_gpu = metal.upload(&Tensor::new(vec![total_len, kv_dim], k_data));
        let v_gpu = metal.upload(&Tensor::new(vec![total_len, kv_dim], v_data));

        let result = metal.batched_causal_attention(
            &q_gpu, &k_gpu, &v_gpu,
            n_tokens, total_len, pos_offset,
            num_heads, num_kv_heads, head_dim,
            attn_scale, softcap,
        );

        let result_data = metal.download(&result);
        assert_vecs_close(
            result_data.as_f32(), &expected, 1e-4,
            "batched_causal_attention_softcap",
        );
    }

    #[test]
    fn test_batched_causal_attention_pos_offset() {
        // Test continuation: pos_offset > 0, simulating appending to existing KV cache
        // Cache has 5 positions already, we add 3 more query tokens
        let metal = backend();
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let n_tokens = 3;
        let pos_offset = 5;
        let total_len = pos_offset + n_tokens; // 8
        let attn_scale = 1.0 / (head_dim as f32).sqrt();

        let total_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q_data: Vec<f32> = (0..n_tokens * total_dim)
            .map(|i| ((i % 11) as f32) * 0.1 - 0.5)
            .collect();
        // Full KV cache (8 positions)
        let k_data: Vec<f32> = (0..total_len * kv_dim)
            .map(|i| ((i % 7) as f32) * 0.05 + 0.1)
            .collect();
        let v_data: Vec<f32> = (0..total_len * kv_dim)
            .map(|i| ((i % 13) as f32) * 0.02)
            .collect();

        // CPU reference
        let mut expected = vec![0.0f32; n_tokens * total_dim];

        for q_idx in 0..n_tokens {
            // Causal: query at position (pos_offset + q_idx) attends to [0, pos_offset + q_idx]
            let max_attend = pos_offset + q_idx + 1;
            for h in 0..num_heads {
                let kv_head = h * num_kv_heads / num_heads;
                let q_off = q_idx * total_dim + h * head_dim;
                let kv_off = kv_head * head_dim;

                let mut scores: Vec<f32> = Vec::new();
                for j in 0..max_attend {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_data[q_off + d] * k_data[j * kv_dim + kv_off + d];
                    }
                    scores.push(dot * attn_scale);
                }

                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
                let sum_exp: f32 = exps.iter().sum();
                let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for j in 0..max_attend {
                        acc += probs[j] * v_data[j * kv_dim + kv_off + d];
                    }
                    expected[q_idx * total_dim + h * head_dim + d] = acc;
                }
            }
        }

        let q_gpu = metal.upload(&Tensor::new(vec![n_tokens, total_dim], q_data));
        let k_gpu = metal.upload(&Tensor::new(vec![total_len, kv_dim], k_data));
        let v_gpu = metal.upload(&Tensor::new(vec![total_len, kv_dim], v_data));

        let result = metal.batched_causal_attention(
            &q_gpu, &k_gpu, &v_gpu,
            n_tokens, total_len, pos_offset,
            num_heads, num_kv_heads, head_dim,
            attn_scale, 0.0,
        );

        let result_data = metal.download(&result);
        assert_vecs_close(
            result_data.as_f32(), &expected, 1e-4,
            "batched_causal_attention_pos_offset",
        );
    }

    #[test]
    fn test_copy_f32_to_f16() {
        // Test F32→F16 conversion kernel
        let metal = backend();
        let count = 64usize;
        let src_data: Vec<f32> = (0..count).map(|i| (i as f32) * 0.1 - 3.0).collect();

        // Create F32 source buffer
        let src_buf = metal.upload(&Tensor::new(vec![count], src_data.clone()));

        // Create F16 dest buffer (2 bytes per element)
        let dest_bytes = count * 2;
        let dest_buf = unsafe {
            let mb = metal.create_buffer_empty(dest_bytes);
            DeviceTensor::from_gpu(vec![count], TensorDtype::F16, Box::new(mb))
        };

        // Dispatch copy kernel
        unsafe {
            let enc = metal.ensure_encoder(metal.pso_copy_f32_to_f16);
            metal.set_buffer(enc, MetalBackend::buf_id(&src_buf), 0);
            metal.set_buffer(enc, MetalBackend::buf_id(&dest_buf), 1);
            metal.set_u32(enc, count as u32, 2);
            metal.set_u32(enc, 0u32, 3); // dest_offset = 0
            let threads = 256usize;
            let groups = (count + threads - 1) / threads;
            ffi::msg_send_dispatch(
                enc,
                metal.sels.dispatch_threadgroups,
                groups, 1, 1,
                threads, 1, 1,
            );
            metal.flush();
        }

        // Read back F16 data and verify
        let mb = dest_buf.gpu_inner::<MetalBuffer>().unwrap();
        let ptr = unsafe { ffi::msg_send_ptr(mb.buffer, metal.sels.contents) as *const u16 };
        let f16_data: Vec<u16> = unsafe { std::slice::from_raw_parts(ptr, count).to_vec() };

        for (i, &f16_val) in f16_data.iter().enumerate() {
            let f32_back = half::f16::from_bits(f16_val).to_f32();
            let expected = src_data[i];
            assert!(
                (f32_back - expected).abs() < 0.01,
                "copy_f32_to_f16 mismatch at {}: got {} expected {} (diff {})",
                i, f32_back, expected, (f32_back - expected).abs()
            );
        }
    }

    #[test]
    fn test_copy_f32_to_f16_with_offset() {
        // Test F32→F16 conversion with non-zero dest_offset
        let metal = backend();
        let count = 16usize;
        let dest_offset = 8usize;
        let total_dest = dest_offset + count;

        let src_data: Vec<f32> = (0..count).map(|i| (i as f32) * 0.5).collect();

        let src_buf = metal.upload(&Tensor::new(vec![count], src_data.clone()));

        // Create F16 dest buffer large enough for offset + data
        let dest_bytes = total_dest * 2;
        let dest_buf = unsafe {
            let mb = metal.create_buffer_empty(dest_bytes);
            DeviceTensor::from_gpu(vec![total_dest], TensorDtype::F16, Box::new(mb))
        };

        unsafe {
            let enc = metal.ensure_encoder(metal.pso_copy_f32_to_f16);
            metal.set_buffer(enc, MetalBackend::buf_id(&src_buf), 0);
            metal.set_buffer(enc, MetalBackend::buf_id(&dest_buf), 1);
            metal.set_u32(enc, count as u32, 2);
            metal.set_u32(enc, dest_offset as u32, 3);
            let threads = 256usize;
            let groups = (count + threads - 1) / threads;
            ffi::msg_send_dispatch(
                enc,
                metal.sels.dispatch_threadgroups,
                groups, 1, 1,
                threads, 1, 1,
            );
            metal.flush();
        }

        // Read back and verify only the offset portion
        let mb = dest_buf.gpu_inner::<MetalBuffer>().unwrap();
        let ptr = unsafe { ffi::msg_send_ptr(mb.buffer, metal.sels.contents) as *const u16 };
        let f16_data: Vec<u16> =
            unsafe { std::slice::from_raw_parts(ptr.add(dest_offset), count).to_vec() };

        for (i, &f16_val) in f16_data.iter().enumerate() {
            let f32_back = half::f16::from_bits(f16_val).to_f32();
            let expected = src_data[i];
            assert!(
                (f32_back - expected).abs() < 0.01,
                "copy_f32_to_f16 offset mismatch at {}: got {} expected {}",
                i, f32_back, expected
            );
        }
    }

    #[test]
    fn test_grouped_attn_decode_f16() {
        // Test F16 KV cache attention against CPU reference (F32)
        let metal = backend();
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let total_len = 5;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();

        let total_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Q: [1, total_dim]
        let q_data: Vec<f32> = (0..total_dim)
            .map(|i| (i as f32) * 0.2 - 0.4)
            .collect();
        // K, V: [total_len, kv_dim] — will be converted to F16
        let k_data: Vec<f32> = (0..total_len * kv_dim)
            .map(|i| ((i % 9) as f32) * 0.1 - 0.3)
            .collect();
        let v_data: Vec<f32> = (0..total_len * kv_dim)
            .map(|i| ((i % 5) as f32) * 0.15)
            .collect();

        // CPU reference
        let mut expected = vec![0.0f32; total_dim];
        for h in 0..num_heads {
            let kv_head = h * num_kv_heads / num_heads;
            let kv_off = kv_head * head_dim;

            let mut scores: Vec<f32> = Vec::new();
            for j in 0..total_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_data[h * head_dim + d] * k_data[j * kv_dim + kv_off + d];
                }
                scores.push(dot * attn_scale);
            }

            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..total_len {
                    acc += probs[j] * v_data[j * kv_dim + kv_off + d];
                }
                expected[h * head_dim + d] = acc;
            }
        }

        // Upload Q as F32
        let q_gpu = metal.upload(&Tensor::new(vec![1, total_dim], q_data));

        // Convert K and V to F16 on GPU
        let k_f32 = metal.upload(&Tensor::new(vec![total_len, kv_dim], k_data));
        let v_f32 = metal.upload(&Tensor::new(vec![total_len, kv_dim], v_data));

        let k_count = total_len * kv_dim;
        let v_count = total_len * kv_dim;
        let k_f16 = unsafe {
            let mb = metal.create_buffer_empty(k_count * 2);
            DeviceTensor::from_gpu(
                vec![total_len, kv_dim], TensorDtype::F16, Box::new(mb),
            )
        };
        let v_f16 = unsafe {
            let mb = metal.create_buffer_empty(v_count * 2);
            DeviceTensor::from_gpu(
                vec![total_len, kv_dim], TensorDtype::F16, Box::new(mb),
            )
        };

        // Copy F32 → F16 for K
        unsafe {
            let enc = metal.ensure_encoder(metal.pso_copy_f32_to_f16);
            metal.set_buffer(enc, MetalBackend::buf_id(&k_f32), 0);
            metal.set_buffer(enc, MetalBackend::buf_id(&k_f16), 1);
            metal.set_u32(enc, k_count as u32, 2);
            metal.set_u32(enc, 0u32, 3);
            let threads = 256usize;
            let groups = (k_count + threads - 1) / threads;
            ffi::msg_send_dispatch(
                enc, metal.sels.dispatch_threadgroups,
                groups, 1, 1, threads, 1, 1,
            );
        }
        // Copy F32 → F16 for V
        unsafe {
            let enc = metal.ensure_encoder(metal.pso_copy_f32_to_f16);
            metal.set_buffer(enc, MetalBackend::buf_id(&v_f32), 0);
            metal.set_buffer(enc, MetalBackend::buf_id(&v_f16), 1);
            metal.set_u32(enc, v_count as u32, 2);
            metal.set_u32(enc, 0u32, 3);
            let threads = 256usize;
            let groups = (v_count + threads - 1) / threads;
            ffi::msg_send_dispatch(
                enc, metal.sels.dispatch_threadgroups,
                groups, 1, 1, threads, 1, 1,
            );
        }

        // Dispatch grouped_attn_decode_f16
        let out_bytes = total_dim * std::mem::size_of::<f32>();
        let result = unsafe {
            let out_buf = metal.create_buffer_empty(out_bytes);
            let enc = metal.ensure_encoder(metal.pso_grouped_attn_decode_f16);
            metal.set_buffer(enc, MetalBackend::buf_id(&q_gpu), 0);
            metal.set_buffer(enc, MetalBackend::buf_id(&k_f16), 1);
            metal.set_buffer(enc, MetalBackend::buf_id(&v_f16), 2);
            metal.set_buffer(enc, out_buf.buffer, 3);
            metal.set_u32(enc, num_heads as u32, 4);
            metal.set_u32(enc, num_kv_heads as u32, 5);
            metal.set_u32(enc, head_dim as u32, 6);
            metal.set_u32(enc, total_len as u32, 7);
            metal.set_f32(enc, attn_scale, 8);
            metal.set_f32(enc, 0.0f32, 9); // no softcap
            ffi::msg_send_dispatch(
                enc, metal.sels.dispatch_threadgroups,
                num_heads, 1, 1, 256, 1, 1,
            );

            MetalBackend::wrap(out_buf, vec![1, total_dim], TensorDtype::F32)
        };

        let result_data = metal.download(&result);
        // Slightly higher tolerance due to F16 quantization
        assert_vecs_close(
            result_data.as_f32(), &expected, 0.02,
            "grouped_attn_decode_f16",
        );
    }

    #[test]
    fn test_grouped_attn_decode_f16_gqa() {
        // GQA test with F16: 4 Q heads, 2 KV heads
        let metal = backend();
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let total_len = 4;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();

        let total_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q_data: Vec<f32> = (0..total_dim)
            .map(|i| ((i % 7) as f32) * 0.15 - 0.5)
            .collect();
        let k_data: Vec<f32> = (0..total_len * kv_dim)
            .map(|i| ((i % 11) as f32) * 0.08 - 0.2)
            .collect();
        let v_data: Vec<f32> = (0..total_len * kv_dim)
            .map(|i| ((i % 5) as f32) * 0.12)
            .collect();

        // CPU reference
        let mut expected = vec![0.0f32; total_dim];
        for h in 0..num_heads {
            let kv_head = h * num_kv_heads / num_heads;
            let kv_off = kv_head * head_dim;

            let mut scores: Vec<f32> = Vec::new();
            for j in 0..total_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_data[h * head_dim + d] * k_data[j * kv_dim + kv_off + d];
                }
                scores.push(dot * attn_scale);
            }

            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..total_len {
                    acc += probs[j] * v_data[j * kv_dim + kv_off + d];
                }
                expected[h * head_dim + d] = acc;
            }
        }

        let q_gpu = metal.upload(&Tensor::new(vec![1, total_dim], q_data));

        // Convert K, V to F16
        let k_f32 = metal.upload(&Tensor::new(vec![total_len, kv_dim], k_data));
        let v_f32 = metal.upload(&Tensor::new(vec![total_len, kv_dim], v_data));
        let k_count = total_len * kv_dim;
        let v_count = total_len * kv_dim;

        let k_f16 = unsafe {
            let mb = metal.create_buffer_empty(k_count * 2);
            DeviceTensor::from_gpu(vec![total_len, kv_dim], TensorDtype::F16, Box::new(mb))
        };
        let v_f16 = unsafe {
            let mb = metal.create_buffer_empty(v_count * 2);
            DeviceTensor::from_gpu(vec![total_len, kv_dim], TensorDtype::F16, Box::new(mb))
        };

        // Copy F32 → F16
        unsafe {
            let enc = metal.ensure_encoder(metal.pso_copy_f32_to_f16);
            metal.set_buffer(enc, MetalBackend::buf_id(&k_f32), 0);
            metal.set_buffer(enc, MetalBackend::buf_id(&k_f16), 1);
            metal.set_u32(enc, k_count as u32, 2);
            metal.set_u32(enc, 0u32, 3);
            ffi::msg_send_dispatch(
                enc, metal.sels.dispatch_threadgroups,
                (k_count + 255) / 256, 1, 1, 256, 1, 1,
            );
        }
        unsafe {
            let enc = metal.ensure_encoder(metal.pso_copy_f32_to_f16);
            metal.set_buffer(enc, MetalBackend::buf_id(&v_f32), 0);
            metal.set_buffer(enc, MetalBackend::buf_id(&v_f16), 1);
            metal.set_u32(enc, v_count as u32, 2);
            metal.set_u32(enc, 0u32, 3);
            ffi::msg_send_dispatch(
                enc, metal.sels.dispatch_threadgroups,
                (v_count + 255) / 256, 1, 1, 256, 1, 1,
            );
        }

        // Dispatch attention
        let out_bytes = total_dim * std::mem::size_of::<f32>();
        let result = unsafe {
            let out_buf = metal.create_buffer_empty(out_bytes);
            let enc = metal.ensure_encoder(metal.pso_grouped_attn_decode_f16);
            metal.set_buffer(enc, MetalBackend::buf_id(&q_gpu), 0);
            metal.set_buffer(enc, MetalBackend::buf_id(&k_f16), 1);
            metal.set_buffer(enc, MetalBackend::buf_id(&v_f16), 2);
            metal.set_buffer(enc, out_buf.buffer, 3);
            metal.set_u32(enc, num_heads as u32, 4);
            metal.set_u32(enc, num_kv_heads as u32, 5);
            metal.set_u32(enc, head_dim as u32, 6);
            metal.set_u32(enc, total_len as u32, 7);
            metal.set_f32(enc, attn_scale, 8);
            metal.set_f32(enc, 0.0f32, 9);
            ffi::msg_send_dispatch(
                enc, metal.sels.dispatch_threadgroups,
                num_heads, 1, 1, 256, 1, 1,
            );
            MetalBackend::wrap(out_buf, vec![1, total_dim], TensorDtype::F32)
        };

        let result_data = metal.download(&result);
        assert_vecs_close(
            result_data.as_f32(), &expected, 0.02,
            "grouped_attn_decode_f16_gqa",
        );
    }
}
