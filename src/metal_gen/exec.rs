//! Tight Metal encoding loop for single-token decode.
//!
//! Encodes ALL decode ops into a single command buffer with concurrent dispatch.
//! No allocation, no mutex, selective barriers.

use crate::backend::metal::ffi::*;

use super::graph::{BufferRef, DecodeGraph, DispatchDims, ParamValue, PrefillGraph};
use super::pool::BufferPool;

/// `MTLBarrierScopeBuffers` — ensures all buffer writes from prior dispatches
/// are visible to subsequent dispatches within the same encoder.
const MTL_BARRIER_SCOPE_BUFFERS: NSUInteger = 1;

/// Dynamic parameters that change per decode token.
pub(crate) struct DynamicParams {
    pub pos: usize,
    pub token_id: u32,
    pub kv_dim: usize,
}

/// Metal resources created once at engine init.
///
/// Holds the device, command queue, pre-registered selectors, and PSOs
/// indexed by `PsoRef`.
pub(crate) struct MetalResources {
    pub device: Id,
    pub command_queue: Id,
    pub sels: Selectors,
    /// PSOs indexed by `PsoRef as usize`.
    pub psos: Vec<Id>,
}

// Metal resources are safe to share across threads (we synchronize with
// waitUntilCompleted before reading results).
unsafe impl Send for MetalResources {}
unsafe impl Sync for MetalResources {}

impl MetalResources {
    /// Build MetalResources by compiling the MSL source and creating PSOs.
    ///
    /// # Safety
    /// Must be called after the Objective-C runtime is initialized.
    pub(crate) unsafe fn new(
        device: Id,
        command_queue: Id,
    ) -> Result<Self, crate::error::InferenceError> {
        let sels = Selectors::new();

        // Compile MSL
        let source = ns_string(crate::backend::metal::kernels::MSL_SOURCE);
        let mut error: Id = NIL;
        let library = msg_send_id_id_id_id(
            device,
            sels.new_library_with_source,
            source,
            NIL,
            &mut error as *mut Id as Id,
        );
        if library == NIL {
            let desc = obj_description(error);
            return Err(crate::error::InferenceError::Backend(format!(
                "Metal MSL compile error: {}", desc
            )));
        }

        // Create PSOs in PsoRef order
        let kernel_names = [
            "layer_norm",              // LayerNorm = 0
            "rms_norm",                // RmsNorm = 1
            "quantized_matmul_q8_0",   // QuantizedMatmulQ8_0 = 2
            "quantized_matmul_q4_0",   // QuantizedMatmulQ4_0 = 3
            "quantized_matmul_q4_k",   // QuantizedMatmulQ4K = 4
            "quantized_matmul_q5_k",   // QuantizedMatmulQ5K = 5
            "quantized_matmul_q6_k",   // QuantizedMatmulQ6K = 6
            "gemm_transpose",          // MatmulTranspose = 7
            "add_tensor",              // AddTensor = 8
            "add_bias",                // AddBias = 9
            "gelu",                    // Gelu = 10
            "silu",                    // Silu = 11
            "swiglu",                  // SwiGlu = 12
            "geglu",                   // GeGlu = 13
            "rope_norm",               // RopeNorm = 14
            "rope_neox",               // RopeNeox = 15
            "embedding_lookup",        // EmbeddingLookup = 16
            "grouped_attn_decode",     // GroupedAttnDecode = 17
            "copy_buffer",             // CopyBuffer = 18
            "scale_kernel",            // ScaleKernel = 19
            // Phase 1: F16 KV cache + online softmax
            "copy_f32_to_f16",               // CopyF32ToF16 = 20
            "grouped_attn_decode_f16",       // GroupedAttnDecodeF16 = 21
            // Phase 2: Fused matmul+bias
            "quantized_matmul_bias_q8_0",    // QuantizedMatmulBiasQ8_0 = 22
            "quantized_matmul_bias_q4_0",    // QuantizedMatmulBiasQ4_0 = 23
            "quantized_matmul_bias_q4_k",    // QuantizedMatmulBiasQ4K = 24
            "quantized_matmul_bias_q5_k",    // QuantizedMatmulBiasQ5K = 25
            "quantized_matmul_bias_q6_k",    // QuantizedMatmulBiasQ6K = 26
            "gemm_transpose_bias",           // MatmulTransposeBias = 27
            "batched_causal_attention",      // BatchedCausalAttention = 28
            "batched_causal_attention_f16",  // BatchedCausalAttentionF16 = 29
        ];

        let mut psos = Vec::with_capacity(kernel_names.len());
        for name in &kernel_names {
            let ns_name = ns_string(name);
            let func = msg_send_id_id(library, sels.new_function_with_name, ns_name);
            if func == NIL {
                msg_send_void(library, sels.release);
                return Err(crate::error::InferenceError::Backend(format!(
                    "Metal kernel function '{}' not found", name
                )));
            }
            let mut pso_error: Id = NIL;
            let pso = msg_send_id_id_err(device, sels.new_compute_pipeline, func, &mut pso_error);
            msg_send_void(func, sels.release);
            if pso == NIL {
                let desc = obj_description(pso_error);
                // Clean up already-created PSOs
                for p in &psos {
                    msg_send_void(*p, sels.release);
                }
                msg_send_void(library, sels.release);
                return Err(crate::error::InferenceError::Backend(format!(
                    "Metal PSO creation failed for '{}': {}", name, desc
                )));
            }
            psos.push(pso);
        }

        msg_send_void(library, sels.release);

        Ok(Self { device, command_queue, sels, psos })
    }
}

impl Drop for MetalResources {
    fn drop(&mut self) {
        unsafe {
            for &pso in &self.psos {
                if pso != NIL {
                    msg_send_void(pso, self.sels.release);
                }
            }
            // Note: device and command_queue are owned by the engine,
            // which releases them separately.
        }
    }
}

/// Encode all decode ops into a single command buffer and execute.
///
/// Returns the logits as a Vec<f32>.
///
/// # Safety
/// All buffer pointers (pool, weight_bufs, kv_bufs) must be valid Metal buffers.
/// The graph must have been built for the same model config.
pub(crate) unsafe fn encode_decode_token(
    graph: &DecodeGraph,
    pool: &BufferPool,
    res: &MetalResources,
    weight_bufs: &[Id],
    kv_bufs: &[Id],
    dynamic: &DynamicParams,
    token_id_buf: Id,
    position_id_buf: Id,
) -> Vec<f32> {
    // 1. Create command buffer
    let cmd = msg_send_id(res.command_queue, res.sels.command_buffer);
    debug_assert!(!cmd.is_null(), "command_buffer returned nil");

    // 2. Create compute command encoder (serial — concurrent dispatch type
    //    requires MTLComputeCommandEncoder created with dispatchType:Concurrent,
    //    which needs a different selector. For now, use the standard encoder
    //    with explicit barriers for correctness, then optimize later.)
    let enc = msg_send_id(cmd, res.sels.compute_command_encoder);
    debug_assert!(!enc.is_null(), "compute_command_encoder returned nil");

    // 3. Pre-compute dynamic values
    let pos = dynamic.pos as u32;
    let total_len = (dynamic.pos + 1) as u32;
    let cache_row_offset = dynamic.pos as u32 * dynamic.kv_dim as u32;

    // Write the token ID into the pre-allocated token_id buffer
    let token_ptr = msg_send_ptr(token_id_buf, res.sels.contents) as *mut u32;
    *token_ptr = dynamic.token_id;

    // Write the position ID into the pre-allocated position_id buffer
    let pos_ptr = msg_send_ptr(position_id_buf, res.sels.contents) as *mut u32;
    *pos_ptr = pos;

    // 4. Build a barrier set for O(1) lookup
    let mut barrier_set = vec![false; graph.ops.len()];
    for &idx in &graph.barriers {
        if idx < barrier_set.len() {
            barrier_set[idx] = true;
        }
    }

    // 5. Tight encoding loop
    for (i, op) in graph.ops.iter().enumerate() {
        // Insert barrier if needed
        if barrier_set[i] {
            msg_send_void_nsuint(
                enc,
                res.sels.memory_barrier_with_scope,
                MTL_BARRIER_SCOPE_BUFFERS,
            );
        }

        // Set PSO
        let pso = res.psos[op.pso as usize];
        msg_send_void_id(enc, res.sels.set_compute_pipeline, pso);

        // Bind buffers
        for &(ref buf_ref, binding, offset) in &op.bindings {
            let buf_id = resolve_buffer(buf_ref, pool, weight_bufs, kv_bufs);
            msg_send_set_buffer(enc, res.sels.set_buffer, buf_id, offset as usize, binding as usize);
        }

        // Bind params
        for &(ref param, binding) in &op.params {
            match param {
                ParamValue::U32(v) => {
                    let bytes = v.to_ne_bytes();
                    msg_send_set_bytes(enc, res.sels.set_bytes, bytes.as_ptr(), 4, binding as usize);
                }
                ParamValue::F32(v) => {
                    let bytes = v.to_ne_bytes();
                    msg_send_set_bytes(enc, res.sels.set_bytes, bytes.as_ptr(), 4, binding as usize);
                }
                ParamValue::TokenId => {
                    // Bind the token_id buffer (single u32) as a buffer
                    msg_send_set_buffer(enc, res.sels.set_buffer, token_id_buf, 0, binding as usize);
                }
                ParamValue::PositionIdBuffer => {
                    // Bind the position_id buffer (single u32) as a buffer
                    msg_send_set_buffer(enc, res.sels.set_buffer, position_id_buf, 0, binding as usize);
                }
                ParamValue::PositionId => {
                    let bytes = pos.to_ne_bytes();
                    msg_send_set_bytes(enc, res.sels.set_bytes, bytes.as_ptr(), 4, binding as usize);
                }
                ParamValue::TotalLen => {
                    let bytes = total_len.to_ne_bytes();
                    msg_send_set_bytes(enc, res.sels.set_bytes, bytes.as_ptr(), 4, binding as usize);
                }
                ParamValue::CacheRowOffset => {
                    let bytes = cache_row_offset.to_ne_bytes();
                    msg_send_set_bytes(enc, res.sels.set_bytes, bytes.as_ptr(), 4, binding as usize);
                }
                // Prefill-only variants — should not appear in decode graphs
                ParamValue::PrefillTokenIds | ParamValue::PrefillPositionIds
                | ParamValue::PrefillNTokens | ParamValue::PrefillTotalLen
                | ParamValue::PrefillPosOffset => {
                    debug_assert!(false, "prefill-only ParamValue in decode graph");
                }
            }
        }

        // Dispatch
        match op.dispatch {
            DispatchDims::D1 { count, threads } => {
                let groups = (count + threads - 1) / threads;
                msg_send_dispatch(
                    enc, res.sels.dispatch_threadgroups,
                    groups as usize, 1, 1,
                    threads as usize, 1, 1,
                );
            }
            DispatchDims::D2 { gx, gy, tx, ty } => {
                msg_send_dispatch(
                    enc, res.sels.dispatch_threadgroups,
                    gx as usize, gy as usize, 1,
                    tx as usize, ty as usize, 1,
                );
            }
            DispatchDims::D3 { gx, gy, gz, tx, ty, tz } => {
                msg_send_dispatch(
                    enc, res.sels.dispatch_threadgroups,
                    gx as usize, gy as usize, gz as usize,
                    tx as usize, ty as usize, tz as usize,
                );
            }
            DispatchDims::Rows { num_rows, threads_per_group } => {
                msg_send_dispatch(
                    enc, res.sels.dispatch_threadgroups,
                    num_rows as usize, 1, 1,
                    threads_per_group as usize, 1, 1,
                );
            }
            DispatchDims::Fixed { gx, gy, gz, tx, ty, tz } => {
                msg_send_dispatch(
                    enc, res.sels.dispatch_threadgroups,
                    gx as usize, gy as usize, gz as usize,
                    tx as usize, ty as usize, tz as usize,
                );
            }
        }
    }

    // 6. End encoding, commit, wait
    msg_send_void(enc, res.sels.end_encoding);
    msg_send_void(cmd, res.sels.commit);
    msg_send_void(cmd, res.sels.wait_until_completed);

    // 7. Read logits from pool
    let logits_buf = pool.get(graph.logits_slot);
    let ptr = msg_send_ptr(logits_buf, res.sels.contents) as *const f32;
    std::slice::from_raw_parts(ptr, graph.logits_count).to_vec()
}

/// Dynamic parameters for multi-token prefill.
pub(crate) struct PrefillParams {
    pub n_tokens: usize,
    pub pos_offset: usize,
}

/// Encode all prefill ops into a single command buffer and execute.
///
/// Returns the logits as a Vec<f32>.
///
/// # Safety
/// All buffer pointers must be valid Metal buffers.
pub(crate) unsafe fn encode_prefill(
    graph: &PrefillGraph,
    pool: &BufferPool,
    res: &MetalResources,
    weight_bufs: &[Id],
    kv_bufs: &[Id],
    params: &PrefillParams,
    token_ids_buf: Id,     // M u32s
    position_ids_buf: Id,  // M u32s
) -> Vec<f32> {
    // 1. Create command buffer
    let cmd = msg_send_id(res.command_queue, res.sels.command_buffer);
    debug_assert!(!cmd.is_null(), "command_buffer returned nil");

    // 2. Create compute command encoder
    let enc = msg_send_id(cmd, res.sels.compute_command_encoder);
    debug_assert!(!enc.is_null(), "compute_command_encoder returned nil");

    // 3. Pre-compute dynamic values
    let n_tokens = params.n_tokens as u32;
    let total_len = (params.pos_offset + params.n_tokens) as u32;
    let pos_offset = params.pos_offset as u32;

    // 4. Build barrier set for O(1) lookup
    let mut barrier_set = vec![false; graph.ops.len()];
    for &idx in &graph.barriers {
        if idx < barrier_set.len() {
            barrier_set[idx] = true;
        }
    }

    // 5. Tight encoding loop
    for (i, op) in graph.ops.iter().enumerate() {
        // Insert barrier if needed
        if barrier_set[i] {
            msg_send_void_nsuint(
                enc,
                res.sels.memory_barrier_with_scope,
                MTL_BARRIER_SCOPE_BUFFERS,
            );
        }

        // Set PSO
        let pso = res.psos[op.pso as usize];
        msg_send_void_id(enc, res.sels.set_compute_pipeline, pso);

        // Bind buffers (with byte offset)
        for &(ref buf_ref, binding, offset) in &op.bindings {
            let buf_id = resolve_buffer(buf_ref, pool, weight_bufs, kv_bufs);
            msg_send_set_buffer(enc, res.sels.set_buffer, buf_id, offset as usize, binding as usize);
        }

        // Bind params
        for &(ref param, binding) in &op.params {
            match param {
                ParamValue::U32(v) => {
                    let bytes = v.to_ne_bytes();
                    msg_send_set_bytes(enc, res.sels.set_bytes, bytes.as_ptr(), 4, binding as usize);
                }
                ParamValue::F32(v) => {
                    let bytes = v.to_ne_bytes();
                    msg_send_set_bytes(enc, res.sels.set_bytes, bytes.as_ptr(), 4, binding as usize);
                }
                ParamValue::PrefillTokenIds => {
                    msg_send_set_buffer(enc, res.sels.set_buffer, token_ids_buf, 0, binding as usize);
                }
                ParamValue::PrefillPositionIds => {
                    msg_send_set_buffer(enc, res.sels.set_buffer, position_ids_buf, 0, binding as usize);
                }
                ParamValue::PrefillNTokens => {
                    let bytes = n_tokens.to_ne_bytes();
                    msg_send_set_bytes(enc, res.sels.set_bytes, bytes.as_ptr(), 4, binding as usize);
                }
                ParamValue::PrefillTotalLen => {
                    let bytes = total_len.to_ne_bytes();
                    msg_send_set_bytes(enc, res.sels.set_bytes, bytes.as_ptr(), 4, binding as usize);
                }
                ParamValue::PrefillPosOffset => {
                    let bytes = pos_offset.to_ne_bytes();
                    msg_send_set_bytes(enc, res.sels.set_bytes, bytes.as_ptr(), 4, binding as usize);
                }
                // Decode-only variants — should not appear in prefill graphs
                ParamValue::TokenId | ParamValue::PositionId
                | ParamValue::PositionIdBuffer | ParamValue::TotalLen
                | ParamValue::CacheRowOffset => {
                    debug_assert!(false, "decode-only ParamValue in prefill graph");
                }
            }
        }

        // Dispatch
        match op.dispatch {
            DispatchDims::D1 { count, threads } => {
                let groups = (count + threads - 1) / threads;
                msg_send_dispatch(
                    enc, res.sels.dispatch_threadgroups,
                    groups as usize, 1, 1,
                    threads as usize, 1, 1,
                );
            }
            DispatchDims::D2 { gx, gy, tx, ty } => {
                msg_send_dispatch(
                    enc, res.sels.dispatch_threadgroups,
                    gx as usize, gy as usize, 1,
                    tx as usize, ty as usize, 1,
                );
            }
            DispatchDims::D3 { gx, gy, gz, tx, ty, tz } => {
                msg_send_dispatch(
                    enc, res.sels.dispatch_threadgroups,
                    gx as usize, gy as usize, gz as usize,
                    tx as usize, ty as usize, tz as usize,
                );
            }
            DispatchDims::Rows { num_rows, threads_per_group } => {
                msg_send_dispatch(
                    enc, res.sels.dispatch_threadgroups,
                    num_rows as usize, 1, 1,
                    threads_per_group as usize, 1, 1,
                );
            }
            DispatchDims::Fixed { gx, gy, gz, tx, ty, tz } => {
                msg_send_dispatch(
                    enc, res.sels.dispatch_threadgroups,
                    gx as usize, gy as usize, gz as usize,
                    tx as usize, ty as usize, tz as usize,
                );
            }
        }
    }

    // 6. End encoding, commit, wait
    msg_send_void(enc, res.sels.end_encoding);
    msg_send_void(cmd, res.sels.commit);
    msg_send_void(cmd, res.sels.wait_until_completed);

    // 7. Read logits from pool
    let logits_buf = pool.get(graph.logits_slot);
    let ptr = msg_send_ptr(logits_buf, res.sels.contents) as *const f32;
    std::slice::from_raw_parts(ptr, graph.logits_count).to_vec()
}

/// Resolve a BufferRef to a raw MTLBuffer Id.
#[inline]
unsafe fn resolve_buffer(
    buf_ref: &BufferRef,
    pool: &BufferPool,
    weight_bufs: &[Id],
    kv_bufs: &[Id],
) -> Id {
    match buf_ref {
        BufferRef::Pool(slot) => pool.get(*slot),
        BufferRef::Weight(idx) => weight_bufs[*idx as usize],
        BufferRef::KvCache(idx) => kv_bufs[*idx as usize],
    }
}
