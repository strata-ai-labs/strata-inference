//! Decode operation list builder and static barrier analysis.
//!
//! Builds a flat `Vec<DecodeOp>` describing every GPU dispatch for one decode
//! token. Barrier positions are computed statically via buffer conflict analysis.

use crate::model::config::{Activation, ModelConfig, NormType, PositionType};
use crate::model::weights::ModelWeights;
use crate::tensor::TensorDtype;

/// Index into the flat buffer pool. Assigned at graph build time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BufferSlot(pub u16);

// ---------------------------------------------------------------------------
// Buffer and PSO references
// ---------------------------------------------------------------------------

/// Identifies a Metal buffer for binding and conflict tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BufferRef {
    /// Intermediate from the pre-allocated pool.
    Pool(BufferSlot),
    /// Model weight buffer (read-only during decode). Index is flat weight ordinal.
    Weight(u16),
    /// KV cache buffer: layer * 2 + {0=K, 1=V}.
    KvCache(u16),
}

/// Which PSO to use, resolved to a raw Id at encode time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[allow(dead_code)]
pub(crate) enum PsoRef {
    LayerNorm = 0,
    RmsNorm,
    QuantizedMatmulQ8_0,
    QuantizedMatmulQ4_0,
    QuantizedMatmulQ4K,
    QuantizedMatmulQ5K,
    QuantizedMatmulQ6K,
    MatmulTranspose,
    AddTensor,
    AddBias,
    Gelu,
    Silu,
    SwiGlu,
    GeGlu,
    RopeNorm,
    RopeNeox,
    EmbeddingLookup,
    GroupedAttnDecode,
    CopyBuffer,
    ScaleKernel,
    // Phase 1: F16 KV cache + online softmax
    CopyF32ToF16,
    GroupedAttnDecodeF16,
    // Phase 2: Fused matmul+bias
    QuantizedMatmulBiasQ8_0,
    QuantizedMatmulBiasQ4_0,
    QuantizedMatmulBiasQ4K,
    QuantizedMatmulBiasQ5K,
    QuantizedMatmulBiasQ6K,
    MatmulTransposeBias,
    // Phase 3: Batched causal attention for prefill
    BatchedCausalAttention,
    BatchedCausalAttentionF16,
    // Phase 4: Batched quantized matmul for prefill
    BatchedMatmulQ8_0,
    BatchedMatmulBiasQ8_0,
    BatchedMatmulQ4_0,
    BatchedMatmulBiasQ4_0,
    BatchedMatmulQ4K,
    BatchedMatmulBiasQ4K,
    BatchedMatmulQ5K,
    BatchedMatmulBiasQ5K,
    BatchedMatmulQ6K,
    BatchedMatmulBiasQ6K,
    // Phase 5: RoPE with per-dimension frequency factors (LongRoPE)
    RopeNeoxFactors,
}

impl PsoRef {
    /// Human-readable kernel name for profiling output.
    pub(crate) fn name(self) -> &'static str {
        match self {
            PsoRef::LayerNorm => "LayerNorm",
            PsoRef::RmsNorm => "RmsNorm",
            PsoRef::QuantizedMatmulQ8_0 => "QuantizedMatmulQ8_0",
            PsoRef::QuantizedMatmulQ4_0 => "QuantizedMatmulQ4_0",
            PsoRef::QuantizedMatmulQ4K => "QuantizedMatmulQ4K",
            PsoRef::QuantizedMatmulQ5K => "QuantizedMatmulQ5K",
            PsoRef::QuantizedMatmulQ6K => "QuantizedMatmulQ6K",
            PsoRef::MatmulTranspose => "MatmulTranspose",
            PsoRef::AddTensor => "AddTensor",
            PsoRef::AddBias => "AddBias",
            PsoRef::Gelu => "Gelu",
            PsoRef::Silu => "Silu",
            PsoRef::SwiGlu => "SwiGlu",
            PsoRef::GeGlu => "GeGlu",
            PsoRef::RopeNorm => "RopeNorm",
            PsoRef::RopeNeox => "RopeNeox",
            PsoRef::EmbeddingLookup => "EmbeddingLookup",
            PsoRef::GroupedAttnDecode => "GroupedAttnDecode",
            PsoRef::CopyBuffer => "CopyBuffer",
            PsoRef::ScaleKernel => "ScaleKernel",
            PsoRef::CopyF32ToF16 => "CopyF32ToF16",
            PsoRef::GroupedAttnDecodeF16 => "GroupedAttnDecodeF16",
            PsoRef::QuantizedMatmulBiasQ8_0 => "QuantizedMatmulBiasQ8_0",
            PsoRef::QuantizedMatmulBiasQ4_0 => "QuantizedMatmulBiasQ4_0",
            PsoRef::QuantizedMatmulBiasQ4K => "QuantizedMatmulBiasQ4K",
            PsoRef::QuantizedMatmulBiasQ5K => "QuantizedMatmulBiasQ5K",
            PsoRef::QuantizedMatmulBiasQ6K => "QuantizedMatmulBiasQ6K",
            PsoRef::MatmulTransposeBias => "MatmulTransposeBias",
            PsoRef::BatchedCausalAttention => "BatchedCausalAttention",
            PsoRef::BatchedCausalAttentionF16 => "BatchedCausalAttentionF16",
            PsoRef::BatchedMatmulQ8_0 => "BatchedMatmulQ8_0",
            PsoRef::BatchedMatmulBiasQ8_0 => "BatchedMatmulBiasQ8_0",
            PsoRef::BatchedMatmulQ4_0 => "BatchedMatmulQ4_0",
            PsoRef::BatchedMatmulBiasQ4_0 => "BatchedMatmulBiasQ4_0",
            PsoRef::BatchedMatmulQ4K => "BatchedMatmulQ4K",
            PsoRef::BatchedMatmulBiasQ4K => "BatchedMatmulBiasQ4K",
            PsoRef::BatchedMatmulQ5K => "BatchedMatmulQ5K",
            PsoRef::BatchedMatmulBiasQ5K => "BatchedMatmulBiasQ5K",
            PsoRef::BatchedMatmulQ6K => "BatchedMatmulQ6K",
            PsoRef::BatchedMatmulBiasQ6K => "BatchedMatmulBiasQ6K",
            PsoRef::RopeNeoxFactors => "RopeNeoxFactors",
        }
    }
}

/// Parameter value: either fixed at build time or patched per-token.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ParamValue {
    U32(u32),
    F32(f32),
    /// Patched to current token ID at encode time (bound as a buffer).
    TokenId,
    /// Patched to current position at encode time (bound as bytes, u32 scalar).
    PositionId,
    /// Patched to current position at encode time (bound as a buffer, single u32).
    /// Used for embedding_lookup where the kernel expects an ID buffer.
    PositionIdBuffer,
    /// Patched to pos + 1 (total_len for attention).
    TotalLen,
    /// Patched to pos (for copy_buffer dest row offset).
    CacheRowOffset,

    // --- Prefill-specific variants ---

    /// Buffer of M u32 token IDs (bound as a buffer).
    PrefillTokenIds,
    /// Buffer of M u32 position IDs [0..M) (bound as a buffer).
    PrefillPositionIds,
    /// M as u32 bytes (number of prefill tokens).
    PrefillNTokens,
    /// (pos_offset + M) as u32 bytes (total sequence length after prefill).
    PrefillTotalLen,
    /// pos_offset as u32 bytes (used for RoPE pos_offset).
    PrefillPosOffset,
}

// ---------------------------------------------------------------------------
// Dispatch configuration
// ---------------------------------------------------------------------------

/// How to compute the threadgroup grid for a dispatch.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) enum DispatchDims {
    /// 1D: `ceil(count / threads_per_group)` groups × 1 × 1, threads_per_group × 1 × 1.
    D1 { count: u32, threads: u32 },
    /// 2D: threadgroup counts and threads per threadgroup.
    D2 { gx: u32, gy: u32, tx: u32, ty: u32 },
    /// 3D: explicit threadgroups and threads.
    D3 { gx: u32, gy: u32, gz: u32, tx: u32, ty: u32, tz: u32 },
    /// One threadgroup per row, with a given number of threads.
    Rows { num_rows: u32, threads_per_group: u32 },
    /// Fixed groups and threads specified directly.
    Fixed { gx: u32, gy: u32, gz: u32, tx: u32, ty: u32, tz: u32 },
}

// ---------------------------------------------------------------------------
// DecodeOp — a single GPU dispatch
// ---------------------------------------------------------------------------

/// A single GPU dispatch operation in the decode graph.
pub(crate) struct DecodeOp {
    pub pso: PsoRef,
    /// (buffer_ref, binding_index, byte_offset) triples for buffer bindings.
    /// The byte_offset is passed to `setBuffer:offset:atIndex:` — always 0
    /// for decode ops, but used by prefill for row-loop quantized matmuls.
    pub bindings: Vec<(BufferRef, u8, u32)>,
    /// (value, binding_index) pairs for small parameter bindings.
    pub params: Vec<(ParamValue, u8)>,
    /// Dispatch dimensions.
    pub dispatch: DispatchDims,
    /// Buffers read by this op (for barrier analysis).
    pub reads: Vec<BufferRef>,
    /// Buffer written by this op, if any (for barrier analysis).
    pub writes: Option<BufferRef>,
}

// ---------------------------------------------------------------------------
// DecodeGraph — the complete decode plan
// ---------------------------------------------------------------------------

/// The complete decode graph: ops, barriers, and buffer layout.
pub(crate) struct DecodeGraph {
    pub ops: Vec<DecodeOp>,
    /// Op indices where a barrier must be inserted BEFORE the op.
    pub barriers: Vec<usize>,
    /// Byte size per BufferSlot.
    pub slot_sizes: Vec<usize>,
    /// Total number of pool slots.
    pub num_slots: u16,
    /// Which pool slot holds the final logits.
    pub logits_slot: BufferSlot,
    /// Number of logit values (vocab_size).
    pub logits_count: usize,
    /// Whether the KV cache uses F16 storage.
    pub kv_f16: bool,
}

// ---------------------------------------------------------------------------
// Weight indexing
// ---------------------------------------------------------------------------

/// Walk ModelWeights in a deterministic order, returning (weight_index, DeviceTensor)
/// pairs. The engine uses the same order to build its weight_buf_ids array.
pub(crate) fn weight_walk_order(weights: &ModelWeights) -> Vec<&crate::backend::DeviceTensor> {
    let mut out = Vec::new();

    // Global weights
    out.push(&weights.token_embedding);
    if let Some(ref w) = weights.position_embedding { out.push(w); }
    if let Some(ref w) = weights.output_norm_w { out.push(w); }
    if let Some(ref w) = weights.output_norm_b { out.push(w); }
    if let Some(ref w) = weights.output_projection { out.push(w); }
    // LongRoPE frequency factors (Phi-3.5, etc.)
    if let Some(ref w) = weights.rope_factors_short { out.push(w); }
    if let Some(ref w) = weights.rope_factors_long { out.push(w); }

    // Per-layer weights
    for layer in &weights.layers {
        if let Some(ref w) = layer.attn_norm_w { out.push(w); }
        if let Some(ref w) = layer.attn_norm_b { out.push(w); }
        out.push(&layer.attn_q);
        if let Some(ref w) = layer.attn_q_bias { out.push(w); }
        out.push(&layer.attn_k);
        if let Some(ref w) = layer.attn_k_bias { out.push(w); }
        out.push(&layer.attn_v);
        if let Some(ref w) = layer.attn_v_bias { out.push(w); }
        out.push(&layer.attn_output);
        if let Some(ref w) = layer.attn_output_bias { out.push(w); }
        if let Some(ref w) = layer.ffn_norm_w { out.push(w); }
        if let Some(ref w) = layer.ffn_norm_b { out.push(w); }
        out.push(&layer.ffn_up);
        if let Some(ref w) = layer.ffn_up_bias { out.push(w); }
        out.push(&layer.ffn_down);
        if let Some(ref w) = layer.ffn_down_bias { out.push(w); }
        if let Some(ref w) = layer.ffn_gate { out.push(w); }

        // Post-norm weights (BERT)
        if let Some(ref w) = layer.attn_output_norm_w { out.push(w); }
        if let Some(ref w) = layer.attn_output_norm_b { out.push(w); }
        if let Some(ref w) = layer.ffn_output_norm_w { out.push(w); }
        if let Some(ref w) = layer.ffn_output_norm_b { out.push(w); }

        // Per-head Q/K norms (Qwen3) and post-projection norms (GemmaEmbedding)
        if let Some(ref w) = layer.attn_q_norm_w { out.push(w); }
        if let Some(ref w) = layer.attn_k_norm_w { out.push(w); }
        if let Some(ref w) = layer.attn_post_norm_w { out.push(w); }
        if let Some(ref w) = layer.ffn_post_norm_w { out.push(w); }
    }

    out
}

// ---------------------------------------------------------------------------
// Graph builder
// ---------------------------------------------------------------------------

/// Internal builder state for constructing the decode graph.
struct GraphBuilder {
    ops: Vec<DecodeOp>,
    /// Tracks allocated pool slots and their byte sizes.
    slot_sizes: Vec<usize>,
    /// Current weight index (auto-incremented when looking up weights).
    weight_idx: u16,
}

impl GraphBuilder {
    fn new() -> Self {
        Self {
            ops: Vec::with_capacity(300),
            slot_sizes: Vec::new(),
            weight_idx: 0,
        }
    }

    /// Allocate a new buffer pool slot with the given byte size.
    fn alloc_slot(&mut self, bytes: usize) -> BufferSlot {
        let idx = self.slot_sizes.len();
        self.slot_sizes.push(bytes);
        BufferSlot(idx as u16)
    }

    /// Get the next weight index and advance.
    fn next_weight(&mut self) -> BufferRef {
        let idx = self.weight_idx;
        self.weight_idx += 1;
        BufferRef::Weight(idx)
    }

    /// Skip an optional weight (advances index only if present).
    fn next_weight_opt(&mut self, present: bool) -> Option<BufferRef> {
        if present {
            Some(self.next_weight())
        } else {
            None
        }
    }

    /// Emit an embedding_lookup op.
    ///
    /// `use_position_id`: if true, the kernel looks up the position index
    /// (patched at encode time) instead of the token ID.
    fn emit_embedding_lookup(
        &mut self,
        table: BufferRef,
        out: BufferSlot,
        hidden_size: usize,
        use_position_id: bool,
    ) {
        let id_param = if use_position_id {
            ParamValue::PositionIdBuffer
        } else {
            ParamValue::TokenId
        };
        self.ops.push(DecodeOp {
            pso: PsoRef::EmbeddingLookup,
            bindings: vec![
                (table, 0, 0),
                // binding 1 = ids buffer (set as buffer at encode time)
                (BufferRef::Pool(out), 2, 0),
            ],
            params: vec![
                (id_param, 1), // token_id or position_id buffer
                (ParamValue::U32(hidden_size as u32), 3),
                (ParamValue::U32(1), 4), // num_tokens = 1
            ],
            dispatch: DispatchDims::D2 {
                gx: ((hidden_size + 15) / 16) as u32,
                gy: 1, // 1 token
                tx: 16,
                ty: 16,
            },
            reads: vec![table],
            writes: Some(BufferRef::Pool(out)),
        });
    }

    /// Emit an add_tensor op: out = a + b (element-wise).
    fn emit_add(&mut self, a: BufferSlot, b: BufferSlot, out: BufferSlot, count: usize) {
        let threads = 256u32;
        let groups = ((count + 255) / 256) as u32;
        self.ops.push(DecodeOp {
            pso: PsoRef::AddTensor,
            bindings: vec![
                (BufferRef::Pool(a), 0, 0),
                (BufferRef::Pool(b), 1, 0),
                (BufferRef::Pool(out), 2, 0),
            ],
            params: vec![(ParamValue::U32(count as u32), 3)],
            dispatch: DispatchDims::Fixed { gx: groups, gy: 1, gz: 1, tx: threads, ty: 1, tz: 1 },
            reads: vec![BufferRef::Pool(a), BufferRef::Pool(b)],
            writes: Some(BufferRef::Pool(out)),
        });
    }

    /// Emit a layer_norm op.
    fn emit_layer_norm(
        &mut self,
        input: BufferSlot,
        weight: BufferRef,
        bias: BufferRef,
        out: BufferSlot,
        cols: usize,
        eps: f32,
    ) {
        let threads_per_group = (cols.next_power_of_two()).min(256) as u32;
        self.ops.push(DecodeOp {
            pso: PsoRef::LayerNorm,
            bindings: vec![
                (BufferRef::Pool(input), 0, 0),
                (weight, 1, 0),
                (bias, 2, 0),
                (BufferRef::Pool(out), 3, 0),
            ],
            params: vec![
                (ParamValue::U32(1), 4), // rows = 1
                (ParamValue::U32(cols as u32), 5),
                (ParamValue::F32(eps), 6),
            ],
            dispatch: DispatchDims::Rows { num_rows: 1, threads_per_group },
            reads: vec![BufferRef::Pool(input), weight, bias],
            writes: Some(BufferRef::Pool(out)),
        });
    }

    /// Emit an rms_norm op.
    fn emit_rms_norm(
        &mut self,
        input: BufferSlot,
        weight: BufferRef,
        out: BufferSlot,
        cols: usize,
        eps: f32,
    ) {
        let threads_per_group = (cols.next_power_of_two()).min(256) as u32;
        self.ops.push(DecodeOp {
            pso: PsoRef::RmsNorm,
            bindings: vec![
                (BufferRef::Pool(input), 0, 0),
                (weight, 1, 0),
                (BufferRef::Pool(out), 2, 0),
            ],
            params: vec![
                (ParamValue::U32(1), 3), // rows = 1
                (ParamValue::U32(cols as u32), 4),
                (ParamValue::F32(eps), 5),
            ],
            dispatch: DispatchDims::Rows { num_rows: 1, threads_per_group },
            reads: vec![BufferRef::Pool(input), weight],
            writes: Some(BufferRef::Pool(out)),
        });
    }

    /// Emit a quantized matmul for single-row decode (M=1).
    fn emit_qmatmul(
        &mut self,
        weight_ref: BufferRef,
        input: BufferSlot,
        out: BufferSlot,
        n: usize, // output dim
        k: usize, // input dim
        dtype: TensorDtype,
    ) {
        let (pso, threadgroups, threads) = match dtype {
            TensorDtype::Q8_0 => (PsoRef::QuantizedMatmulQ8_0, ((n + 7) / 8) as u32, 128u32),
            TensorDtype::Q4_0 => (PsoRef::QuantizedMatmulQ4_0, ((n + 7) / 8) as u32, 64u32),
            TensorDtype::Q4_K => (PsoRef::QuantizedMatmulQ4K, ((n + 3) / 4) as u32, 64u32),
            TensorDtype::Q5_K => (PsoRef::QuantizedMatmulQ5K, ((n + 3) / 4) as u32, 64u32),
            TensorDtype::Q6_K => (PsoRef::QuantizedMatmulQ6K, ((n + 3) / 4) as u32, 64u32),
            _ => panic!("unsupported quantized dtype for graph: {:?}", dtype),
        };
        self.ops.push(DecodeOp {
            pso,
            bindings: vec![
                (weight_ref, 0, 0),
                (BufferRef::Pool(input), 1, 0),
                (BufferRef::Pool(out), 2, 0),
            ],
            params: vec![
                (ParamValue::U32(n as u32), 3),
                (ParamValue::U32(k as u32), 4),
            ],
            dispatch: DispatchDims::Fixed { gx: threadgroups, gy: 1, gz: 1, tx: threads, ty: 1, tz: 1 },
            reads: vec![weight_ref, BufferRef::Pool(input)],
            writes: Some(BufferRef::Pool(out)),
        });
    }

    /// Emit an F32 matmul_transpose (for non-quantized weights, M=1).
    fn emit_matmul_transpose(
        &mut self,
        input: BufferSlot,
        weight_ref: BufferRef,
        out: BufferSlot,
        n: usize,
        k: usize,
    ) {
        let gx = ((n + 31) / 32) as u32;
        let gy = 1u32; // M = 1
        self.ops.push(DecodeOp {
            pso: PsoRef::MatmulTranspose,
            bindings: vec![
                (BufferRef::Pool(input), 0, 0),
                (weight_ref, 1, 0),
                (BufferRef::Pool(out), 2, 0),
            ],
            params: vec![
                (ParamValue::U32(1), 3), // M = 1
                (ParamValue::U32(k as u32), 4),
                (ParamValue::U32(n as u32), 5),
            ],
            dispatch: DispatchDims::Fixed { gx, gy, gz: 1, tx: 128, ty: 1, tz: 1 },
            reads: vec![BufferRef::Pool(input), weight_ref],
            writes: Some(BufferRef::Pool(out)),
        });
    }

    /// Emit a fused quantized matmul+bias for single-row decode (M=1).
    fn emit_qmatmul_bias(
        &mut self,
        weight_ref: BufferRef,
        input: BufferSlot,
        out: BufferSlot,
        bias_ref: BufferRef,
        n: usize,
        k: usize,
        dtype: TensorDtype,
    ) {
        let (pso, threadgroups, threads) = match dtype {
            TensorDtype::Q8_0 => (PsoRef::QuantizedMatmulBiasQ8_0, ((n + 7) / 8) as u32, 128u32),
            TensorDtype::Q4_0 => (PsoRef::QuantizedMatmulBiasQ4_0, ((n + 7) / 8) as u32, 64u32),
            TensorDtype::Q4_K => (PsoRef::QuantizedMatmulBiasQ4K, ((n + 3) / 4) as u32, 64u32),
            TensorDtype::Q5_K => (PsoRef::QuantizedMatmulBiasQ5K, ((n + 3) / 4) as u32, 64u32),
            TensorDtype::Q6_K => (PsoRef::QuantizedMatmulBiasQ6K, ((n + 3) / 4) as u32, 64u32),
            _ => panic!("unsupported quantized dtype for fused matmul+bias: {:?}", dtype),
        };
        self.ops.push(DecodeOp {
            pso,
            bindings: vec![
                (weight_ref, 0, 0),
                (BufferRef::Pool(input), 1, 0),
                (BufferRef::Pool(out), 2, 0),
                (bias_ref, 5, 0),
            ],
            params: vec![
                (ParamValue::U32(n as u32), 3),
                (ParamValue::U32(k as u32), 4),
            ],
            dispatch: DispatchDims::Fixed { gx: threadgroups, gy: 1, gz: 1, tx: threads, ty: 1, tz: 1 },
            reads: vec![weight_ref, BufferRef::Pool(input), bias_ref],
            writes: Some(BufferRef::Pool(out)),
        });
    }

    /// Emit a fused F32 matmul_transpose+bias (M=1).
    fn emit_matmul_transpose_bias(
        &mut self,
        input: BufferSlot,
        weight_ref: BufferRef,
        out: BufferSlot,
        bias_ref: BufferRef,
        n: usize,
        k: usize,
    ) {
        let gx = ((n + 31) / 32) as u32;
        let gy = 1u32; // M = 1
        self.ops.push(DecodeOp {
            pso: PsoRef::MatmulTransposeBias,
            bindings: vec![
                (BufferRef::Pool(input), 0, 0),
                (weight_ref, 1, 0),
                (BufferRef::Pool(out), 2, 0),
                (bias_ref, 6, 0),
            ],
            params: vec![
                (ParamValue::U32(1), 3), // M = 1
                (ParamValue::U32(k as u32), 4),
                (ParamValue::U32(n as u32), 5),
            ],
            dispatch: DispatchDims::Fixed { gx, gy, gz: 1, tx: 128, ty: 1, tz: 1 },
            reads: vec![BufferRef::Pool(input), weight_ref, bias_ref],
            writes: Some(BufferRef::Pool(out)),
        });
    }

    /// Emit a linear_forward: dispatches quantized_matmul or matmul_transpose
    /// based on weight dtype, with optional fused bias add.
    ///
    /// When bias is present, uses fused matmul+bias kernels to eliminate
    /// the separate add_bias dispatch.
    fn emit_linear(
        &mut self,
        input: BufferSlot,
        weight_ref: BufferRef,
        bias_ref: Option<BufferRef>,
        out: BufferSlot,
        n: usize, // output features
        k: usize, // input features
        dtype: TensorDtype,
    ) -> BufferSlot {
        if let Some(bias) = bias_ref {
            // Fused matmul+bias: single kernel, no separate add_bias dispatch
            match dtype {
                TensorDtype::F32 | TensorDtype::F16 => {
                    self.emit_matmul_transpose_bias(input, weight_ref, out, bias, n, k);
                }
                _ => {
                    self.emit_qmatmul_bias(weight_ref, input, out, bias, n, k, dtype);
                }
            }
            out
        } else {
            // No bias: use standard matmul
            match dtype {
                TensorDtype::F32 | TensorDtype::F16 => {
                    self.emit_matmul_transpose(input, weight_ref, out, n, k);
                }
                _ => {
                    self.emit_qmatmul(weight_ref, input, out, n, k, dtype);
                }
            }
            out
        }
    }

    /// Emit a GELU activation.
    fn emit_gelu(&mut self, input: BufferSlot, out: BufferSlot, count: usize) {
        let groups = ((count + 255) / 256) as u32;
        self.ops.push(DecodeOp {
            pso: PsoRef::Gelu,
            bindings: vec![
                (BufferRef::Pool(input), 0, 0),
                (BufferRef::Pool(out), 1, 0),
            ],
            params: vec![(ParamValue::U32(count as u32), 2)],
            dispatch: DispatchDims::Fixed { gx: groups, gy: 1, gz: 1, tx: 256, ty: 1, tz: 1 },
            reads: vec![BufferRef::Pool(input)],
            writes: Some(BufferRef::Pool(out)),
        });
    }

    /// Emit SwiGLU: silu(gate) * up.
    fn emit_swiglu(&mut self, gate: BufferSlot, up: BufferSlot, out: BufferSlot, count: usize) {
        let groups = ((count + 255) / 256) as u32;
        self.ops.push(DecodeOp {
            pso: PsoRef::SwiGlu,
            bindings: vec![
                (BufferRef::Pool(gate), 0, 0),
                (BufferRef::Pool(up), 1, 0),
                (BufferRef::Pool(out), 2, 0),
            ],
            params: vec![(ParamValue::U32(count as u32), 3)],
            dispatch: DispatchDims::Fixed { gx: groups, gy: 1, gz: 1, tx: 256, ty: 1, tz: 1 },
            reads: vec![BufferRef::Pool(gate), BufferRef::Pool(up)],
            writes: Some(BufferRef::Pool(out)),
        });
    }

    /// Emit GeGLU: gelu(gate) * up.
    fn emit_geglu(&mut self, gate: BufferSlot, up: BufferSlot, out: BufferSlot, count: usize) {
        let groups = ((count + 255) / 256) as u32;
        self.ops.push(DecodeOp {
            pso: PsoRef::GeGlu,
            bindings: vec![
                (BufferRef::Pool(gate), 0, 0),
                (BufferRef::Pool(up), 1, 0),
                (BufferRef::Pool(out), 2, 0),
            ],
            params: vec![(ParamValue::U32(count as u32), 3)],
            dispatch: DispatchDims::Fixed { gx: groups, gy: 1, gz: 1, tx: 256, ty: 1, tz: 1 },
            reads: vec![BufferRef::Pool(gate), BufferRef::Pool(up)],
            writes: Some(BufferRef::Pool(out)),
        });
    }

    /// Emit scale: out[i] = input[i] * factor.
    fn emit_scale(&mut self, input: BufferSlot, out: BufferSlot, count: usize, factor: f32) {
        let groups = ((count + 255) / 256) as u32;
        self.ops.push(DecodeOp {
            pso: PsoRef::ScaleKernel,
            bindings: vec![
                (BufferRef::Pool(input), 0, 0),
                (BufferRef::Pool(out), 1, 0),
            ],
            params: vec![
                (ParamValue::F32(factor), 2),
                (ParamValue::U32(count as u32), 3),
            ],
            dispatch: DispatchDims::Fixed { gx: groups, gy: 1, gz: 1, tx: 256, ty: 1, tz: 1 },
            reads: vec![BufferRef::Pool(input)],
            writes: Some(BufferRef::Pool(out)),
        });
    }

    /// Emit RoPE (norm or neox) for Q or K (single token, [1, n_heads * head_dim]).
    /// When `rope_factors` is Some, uses the `rope_neox_factors` kernel which multiplies
    /// per-dimension frequency factors into the RoPE computation (LongRoPE).
    /// `mscale` is the YaRN magnitude scaling factor (1.0 = no scaling).
    fn emit_rope(
        &mut self,
        neox: bool,
        input: BufferSlot,
        out: BufferSlot,
        n_heads: usize,
        head_dim: usize,
        rope_dim: usize,
        freq_base: f32,
        rope_factors: Option<BufferRef>,
        mscale: f32,
    ) {
        let pso = match (neox, rope_factors.is_some()) {
            (true, true) => PsoRef::RopeNeoxFactors,
            (true, false) => PsoRef::RopeNeox,
            (false, _) => PsoRef::RopeNorm, // factors not yet supported for norm variant
        };
        let half_rope = rope_dim / 2;
        let gx = ((half_rope + 15) / 16) as u32;
        let gy = ((n_heads + 15) / 16) as u32;
        let gz = 1u32; // seq_len = 1

        let mut bindings = vec![
            (BufferRef::Pool(input), 0, 0),
            (BufferRef::Pool(out), 1, 0),
        ];
        if let Some(factors_ref) = rope_factors {
            bindings.push((factors_ref, 8, 0));
        }

        let mut params = vec![
            (ParamValue::PositionId, 2), // pos_offset (patched at encode)
            (ParamValue::F32(freq_base), 3),
            (ParamValue::U32(head_dim as u32), 4),
            (ParamValue::U32(rope_dim as u32), 5),
            (ParamValue::U32(n_heads as u32), 6),
            (ParamValue::U32(1), 7), // seq_len = 1
        ];
        if rope_factors.is_some() {
            params.push((ParamValue::F32(mscale), 9));
        }

        let mut reads = vec![BufferRef::Pool(input)];
        if let Some(factors_ref) = rope_factors {
            reads.push(factors_ref);
        }

        self.ops.push(DecodeOp {
            pso,
            bindings,
            params,
            dispatch: DispatchDims::Fixed { gx, gy, gz, tx: 16, ty: 16, tz: 1 },
            reads,
            writes: Some(BufferRef::Pool(out)),
        });
    }

    /// Emit copy to KV cache at the current position.
    /// When `kv_f16` is true, emits `copy_f32_to_f16` kernel; otherwise `copy_buffer`.
    fn emit_copy_to_cache(
        &mut self,
        src: BufferSlot,
        kv_buf: BufferRef,
        kv_dim: usize,
        kv_f16: bool,
    ) {
        let count = kv_dim; // 1 row of kv_dim elements
        let groups = ((count + 255) / 256) as u32;
        let pso = if kv_f16 { PsoRef::CopyF32ToF16 } else { PsoRef::CopyBuffer };
        self.ops.push(DecodeOp {
            pso,
            bindings: vec![
                (BufferRef::Pool(src), 0, 0),
                (kv_buf, 1, 0),
            ],
            params: vec![
                (ParamValue::U32(count as u32), 2),
                (ParamValue::CacheRowOffset, 3), // dest_offset = pos * kv_dim (patched at encode)
            ],
            dispatch: DispatchDims::Fixed { gx: groups, gy: 1, gz: 1, tx: 256, ty: 1, tz: 1 },
            reads: vec![BufferRef::Pool(src)],
            writes: Some(kv_buf),
        });
    }

    /// Emit grouped_attention_decode (fused Q@K^T + softmax + @V).
    /// When `kv_f16` is true, uses the online softmax F16 variant.
    fn emit_grouped_attn_decode(
        &mut self,
        q: BufferSlot,
        k_cache: BufferRef,
        v_cache: BufferRef,
        out: BufferSlot,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        attn_scale: f32,
        softcap: f32,
        kv_f16: bool,
    ) {
        let pso = if kv_f16 { PsoRef::GroupedAttnDecodeF16 } else { PsoRef::GroupedAttnDecode };
        self.ops.push(DecodeOp {
            pso,
            bindings: vec![
                (BufferRef::Pool(q), 0, 0),
                (k_cache, 1, 0),
                (v_cache, 2, 0),
                (BufferRef::Pool(out), 3, 0),
            ],
            params: vec![
                (ParamValue::U32(num_heads as u32), 4),
                (ParamValue::U32(num_kv_heads as u32), 5),
                (ParamValue::U32(head_dim as u32), 6),
                (ParamValue::TotalLen, 7), // total_len = pos + 1 (patched at encode)
                (ParamValue::F32(attn_scale), 8),
                (ParamValue::F32(softcap), 9),
            ],
            dispatch: DispatchDims::Fixed {
                gx: num_heads as u32, gy: 1, gz: 1,
                tx: 256, ty: 1, tz: 1,
            },
            reads: vec![BufferRef::Pool(q), k_cache, v_cache],
            writes: Some(BufferRef::Pool(out)),
        });
    }

    /// Emit a normalize op (LayerNorm or RMSNorm) based on config.
    fn emit_normalize(
        &mut self,
        input: BufferSlot,
        weight: BufferRef,
        bias: Option<BufferRef>,
        out: BufferSlot,
        config: &ModelConfig,
    ) {
        match config.norm_type {
            NormType::LayerNorm => {
                let b = bias.expect("LayerNorm requires bias");
                self.emit_layer_norm(input, weight, b, out, config.hidden_size, config.norm_eps);
            }
            NormType::RMSNorm => {
                self.emit_rms_norm(input, weight, out, config.hidden_size, config.norm_eps);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DecodeGraph::build
// ---------------------------------------------------------------------------

impl DecodeGraph {
    /// Build the decode graph for single-token generation.
    ///
    /// This mirrors the logic in `model_forward_step` + `transformer_layer_forward_cached`
    /// + `ffn_forward` but emits `DecodeOp` entries instead of executing.
    ///
    /// When `kv_f16` is true, the graph emits `copy_f32_to_f16` for KV cache writes
    /// and `grouped_attn_decode_f16` (online softmax) for attention reads.
    pub(crate) fn build(config: &ModelConfig, weights: &ModelWeights, kv_f16: bool) -> Self {
        let mut b = GraphBuilder::new();

        let h = config.hidden_size;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let total_dim = config.num_heads * config.head_dim;
        let ffn_h = config.ffn_hidden;
        let f32_size = std::mem::size_of::<f32>();
        // LongRoPE mscale: the raw attn_factor is passed to the RoPE kernel as mscale,
        // which scales cos/sin for both Q and K.  This naturally produces mscale^2 in
        // the attention dot product, so we do NOT additionally modify attn_scale here.
        // (The 0.1*ln(f)+1.0 transform only applies to YaRN with ext_factor!=0.)
        let yarn_mscale = config.rope_scaling_attn_factor;
        let attn_scale = config.attn_scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());

        // ===================================================================
        // Index global weights (same order as weight_walk_order)
        // ===================================================================
        let token_emb = b.next_weight(); // token_embedding
        let pos_emb = if weights.position_embedding.is_some() {
            Some(b.next_weight())
        } else {
            None
        };
        let output_norm_w = b.next_weight_opt(weights.output_norm_w.is_some());
        let output_norm_b = b.next_weight_opt(weights.output_norm_b.is_some());
        let output_proj = b.next_weight_opt(weights.output_projection.is_some());
        // LongRoPE frequency factors — short factors used at graph level.
        // At runtime, MetalExecutor swaps in long factors when seq_len >= original_ctx.
        let rope_factors = b.next_weight_opt(weights.rope_factors_short.is_some());
        let rope_factors_long = b.next_weight_opt(weights.rope_factors_long.is_some());
        let _ = rope_factors_long; // consumed by next_weight_opt to maintain walk order

        // ===================================================================
        // Pre-layer: embedding lookup + position embedding + scaling
        // ===================================================================

        // The embedding_lookup kernel expects a token ID buffer.
        // We pass it as a special param binding (TokenId) that the encoder
        // will create as a tiny buffer with a single u32.
        let hidden_slot = b.alloc_slot(h * f32_size);
        b.emit_embedding_lookup(token_emb, hidden_slot, h, false);

        // Embedding scaling
        let mut current_hidden = hidden_slot;
        if (config.embedding_scale - 1.0).abs() > f32::EPSILON {
            let scaled = b.alloc_slot(h * f32_size);
            b.emit_scale(current_hidden, scaled, h, config.embedding_scale);
            current_hidden = scaled;
        }

        // Position embedding (Learned, GPT-2/BERT)
        if config.position_type == PositionType::Learned {
            if let Some(pos_emb_ref) = pos_emb {
                let pos_slot = b.alloc_slot(h * f32_size);
                b.emit_embedding_lookup(pos_emb_ref, pos_slot, h, true);

                let combined = b.alloc_slot(h * f32_size);
                b.emit_add(current_hidden, pos_slot, combined, h);
                current_hidden = combined;
            }
        }

        // ===================================================================
        // Per-layer transformer (cached single-token decode)
        // ===================================================================

        // Index per-layer weights and emit per-layer ops
        for layer_idx in 0..config.num_layers {
            let layer = &weights.layers[layer_idx];

            // Index this layer's weights (must match weight_walk_order)
            let attn_norm_w = b.next_weight_opt(layer.attn_norm_w.is_some());
            let attn_norm_b = b.next_weight_opt(layer.attn_norm_b.is_some());
            let w_q = b.next_weight();
            let q_bias = b.next_weight_opt(layer.attn_q_bias.is_some());
            let w_k = b.next_weight();
            let k_bias = b.next_weight_opt(layer.attn_k_bias.is_some());
            let w_v = b.next_weight();
            let v_bias = b.next_weight_opt(layer.attn_v_bias.is_some());
            let w_o = b.next_weight();
            let o_bias = b.next_weight_opt(layer.attn_output_bias.is_some());
            let ffn_norm_w = b.next_weight_opt(layer.ffn_norm_w.is_some());
            let ffn_norm_b = b.next_weight_opt(layer.ffn_norm_b.is_some());
            let w_up = b.next_weight();
            let up_bias = b.next_weight_opt(layer.ffn_up_bias.is_some());
            let w_down = b.next_weight();
            let down_bias = b.next_weight_opt(layer.ffn_down_bias.is_some());
            let w_gate = b.next_weight_opt(layer.ffn_gate.is_some());

            // Post-norm weights (BERT)
            let attn_output_norm_w = b.next_weight_opt(layer.attn_output_norm_w.is_some());
            let attn_output_norm_b = b.next_weight_opt(layer.attn_output_norm_b.is_some());
            let ffn_output_norm_w = b.next_weight_opt(layer.ffn_output_norm_w.is_some());
            let ffn_output_norm_b = b.next_weight_opt(layer.ffn_output_norm_b.is_some());

            // Per-head Q/K norms (Qwen3) and post-projection norms (GemmaEmbedding)
            // Indexed to keep weight walk order consistent; ops not yet emitted.
            let _attn_q_norm_w = b.next_weight_opt(layer.attn_q_norm_w.is_some());
            let _attn_k_norm_w = b.next_weight_opt(layer.attn_k_norm_w.is_some());
            let _attn_post_norm_w = b.next_weight_opt(layer.attn_post_norm_w.is_some());
            let _ffn_post_norm_w = b.next_weight_opt(layer.ffn_post_norm_w.is_some());

            // KV cache references for this layer
            let k_cache = BufferRef::KvCache((layer_idx * 2) as u16);
            let v_cache = BufferRef::KvCache((layer_idx * 2 + 1) as u16);

            // Weight dtypes for dispatching quantized vs F32 matmul
            let q_dtype = layer.attn_q.dtype();
            let k_dtype = layer.attn_k.dtype();
            let v_dtype = layer.attn_v.dtype();
            let o_dtype = layer.attn_output.dtype();
            let up_dtype = layer.ffn_up.dtype();
            let down_dtype = layer.ffn_down.dtype();
            let gate_dtype = layer.ffn_gate.as_ref().map(|w| w.dtype());

            if config.pre_norm {
                // ===========================================================
                // Pre-norm path (Gemma/LLaMA/GPT-2)
                // ===========================================================

                // 1. Pre-attention norm
                let normed = b.alloc_slot(h * f32_size);
                b.emit_normalize(
                    current_hidden,
                    attn_norm_w.expect("pre_norm requires attn_norm_w"),
                    attn_norm_b,
                    normed,
                    config,
                );

                // 2. QKV projections (all read normed, write to different outputs)
                let q_raw = b.alloc_slot(total_dim * f32_size);
                let q_out = b.emit_linear(normed, w_q, q_bias, q_raw, total_dim, h, q_dtype);

                let k_raw = b.alloc_slot(kv_dim * f32_size);
                let k_out = b.emit_linear(normed, w_k, k_bias, k_raw, kv_dim, h, k_dtype);

                let v_raw = b.alloc_slot(kv_dim * f32_size);
                let v_out = b.emit_linear(normed, w_v, v_bias, v_raw, kv_dim, h, v_dtype);

                // 3. RoPE (if configured) — applies to Q and K only
                let (q_final, k_final) = if config.position_type == PositionType::RoPE {
                    let q_rope = b.alloc_slot(total_dim * f32_size);
                    b.emit_rope(
                        config.rope_neox, q_out, q_rope,
                        config.num_heads, config.head_dim, config.rope_dim, config.rope_freq_base,
                        rope_factors, yarn_mscale,
                    );
                    let k_rope = b.alloc_slot(kv_dim * f32_size);
                    b.emit_rope(
                        config.rope_neox, k_out, k_rope,
                        config.num_kv_heads, config.head_dim, config.rope_dim, config.rope_freq_base,
                        rope_factors, yarn_mscale,
                    );
                    (q_rope, k_rope)
                } else {
                    (q_out, k_out)
                };

                // 4. Copy K, V into KV cache
                b.emit_copy_to_cache(k_final, k_cache, kv_dim, kv_f16);
                b.emit_copy_to_cache(v_out, v_cache, kv_dim, kv_f16);

                // 5. Fused grouped attention decode
                let attn_out = b.alloc_slot(total_dim * f32_size);
                b.emit_grouped_attn_decode(
                    q_final, k_cache, v_cache, attn_out,
                    config.num_heads, config.num_kv_heads, config.head_dim,
                    attn_scale, config.attn_logit_softcap, kv_f16,
                );

                // 6. Output projection + residual
                let proj_raw = b.alloc_slot(h * f32_size);
                let proj_out = b.emit_linear(attn_out, w_o, o_bias, proj_raw, h, total_dim, o_dtype);

                let residual = b.alloc_slot(h * f32_size);
                b.emit_add(current_hidden, proj_out, residual, h);

                // 7. Pre-FFN norm
                let normed2 = b.alloc_slot(h * f32_size);
                b.emit_normalize(
                    residual,
                    ffn_norm_w.expect("pre_norm requires ffn_norm_w"),
                    ffn_norm_b,
                    normed2,
                    config,
                );

                // 8. FFN
                let ffn_out = emit_ffn(
                    &mut b, normed2, w_up, up_bias, w_down, down_bias, w_gate,
                    config.activation, h, ffn_h,
                    up_dtype, down_dtype, gate_dtype,
                );

                // 9. Residual
                let layer_out = b.alloc_slot(h * f32_size);
                b.emit_add(residual, ffn_out, layer_out, h);
                current_hidden = layer_out;

            } else {
                // ===========================================================
                // Post-norm path (BERT — rare for generation but supported)
                // ===========================================================

                // 1. QKV projections (no pre-norm)
                let q_raw = b.alloc_slot(total_dim * f32_size);
                let q_out = b.emit_linear(current_hidden, w_q, q_bias, q_raw, total_dim, h, q_dtype);

                let k_raw = b.alloc_slot(kv_dim * f32_size);
                let k_out = b.emit_linear(current_hidden, w_k, k_bias, k_raw, kv_dim, h, k_dtype);

                let v_raw = b.alloc_slot(kv_dim * f32_size);
                let v_out = b.emit_linear(current_hidden, w_v, v_bias, v_raw, kv_dim, h, v_dtype);

                // 2. RoPE (unlikely for post-norm, but be general)
                let (q_final, k_final) = if config.position_type == PositionType::RoPE {
                    let q_rope = b.alloc_slot(total_dim * f32_size);
                    b.emit_rope(
                        config.rope_neox, q_out, q_rope,
                        config.num_heads, config.head_dim, config.rope_dim, config.rope_freq_base,
                        rope_factors, yarn_mscale,
                    );
                    let k_rope = b.alloc_slot(kv_dim * f32_size);
                    b.emit_rope(
                        config.rope_neox, k_out, k_rope,
                        config.num_kv_heads, config.head_dim, config.rope_dim, config.rope_freq_base,
                        rope_factors, yarn_mscale,
                    );
                    (q_rope, k_rope)
                } else {
                    (q_out, k_out)
                };

                // 3. Copy K, V into KV cache
                b.emit_copy_to_cache(k_final, k_cache, kv_dim, kv_f16);
                b.emit_copy_to_cache(v_out, v_cache, kv_dim, kv_f16);

                // 4. Fused grouped attention decode
                let attn_out = b.alloc_slot(total_dim * f32_size);
                b.emit_grouped_attn_decode(
                    q_final, k_cache, v_cache, attn_out,
                    config.num_heads, config.num_kv_heads, config.head_dim,
                    attn_scale, config.attn_logit_softcap, kv_f16,
                );

                // 5. Output projection + residual
                let proj_raw = b.alloc_slot(h * f32_size);
                let proj_out = b.emit_linear(attn_out, w_o, o_bias, proj_raw, h, total_dim, o_dtype);

                let mut residual_slot = b.alloc_slot(h * f32_size);
                b.emit_add(current_hidden, proj_out, residual_slot, h);

                // 6. Post-attention norm
                if let (Some(nw), Some(nb)) = (attn_output_norm_w, attn_output_norm_b) {
                    let normed = b.alloc_slot(h * f32_size);
                    b.emit_normalize(residual_slot, nw, Some(nb), normed, config);
                    residual_slot = normed;
                }

                // 7. FFN
                let ffn_out = emit_ffn(
                    &mut b, residual_slot, w_up, up_bias, w_down, down_bias, w_gate,
                    config.activation, h, ffn_h,
                    up_dtype, down_dtype, gate_dtype,
                );

                // 8. Residual + post-FFN norm
                let mut output = b.alloc_slot(h * f32_size);
                b.emit_add(residual_slot, ffn_out, output, h);

                if let (Some(nw), Some(nb)) = (ffn_output_norm_w, ffn_output_norm_b) {
                    let normed = b.alloc_slot(h * f32_size);
                    b.emit_normalize(output, nw, Some(nb), normed, config);
                    output = normed;
                }

                current_hidden = output;
            }
        }

        // ===================================================================
        // Post-layer: final norm + logits projection
        // ===================================================================

        if let Some(norm_w) = output_norm_w {
            let normed = b.alloc_slot(h * f32_size);
            b.emit_normalize(current_hidden, norm_w, output_norm_b, normed, config);
            current_hidden = normed;
        }

        // Logits projection: [1, hidden_size] x [vocab_size, hidden_size]^T -> [1, vocab_size]
        let proj_weight = output_proj.unwrap_or(token_emb); // tied embeddings fallback
        let proj_dtype = weights.output_projection.as_ref()
            .map(|w| w.dtype())
            .unwrap_or(weights.token_embedding.dtype());

        let logits_raw = b.alloc_slot(config.vocab_size * f32_size);
        let logits_slot = b.emit_linear(
            current_hidden, proj_weight, None, logits_raw,
            config.vocab_size, h, proj_dtype,
        );

        // ===================================================================
        // Barrier analysis
        // ===================================================================
        let barriers = compute_barriers(&b.ops);
        let num_slots = b.slot_sizes.len() as u16;

        DecodeGraph {
            ops: b.ops,
            barriers,
            slot_sizes: b.slot_sizes,
            num_slots,
            logits_slot,
            logits_count: config.vocab_size,
            kv_f16,
        }
    }
}

/// Emit FFN ops based on activation type.
fn emit_ffn(
    b: &mut GraphBuilder,
    input: BufferSlot,
    w_up: BufferRef,
    up_bias: Option<BufferRef>,
    w_down: BufferRef,
    down_bias: Option<BufferRef>,
    w_gate: Option<BufferRef>,
    activation: Activation,
    h: usize,
    ffn_h: usize,
    up_dtype: TensorDtype,
    down_dtype: TensorDtype,
    gate_dtype: Option<TensorDtype>,
) -> BufferSlot {
    let f32_size = std::mem::size_of::<f32>();

    match activation {
        Activation::SwiGLU => {
            let gate_w = w_gate.expect("SwiGLU requires gate weight");
            let gate_d = gate_dtype.expect("SwiGLU requires gate dtype");

            let gate_raw = b.alloc_slot(ffn_h * f32_size);
            let gate_out = b.emit_linear(input, gate_w, None, gate_raw, ffn_h, h, gate_d);

            let up_raw = b.alloc_slot(ffn_h * f32_size);
            let up_out = b.emit_linear(input, w_up, up_bias, up_raw, ffn_h, h, up_dtype);

            let activated = b.alloc_slot(ffn_h * f32_size);
            b.emit_swiglu(gate_out, up_out, activated, ffn_h);

            let down_raw = b.alloc_slot(h * f32_size);
            b.emit_linear(activated, w_down, down_bias, down_raw, h, ffn_h, down_dtype)
        }
        Activation::GeGLU => {
            let gate_w = w_gate.expect("GeGLU requires gate weight");
            let gate_d = gate_dtype.expect("GeGLU requires gate dtype");

            let gate_raw = b.alloc_slot(ffn_h * f32_size);
            let gate_out = b.emit_linear(input, gate_w, None, gate_raw, ffn_h, h, gate_d);

            let up_raw = b.alloc_slot(ffn_h * f32_size);
            let up_out = b.emit_linear(input, w_up, up_bias, up_raw, ffn_h, h, up_dtype);

            let activated = b.alloc_slot(ffn_h * f32_size);
            b.emit_geglu(gate_out, up_out, activated, ffn_h);

            let down_raw = b.alloc_slot(h * f32_size);
            b.emit_linear(activated, w_down, down_bias, down_raw, h, ffn_h, down_dtype)
        }
        Activation::GELU => {
            let up_raw = b.alloc_slot(ffn_h * f32_size);
            let up_out = b.emit_linear(input, w_up, up_bias, up_raw, ffn_h, h, up_dtype);

            let activated = b.alloc_slot(ffn_h * f32_size);
            b.emit_gelu(up_out, activated, ffn_h);

            let down_raw = b.alloc_slot(h * f32_size);
            b.emit_linear(activated, w_down, down_bias, down_raw, h, ffn_h, down_dtype)
        }
    }
}

// ---------------------------------------------------------------------------
// Barrier computation
// ---------------------------------------------------------------------------

/// Compute barrier positions by tracking read/write conflicts.
///
/// Walk ops in order. Track in-flight reads and writes by `BufferRef`.
/// When a new op conflicts with in-flight state (new read overlaps write,
/// or new write overlaps any access), emit a barrier and clear in-flight sets.
///
/// `Weight(*)` buffers are excluded since they are read-only during decode.
pub(crate) fn compute_barriers(ops: &[DecodeOp]) -> Vec<usize> {
    let mut barriers = Vec::new();
    let mut inflight_writes: Vec<BufferRef> = Vec::new();
    let mut inflight_reads: Vec<BufferRef> = Vec::new();

    for (i, op) in ops.iter().enumerate() {
        let mut needs_barrier = false;

        // Check if any of this op's reads conflict with in-flight writes
        for r in &op.reads {
            if is_trackable(r) && inflight_writes.contains(r) {
                needs_barrier = true;
                break;
            }
        }

        // Check if this op's write conflicts with any in-flight access
        if !needs_barrier {
            if let Some(ref w) = op.writes {
                if is_trackable(w) {
                    if inflight_reads.contains(w) || inflight_writes.contains(w) {
                        needs_barrier = true;
                    }
                }
            }
        }

        if needs_barrier {
            barriers.push(i);
            inflight_reads.clear();
            inflight_writes.clear();
        }

        // Add this op's accesses to in-flight sets
        for r in &op.reads {
            if is_trackable(r) && !inflight_reads.contains(r) {
                inflight_reads.push(*r);
            }
        }
        if let Some(ref w) = op.writes {
            if is_trackable(w) && !inflight_writes.contains(w) {
                inflight_writes.push(*w);
            }
        }
    }

    barriers
}

/// Whether a buffer ref should be tracked for conflicts.
/// Weight buffers are read-only so never conflict with each other.
fn is_trackable(r: &BufferRef) -> bool {
    !matches!(r, BufferRef::Weight(_))
}

// ---------------------------------------------------------------------------
// PrefillGraph — multi-token prefill operation graph
// ---------------------------------------------------------------------------

/// The complete prefill graph: ops, barriers, and buffer layout.
///
/// Same shape as `DecodeGraph` but sized for M tokens. Quantized matmuls
/// are handled via M row-loop dispatches (one per input row) with buffer
/// offsets, while native multi-token kernels (norm, rope, attention, etc.)
/// use a single dispatch.
#[allow(dead_code)]
pub(crate) struct PrefillGraph {
    pub ops: Vec<DecodeOp>,
    pub barriers: Vec<usize>,
    pub slot_sizes: Vec<usize>,
    pub num_slots: u16,
    pub logits_slot: BufferSlot,
    pub logits_count: usize,
    pub kv_f16: bool,
}

/// Patch EmbeddingLookup ops in an op list to use a different weight index.
///
/// Generic over both DecodeGraph and PrefillGraph ops.
pub(crate) fn patch_ops(ops: &mut Vec<DecodeOp>, old_ref: BufferRef, new_ref: BufferRef) {
    for op in ops.iter_mut() {
        if op.pso == PsoRef::EmbeddingLookup {
            for binding in &mut op.bindings {
                if binding.0 == old_ref && binding.1 == 0 {
                    binding.0 = new_ref;
                }
            }
            for read in &mut op.reads {
                if *read == old_ref {
                    *read = new_ref;
                }
            }
        }
    }
}

impl PrefillGraph {
    /// Build the prefill graph for `n_tokens` tokens.
    ///
    /// Mirrors `DecodeGraph::build()` but with multi-token support:
    /// - Intermediate buffers sized for M rows
    /// - Quantized matmuls emit M row-loop dispatches with buffer offsets
    /// - F32 matmuls use native M>1 support
    /// - Norm/RoPE/attention kernels use native multi-row dispatch
    /// - Final output extracts last-token hidden state for logits projection
    pub(crate) fn build(
        config: &ModelConfig,
        weights: &ModelWeights,
        n_tokens: usize,
        kv_f16: bool,
    ) -> Self {
        let mut b = GraphBuilder::new();

        let h = config.hidden_size;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let total_dim = config.num_heads * config.head_dim;
        let ffn_h = config.ffn_hidden;
        let f32_size = std::mem::size_of::<f32>();
        let m = n_tokens;

        // LongRoPE mscale: applied inside RoPE kernel to both Q and K, so we do NOT
        // modify attn_scale here.  See DecodeGraph::build comment for details.
        let yarn_mscale = config.rope_scaling_attn_factor;
        let attn_scale = config.attn_scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());

        // ===================================================================
        // Index global weights (same order as weight_walk_order)
        // ===================================================================
        let token_emb = b.next_weight();
        let pos_emb = if weights.position_embedding.is_some() {
            Some(b.next_weight())
        } else {
            None
        };
        let output_norm_w = b.next_weight_opt(weights.output_norm_w.is_some());
        let output_norm_b = b.next_weight_opt(weights.output_norm_b.is_some());
        let output_proj = b.next_weight_opt(weights.output_projection.is_some());
        let rope_factors = b.next_weight_opt(weights.rope_factors_short.is_some());
        let rope_factors_long = b.next_weight_opt(weights.rope_factors_long.is_some());
        let _ = rope_factors_long; // consumed by next_weight_opt to maintain walk order

        // ===================================================================
        // Pre-layer: embedding lookup + position embedding + scaling
        // ===================================================================

        // Multi-token embedding lookup: [M, hidden_size]
        let hidden_slot = b.alloc_slot(m * h * f32_size);
        emit_embedding_lookup_multi(&mut b, token_emb, hidden_slot, h, m);

        let mut current_hidden = hidden_slot;

        // Embedding scaling (element-wise, count = M * h)
        if (config.embedding_scale - 1.0).abs() > f32::EPSILON {
            let scaled = b.alloc_slot(m * h * f32_size);
            emit_scale_multi(&mut b, current_hidden, scaled, m * h, config.embedding_scale);
            current_hidden = scaled;
        }

        // Position embedding (Learned, GPT-2/BERT)
        if config.position_type == PositionType::Learned {
            if let Some(pos_emb_ref) = pos_emb {
                let pos_slot = b.alloc_slot(m * h * f32_size);
                emit_pos_embedding_lookup_multi(&mut b, pos_emb_ref, pos_slot, h, m);

                let combined = b.alloc_slot(m * h * f32_size);
                emit_add_multi(&mut b, current_hidden, pos_slot, combined, m * h);
                current_hidden = combined;
            }
        }

        // ===================================================================
        // Per-layer transformer (multi-token prefill)
        // ===================================================================

        for layer_idx in 0..config.num_layers {
            let layer = &weights.layers[layer_idx];

            // Index this layer's weights (must match weight_walk_order)
            let attn_norm_w = b.next_weight_opt(layer.attn_norm_w.is_some());
            let attn_norm_b = b.next_weight_opt(layer.attn_norm_b.is_some());
            let w_q = b.next_weight();
            let q_bias = b.next_weight_opt(layer.attn_q_bias.is_some());
            let w_k = b.next_weight();
            let k_bias = b.next_weight_opt(layer.attn_k_bias.is_some());
            let w_v = b.next_weight();
            let v_bias = b.next_weight_opt(layer.attn_v_bias.is_some());
            let w_o = b.next_weight();
            let o_bias = b.next_weight_opt(layer.attn_output_bias.is_some());
            let ffn_norm_w = b.next_weight_opt(layer.ffn_norm_w.is_some());
            let ffn_norm_b = b.next_weight_opt(layer.ffn_norm_b.is_some());
            let w_up = b.next_weight();
            let up_bias = b.next_weight_opt(layer.ffn_up_bias.is_some());
            let w_down = b.next_weight();
            let down_bias = b.next_weight_opt(layer.ffn_down_bias.is_some());
            let w_gate = b.next_weight_opt(layer.ffn_gate.is_some());

            // Post-norm weights (BERT)
            let attn_output_norm_w = b.next_weight_opt(layer.attn_output_norm_w.is_some());
            let attn_output_norm_b = b.next_weight_opt(layer.attn_output_norm_b.is_some());
            let ffn_output_norm_w = b.next_weight_opt(layer.ffn_output_norm_w.is_some());
            let ffn_output_norm_b = b.next_weight_opt(layer.ffn_output_norm_b.is_some());

            let _attn_q_norm_w = b.next_weight_opt(layer.attn_q_norm_w.is_some());
            let _attn_k_norm_w = b.next_weight_opt(layer.attn_k_norm_w.is_some());
            let _attn_post_norm_w = b.next_weight_opt(layer.attn_post_norm_w.is_some());
            let _ffn_post_norm_w = b.next_weight_opt(layer.ffn_post_norm_w.is_some());

            let k_cache = BufferRef::KvCache((layer_idx * 2) as u16);
            let v_cache = BufferRef::KvCache((layer_idx * 2 + 1) as u16);

            let q_dtype = layer.attn_q.dtype();
            let k_dtype = layer.attn_k.dtype();
            let v_dtype = layer.attn_v.dtype();
            let o_dtype = layer.attn_output.dtype();
            let up_dtype = layer.ffn_up.dtype();
            let down_dtype = layer.ffn_down.dtype();
            let gate_dtype = layer.ffn_gate.as_ref().map(|w| w.dtype());

            if config.pre_norm {
                // ===========================================================
                // Pre-norm path (Gemma/LLaMA/GPT-2)
                // ===========================================================

                // 1. Pre-attention norm [M, h] -> [M, h]
                let normed = b.alloc_slot(m * h * f32_size);
                emit_norm_multi(
                    &mut b, current_hidden,
                    attn_norm_w.expect("pre_norm requires attn_norm_w"),
                    attn_norm_b, normed, h, m, config,
                );

                // 2. QKV projections [M, h] -> [M, total_dim] / [M, kv_dim]
                let q_raw = b.alloc_slot(m * total_dim * f32_size);
                emit_linear_multi(&mut b, normed, w_q, q_bias, q_raw, total_dim, h, q_dtype, m);

                let k_raw = b.alloc_slot(m * kv_dim * f32_size);
                emit_linear_multi(&mut b, normed, w_k, k_bias, k_raw, kv_dim, h, k_dtype, m);

                let v_raw = b.alloc_slot(m * kv_dim * f32_size);
                emit_linear_multi(&mut b, normed, w_v, v_bias, v_raw, kv_dim, h, v_dtype, m);

                // 3. RoPE (if configured) — multi-token, gz = M
                let (q_final, k_final) = if config.position_type == PositionType::RoPE {
                    let q_rope = b.alloc_slot(m * total_dim * f32_size);
                    emit_rope_multi(
                        &mut b, config.rope_neox, q_raw, q_rope,
                        config.num_heads, config.head_dim, config.rope_dim,
                        config.rope_freq_base, m, rope_factors, yarn_mscale,
                    );
                    let k_rope = b.alloc_slot(m * kv_dim * f32_size);
                    emit_rope_multi(
                        &mut b, config.rope_neox, k_raw, k_rope,
                        config.num_kv_heads, config.head_dim, config.rope_dim,
                        config.rope_freq_base, m, rope_factors, yarn_mscale,
                    );
                    (q_rope, k_rope)
                } else {
                    (q_raw, k_raw)
                };

                // 4. Copy K, V into KV cache (M rows at once)
                emit_copy_to_cache_multi(&mut b, k_final, k_cache, kv_dim, m, kv_f16);
                emit_copy_to_cache_multi(&mut b, v_raw, v_cache, kv_dim, m, kv_f16);

                // 5. Batched causal attention
                let attn_out = b.alloc_slot(m * total_dim * f32_size);
                emit_batched_attention(
                    &mut b, q_final, k_cache, v_cache, attn_out,
                    config.num_heads, config.num_kv_heads, config.head_dim,
                    attn_scale, config.attn_logit_softcap, m, kv_f16,
                );

                // 6. Output projection + residual [M, total_dim] -> [M, h]
                let proj_raw = b.alloc_slot(m * h * f32_size);
                emit_linear_multi(&mut b, attn_out, w_o, o_bias, proj_raw, h, total_dim, o_dtype, m);

                let residual = b.alloc_slot(m * h * f32_size);
                emit_add_multi(&mut b, current_hidden, proj_raw, residual, m * h);

                // 7. Pre-FFN norm [M, h] -> [M, h]
                let normed2 = b.alloc_slot(m * h * f32_size);
                emit_norm_multi(
                    &mut b, residual,
                    ffn_norm_w.expect("pre_norm requires ffn_norm_w"),
                    ffn_norm_b, normed2, h, m, config,
                );

                // 8. FFN
                let ffn_out = emit_ffn_multi(
                    &mut b, normed2, w_up, up_bias, w_down, down_bias, w_gate,
                    config.activation, h, ffn_h,
                    up_dtype, down_dtype, gate_dtype, m,
                );

                // 9. Residual
                let layer_out = b.alloc_slot(m * h * f32_size);
                emit_add_multi(&mut b, residual, ffn_out, layer_out, m * h);
                current_hidden = layer_out;

            } else {
                // ===========================================================
                // Post-norm path (BERT)
                // ===========================================================

                let q_raw = b.alloc_slot(m * total_dim * f32_size);
                emit_linear_multi(&mut b, current_hidden, w_q, q_bias, q_raw, total_dim, h, q_dtype, m);

                let k_raw = b.alloc_slot(m * kv_dim * f32_size);
                emit_linear_multi(&mut b, current_hidden, w_k, k_bias, k_raw, kv_dim, h, k_dtype, m);

                let v_raw = b.alloc_slot(m * kv_dim * f32_size);
                emit_linear_multi(&mut b, current_hidden, w_v, v_bias, v_raw, kv_dim, h, v_dtype, m);

                let (q_final, k_final) = if config.position_type == PositionType::RoPE {
                    let q_rope = b.alloc_slot(m * total_dim * f32_size);
                    emit_rope_multi(
                        &mut b, config.rope_neox, q_raw, q_rope,
                        config.num_heads, config.head_dim, config.rope_dim,
                        config.rope_freq_base, m, rope_factors, yarn_mscale,
                    );
                    let k_rope = b.alloc_slot(m * kv_dim * f32_size);
                    emit_rope_multi(
                        &mut b, config.rope_neox, k_raw, k_rope,
                        config.num_kv_heads, config.head_dim, config.rope_dim,
                        config.rope_freq_base, m, rope_factors, yarn_mscale,
                    );
                    (q_rope, k_rope)
                } else {
                    (q_raw, k_raw)
                };

                emit_copy_to_cache_multi(&mut b, k_final, k_cache, kv_dim, m, kv_f16);
                emit_copy_to_cache_multi(&mut b, v_raw, v_cache, kv_dim, m, kv_f16);

                let attn_out = b.alloc_slot(m * total_dim * f32_size);
                emit_batched_attention(
                    &mut b, q_final, k_cache, v_cache, attn_out,
                    config.num_heads, config.num_kv_heads, config.head_dim,
                    attn_scale, config.attn_logit_softcap, m, kv_f16,
                );

                let proj_raw = b.alloc_slot(m * h * f32_size);
                emit_linear_multi(&mut b, attn_out, w_o, o_bias, proj_raw, h, total_dim, o_dtype, m);

                let mut residual_slot = b.alloc_slot(m * h * f32_size);
                emit_add_multi(&mut b, current_hidden, proj_raw, residual_slot, m * h);

                if let (Some(nw), Some(nb)) = (attn_output_norm_w, attn_output_norm_b) {
                    let normed = b.alloc_slot(m * h * f32_size);
                    emit_norm_multi(&mut b, residual_slot, nw, Some(nb), normed, h, m, config);
                    residual_slot = normed;
                }

                let ffn_out = emit_ffn_multi(
                    &mut b, residual_slot, w_up, up_bias, w_down, down_bias, w_gate,
                    config.activation, h, ffn_h,
                    up_dtype, down_dtype, gate_dtype, m,
                );

                let mut output = b.alloc_slot(m * h * f32_size);
                emit_add_multi(&mut b, residual_slot, ffn_out, output, m * h);

                if let (Some(nw), Some(nb)) = (ffn_output_norm_w, ffn_output_norm_b) {
                    let normed = b.alloc_slot(m * h * f32_size);
                    emit_norm_multi(&mut b, output, nw, Some(nb), normed, h, m, config);
                    output = normed;
                }

                current_hidden = output;
            }
        }

        // ===================================================================
        // Post-layer: final norm + last-token extraction + logits projection
        // ===================================================================

        if let Some(norm_w) = output_norm_w {
            let normed = b.alloc_slot(m * h * f32_size);
            emit_norm_multi(&mut b, current_hidden, norm_w, output_norm_b, normed, h, m, config);
            current_hidden = normed;
        }

        // Extract last token's hidden state: copy [1, h] from row (M-1)
        let last_hidden = b.alloc_slot(h * f32_size);
        let src_offset = ((m - 1) * h * f32_size) as u32;
        b.ops.push(DecodeOp {
            pso: PsoRef::CopyBuffer,
            bindings: vec![
                (BufferRef::Pool(current_hidden), 0, src_offset),
                (BufferRef::Pool(last_hidden), 1, 0),
            ],
            params: vec![
                (ParamValue::U32(h as u32), 2),  // count = hidden_size
                (ParamValue::U32(0), 3),          // dest_offset = 0
            ],
            dispatch: DispatchDims::Fixed {
                gx: ((h + 255) / 256) as u32, gy: 1, gz: 1,
                tx: 256, ty: 1, tz: 1,
            },
            reads: vec![BufferRef::Pool(current_hidden)],
            writes: Some(BufferRef::Pool(last_hidden)),
        });

        // Logits projection: [1, h] -> [1, vocab_size]
        let proj_weight = output_proj.unwrap_or(token_emb);
        let proj_dtype = weights.output_projection.as_ref()
            .map(|w| w.dtype())
            .unwrap_or(weights.token_embedding.dtype());

        let logits_raw = b.alloc_slot(config.vocab_size * f32_size);
        // Single-row logits projection (same as decode)
        let logits_slot = b.emit_linear(
            last_hidden, proj_weight, None, logits_raw,
            config.vocab_size, h, proj_dtype,
        );

        // ===================================================================
        // Barrier analysis
        // ===================================================================
        let barriers = compute_barriers(&b.ops);
        let num_slots = b.slot_sizes.len() as u16;

        PrefillGraph {
            ops: b.ops,
            barriers,
            slot_sizes: b.slot_sizes,
            num_slots,
            logits_slot,
            logits_count: config.vocab_size,
            kv_f16,
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-token emit helpers for PrefillGraph
// ---------------------------------------------------------------------------

/// Multi-token embedding lookup: dispatch gy = n_tokens, bind PrefillTokenIds buffer.
fn emit_embedding_lookup_multi(
    b: &mut GraphBuilder,
    table: BufferRef,
    out: BufferSlot,
    hidden_size: usize,
    n_tokens: usize,
) {
    b.ops.push(DecodeOp {
        pso: PsoRef::EmbeddingLookup,
        bindings: vec![
            (table, 0, 0),
            (BufferRef::Pool(out), 2, 0),
        ],
        params: vec![
            (ParamValue::PrefillTokenIds, 1),
            (ParamValue::U32(hidden_size as u32), 3),
            (ParamValue::PrefillNTokens, 4),
        ],
        dispatch: DispatchDims::D2 {
            gx: ((hidden_size + 15) / 16) as u32,
            gy: ((n_tokens + 15) / 16) as u32,
            tx: 16,
            ty: 16,
        },
        reads: vec![table],
        writes: Some(BufferRef::Pool(out)),
    });
}

/// Multi-token position embedding lookup: dispatch gy = n_tokens.
fn emit_pos_embedding_lookup_multi(
    b: &mut GraphBuilder,
    table: BufferRef,
    out: BufferSlot,
    hidden_size: usize,
    n_tokens: usize,
) {
    b.ops.push(DecodeOp {
        pso: PsoRef::EmbeddingLookup,
        bindings: vec![
            (table, 0, 0),
            (BufferRef::Pool(out), 2, 0),
        ],
        params: vec![
            (ParamValue::PrefillPositionIds, 1),
            (ParamValue::U32(hidden_size as u32), 3),
            (ParamValue::PrefillNTokens, 4),
        ],
        dispatch: DispatchDims::D2 {
            gx: ((hidden_size + 15) / 16) as u32,
            gy: ((n_tokens + 15) / 16) as u32,
            tx: 16,
            ty: 16,
        },
        reads: vec![table],
        writes: Some(BufferRef::Pool(out)),
    });
}

/// Multi-token norm: dispatch num_rows = n_tokens.
fn emit_norm_multi(
    b: &mut GraphBuilder,
    input: BufferSlot,
    weight: BufferRef,
    bias: Option<BufferRef>,
    out: BufferSlot,
    cols: usize,
    n_tokens: usize,
    config: &ModelConfig,
) {
    let threads_per_group = (cols.next_power_of_two()).min(256) as u32;
    match config.norm_type {
        NormType::LayerNorm => {
            let bias_ref = bias.expect("LayerNorm requires bias");
            b.ops.push(DecodeOp {
                pso: PsoRef::LayerNorm,
                bindings: vec![
                    (BufferRef::Pool(input), 0, 0),
                    (weight, 1, 0),
                    (bias_ref, 2, 0),
                    (BufferRef::Pool(out), 3, 0),
                ],
                params: vec![
                    (ParamValue::U32(n_tokens as u32), 4),
                    (ParamValue::U32(cols as u32), 5),
                    (ParamValue::F32(config.norm_eps), 6),
                ],
                dispatch: DispatchDims::Rows { num_rows: n_tokens as u32, threads_per_group },
                reads: vec![BufferRef::Pool(input), weight, bias_ref],
                writes: Some(BufferRef::Pool(out)),
            });
        }
        NormType::RMSNorm => {
            b.ops.push(DecodeOp {
                pso: PsoRef::RmsNorm,
                bindings: vec![
                    (BufferRef::Pool(input), 0, 0),
                    (weight, 1, 0),
                    (BufferRef::Pool(out), 2, 0),
                ],
                params: vec![
                    (ParamValue::U32(n_tokens as u32), 3),
                    (ParamValue::U32(cols as u32), 4),
                    (ParamValue::F32(config.norm_eps), 5),
                ],
                dispatch: DispatchDims::Rows { num_rows: n_tokens as u32, threads_per_group },
                reads: vec![BufferRef::Pool(input), weight],
                writes: Some(BufferRef::Pool(out)),
            });
        }
    }
}

/// Multi-token linear: both quantized and F32 matmuls use native M>1 via the M
/// parameter. Quantized types use batched GEMM kernels with fused dequant;
/// F32/F16 types use gemm_transpose / gemm_transpose_bias.
fn emit_linear_multi(
    b: &mut GraphBuilder,
    input: BufferSlot,
    weight_ref: BufferRef,
    bias_ref: Option<BufferRef>,
    out: BufferSlot,
    n: usize,
    k: usize,
    dtype: TensorDtype,
    m: usize,
) {
    match dtype {
        TensorDtype::F32 | TensorDtype::F16 => {
            // Native M>1 support in gemm_transpose / gemm_transpose_bias
            if let Some(bias) = bias_ref {
                let gx = ((n + 31) / 32) as u32;
                let gy = ((m + 31) / 32) as u32; // BM = 32
                b.ops.push(DecodeOp {
                    pso: PsoRef::MatmulTransposeBias,
                    bindings: vec![
                        (BufferRef::Pool(input), 0, 0),
                        (weight_ref, 1, 0),
                        (BufferRef::Pool(out), 2, 0),
                        (bias, 6, 0),
                    ],
                    params: vec![
                        (ParamValue::U32(m as u32), 3),
                        (ParamValue::U32(k as u32), 4),
                        (ParamValue::U32(n as u32), 5),
                    ],
                    dispatch: DispatchDims::Fixed { gx, gy, gz: 1, tx: 128, ty: 1, tz: 1 },
                    reads: vec![BufferRef::Pool(input), weight_ref, bias],
                    writes: Some(BufferRef::Pool(out)),
                });
            } else {
                let gx = ((n + 31) / 32) as u32;
                let gy = ((m + 31) / 32) as u32; // BM = 32
                b.ops.push(DecodeOp {
                    pso: PsoRef::MatmulTranspose,
                    bindings: vec![
                        (BufferRef::Pool(input), 0, 0),
                        (weight_ref, 1, 0),
                        (BufferRef::Pool(out), 2, 0),
                    ],
                    params: vec![
                        (ParamValue::U32(m as u32), 3),
                        (ParamValue::U32(k as u32), 4),
                        (ParamValue::U32(n as u32), 5),
                    ],
                    dispatch: DispatchDims::Fixed { gx, gy, gz: 1, tx: 128, ty: 1, tz: 1 },
                    reads: vec![BufferRef::Pool(input), weight_ref],
                    writes: Some(BufferRef::Pool(out)),
                });
            }
        }
        _ if m == 1 => {
            // Single-row quantized matmul: optimized fused dequant+dot for M=1.
            if let Some(bias) = bias_ref {
                let (pso, threadgroups, threads) = match dtype {
                    TensorDtype::Q8_0 => (PsoRef::QuantizedMatmulBiasQ8_0, ((n + 7) / 8) as u32, 128u32),
                    TensorDtype::Q4_0 => (PsoRef::QuantizedMatmulBiasQ4_0, ((n + 7) / 8) as u32, 64u32),
                    TensorDtype::Q4_K => (PsoRef::QuantizedMatmulBiasQ4K, ((n + 3) / 4) as u32, 64u32),
                    TensorDtype::Q5_K => (PsoRef::QuantizedMatmulBiasQ5K, ((n + 3) / 4) as u32, 64u32),
                    TensorDtype::Q6_K => (PsoRef::QuantizedMatmulBiasQ6K, ((n + 3) / 4) as u32, 64u32),
                    _ => panic!("unsupported quantized dtype: {:?}", dtype),
                };
                b.ops.push(DecodeOp {
                    pso,
                    bindings: vec![
                        (weight_ref, 0, 0),
                        (BufferRef::Pool(input), 1, 0),
                        (BufferRef::Pool(out), 2, 0),
                        (bias, 5, 0),
                    ],
                    params: vec![
                        (ParamValue::U32(n as u32), 3),
                        (ParamValue::U32(k as u32), 4),
                    ],
                    dispatch: DispatchDims::Fixed { gx: threadgroups, gy: 1, gz: 1, tx: threads, ty: 1, tz: 1 },
                    reads: vec![weight_ref, BufferRef::Pool(input), bias],
                    writes: Some(BufferRef::Pool(out)),
                });
            } else {
                let (pso, threadgroups, threads) = match dtype {
                    TensorDtype::Q8_0 => (PsoRef::QuantizedMatmulQ8_0, ((n + 7) / 8) as u32, 128u32),
                    TensorDtype::Q4_0 => (PsoRef::QuantizedMatmulQ4_0, ((n + 7) / 8) as u32, 64u32),
                    TensorDtype::Q4_K => (PsoRef::QuantizedMatmulQ4K, ((n + 3) / 4) as u32, 64u32),
                    TensorDtype::Q5_K => (PsoRef::QuantizedMatmulQ5K, ((n + 3) / 4) as u32, 64u32),
                    TensorDtype::Q6_K => (PsoRef::QuantizedMatmulQ6K, ((n + 3) / 4) as u32, 64u32),
                    _ => panic!("unsupported quantized dtype: {:?}", dtype),
                };
                b.ops.push(DecodeOp {
                    pso,
                    bindings: vec![
                        (weight_ref, 0, 0),
                        (BufferRef::Pool(input), 1, 0),
                        (BufferRef::Pool(out), 2, 0),
                    ],
                    params: vec![
                        (ParamValue::U32(n as u32), 3),
                        (ParamValue::U32(k as u32), 4),
                    ],
                    dispatch: DispatchDims::Fixed { gx: threadgroups, gy: 1, gz: 1, tx: threads, ty: 1, tz: 1 },
                    reads: vec![weight_ref, BufferRef::Pool(input)],
                    writes: Some(BufferRef::Pool(out)),
                });
            }
        }
        _ => {
            // Batched quantized GEMM for M>1 (prefill): single dispatch per linear
            // using tiled matrix multiplication with fused dequantization.
            // BBM=64, BBN=32, BBK=32 tiles with half-precision threadgroup memory
            // and block-level dequantization for high throughput.
            let gx = ((n + 31) / 32) as u32; // ceil(N / BBN)
            let gy = ((m + 63) / 64) as u32; // ceil(M / BBM)

            if let Some(bias) = bias_ref {
                let pso = match dtype {
                    TensorDtype::Q8_0 => PsoRef::BatchedMatmulBiasQ8_0,
                    TensorDtype::Q4_0 => PsoRef::BatchedMatmulBiasQ4_0,
                    TensorDtype::Q4_K => PsoRef::BatchedMatmulBiasQ4K,
                    TensorDtype::Q5_K => PsoRef::BatchedMatmulBiasQ5K,
                    TensorDtype::Q6_K => PsoRef::BatchedMatmulBiasQ6K,
                    _ => unreachable!(),
                };
                b.ops.push(DecodeOp {
                    pso,
                    bindings: vec![
                        (weight_ref, 0, 0),
                        (BufferRef::Pool(input), 1, 0),
                        (BufferRef::Pool(out), 2, 0),
                        (bias, 6, 0),
                    ],
                    params: vec![
                        (ParamValue::U32(m as u32), 3),
                        (ParamValue::U32(k as u32), 4),
                        (ParamValue::U32(n as u32), 5),
                    ],
                    dispatch: DispatchDims::Fixed { gx, gy, gz: 1, tx: 128, ty: 1, tz: 1 },
                    reads: vec![weight_ref, BufferRef::Pool(input), bias],
                    writes: Some(BufferRef::Pool(out)),
                });
            } else {
                let pso = match dtype {
                    TensorDtype::Q8_0 => PsoRef::BatchedMatmulQ8_0,
                    TensorDtype::Q4_0 => PsoRef::BatchedMatmulQ4_0,
                    TensorDtype::Q4_K => PsoRef::BatchedMatmulQ4K,
                    TensorDtype::Q5_K => PsoRef::BatchedMatmulQ5K,
                    TensorDtype::Q6_K => PsoRef::BatchedMatmulQ6K,
                    _ => panic!("unsupported quantized dtype for prefill: {:?}", dtype),
                };
                b.ops.push(DecodeOp {
                    pso,
                    bindings: vec![
                        (weight_ref, 0, 0),
                        (BufferRef::Pool(input), 1, 0),
                        (BufferRef::Pool(out), 2, 0),
                    ],
                    params: vec![
                        (ParamValue::U32(m as u32), 3),
                        (ParamValue::U32(k as u32), 4),
                        (ParamValue::U32(n as u32), 5),
                    ],
                    dispatch: DispatchDims::Fixed { gx, gy, gz: 1, tx: 128, ty: 1, tz: 1 },
                    reads: vec![weight_ref, BufferRef::Pool(input)],
                    writes: Some(BufferRef::Pool(out)),
                });
            }
        }
    }
}

/// Multi-token add: count = n_tokens * dim.
fn emit_add_multi(
    b: &mut GraphBuilder,
    a: BufferSlot,
    b_slot: BufferSlot,
    out: BufferSlot,
    count: usize,
) {
    let threads = 256u32;
    let groups = ((count + 255) / 256) as u32;
    b.ops.push(DecodeOp {
        pso: PsoRef::AddTensor,
        bindings: vec![
            (BufferRef::Pool(a), 0, 0),
            (BufferRef::Pool(b_slot), 1, 0),
            (BufferRef::Pool(out), 2, 0),
        ],
        params: vec![(ParamValue::U32(count as u32), 3)],
        dispatch: DispatchDims::Fixed { gx: groups, gy: 1, gz: 1, tx: threads, ty: 1, tz: 1 },
        reads: vec![BufferRef::Pool(a), BufferRef::Pool(b_slot)],
        writes: Some(BufferRef::Pool(out)),
    });
}

/// Multi-token scale: count = n_tokens * dim.
fn emit_scale_multi(
    b: &mut GraphBuilder,
    input: BufferSlot,
    out: BufferSlot,
    count: usize,
    factor: f32,
) {
    let groups = ((count + 255) / 256) as u32;
    b.ops.push(DecodeOp {
        pso: PsoRef::ScaleKernel,
        bindings: vec![
            (BufferRef::Pool(input), 0, 0),
            (BufferRef::Pool(out), 1, 0),
        ],
        params: vec![
            (ParamValue::F32(factor), 2),
            (ParamValue::U32(count as u32), 3),
        ],
        dispatch: DispatchDims::Fixed { gx: groups, gy: 1, gz: 1, tx: 256, ty: 1, tz: 1 },
        reads: vec![BufferRef::Pool(input)],
        writes: Some(BufferRef::Pool(out)),
    });
}

/// Multi-token RoPE: dispatch gz = n_tokens, use PrefillPosOffset.
/// When `rope_factors` is Some, uses the `rope_neox_factors` kernel (LongRoPE).
/// `mscale` is the YaRN magnitude scaling factor (1.0 = no scaling).
fn emit_rope_multi(
    b: &mut GraphBuilder,
    neox: bool,
    input: BufferSlot,
    out: BufferSlot,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    freq_base: f32,
    n_tokens: usize,
    rope_factors: Option<BufferRef>,
    mscale: f32,
) {
    let pso = match (neox, rope_factors.is_some()) {
        (true, true) => PsoRef::RopeNeoxFactors,
        (true, false) => PsoRef::RopeNeox,
        (false, _) => PsoRef::RopeNorm,
    };
    let half_rope = rope_dim / 2;
    let gx = ((half_rope + 15) / 16) as u32;
    let gy = ((n_heads + 15) / 16) as u32;
    let gz = ((n_tokens + 0) / 1) as u32; // 1 group per token in z

    let mut bindings = vec![
        (BufferRef::Pool(input), 0, 0),
        (BufferRef::Pool(out), 1, 0),
    ];
    if let Some(factors_ref) = rope_factors {
        bindings.push((factors_ref, 8, 0));
    }

    let mut reads = vec![BufferRef::Pool(input)];
    if let Some(factors_ref) = rope_factors {
        reads.push(factors_ref);
    }

    let mut params = vec![
        (ParamValue::PrefillPosOffset, 2), // pos_offset (patched at encode)
        (ParamValue::F32(freq_base), 3),
        (ParamValue::U32(head_dim as u32), 4),
        (ParamValue::U32(rope_dim as u32), 5),
        (ParamValue::U32(n_heads as u32), 6),
        (ParamValue::U32(n_tokens as u32), 7), // seq_len = M
    ];
    if rope_factors.is_some() {
        params.push((ParamValue::F32(mscale), 9));
    }

    b.ops.push(DecodeOp {
        pso,
        bindings,
        params,
        dispatch: DispatchDims::Fixed { gx, gy, gz, tx: 16, ty: 16, tz: 1 },
        reads,
        writes: Some(BufferRef::Pool(out)),
    });
}

/// Multi-token copy to KV cache: count = M * kv_dim, dest_offset based on pos_offset.
fn emit_copy_to_cache_multi(
    b: &mut GraphBuilder,
    src: BufferSlot,
    kv_buf: BufferRef,
    kv_dim: usize,
    n_tokens: usize,
    kv_f16: bool,
) {
    let count = n_tokens * kv_dim;
    let groups = ((count + 255) / 256) as u32;
    let pso = if kv_f16 { PsoRef::CopyF32ToF16 } else { PsoRef::CopyBuffer };
    b.ops.push(DecodeOp {
        pso,
        bindings: vec![
            (BufferRef::Pool(src), 0, 0),
            (kv_buf, 1, 0),
        ],
        params: vec![
            (ParamValue::U32(count as u32), 2),
            // dest_offset = pos_offset * kv_dim (always 0 for initial prefill,
            // but using a fixed U32(0) here since the plan only supports pos_offset=0)
            (ParamValue::U32(0), 3),
        ],
        dispatch: DispatchDims::Fixed { gx: groups, gy: 1, gz: 1, tx: 256, ty: 1, tz: 1 },
        reads: vec![BufferRef::Pool(src)],
        writes: Some(kv_buf),
    });
}

/// Multi-token batched causal attention.
fn emit_batched_attention(
    b: &mut GraphBuilder,
    q: BufferSlot,
    k_cache: BufferRef,
    v_cache: BufferRef,
    out: BufferSlot,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_scale: f32,
    softcap: f32,
    n_tokens: usize,
    kv_f16: bool,
) {
    let pso = if kv_f16 { PsoRef::BatchedCausalAttentionF16 } else { PsoRef::BatchedCausalAttention };
    // gid = q_idx * num_heads + h, so dispatch n_tokens * num_heads groups
    b.ops.push(DecodeOp {
        pso,
        bindings: vec![
            (BufferRef::Pool(q), 0, 0),
            (k_cache, 1, 0),
            (v_cache, 2, 0),
            (BufferRef::Pool(out), 3, 0),
        ],
        params: vec![
            (ParamValue::U32(num_heads as u32), 4),
            (ParamValue::U32(num_kv_heads as u32), 5),
            (ParamValue::U32(head_dim as u32), 6),
            (ParamValue::PrefillNTokens, 7),   // n_tokens = M
            (ParamValue::PrefillTotalLen, 8),   // total_len = pos_offset + M
            (ParamValue::PrefillPosOffset, 9),  // pos_offset = 0 for initial prefill
            (ParamValue::F32(attn_scale), 10),
            (ParamValue::F32(softcap), 11),
        ],
        dispatch: DispatchDims::Fixed {
            gx: (n_tokens * num_heads) as u32, gy: 1, gz: 1,
            tx: 256, ty: 1, tz: 1,
        },
        reads: vec![BufferRef::Pool(q), k_cache, v_cache],
        writes: Some(BufferRef::Pool(out)),
    });
}

/// Multi-token FFN.
fn emit_ffn_multi(
    b: &mut GraphBuilder,
    input: BufferSlot,
    w_up: BufferRef,
    up_bias: Option<BufferRef>,
    w_down: BufferRef,
    down_bias: Option<BufferRef>,
    w_gate: Option<BufferRef>,
    activation: Activation,
    h: usize,
    ffn_h: usize,
    up_dtype: TensorDtype,
    down_dtype: TensorDtype,
    gate_dtype: Option<TensorDtype>,
    m: usize,
) -> BufferSlot {
    let f32_size = std::mem::size_of::<f32>();

    match activation {
        Activation::SwiGLU => {
            let gate_w = w_gate.expect("SwiGLU requires gate weight");
            let gate_d = gate_dtype.expect("SwiGLU requires gate dtype");

            let gate_raw = b.alloc_slot(m * ffn_h * f32_size);
            emit_linear_multi(b, input, gate_w, None, gate_raw, ffn_h, h, gate_d, m);

            let up_raw = b.alloc_slot(m * ffn_h * f32_size);
            emit_linear_multi(b, input, w_up, up_bias, up_raw, ffn_h, h, up_dtype, m);

            // SwiGLU: element-wise, count = M * ffn_h
            let activated = b.alloc_slot(m * ffn_h * f32_size);
            let count = m * ffn_h;
            let groups = ((count + 255) / 256) as u32;
            b.ops.push(DecodeOp {
                pso: PsoRef::SwiGlu,
                bindings: vec![
                    (BufferRef::Pool(gate_raw), 0, 0),
                    (BufferRef::Pool(up_raw), 1, 0),
                    (BufferRef::Pool(activated), 2, 0),
                ],
                params: vec![(ParamValue::U32(count as u32), 3)],
                dispatch: DispatchDims::Fixed { gx: groups, gy: 1, gz: 1, tx: 256, ty: 1, tz: 1 },
                reads: vec![BufferRef::Pool(gate_raw), BufferRef::Pool(up_raw)],
                writes: Some(BufferRef::Pool(activated)),
            });

            let down_raw = b.alloc_slot(m * h * f32_size);
            emit_linear_multi(b, activated, w_down, down_bias, down_raw, h, ffn_h, down_dtype, m);
            down_raw
        }
        Activation::GeGLU => {
            let gate_w = w_gate.expect("GeGLU requires gate weight");
            let gate_d = gate_dtype.expect("GeGLU requires gate dtype");

            let gate_raw = b.alloc_slot(m * ffn_h * f32_size);
            emit_linear_multi(b, input, gate_w, None, gate_raw, ffn_h, h, gate_d, m);

            let up_raw = b.alloc_slot(m * ffn_h * f32_size);
            emit_linear_multi(b, input, w_up, up_bias, up_raw, ffn_h, h, up_dtype, m);

            let activated = b.alloc_slot(m * ffn_h * f32_size);
            let count = m * ffn_h;
            let groups = ((count + 255) / 256) as u32;
            b.ops.push(DecodeOp {
                pso: PsoRef::GeGlu,
                bindings: vec![
                    (BufferRef::Pool(gate_raw), 0, 0),
                    (BufferRef::Pool(up_raw), 1, 0),
                    (BufferRef::Pool(activated), 2, 0),
                ],
                params: vec![(ParamValue::U32(count as u32), 3)],
                dispatch: DispatchDims::Fixed { gx: groups, gy: 1, gz: 1, tx: 256, ty: 1, tz: 1 },
                reads: vec![BufferRef::Pool(gate_raw), BufferRef::Pool(up_raw)],
                writes: Some(BufferRef::Pool(activated)),
            });

            let down_raw = b.alloc_slot(m * h * f32_size);
            emit_linear_multi(b, activated, w_down, down_bias, down_raw, h, ffn_h, down_dtype, m);
            down_raw
        }
        Activation::GELU => {
            let up_raw = b.alloc_slot(m * ffn_h * f32_size);
            emit_linear_multi(b, input, w_up, up_bias, up_raw, ffn_h, h, up_dtype, m);

            let activated = b.alloc_slot(m * ffn_h * f32_size);
            let count = m * ffn_h;
            let groups = ((count + 255) / 256) as u32;
            b.ops.push(DecodeOp {
                pso: PsoRef::Gelu,
                bindings: vec![
                    (BufferRef::Pool(up_raw), 0, 0),
                    (BufferRef::Pool(activated), 1, 0),
                ],
                params: vec![(ParamValue::U32(count as u32), 2)],
                dispatch: DispatchDims::Fixed { gx: groups, gy: 1, gz: 1, tx: 256, ty: 1, tz: 1 },
                reads: vec![BufferRef::Pool(up_raw)],
                writes: Some(BufferRef::Pool(activated)),
            });

            let down_raw = b.alloc_slot(m * h * f32_size);
            emit_linear_multi(b, activated, w_down, down_bias, down_raw, h, ffn_h, down_dtype, m);
            down_raw
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::DeviceTensor;
    use crate::model::config::*;
    use crate::model::weights::{LayerWeights, ModelWeights};
    use crate::tensor::Tensor;

    fn dt(shape: Vec<usize>, dtype: TensorDtype) -> DeviceTensor {
        let count: usize = shape.iter().product();
        match dtype {
            TensorDtype::F32 => DeviceTensor::new(Tensor::new(shape, vec![0.0f32; count])),
            _ => {
                // All quantized types
                let block_size = dtype.block_size();
                let block_bytes = dtype.block_byte_size();
                let n_blocks = (count + block_size - 1) / block_size;
                DeviceTensor::new(Tensor::from_quantized(shape, dtype, vec![0u8; n_blocks * block_bytes]))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Config helpers
    // -----------------------------------------------------------------------

    fn gpt2_config() -> ModelConfig {
        ModelConfig {
            arch: ModelArch::GPT2,
            arch_name: "gpt2".to_string(),
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 12,
            head_dim: 64,
            ffn_hidden: 3072,
            vocab_size: 50257,
            max_seq_len: 1024,
            norm_type: NormType::LayerNorm,
            norm_eps: 1e-5,
            activation: Activation::GELU,
            position_type: PositionType::Learned,
            rope_freq_base: 10000.0,
            rope_dim: 64,
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

    fn llama_config() -> ModelConfig {
        ModelConfig {
            arch: ModelArch::LLaMA,
            arch_name: "llama".to_string(),
            hidden_size: 2048,
            num_layers: 22,
            num_heads: 32,
            num_kv_heads: 4, // GQA
            head_dim: 64,
            ffn_hidden: 5632,
            vocab_size: 32000,
            max_seq_len: 2048,
            norm_type: NormType::RMSNorm,
            norm_eps: 1e-5,
            activation: Activation::SwiGLU,
            position_type: PositionType::RoPE,
            rope_freq_base: 10000.0,
            rope_dim: 64,
            rope_neox: false,
            rope_scaling_original_ctx: 0,
            rope_scaling_attn_factor: 1.0,
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

    fn gpt2_1layer_config() -> ModelConfig {
        ModelConfig {
            num_layers: 1,
            ..gpt2_config()
        }
    }

    fn llama_1layer_config() -> ModelConfig {
        ModelConfig {
            num_layers: 1,
            ..llama_config()
        }
    }

    // -----------------------------------------------------------------------
    // Weight helpers
    // -----------------------------------------------------------------------

    fn gpt2_weights(config: &ModelConfig) -> ModelWeights {
        let h = config.hidden_size;
        let ffn = config.ffn_hidden;
        let v = config.vocab_size;
        let qtype = TensorDtype::Q8_0;

        let layers: Vec<LayerWeights> = (0..config.num_layers)
            .map(|_| LayerWeights {
                attn_norm_w: Some(dt(vec![h], TensorDtype::F32)),
                attn_norm_b: Some(dt(vec![h], TensorDtype::F32)),
                ffn_norm_w: Some(dt(vec![h], TensorDtype::F32)),
                ffn_norm_b: Some(dt(vec![h], TensorDtype::F32)),
                attn_output_norm_w: None,
                attn_output_norm_b: None,
                ffn_output_norm_w: None,
                ffn_output_norm_b: None,
                attn_q: dt(vec![h, h], qtype),
                attn_k: dt(vec![h, h], qtype),
                attn_v: dt(vec![h, h], qtype),
                attn_output: dt(vec![h, h], qtype),
                attn_q_bias: Some(dt(vec![h], TensorDtype::F32)),
                attn_k_bias: Some(dt(vec![h], TensorDtype::F32)),
                attn_v_bias: Some(dt(vec![h], TensorDtype::F32)),
                attn_output_bias: Some(dt(vec![h], TensorDtype::F32)),
                ffn_up: dt(vec![ffn, h], qtype),
                ffn_down: dt(vec![h, ffn], qtype),
                ffn_gate: None,
                ffn_up_bias: Some(dt(vec![ffn], TensorDtype::F32)),
                ffn_down_bias: Some(dt(vec![h], TensorDtype::F32)),
                attn_q_norm_w: None,
                attn_k_norm_w: None,
                attn_post_norm_w: None,
                ffn_post_norm_w: None,
            })
            .collect();

        ModelWeights {
            token_embedding: dt(vec![v, h], TensorDtype::F32),
            position_embedding: Some(dt(vec![1024, h], TensorDtype::F32)),
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers,
            output_norm_w: Some(dt(vec![h], TensorDtype::F32)),
            output_norm_b: Some(dt(vec![h], TensorDtype::F32)),
            output_projection: None, // tied embeddings
            rope_factors_short: None,
            rope_factors_long: None,
        }
    }

    fn llama_weights(config: &ModelConfig) -> ModelWeights {
        let h = config.hidden_size;
        let ffn = config.ffn_hidden;
        let v = config.vocab_size;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let total_dim = config.num_heads * config.head_dim;
        let qtype = TensorDtype::Q8_0;

        let layers: Vec<LayerWeights> = (0..config.num_layers)
            .map(|_| LayerWeights {
                attn_norm_w: Some(dt(vec![h], TensorDtype::F32)),
                attn_norm_b: None,
                ffn_norm_w: Some(dt(vec![h], TensorDtype::F32)),
                ffn_norm_b: None,
                attn_output_norm_w: None,
                attn_output_norm_b: None,
                ffn_output_norm_w: None,
                ffn_output_norm_b: None,
                attn_q: dt(vec![total_dim, h], qtype),
                attn_k: dt(vec![kv_dim, h], qtype),
                attn_v: dt(vec![kv_dim, h], qtype),
                attn_output: dt(vec![h, total_dim], qtype),
                attn_q_bias: None,
                attn_k_bias: None,
                attn_v_bias: None,
                attn_output_bias: None,
                ffn_up: dt(vec![ffn, h], qtype),
                ffn_down: dt(vec![h, ffn], qtype),
                ffn_gate: Some(dt(vec![ffn, h], qtype)),
                ffn_up_bias: None,
                ffn_down_bias: None,
                attn_q_norm_w: None,
                attn_k_norm_w: None,
                attn_post_norm_w: None,
                ffn_post_norm_w: None,
            })
            .collect();

        ModelWeights {
            token_embedding: dt(vec![v, h], TensorDtype::F32),
            position_embedding: None, // RoPE, no learned positions
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers,
            output_norm_w: Some(dt(vec![h], TensorDtype::F32)),
            output_norm_b: None,
            output_projection: None, // tied embeddings
            rope_factors_short: None,
            rope_factors_long: None,
        }
    }

    // -----------------------------------------------------------------------
    // Helper: count ops by PSO type
    // -----------------------------------------------------------------------

    fn count_pso(graph: &DecodeGraph, pso: PsoRef) -> usize {
        graph.ops.iter().filter(|op| op.pso == pso).count()
    }

    /// Check that a specific op has the expected PSO.
    fn assert_op_pso(graph: &DecodeGraph, idx: usize, expected: PsoRef) {
        assert_eq!(
            graph.ops[idx].pso, expected,
            "Op[{}] expected {:?}, got {:?}",
            idx, expected, graph.ops[idx].pso
        );
    }

    // -----------------------------------------------------------------------
    // GPT-2 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gpt2_graph_structure() {
        let config = gpt2_config();
        let weights = gpt2_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        // GPT-2 12 layers (with fused matmul+bias):
        //   Pre-layer: embedding_lookup + embedding_lookup(pos) + add = 3 ops
        //   Per layer (14 ops): layer_norm + qmatmul_bias(Q) + qmatmul_bias(K) +
        //     qmatmul_bias(V) + copy_k + copy_v + grouped_attn + qmatmul_bias(O) +
        //     add(residual) + layer_norm(FFN) + qmatmul_bias(up) + gelu +
        //     qmatmul_bias(down) + add(residual)
        //   Post-layer: layer_norm(final) + matmul(logits) = 2 ops
        //   Total = 3 + 12*14 + 2 = 173
        assert_eq!(
            graph.ops.len(), 173,
            "GPT-2 12-layer should have exactly 173 ops, got {}",
            graph.ops.len()
        );

        // Barriers should be fewer than ops (concurrent dispatch benefit)
        assert!(
            graph.barriers.len() < graph.ops.len(),
            "Barriers ({}) should be < ops ({})",
            graph.barriers.len(),
            graph.ops.len()
        );

        // Logits slot should be valid
        assert!(
            (graph.logits_slot.0 as usize) < graph.slot_sizes.len(),
            "logits_slot out of bounds"
        );

        assert_eq!(graph.logits_count, 50257);
    }

    #[test]
    fn test_gpt2_1layer_op_sequence() {
        let config = gpt2_1layer_config();
        let weights = gpt2_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        // 3 + 1*14 + 2 = 19 ops (fused matmul+bias eliminates 6 add_bias)
        assert_eq!(graph.ops.len(), 19, "GPT-2 1-layer: expected 19 ops, got {}", graph.ops.len());

        // Verify first three ops are embedding_lookup, embedding_lookup(pos), add
        assert_op_pso(&graph, 0, PsoRef::EmbeddingLookup);
        assert_op_pso(&graph, 1, PsoRef::EmbeddingLookup);
        assert_op_pso(&graph, 2, PsoRef::AddTensor);

        // First embedding lookup should use TokenId (token embedding)
        let has_token_id = graph.ops[0].params.iter().any(|(p, _)| matches!(p, ParamValue::TokenId));
        assert!(has_token_id, "Token embedding lookup should use TokenId param");

        // Second embedding lookup should use PositionIdBuffer (position embedding)
        let has_pos_id = graph.ops[1].params.iter().any(|(p, _)| matches!(p, ParamValue::PositionIdBuffer));
        assert!(has_pos_id, "Position embedding lookup should use PositionIdBuffer param");

        // Verify the layer starts with LayerNorm (pre-norm GPT-2)
        assert_op_pso(&graph, 3, PsoRef::LayerNorm);

        // Verify fused QKV projections (matmul+bias in one kernel)
        assert_op_pso(&graph, 4, PsoRef::QuantizedMatmulBiasQ8_0); // Q+bias
        assert_op_pso(&graph, 5, PsoRef::QuantizedMatmulBiasQ8_0); // K+bias
        assert_op_pso(&graph, 6, PsoRef::QuantizedMatmulBiasQ8_0); // V+bias

        // Copy to cache
        assert_op_pso(&graph, 7, PsoRef::CopyBuffer); // K cache
        assert_op_pso(&graph, 8, PsoRef::CopyBuffer); // V cache

        // Attention
        assert_op_pso(&graph, 9, PsoRef::GroupedAttnDecode);

        // Output projection + residual (fused matmul+bias)
        assert_op_pso(&graph, 10, PsoRef::QuantizedMatmulBiasQ8_0); // O+bias
        assert_op_pso(&graph, 11, PsoRef::AddTensor);                // residual

        // FFN pre-norm
        assert_op_pso(&graph, 12, PsoRef::LayerNorm);

        // FFN: fused up+bias + gelu + fused down+bias + residual
        assert_op_pso(&graph, 13, PsoRef::QuantizedMatmulBiasQ8_0); // up+bias
        assert_op_pso(&graph, 14, PsoRef::Gelu);
        assert_op_pso(&graph, 15, PsoRef::QuantizedMatmulBiasQ8_0); // down+bias
        assert_op_pso(&graph, 16, PsoRef::AddTensor);                // FFN residual

        // Post-layer: final norm + logits
        assert_op_pso(&graph, 17, PsoRef::LayerNorm);
        assert_op_pso(&graph, 18, PsoRef::MatmulTranspose); // tied embeddings → F32
    }

    #[test]
    fn test_gpt2_pso_counts() {
        let config = gpt2_config();
        let weights = gpt2_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        // Per layer: all 6 matmuls (Q,K,V,O,up,down) are fused matmul+bias
        assert_eq!(count_pso(&graph, PsoRef::QuantizedMatmulBiasQ8_0), 6 * 12);

        // No separate add_bias ops (all fused)
        assert_eq!(count_pso(&graph, PsoRef::AddBias), 0);

        // No unfused quantized matmul (all have bias for GPT-2)
        assert_eq!(count_pso(&graph, PsoRef::QuantizedMatmulQ8_0), 0);

        // Per layer: 1 gelu
        assert_eq!(count_pso(&graph, PsoRef::Gelu), 12);

        // Per layer: 2 layer_norm (attn pre-norm + FFN pre-norm) + 1 final
        assert_eq!(count_pso(&graph, PsoRef::LayerNorm), 2 * 12 + 1);

        // 2 embedding lookups (token + position)
        assert_eq!(count_pso(&graph, PsoRef::EmbeddingLookup), 2);

        // Per layer: 2 copy_buffer (K, V)
        assert_eq!(count_pso(&graph, PsoRef::CopyBuffer), 2 * 12);

        // Per layer: 1 grouped_attn + 2 add_tensor (attn residual + FFN residual)
        assert_eq!(count_pso(&graph, PsoRef::GroupedAttnDecode), 12);
        assert_eq!(count_pso(&graph, PsoRef::AddTensor), 2 * 12 + 1); // +1 for pos+token add

        // 1 matmul_transpose for logits (tied embeddings = F32)
        assert_eq!(count_pso(&graph, PsoRef::MatmulTranspose), 1);

        // No RoPE, SwiGLU, RMSNorm, scale for GPT-2
        assert_eq!(count_pso(&graph, PsoRef::RopeNorm), 0);
        assert_eq!(count_pso(&graph, PsoRef::RopeNeox), 0);
        assert_eq!(count_pso(&graph, PsoRef::SwiGlu), 0);
        assert_eq!(count_pso(&graph, PsoRef::RmsNorm), 0);
        assert_eq!(count_pso(&graph, PsoRef::ScaleKernel), 0);
    }

    // -----------------------------------------------------------------------
    // LLaMA/Gemma tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_llama_graph_structure() {
        let config = llama_config();
        let weights = llama_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        // LLaMA 22 layers:
        //   Pre-layer: embedding_lookup = 1 op (no pos embedding, no scale)
        //   Per layer (18 ops): rms_norm + qmatmul(Q) + qmatmul(K) + qmatmul(V)
        //     + rope(Q) + rope(K) + copy_k + copy_v + grouped_attn
        //     + qmatmul(O) + add(residual) + rms_norm(FFN)
        //     + qmatmul(gate) + qmatmul(up) + swiglu + qmatmul(down) + add(residual)
        //   Actually let me count more carefully:
        //   rms_norm(attn) + qmatmul_Q + qmatmul_K + qmatmul_V +
        //   rope_Q + rope_K + copy_K + copy_V + grouped_attn +
        //   qmatmul_O + add(residual) + rms_norm(FFN) +
        //   qmatmul_gate + qmatmul_up + swiglu + qmatmul_down + add(residual) = 17
        //   Post-layer: rms_norm(final) + matmul(logits) = 2 ops
        //   Total = 1 + 22*17 + 2 = 377
        assert_eq!(
            graph.ops.len(), 377,
            "LLaMA 22-layer should have exactly 377 ops, got {}",
            graph.ops.len()
        );

        assert_eq!(graph.logits_count, 32000);

        // Should have RoPE, SwiGLU, RMSNorm, no LayerNorm/Gelu/AddBias
        assert_eq!(count_pso(&graph, PsoRef::RopeNorm), 2 * 22); // Q+K per layer
        assert_eq!(count_pso(&graph, PsoRef::SwiGlu), 22);
        assert_eq!(count_pso(&graph, PsoRef::RmsNorm), 2 * 22 + 1); // attn+FFN per layer + final
        assert_eq!(count_pso(&graph, PsoRef::LayerNorm), 0);
        assert_eq!(count_pso(&graph, PsoRef::Gelu), 0);
        assert_eq!(count_pso(&graph, PsoRef::AddBias), 0);
        assert_eq!(count_pso(&graph, PsoRef::EmbeddingLookup), 1); // no position embedding
    }

    #[test]
    fn test_llama_1layer_op_sequence() {
        let config = llama_1layer_config();
        let weights = llama_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        // 1 + 1*17 + 2 = 20 ops
        assert_eq!(graph.ops.len(), 20, "LLaMA 1-layer: expected 20 ops, got {}", graph.ops.len());

        // Embedding lookup (token only, no position)
        assert_op_pso(&graph, 0, PsoRef::EmbeddingLookup);

        // Layer: pre-norm
        assert_op_pso(&graph, 1, PsoRef::RmsNorm);

        // QKV without biases
        assert_op_pso(&graph, 2, PsoRef::QuantizedMatmulQ8_0); // Q
        assert_op_pso(&graph, 3, PsoRef::QuantizedMatmulQ8_0); // K
        assert_op_pso(&graph, 4, PsoRef::QuantizedMatmulQ8_0); // V

        // RoPE on Q and K
        assert_op_pso(&graph, 5, PsoRef::RopeNorm); // Q
        assert_op_pso(&graph, 6, PsoRef::RopeNorm); // K

        // Copy to KV cache
        assert_op_pso(&graph, 7, PsoRef::CopyBuffer);
        assert_op_pso(&graph, 8, PsoRef::CopyBuffer);

        // Attention
        assert_op_pso(&graph, 9, PsoRef::GroupedAttnDecode);

        // Output projection + residual (no bias)
        assert_op_pso(&graph, 10, PsoRef::QuantizedMatmulQ8_0);
        assert_op_pso(&graph, 11, PsoRef::AddTensor); // residual

        // FFN pre-norm
        assert_op_pso(&graph, 12, PsoRef::RmsNorm);

        // FFN: SwiGLU path (gate + up + swiglu + down)
        assert_op_pso(&graph, 13, PsoRef::QuantizedMatmulQ8_0); // gate
        assert_op_pso(&graph, 14, PsoRef::QuantizedMatmulQ8_0); // up
        assert_op_pso(&graph, 15, PsoRef::SwiGlu);
        assert_op_pso(&graph, 16, PsoRef::QuantizedMatmulQ8_0); // down
        assert_op_pso(&graph, 17, PsoRef::AddTensor); // FFN residual

        // Post-layer: final norm + logits
        assert_op_pso(&graph, 18, PsoRef::RmsNorm);
        assert_op_pso(&graph, 19, PsoRef::MatmulTranspose);
    }

    // -----------------------------------------------------------------------
    // Phi-3 tests
    // -----------------------------------------------------------------------

    fn phi3_config() -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Phi3,
            arch_name: "phi3".to_string(),
            hidden_size: 3072,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32, // MHA
            head_dim: 96,
            ffn_hidden: 8192,
            vocab_size: 32064,
            max_seq_len: 4096,
            norm_type: NormType::RMSNorm,
            norm_eps: 1e-5,
            activation: Activation::SwiGLU,
            position_type: PositionType::RoPE,
            rope_freq_base: 10000.0,
            rope_dim: 96,
            rope_neox: true,
            rope_scaling_original_ctx: 0,
            rope_scaling_attn_factor: 1.0,
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

    fn phi3_1layer_config() -> ModelConfig {
        ModelConfig {
            num_layers: 1,
            ..phi3_config()
        }
    }

    fn phi3_weights(config: &ModelConfig) -> ModelWeights {
        let h = config.hidden_size;
        let ffn = config.ffn_hidden;
        let v = config.vocab_size;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let total_dim = config.num_heads * config.head_dim;
        let qtype = TensorDtype::Q8_0;

        let layers: Vec<LayerWeights> = (0..config.num_layers)
            .map(|_| LayerWeights {
                attn_norm_w: Some(dt(vec![h], TensorDtype::F32)),
                attn_norm_b: None,
                ffn_norm_w: Some(dt(vec![h], TensorDtype::F32)),
                ffn_norm_b: None,
                attn_output_norm_w: None,
                attn_output_norm_b: None,
                ffn_output_norm_w: None,
                ffn_output_norm_b: None,
                attn_q: dt(vec![total_dim, h], qtype),
                attn_k: dt(vec![kv_dim, h], qtype),
                attn_v: dt(vec![kv_dim, h], qtype),
                attn_output: dt(vec![h, total_dim], qtype),
                attn_q_bias: None,
                attn_k_bias: None,
                attn_v_bias: None,
                attn_output_bias: None,
                ffn_up: dt(vec![ffn, h], qtype),
                ffn_down: dt(vec![h, ffn], qtype),
                ffn_gate: Some(dt(vec![ffn, h], qtype)),
                ffn_up_bias: None,
                ffn_down_bias: None,
                attn_q_norm_w: None,
                attn_k_norm_w: None,
                attn_post_norm_w: None,
                ffn_post_norm_w: None,
            })
            .collect();

        ModelWeights {
            token_embedding: dt(vec![v, h], TensorDtype::F32),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers,
            output_norm_w: Some(dt(vec![h], TensorDtype::F32)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        }
    }

    fn phi3_longrope_weights(config: &ModelConfig) -> ModelWeights {
        let half_rope = config.rope_dim / 2;
        ModelWeights {
            rope_factors_short: Some(dt(vec![half_rope], TensorDtype::F32)),
            rope_factors_long: Some(dt(vec![half_rope], TensorDtype::F32)),
            ..phi3_weights(config)
        }
    }

    #[test]
    fn test_phi3_graph_structure() {
        let config = phi3_config();
        let weights = phi3_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        // Phi-3 32 layers:
        //   Pre-layer: embedding_lookup = 1 op
        //   Per layer (17 ops): rms_norm + qmatmul(Q) + qmatmul(K) + qmatmul(V) +
        //     rope_neox(Q) + rope_neox(K) + copy_k + copy_v + grouped_attn +
        //     qmatmul(O) + add(residual) + rms_norm(FFN) +
        //     qmatmul(gate) + qmatmul(up) + swiglu + qmatmul(down) + add(residual) = 17
        //   Post-layer: rms_norm(final) + matmul(logits) = 2 ops
        //   Total = 1 + 32*17 + 2 = 547
        assert_eq!(
            graph.ops.len(), 547,
            "Phi-3 32-layer should have exactly 547 ops, got {}",
            graph.ops.len()
        );

        // PSO distribution
        assert_eq!(count_pso(&graph, PsoRef::RopeNeox), 2 * 32); // Q+K per layer
        assert_eq!(count_pso(&graph, PsoRef::SwiGlu), 32);
        assert_eq!(count_pso(&graph, PsoRef::RmsNorm), 2 * 32 + 1); // attn+FFN per layer + final
        assert_eq!(count_pso(&graph, PsoRef::LayerNorm), 0);
        assert_eq!(count_pso(&graph, PsoRef::Gelu), 0);
        assert_eq!(count_pso(&graph, PsoRef::AddBias), 0);
        assert_eq!(count_pso(&graph, PsoRef::EmbeddingLookup), 1);
    }

    #[test]
    fn test_phi3_1layer_op_sequence() {
        let config = phi3_1layer_config();
        let weights = phi3_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        // 1 + 1*17 + 2 = 20 ops
        assert_eq!(graph.ops.len(), 20, "Phi-3 1-layer: expected 20 ops, got {}", graph.ops.len());

        // Embedding lookup
        assert_op_pso(&graph, 0, PsoRef::EmbeddingLookup);

        // Layer: pre-norm
        assert_op_pso(&graph, 1, PsoRef::RmsNorm);

        // QKV without biases
        assert_op_pso(&graph, 2, PsoRef::QuantizedMatmulQ8_0); // Q
        assert_op_pso(&graph, 3, PsoRef::QuantizedMatmulQ8_0); // K
        assert_op_pso(&graph, 4, PsoRef::QuantizedMatmulQ8_0); // V

        // RoPE NeoX on Q and K
        assert_op_pso(&graph, 5, PsoRef::RopeNeox); // Q
        assert_op_pso(&graph, 6, PsoRef::RopeNeox); // K

        // Copy to KV cache
        assert_op_pso(&graph, 7, PsoRef::CopyBuffer);
        assert_op_pso(&graph, 8, PsoRef::CopyBuffer);

        // Attention
        assert_op_pso(&graph, 9, PsoRef::GroupedAttnDecode);

        // Output projection + residual
        assert_op_pso(&graph, 10, PsoRef::QuantizedMatmulQ8_0);
        assert_op_pso(&graph, 11, PsoRef::AddTensor);

        // FFN pre-norm
        assert_op_pso(&graph, 12, PsoRef::RmsNorm);

        // FFN: SwiGLU path
        assert_op_pso(&graph, 13, PsoRef::QuantizedMatmulQ8_0); // gate
        assert_op_pso(&graph, 14, PsoRef::QuantizedMatmulQ8_0); // up
        assert_op_pso(&graph, 15, PsoRef::SwiGlu);
        assert_op_pso(&graph, 16, PsoRef::QuantizedMatmulQ8_0); // down
        assert_op_pso(&graph, 17, PsoRef::AddTensor);

        // Post-layer
        assert_op_pso(&graph, 18, PsoRef::RmsNorm);
        assert_op_pso(&graph, 19, PsoRef::MatmulTranspose);
    }

    #[test]
    fn test_phi3_rope_uses_neox() {
        let config = phi3_1layer_config();
        let weights = phi3_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        // Should use RopeNeox (not RopeNorm)
        assert_eq!(count_pso(&graph, PsoRef::RopeNeox), 2);
        assert_eq!(count_pso(&graph, PsoRef::RopeNorm), 0);
    }

    #[test]
    fn test_phi3_no_bias_ops() {
        let config = phi3_1layer_config();
        let weights = phi3_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        // No fused matmul+bias PSOs should appear (Phi-3 has no biases)
        assert_eq!(count_pso(&graph, PsoRef::QuantizedMatmulBiasQ8_0), 0);
        assert_eq!(count_pso(&graph, PsoRef::QuantizedMatmulBiasQ4_0), 0);
        assert_eq!(count_pso(&graph, PsoRef::MatmulTransposeBias), 0);
        assert_eq!(count_pso(&graph, PsoRef::AddBias), 0);
    }

    #[test]
    fn test_phi3_longrope_factors() {
        let config = phi3_1layer_config();
        let weights = phi3_longrope_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        // With rope factors, should use RopeNeoxFactors instead of RopeNeox
        assert_eq!(count_pso(&graph, PsoRef::RopeNeoxFactors), 2);
        assert_eq!(count_pso(&graph, PsoRef::RopeNeox), 0);
        assert_eq!(count_pso(&graph, PsoRef::RopeNorm), 0);
    }

    #[test]
    fn test_phi3_weight_walk_order_consistency() {
        let config = phi3_config();
        let weights = phi3_weights(&config);

        let graph = DecodeGraph::build(&config, &weights, false);
        let walked = weight_walk_order(&weights);

        let max_weight_idx = graph.ops.iter()
            .flat_map(|op| {
                op.bindings.iter().map(|(br, _, _)| br)
                    .chain(op.reads.iter())
            })
            .filter_map(|br| match br {
                BufferRef::Weight(i) => Some(*i as usize),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        assert!(
            max_weight_idx < walked.len(),
            "Weight index {} out of bounds (walked {} weights)",
            max_weight_idx,
            walked.len()
        );
    }

    // -----------------------------------------------------------------------
    // Binding and dispatch dimension tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_embedding_lookup_bindings() {
        let config = gpt2_1layer_config();
        let weights = gpt2_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        // Op 0: token embedding lookup
        let op = &graph.ops[0];
        assert_eq!(op.pso, PsoRef::EmbeddingLookup);
        // Should bind: table(weight) at 0, output(pool) at 2
        assert_eq!(op.bindings.len(), 2);
        assert!(matches!(op.bindings[0].0, BufferRef::Weight(_)));
        assert_eq!(op.bindings[0].1, 0);
        assert!(matches!(op.bindings[1].0, BufferRef::Pool(_)));
        assert_eq!(op.bindings[1].1, 2);
        // Should have params: TokenId at 1, hidden_size at 3, num_tokens=1 at 4
        assert_eq!(op.params.len(), 3);
        assert!(matches!(op.params[0].0, ParamValue::TokenId));
        assert_eq!(op.params[0].1, 1);
        assert!(matches!(op.params[1].0, ParamValue::U32(768)));
        assert_eq!(op.params[1].1, 3);
        assert!(matches!(op.params[2].0, ParamValue::U32(1)));
        assert_eq!(op.params[2].1, 4);

        // Op 1: position embedding lookup
        let op = &graph.ops[1];
        assert_eq!(op.pso, PsoRef::EmbeddingLookup);
        // Should have PositionIdBuffer at binding 1
        assert!(matches!(op.params[0].0, ParamValue::PositionIdBuffer));
        assert_eq!(op.params[0].1, 1);
    }

    #[test]
    fn test_copy_buffer_has_cache_row_offset() {
        let config = gpt2_1layer_config();
        let weights = gpt2_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        // Find copy_buffer ops (should be at indices 7, 8 in GPT-2 1-layer with fused bias)
        let copy_ops: Vec<usize> = graph.ops.iter().enumerate()
            .filter(|(_, op)| op.pso == PsoRef::CopyBuffer)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(copy_ops.len(), 2);

        for &idx in &copy_ops {
            let op = &graph.ops[idx];
            // Should bind: src(pool) at 0, kv_cache at 1
            assert!(matches!(op.bindings[0].0, BufferRef::Pool(_)));
            assert_eq!(op.bindings[0].1, 0);
            assert!(matches!(op.bindings[1].0, BufferRef::KvCache(_)));
            assert_eq!(op.bindings[1].1, 1);
            // Should have CacheRowOffset param at binding 3
            let has_offset = op.params.iter().any(|(p, b)| matches!(p, ParamValue::CacheRowOffset) && *b == 3);
            assert!(has_offset, "copy_buffer should have CacheRowOffset at binding 3");
        }
    }

    #[test]
    fn test_grouped_attn_has_total_len() {
        let config = gpt2_1layer_config();
        let weights = gpt2_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        let attn_op = graph.ops.iter()
            .find(|op| op.pso == PsoRef::GroupedAttnDecode)
            .expect("Should have a grouped_attn_decode op");

        // Should have TotalLen param at binding 7
        let has_total_len = attn_op.params.iter().any(|(p, b)| matches!(p, ParamValue::TotalLen) && *b == 7);
        assert!(has_total_len, "grouped_attn should have TotalLen at binding 7");

        // Should bind Q(pool) at 0, K_cache at 1, V_cache at 2, out(pool) at 3
        assert_eq!(attn_op.bindings.len(), 4);
        assert!(matches!(attn_op.bindings[0].0, BufferRef::Pool(_)));
        assert!(matches!(attn_op.bindings[1].0, BufferRef::KvCache(_)));
        assert!(matches!(attn_op.bindings[2].0, BufferRef::KvCache(_)));
        assert!(matches!(attn_op.bindings[3].0, BufferRef::Pool(_)));

        // Dispatch should be 1 group per head
        match attn_op.dispatch {
            DispatchDims::Fixed { gx, tx, .. } => {
                assert_eq!(gx, 12); // num_heads for GPT-2
                assert_eq!(tx, 256);
            }
            _ => panic!("grouped_attn dispatch should be Fixed"),
        }
    }

    #[test]
    fn test_rope_dispatch_dims() {
        let config = llama_1layer_config();
        let weights = llama_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        let rope_ops: Vec<&DecodeOp> = graph.ops.iter()
            .filter(|op| op.pso == PsoRef::RopeNorm)
            .collect();
        assert_eq!(rope_ops.len(), 2); // Q and K

        // Q rope: n_heads=32, head_dim=64, rope_dim=64, half_rope=32
        match rope_ops[0].dispatch {
            DispatchDims::Fixed { gx, gy, gz, tx, ty, .. } => {
                assert_eq!(gx, (32 + 15) / 16); // half_rope / 16 = 2
                assert_eq!(gy, (32 + 15) / 16); // num_heads(32) / 16 = 2
                assert_eq!(gz, 1); // seq_len = 1
                assert_eq!(tx, 16);
                assert_eq!(ty, 16);
            }
            _ => panic!("rope dispatch should be Fixed"),
        }

        // K rope: n_kv_heads=4 (GQA)
        match rope_ops[1].dispatch {
            DispatchDims::Fixed { gx, gy, .. } => {
                assert_eq!(gx, 2); // half_rope / 16
                assert_eq!(gy, (4 + 15) / 16); // num_kv_heads / 16 = 1
            }
            _ => panic!("rope dispatch should be Fixed"),
        }

        // Should have PositionId at binding 2 (as bytes, not buffer)
        let has_pos = rope_ops[0].params.iter().any(|(p, b)| matches!(p, ParamValue::PositionId) && *b == 2);
        assert!(has_pos, "RoPE should have PositionId at binding 2");
    }

    #[test]
    fn test_qmatmul_dispatch_dims() {
        let config = llama_1layer_config();
        let weights = llama_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        // Find the Q projection qmatmul (op index 2: after embedding + rms_norm)
        let op = &graph.ops[2];
        assert_eq!(op.pso, PsoRef::QuantizedMatmulQ8_0);

        // Q8_0: threadgroups = (n + 7) / 8, threads = 128
        let total_dim = 32 * 64; // num_heads * head_dim = 2048
        match op.dispatch {
            DispatchDims::Fixed { gx, tx, .. } => {
                assert_eq!(gx, ((total_dim + 7) / 8) as u32);
                assert_eq!(tx, 128);
            }
            _ => panic!("qmatmul dispatch should be Fixed"),
        }
    }

    // -----------------------------------------------------------------------
    // Weight walk order tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_weight_walk_order_consistency() {
        let config = gpt2_config();
        let weights = gpt2_weights(&config);

        let graph = DecodeGraph::build(&config, &weights, false);
        let walked = weight_walk_order(&weights);

        // The graph builder should have consumed exactly as many weights
        // as weight_walk_order returns
        let max_weight_idx = graph.ops.iter()
            .flat_map(|op| {
                op.bindings.iter().map(|(br, _, _)| br)
                    .chain(op.reads.iter())
            })
            .filter_map(|br| match br {
                BufferRef::Weight(i) => Some(*i as usize),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        assert!(
            max_weight_idx < walked.len(),
            "Weight index {} out of bounds (walked {} weights)",
            max_weight_idx,
            walked.len()
        );
    }

    #[test]
    fn test_weight_walk_exact_count_gpt2() {
        let config = gpt2_config();
        let weights = gpt2_weights(&config);
        let walked = weight_walk_order(&weights);

        // GPT-2 global weights: token_emb + pos_emb + output_norm_w + output_norm_b = 4
        // (no output_projection — tied embeddings)
        // Per layer: attn_norm_w + attn_norm_b + Q + Q_bias + K + K_bias + V + V_bias
        //   + O + O_bias + ffn_norm_w + ffn_norm_b + up + up_bias + down + down_bias = 16
        // Total: 4 + 12*16 = 196
        assert_eq!(walked.len(), 196, "GPT-2 should walk 196 weights, got {}", walked.len());
    }

    #[test]
    fn test_weight_walk_exact_count_llama() {
        let config = llama_config();
        let weights = llama_weights(&config);
        let walked = weight_walk_order(&weights);

        // LLaMA global: token_emb + output_norm_w = 2
        // Per layer: attn_norm_w + Q + K + V + O + ffn_norm_w + up + down + gate = 9
        // Total: 2 + 22*9 = 200
        assert_eq!(walked.len(), 200, "LLaMA should walk 200 weights, got {}", walked.len());
    }

    // -----------------------------------------------------------------------
    // Barrier analysis tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_barriers_no_false_positives() {
        let config = gpt2_1layer_config();
        let weights = gpt2_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        // No barrier should be at index 0 (nothing to barrier before the first op)
        assert!(
            !graph.barriers.contains(&0),
            "There should be no barrier before the first op"
        );

        // All barrier indices should be valid op indices
        for &idx in &graph.barriers {
            assert!(idx < graph.ops.len(), "Barrier index {} out of bounds", idx);
        }

        // Barriers should be sorted (monotonically increasing)
        for i in 1..graph.barriers.len() {
            assert!(
                graph.barriers[i] > graph.barriers[i - 1],
                "Barriers should be sorted: [{}]={} <= [{}]={}",
                i - 1, graph.barriers[i - 1], i, graph.barriers[i]
            );
        }
    }

    #[test]
    fn test_barrier_reduction_gpt2() {
        let config = gpt2_config();
        let weights = gpt2_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        // The old MetalBackend inserts a barrier before EVERY dispatch (245 barriers).
        // The graph should have fewer barriers due to concurrent-safe ops (e.g., Q/K/V
        // matmuls all read the same normed input without conflicts).
        let reduction_pct = 100.0 * (1.0 - graph.barriers.len() as f64 / graph.ops.len() as f64);
        assert!(
            reduction_pct > 10.0,
            "Expected >10% barrier reduction, got {:.1}% ({} barriers / {} ops)",
            reduction_pct, graph.barriers.len(), graph.ops.len()
        );
    }

    // -----------------------------------------------------------------------
    // Embedding scale test
    // -----------------------------------------------------------------------

    #[test]
    fn test_embedding_scale() {
        let mut config = llama_1layer_config();
        config.embedding_scale = (config.hidden_size as f32).sqrt();
        let weights = llama_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        // With embedding_scale != 1.0: embedding_lookup + scale + ... layer ... + norm + logits
        // = 1 + 1 + 17 + 2 = 21
        assert_eq!(graph.ops.len(), 21, "With scale: expected 21 ops, got {}", graph.ops.len());
        assert_op_pso(&graph, 0, PsoRef::EmbeddingLookup);
        assert_op_pso(&graph, 1, PsoRef::ScaleKernel);
        assert_op_pso(&graph, 2, PsoRef::RmsNorm); // layer starts

        // Scale kernel should have the correct factor
        let scale_op = &graph.ops[1];
        let factor = scale_op.params.iter().find_map(|(p, _)| match p {
            ParamValue::F32(v) => Some(*v),
            _ => None,
        });
        assert!(
            (factor.unwrap() - config.embedding_scale).abs() < 1e-6,
            "Scale factor should match embedding_scale"
        );
    }

    // -----------------------------------------------------------------------
    // Logits slot validity
    // -----------------------------------------------------------------------

    #[test]
    fn test_logits_slot_size() {
        let config = gpt2_config();
        let weights = gpt2_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        let slot_idx = graph.logits_slot.0 as usize;
        assert!(slot_idx < graph.slot_sizes.len());
        let expected_bytes = config.vocab_size * std::mem::size_of::<f32>();
        assert_eq!(
            graph.slot_sizes[slot_idx], expected_bytes,
            "Logits slot should be vocab_size * 4 bytes"
        );
    }

    // -----------------------------------------------------------------------
    // KV cache buffer ref tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_kv_cache_indices() {
        let config = gpt2_config();
        let weights = gpt2_weights(&config);
        let graph = DecodeGraph::build(&config, &weights, false);

        // Collect all KvCache references
        let kv_indices: Vec<u16> = graph.ops.iter()
            .flat_map(|op| {
                op.bindings.iter().map(|(br, _, _)| br)
                    .chain(op.reads.iter())
                    .chain(op.writes.iter())
            })
            .filter_map(|br| match br {
                BufferRef::KvCache(i) => Some(*i),
                _ => None,
            })
            .collect();

        // Should cover indices 0..num_layers*2
        let max_kv = kv_indices.iter().max().copied().unwrap_or(0);
        assert_eq!(
            max_kv as usize, (config.num_layers * 2) - 1,
            "Max KV cache index should be num_layers*2-1"
        );

        // Each layer should have K (even) and V (odd)
        for layer in 0..config.num_layers {
            let k_idx = (layer * 2) as u16;
            let v_idx = (layer * 2 + 1) as u16;
            assert!(kv_indices.contains(&k_idx), "Missing K cache for layer {}", layer);
            assert!(kv_indices.contains(&v_idx), "Missing V cache for layer {}", layer);
        }
    }

    // ===================================================================
    // PrefillGraph tests
    // ===================================================================

    fn count_prefill_pso(graph: &PrefillGraph, pso: PsoRef) -> usize {
        graph.ops.iter().filter(|op| op.pso == pso).count()
    }

    fn assert_prefill_op_pso(graph: &PrefillGraph, idx: usize, expected: PsoRef) {
        assert_eq!(
            graph.ops[idx].pso, expected,
            "Prefill Op[{}] expected {:?}, got {:?}",
            idx, expected, graph.ops[idx].pso
        );
    }

    // -----------------------------------------------------------------------
    // Basic structure: GPT-2 prefill
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_gpt2_1layer_structure() {
        let config = gpt2_1layer_config();
        let weights = gpt2_weights(&config);
        let m = 8;
        let graph = PrefillGraph::build(&config, &weights, m, false);

        // GPT-2 1-layer prefill with batched quantized GEMM:
        //   Pre-layer: embedding_lookup(tok) + embedding_lookup(pos) + add = 3 ops
        //   Per layer:
        //     norm(1) + Q(1) + K(1) + V(1) + copy_k(1) + copy_v(1) + attn(1)
        //     + O(1) + add(1) + norm(1) + up(1) + gelu(1) + down(1) + add(1) = 8 + 6
        //   Post-layer: norm(1) + copy_last_token(1) + logits_proj(1 for F32 tied emb)
        let expected_per_layer = 8 + 6;
        let expected_total = 3 + config.num_layers * expected_per_layer + 3;
        assert_eq!(
            graph.ops.len(), expected_total,
            "GPT-2 1-layer M={} should have {} ops, got {}",
            m, expected_total, graph.ops.len()
        );

        assert!(graph.logits_count == config.vocab_size);
        assert!((graph.logits_slot.0 as usize) < graph.slot_sizes.len());
    }

    #[test]
    fn test_prefill_llama_1layer_structure() {
        let config = llama_1layer_config();
        let weights = llama_weights(&config);
        let m = 4;
        let graph = PrefillGraph::build(&config, &weights, m, false);

        // LLaMA 1-layer prefill with batched quantized GEMM:
        //   Pre-layer: embedding_lookup(tok) = 1 op (no pos embedding)
        //   Per layer:
        //     rms_norm(1) + Q(1) + K(1) + V(1) + rope_q(1) + rope_k(1)
        //     + copy_k(1) + copy_v(1) + batched_attn(1) + O_proj(1) + add(1)
        //     + rms_norm(1) + gate(1) + up(1) + swiglu(1) + down(1) + add(1) = 10 + 7
        //   Post-layer: rms_norm(1) + copy_last_token(1) + logits_proj(1 for F32)
        let expected_per_layer = 10 + 7;
        let expected_total = 1 + config.num_layers * expected_per_layer + 3;
        assert_eq!(
            graph.ops.len(), expected_total,
            "LLaMA 1-layer M={} should have {} ops, got {}",
            m, expected_total, graph.ops.len()
        );

        // Should have RoPE but no LayerNorm
        assert_eq!(count_prefill_pso(&graph, PsoRef::LayerNorm), 0);
        assert!(count_prefill_pso(&graph, PsoRef::RopeNorm) > 0 || count_prefill_pso(&graph, PsoRef::RopeNeox) > 0);
        assert!(count_prefill_pso(&graph, PsoRef::SwiGlu) > 0);
        assert_eq!(count_prefill_pso(&graph, PsoRef::Gelu), 0);
    }

    // -----------------------------------------------------------------------
    // M=1 prefill should produce same op types as decode (but with prefill params)
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_m1_op_types_match_decode() {
        let config = gpt2_1layer_config();
        let weights = gpt2_weights(&config);

        let decode = DecodeGraph::build(&config, &weights, false);
        let prefill = PrefillGraph::build(&config, &weights, 1, false);

        // With M=1, prefill uses single-row quantized matmul (same PSOs as decode).
        // The only PSO differences should be:
        //   - decode uses GroupedAttnDecode, prefill uses BatchedCausalAttention
        //   - prefill has an extra CopyBuffer op for last-token extraction before logits
        assert_eq!(
            prefill.ops.len(), decode.ops.len() + 1,
            "M=1 prefill should have 1 extra op (last-token extraction) vs decode: decode={}, prefill={}",
            decode.ops.len(), prefill.ops.len()
        );

        // Compare the layer ops (before the final norm/logits section).
        let layer_end = decode.ops.len() - 2; // skip final norm + logits
        for i in 0..layer_end {
            let d = decode.ops[i].pso;
            let p = prefill.ops[i].pso;
            match (d, p) {
                (PsoRef::GroupedAttnDecode, PsoRef::BatchedCausalAttention) => {}
                (PsoRef::GroupedAttnDecodeF16, PsoRef::BatchedCausalAttentionF16) => {}
                _ => {
                    assert_eq!(d, p, "Op[{}] PSO mismatch: decode={:?}, prefill={:?}", i, d, p);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Batched quantized matmul: verify single dispatch per matmul
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_batched_quantized_matmul() {
        let config = gpt2_1layer_config();
        let weights = gpt2_weights(&config);
        let m = 5;
        let graph = PrefillGraph::build(&config, &weights, m, false);

        // GPT-2 has bias, so uses BatchedMatmulBiasQ8_0 (1 dispatch per linear).
        let batched_ops: Vec<(usize, &DecodeOp)> = graph.ops.iter().enumerate()
            .filter(|(_, op)| op.pso == PsoRef::BatchedMatmulBiasQ8_0)
            .collect();

        // GPT-2 1-layer: Q, K, V, O, up, down = 6 linears × 1 dispatch each = 6
        assert_eq!(
            batched_ops.len(), 6,
            "Expected 6 BatchedMatmulBiasQ8_0 ops (6 linears × 1 dispatch), got {}",
            batched_ops.len()
        );

        // Each op should have 3 params: M(3), K(4), N(5)
        for (i, (_, op)) in batched_ops.iter().enumerate() {
            assert_eq!(op.params.len(), 3, "Op {} should have 3 params (M, K, N)", i);
        }

        // Bias binding should be at index 6 (batched kernel layout)
        for (i, (_, op)) in batched_ops.iter().enumerate() {
            let bias_binding = op.bindings.iter().find(|(_, idx, _)| *idx == 6);
            assert!(bias_binding.is_some(), "Op {} should have bias at binding index 6", i);
        }

        // Input/output bindings should have zero offset (entire M×K / M×N buffer)
        for (i, (_, op)) in batched_ops.iter().enumerate() {
            let input_binding = op.bindings.iter().find(|(_, idx, _)| *idx == 1).unwrap();
            let output_binding = op.bindings.iter().find(|(_, idx, _)| *idx == 2).unwrap();
            assert_eq!(input_binding.2, 0, "Op {} input offset should be 0", i);
            assert_eq!(output_binding.2, 0, "Op {} output offset should be 0", i);
        }
    }

    // -----------------------------------------------------------------------
    // Batched attention: verify kv_f16 selects correct PSO
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_batched_attention_f16_pso() {
        let config = llama_1layer_config();
        let weights = llama_weights(&config);

        let graph_f32 = PrefillGraph::build(&config, &weights, 4, false);
        let graph_f16 = PrefillGraph::build(&config, &weights, 4, true);

        let f32_attn_count = count_prefill_pso(&graph_f32, PsoRef::BatchedCausalAttention);
        let f16_attn_count = count_prefill_pso(&graph_f16, PsoRef::BatchedCausalAttentionF16);

        assert_eq!(f32_attn_count, 1, "F32 graph should have 1 BatchedCausalAttention");
        assert_eq!(f16_attn_count, 1, "F16 graph should have 1 BatchedCausalAttentionF16");

        // F32 graph should NOT have F16 attention
        assert_eq!(count_prefill_pso(&graph_f32, PsoRef::BatchedCausalAttentionF16), 0);
        // F16 graph should NOT have F32 attention
        assert_eq!(count_prefill_pso(&graph_f16, PsoRef::BatchedCausalAttention), 0);
    }

    // -----------------------------------------------------------------------
    // Batched attention: verify dispatch dimensions
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_batched_attention_dispatch() {
        let config = llama_1layer_config();
        let weights = llama_weights(&config);
        let m = 7;
        let graph = PrefillGraph::build(&config, &weights, m, false);

        let attn_ops: Vec<&DecodeOp> = graph.ops.iter()
            .filter(|op| op.pso == PsoRef::BatchedCausalAttention)
            .collect();

        assert_eq!(attn_ops.len(), 1);
        let attn = attn_ops[0];

        // Dispatch should be: gx = M * num_heads, gy = 1, gz = 1, tx = 256
        if let DispatchDims::Fixed { gx, gy, gz, tx, .. } = attn.dispatch {
            assert_eq!(gx, (m * config.num_heads) as u32,
                "Attention dispatch gx should be M*num_heads={}", m * config.num_heads);
            assert_eq!(gy, 1);
            assert_eq!(gz, 1);
            assert_eq!(tx, 256);
        } else {
            panic!("Expected Fixed dispatch for batched attention");
        }

        // Should have PrefillNTokens, PrefillTotalLen, PrefillPosOffset params
        let has_n_tokens = attn.params.iter().any(|(p, _)| matches!(p, ParamValue::PrefillNTokens));
        let has_total_len = attn.params.iter().any(|(p, _)| matches!(p, ParamValue::PrefillTotalLen));
        let has_pos_offset = attn.params.iter().any(|(p, _)| matches!(p, ParamValue::PrefillPosOffset));
        assert!(has_n_tokens, "Batched attention missing PrefillNTokens param");
        assert!(has_total_len, "Batched attention missing PrefillTotalLen param");
        assert!(has_pos_offset, "Batched attention missing PrefillPosOffset param");
    }

    // -----------------------------------------------------------------------
    // Last-token extraction: CopyBuffer with source byte offset
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_last_token_extraction() {
        let config = gpt2_1layer_config();
        let weights = gpt2_weights(&config);

        for m in [1, 2, 5, 16] {
            let graph = PrefillGraph::build(&config, &weights, m, false);
            let h = config.hidden_size;

            // The last-token extraction CopyBuffer writes to a Pool slot (not KvCache).
            // It copies h elements (one hidden-state row) with:
            //   - source binding offset = (M-1) * h * 4 bytes
            //   - dest binding offset = 0
            //   - count param = h
            //   - dest_offset param = 0
            // Distinguish it from KV cache copy ops which write to KvCache buffers.
            let expected_src_offset = ((m - 1) * h * 4) as u32;

            let extraction_ops: Vec<&DecodeOp> = graph.ops.iter()
                .filter(|op| op.pso == PsoRef::CopyBuffer)
                .filter(|op| {
                    // Extraction writes to Pool (not KvCache)
                    op.writes.as_ref().map_or(false, |w| matches!(w, BufferRef::Pool(_)))
                    // And the dest binding (index 1) is a Pool reference
                    && op.bindings.iter().any(|(br, idx, _)| *idx == 1 && matches!(br, BufferRef::Pool(_)))
                    // And has source binding offset matching expected
                    && op.bindings.iter().any(|(_, idx, off)| *idx == 0 && *off == expected_src_offset)
                    // And count = h (not M * kv_dim as in cache copies)
                    && op.params.iter().any(|(p, idx)| *idx == 2 && matches!(p, ParamValue::U32(v) if *v == h as u32))
                })
                .collect();

            assert_eq!(
                extraction_ops.len(), 1,
                "M={}: Expected 1 last-token extraction CopyBuffer (Pool-to-Pool, count=h={}, src_offset={}), found {}",
                m, h, expected_src_offset, extraction_ops.len()
            );

            let extraction = extraction_ops[0];

            // Verify count = h
            let count_param = extraction.params.iter().find(|(_, idx)| *idx == 2).unwrap();
            if let ParamValue::U32(count) = count_param.0 {
                assert_eq!(count, h as u32, "M={}: extraction count should be h={}", m, h);
            } else {
                panic!("M={}: extraction count param should be U32", m);
            }

            // Verify dest_offset = 0
            let dest_param = extraction.params.iter().find(|(_, idx)| *idx == 3).unwrap();
            if let ParamValue::U32(offset) = dest_param.0 {
                assert_eq!(offset, 0, "M={}: extraction dest_offset should be 0", m);
            } else {
                panic!("M={}: extraction dest_offset param should be U32", m);
            }
        }
    }

    // -----------------------------------------------------------------------
    // KV cache F16: CopyF32ToF16 vs CopyBuffer
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_kv_cache_f16_vs_f32() {
        let config = llama_1layer_config();
        let weights = llama_weights(&config);
        let m = 4;

        let graph_f32 = PrefillGraph::build(&config, &weights, m, false);
        let graph_f16 = PrefillGraph::build(&config, &weights, m, true);

        // F32 cache: KV writes use CopyBuffer (2 per layer: K, V)
        let f32_copy_count = graph_f32.ops.iter()
            .filter(|op| op.pso == PsoRef::CopyBuffer && op.writes.iter().any(|w| matches!(w, BufferRef::KvCache(_))))
            .count();
        assert_eq!(f32_copy_count, 2, "F32 KV: should have 2 CopyBuffer ops for KV cache");

        // F16 cache: KV writes use CopyF32ToF16
        let f16_copy_count = graph_f16.ops.iter()
            .filter(|op| op.pso == PsoRef::CopyF32ToF16)
            .count();
        assert_eq!(f16_copy_count, 2, "F16 KV: should have 2 CopyF32ToF16 ops");

        // F16 graph should NOT have CopyBuffer writing to KvCache
        let f16_plain_copy_to_kv = graph_f16.ops.iter()
            .filter(|op| op.pso == PsoRef::CopyBuffer && op.writes.iter().any(|w| matches!(w, BufferRef::KvCache(_))))
            .count();
        assert_eq!(f16_plain_copy_to_kv, 0, "F16: CopyBuffer should not write to KV cache");
    }

    // -----------------------------------------------------------------------
    // Buffer slot sizing: intermediates scale with M
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_slot_sizes_scale_with_m() {
        let config = gpt2_1layer_config();
        let weights = gpt2_weights(&config);

        let graph_m1 = PrefillGraph::build(&config, &weights, 1, false);
        let graph_m4 = PrefillGraph::build(&config, &weights, 4, false);

        // M=4 should have more total buffer memory than M=1
        let total_m1: usize = graph_m1.slot_sizes.iter().sum();
        let total_m4: usize = graph_m4.slot_sizes.iter().sum();
        assert!(
            total_m4 > total_m1,
            "M=4 total buffer ({}) should be larger than M=1 ({})",
            total_m4, total_m1
        );

        // The hidden state slot should be exactly M * h * 4 bytes
        let h = config.hidden_size;
        // Slot 0 is typically the first intermediate (token embedding output)
        // but slot ordering varies. Check that at least one slot has the expected size.
        let expected_hidden_m4 = 4 * h * 4; // M=4 * h * sizeof(f32)
        assert!(
            graph_m4.slot_sizes.iter().any(|&s| s == expected_hidden_m4),
            "M=4: expected at least one slot of size {} (4*{}*4), found sizes: {:?}",
            expected_hidden_m4, h, &graph_m4.slot_sizes[..graph_m4.slot_sizes.len().min(10)]
        );
    }

    // -----------------------------------------------------------------------
    // Weight walk order: PrefillGraph references same weights as DecodeGraph
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_weight_walk_order_consistency() {
        let config = gpt2_config();
        let weights = gpt2_weights(&config);

        let walked = weight_walk_order(&weights);
        let graph = PrefillGraph::build(&config, &weights, 8, false);

        let max_weight_idx = graph.ops.iter()
            .flat_map(|op| op.bindings.iter().map(|(br, _, _)| br))
            .filter_map(|br| match br {
                BufferRef::Weight(idx) => Some(*idx as usize),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        assert!(
            max_weight_idx < walked.len(),
            "Prefill graph references weight index {} but only {} walked",
            max_weight_idx, walked.len()
        );
    }

    #[test]
    fn test_prefill_weight_walk_matches_decode() {
        let config = llama_config();
        let weights = llama_weights(&config);

        let decode = DecodeGraph::build(&config, &weights, false);
        let prefill = PrefillGraph::build(&config, &weights, 4, false);

        // Both should reference the same set of weight indices
        let decode_weights: std::collections::BTreeSet<u16> = decode.ops.iter()
            .flat_map(|op| op.bindings.iter().map(|(br, _, _)| br))
            .filter_map(|br| match br {
                BufferRef::Weight(idx) => Some(*idx),
                _ => None,
            })
            .collect();

        let prefill_weights: std::collections::BTreeSet<u16> = prefill.ops.iter()
            .flat_map(|op| op.bindings.iter().map(|(br, _, _)| br))
            .filter_map(|br| match br {
                BufferRef::Weight(idx) => Some(*idx),
                _ => None,
            })
            .collect();

        assert_eq!(
            decode_weights, prefill_weights,
            "Decode and prefill should reference the same weight indices"
        );
    }

    // -----------------------------------------------------------------------
    // Barrier computation for prefill
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_barriers_valid() {
        let config = gpt2_config();
        let weights = gpt2_weights(&config);
        let graph = PrefillGraph::build(&config, &weights, 8, false);

        // No barrier before op 0
        assert!(
            !graph.barriers.contains(&0),
            "Should not have barrier before first op"
        );

        // All barrier indices should be within op range
        for &b in &graph.barriers {
            assert!(b < graph.ops.len(), "Barrier index {} out of range ({})", b, graph.ops.len());
        }

        // Barriers should be sorted
        let mut sorted = graph.barriers.clone();
        sorted.sort();
        assert_eq!(graph.barriers, sorted, "Barriers should be sorted");

        // Should have fewer barriers than ops
        assert!(
            graph.barriers.len() < graph.ops.len(),
            "Barriers ({}) should be fewer than ops ({})",
            graph.barriers.len(), graph.ops.len()
        );
    }

    // -----------------------------------------------------------------------
    // RoPE multi-token dispatch: gz = M
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_rope_dispatch() {
        let config = llama_1layer_config();
        let weights = llama_weights(&config);
        let m = 6;
        let graph = PrefillGraph::build(&config, &weights, m, false);

        let rope_ops: Vec<&DecodeOp> = graph.ops.iter()
            .filter(|op| op.pso == PsoRef::RopeNorm || op.pso == PsoRef::RopeNeox)
            .collect();

        // LLaMA 1-layer should have 2 RoPE ops per layer (Q and K)
        assert_eq!(rope_ops.len(), 2, "Expected 2 RoPE ops, got {}", rope_ops.len());

        for rope in &rope_ops {
            if let DispatchDims::Fixed { gz, .. } = rope.dispatch {
                assert_eq!(gz, m as u32, "RoPE dispatch gz should be M={}, got {}", m, gz);
            } else {
                panic!("Expected Fixed dispatch for RoPE");
            }

            // Should have PrefillPosOffset param
            let has_pos = rope.params.iter().any(|(p, _)| matches!(p, ParamValue::PrefillPosOffset));
            assert!(has_pos, "RoPE should have PrefillPosOffset param");
        }
    }

    // -----------------------------------------------------------------------
    // F32 matmul dispatch uses BM=32
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_f32_matmul_dispatch_bm32() {
        // Create config with F32 weights to exercise the F32 matmul path
        let config = gpt2_1layer_config();
        let h = config.hidden_size;
        let ffn = config.ffn_hidden;

        // Build weights with F32 dtype instead of Q8_0
        let layers: Vec<LayerWeights> = (0..config.num_layers)
            .map(|_| LayerWeights {
                attn_norm_w: Some(dt(vec![h], TensorDtype::F32)),
                attn_norm_b: Some(dt(vec![h], TensorDtype::F32)),
                ffn_norm_w: Some(dt(vec![h], TensorDtype::F32)),
                ffn_norm_b: Some(dt(vec![h], TensorDtype::F32)),
                attn_output_norm_w: None,
                attn_output_norm_b: None,
                ffn_output_norm_w: None,
                ffn_output_norm_b: None,
                attn_q: dt(vec![h, h], TensorDtype::F32),
                attn_k: dt(vec![h, h], TensorDtype::F32),
                attn_v: dt(vec![h, h], TensorDtype::F32),
                attn_output: dt(vec![h, h], TensorDtype::F32),
                attn_q_bias: Some(dt(vec![h], TensorDtype::F32)),
                attn_k_bias: Some(dt(vec![h], TensorDtype::F32)),
                attn_v_bias: Some(dt(vec![h], TensorDtype::F32)),
                attn_output_bias: Some(dt(vec![h], TensorDtype::F32)),
                ffn_up: dt(vec![ffn, h], TensorDtype::F32),
                ffn_down: dt(vec![h, ffn], TensorDtype::F32),
                ffn_gate: None,
                ffn_up_bias: Some(dt(vec![ffn], TensorDtype::F32)),
                ffn_down_bias: Some(dt(vec![h], TensorDtype::F32)),
                attn_q_norm_w: None,
                attn_k_norm_w: None,
                attn_post_norm_w: None,
                ffn_post_norm_w: None,
            })
            .collect();

        let weights = ModelWeights {
            token_embedding: dt(vec![config.vocab_size, h], TensorDtype::F32),
            position_embedding: Some(dt(vec![1024, h], TensorDtype::F32)),
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers,
            output_norm_w: Some(dt(vec![h], TensorDtype::F32)),
            output_norm_b: Some(dt(vec![h], TensorDtype::F32)),
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let m = 10;
        let graph = PrefillGraph::build(&config, &weights, m, false);

        // Find MatmulTransposeBias ops (F32 path)
        let f32_matmul_ops: Vec<&DecodeOp> = graph.ops.iter()
            .filter(|op| op.pso == PsoRef::MatmulTransposeBias || op.pso == PsoRef::MatmulTranspose)
            .collect();

        // Should have F32 matmuls (not quantized)
        assert!(!f32_matmul_ops.is_empty(), "Expected F32 matmul ops");

        // Each F32 matmul should emit exactly 1 op (not M like quantized)
        // And gy should be ceil(M/32) = ceil(10/32) = 1
        for op in &f32_matmul_ops {
            if let DispatchDims::Fixed { gy, .. } = op.dispatch {
                let expected_gy = ((m + 31) / 32) as u32;
                assert_eq!(gy, expected_gy, "F32 matmul gy should be ceil(M/32)={}", expected_gy);
            }
        }

        // Total F32 matmul ops should be one per matmul (not M per matmul)
        // GPT-2 1-layer F32: Q, K, V, O, up, down = 6 matmuls + logits = 7
        assert!(
            f32_matmul_ops.len() <= 7,
            "F32 path should emit 1 op per matmul (not M), got {} ops",
            f32_matmul_ops.len()
        );
    }

    // -----------------------------------------------------------------------
    // Embedding ops use prefill param variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_embedding_uses_prefill_params() {
        let config = gpt2_1layer_config();
        let weights = gpt2_weights(&config);
        let graph = PrefillGraph::build(&config, &weights, 4, false);

        // First op should be EmbeddingLookup for token embedding
        assert_prefill_op_pso(&graph, 0, PsoRef::EmbeddingLookup);

        let emb_op = &graph.ops[0];

        // Should bind PrefillTokenIds (buffer of M token IDs)
        let has_prefill_token_ids = emb_op.params.iter()
            .any(|(p, _)| matches!(p, ParamValue::PrefillTokenIds));
        assert!(has_prefill_token_ids, "Token embedding should use PrefillTokenIds");

        // Should have PrefillNTokens param
        let has_n_tokens = emb_op.params.iter()
            .any(|(p, _)| matches!(p, ParamValue::PrefillNTokens));
        assert!(has_n_tokens, "Token embedding should use PrefillNTokens");

        // GPT-2 has position embedding (op 1)
        assert_prefill_op_pso(&graph, 1, PsoRef::EmbeddingLookup);
        let pos_op = &graph.ops[1];
        let has_prefill_pos_ids = pos_op.params.iter()
            .any(|(p, _)| matches!(p, ParamValue::PrefillPositionIds));
        assert!(has_prefill_pos_ids, "Position embedding should use PrefillPositionIds");
    }

    // -----------------------------------------------------------------------
    // Norm multi-token dispatch: num_rows = M
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_norm_dispatch_rows() {
        let config = gpt2_1layer_config();
        let weights = gpt2_weights(&config);
        let m = 12;
        let graph = PrefillGraph::build(&config, &weights, m, false);

        let norm_ops: Vec<&DecodeOp> = graph.ops.iter()
            .filter(|op| op.pso == PsoRef::LayerNorm || op.pso == PsoRef::RmsNorm)
            .collect();

        // GPT-2 1-layer: 2 LayerNorm per layer + 1 final = 3 norms
        assert!(norm_ops.len() >= 3, "Expected at least 3 norm ops, got {}", norm_ops.len());

        // Norms dispatched with Rows variant should have num_rows = M
        for op in &norm_ops {
            if let DispatchDims::Rows { num_rows, .. } = op.dispatch {
                assert_eq!(
                    num_rows, m as u32,
                    "Norm dispatch num_rows should be M={}, got {}",
                    m, num_rows
                );
            } else {
                panic!("Expected Rows dispatch for norm op, got {:?}", op.dispatch);
            }
        }
    }

    // -----------------------------------------------------------------------
    // patch_ops works on PrefillGraph ops
    // -----------------------------------------------------------------------

    #[test]
    fn test_patch_ops_on_prefill_graph() {
        let config = gpt2_1layer_config();
        let weights = gpt2_weights(&config);
        let mut graph = PrefillGraph::build(&config, &weights, 4, false);

        // Count EmbeddingLookup ops referencing Weight(0)
        let before_count = graph.ops.iter()
            .filter(|op| op.pso == PsoRef::EmbeddingLookup)
            .filter(|op| op.bindings.iter().any(|(br, idx, _)| *br == BufferRef::Weight(0) && *idx == 0))
            .count();
        assert!(before_count > 0, "Should have EmbeddingLookup ops using Weight(0)");

        // Patch Weight(0) -> Weight(99)
        patch_ops(&mut graph.ops, BufferRef::Weight(0), BufferRef::Weight(99));

        // After patching, no EmbeddingLookup should reference Weight(0)
        let after_count = graph.ops.iter()
            .filter(|op| op.pso == PsoRef::EmbeddingLookup)
            .filter(|op| op.bindings.iter().any(|(br, idx, _)| *br == BufferRef::Weight(0) && *idx == 0))
            .count();
        assert_eq!(after_count, 0, "After patching, no EmbeddingLookup should use Weight(0)");

        // Should now reference Weight(99)
        let patched_count = graph.ops.iter()
            .filter(|op| op.pso == PsoRef::EmbeddingLookup)
            .filter(|op| op.bindings.iter().any(|(br, idx, _)| *br == BufferRef::Weight(99) && *idx == 0))
            .count();
        assert_eq!(patched_count, before_count, "Patched ops should now use Weight(99)");
    }

    // -----------------------------------------------------------------------
    // Multi-layer: KV cache indices still correct for prefill
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_kv_cache_indices() {
        let config = gpt2_config();
        let weights = gpt2_weights(&config);
        let graph = PrefillGraph::build(&config, &weights, 8, false);

        let kv_indices: Vec<u16> = graph.ops.iter()
            .flat_map(|op| {
                op.bindings.iter().map(|(br, _, _)| br)
                    .chain(op.reads.iter())
                    .chain(op.writes.iter())
            })
            .filter_map(|br| match br {
                BufferRef::KvCache(i) => Some(*i),
                _ => None,
            })
            .collect();

        let max_kv = kv_indices.iter().max().copied().unwrap_or(0);
        assert_eq!(
            max_kv as usize, (config.num_layers * 2) - 1,
            "Max KV cache index should be num_layers*2-1"
        );

        for layer in 0..config.num_layers {
            let k_idx = (layer * 2) as u16;
            let v_idx = (layer * 2 + 1) as u16;
            assert!(kv_indices.contains(&k_idx), "Missing K cache for layer {}", layer);
            assert!(kv_indices.contains(&v_idx), "Missing V cache for layer {}", layer);
        }
    }

    // -----------------------------------------------------------------------
    // Batched matmul no-bias path (LLaMA has no bias)
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_batched_matmul_no_bias() {
        let config = llama_1layer_config();
        let weights = llama_weights(&config);
        let m = 7;
        let graph = PrefillGraph::build(&config, &weights, m, false);

        // LLaMA uses Q8_0 without bias → BatchedMatmulQ8_0 (1 dispatch per linear)
        let batched_nobias: Vec<&DecodeOp> = graph.ops.iter()
            .filter(|op| op.pso == PsoRef::BatchedMatmulQ8_0)
            .collect();

        // LLaMA 1-layer: Q, K, V, O, gate, up, down = 7 linears × 1 dispatch = 7
        assert_eq!(
            batched_nobias.len(), 7,
            "Expected 7 BatchedMatmulQ8_0 ops (7 linears × 1 dispatch), got {}",
            batched_nobias.len()
        );

        // No bias ops should exist
        let bias_count: usize = graph.ops.iter()
            .filter(|op| op.pso == PsoRef::BatchedMatmulBiasQ8_0)
            .count();
        assert_eq!(bias_count, 0, "LLaMA should have no bias matmul ops");

        // Each no-bias op should NOT have a binding at index 6 (no bias buffer)
        for (i, op) in batched_nobias.iter().enumerate() {
            let has_bias_binding = op.bindings.iter().any(|(_, idx, _)| *idx == 6);
            assert!(!has_bias_binding, "No-bias op {} should not have binding at index 6", i);

            // Should have 3 bindings: weights(0), input(1), output(2)
            assert_eq!(op.bindings.len(), 3, "No-bias op {} should have 3 bindings", i);

            // Should have 3 params: M(3), K(4), N(5)
            assert_eq!(op.params.len(), 3, "No-bias op {} should have 3 params", i);
        }
    }

    // -----------------------------------------------------------------------
    // Batched matmul M boundary cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_batched_matmul_m_boundary() {
        let config = gpt2_1layer_config();
        let weights = gpt2_weights(&config);

        // GPT-2: 6 quantized linears per layer, non-matmul ops = 8, pre=3, post=3.
        let num_linears = 6;
        let non_matmul = 8;
        let pre = 3;
        let post = 3;

        // M=1: uses single-row kernels (optimized fused dequant+dot)
        {
            let graph = PrefillGraph::build(&config, &weights, 1, false);
            let single_row_ops: usize = graph.ops.iter()
                .filter(|op| op.pso == PsoRef::QuantizedMatmulBiasQ8_0)
                .count();
            assert_eq!(
                single_row_ops, num_linears,
                "M=1: expected {} single-row matmul ops, got {}",
                num_linears, single_row_ops
            );
            let expected_total = pre + config.num_layers * (non_matmul + num_linears) + post;
            assert_eq!(graph.ops.len(), expected_total);
        }

        // M>1: uses batched GEMM (single dispatch per linear)
        for m in [32, 33] {
            let graph = PrefillGraph::build(&config, &weights, m, false);
            let batched_ops: usize = graph.ops.iter()
                .filter(|op| op.pso == PsoRef::BatchedMatmulBiasQ8_0)
                .count();

            assert_eq!(
                batched_ops, num_linears,
                "M={}: expected {} batched matmul ops, got {}",
                m, num_linears, batched_ops
            );

            let expected_total = pre + config.num_layers * (non_matmul + num_linears) + post;
            assert_eq!(
                graph.ops.len(), expected_total,
                "M={}: expected {} total ops, got {}",
                m, expected_total, graph.ops.len()
            );
        }
    }

    // -----------------------------------------------------------------------
    // K-quant batched matmul: Q4_K weights produce BatchedMatmulQ4K ops
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_batched_matmul_kquant() {
        // Create a LLaMA config with Q4_K weights
        let config = llama_1layer_config();
        let h = config.hidden_size;
        let ffn = config.ffn_hidden;
        let v = config.vocab_size;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let total_dim = config.num_heads * config.head_dim;
        let qtype = TensorDtype::Q4_K;

        let layers: Vec<LayerWeights> = (0..config.num_layers)
            .map(|_| LayerWeights {
                attn_norm_w: Some(dt(vec![h], TensorDtype::F32)),
                attn_norm_b: None,
                ffn_norm_w: Some(dt(vec![h], TensorDtype::F32)),
                ffn_norm_b: None,
                attn_output_norm_w: None,
                attn_output_norm_b: None,
                ffn_output_norm_w: None,
                ffn_output_norm_b: None,
                attn_q: dt(vec![total_dim, h], qtype),
                attn_k: dt(vec![kv_dim, h], qtype),
                attn_v: dt(vec![kv_dim, h], qtype),
                attn_output: dt(vec![h, total_dim], qtype),
                attn_q_bias: None,
                attn_k_bias: None,
                attn_v_bias: None,
                attn_output_bias: None,
                ffn_up: dt(vec![ffn, h], qtype),
                ffn_down: dt(vec![h, ffn], qtype),
                ffn_gate: Some(dt(vec![ffn, h], qtype)),
                ffn_up_bias: None,
                ffn_down_bias: None,
                attn_q_norm_w: None,
                attn_k_norm_w: None,
                attn_post_norm_w: None,
                ffn_post_norm_w: None,
            })
            .collect();

        let weights = ModelWeights {
            token_embedding: dt(vec![v, h], TensorDtype::F32),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers,
            output_norm_w: Some(dt(vec![h], TensorDtype::F32)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let m = 5;
        let graph = PrefillGraph::build(&config, &weights, m, false);

        // Should use BatchedMatmulQ4K (not Q8_0 variants)
        let q4k_ops = count_prefill_pso(&graph, PsoRef::BatchedMatmulQ4K);
        let q8_ops = count_prefill_pso(&graph, PsoRef::BatchedMatmulQ8_0);

        // Q, K, V, O, gate, up, down = 7 linears × 1 dispatch = 7
        assert_eq!(q4k_ops, 7, "Q4_K weights should produce 7 BatchedMatmulQ4K ops, got {}", q4k_ops);
        assert_eq!(q8_ops, 0, "Q4_K weights should produce 0 BatchedMatmulQ8_0 ops");

        // Per-layer: 10 non-matmul ops + 7 batched matmul ops
        let expected_per_layer = 10 + 7;
        let expected_total = 1 + config.num_layers * expected_per_layer + 3;
        assert_eq!(
            graph.ops.len(), expected_total,
            "Q4_K LLaMA M={} should have {} ops, got {}",
            m, expected_total, graph.ops.len()
        );

        // Verify batched kernel uses 128 threads (tiled GEMM)
        let q4k_op = graph.ops.iter().find(|op| op.pso == PsoRef::BatchedMatmulQ4K).unwrap();
        if let DispatchDims::Fixed { tx, .. } = q4k_op.dispatch {
            assert_eq!(tx, 128, "Batched Q4_K matmul should use 128 threads");
        }
    }

    // -----------------------------------------------------------------------
    // M=1 prefill op count with batched kernels matches decode + 1
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_m1_op_count_invariant() {
        // Verify that M=1 prefill has exactly 1 more op than decode
        // regardless of quant type (Q8_0 or K-quant)
        for (name, config, weights) in [
            ("GPT-2 Q8_0", gpt2_1layer_config(), gpt2_weights(&gpt2_1layer_config())),
            ("LLaMA Q8_0", llama_1layer_config(), llama_weights(&llama_1layer_config())),
        ] {
            let decode = DecodeGraph::build(&config, &weights, false);
            let prefill = PrefillGraph::build(&config, &weights, 1, false);

            assert_eq!(
                prefill.ops.len(), decode.ops.len() + 1,
                "{}: M=1 prefill ({}) should have exactly 1 more op than decode ({})",
                name, prefill.ops.len(), decode.ops.len()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Batched matmul dispatch grid correctness for non-square dimensions
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Phi-3 prefill graph
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefill_phi3_1layer_structure() {
        let config = phi3_1layer_config();
        let weights = phi3_weights(&config);
        let m = 8;
        let graph = PrefillGraph::build(&config, &weights, m, false);

        // Phi-3 1-layer prefill with batched quantized GEMM:
        //   Pre-layer: embedding_lookup(tok) = 1 op
        //   Per layer:
        //     rms_norm(1) + Q(1) + K(1) + V(1) + rope_q(1) + rope_k(1)
        //     + copy_k(1) + copy_v(1) + batched_attn(1) + O_proj(1) + add(1)
        //     + rms_norm(1) + gate(1) + up(1) + swiglu(1) + down(1) + add(1) = 10 + 7 = 17
        //   Post-layer: rms_norm(1) + copy_last_token(1) + logits_proj(1) = 3
        let expected_per_layer = 17;
        let expected_total = 1 + config.num_layers * expected_per_layer + 3;
        assert_eq!(
            graph.ops.len(), expected_total,
            "Phi-3 1-layer M={} should have {} ops, got {}",
            m, expected_total, graph.ops.len()
        );

        // Should use NeoX-style RoPE (not RopeNorm)
        assert_eq!(count_prefill_pso(&graph, PsoRef::RopeNeox), 2);
        assert_eq!(count_prefill_pso(&graph, PsoRef::RopeNorm), 0);

        // Should use SwiGLU, not GELU
        assert!(count_prefill_pso(&graph, PsoRef::SwiGlu) > 0);
        assert_eq!(count_prefill_pso(&graph, PsoRef::Gelu), 0);

        // No bias ops (Phi-3 has no biases)
        assert_eq!(count_prefill_pso(&graph, PsoRef::BatchedMatmulBiasQ8_0), 0);
        assert_eq!(count_prefill_pso(&graph, PsoRef::AddBias), 0);

        // M>1 should use batched matmul kernels (not per-row)
        // 7 linears per layer: Q, K, V, O, gate, up, down
        assert_eq!(count_prefill_pso(&graph, PsoRef::BatchedMatmulQ8_0), 7);
    }

    #[test]
    fn test_prefill_phi3_longrope_uses_factors_kernel() {
        let config = phi3_1layer_config();
        let weights = phi3_longrope_weights(&config);
        let m = 4;
        let graph = PrefillGraph::build(&config, &weights, m, false);

        // With LongRoPE factors, should use RopeNeoxFactors instead of RopeNeox
        assert_eq!(count_prefill_pso(&graph, PsoRef::RopeNeoxFactors), 2);
        assert_eq!(count_prefill_pso(&graph, PsoRef::RopeNeox), 0);
    }

    #[test]
    fn test_prefill_phi3_weight_walk_order_matches_decode() {
        let config = phi3_1layer_config();
        let weights = phi3_weights(&config);

        let decode = DecodeGraph::build(&config, &weights, false);
        let prefill = PrefillGraph::build(&config, &weights, 8, false);

        // Both should reference the same weights in the same walk order
        let walked = weight_walk_order(&weights);

        let decode_max = decode.ops.iter()
            .flat_map(|op| op.bindings.iter().map(|(br, _, _)| br).chain(op.reads.iter()))
            .filter_map(|br| match br { BufferRef::Weight(i) => Some(*i as usize), _ => None })
            .max()
            .unwrap_or(0);

        let prefill_max = prefill.ops.iter()
            .flat_map(|op| op.bindings.iter().map(|(br, _, _)| br).chain(op.reads.iter()))
            .filter_map(|br| match br { BufferRef::Weight(i) => Some(*i as usize), _ => None })
            .max()
            .unwrap_or(0);

        assert_eq!(
            decode_max, prefill_max,
            "Decode and prefill should reference the same max weight index"
        );
        assert!(
            prefill_max < walked.len(),
            "Weight index {} out of bounds (walked {} weights)",
            prefill_max, walked.len()
        );
    }

    #[test]
    fn test_prefill_batched_matmul_grid_dims() {
        let config = llama_1layer_config();
        let weights = llama_weights(&config);
        let m = 13;
        let graph = PrefillGraph::build(&config, &weights, m, false);

        let h = config.hidden_size;          // 2048
        let kv_dim = config.num_kv_heads * config.head_dim; // 4 * 64 = 256
        let total_dim = config.num_heads * config.head_dim;  // 32 * 64 = 2048
        let ffn = config.ffn_hidden;          // 5632

        // Collect all batched matmul ops
        let batched_ops: Vec<&DecodeOp> = graph.ops.iter()
            .filter(|op| op.pso == PsoRef::BatchedMatmulQ8_0)
            .collect();

        // LLaMA: Q(total_dim), K(kv_dim), V(kv_dim), O(h), gate(ffn), up(ffn), down(h) = 7 linears
        assert_eq!(batched_ops.len(), 7);

        // Check dispatch grid for each linear.
        // Batched GEMM: gx=ceil(N/32), gy=ceil(M/64), tx=128.
        let expected_n_dims = [total_dim, kv_dim, kv_dim, h, ffn, ffn, h];
        let expected_gy = ((m + 63) / 64) as u32; // ceil(M/BBM)
        for (linear_idx, &expected_n) in expected_n_dims.iter().enumerate() {
            let op = batched_ops[linear_idx];
            if let DispatchDims::Fixed { gx, gy, gz, tx, ty, tz } = op.dispatch {
                let expected_gx = ((expected_n + 31) / 32) as u32; // ceil(N/BBN)
                assert_eq!(gx, expected_gx,
                    "Linear {} (N={}): gx should be ceil({}/32)={}, got {}",
                    linear_idx, expected_n, expected_n, expected_gx, gx);
                assert_eq!(gy, expected_gy, "Batched dispatch gy should be ceil(M/64)");
                assert_eq!(gz, 1);
                assert_eq!(tx, 128);
                assert_eq!(ty, 1);
                assert_eq!(tz, 1);
            } else {
                panic!("Linear {}: expected Fixed dispatch", linear_idx);
            }

            // Verify N param (binding 5) matches expected_n
            let n_param = op.params.iter().find(|(_, idx)| *idx == 5).unwrap();
            if let ParamValue::U32(n_val) = n_param.0 {
                assert_eq!(n_val, expected_n as u32,
                    "Linear {}: N param should be {}, got {}", linear_idx, expected_n, n_val);
            }
        }
    }
}
