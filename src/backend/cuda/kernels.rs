//! CUDA PTX kernel sources.
//!
//! All compute kernels compiled to PTX assembly targeting sm_50.
//! The PTX is loaded at runtime via cuModuleLoadData.

/// Null-terminated PTX source containing all kernels.
pub const PTX_MODULE: &str = concat!(
    r#"
.version 7.0
.target sm_53
.address_size 64

// =========================================================================
// PORTED KERNELS (from strata-core)
// =========================================================================

// -------------------------------------------------------------------------
// gemm: C = A * B  (tiled 16x16 shared-memory GEMM)
//
// Parameters (in order):
//   param_A   : .u64  pointer to A (M x K, row-major)
//   param_B   : .u64  pointer to B (K x N, row-major)
//   param_C   : .u64  pointer to C (M x N, row-major)
//   param_M   : .u32  number of rows in A / C
//   param_K   : .u32  inner dimension
//   param_N   : .u32  number of cols in B / C
//
// Grid:  (ceil(N/16), ceil(M/16), 1)
// Block: (16, 16, 1)
// -------------------------------------------------------------------------
.visible .entry gemm(
    .param .u64 param_A,
    .param .u64 param_B,
    .param .u64 param_C,
    .param .u32 param_M,
    .param .u32 param_K,
    .param .u32 param_N
)
{
    .reg .u64 %rd<20>;
    .reg .u32 %r<30>;
    .reg .f32 %f<10>;
    .reg .pred %p<5>;
    .shared .align 4 .f32 tile_A[256];
    .shared .align 4 .f32 tile_B[256];

    // Load parameters
    ld.param.u64 %rd0, [param_A];
    ld.param.u64 %rd1, [param_B];
    ld.param.u64 %rd2, [param_C];
    ld.param.u32 %r0, [param_M];
    ld.param.u32 %r1, [param_K];
    ld.param.u32 %r2, [param_N];

    // Thread indices
    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %tid.y;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ctaid.y;

    // Global row/col for this thread
    shl.b32 %r7, %r6, 4;
    add.u32 %r7, %r7, %r4;       // row = by*16 + ty
    shl.b32 %r8, %r5, 4;
    add.u32 %r8, %r8, %r3;       // col = bx*16 + tx

    // Accumulator
    mov.f32 %f0, 0f00000000;

    // Shared memory index: ty*16+tx
    shl.b32 %r9, %r4, 4;
    add.u32 %r9, %r9, %r3;

    // Tile loop: t = 0, 16, 32, ...
    mov.u32 %r10, 0;
GEMM_TILE_LOOP:
    // --- Load tile_A[ty][tx] = A[row][t+tx] ---
    add.u32 %r11, %r10, %r3;
    setp.lt.u32 %p0, %r7, %r0;
    setp.lt.u32 %p1, %r11, %r1;
    and.pred %p2, %p0, %p1;
    @!%p2 bra GEMM_ZERO_A;
    mul.lo.u32 %r12, %r7, %r1;
    add.u32 %r12, %r12, %r11;
    mul.wide.u32 %rd3, %r12, 4;
    add.u64 %rd3, %rd0, %rd3;
    ld.global.f32 %f1, [%rd3];
    bra GEMM_STORE_A;
GEMM_ZERO_A:
    mov.f32 %f1, 0f00000000;
GEMM_STORE_A:
    mov.u32 %r13, tile_A;
    shl.b32 %r14, %r9, 2;
    add.u32 %r13, %r13, %r14;
    st.shared.f32 [%r13], %f1;

    // --- Load tile_B[ty][tx] = B[t+ty][col] ---
    add.u32 %r15, %r10, %r4;
    setp.lt.u32 %p0, %r15, %r1;
    setp.lt.u32 %p1, %r8, %r2;
    and.pred %p2, %p0, %p1;
    @!%p2 bra GEMM_ZERO_B;
    mul.lo.u32 %r16, %r15, %r2;
    add.u32 %r16, %r16, %r8;
    mul.wide.u32 %rd4, %r16, 4;
    add.u64 %rd4, %rd1, %rd4;
    ld.global.f32 %f2, [%rd4];
    bra GEMM_STORE_B;
GEMM_ZERO_B:
    mov.f32 %f2, 0f00000000;
GEMM_STORE_B:
    mov.u32 %r17, tile_B;
    shl.b32 %r18, %r9, 2;
    add.u32 %r17, %r17, %r18;
    st.shared.f32 [%r17], %f2;

    bar.sync 0;

    // --- Accumulate: acc += tile_A[ty][k] * tile_B[k][tx] ---
    mov.u32 %r19, 0;
GEMM_K_LOOP:
    shl.b32 %r20, %r4, 4;
    add.u32 %r20, %r20, %r19;
    shl.b32 %r20, %r20, 2;
    mov.u32 %r21, tile_A;
    add.u32 %r21, %r21, %r20;
    ld.shared.f32 %f3, [%r21];
    shl.b32 %r22, %r19, 4;
    add.u32 %r22, %r22, %r3;
    shl.b32 %r22, %r22, 2;
    mov.u32 %r23, tile_B;
    add.u32 %r23, %r23, %r22;
    ld.shared.f32 %f4, [%r23];
    fma.rn.f32 %f0, %f3, %f4, %f0;
    add.u32 %r19, %r19, 1;
    setp.lt.u32 %p0, %r19, 16;
    @%p0 bra GEMM_K_LOOP;

    bar.sync 0;

    add.u32 %r10, %r10, 16;
    setp.lt.u32 %p0, %r10, %r1;
    @%p0 bra GEMM_TILE_LOOP;

    // --- Write result ---
    setp.lt.u32 %p0, %r7, %r0;
    setp.lt.u32 %p1, %r8, %r2;
    and.pred %p2, %p0, %p1;
    @!%p2 bra GEMM_DONE;
    mul.lo.u32 %r24, %r7, %r2;
    add.u32 %r24, %r24, %r8;
    mul.wide.u32 %rd5, %r24, 4;
    add.u64 %rd5, %rd2, %rd5;
    st.global.f32 [%rd5], %f0;
GEMM_DONE:
    ret;
}

// -------------------------------------------------------------------------
// gemm_transpose: C = A * B^T  (tiled 16x16, B is (N,K) read transposed)
//
// Parameters: same as gemm -- A (M,K), B (N,K), C (M,N), M, K, N
// Grid:  (ceil(N/16), ceil(M/16), 1)
// Block: (16, 16, 1)
// -------------------------------------------------------------------------
.visible .entry gemm_transpose(
    .param .u64 param_A,
    .param .u64 param_B,
    .param .u64 param_C,
    .param .u32 param_M,
    .param .u32 param_K,
    .param .u32 param_N
)
{
    .reg .u64 %rd<20>;
    .reg .u32 %r<30>;
    .reg .f32 %f<10>;
    .reg .pred %p<5>;
    .shared .align 4 .f32 tile_A[256];
    .shared .align 4 .f32 tile_B[256];

    ld.param.u64 %rd0, [param_A];
    ld.param.u64 %rd1, [param_B];
    ld.param.u64 %rd2, [param_C];
    ld.param.u32 %r0, [param_M];
    ld.param.u32 %r1, [param_K];
    ld.param.u32 %r2, [param_N];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %tid.y;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ctaid.y;

    shl.b32 %r7, %r6, 4;
    add.u32 %r7, %r7, %r4;       // row = by*16 + ty
    shl.b32 %r8, %r5, 4;
    add.u32 %r8, %r8, %r3;       // col = bx*16 + tx

    mov.f32 %f0, 0f00000000;
    shl.b32 %r9, %r4, 4;
    add.u32 %r9, %r9, %r3;

    mov.u32 %r10, 0;
GEMMT_TILE_LOOP:
    // tile_A[ty][tx] = A[row][t+tx]
    add.u32 %r11, %r10, %r3;
    setp.lt.u32 %p0, %r7, %r0;
    setp.lt.u32 %p1, %r11, %r1;
    and.pred %p2, %p0, %p1;
    @!%p2 bra GEMMT_ZERO_A;
    mul.lo.u32 %r12, %r7, %r1;
    add.u32 %r12, %r12, %r11;
    mul.wide.u32 %rd3, %r12, 4;
    add.u64 %rd3, %rd0, %rd3;
    ld.global.f32 %f1, [%rd3];
    bra GEMMT_STORE_A;
GEMMT_ZERO_A:
    mov.f32 %f1, 0f00000000;
GEMMT_STORE_A:
    mov.u32 %r13, tile_A;
    shl.b32 %r14, %r9, 2;
    add.u32 %r13, %r13, %r14;
    st.shared.f32 [%r13], %f1;

    // tile_B[ty][tx] = B[col][t+ty]  (B is transposed: stored as (N,K))
    add.u32 %r15, %r10, %r4;
    setp.lt.u32 %p0, %r8, %r2;
    setp.lt.u32 %p1, %r15, %r1;
    and.pred %p2, %p0, %p1;
    @!%p2 bra GEMMT_ZERO_B;
    mul.lo.u32 %r16, %r8, %r1;
    add.u32 %r16, %r16, %r15;
    mul.wide.u32 %rd4, %r16, 4;
    add.u64 %rd4, %rd1, %rd4;
    ld.global.f32 %f2, [%rd4];
    bra GEMMT_STORE_B;
GEMMT_ZERO_B:
    mov.f32 %f2, 0f00000000;
GEMMT_STORE_B:
    mov.u32 %r17, tile_B;
    shl.b32 %r18, %r9, 2;
    add.u32 %r17, %r17, %r18;
    st.shared.f32 [%r17], %f2;

    bar.sync 0;

    mov.u32 %r19, 0;
GEMMT_K_LOOP:
    shl.b32 %r20, %r4, 4;
    add.u32 %r20, %r20, %r19;
    shl.b32 %r20, %r20, 2;
    mov.u32 %r21, tile_A;
    add.u32 %r21, %r21, %r20;
    ld.shared.f32 %f3, [%r21];
    shl.b32 %r22, %r19, 4;
    add.u32 %r22, %r22, %r3;
    shl.b32 %r22, %r22, 2;
    mov.u32 %r23, tile_B;
    add.u32 %r23, %r23, %r22;
    ld.shared.f32 %f4, [%r23];
    fma.rn.f32 %f0, %f3, %f4, %f0;
    add.u32 %r19, %r19, 1;
    setp.lt.u32 %p0, %r19, 16;
    @%p0 bra GEMMT_K_LOOP;

    bar.sync 0;

    add.u32 %r10, %r10, 16;
    setp.lt.u32 %p0, %r10, %r1;
    @%p0 bra GEMMT_TILE_LOOP;

    setp.lt.u32 %p0, %r7, %r0;
    setp.lt.u32 %p1, %r8, %r2;
    and.pred %p2, %p0, %p1;
    @!%p2 bra GEMMT_DONE;
    mul.lo.u32 %r24, %r7, %r2;
    add.u32 %r24, %r24, %r8;
    mul.wide.u32 %rd5, %r24, 4;
    add.u64 %rd5, %rd2, %rd5;
    st.global.f32 [%rd5], %f0;
GEMMT_DONE:
    ret;
}

// -------------------------------------------------------------------------
// gelu: element-wise GELU approximation
//   y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Parameters: input ptr (.u64), output ptr (.u64), n (.u32)
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
//
// tanh(a) via: tanh(a) = 1 - 2/(exp(2a)+1)
//   exp(2a) = exp2(2a * log2(e))
// -------------------------------------------------------------------------
.visible .entry gelu(
    .param .u64 param_in,
    .param .u64 param_out,
    .param .u32 param_n
)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<6>;
    .reg .f32 %f<20>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_in];
    ld.param.u64 %rd1, [param_out];
    ld.param.u32 %r0, [param_n];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r1, %r1, %r2;
    mov.u32 %r3, %tid.x;
    add.u32 %r1, %r1, %r3;
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra GELU_DONE;

    // Load x
    mul.wide.u32 %rd2, %r1, 4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];

    // Constants
    mov.f32 %f1, 0f3F4C422A;      // sqrt(2/pi) = 0.7978845608
    mov.f32 %f2, 0f3D372713;      // 0.044715
    mov.f32 %f3, 0f3F000000;      // 0.5
    mov.f32 %f10, 0f3FB8AA3B;     // log2(e) = 1.4426950408
    mov.f32 %f11, 0f40000000;     // 2.0
    mov.f32 %f12, 0f3F800000;     // 1.0

    // x^3
    mul.rn.f32 %f4, %f0, %f0;
    mul.rn.f32 %f4, %f4, %f0;
    // 0.044715 * x^3
    mul.rn.f32 %f4, %f2, %f4;
    // x + 0.044715 * x^3
    add.rn.f32 %f4, %f0, %f4;
    // sqrt(2/pi) * (x + 0.044715 * x^3)
    mul.rn.f32 %f4, %f1, %f4;

    // tanh(a) = 1 - 2/(exp(2a)+1)
    mul.rn.f32 %f5, %f4, %f11;
    mul.rn.f32 %f5, %f5, %f10;
    ex2.approx.f32 %f5, %f5;
    add.rn.f32 %f6, %f5, %f12;
    div.approx.f32 %f6, %f11, %f6;
    sub.rn.f32 %f7, %f12, %f6;

    // y = 0.5 * x * (1 + tanh)
    add.rn.f32 %f8, %f12, %f7;
    mul.rn.f32 %f8, %f3, %f8;
    mul.rn.f32 %f9, %f0, %f8;

    // Store
    add.u64 %rd4, %rd1, %rd2;
    st.global.f32 [%rd4], %f9;
GELU_DONE:
    ret;
}

// -------------------------------------------------------------------------
// add_tensor: c[i] = a[i] + b[i]
//
// Parameters: a ptr (.u64), b ptr (.u64), c ptr (.u64), n (.u32)
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry add_tensor(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n
)
{
    .reg .u64 %rd<8>;
    .reg .u32 %r<5>;
    .reg .f32 %f<4>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_a];
    ld.param.u64 %rd1, [param_b];
    ld.param.u64 %rd2, [param_c];
    ld.param.u32 %r0, [param_n];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r1, %r1, %r2;
    mov.u32 %r3, %tid.x;
    add.u32 %r1, %r1, %r3;
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra ADD_DONE;

    mul.wide.u32 %rd3, %r1, 4;
    add.u64 %rd4, %rd0, %rd3;
    add.u64 %rd5, %rd1, %rd3;
    add.u64 %rd6, %rd2, %rd3;
    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];
    add.rn.f32 %f2, %f0, %f1;
    st.global.f32 [%rd6], %f2;
ADD_DONE:
    ret;
}

// -------------------------------------------------------------------------
// add_bias: t[r*cols + c] += bias[c]
//
// Parameters: t ptr (.u64), bias ptr (.u64), rows (.u32), cols (.u32)
// Grid: (rows, ceil(cols/256), 1), Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry add_bias(
    .param .u64 param_t,
    .param .u64 param_bias,
    .param .u32 param_rows,
    .param .u32 param_cols
)
{
    .reg .u64 %rd<8>;
    .reg .u32 %r<8>;
    .reg .f32 %f<4>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_t];
    ld.param.u64 %rd1, [param_bias];
    ld.param.u32 %r0, [param_rows];
    ld.param.u32 %r1, [param_cols];

    // r = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r2, %ctaid.y;
    mov.u32 %r3, %ntid.x;
    mul.lo.u32 %r2, %r2, %r3;
    mov.u32 %r4, %tid.x;
    add.u32 %r2, %r2, %r4;

    setp.ge.u32 %p0, %r2, %r1;
    @%p0 bra BIAS_DONE;

    // offset = r * cols + c
    mul.lo.u32 %r6, %r5, %r1;
    add.u32 %r6, %r6, %r2;
    mul.wide.u32 %rd2, %r6, 4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];

    // bias[c]
    mul.wide.u32 %rd4, %r2, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f1, [%rd5];

    add.rn.f32 %f2, %f0, %f1;
    st.global.f32 [%rd3], %f2;
BIAS_DONE:
    ret;
}

// -------------------------------------------------------------------------
// scale: t[i] *= factor
//
// Parameters: t ptr (.u64), factor (.f32), n (.u32)
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry scale(
    .param .u64 param_t,
    .param .f32 param_factor,
    .param .u32 param_n
)
{
    .reg .u64 %rd<4>;
    .reg .u32 %r<5>;
    .reg .f32 %f<3>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_t];
    ld.param.f32 %f0, [param_factor];
    ld.param.u32 %r0, [param_n];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r1, %r1, %r2;
    mov.u32 %r3, %tid.x;
    add.u32 %r1, %r1, %r3;
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra SCALE_DONE;

    mul.wide.u32 %rd1, %r1, 4;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.f32 %f1, [%rd2];
    mul.rn.f32 %f2, %f1, %f0;
    st.global.f32 [%rd2], %f2;
SCALE_DONE:
    ret;
}

// -------------------------------------------------------------------------
// layer_norm: per-row normalization with weight and bias
//
// Parameters:
//   input ptr (.u64), output ptr (.u64),
//   weight ptr (.u64), bias ptr (.u64),
//   rows (.u32), cols (.u32), eps (.f32)
//
// Grid:  (rows, 1, 1) -- one block per row
// Block: (256, 1, 1)
//
// Algorithm per row:
//   1. Parallel reduction for sum (mean)
//   2. Parallel reduction for variance
//   3. Normalize: out[c] = (in[c] - mean) * rsqrt(var + eps) * w[c] + b[c]
// -------------------------------------------------------------------------
.visible .entry layer_norm(
    .param .u64 param_in,
    .param .u64 param_out,
    .param .u64 param_w,
    .param .u64 param_b,
    .param .u32 param_rows,
    .param .u32 param_cols,
    .param .f32 param_eps
)
{
    .reg .u64 %rd<20>;
    .reg .u32 %r<20>;
    .reg .f32 %f<20>;
    .reg .pred %p<4>;
    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd0, [param_in];
    ld.param.u64 %rd1, [param_out];
    ld.param.u64 %rd2, [param_w];
    ld.param.u64 %rd3, [param_b];
    ld.param.u32 %r0, [param_rows];
    ld.param.u32 %r1, [param_cols];
    ld.param.f32 %f0, [param_eps];

    mov.u32 %r2, %ctaid.x;       // row index
    mov.u32 %r3, %tid.x;         // thread index
    mov.u32 %r4, %ntid.x;        // blockDim.x = 256

    // Base pointer for this row's input
    mul.lo.u32 %r5, %r2, %r1;
    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd0, %rd4;

    // --- Phase 1: compute mean via parallel reduction ---
    mov.f32 %f1, 0f00000000;
    mov.u32 %r6, %r3;
LN_SUM_LOOP:
    setp.ge.u32 %p0, %r6, %r1;
    @%p0 bra LN_SUM_DONE;
    mul.wide.u32 %rd6, %r6, 4;
    add.u64 %rd7, %rd5, %rd6;
    ld.global.f32 %f2, [%rd7];
    add.rn.f32 %f1, %f1, %f2;
    add.u32 %r6, %r6, %r4;
    bra LN_SUM_LOOP;
LN_SUM_DONE:

    // Store partial sum to shared memory
    mov.u32 %r7, sdata;
    shl.b32 %r8, %r3, 2;
    add.u32 %r7, %r7, %r8;
    st.shared.f32 [%r7], %f1;
    bar.sync 0;

    // Tree reduction (start at ntid.x/2, not hardcoded 128)
    shr.u32 %r9, %r4, 1;
LN_RED1:
    setp.ge.u32 %p0, %r3, %r9;
    @%p0 bra LN_RED1_SKIP;
    add.u32 %r10, %r3, %r9;
    mov.u32 %r11, sdata;
    shl.b32 %r12, %r10, 2;
    add.u32 %r11, %r11, %r12;
    ld.shared.f32 %f3, [%r11];
    mov.u32 %r11, sdata;
    shl.b32 %r12, %r3, 2;
    add.u32 %r11, %r11, %r12;
    ld.shared.f32 %f4, [%r11];
    add.rn.f32 %f4, %f4, %f3;
    st.shared.f32 [%r11], %f4;
LN_RED1_SKIP:
    bar.sync 0;
    shr.u32 %r9, %r9, 1;
    setp.ge.u32 %p0, %r9, 1;
    @%p0 bra LN_RED1;

    // mean = sdata[0] / cols
    mov.u32 %r11, sdata;
    ld.shared.f32 %f5, [%r11];
    cvt.rn.f32.u32 %f6, %r1;
    div.approx.f32 %f5, %f5, %f6;
    bar.sync 0;

    // --- Phase 2: compute variance via parallel reduction ---
    mov.f32 %f7, 0f00000000;
    mov.u32 %r6, %r3;
LN_VAR_LOOP:
    setp.ge.u32 %p0, %r6, %r1;
    @%p0 bra LN_VAR_DONE;
    mul.wide.u32 %rd6, %r6, 4;
    add.u64 %rd7, %rd5, %rd6;
    ld.global.f32 %f8, [%rd7];
    sub.rn.f32 %f8, %f8, %f5;
    fma.rn.f32 %f7, %f8, %f8, %f7;
    add.u32 %r6, %r6, %r4;
    bra LN_VAR_LOOP;
LN_VAR_DONE:

    mov.u32 %r7, sdata;
    shl.b32 %r8, %r3, 2;
    add.u32 %r7, %r7, %r8;
    st.shared.f32 [%r7], %f7;
    bar.sync 0;

    shr.u32 %r9, %r4, 1;
LN_RED2:
    setp.ge.u32 %p0, %r3, %r9;
    @%p0 bra LN_RED2_SKIP;
    add.u32 %r10, %r3, %r9;
    mov.u32 %r11, sdata;
    shl.b32 %r12, %r10, 2;
    add.u32 %r11, %r11, %r12;
    ld.shared.f32 %f3, [%r11];
    mov.u32 %r11, sdata;
    shl.b32 %r12, %r3, 2;
    add.u32 %r11, %r11, %r12;
    ld.shared.f32 %f4, [%r11];
    add.rn.f32 %f4, %f4, %f3;
    st.shared.f32 [%r11], %f4;
LN_RED2_SKIP:
    bar.sync 0;
    shr.u32 %r9, %r9, 1;
    setp.ge.u32 %p0, %r9, 1;
    @%p0 bra LN_RED2;

    // var = sdata[0] / cols
    mov.u32 %r11, sdata;
    ld.shared.f32 %f9, [%r11];
    div.approx.f32 %f9, %f9, %f6;
    // inv_std = rsqrt(var + eps)
    add.rn.f32 %f9, %f9, %f0;
    rsqrt.approx.f32 %f10, %f9;
    bar.sync 0;

    // --- Phase 3: normalize and write output ---
    add.u64 %rd8, %rd1, %rd4;
    mov.u32 %r6, %r3;
LN_NORM_LOOP:
    setp.ge.u32 %p0, %r6, %r1;
    @%p0 bra LN_NORM_DONE;
    // Load input[c]
    mul.wide.u32 %rd6, %r6, 4;
    add.u64 %rd7, %rd5, %rd6;
    ld.global.f32 %f11, [%rd7];
    // (x - mean) * inv_std
    sub.rn.f32 %f11, %f11, %f5;
    mul.rn.f32 %f11, %f11, %f10;
    // * weight[c]
    add.u64 %rd9, %rd2, %rd6;
    ld.global.f32 %f12, [%rd9];
    mul.rn.f32 %f11, %f11, %f12;
    // + bias[c]
    add.u64 %rd10, %rd3, %rd6;
    ld.global.f32 %f13, [%rd10];
    add.rn.f32 %f11, %f11, %f13;
    // Store
    add.u64 %rd11, %rd8, %rd6;
    st.global.f32 [%rd11], %f11;
    add.u32 %r6, %r6, %r4;
    bra LN_NORM_LOOP;
LN_NORM_DONE:
    ret;
}

// -------------------------------------------------------------------------
// softmax_rows: per-row softmax with max subtraction
//
// Parameters: data ptr (.u64), rows (.u32), cols (.u32)
// Grid:  (rows, 1, 1)
// Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry softmax_rows(
    .param .u64 param_data,
    .param .u32 param_rows,
    .param .u32 param_cols
)
{
    .reg .u64 %rd<10>;
    .reg .u32 %r<16>;
    .reg .f32 %f<16>;
    .reg .pred %p<4>;
    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd0, [param_data];
    ld.param.u32 %r0, [param_rows];
    ld.param.u32 %r1, [param_cols];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ntid.x;

    // Row base pointer
    mul.lo.u32 %r5, %r2, %r1;
    mul.wide.u32 %rd1, %r5, 4;
    add.u64 %rd2, %rd0, %rd1;

    // --- Phase 1: find max ---
    mov.f32 %f0, 0fFF7FFFFF;     // -FLT_MAX
    mov.u32 %r6, %r3;
SM_MAX_LOOP:
    setp.ge.u32 %p0, %r6, %r1;
    @%p0 bra SM_MAX_DONE;
    mul.wide.u32 %rd3, %r6, 4;
    add.u64 %rd4, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    max.f32 %f0, %f0, %f1;
    add.u32 %r6, %r6, %r4;
    bra SM_MAX_LOOP;
SM_MAX_DONE:

    mov.u32 %r7, sdata;
    shl.b32 %r8, %r3, 2;
    add.u32 %r7, %r7, %r8;
    st.shared.f32 [%r7], %f0;
    bar.sync 0;

    // Tree reduction for max (start at ntid.x/2)
    shr.u32 %r9, %r4, 1;
SM_RED_MAX:
    setp.ge.u32 %p0, %r3, %r9;
    @%p0 bra SM_RED_MAX_SKIP;
    add.u32 %r10, %r3, %r9;
    mov.u32 %r11, sdata;
    shl.b32 %r12, %r10, 2;
    add.u32 %r11, %r11, %r12;
    ld.shared.f32 %f2, [%r11];
    mov.u32 %r11, sdata;
    shl.b32 %r12, %r3, 2;
    add.u32 %r11, %r11, %r12;
    ld.shared.f32 %f3, [%r11];
    max.f32 %f3, %f3, %f2;
    st.shared.f32 [%r11], %f3;
SM_RED_MAX_SKIP:
    bar.sync 0;
    shr.u32 %r9, %r9, 1;
    setp.ge.u32 %p0, %r9, 1;
    @%p0 bra SM_RED_MAX;

    // Broadcast max from sdata[0]
    mov.u32 %r11, sdata;
    ld.shared.f32 %f4, [%r11];
    bar.sync 0;

    // --- Phase 2: compute exp(x - max) in-place and sum ---
    mov.f32 %f12, 0f3FB8AA3B;    // log2(e)
    mov.f32 %f5, 0f00000000;
    mov.u32 %r6, %r3;
SM_EXP_LOOP:
    setp.ge.u32 %p0, %r6, %r1;
    @%p0 bra SM_EXP_DONE;
    mul.wide.u32 %rd3, %r6, 4;
    add.u64 %rd4, %rd2, %rd3;
    ld.global.f32 %f6, [%rd4];
    sub.rn.f32 %f6, %f6, %f4;
    mul.rn.f32 %f7, %f6, %f12;
    ex2.approx.f32 %f7, %f7;
    st.global.f32 [%rd4], %f7;
    add.rn.f32 %f5, %f5, %f7;
    add.u32 %r6, %r6, %r4;
    bra SM_EXP_LOOP;
SM_EXP_DONE:

    mov.u32 %r7, sdata;
    shl.b32 %r8, %r3, 2;
    add.u32 %r7, %r7, %r8;
    st.shared.f32 [%r7], %f5;
    bar.sync 0;

    // Tree reduction for sum (start at ntid.x/2)
    shr.u32 %r9, %r4, 1;
SM_RED_SUM:
    setp.ge.u32 %p0, %r3, %r9;
    @%p0 bra SM_RED_SUM_SKIP;
    add.u32 %r10, %r3, %r9;
    mov.u32 %r11, sdata;
    shl.b32 %r12, %r10, 2;
    add.u32 %r11, %r11, %r12;
    ld.shared.f32 %f8, [%r11];
    mov.u32 %r11, sdata;
    shl.b32 %r12, %r3, 2;
    add.u32 %r11, %r11, %r12;
    ld.shared.f32 %f9, [%r11];
    add.rn.f32 %f9, %f9, %f8;
    st.shared.f32 [%r11], %f9;
SM_RED_SUM_SKIP:
    bar.sync 0;
    shr.u32 %r9, %r9, 1;
    setp.ge.u32 %p0, %r9, 1;
    @%p0 bra SM_RED_SUM;

    // Broadcast sum
    mov.u32 %r11, sdata;
    ld.shared.f32 %f10, [%r11];
    bar.sync 0;

    // --- Phase 3: divide by sum ---
    mov.u32 %r6, %r3;
SM_DIV_LOOP:
    setp.ge.u32 %p0, %r6, %r1;
    @%p0 bra SM_DIV_DONE;
    mul.wide.u32 %rd3, %r6, 4;
    add.u64 %rd4, %rd2, %rd3;
    ld.global.f32 %f11, [%rd4];
    div.approx.f32 %f11, %f11, %f10;
    st.global.f32 [%rd4], %f11;
    add.u32 %r6, %r6, %r4;
    bra SM_DIV_LOOP;
SM_DIV_DONE:
    ret;
}

// -------------------------------------------------------------------------
// mean_pool: sum rows where mask==1, divide by count
//
// Parameters: hidden ptr (.u64), mask ptr (.u64), output ptr (.u64),
//             rows (.u32), cols (.u32)
//
// Grid: (1, 1, 1), Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry mean_pool(
    .param .u64 param_hidden,
    .param .u64 param_mask,
    .param .u64 param_output,
    .param .u32 param_rows,
    .param .u32 param_cols
)
{
    .reg .u64 %rd<14>;
    .reg .u32 %r<16>;
    .reg .f32 %f<10>;
    .reg .pred %p<4>;
    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd0, [param_hidden];
    ld.param.u64 %rd1, [param_mask];
    ld.param.u64 %rd2, [param_output];
    ld.param.u32 %r0, [param_rows];
    ld.param.u32 %r1, [param_cols];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ntid.x;

    mov.f32 %f0, 0f00000000;     // count

    // Zero the output array
    mov.u32 %r4, %r2;
MP_ZERO_LOOP:
    setp.ge.u32 %p0, %r4, %r1;
    @%p0 bra MP_ZERO_DONE;
    mul.wide.u32 %rd3, %r4, 4;
    add.u64 %rd4, %rd2, %rd3;
    mov.f32 %f1, 0f00000000;
    st.global.f32 [%rd4], %f1;
    add.u32 %r4, %r4, %r3;
    bra MP_ZERO_LOOP;
MP_ZERO_DONE:
    bar.sync 0;

    // Iterate over rows
    mov.u32 %r5, 0;
MP_ROW_LOOP:
    setp.ge.u32 %p0, %r5, %r0;
    @%p0 bra MP_ROW_DONE;

    // Check mask[row]
    mul.wide.u32 %rd5, %r5, 4;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.u32 %r6, [%rd6];
    setp.eq.u32 %p1, %r6, 0;
    @%p1 bra MP_ROW_SKIP;

    // mask[row] == 1: increment count (thread 0 only)
    setp.eq.u32 %p2, %r2, 0;
    @!%p2 bra MP_SKIP_COUNT;
    mov.f32 %f2, 0f3F800000;
    add.rn.f32 %f0, %f0, %f2;
MP_SKIP_COUNT:

    // Row base
    mul.lo.u32 %r7, %r5, %r1;
    mul.wide.u32 %rd7, %r7, 4;
    add.u64 %rd8, %rd0, %rd7;

    mov.u32 %r8, %r2;
MP_COL_LOOP:
    setp.ge.u32 %p0, %r8, %r1;
    @%p0 bra MP_COL_DONE;
    mul.wide.u32 %rd9, %r8, 4;
    add.u64 %rd10, %rd8, %rd9;
    ld.global.f32 %f3, [%rd10];
    add.u64 %rd11, %rd2, %rd9;
    ld.global.f32 %f4, [%rd11];
    add.rn.f32 %f4, %f4, %f3;
    st.global.f32 [%rd11], %f4;
    add.u32 %r8, %r8, %r3;
    bra MP_COL_LOOP;
MP_COL_DONE:

MP_ROW_SKIP:
    bar.sync 0;
    add.u32 %r5, %r5, 1;
    bra MP_ROW_LOOP;
MP_ROW_DONE:

    // Broadcast count from thread 0 via shared memory
    setp.eq.u32 %p0, %r2, 0;
    @!%p0 bra MP_SKIP_STORE_CNT;
    mov.u32 %r9, sdata;
    st.shared.f32 [%r9], %f0;
MP_SKIP_STORE_CNT:
    bar.sync 0;
    mov.u32 %r9, sdata;
    ld.shared.f32 %f5, [%r9];

    // Guard: if count == 0, skip divide
    mov.f32 %f6, 0f00000000;
    setp.eq.f32 %p0, %f5, %f6;
    @%p0 bra MP_FINAL_DONE;

    // Divide output[c] by count
    mov.u32 %r10, %r2;
MP_DIV_LOOP:
    setp.ge.u32 %p0, %r10, %r1;
    @%p0 bra MP_FINAL_DONE;
    mul.wide.u32 %rd12, %r10, 4;
    add.u64 %rd13, %rd2, %rd12;
    ld.global.f32 %f7, [%rd13];
    div.approx.f32 %f7, %f7, %f5;
    st.global.f32 [%rd13], %f7;
    add.u32 %r10, %r10, %r3;
    bra MP_DIV_LOOP;
MP_FINAL_DONE:
    ret;
}


// =========================================================================
// NEW KERNELS (strata-inference specific)
// =========================================================================

// -------------------------------------------------------------------------
// quantized_matmul_q8_0: Fused Q8_0 dequantize + matrix-vector product
//
// Q8_0 block layout (34 bytes per block of 32 values):
//   - 2-byte f16 scale (d)
//   - 32 x int8 quantized values (qs)
//   Dequant: y[i] = d * qs[i]
//
// Computes output[m*N + n] = sum_k( dequant(weights[n][k]) * input[m*K + k] )
//
// Parameters:
//   param_weights : .u64  raw byte pointer to Q8_0 weight data (N rows, K cols packed)
//   param_input   : .u64  f32 pointer to input (M x K)
//   param_output  : .u64  f32 pointer to output (M x N)
//   param_N       : .u32  number of weight rows (output cols)
//   param_K       : .u32  number of columns (must be multiple of 32)
//
// Grid:  (N, M, 1) -- one block per (output_row, input_row)
// Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry quantized_matmul_q8_0(
    .param .u64 param_weights,
    .param .u64 param_input,
    .param .u64 param_output,
    .param .u32 param_N,
    .param .u32 param_K
)
{
    .reg .u64 %rd<20>;
    .reg .u32 %r<30>;
    .reg .f32 %f<12>;
    .reg .f16 %h0;
    .reg .pred %p<4>;
    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_input];
    ld.param.u64 %rd2, [param_output];
    ld.param.u32 %r0, [param_N];
    ld.param.u32 %r1, [param_K];

    mov.u32 %r2, %tid.x;          // tid
    mov.u32 %r3, %ntid.x;         // 256
    mov.u32 %r4, %ctaid.x;        // n (weight row / output col)
    mov.u32 %r5, %ctaid.y;        // m (input row / output row)

    // Number of Q8_0 blocks per weight row = K / 32
    shr.u32 %r6, %r1, 5;          // num_blocks = K / 32

    // Weight row base: n * num_blocks * 34 bytes
    mul.lo.u32 %r7, %r4, %r6;     // n * num_blocks
    mul.lo.u32 %r8, %r7, 34;      // n * num_blocks * 34
    cvt.u64.u32 %rd3, %r8;
    add.u64 %rd4, %rd0, %rd3;     // &weights[n * num_blocks * 34]

    // Input row base: m * K
    mul.lo.u32 %r9, %r5, %r1;     // m * K
    mul.wide.u32 %rd5, %r9, 4;
    add.u64 %rd6, %rd1, %rd5;     // &input[m * K]

    // Each thread processes blocks at stride blockDim
    mov.f32 %f0, 0f00000000;      // partial sum
    mov.u32 %r10, %r2;            // block_idx = tid

QMQ8_BLOCK_LOOP:
    setp.ge.u32 %p0, %r10, %r6;
    @%p0 bra QMQ8_BLOCK_DONE;

    // Byte offset of this block within the weight row: block_idx * 34
    mul.lo.u32 %r11, %r10, 34;
    cvt.u64.u32 %rd7, %r11;
    add.u64 %rd8, %rd4, %rd7;     // &block

    // Load f16 scale (2 bytes at offset 0)
    ld.global.b16 %h0, [%rd8];
    cvt.f32.f16 %f1, %h0;         // scale as f32

    // Input base for this block: &input[m*K + block_idx*32]
    shl.b32 %r14, %r10, 5;        // block_idx * 32
    mul.wide.u32 %rd9, %r14, 4;
    add.u64 %rd10, %rd6, %rd9;    // &input[m*K + block_idx*32]

    // Process 32 quantized values in this block
    // qs start at offset 2 in the block
    add.u64 %rd11, %rd8, 2;       // &qs[0]

    mov.f32 %f2, 0f00000000;      // block accum
    mov.u32 %r15, 0;              // j = 0
QMQ8_INNER_LOOP:
    setp.ge.u32 %p1, %r15, 32;
    @%p1 bra QMQ8_INNER_DONE;

    // Load qs[j] (int8)
    cvt.u64.u32 %rd12, %r15;
    add.u64 %rd13, %rd11, %rd12;
    ld.global.s8 %r16, [%rd13];
    cvt.rn.f32.s32 %f3, %r16;     // (float)qs[j]

    // Load input[m*K + block_idx*32 + j]
    mul.wide.u32 %rd14, %r15, 4;
    add.u64 %rd15, %rd10, %rd14;
    ld.global.f32 %f4, [%rd15];

    // accum += qs[j] * input[j]
    fma.rn.f32 %f2, %f3, %f4, %f2;

    add.u32 %r15, %r15, 1;
    bra QMQ8_INNER_LOOP;
QMQ8_INNER_DONE:

    // partial_sum += scale * block_accum
    fma.rn.f32 %f0, %f1, %f2, %f0;

    add.u32 %r10, %r10, %r3;      // next block at stride
    bra QMQ8_BLOCK_LOOP;
QMQ8_BLOCK_DONE:

    // Shared-memory parallel reduction
    mov.u32 %r17, sdata;
    shl.b32 %r18, %r2, 2;
    add.u32 %r17, %r17, %r18;
    st.shared.f32 [%r17], %f0;
    bar.sync 0;

    mov.u32 %r19, 128;
QMQ8_RED:
    setp.ge.u32 %p0, %r2, %r19;
    @%p0 bra QMQ8_RED_SKIP;
    add.u32 %r20, %r2, %r19;
    mov.u32 %r21, sdata;
    shl.b32 %r22, %r20, 2;
    add.u32 %r21, %r21, %r22;
    ld.shared.f32 %f5, [%r21];
    mov.u32 %r21, sdata;
    shl.b32 %r22, %r2, 2;
    add.u32 %r21, %r21, %r22;
    ld.shared.f32 %f6, [%r21];
    add.rn.f32 %f6, %f6, %f5;
    st.shared.f32 [%r21], %f6;
QMQ8_RED_SKIP:
    bar.sync 0;
    shr.u32 %r19, %r19, 1;
    setp.ge.u32 %p0, %r19, 1;
    @%p0 bra QMQ8_RED;

    // Thread 0 writes result
    setp.ne.u32 %p0, %r2, 0;
    @%p0 bra QMQ8_DONE;
    mov.u32 %r23, sdata;
    ld.shared.f32 %f7, [%r23];
    // output[m * N + n]
    mul.lo.u32 %r24, %r5, %r0;
    add.u32 %r24, %r24, %r4;
    mul.wide.u32 %rd16, %r24, 4;
    add.u64 %rd17, %rd2, %rd16;
    st.global.f32 [%rd17], %f7;
QMQ8_DONE:
    ret;
}

// -------------------------------------------------------------------------
// quantized_matmul_q4_0: Fused Q4_0 dequantize + matrix-vector product
//
// Q4_0 block layout (18 bytes per block of 32 values):
//   - 2-byte f16 scale (d)
//   - 16 packed nibble bytes (each byte holds 2 4-bit values)
//   Nibble extraction: lo = (byte & 0x0F) - 8, hi = (byte >> 4) - 8
//   Dequant: y[i] = d * nibble[i]
//
// Parameters: same as quantized_matmul_q8_0
// Grid:  (N, M, 1)
// Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry quantized_matmul_q4_0(
    .param .u64 param_weights,
    .param .u64 param_input,
    .param .u64 param_output,
    .param .u32 param_N,
    .param .u32 param_K
)
{
    .reg .u64 %rd<22>;
    .reg .u32 %r<34>;
    .reg .f32 %f<14>;
    .reg .f16 %h0;
    .reg .pred %p<4>;
    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_input];
    ld.param.u64 %rd2, [param_output];
    ld.param.u32 %r0, [param_N];
    ld.param.u32 %r1, [param_K];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ntid.x;          // 256
    mov.u32 %r4, %ctaid.x;         // n (weight row)
    mov.u32 %r5, %ctaid.y;         // m (input row)

    // num_blocks = K / 32
    shr.u32 %r6, %r1, 5;

    // Weight row base: n * num_blocks * 18 bytes
    mul.lo.u32 %r7, %r4, %r6;
    mul.lo.u32 %r8, %r7, 18;
    cvt.u64.u32 %rd3, %r8;
    add.u64 %rd4, %rd0, %rd3;

    // Input row base: m * K
    mul.lo.u32 %r9, %r5, %r1;
    mul.wide.u32 %rd5, %r9, 4;
    add.u64 %rd6, %rd1, %rd5;

    mov.f32 %f0, 0f00000000;
    mov.u32 %r10, %r2;

QMQ4_BLOCK_LOOP:
    setp.ge.u32 %p0, %r10, %r6;
    @%p0 bra QMQ4_BLOCK_DONE;

    // Block byte offset: block_idx * 18
    mul.lo.u32 %r11, %r10, 18;
    cvt.u64.u32 %rd7, %r11;
    add.u64 %rd8, %rd4, %rd7;

    // Load f16 scale
    ld.global.b16 %h0, [%rd8];
    cvt.f32.f16 %f1, %h0;

    // Input base for this block
    shl.b32 %r14, %r10, 5;
    mul.wide.u32 %rd9, %r14, 4;
    add.u64 %rd10, %rd6, %rd9;

    // Nibble data starts at offset 2
    add.u64 %rd11, %rd8, 2;

    mov.f32 %f2, 0f00000000;
    mov.f32 %f8, 0f41000000;       // 8.0
    mov.u32 %r15, 0;               // byte_idx = 0

QMQ4_NIBBLE_LOOP:
    setp.ge.u32 %p1, %r15, 16;
    @%p1 bra QMQ4_NIBBLE_DONE;

    // Load packed byte
    cvt.u64.u32 %rd12, %r15;
    add.u64 %rd13, %rd11, %rd12;
    ld.global.u8 %r16, [%rd13];

    // Extract low nibble: (byte & 0x0F) - 8
    and.b32 %r17, %r16, 15;
    cvt.rn.f32.u32 %f3, %r17;
    sub.rn.f32 %f3, %f3, %f8;

    // Load input[byte_idx] for low nibble
    mul.wide.u32 %rd14, %r15, 4;
    add.u64 %rd15, %rd10, %rd14;
    ld.global.f32 %f4, [%rd15];

    fma.rn.f32 %f2, %f3, %f4, %f2;

    // Extract high nibble: (byte >> 4) - 8
    shr.u32 %r18, %r16, 4;
    cvt.rn.f32.u32 %f5, %r18;
    sub.rn.f32 %f5, %f5, %f8;

    // Load input[byte_idx + 16] for high nibble
    add.u32 %r19, %r15, 16;
    mul.wide.u32 %rd16, %r19, 4;
    add.u64 %rd17, %rd10, %rd16;
    ld.global.f32 %f6, [%rd17];

    fma.rn.f32 %f2, %f5, %f6, %f2;

    add.u32 %r15, %r15, 1;
    bra QMQ4_NIBBLE_LOOP;
QMQ4_NIBBLE_DONE:

    fma.rn.f32 %f0, %f1, %f2, %f0;

    add.u32 %r10, %r10, %r3;
    bra QMQ4_BLOCK_LOOP;
QMQ4_BLOCK_DONE:

    // Shared-memory reduction
    mov.u32 %r20, sdata;
    shl.b32 %r21, %r2, 2;
    add.u32 %r20, %r20, %r21;
    st.shared.f32 [%r20], %f0;
    bar.sync 0;

    mov.u32 %r22, 128;
QMQ4_RED:
    setp.ge.u32 %p0, %r2, %r22;
    @%p0 bra QMQ4_RED_SKIP;
    add.u32 %r23, %r2, %r22;
    mov.u32 %r24, sdata;
    shl.b32 %r25, %r23, 2;
    add.u32 %r24, %r24, %r25;
    ld.shared.f32 %f9, [%r24];
    mov.u32 %r24, sdata;
    shl.b32 %r25, %r2, 2;
    add.u32 %r24, %r24, %r25;
    ld.shared.f32 %f10, [%r24];
    add.rn.f32 %f10, %f10, %f9;
    st.shared.f32 [%r24], %f10;
QMQ4_RED_SKIP:
    bar.sync 0;
    shr.u32 %r22, %r22, 1;
    setp.ge.u32 %p0, %r22, 1;
    @%p0 bra QMQ4_RED;

    // Thread 0 writes result
    setp.ne.u32 %p0, %r2, 0;
    @%p0 bra QMQ4_DONE;
    mov.u32 %r26, sdata;
    ld.shared.f32 %f11, [%r26];
    mul.lo.u32 %r27, %r5, %r0;
    add.u32 %r27, %r27, %r4;
    mul.wide.u32 %rd18, %r27, 4;
    add.u64 %rd19, %rd2, %rd18;
    st.global.f32 [%rd19], %f11;
QMQ4_DONE:
    ret;
}

// -------------------------------------------------------------------------
// quantized_matmul_q4_k: Fused Q4_K dequantize + matrix multiply
//
// Q4_K block layout (144 bytes per block of 256 values):
//   [0:2]   d    (f16) -- super-block scale
//   [2:4]   dmin (f16) -- super-block min
//   [4:16]  scales[12] -- packed 6-bit sub-block scales/mins (8 pairs)
//   [16:144] qs[128]   -- 256 x 4-bit values (low/high nibbles)
//
// Scale extraction (get_scale_min_k4):
//   j < 4: sc = scales[j] & 63,               m = scales[j+4] & 63
//   j >= 4: sc = (scales[j+4]&0xF)|((scales[j-4]>>6)<<4)
//           m  = (scales[j+4]>>4)|((scales[j]>>6)<<4)
//
// Dequant: val = d * sc * q_nibble - dmin * m
//
// Each block of 256 threads processes one output element [i,j].
// Thread tid handles element tid within each Q4_K super-block.
//
// Grid:  (N, M, 1)
// Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry quantized_matmul_q4_k(
    .param .u64 param_weights,
    .param .u64 param_input,
    .param .u64 param_output,
    .param .u32 param_N,
    .param .u32 param_K
)
{
    .reg .u64 %rd<20>;
    .reg .u32 %r<40>;
    .reg .f32 %f<16>;
    .reg .f16 %h0, %h1;
    .reg .pred %p<6>;
    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_input];
    ld.param.u64 %rd2, [param_output];
    ld.param.u32 %r0, [param_N];
    ld.param.u32 %r1, [param_K];

    mov.u32 %r2, %tid.x;          // tid (0-255)
    mov.u32 %r4, %ctaid.x;        // j (weight row / output col)
    mov.u32 %r5, %ctaid.y;        // i (input row)

    // blocks_per_row = K / 256 = K >> 8
    shr.u32 %r6, %r1, 8;

    // bytes_per_row = blocks_per_row * 144
    mul.lo.u32 %r7, %r6, 144;

    // Weight row base: &weights[j * bytes_per_row]
    mul.lo.u32 %r8, %r4, %r7;
    cvt.u64.u32 %rd3, %r8;
    add.u64 %rd4, %rd0, %rd3;

    // Input row base: &input[i * K]
    mul.lo.u32 %r9, %r5, %r1;
    mul.wide.u32 %rd5, %r9, 4;
    add.u64 %rd6, %rd1, %rd5;

    // Decompose tid for Q4_K block structure:
    // chunk = tid / 64 (0-3)
    shr.u32 %r11, %r2, 6;
    // local = tid % 64 (0-63)
    and.b32 %r12, %r2, 63;
    // is_low_nibble: local < 32
    setp.lt.u32 %p0, %r12, 32;
    // l = is_low ? local : (local - 32)
    @%p0 mov.u32 %r13, %r12;
    @!%p0 sub.u32 %r13, %r12, 32;
    // q_off = chunk * 32 (byte offset within qs[128])
    shl.b32 %r14, %r11, 5;
    // scale_idx = chunk * 2 + (is_low ? 0 : 1)
    shl.b32 %r15, %r11, 1;
    @!%p0 add.u32 %r15, %r15, 1;

    // Initialize partial sum
    mov.f32 %f0, 0f00000000;

    // Loop over Q4_K super-blocks
    mov.u32 %r10, 0;
Q4K_BLOCK_LOOP:
    setp.ge.u32 %p1, %r10, %r6;
    @%p1 bra Q4K_BLOCK_DONE;

    // Block address: weight_row + block_idx * 144
    mul.lo.u32 %r16, %r10, 144;
    cvt.u64.u32 %rd7, %r16;
    add.u64 %rd8, %rd4, %rd7;

    // Load d (f16) at offset 0, dmin (f16) at offset 2
    ld.global.b16 %h0, [%rd8];
    ld.global.b16 %h1, [%rd8+2];
    cvt.f32.f16 %f1, %h0;           // d as f32
    cvt.f32.f16 %f2, %h1;           // dmin as f32

    // --- Extract scale and min for this thread's scale_idx ---
    // scales[12] at block + 4; load scales[j], scales[j+4], and scales[j-4] if needed
    add.u64 %rd9, %rd8, 4;          // &scales[0]

    // Load scales[scale_idx]
    cvt.u64.u32 %rd10, %r15;
    add.u64 %rd11, %rd9, %rd10;
    ld.global.u8 %r17, [%rd11];     // scales[j]

    // Load scales[scale_idx + 4]
    add.u64 %rd12, %rd11, 4;
    ld.global.u8 %r18, [%rd12];     // scales[j+4]

    // Branch: j < 4 vs j >= 4
    setp.lt.u32 %p2, %r15, 4;
    @%p2 bra Q4K_SCALE_LOW;

    // j >= 4 path:
    // Load scales[j-4]
    sub.u64 %rd13, %rd11, 4;
    ld.global.u8 %r19, [%rd13];

    // sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
    and.b32 %r20, %r18, 0xF;
    shr.u32 %r21, %r19, 6;
    shl.b32 %r21, %r21, 4;
    or.b32 %r22, %r20, %r21;

    // m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
    shr.u32 %r23, %r18, 4;
    shr.u32 %r24, %r17, 6;
    shl.b32 %r24, %r24, 4;
    or.b32 %r25, %r23, %r24;
    bra Q4K_SCALE_DONE;

Q4K_SCALE_LOW:
    // j < 4 path:
    // sc = scales[j] & 63
    and.b32 %r22, %r17, 63;
    // m = scales[j+4] & 63
    and.b32 %r25, %r18, 63;

Q4K_SCALE_DONE:
    // %r22 = sc, %r25 = m_val

    // Load qs byte: qs[q_off + l] at block + 16 + q_off + l
    add.u32 %r26, %r14, %r13;       // q_off + l
    add.u32 %r26, %r26, 16;         // + 16 (header size: 2+2+12)
    cvt.u64.u32 %rd14, %r26;
    add.u64 %rd15, %rd8, %rd14;
    ld.global.u8 %r27, [%rd15];

    // Extract nibble: low if is_low, high otherwise
    @%p0 and.b32 %r28, %r27, 0xF;
    @!%p0 shr.u32 %r28, %r27, 4;

    // Dequantize: val = d * sc * q - dmin * m
    cvt.rn.f32.u32 %f3, %r22;       // sc as f32
    cvt.rn.f32.u32 %f4, %r25;       // m as f32
    cvt.rn.f32.u32 %f5, %r28;       // q as f32
    mul.rn.f32 %f6, %f1, %f3;       // d * sc
    mul.rn.f32 %f7, %f6, %f5;       // d * sc * q
    mul.rn.f32 %f8, %f2, %f4;       // dmin * m
    sub.rn.f32 %f9, %f7, %f8;       // val

    // Load input element: input[i*K + block_idx*256 + tid]
    shl.b32 %r29, %r10, 8;          // block_idx * 256
    add.u32 %r29, %r29, %r2;        // + tid
    mul.wide.u32 %rd16, %r29, 4;
    add.u64 %rd17, %rd6, %rd16;
    ld.global.f32 %f10, [%rd17];

    // partial_sum += val * input
    fma.rn.f32 %f0, %f9, %f10, %f0;

    // Next super-block
    add.u32 %r10, %r10, 1;
    bra Q4K_BLOCK_LOOP;
Q4K_BLOCK_DONE:

    // === Shared-memory tree reduction ===
    mov.u32 %r30, sdata;
    shl.b32 %r31, %r2, 2;
    add.u32 %r30, %r30, %r31;
    st.shared.f32 [%r30], %f0;
    bar.sync 0;

    mov.u32 %r32, 128;
Q4K_RED:
    setp.ge.u32 %p3, %r2, %r32;
    @%p3 bra Q4K_RED_SKIP;
    add.u32 %r33, %r2, %r32;
    mov.u32 %r34, sdata;
    shl.b32 %r35, %r33, 2;
    add.u32 %r34, %r34, %r35;
    ld.shared.f32 %f11, [%r34];
    mov.u32 %r34, sdata;
    shl.b32 %r35, %r2, 2;
    add.u32 %r34, %r34, %r35;
    ld.shared.f32 %f12, [%r34];
    add.rn.f32 %f12, %f12, %f11;
    st.shared.f32 [%r34], %f12;
Q4K_RED_SKIP:
    bar.sync 0;
    shr.u32 %r32, %r32, 1;
    setp.ge.u32 %p3, %r32, 1;
    @%p3 bra Q4K_RED;

    // Thread 0 writes result
    setp.ne.u32 %p4, %r2, 0;
    @%p4 bra Q4K_DONE;
    mov.u32 %r36, sdata;
    ld.shared.f32 %f13, [%r36];
    // output[i * N + j]
    mul.lo.u32 %r37, %r5, %r0;
    add.u32 %r37, %r37, %r4;
    mul.wide.u32 %rd18, %r37, 4;
    add.u64 %rd19, %rd2, %rd18;
    st.global.f32 [%rd19], %f13;
Q4K_DONE:
    ret;
}

// -------------------------------------------------------------------------
// quantized_matmul_q5_k: Fused Q5_K dequantize + matrix multiply
//
// Q5_K block layout (176 bytes per block of 256 values):
//   [0:2]   d    (f16) -- super-block scale
//   [2:4]   dmin (f16) -- super-block min
//   [4:16]  scales[12] -- packed 6-bit sub-block scales/mins (same as Q4_K)
//   [16:48] qh[32]     -- high bit per value (1 bit each, packed in bytes)
//   [48:176] qs[128]   -- low 4-bit values (same as Q4_K)
//
// Dequant: val = d * sc * (q_low + high*16) - dmin * m
//   high = (qh[l] >> bit_pos) & 1
//   bit_pos for low_nibble:  chunk * 2
//   bit_pos for high_nibble: chunk * 2 + 1
//
// Grid:  (N, M, 1)
// Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry quantized_matmul_q5_k(
    .param .u64 param_weights,
    .param .u64 param_input,
    .param .u64 param_output,
    .param .u32 param_N,
    .param .u32 param_K
)
{
    .reg .u64 %rd<20>;
    .reg .u32 %r<42>;
    .reg .f32 %f<16>;
    .reg .f16 %h0, %h1;
    .reg .pred %p<6>;
    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_input];
    ld.param.u64 %rd2, [param_output];
    ld.param.u32 %r0, [param_N];
    ld.param.u32 %r1, [param_K];

    mov.u32 %r2, %tid.x;
    mov.u32 %r4, %ctaid.x;        // j (weight row)
    mov.u32 %r5, %ctaid.y;        // i (input row)

    // blocks_per_row = K >> 8
    shr.u32 %r6, %r1, 8;

    // bytes_per_row = blocks_per_row * 176
    mul.lo.u32 %r7, %r6, 176;

    // Weight row base
    mul.lo.u32 %r8, %r4, %r7;
    cvt.u64.u32 %rd3, %r8;
    add.u64 %rd4, %rd0, %rd3;

    // Input row base
    mul.lo.u32 %r9, %r5, %r1;
    mul.wide.u32 %rd5, %r9, 4;
    add.u64 %rd6, %rd1, %rd5;

    // Decompose tid (same as Q4_K)
    shr.u32 %r11, %r2, 6;         // chunk
    and.b32 %r12, %r2, 63;        // local
    setp.lt.u32 %p0, %r12, 32;    // is_low
    @%p0 mov.u32 %r13, %r12;      // l
    @!%p0 sub.u32 %r13, %r12, 32;
    shl.b32 %r14, %r11, 5;        // q_off = chunk*32
    shl.b32 %r15, %r11, 1;        // scale_idx = chunk*2
    @!%p0 add.u32 %r15, %r15, 1;  // + 1 if high nibble

    // qh bit mask: 1 << scale_idx (= 1 << (chunk*2 + nibble))
    mov.u32 %r38, 1;
    shl.b32 %r38, %r38, %r15;     // qh_mask

    mov.f32 %f0, 0f00000000;
    mov.u32 %r10, 0;
Q5K_BLOCK_LOOP:
    setp.ge.u32 %p1, %r10, %r6;
    @%p1 bra Q5K_BLOCK_DONE;

    // Block address
    mul.lo.u32 %r16, %r10, 176;
    cvt.u64.u32 %rd7, %r16;
    add.u64 %rd8, %rd4, %rd7;

    // Load d, dmin
    ld.global.b16 %h0, [%rd8];
    ld.global.b16 %h1, [%rd8+2];
    cvt.f32.f16 %f1, %h0;
    cvt.f32.f16 %f2, %h1;

    // Scale extraction (identical to Q4_K)
    add.u64 %rd9, %rd8, 4;
    cvt.u64.u32 %rd10, %r15;
    add.u64 %rd11, %rd9, %rd10;
    ld.global.u8 %r17, [%rd11];
    add.u64 %rd12, %rd11, 4;
    ld.global.u8 %r18, [%rd12];

    setp.lt.u32 %p2, %r15, 4;
    @%p2 bra Q5K_SCALE_LOW;

    sub.u64 %rd13, %rd11, 4;
    ld.global.u8 %r19, [%rd13];
    and.b32 %r20, %r18, 0xF;
    shr.u32 %r21, %r19, 6;
    shl.b32 %r21, %r21, 4;
    or.b32 %r22, %r20, %r21;
    shr.u32 %r23, %r18, 4;
    shr.u32 %r24, %r17, 6;
    shl.b32 %r24, %r24, 4;
    or.b32 %r25, %r23, %r24;
    bra Q5K_SCALE_DONE;

Q5K_SCALE_LOW:
    and.b32 %r22, %r17, 63;
    and.b32 %r25, %r18, 63;

Q5K_SCALE_DONE:

    // Load qs byte: qs[q_off + l] at block + 48 + q_off + l
    add.u32 %r26, %r14, %r13;
    add.u32 %r26, %r26, 48;       // qs start at offset 48 (2+2+12+32)
    cvt.u64.u32 %rd14, %r26;
    add.u64 %rd15, %rd8, %rd14;
    ld.global.u8 %r27, [%rd15];

    // Extract low nibble
    @%p0 and.b32 %r28, %r27, 0xF;
    @!%p0 shr.u32 %r28, %r27, 4;

    // Load qh byte: qh[l] at block + 16 + l
    add.u32 %r39, %r13, 16;       // qh start at offset 16
    cvt.u64.u32 %rd14, %r39;
    add.u64 %rd15, %rd8, %rd14;
    ld.global.u8 %r40, [%rd15];

    // high = (qh[l] & qh_mask) ? 16 : 0
    and.b32 %r41, %r40, %r38;
    setp.ne.u32 %p4, %r41, 0;
    selp.u32 %r41, 16, 0, %p4;

    // q_total = q_low + high
    add.u32 %r28, %r28, %r41;

    // Dequantize: val = d * sc * q_total - dmin * m
    cvt.rn.f32.u32 %f3, %r22;
    cvt.rn.f32.u32 %f4, %r25;
    cvt.rn.f32.u32 %f5, %r28;
    mul.rn.f32 %f6, %f1, %f3;
    mul.rn.f32 %f7, %f6, %f5;
    mul.rn.f32 %f8, %f2, %f4;
    sub.rn.f32 %f9, %f7, %f8;

    // Load input element
    shl.b32 %r29, %r10, 8;
    add.u32 %r29, %r29, %r2;
    mul.wide.u32 %rd16, %r29, 4;
    add.u64 %rd17, %rd6, %rd16;
    ld.global.f32 %f10, [%rd17];

    fma.rn.f32 %f0, %f9, %f10, %f0;

    add.u32 %r10, %r10, 1;
    bra Q5K_BLOCK_LOOP;
Q5K_BLOCK_DONE:

    // Tree reduction (same pattern)
    mov.u32 %r30, sdata;
    shl.b32 %r31, %r2, 2;
    add.u32 %r30, %r30, %r31;
    st.shared.f32 [%r30], %f0;
    bar.sync 0;

    mov.u32 %r32, 128;
Q5K_RED:
    setp.ge.u32 %p3, %r2, %r32;
    @%p3 bra Q5K_RED_SKIP;
    add.u32 %r33, %r2, %r32;
    mov.u32 %r34, sdata;
    shl.b32 %r35, %r33, 2;
    add.u32 %r34, %r34, %r35;
    ld.shared.f32 %f11, [%r34];
    mov.u32 %r34, sdata;
    shl.b32 %r35, %r2, 2;
    add.u32 %r34, %r34, %r35;
    ld.shared.f32 %f12, [%r34];
    add.rn.f32 %f12, %f12, %f11;
    st.shared.f32 [%r34], %f12;
Q5K_RED_SKIP:
    bar.sync 0;
    shr.u32 %r32, %r32, 1;
    setp.ge.u32 %p3, %r32, 1;
    @%p3 bra Q5K_RED;

    setp.ne.u32 %p4, %r2, 0;
    @%p4 bra Q5K_DONE;
    mov.u32 %r36, sdata;
    ld.shared.f32 %f13, [%r36];
    mul.lo.u32 %r37, %r5, %r0;
    add.u32 %r37, %r37, %r4;
    mul.wide.u32 %rd18, %r37, 4;
    add.u64 %rd19, %rd2, %rd18;
    st.global.f32 [%rd19], %f13;
Q5K_DONE:
    ret;
}

// -------------------------------------------------------------------------
// quantized_matmul_q6_k: Fused Q6_K dequantize + matrix multiply
//
// Q6_K block layout (210 bytes per block of 256 values):
//   [0:128]   ql[128]    -- low 4-bit values (packed nibbles)
//   [128:192] qh[64]     -- high 2-bit values (packed 2-bit pairs)
//   [192:208] scales[16] -- signed i8, one per 16-element group
//   [208:210] d (f16)    -- super-block scale
//
// Organization: 2 chunks of 128 elements. Each chunk has 4 sub-groups of 32:
//   sub 0 (pos 0-31):   low nibble of ql[off+l],     qh bits 0-1, scales[off+is]
//   sub 1 (pos 32-63):  low nibble of ql[off+l+32],  qh bits 2-3, scales[off+is+2]
//   sub 2 (pos 64-95):  high nibble of ql[off+l],    qh bits 4-5, scales[off+is+4]
//   sub 3 (pos 96-127): high nibble of ql[off+l+32], qh bits 6-7, scales[off+is+6]
//
// Dequant: val = d * scales[sc_idx] * ((ql_nibble | (qh_bits << 4)) - 32)
//
// Grid:  (N, M, 1)
// Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry quantized_matmul_q6_k(
    .param .u64 param_weights,
    .param .u64 param_input,
    .param .u64 param_output,
    .param .u32 param_N,
    .param .u32 param_K
)
{
    .reg .u64 %rd<20>;
    .reg .u32 %r<44>;
    .reg .f32 %f<16>;
    .reg .f16 %h0;
    .reg .pred %p<6>;
    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_input];
    ld.param.u64 %rd2, [param_output];
    ld.param.u32 %r0, [param_N];
    ld.param.u32 %r1, [param_K];

    mov.u32 %r2, %tid.x;          // tid (0-255)
    mov.u32 %r4, %ctaid.x;        // j (weight row)
    mov.u32 %r5, %ctaid.y;        // i (input row)

    // blocks_per_row = K >> 8
    shr.u32 %r6, %r1, 8;

    // bytes_per_row = blocks_per_row * 210
    mul.lo.u32 %r7, %r6, 210;

    // Weight row base
    mul.lo.u32 %r8, %r4, %r7;
    cvt.u64.u32 %rd3, %r8;
    add.u64 %rd4, %rd0, %rd3;

    // Input row base
    mul.lo.u32 %r9, %r5, %r1;
    mul.wide.u32 %rd5, %r9, 4;
    add.u64 %rd6, %rd1, %rd5;

    // Decompose tid into Q6_K position:
    // chunk = tid / 128 (0 or 1)
    shr.u32 %r11, %r2, 7;
    // pos_in_chunk = tid % 128
    and.b32 %r12, %r2, 127;
    // sub = pos_in_chunk / 32 (0-3)
    shr.u32 %r13, %r12, 5;
    // l = pos_in_chunk % 32 (0-31)
    and.b32 %r14, %r12, 31;

    // ql_off = chunk * 64
    shl.b32 %r15, %r11, 6;
    // qh_off = chunk * 32
    shl.b32 %r16, %r11, 5;
    // sc_off = chunk * 8
    shl.b32 %r17, %r11, 3;
    // is = l / 16 (0 or 1)
    shr.u32 %r18, %r14, 4;

    // ql byte index: ql_off + l + (sub & 1) * 32
    and.b32 %r19, %r13, 1;
    shl.b32 %r19, %r19, 5;         // (sub & 1) * 32
    add.u32 %r19, %r19, %r15;      // + ql_off
    add.u32 %r19, %r19, %r14;      // + l
    // %r19 = ql byte index within ql[128]

    // use_high_nibble = (sub >= 2)
    setp.ge.u32 %p0, %r13, 2;

    // qh shift = sub * 2
    shl.b32 %r20, %r13, 1;         // sub * 2

    // scale index = sc_off + is + sub * 2
    add.u32 %r21, %r17, %r18;      // sc_off + is
    add.u32 %r21, %r21, %r20;      // + sub * 2
    // %r21 = scale index within scales[16]

    mov.f32 %f0, 0f00000000;
    mov.u32 %r10, 0;
Q6K_BLOCK_LOOP:
    setp.ge.u32 %p1, %r10, %r6;
    @%p1 bra Q6K_BLOCK_DONE;

    // Block address: weight_row + block_idx * 210
    mul.lo.u32 %r22, %r10, 210;
    cvt.u64.u32 %rd7, %r22;
    add.u64 %rd8, %rd4, %rd7;

    // Load d (f16) at block + 208
    ld.global.b16 %h0, [%rd8+208];
    cvt.f32.f16 %f1, %h0;           // d as f32

    // Load ql byte at block + ql_byte_index
    cvt.u64.u32 %rd9, %r19;
    add.u64 %rd10, %rd8, %rd9;
    ld.global.u8 %r23, [%rd10];

    // Extract nibble
    @%p0 shr.u32 %r24, %r23, 4;     // high nibble
    @!%p0 mov.u32 %r24, %r23;
    and.b32 %r24, %r24, 0xF;         // mask to 4 bits

    // Load qh byte at block + 128 + qh_off + l
    add.u32 %r25, %r16, %r14;        // qh_off + l
    add.u32 %r25, %r25, 128;         // + 128 (qh starts at offset 128)
    cvt.u64.u32 %rd11, %r25;
    add.u64 %rd12, %rd8, %rd11;
    ld.global.u8 %r26, [%rd12];

    // Extract 2 high bits: (qh_byte >> qh_shift) & 3
    shr.u32 %r27, %r26, %r20;
    and.b32 %r27, %r27, 3;

    // q6 = ql_nibble | (qh_bits << 4)
    shl.b32 %r28, %r27, 4;
    or.b32 %r28, %r24, %r28;

    // q6_centered = q6 - 32 (signed)
    sub.u32 %r29, %r28, 32;

    // Load scale (signed i8) at block + 192 + scale_index
    add.u32 %r30, %r21, 192;
    cvt.u64.u32 %rd13, %r30;
    add.u64 %rd14, %rd8, %rd13;
    ld.global.s8 %r31, [%rd14];      // signed byte

    // Dequantize: val = d * sc * q6_centered
    cvt.rn.f32.s32 %f2, %r31;        // sc as f32 (signed)
    cvt.rn.f32.s32 %f3, %r29;        // q6_centered as f32 (signed)
    mul.rn.f32 %f4, %f1, %f2;        // d * sc
    mul.rn.f32 %f5, %f4, %f3;        // d * sc * q6_centered

    // Load input element
    shl.b32 %r32, %r10, 8;           // block_idx * 256
    add.u32 %r32, %r32, %r2;         // + tid
    mul.wide.u32 %rd15, %r32, 4;
    add.u64 %rd16, %rd6, %rd15;
    ld.global.f32 %f6, [%rd16];

    fma.rn.f32 %f0, %f5, %f6, %f0;

    add.u32 %r10, %r10, 1;
    bra Q6K_BLOCK_LOOP;
Q6K_BLOCK_DONE:

    // Tree reduction
    mov.u32 %r33, sdata;
    shl.b32 %r34, %r2, 2;
    add.u32 %r33, %r33, %r34;
    st.shared.f32 [%r33], %f0;
    bar.sync 0;

    mov.u32 %r35, 128;
Q6K_RED:
    setp.ge.u32 %p3, %r2, %r35;
    @%p3 bra Q6K_RED_SKIP;
    add.u32 %r36, %r2, %r35;
    mov.u32 %r37, sdata;
    shl.b32 %r38, %r36, 2;
    add.u32 %r37, %r37, %r38;
    ld.shared.f32 %f7, [%r37];
    mov.u32 %r37, sdata;
    shl.b32 %r38, %r2, 2;
    add.u32 %r37, %r37, %r38;
    ld.shared.f32 %f8, [%r37];
    add.rn.f32 %f8, %f8, %f7;
    st.shared.f32 [%r37], %f8;
Q6K_RED_SKIP:
    bar.sync 0;
    shr.u32 %r35, %r35, 1;
    setp.ge.u32 %p3, %r35, 1;
    @%p3 bra Q6K_RED;

    setp.ne.u32 %p4, %r2, 0;
    @%p4 bra Q6K_DONE;
    mov.u32 %r39, sdata;
    ld.shared.f32 %f9, [%r39];
    mul.lo.u32 %r40, %r5, %r0;
    add.u32 %r40, %r40, %r4;
    mul.wide.u32 %rd17, %r40, 4;
    add.u64 %rd18, %rd2, %rd17;
    st.global.f32 [%rd18], %f9;
Q6K_DONE:
    ret;
}

// -------------------------------------------------------------------------
// rms_norm: per-row RMSNorm
//   out[c] = x[c] * rsqrt(mean(x^2) + eps) * w[c]
//
// Parameters:
//   input ptr (.u64), weight ptr (.u64), output ptr (.u64),
//   rows (.u32), cols (.u32), eps (.f32)
//
// Grid:  (rows, 1, 1) -- one block per row
// Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry rms_norm(
    .param .u64 param_in,
    .param .u64 param_w,
    .param .u64 param_out,
    .param .u32 param_rows,
    .param .u32 param_cols,
    .param .f32 param_eps
)
{
    .reg .u64 %rd<16>;
    .reg .u32 %r<16>;
    .reg .f32 %f<16>;
    .reg .pred %p<4>;
    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd0, [param_in];
    ld.param.u64 %rd1, [param_w];
    ld.param.u64 %rd2, [param_out];
    ld.param.u32 %r0, [param_rows];
    ld.param.u32 %r1, [param_cols];
    ld.param.f32 %f0, [param_eps];

    mov.u32 %r2, %ctaid.x;        // row
    mov.u32 %r3, %tid.x;          // tid
    mov.u32 %r4, %ntid.x;         // 256

    // Row base pointer
    mul.lo.u32 %r5, %r2, %r1;
    mul.wide.u32 %rd3, %r5, 4;
    add.u64 %rd4, %rd0, %rd3;     // &input[row * cols]

    // --- Phase 1: compute sum of squares ---
    mov.f32 %f1, 0f00000000;
    mov.u32 %r6, %r3;
RMS_SQ_LOOP:
    setp.ge.u32 %p0, %r6, %r1;
    @%p0 bra RMS_SQ_DONE;
    mul.wide.u32 %rd5, %r6, 4;
    add.u64 %rd6, %rd4, %rd5;
    ld.global.f32 %f2, [%rd6];
    fma.rn.f32 %f1, %f2, %f2, %f1;
    add.u32 %r6, %r6, %r4;
    bra RMS_SQ_LOOP;
RMS_SQ_DONE:

    // Store to shared memory for reduction
    mov.u32 %r7, sdata;
    shl.b32 %r8, %r3, 2;
    add.u32 %r7, %r7, %r8;
    st.shared.f32 [%r7], %f1;
    bar.sync 0;

    // Tree reduction (start at ntid.x/2)
    shr.u32 %r9, %r4, 1;
RMS_RED:
    setp.ge.u32 %p0, %r3, %r9;
    @%p0 bra RMS_RED_SKIP;
    add.u32 %r10, %r3, %r9;
    mov.u32 %r11, sdata;
    shl.b32 %r12, %r10, 2;
    add.u32 %r11, %r11, %r12;
    ld.shared.f32 %f3, [%r11];
    mov.u32 %r11, sdata;
    shl.b32 %r12, %r3, 2;
    add.u32 %r11, %r11, %r12;
    ld.shared.f32 %f4, [%r11];
    add.rn.f32 %f4, %f4, %f3;
    st.shared.f32 [%r11], %f4;
RMS_RED_SKIP:
    bar.sync 0;
    shr.u32 %r9, %r9, 1;
    setp.ge.u32 %p0, %r9, 1;
    @%p0 bra RMS_RED;

    // mean_sq = sdata[0] / cols, then rsqrt(mean_sq + eps)
    mov.u32 %r11, sdata;
    ld.shared.f32 %f5, [%r11];
    cvt.rn.f32.u32 %f6, %r1;
    div.approx.f32 %f5, %f5, %f6;
    add.rn.f32 %f5, %f5, %f0;
    rsqrt.approx.f32 %f7, %f5;    // inv_rms
    bar.sync 0;

    // --- Phase 2: normalize and write output ---
    add.u64 %rd7, %rd2, %rd3;     // &output[row * cols]
    mov.u32 %r6, %r3;
RMS_NORM_LOOP:
    setp.ge.u32 %p0, %r6, %r1;
    @%p0 bra RMS_NORM_DONE;
    mul.wide.u32 %rd5, %r6, 4;
    add.u64 %rd6, %rd4, %rd5;
    ld.global.f32 %f8, [%rd6];
    // x * inv_rms
    mul.rn.f32 %f8, %f8, %f7;
    // * weight[c]
    add.u64 %rd8, %rd1, %rd5;
    ld.global.f32 %f9, [%rd8];
    mul.rn.f32 %f8, %f8, %f9;
    // Store
    add.u64 %rd9, %rd7, %rd5;
    st.global.f32 [%rd9], %f8;
    add.u32 %r6, %r6, %r4;
    bra RMS_NORM_LOOP;
RMS_NORM_DONE:
    ret;
}

// -------------------------------------------------------------------------
// silu: element-wise SiLU (Swish)
//   y = x / (1 + exp(-x))
//   = x * sigmoid(x)
//
// exp(-x) = exp2(-x * log2(e))
//
// Parameters: input ptr (.u64), output ptr (.u64), n (.u32)
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry silu(
    .param .u64 param_in,
    .param .u64 param_out,
    .param .u32 param_n
)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<6>;
    .reg .f32 %f<10>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_in];
    ld.param.u64 %rd1, [param_out];
    ld.param.u32 %r0, [param_n];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r1, %r1, %r2;
    mov.u32 %r3, %tid.x;
    add.u32 %r1, %r1, %r3;
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra SILU_DONE;

    mul.wide.u32 %rd2, %r1, 4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];    // x

    // sigmoid(x) = 1 / (1 + exp(-x))
    // exp(-x) = exp2(-x * log2(e))
    mov.f32 %f1, 0f3FB8AA3B;      // log2(e)
    neg.f32 %f2, %f0;             // -x
    mul.rn.f32 %f2, %f2, %f1;     // -x * log2(e)
    ex2.approx.f32 %f2, %f2;      // exp(-x)
    mov.f32 %f3, 0f3F800000;      // 1.0
    add.rn.f32 %f4, %f2, %f3;     // 1 + exp(-x)
    div.approx.f32 %f5, %f0, %f4; // x / (1 + exp(-x))

    add.u64 %rd4, %rd1, %rd2;
    st.global.f32 [%rd4], %f5;
SILU_DONE:
    ret;
}

// -------------------------------------------------------------------------
// swiglu: y = silu(gate) * up
//
// Parameters: gate ptr (.u64), up ptr (.u64), output ptr (.u64), n (.u32)
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry swiglu(
    .param .u64 param_gate,
    .param .u64 param_up,
    .param .u64 param_out,
    .param .u32 param_n
)
{
    .reg .u64 %rd<8>;
    .reg .u32 %r<6>;
    .reg .f32 %f<10>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_gate];
    ld.param.u64 %rd1, [param_up];
    ld.param.u64 %rd2, [param_out];
    ld.param.u32 %r0, [param_n];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r1, %r1, %r2;
    mov.u32 %r3, %tid.x;
    add.u32 %r1, %r1, %r3;
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra SWIGLU_DONE;

    mul.wide.u32 %rd3, %r1, 4;

    // Load gate value
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];    // gate

    // silu(gate) = gate / (1 + exp(-gate))
    mov.f32 %f1, 0f3FB8AA3B;      // log2(e)
    neg.f32 %f2, %f0;
    mul.rn.f32 %f2, %f2, %f1;
    ex2.approx.f32 %f2, %f2;
    mov.f32 %f3, 0f3F800000;      // 1.0
    add.rn.f32 %f4, %f2, %f3;
    div.approx.f32 %f5, %f0, %f4; // silu(gate)

    // Load up value
    add.u64 %rd5, %rd1, %rd3;
    ld.global.f32 %f6, [%rd5];

    // y = silu(gate) * up
    mul.rn.f32 %f7, %f5, %f6;

    add.u64 %rd6, %rd2, %rd3;
    st.global.f32 [%rd6], %f7;
SWIGLU_DONE:
    ret;
}

// -------------------------------------------------------------------------
// geglu: y = gelu(gate) * up
//
// Parameters: gate ptr (.u64), up ptr (.u64), output ptr (.u64), n (.u32)
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry geglu(
    .param .u64 param_gate,
    .param .u64 param_up,
    .param .u64 param_out,
    .param .u32 param_n
)
{
    .reg .u64 %rd<8>;
    .reg .u32 %r<6>;
    .reg .f32 %f<20>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_gate];
    ld.param.u64 %rd1, [param_up];
    ld.param.u64 %rd2, [param_out];
    ld.param.u32 %r0, [param_n];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r1, %r1, %r2;
    mov.u32 %r3, %tid.x;
    add.u32 %r1, %r1, %r3;
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra GEGLU_DONE;

    mul.wide.u32 %rd3, %r1, 4;

    // Load gate value
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];    // x = gate

    // gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    mov.f32 %f1, 0f3F4C422A;      // sqrt(2/pi)
    mov.f32 %f2, 0f3D372713;      // 0.044715
    mov.f32 %f3, 0f3F000000;      // 0.5
    mov.f32 %f10, 0f3FB8AA3B;     // log2(e)
    mov.f32 %f11, 0f40000000;     // 2.0
    mov.f32 %f12, 0f3F800000;     // 1.0

    mul.rn.f32 %f4, %f0, %f0;
    mul.rn.f32 %f4, %f4, %f0;     // x^3
    mul.rn.f32 %f4, %f2, %f4;     // 0.044715 * x^3
    add.rn.f32 %f4, %f0, %f4;     // x + 0.044715 * x^3
    mul.rn.f32 %f4, %f1, %f4;     // a

    // tanh(a) = 1 - 2/(exp(2a)+1)
    mul.rn.f32 %f5, %f4, %f11;
    mul.rn.f32 %f5, %f5, %f10;
    ex2.approx.f32 %f5, %f5;
    add.rn.f32 %f6, %f5, %f12;
    div.approx.f32 %f6, %f11, %f6;
    sub.rn.f32 %f7, %f12, %f6;    // tanh(a)

    add.rn.f32 %f8, %f12, %f7;    // 1 + tanh
    mul.rn.f32 %f8, %f3, %f8;     // 0.5 * (1 + tanh)
    mul.rn.f32 %f9, %f0, %f8;     // gelu(gate)

    // Load up value
    add.u64 %rd5, %rd1, %rd3;
    ld.global.f32 %f13, [%rd5];

    // y = gelu(gate) * up
    mul.rn.f32 %f14, %f9, %f13;

    add.u64 %rd6, %rd2, %rd3;
    st.global.f32 [%rd6], %f14;
GEGLU_DONE:
    ret;
}

// -------------------------------------------------------------------------
// rope_norm: Interleaved RoPE (rotary position embeddings)
//
// Pairs: (x[2i], x[2i+1]) for each head dimension pair.
//   x_new[2i]   = x[2i]   * cos(theta) - x[2i+1] * sin(theta)
//   x_new[2i+1] = x[2i]   * sin(theta) + x[2i+1] * cos(theta)
// where theta = (pos_offset + seq_idx) * base^(-2i/head_dim)
//
// Parameters:
//   input ptr (.u64)    -- input tensor [seq_len, n_heads * head_dim]
//   output ptr (.u64)   -- output tensor (same shape)
//   pos_offset (.u32)   -- starting position offset
//   freq_base (.f32)    -- RoPE base frequency (e.g. 10000.0)
//   head_dim (.u32)     -- dimension per head
//   rope_dim (.u32)     -- number of rotated dims per head (must be even)
//   n_heads (.u32)      -- number of attention heads
//   seq_len (.u32)      -- sequence length
//
// Grid:  (ceil(rope_dim/2 / 16), ceil(n_heads / 16), seq_len)
// Block: (16, 16, 1)
//   threadIdx.x + blockIdx.x*16 -> pair_idx (0..rope_dim/2)
//   threadIdx.y + blockIdx.y*16 -> head_idx (0..n_heads)
//   blockIdx.z -> seq_idx (0..seq_len)
// -------------------------------------------------------------------------
.visible .entry rope_norm(
    .param .u64 param_input,
    .param .u64 param_output,
    .param .u32 param_pos_offset,
    .param .f32 param_freq_base,
    .param .u32 param_head_dim,
    .param .u32 param_rope_dim,
    .param .u32 param_n_heads,
    .param .u32 param_seq_len
)
{
    .reg .u64 %rd<10>;
    .reg .u32 %r<20>;
    .reg .f32 %f<20>;
    .reg .pred %p<3>;

    ld.param.u64 %rd0, [param_input];
    ld.param.u64 %rd1, [param_output];
    ld.param.u32 %r0, [param_pos_offset];
    ld.param.f32 %f0, [param_freq_base];
    ld.param.u32 %r1, [param_head_dim];
    ld.param.u32 %r2, [param_rope_dim];
    ld.param.u32 %r3, [param_n_heads];
    ld.param.u32 %r4, [param_seq_len];

    // pair_idx = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mul.lo.u32 %r5, %r5, %r6;
    mov.u32 %r7, %tid.x;
    add.u32 %r5, %r5, %r7;       // pair_idx

    // head_idx = blockIdx.y * blockDim.y + threadIdx.y
    mov.u32 %r8, %ctaid.y;
    mov.u32 %r9, %ntid.y;
    mul.lo.u32 %r8, %r8, %r9;
    mov.u32 %r10, %tid.y;
    add.u32 %r8, %r8, %r10;      // head_idx

    // seq_idx = blockIdx.z
    mov.u32 %r11, %ctaid.z;

    // Bounds check: pair_idx < rope_dim/2, head_idx < n_heads
    shr.u32 %r12, %r2, 1;         // half_rope = rope_dim / 2
    setp.ge.u32 %p0, %r5, %r12;
    @%p0 bra ROPE_N_DONE;
    setp.ge.u32 %p1, %r8, %r3;
    @%p1 bra ROPE_N_DONE;

    // Compute theta = (pos_offset + seq_idx) * freq_base^(-2*pair_idx / rope_dim)
    add.u32 %r13, %r0, %r11;      // pos = pos_offset + seq_idx
    shl.b32 %r14, %r5, 1;         // 2 * pair_idx
    cvt.rn.f32.u32 %f1, %r14;     // (float)(2*pair_idx)
    cvt.rn.f32.u32 %f2, %r2;      // (float)rope_dim
    div.approx.f32 %f3, %f1, %f2; // 2*pair_idx / rope_dim
    neg.f32 %f3, %f3;             // -2*pair_idx / rope_dim
    lg2.approx.f32 %f4, %f0;      // log2(freq_base)
    mul.rn.f32 %f3, %f3, %f4;     // -2*pair_idx/rope_dim * log2(freq_base)
    ex2.approx.f32 %f5, %f3;      // freq_base^(-2*pair_idx/rope_dim)
    cvt.rn.f32.u32 %f6, %r13;     // (float)pos
    mul.rn.f32 %f7, %f6, %f5;     // theta = pos * freq

    cos.approx.f32 %f8, %f7;
    sin.approx.f32 %f9, %f7;

    // Compute linear index: (seq_idx * n_heads + head_idx) * head_dim + 2*pair_idx
    mul.lo.u32 %r15, %r11, %r3;   // seq_idx * n_heads
    add.u32 %r15, %r15, %r8;      // + head_idx
    mul.lo.u32 %r15, %r15, %r1;   // * head_dim
    add.u32 %r15, %r15, %r14;     // + 2*pair_idx

    // Input addresses
    mul.wide.u32 %rd2, %r15, 4;
    add.u64 %rd3, %rd0, %rd2;     // &input[idx]

    // Output addresses
    add.u64 %rd4, %rd1, %rd2;     // &output[idx]

    // Load x[2i] and x[2i+1] from input
    ld.global.f32 %f10, [%rd3];
    ld.global.f32 %f11, [%rd3+4];

    // Rotate
    // x_new[2i]   = x[2i]   * cos - x[2i+1] * sin
    // x_new[2i+1] = x[2i]   * sin + x[2i+1] * cos
    mul.rn.f32 %f12, %f10, %f8;
    mul.rn.f32 %f13, %f11, %f9;
    sub.rn.f32 %f14, %f12, %f13;  // new_x0

    mul.rn.f32 %f15, %f10, %f9;
    mul.rn.f32 %f16, %f11, %f8;
    add.rn.f32 %f17, %f15, %f16;  // new_x1

    // Store to output
    st.global.f32 [%rd4], %f14;
    st.global.f32 [%rd4+4], %f17;
ROPE_N_DONE:
    ret;
}

// -------------------------------------------------------------------------
// rope_neox: Half-split RoPE (GPT-NeoX / LLaMA style)
//
// Pairs: (x[i], x[i + half]) for each head.
//   x_new[i]        = x[i]      * cos(theta) - x[i+half] * sin(theta)
//   x_new[i + half] = x[i]      * sin(theta) + x[i+half] * cos(theta)
// where theta = (pos_offset + seq_idx) * base^(-2i/head_dim)
//
// Parameters: same as rope_norm (8 params)
// Grid:  (ceil(rope_dim/2 / 16), ceil(n_heads / 16), seq_len)
// Block: (16, 16, 1)
// -------------------------------------------------------------------------
.visible .entry rope_neox(
    .param .u64 param_input,
    .param .u64 param_output,
    .param .u32 param_pos_offset,
    .param .f32 param_freq_base,
    .param .u32 param_head_dim,
    .param .u32 param_rope_dim,
    .param .u32 param_n_heads,
    .param .u32 param_seq_len
)
{
    .reg .u64 %rd<12>;
    .reg .u32 %r<20>;
    .reg .f32 %f<20>;
    .reg .pred %p<3>;

    ld.param.u64 %rd0, [param_input];
    ld.param.u64 %rd1, [param_output];
    ld.param.u32 %r0, [param_pos_offset];
    ld.param.f32 %f0, [param_freq_base];
    ld.param.u32 %r1, [param_head_dim];
    ld.param.u32 %r2, [param_rope_dim];
    ld.param.u32 %r3, [param_n_heads];
    ld.param.u32 %r4, [param_seq_len];

    // pair_idx = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mul.lo.u32 %r5, %r5, %r6;
    mov.u32 %r7, %tid.x;
    add.u32 %r5, %r5, %r7;       // pair_idx (i)

    // head_idx = blockIdx.y * blockDim.y + threadIdx.y
    mov.u32 %r8, %ctaid.y;
    mov.u32 %r9, %ntid.y;
    mul.lo.u32 %r8, %r8, %r9;
    mov.u32 %r10, %tid.y;
    add.u32 %r8, %r8, %r10;      // head_idx

    // seq_idx = blockIdx.z
    mov.u32 %r11, %ctaid.z;

    // Bounds check: pair_idx < rope_dim/2, head_idx < n_heads
    shr.u32 %r12, %r2, 1;         // half_rope = rope_dim / 2
    setp.ge.u32 %p0, %r5, %r12;
    @%p0 bra ROPE_X_DONE;
    setp.ge.u32 %p1, %r8, %r3;
    @%p1 bra ROPE_X_DONE;

    // Compute theta = (pos_offset + seq_idx) * freq_base^(-2*i / rope_dim)
    add.u32 %r13, %r0, %r11;      // pos = pos_offset + seq_idx
    shl.b32 %r14, %r5, 1;         // 2 * pair_idx
    cvt.rn.f32.u32 %f1, %r14;
    cvt.rn.f32.u32 %f2, %r2;      // rope_dim (not head_dim)
    div.approx.f32 %f3, %f1, %f2;
    neg.f32 %f3, %f3;
    lg2.approx.f32 %f4, %f0;
    mul.rn.f32 %f3, %f3, %f4;
    ex2.approx.f32 %f5, %f3;
    cvt.rn.f32.u32 %f6, %r13;
    mul.rn.f32 %f7, %f6, %f5;

    cos.approx.f32 %f8, %f7;
    sin.approx.f32 %f9, %f7;

    // Base offset: (seq_idx * n_heads + head_idx) * head_dim
    mul.lo.u32 %r15, %r11, %r3;
    add.u32 %r15, %r15, %r8;
    mul.lo.u32 %r15, %r15, %r1;

    // NeoX pairing: (x[base + i], x[base + i + half_rope])
    add.u32 %r16, %r15, %r5;      // base + i
    add.u32 %r17, %r16, %r12;     // base + i + half_rope

    // Input addresses
    mul.wide.u32 %rd2, %r16, 4;
    add.u64 %rd3, %rd0, %rd2;     // &input[base + i]
    mul.wide.u32 %rd4, %r17, 4;
    add.u64 %rd5, %rd0, %rd4;     // &input[base + i + half]

    // Output addresses
    add.u64 %rd6, %rd1, %rd2;     // &output[base + i]
    add.u64 %rd7, %rd1, %rd4;     // &output[base + i + half]

    // Load from input
    ld.global.f32 %f10, [%rd3];
    ld.global.f32 %f11, [%rd5];

    // Rotate (NeoX style: first half paired with second half)
    mul.rn.f32 %f12, %f10, %f8;
    mul.rn.f32 %f13, %f11, %f9;
    sub.rn.f32 %f14, %f12, %f13;

    mul.rn.f32 %f15, %f10, %f9;
    mul.rn.f32 %f16, %f11, %f8;
    add.rn.f32 %f17, %f15, %f16;

    // Store to output
    st.global.f32 [%rd6], %f14;
    st.global.f32 [%rd7], %f17;
ROPE_X_DONE:
    ret;
}

// -------------------------------------------------------------------------
// causal_mask: set scores[i][j] = -inf where j > i + offset
//
// Parameters:
//   scores ptr (.u64), rows (.u32), cols (.u32), offset (.u32)
//
// Grid:  (ceil(cols/16), ceil(rows/16), 1)
// Block: (16, 16, 1)
// -------------------------------------------------------------------------
.visible .entry causal_mask(
    .param .u64 param_scores,
    .param .u32 param_rows,
    .param .u32 param_cols,
    .param .u32 param_offset
)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<12>;
    .reg .f32 %f0;
    .reg .pred %p<4>;

    ld.param.u64 %rd0, [param_scores];
    ld.param.u32 %r0, [param_rows];
    ld.param.u32 %r1, [param_cols];
    ld.param.u32 %r2, [param_offset];

    // col = blockIdx.x * 16 + threadIdx.x
    mov.u32 %r3, %ctaid.x;
    shl.b32 %r3, %r3, 4;
    mov.u32 %r4, %tid.x;
    add.u32 %r3, %r3, %r4;

    // row = blockIdx.y * 16 + threadIdx.y
    mov.u32 %r5, %ctaid.y;
    shl.b32 %r5, %r5, 4;
    mov.u32 %r6, %tid.y;
    add.u32 %r5, %r5, %r6;

    // Bounds check
    setp.ge.u32 %p0, %r3, %r1;
    setp.ge.u32 %p1, %r5, %r0;
    or.pred %p2, %p0, %p1;
    @%p2 bra CMASK_DONE;

    // Check if j > i + offset (causal condition)
    add.u32 %r7, %r5, %r2;        // i + offset
    setp.le.u32 %p0, %r3, %r7;    // j <= i + offset => keep
    @%p0 bra CMASK_DONE;

    // Set to -infinity (-FLT_MAX)
    mul.lo.u32 %r8, %r5, %r1;
    add.u32 %r8, %r8, %r3;
    mul.wide.u32 %rd1, %r8, 4;
    add.u64 %rd2, %rd0, %rd1;
    mov.f32 %f0, 0fFF800000;      // -inf (IEEE 754)
    st.global.f32 [%rd2], %f0;
CMASK_DONE:
    ret;
}

// -------------------------------------------------------------------------
// mul_elementwise: c[i] = a[i] * b[i % b_len]
// Supports broadcast on b.
//
// Parameters: a ptr (.u64), b ptr (.u64), c ptr (.u64),
//             n (.u32), b_len (.u32)
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry mul_elementwise(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n,
    .param .u32 param_b_len
)
{
    .reg .u64 %rd<8>;
    .reg .u32 %r<8>;
    .reg .f32 %f<4>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_a];
    ld.param.u64 %rd1, [param_b];
    ld.param.u64 %rd2, [param_c];
    ld.param.u32 %r0, [param_n];
    ld.param.u32 %r1, [param_b_len];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mul.lo.u32 %r2, %r2, %r3;
    mov.u32 %r4, %tid.x;
    add.u32 %r2, %r2, %r4;       // idx
    setp.ge.u32 %p0, %r2, %r0;
    @%p0 bra MUL_EW_DONE;

    // Load a[idx]
    mul.wide.u32 %rd3, %r2, 4;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];

    // b_idx = idx % b_len
    rem.u32 %r5, %r2, %r1;
    mul.wide.u32 %rd5, %r5, 4;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.f32 %f1, [%rd6];

    mul.rn.f32 %f2, %f0, %f1;

    add.u64 %rd7, %rd2, %rd3;
    st.global.f32 [%rd7], %f2;
MUL_EW_DONE:
    ret;
}

// -------------------------------------------------------------------------
// tanh_kernel: element-wise tanh
//   tanh(x) = 1 - 2/(exp(2x)+1)
//
// Parameters: input ptr (.u64), output ptr (.u64), n (.u32)
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry tanh_kernel(
    .param .u64 param_in,
    .param .u64 param_out,
    .param .u32 param_n
)
{
    .reg .u64 %rd<6>;
    .reg .u32 %r<6>;
    .reg .f32 %f<10>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_in];
    ld.param.u64 %rd1, [param_out];
    ld.param.u32 %r0, [param_n];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r1, %r1, %r2;
    mov.u32 %r3, %tid.x;
    add.u32 %r1, %r1, %r3;
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra TANH_DONE;

    mul.wide.u32 %rd2, %r1, 4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];    // x

    // tanh(x) = 1 - 2/(exp(2x)+1)
    mov.f32 %f1, 0f40000000;      // 2.0
    mov.f32 %f2, 0f3FB8AA3B;      // log2(e)
    mov.f32 %f3, 0f3F800000;      // 1.0

    mul.rn.f32 %f4, %f0, %f1;     // 2x
    mul.rn.f32 %f4, %f4, %f2;     // 2x * log2(e)
    ex2.approx.f32 %f4, %f4;      // exp(2x)
    add.rn.f32 %f5, %f4, %f3;     // exp(2x) + 1
    div.approx.f32 %f6, %f1, %f5; // 2 / (exp(2x)+1)
    sub.rn.f32 %f7, %f3, %f6;     // 1 - 2/(exp(2x)+1)

    add.u64 %rd4, %rd1, %rd2;
    st.global.f32 [%rd4], %f7;
TANH_DONE:
    ret;
}

// -------------------------------------------------------------------------
// l2_normalize: in-place L2 normalization
//   out[i] = x[i] / sqrt(sum(x[j]^2))
//
// Parameters: data ptr (.u64), n (.u32)
// Grid:  (1, 1, 1) -- single block
// Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry l2_normalize(
    .param .u64 param_data,
    .param .u32 param_n
)
{
    .reg .u64 %rd<8>;
    .reg .u32 %r<16>;
    .reg .f32 %f<10>;
    .reg .pred %p<3>;
    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd0, [param_data];
    ld.param.u32 %r0, [param_n];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ntid.x;        // 256

    // --- Phase 1: compute sum of squares ---
    mov.f32 %f0, 0f00000000;
    mov.u32 %r3, %r1;
L2_SQ_LOOP:
    setp.ge.u32 %p0, %r3, %r0;
    @%p0 bra L2_SQ_DONE;
    mul.wide.u32 %rd1, %r3, 4;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.f32 %f1, [%rd2];
    fma.rn.f32 %f0, %f1, %f1, %f0;
    add.u32 %r3, %r3, %r2;
    bra L2_SQ_LOOP;
L2_SQ_DONE:

    mov.u32 %r4, sdata;
    shl.b32 %r5, %r1, 2;
    add.u32 %r4, %r4, %r5;
    st.shared.f32 [%r4], %f0;
    bar.sync 0;

    // Tree reduction (start at ntid.x/2)
    shr.u32 %r6, %r2, 1;
L2_RED:
    setp.ge.u32 %p0, %r1, %r6;
    @%p0 bra L2_RED_SKIP;
    add.u32 %r7, %r1, %r6;
    mov.u32 %r8, sdata;
    shl.b32 %r9, %r7, 2;
    add.u32 %r8, %r8, %r9;
    ld.shared.f32 %f2, [%r8];
    mov.u32 %r8, sdata;
    shl.b32 %r9, %r1, 2;
    add.u32 %r8, %r8, %r9;
    ld.shared.f32 %f3, [%r8];
    add.rn.f32 %f3, %f3, %f2;
    st.shared.f32 [%r8], %f3;
L2_RED_SKIP:
    bar.sync 0;
    shr.u32 %r6, %r6, 1;
    setp.ge.u32 %p0, %r6, 1;
    @%p0 bra L2_RED;

    // inv_norm = rsqrt(sum_sq)
    mov.u32 %r8, sdata;
    ld.shared.f32 %f4, [%r8];
    // Guard against zero
    mov.f32 %f5, 0f00000000;
    setp.eq.f32 %p1, %f4, %f5;
    @%p1 bra L2_NORM_DONE;
    rsqrt.approx.f32 %f6, %f4;
    bar.sync 0;

    // --- Phase 2: normalize ---
    mov.u32 %r3, %r1;
L2_NORM_LOOP:
    setp.ge.u32 %p0, %r3, %r0;
    @%p0 bra L2_NORM_DONE;
    mul.wide.u32 %rd1, %r3, 4;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.f32 %f7, [%rd2];
    mul.rn.f32 %f7, %f7, %f6;
    st.global.f32 [%rd2], %f7;
    add.u32 %r3, %r3, %r2;
    bra L2_NORM_LOOP;
L2_NORM_DONE:
    ret;
}

// -------------------------------------------------------------------------
// embedding_lookup: gather rows from embedding table
//   output[i * hidden + j] = table[ids[i] * hidden + j]
//
// Parameters:
//   table ptr (.u64)   -- embedding table (vocab_size x hidden)
//   ids ptr (.u64)     -- token IDs (u32 array, length num_tokens)
//   output ptr (.u64)  -- output (num_tokens x hidden)
//   num_tokens (.u32)
//   hidden (.u32)
//
// Grid:  (ceil(hidden/256), num_tokens, 1)
// Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry embedding_lookup(
    .param .u64 param_table,
    .param .u64 param_ids,
    .param .u64 param_output,
    .param .u32 param_num_tokens,
    .param .u32 param_hidden
)
{
    .reg .u64 %rd<10>;
    .reg .u32 %r<12>;
    .reg .f32 %f0;
    .reg .pred %p<3>;

    ld.param.u64 %rd0, [param_table];
    ld.param.u64 %rd1, [param_ids];
    ld.param.u64 %rd2, [param_output];
    ld.param.u32 %r0, [param_num_tokens];
    ld.param.u32 %r1, [param_hidden];

    // j = blockIdx.x * blockDim.x + threadIdx.x  (hidden dim index)
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mul.lo.u32 %r2, %r2, %r3;
    mov.u32 %r4, %tid.x;
    add.u32 %r2, %r2, %r4;       // j

    // i = blockIdx.y  (token index)
    mov.u32 %r5, %ctaid.y;

    // Bounds check
    setp.ge.u32 %p0, %r2, %r1;
    setp.ge.u32 %p1, %r5, %r0;
    or.pred %p2, %p0, %p1;
    @%p2 bra EMBD_DONE;

    // Load ids[i]
    mul.wide.u32 %rd3, %r5, 4;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.u32 %r6, [%rd4];    // token_id

    // table offset: token_id * hidden + j
    mul.lo.u32 %r7, %r6, %r1;
    add.u32 %r7, %r7, %r2;
    mul.wide.u32 %rd5, %r7, 4;
    add.u64 %rd6, %rd0, %rd5;
    ld.global.f32 %f0, [%rd6];

    // output offset: i * hidden + j
    mul.lo.u32 %r8, %r5, %r1;
    add.u32 %r8, %r8, %r2;
    mul.wide.u32 %rd7, %r8, 4;
    add.u64 %rd8, %rd2, %rd7;
    st.global.f32 [%rd8], %f0;
EMBD_DONE:
    ret;
}

// -------------------------------------------------------------------------
// grouped_attn_decode: fused grouped-query attention for single-token decode
//
// Implements online-softmax: pass 1 finds global max, pass 2 does tiled
// exp(score-max) + weighted V accumulation in shared memory.
//
// Supports GQA via kv_head = h * num_kv_heads / num_heads mapping.
// Supports softcap via softcap * tanh(score / softcap).
//
// Parameters:
//   Q          (.u64)  [1, num_heads * head_dim]
//   K          (.u64)  [max_len, num_kv_heads * head_dim]
//   V          (.u64)  [max_len, num_kv_heads * head_dim]
//   output     (.u64)  [1, num_heads * head_dim]
//   num_heads  (.u32)
//   num_kv_heads (.u32)
//   head_dim   (.u32)
//   total_len  (.u32)
//   attn_scale (.f32)
//   softcap    (.f32)
//
// Grid:  (num_heads, 1, 1) -- one block per Q head
// Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry grouped_attn_decode(
    .param .u64 param_Q,
    .param .u64 param_K,
    .param .u64 param_V,
    .param .u64 param_output,
    .param .u32 param_num_heads,
    .param .u32 param_num_kv_heads,
    .param .u32 param_head_dim,
    .param .u32 param_total_len,
    .param .f32 param_attn_scale,
    .param .f32 param_softcap
)
{
    .reg .u64 %rd<16>;
    .reg .u32 %r<30>;
    .reg .f32 %f<24>;
    .reg .pred %p<8>;
    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd0, [param_Q];
    ld.param.u64 %rd1, [param_K];
    ld.param.u64 %rd2, [param_V];
    ld.param.u64 %rd3, [param_output];
    ld.param.u32 %r0, [param_num_heads];
    ld.param.u32 %r1, [param_num_kv_heads];
    ld.param.u32 %r2, [param_head_dim];
    ld.param.u32 %r3, [param_total_len];
    ld.param.f32 %f0, [param_attn_scale];
    ld.param.f32 %f1, [param_softcap];

    // h = blockIdx.x
    mov.u32 %r4, %ctaid.x;
    setp.ge.u32 %p0, %r4, %r0;
    @%p0 bra GAD_EXIT;

    // lid = threadIdx.x, tpg = blockDim.x
    mov.u32 %r5, %tid.x;
    mov.u32 %r6, %ntid.x;

    // kv_dim = num_kv_heads * head_dim
    mul.lo.u32 %r7, %r1, %r2;

    // kv_head = h * num_kv_heads / num_heads (integer)
    mul.lo.u32 %r8, %r4, %r1;
    cvt.rn.f32.u32 %f2, %r8;
    cvt.rn.f32.u32 %f3, %r0;
    div.approx.f32 %f2, %f2, %f3;
    cvt.rzi.u32.f32 %r8, %f2;

    // kv_off = kv_head * head_dim
    mul.lo.u32 %r9, %r8, %r2;

    // q_head = Q + h * head_dim (byte offset)
    mul.lo.u32 %r10, %r4, %r2;
    mul.wide.u32 %rd4, %r10, 4;
    add.u64 %rd4, %rd0, %rd4;

    // Constants
    mov.f32 %f4, 0f00000000;          // 0.0
    mov.f32 %f5, 0f3FB8AA3B;          // log2(e)
    mov.f32 %f6, 0f40000000;          // 2.0
    mov.f32 %f7, 0f3F800000;          // 1.0

    // ---- Pass 1: find global max of attention scores ----
    mov.f32 %f8, 0fFF800000;          // local_max = -inf
    mov.u32 %r11, %r5;                // j = lid
GAD_MAX_LOOP:
    setp.ge.u32 %p0, %r11, %r3;
    @%p0 bra GAD_MAX_REDUCE;

    // dot = sum(q_head[d] * K[j * kv_dim + kv_off + d])
    mov.f32 %f9, 0f00000000;
    mul.lo.u32 %r12, %r11, %r7;
    add.u32 %r12, %r12, %r9;
    mul.wide.u32 %rd5, %r12, 4;
    add.u64 %rd5, %rd1, %rd5;
    mov.u32 %r13, 0;
GAD_DOT1:
    setp.ge.u32 %p1, %r13, %r2;
    @%p1 bra GAD_DOT1_END;
    mul.wide.u32 %rd6, %r13, 4;
    add.u64 %rd7, %rd4, %rd6;
    ld.global.f32 %f10, [%rd7];
    add.u64 %rd8, %rd5, %rd6;
    ld.global.f32 %f11, [%rd8];
    fma.rn.f32 %f9, %f10, %f11, %f9;
    add.u32 %r13, %r13, 1;
    bra GAD_DOT1;
GAD_DOT1_END:

    // s = dot * attn_scale
    mul.rn.f32 %f9, %f9, %f0;

    // softcap: if softcap > 0, s = softcap * tanh(s / softcap)
    setp.le.f32 %p2, %f1, %f4;
    @%p2 bra GAD_NOSC1;
    div.approx.f32 %f12, %f9, %f1;
    // tanh(x) = 1 - 2/(exp(2x)+1)
    mul.rn.f32 %f12, %f12, %f6;
    mul.rn.f32 %f12, %f12, %f5;
    ex2.approx.f32 %f12, %f12;
    add.rn.f32 %f12, %f12, %f7;
    div.approx.f32 %f12, %f6, %f12;
    sub.rn.f32 %f12, %f7, %f12;
    mul.rn.f32 %f9, %f1, %f12;
GAD_NOSC1:

    max.f32 %f8, %f8, %f9;
    add.u32 %r11, %r11, %r6;
    bra GAD_MAX_LOOP;
GAD_MAX_REDUCE:

    // Reduce max across threadgroup via shared memory
    mov.u32 %r14, sdata;
    shl.b32 %r15, %r5, 2;
    add.u32 %r14, %r14, %r15;
    st.shared.f32 [%r14], %f8;
    bar.sync 0;

    mov.u32 %r16, 128;
GAD_RMAX:
    setp.ge.u32 %p0, %r5, %r16;
    @%p0 bra GAD_RMAX_SKIP;
    add.u32 %r17, %r5, %r16;
    mov.u32 %r18, sdata;
    shl.b32 %r19, %r17, 2;
    add.u32 %r18, %r18, %r19;
    ld.shared.f32 %f10, [%r18];
    mov.u32 %r18, sdata;
    shl.b32 %r19, %r5, 2;
    add.u32 %r18, %r18, %r19;
    ld.shared.f32 %f11, [%r18];
    max.f32 %f11, %f11, %f10;
    st.shared.f32 [%r18], %f11;
GAD_RMAX_SKIP:
    bar.sync 0;
    shr.u32 %r16, %r16, 1;
    setp.ge.u32 %p0, %r16, 1;
    @%p0 bra GAD_RMAX;

    mov.u32 %r18, sdata;
    ld.shared.f32 %f13, [%r18];       // gmax
    bar.sync 0;

    // ---- Pass 2: tiled score + V accumulation ----
    mov.f32 %f14, 0f00000000;         // v_acc (per thread, for dim lid)
    mov.f32 %f15, 0f00000000;         // local_sum_exp

    mov.u32 %r20, 0;                  // tile = 0
GAD_TILE:
    setp.ge.u32 %p0, %r20, %r3;
    @%p0 bra GAD_TILE_END;

    // Step A: each thread computes w = exp(score - gmax) for position tile + lid
    add.u32 %r11, %r20, %r5;          // j = tile + lid
    mov.f32 %f16, 0f00000000;         // w = 0
    setp.ge.u32 %p3, %r11, %r3;
    @%p3 bra GAD_STORE_W;

    // dot product Q . K[j]
    mov.f32 %f9, 0f00000000;
    mul.lo.u32 %r12, %r11, %r7;
    add.u32 %r12, %r12, %r9;
    mul.wide.u32 %rd5, %r12, 4;
    add.u64 %rd5, %rd1, %rd5;
    mov.u32 %r13, 0;
GAD_DOT2:
    setp.ge.u32 %p1, %r13, %r2;
    @%p1 bra GAD_DOT2_END;
    mul.wide.u32 %rd6, %r13, 4;
    add.u64 %rd7, %rd4, %rd6;
    ld.global.f32 %f10, [%rd7];
    add.u64 %rd8, %rd5, %rd6;
    ld.global.f32 %f11, [%rd8];
    fma.rn.f32 %f9, %f10, %f11, %f9;
    add.u32 %r13, %r13, 1;
    bra GAD_DOT2;
GAD_DOT2_END:

    mul.rn.f32 %f9, %f9, %f0;

    // softcap
    setp.le.f32 %p2, %f1, %f4;
    @%p2 bra GAD_NOSC2;
    div.approx.f32 %f12, %f9, %f1;
    mul.rn.f32 %f12, %f12, %f6;
    mul.rn.f32 %f12, %f12, %f5;
    ex2.approx.f32 %f12, %f12;
    add.rn.f32 %f12, %f12, %f7;
    div.approx.f32 %f12, %f6, %f12;
    sub.rn.f32 %f12, %f7, %f12;
    mul.rn.f32 %f9, %f1, %f12;
GAD_NOSC2:

    // w = exp(s - gmax) = exp2((s - gmax) * log2e)
    sub.rn.f32 %f16, %f9, %f13;
    mul.rn.f32 %f16, %f16, %f5;
    ex2.approx.f32 %f16, %f16;

GAD_STORE_W:
    add.rn.f32 %f15, %f15, %f16;
    mov.u32 %r14, sdata;
    shl.b32 %r15, %r5, 2;
    add.u32 %r14, %r14, %r15;
    st.shared.f32 [%r14], %f16;
    bar.sync 0;

    // Step B: threads parallel over head_dim, accumulate weight * V
    setp.ge.u32 %p4, %r5, %r2;
    @%p4 bra GAD_VACC_SKIP;

    // active = min(tpg, total_len - tile)
    sub.u32 %r21, %r3, %r20;
    min.u32 %r21, %r6, %r21;
    mov.u32 %r22, 0;
GAD_VACC:
    setp.ge.u32 %p5, %r22, %r21;
    @%p5 bra GAD_VACC_SKIP;
    // w = shared[t]
    mov.u32 %r14, sdata;
    shl.b32 %r15, %r22, 2;
    add.u32 %r14, %r14, %r15;
    ld.shared.f32 %f17, [%r14];
    // V[(tile+t) * kv_dim + kv_off + lid]
    add.u32 %r23, %r20, %r22;
    mul.lo.u32 %r24, %r23, %r7;
    add.u32 %r24, %r24, %r9;
    add.u32 %r24, %r24, %r5;
    mul.wide.u32 %rd9, %r24, 4;
    add.u64 %rd10, %rd2, %rd9;
    ld.global.f32 %f18, [%rd10];
    fma.rn.f32 %f14, %f17, %f18, %f14;
    add.u32 %r22, %r22, 1;
    bra GAD_VACC;
GAD_VACC_SKIP:
    bar.sync 0;

    add.u32 %r20, %r20, %r6;
    bra GAD_TILE;
GAD_TILE_END:

    // ---- Reduce sum_exp across threadgroup ----
    mov.u32 %r14, sdata;
    shl.b32 %r15, %r5, 2;
    add.u32 %r14, %r14, %r15;
    st.shared.f32 [%r14], %f15;
    bar.sync 0;

    mov.u32 %r16, 128;
GAD_RSUM:
    setp.ge.u32 %p0, %r5, %r16;
    @%p0 bra GAD_RSUM_SKIP;
    add.u32 %r17, %r5, %r16;
    mov.u32 %r18, sdata;
    shl.b32 %r19, %r17, 2;
    add.u32 %r18, %r18, %r19;
    ld.shared.f32 %f10, [%r18];
    mov.u32 %r18, sdata;
    shl.b32 %r19, %r5, 2;
    add.u32 %r18, %r18, %r19;
    ld.shared.f32 %f11, [%r18];
    add.rn.f32 %f11, %f11, %f10;
    st.shared.f32 [%r18], %f11;
GAD_RSUM_SKIP:
    bar.sync 0;
    shr.u32 %r16, %r16, 1;
    setp.ge.u32 %p0, %r16, 1;
    @%p0 bra GAD_RSUM;

    mov.u32 %r18, sdata;
    ld.shared.f32 %f19, [%r18];       // total sum_exp

    // Write normalized output: out[h*head_dim + lid] = v_acc / sum_exp
    setp.ge.u32 %p6, %r5, %r2;
    @%p6 bra GAD_EXIT;
    setp.le.f32 %p7, %f19, %f4;
    @%p7 bra GAD_EXIT;
    div.approx.f32 %f20, %f7, %f19;   // inv_sum = 1.0 / sum_exp
    mul.rn.f32 %f14, %f14, %f20;

    mul.lo.u32 %r25, %r4, %r2;
    add.u32 %r25, %r25, %r5;
    mul.wide.u32 %rd11, %r25, 4;
    add.u64 %rd12, %rd3, %rd11;
    st.global.f32 [%rd12], %f14;
GAD_EXIT:
    ret;
}

// -------------------------------------------------------------------------
// batched_causal_attention: multi-token prefill with online softmax
//
// Fused Q@K^T + causal mask + softmax + @V for multiple query tokens.
// Ported from Metal kernel. Uses online softmax (single pass, no separate
// max-find pass) for numerical stability and efficiency.
//
// Parameters:
//   Q          (.u64)  [n_tokens, num_heads * head_dim]
//   K          (.u64)  [max_len, num_kv_heads * head_dim]
//   V          (.u64)  [max_len, num_kv_heads * head_dim]
//   output     (.u64)  [n_tokens, num_heads * head_dim]
//   num_heads     (.u32)
//   num_kv_heads  (.u32)
//   head_dim      (.u32)
//   n_tokens      (.u32)
//   total_len     (.u32)
//   pos_offset    (.u32)
//   attn_scale    (.f32)
//   softcap       (.f32)
//
// Grid:  (num_heads * n_tokens, 1, 1) -- one block per (head, query_token)
// Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry batched_causal_attention(
    .param .u64 param_Q,
    .param .u64 param_K,
    .param .u64 param_V,
    .param .u64 param_output,
    .param .u32 param_num_heads,
    .param .u32 param_num_kv_heads,
    .param .u32 param_head_dim,
    .param .u32 param_n_tokens,
    .param .u32 param_total_len,
    .param .u32 param_pos_offset,
    .param .f32 param_attn_scale,
    .param .f32 param_softcap
)
{
    .reg .u64 %rd<20>;
    .reg .u32 %r<35>;
    .reg .f32 %f<28>;
    .reg .pred %p<10>;
    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd0, [param_Q];
    ld.param.u64 %rd1, [param_K];
    ld.param.u64 %rd2, [param_V];
    ld.param.u64 %rd3, [param_output];
    ld.param.u32 %r0, [param_num_heads];
    ld.param.u32 %r1, [param_num_kv_heads];
    ld.param.u32 %r2, [param_head_dim];
    ld.param.u32 %r3, [param_n_tokens];
    ld.param.u32 %r4, [param_total_len];
    ld.param.u32 %r5, [param_pos_offset];
    ld.param.f32 %f0, [param_attn_scale];
    ld.param.f32 %f1, [param_softcap];

    // gid = blockIdx.x
    mov.u32 %r6, %ctaid.x;
    // lid = threadIdx.x
    mov.u32 %r7, %tid.x;
    // tpg = blockDim.x
    mov.u32 %r8, %ntid.x;

    // Decompose gid: h = gid % num_heads, q_idx = gid / num_heads
    cvt.rn.f32.u32 %f2, %r6;
    cvt.rn.f32.u32 %f3, %r0;
    div.approx.f32 %f2, %f2, %f3;
    cvt.rzi.u32.f32 %r9, %f2;           // q_idx = gid / num_heads
    mul.lo.u32 %r10, %r9, %r0;
    sub.u32 %r10, %r6, %r10;            // h = gid - q_idx * num_heads

    // Bounds check
    setp.ge.u32 %p0, %r9, %r3;
    @%p0 bra BCA_EXIT;

    // GQA: kv_head = h * num_kv_heads / num_heads
    mul.lo.u32 %r11, %r10, %r1;
    cvt.rn.f32.u32 %f2, %r11;
    div.approx.f32 %f2, %f2, %f3;
    cvt.rzi.u32.f32 %r11, %f2;          // kv_head

    // kv_dim = num_kv_heads * head_dim
    mul.lo.u32 %r12, %r1, %r2;
    // total_dim = num_heads * head_dim
    mul.lo.u32 %r13, %r0, %r2;
    // kv_off = kv_head * head_dim
    mul.lo.u32 %r14, %r11, %r2;

    // Causal mask: max_attend = min(pos_offset + q_idx + 1, total_len)
    add.u32 %r15, %r5, %r9;
    add.u32 %r15, %r15, 1;              // pos_offset + q_idx + 1
    min.u32 %r15, %r15, %r4;            // max_attend

    // q_head = Q + q_idx * total_dim + h * head_dim (byte offset)
    mul.lo.u32 %r16, %r9, %r13;
    mul.lo.u32 %r17, %r10, %r2;
    add.u32 %r16, %r16, %r17;
    mul.wide.u32 %rd4, %r16, 4;
    add.u64 %rd4, %rd0, %rd4;           // q_head ptr

    // Constants
    mov.f32 %f4, 0f00000000;            // 0.0
    mov.f32 %f5, 0f3FB8AA3B;            // log2(e) = 1.4426950
    mov.f32 %f6, 0f40000000;            // 2.0
    mov.f32 %f7, 0f3F800000;            // 1.0

    // Online softmax state
    mov.f32 %f8, 0fFF800000;            // running_max = -inf
    mov.f32 %f9, 0f00000000;            // running_sum = 0
    mov.f32 %f10, 0f00000000;           // v_acc = 0

    // Tile loop: tile = 0, tpg, 2*tpg, ...
    mov.u32 %r18, 0;
BCA_TILE:
    setp.ge.u32 %p1, %r18, %r15;
    @%p1 bra BCA_TILE_END;

    // --- Step A: each thread computes Q.K score for position j = tile + lid ---
    add.u32 %r19, %r18, %r7;            // j = tile + lid
    mov.f32 %f11, 0fFF800000;           // score = -inf (masked by default)
    setp.ge.u32 %p2, %r19, %r15;
    @%p2 bra BCA_STORE_SCORE;

    // dot = sum(q_head[d] * K[j * kv_dim + kv_off + d])
    mov.f32 %f11, 0f00000000;
    mul.lo.u32 %r20, %r19, %r12;
    add.u32 %r20, %r20, %r14;
    mul.wide.u32 %rd5, %r20, 4;
    add.u64 %rd5, %rd1, %rd5;           // K[j] row ptr
    mov.u32 %r21, 0;
BCA_DOT:
    setp.ge.u32 %p3, %r21, %r2;
    @%p3 bra BCA_DOT_END;
    mul.wide.u32 %rd6, %r21, 4;
    add.u64 %rd7, %rd4, %rd6;
    ld.global.f32 %f12, [%rd7];
    add.u64 %rd8, %rd5, %rd6;
    ld.global.f32 %f13, [%rd8];
    fma.rn.f32 %f11, %f12, %f13, %f11;
    add.u32 %r21, %r21, 1;
    bra BCA_DOT;
BCA_DOT_END:

    // score = dot * attn_scale
    mul.rn.f32 %f11, %f11, %f0;

    // softcap: if softcap > 0, score = softcap * tanh(score / softcap)
    setp.le.f32 %p4, %f1, %f4;
    @%p4 bra BCA_STORE_SCORE;
    div.approx.f32 %f14, %f11, %f1;
    // tanh(x) = 1 - 2/(exp(2x)+1)
    mul.rn.f32 %f14, %f14, %f6;
    mul.rn.f32 %f14, %f14, %f5;
    ex2.approx.f32 %f14, %f14;
    add.rn.f32 %f14, %f14, %f7;
    div.approx.f32 %f14, %f6, %f14;
    sub.rn.f32 %f14, %f7, %f14;
    mul.rn.f32 %f11, %f1, %f14;

BCA_STORE_SCORE:
    // Store score to shared memory
    mov.u32 %r22, sdata;
    shl.b32 %r23, %r7, 2;
    add.u32 %r22, %r22, %r23;
    st.shared.f32 [%r22], %f11;
    bar.sync 0;

    // --- Step B: threads parallel over head_dim, online softmax accumulation ---
    setp.ge.u32 %p5, %r7, %r2;
    @%p5 bra BCA_VACC_SKIP;

    // active = min(tpg, max_attend - tile)
    sub.u32 %r24, %r15, %r18;
    min.u32 %r24, %r8, %r24;
    mov.u32 %r25, 0;
BCA_VACC:
    setp.ge.u32 %p6, %r25, %r24;
    @%p6 bra BCA_VACC_SKIP;

    // s = shared[t]
    mov.u32 %r26, sdata;
    shl.b32 %r27, %r25, 2;
    add.u32 %r26, %r26, %r27;
    ld.shared.f32 %f15, [%r26];

    // Online softmax update
    setp.gt.f32 %p7, %f15, %f8;
    @!%p7 bra BCA_SCORE_LE;

    // Case: score > running_max -- apply correction
    // correction = exp(running_max - score) = exp2((running_max - score) * log2e)
    sub.rn.f32 %f16, %f8, %f15;
    mul.rn.f32 %f16, %f16, %f5;
    ex2.approx.f32 %f16, %f16;
    mul.rn.f32 %f9, %f9, %f16;          // running_sum *= correction
    add.rn.f32 %f9, %f9, %f7;           // running_sum += 1.0
    mul.rn.f32 %f10, %f10, %f16;        // v_acc *= correction
    // v_acc += V[(tile+t) * kv_dim + kv_off + lid]
    add.u32 %r28, %r18, %r25;
    mul.lo.u32 %r29, %r28, %r12;
    add.u32 %r29, %r29, %r14;
    add.u32 %r29, %r29, %r7;
    mul.wide.u32 %rd9, %r29, 4;
    add.u64 %rd10, %rd2, %rd9;
    ld.global.f32 %f17, [%rd10];
    add.rn.f32 %f10, %f10, %f17;
    mov.f32 %f8, %f15;                  // running_max = score
    bra BCA_VACC_NEXT;

BCA_SCORE_LE:
    // Case: score <= running_max -- accumulate normally
    // w = exp(score - running_max) = exp2((score - running_max) * log2e)
    sub.rn.f32 %f16, %f15, %f8;
    mul.rn.f32 %f16, %f16, %f5;
    ex2.approx.f32 %f16, %f16;
    add.rn.f32 %f9, %f9, %f16;          // running_sum += w
    // v_acc += w * V[(tile+t) * kv_dim + kv_off + lid]
    add.u32 %r28, %r18, %r25;
    mul.lo.u32 %r29, %r28, %r12;
    add.u32 %r29, %r29, %r14;
    add.u32 %r29, %r29, %r7;
    mul.wide.u32 %rd9, %r29, 4;
    add.u64 %rd10, %rd2, %rd9;
    ld.global.f32 %f17, [%rd10];
    fma.rn.f32 %f10, %f16, %f17, %f10;

BCA_VACC_NEXT:
    add.u32 %r25, %r25, 1;
    bra BCA_VACC;

BCA_VACC_SKIP:
    bar.sync 0;

    // Advance tile
    add.u32 %r18, %r18, %r8;
    bra BCA_TILE;
BCA_TILE_END:

    // --- Write normalized output ---
    setp.ge.u32 %p8, %r7, %r2;
    @%p8 bra BCA_EXIT;
    setp.le.f32 %p9, %f9, %f4;
    @%p9 bra BCA_EXIT;
    div.approx.f32 %f18, %f10, %f9;     // v_acc / running_sum

    // out_head = output + q_idx * total_dim + h * head_dim + lid
    mul.lo.u32 %r30, %r9, %r13;
    mul.lo.u32 %r31, %r10, %r2;
    add.u32 %r30, %r30, %r31;
    add.u32 %r30, %r30, %r7;
    mul.wide.u32 %rd11, %r30, 4;
    add.u64 %rd12, %rd3, %rd11;
    st.global.f32 [%rd12], %f18;
BCA_EXIT:
    ret;
}

// -------------------------------------------------------------------------
// dequant_q4_k: Expand Q4_K quantized weights to F32
//
// Grid:  (N, blocks_per_row, 1) -- one block per (row, super-block)
// Block: (256, 1, 1)            -- one thread per element in super-block
//
// Each thread dequantizes one element and writes F32 to output.
// No dot product, no reduction -- pure dequant for cuBLAS to consume.
// -------------------------------------------------------------------------
.visible .entry dequant_q4_k(
    .param .u64 param_weights,
    .param .u64 param_output,
    .param .u32 param_N,
    .param .u32 param_K
)
{
    .reg .u64 %rd<12>;
    .reg .u32 %r<28>;
    .reg .f32 %f<10>;
    .reg .f16 %h0, %h1;
    .reg .pred %p<3>;

    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_output];
    ld.param.u32 %r0, [param_N];
    ld.param.u32 %r1, [param_K];

    mov.u32 %r2, %tid.x;          // tid (0-255)
    mov.u32 %r3, %ctaid.x;        // row index
    mov.u32 %r4, %ctaid.y;        // super-block index

    // blocks_per_row = K >> 8
    shr.u32 %r5, %r1, 8;
    // bytes_per_row = blocks_per_row * 144
    mul.lo.u32 %r6, %r5, 144;

    // Weight block address: weights + row * bytes_per_row + block * 144
    mul.lo.u32 %r7, %r3, %r6;
    mul.lo.u32 %r8, %r4, 144;
    add.u32 %r7, %r7, %r8;
    cvt.u64.u32 %rd2, %r7;
    add.u64 %rd2, %rd0, %rd2;

    // Decompose tid for Q4_K block structure
    shr.u32 %r9, %r2, 6;          // chunk = tid / 64
    and.b32 %r10, %r2, 63;        // local = tid % 64
    setp.lt.u32 %p0, %r10, 32;    // is_low
    @%p0 mov.u32 %r11, %r10;
    @!%p0 sub.u32 %r11, %r10, 32; // l
    shl.b32 %r12, %r9, 5;         // q_off = chunk * 32
    shl.b32 %r13, %r9, 1;         // scale_idx = chunk * 2
    @!%p0 add.u32 %r13, %r13, 1;

    // Load d, dmin
    ld.global.b16 %h0, [%rd2];
    ld.global.b16 %h1, [%rd2+2];
    cvt.f32.f16 %f1, %h0;
    cvt.f32.f16 %f2, %h1;

    // Scale extraction
    add.u64 %rd3, %rd2, 4;
    cvt.u64.u32 %rd4, %r13;
    add.u64 %rd5, %rd3, %rd4;
    ld.global.u8 %r14, [%rd5];
    add.u64 %rd6, %rd5, 4;
    ld.global.u8 %r15, [%rd6];

    setp.lt.u32 %p1, %r13, 4;
    @%p1 bra DQ4K_SCALE_LOW;

    sub.u64 %rd7, %rd5, 4;
    ld.global.u8 %r16, [%rd7];
    and.b32 %r17, %r15, 0xF;
    shr.u32 %r18, %r16, 6;
    shl.b32 %r18, %r18, 4;
    or.b32 %r17, %r17, %r18;
    shr.u32 %r18, %r15, 4;
    shr.u32 %r19, %r14, 6;
    shl.b32 %r19, %r19, 4;
    or.b32 %r18, %r18, %r19;
    bra DQ4K_SCALE_DONE;

DQ4K_SCALE_LOW:
    and.b32 %r17, %r14, 63;
    and.b32 %r18, %r15, 63;

DQ4K_SCALE_DONE:
    // Load qs byte
    add.u32 %r20, %r12, %r11;
    add.u32 %r20, %r20, 16;
    cvt.u64.u32 %rd8, %r20;
    add.u64 %rd8, %rd2, %rd8;
    ld.global.u8 %r21, [%rd8];

    @%p0 and.b32 %r22, %r21, 0xF;
    @!%p0 shr.u32 %r22, %r21, 4;

    // Dequantize: val = d * sc * q - dmin * m
    cvt.rn.f32.u32 %f3, %r17;
    cvt.rn.f32.u32 %f4, %r18;
    cvt.rn.f32.u32 %f5, %r22;
    mul.rn.f32 %f6, %f1, %f3;
    mul.rn.f32 %f7, %f6, %f5;
    mul.rn.f32 %f8, %f2, %f4;
    sub.rn.f32 %f9, %f7, %f8;

    // Write output[row * K + block * 256 + tid]
    mul.lo.u32 %r23, %r3, %r1;
    shl.b32 %r24, %r4, 8;
    add.u32 %r23, %r23, %r24;
    add.u32 %r23, %r23, %r2;
    mul.wide.u32 %rd9, %r23, 4;
    add.u64 %rd10, %rd1, %rd9;
    st.global.f32 [%rd10], %f9;

    ret;
}

// -------------------------------------------------------------------------
// dequant_q5_k: Expand Q5_K quantized weights to F32
//
// Grid:  (N, blocks_per_row, 1)
// Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry dequant_q5_k(
    .param .u64 param_weights,
    .param .u64 param_output,
    .param .u32 param_N,
    .param .u32 param_K
)
{
    .reg .u64 %rd<12>;
    .reg .u32 %r<30>;
    .reg .f32 %f<10>;
    .reg .f16 %h0, %h1;
    .reg .pred %p<4>;

    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_output];
    ld.param.u32 %r0, [param_N];
    ld.param.u32 %r1, [param_K];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;        // row
    mov.u32 %r4, %ctaid.y;        // super-block

    shr.u32 %r5, %r1, 8;
    mul.lo.u32 %r6, %r5, 176;

    // Block address
    mul.lo.u32 %r7, %r3, %r6;
    mul.lo.u32 %r8, %r4, 176;
    add.u32 %r7, %r7, %r8;
    cvt.u64.u32 %rd2, %r7;
    add.u64 %rd2, %rd0, %rd2;

    // Decompose tid
    shr.u32 %r9, %r2, 6;
    and.b32 %r10, %r2, 63;
    setp.lt.u32 %p0, %r10, 32;
    @%p0 mov.u32 %r11, %r10;
    @!%p0 sub.u32 %r11, %r10, 32;
    shl.b32 %r12, %r9, 5;
    shl.b32 %r13, %r9, 1;
    @!%p0 add.u32 %r13, %r13, 1;

    // qh bit mask
    mov.u32 %r25, 1;
    shl.b32 %r25, %r25, %r13;

    // Load d, dmin
    ld.global.b16 %h0, [%rd2];
    ld.global.b16 %h1, [%rd2+2];
    cvt.f32.f16 %f1, %h0;
    cvt.f32.f16 %f2, %h1;

    // Scale extraction (same as Q4_K)
    add.u64 %rd3, %rd2, 4;
    cvt.u64.u32 %rd4, %r13;
    add.u64 %rd5, %rd3, %rd4;
    ld.global.u8 %r14, [%rd5];
    add.u64 %rd6, %rd5, 4;
    ld.global.u8 %r15, [%rd6];

    setp.lt.u32 %p1, %r13, 4;
    @%p1 bra DQ5K_SCALE_LOW;

    sub.u64 %rd7, %rd5, 4;
    ld.global.u8 %r16, [%rd7];
    and.b32 %r17, %r15, 0xF;
    shr.u32 %r18, %r16, 6;
    shl.b32 %r18, %r18, 4;
    or.b32 %r17, %r17, %r18;
    shr.u32 %r18, %r15, 4;
    shr.u32 %r19, %r14, 6;
    shl.b32 %r19, %r19, 4;
    or.b32 %r18, %r18, %r19;
    bra DQ5K_SCALE_DONE;

DQ5K_SCALE_LOW:
    and.b32 %r17, %r14, 63;
    and.b32 %r18, %r15, 63;

DQ5K_SCALE_DONE:
    // Load qs byte at block + 48 + q_off + l
    add.u32 %r20, %r12, %r11;
    add.u32 %r20, %r20, 48;
    cvt.u64.u32 %rd8, %r20;
    add.u64 %rd8, %rd2, %rd8;
    ld.global.u8 %r21, [%rd8];

    @%p0 and.b32 %r22, %r21, 0xF;
    @!%p0 shr.u32 %r22, %r21, 4;

    // Load qh byte at block + 16 + l
    add.u32 %r26, %r11, 16;
    cvt.u64.u32 %rd8, %r26;
    add.u64 %rd8, %rd2, %rd8;
    ld.global.u8 %r27, [%rd8];

    // high = (qh[l] & qh_mask) ? 16 : 0
    and.b32 %r28, %r27, %r25;
    setp.ne.u32 %p2, %r28, 0;
    selp.u32 %r28, 16, 0, %p2;
    add.u32 %r22, %r22, %r28;

    // Dequantize: val = d * sc * q_total - dmin * m
    cvt.rn.f32.u32 %f3, %r17;
    cvt.rn.f32.u32 %f4, %r18;
    cvt.rn.f32.u32 %f5, %r22;
    mul.rn.f32 %f6, %f1, %f3;
    mul.rn.f32 %f7, %f6, %f5;
    mul.rn.f32 %f8, %f2, %f4;
    sub.rn.f32 %f9, %f7, %f8;

    // Write output
    mul.lo.u32 %r23, %r3, %r1;
    shl.b32 %r24, %r4, 8;
    add.u32 %r23, %r23, %r24;
    add.u32 %r23, %r23, %r2;
    mul.wide.u32 %rd9, %r23, 4;
    add.u64 %rd10, %rd1, %rd9;
    st.global.f32 [%rd10], %f9;

    ret;
}

// -------------------------------------------------------------------------
// dequant_q6_k: Expand Q6_K quantized weights to F32
//
// Grid:  (N, blocks_per_row, 1)
// Block: (256, 1, 1)
// -------------------------------------------------------------------------
.visible .entry dequant_q6_k(
    .param .u64 param_weights,
    .param .u64 param_output,
    .param .u32 param_N,
    .param .u32 param_K
)
{
    .reg .u64 %rd<12>;
    .reg .u32 %r<30>;
    .reg .f32 %f<8>;
    .reg .f16 %h0;
    .reg .pred %p<2>;

    ld.param.u64 %rd0, [param_weights];
    ld.param.u64 %rd1, [param_output];
    ld.param.u32 %r0, [param_N];
    ld.param.u32 %r1, [param_K];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;        // row
    mov.u32 %r4, %ctaid.y;        // super-block

    shr.u32 %r5, %r1, 8;
    mul.lo.u32 %r6, %r5, 210;

    // Block address
    mul.lo.u32 %r7, %r3, %r6;
    mul.lo.u32 %r8, %r4, 210;
    add.u32 %r7, %r7, %r8;
    cvt.u64.u32 %rd2, %r7;
    add.u64 %rd2, %rd0, %rd2;

    // Decompose tid into Q6_K position
    shr.u32 %r9, %r2, 7;          // chunk = tid / 128
    and.b32 %r10, %r2, 127;       // pos_in_chunk
    shr.u32 %r11, %r10, 5;        // sub = pos_in_chunk / 32
    and.b32 %r12, %r10, 31;       // l = pos_in_chunk % 32

    // ql_off = chunk * 64
    shl.b32 %r13, %r9, 6;
    // qh_off = chunk * 32
    shl.b32 %r14, %r9, 5;
    // sc_off = chunk * 8
    shl.b32 %r15, %r9, 3;
    // is = l / 16
    shr.u32 %r16, %r12, 4;

    // ql byte index: ql_off + l + (sub & 1) * 32
    and.b32 %r17, %r11, 1;
    shl.b32 %r17, %r17, 5;
    add.u32 %r17, %r17, %r13;
    add.u32 %r17, %r17, %r12;

    // use_high_nibble = (sub >= 2)
    setp.ge.u32 %p0, %r11, 2;

    // qh shift = sub * 2
    shl.b32 %r18, %r11, 1;

    // scale index = sc_off + is + sub * 2
    add.u32 %r19, %r15, %r16;
    add.u32 %r19, %r19, %r18;

    // Load d at block + 208
    ld.global.b16 %h0, [%rd2+208];
    cvt.f32.f16 %f1, %h0;

    // Load ql byte
    cvt.u64.u32 %rd3, %r17;
    add.u64 %rd4, %rd2, %rd3;
    ld.global.u8 %r20, [%rd4];

    @%p0 shr.u32 %r21, %r20, 4;
    @!%p0 mov.u32 %r21, %r20;
    and.b32 %r21, %r21, 0xF;

    // Load qh byte at block + 128 + qh_off + l
    add.u32 %r22, %r14, %r12;
    add.u32 %r22, %r22, 128;
    cvt.u64.u32 %rd5, %r22;
    add.u64 %rd6, %rd2, %rd5;
    ld.global.u8 %r23, [%rd6];

    // Extract 2 high bits
    shr.u32 %r24, %r23, %r18;
    and.b32 %r24, %r24, 3;

    // q6 = ql_nibble | (qh_bits << 4)
    shl.b32 %r25, %r24, 4;
    or.b32 %r25, %r21, %r25;

    // q6_centered = q6 - 32
    sub.u32 %r26, %r25, 32;

    // Load scale (signed i8) at block + 192 + scale_index
    add.u32 %r27, %r19, 192;
    cvt.u64.u32 %rd7, %r27;
    add.u64 %rd8, %rd2, %rd7;
    ld.global.s8 %r28, [%rd8];

    // Dequantize: val = d * sc * q6_centered
    cvt.rn.f32.s32 %f2, %r28;
    cvt.rn.f32.s32 %f3, %r26;
    mul.rn.f32 %f4, %f1, %f2;
    mul.rn.f32 %f5, %f4, %f3;

    // Write output
    mul.lo.u32 %r23, %r3, %r1;
    shl.b32 %r24, %r4, 8;
    add.u32 %r23, %r23, %r24;
    add.u32 %r23, %r23, %r2;
    mul.wide.u32 %rd9, %r23, 4;
    add.u64 %rd10, %rd1, %rd9;
    st.global.f32 [%rd10], %f5;

    ret;
}

"#,
    "\0"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_module_is_null_terminated() {
        assert!(PTX_MODULE.ends_with('\0'));
    }

    #[test]
    fn ptx_module_has_header() {
        assert!(PTX_MODULE.contains(".version 7.0"));
        assert!(PTX_MODULE.contains(".target sm_53"));
        assert!(PTX_MODULE.contains(".address_size 64"));
    }

    #[test]
    fn ptx_module_contains_ported_kernels() {
        let ported = [
            "gemm",
            "gemm_transpose",
            "gelu",
            "add_tensor",
            "add_bias",
            "scale",
            "layer_norm",
            "softmax_rows",
            "mean_pool",
        ];
        for name in &ported {
            let entry = format!(".visible .entry {}(", name);
            assert!(
                PTX_MODULE.contains(&entry),
                "Missing ported kernel: {}",
                name
            );
        }
    }

    #[test]
    fn ptx_module_contains_new_kernels() {
        let new_kernels = [
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
            "batched_causal_attention",
            "quantized_matmul_q4_k",
            "quantized_matmul_q5_k",
            "quantized_matmul_q6_k",
            "dequant_q4_k",
            "dequant_q5_k",
            "dequant_q6_k",
        ];
        for name in &new_kernels {
            let entry = format!(".visible .entry {}(", name);
            assert!(PTX_MODULE.contains(&entry), "Missing new kernel: {}", name);
        }
    }

    #[test]
    fn ptx_module_total_kernel_count() {
        let count = PTX_MODULE.matches(".visible .entry ").count();
        // 9 ported + 14 new + 1 batched_causal_attention + 3 Q4_K/Q5_K/Q6_K + 3 dequant = 30 kernels
        assert_eq!(count, 30, "Expected 30 kernels, found {}", count);
    }

    #[test]
    fn ptx_module_does_not_contain_excluded_kernels() {
        let excluded = [
            "batched_gemm_transpose",
            "batched_gemm",
            "batched_attention_mask",
            "batched_mean_pool",
            "attention_mask",
            "transpose_heads",
            "untranspose_heads",
            "slice_columns",
            "scatter_columns",
        ];
        for name in &excluded {
            let entry = format!(".visible .entry {}(", name);
            assert!(
                !PTX_MODULE.contains(&entry),
                "Should not contain excluded kernel: {}",
                name
            );
        }
    }
}
