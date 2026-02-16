//! Metal Shading Language (MSL) kernel sources for strata-inference.
//!
//! All compute kernels are compiled at runtime from this single source string.
//! MSL is a C++14-based language; we use `metal_stdlib` for math intrinsics.
//!
//! Ported f32 kernels from strata-core: gemm, gemm_transpose, gelu, add_tensor,
//! add_bias, scale_kernel, layer_norm, softmax_rows, mean_pool.
//!
//! New kernels for quantized inference: quantized_matmul_q8_0, quantized_matmul_q4_0,
//! rms_norm, silu, swiglu, geglu, rope_norm, rope_neox, causal_mask,
//! mul_elementwise, tanh_kernel, l2_normalize, embedding_lookup.

/// Complete MSL source containing all kernels needed by the inference engine:
/// f32 GEMM, activations, normalization, attention, pooling, quantized matmul,
/// and rotary position embeddings.
pub const MSL_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// =========================================================================
// Constants
// =========================================================================

// Simdgroup-matrix GEMM tiling constants (32x32 tiles, 4 simdgroups of 32)
constant constexpr uint BM = 32;
constant constexpr uint BN = 32;
constant constexpr uint BK = 32;

// Q8_0 block: 34 bytes per 32 values (2 bytes f16 scale + 32 bytes i8 quants)
constant constexpr uint QK8_0 = 32;
// Q4_0 block: 18 bytes per 32 values (2 bytes f16 scale + 16 bytes packed nibbles)
constant constexpr uint QK4_0 = 32;

// Quantized matmul tiling
constant constexpr short N_R0_Q8 = 2;   // rows per simdgroup for Q8_0
constant constexpr short N_SG_Q8 = 4;   // simdgroups per threadgroup for Q8_0 (128 threads)
constant constexpr short N_R0_Q4 = 4;   // rows per simdgroup for Q4_0
constant constexpr short N_SG_Q4 = 2;   // simdgroups per threadgroup for Q4_0 (64 threads)
constant constexpr short NW = 32;       // SIMD group width (Apple GPU)

// Q8_0 block struct (packed, 34 bytes)
struct block_q8_0 {
    half d;            // scale factor
    int8_t qs[QK8_0];  // quantized values
};

// Q4_0 block struct (packed, 18 bytes)
struct block_q4_0 {
    half d;                 // scale factor
    uint8_t qs[QK4_0 / 2]; // packed nibbles (2 values per byte)
};

// =========================================================================
// Ported f32 kernels from strata-core
// =========================================================================

// -----------------------------------------------------------------------
// gemm — simdgroup_matrix 32x32 tiled matrix multiply
//   C[M,N] = A[M,K] * B[K,N]     (both row-major)
//   128 threads (4 simdgroups) per threadgroup, each owns a 16x16 sub-tile.
// -----------------------------------------------------------------------
kernel void gemm(
    device const float* A       [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device       float* C       [[buffer(2)]],
    constant     uint&  M       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    constant     uint&  N       [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lid  [[thread_index_in_threadgroup]])
{
    uint tile_row = tgid.y * BM;
    uint tile_col = tgid.x * BN;

    // 2x2 simdgroup layout within the 32x32 tile
    uint sg_row = (sgid / 2) * 16;
    uint sg_col = (sgid % 2) * 16;

    // 2x2 grid of 8x8 accumulators per simdgroup = 16x16 sub-tile
    simdgroup_matrix<float, 8, 8> acc[2][2];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0);

    threadgroup float tgA[BM * BK];  // 32x32
    threadgroup float tgB[BK * BN];  // 32x32

    uint num_k_tiles = (K + BK - 1) / BK;

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_base = kt * BK;

        // Cooperative load: 128 threads load 32x32 = 1024 elements (8 each)
        for (uint idx = lid; idx < BM * BK; idx += 128) {
            uint r = idx / BK;
            uint c = idx % BK;
            uint gr = tile_row + r;
            uint gc = k_base + c;
            tgA[r * BK + c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint idx = lid; idx < BK * BN; idx += 128) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint gr = k_base + r;
            uint gc = tile_col + c;
            tgB[r * BN + c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner K-loop: 4 steps of 8 along BK=32
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<float, 8, 8> a_mat[2];
            simdgroup_matrix<float, 8, 8> b_mat[2];

            // Load 2 A sub-matrices (rows sg_row..sg_row+15, cols kk..kk+7)
            simdgroup_load(a_mat[0], &tgA[(sg_row + 0) * BK + kk], BK);
            simdgroup_load(a_mat[1], &tgA[(sg_row + 8) * BK + kk], BK);

            // Load 2 B sub-matrices (rows kk..kk+7, cols sg_col..sg_col+15)
            simdgroup_load(b_mat[0], &tgB[kk * BN + (sg_col + 0)], BN);
            simdgroup_load(b_mat[1], &tgB[kk * BN + (sg_col + 8)], BN);

            // 2x2 multiply-accumulate
            simdgroup_multiply_accumulate(acc[0][0], a_mat[0], b_mat[0], acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a_mat[0], b_mat[1], acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a_mat[1], b_mat[0], acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a_mat[1], b_mat[1], acc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results — threadgroup-level decision to avoid divergent barriers
    uint out_row = tile_row + sg_row;
    uint out_col = tile_col + sg_col;

    // Check if the entire 32x32 threadgroup tile is in-bounds
    if (tile_row + BM <= M && tile_col + BN <= N) {
        // Fast path: all simdgroups store directly to device memory
        simdgroup_store(acc[0][0], &C[(out_row + 0) * N + (out_col + 0)], N);
        simdgroup_store(acc[0][1], &C[(out_row + 0) * N + (out_col + 8)], N);
        simdgroup_store(acc[1][0], &C[(out_row + 8) * N + (out_col + 0)], N);
        simdgroup_store(acc[1][1], &C[(out_row + 8) * N + (out_col + 8)], N);
    } else {
        // Edge tile: all simdgroups store to staging, then bounds-checked write
        threadgroup float staging[BM * BN];
        uint base = sg_row * BN + sg_col;
        simdgroup_store(acc[0][0], &staging[base + 0 * BN + 0], BN);
        simdgroup_store(acc[0][1], &staging[base + 0 * BN + 8], BN);
        simdgroup_store(acc[1][0], &staging[base + 8 * BN + 0], BN);
        simdgroup_store(acc[1][1], &staging[base + 8 * BN + 8], BN);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = lid; idx < BM * BN; idx += 128) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint gr = tile_row + r;
            uint gc = tile_col + c;
            if (gr < M && gc < N) {
                C[gr * N + gc] = staging[r * BN + c];
            }
        }
    }
}

// -----------------------------------------------------------------------
// gemm_transpose — simdgroup_matrix GEMM where B is (N,K) accessed transposed
//   C[M,N] = A[M,K] * B^T   where B is stored as (N,K)
// -----------------------------------------------------------------------
kernel void gemm_transpose(
    device const float* A       [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device       float* C       [[buffer(2)]],
    constant     uint&  M       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    constant     uint&  N       [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lid  [[thread_index_in_threadgroup]])
{
    uint tile_row = tgid.y * BM;
    uint tile_col = tgid.x * BN;

    uint sg_row = (sgid / 2) * 16;
    uint sg_col = (sgid % 2) * 16;

    simdgroup_matrix<float, 8, 8> acc[2][2];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0);

    threadgroup float tgA[BM * BK];
    threadgroup float tgB[BK * BN];

    uint num_k_tiles = (K + BK - 1) / BK;

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_base = kt * BK;

        // Load A tile (same as gemm)
        for (uint idx = lid; idx < BM * BK; idx += 128) {
            uint r = idx / BK;
            uint c = idx % BK;
            uint gr = tile_row + r;
            uint gc = k_base + c;
            tgA[r * BK + c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        // Load B tile transposed: B is (N,K), we want B^T[k,n] = B[n,k]
        // tgB[r][c] corresponds to K-dim r, N-dim c
        for (uint idx = lid; idx < BK * BN; idx += 128) {
            uint r = idx / BN;  // K-dim offset
            uint c = idx % BN;  // N-dim offset
            uint gk = k_base + r;
            uint gn = tile_col + c;
            tgB[r * BN + c] = (gk < K && gn < N) ? B[gn * K + gk] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<float, 8, 8> a_mat[2];
            simdgroup_matrix<float, 8, 8> b_mat[2];

            simdgroup_load(a_mat[0], &tgA[(sg_row + 0) * BK + kk], BK);
            simdgroup_load(a_mat[1], &tgA[(sg_row + 8) * BK + kk], BK);

            simdgroup_load(b_mat[0], &tgB[kk * BN + (sg_col + 0)], BN);
            simdgroup_load(b_mat[1], &tgB[kk * BN + (sg_col + 8)], BN);

            simdgroup_multiply_accumulate(acc[0][0], a_mat[0], b_mat[0], acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a_mat[0], b_mat[1], acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a_mat[1], b_mat[0], acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a_mat[1], b_mat[1], acc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint out_row = tile_row + sg_row;
    uint out_col = tile_col + sg_col;

    if (tile_row + BM <= M && tile_col + BN <= N) {
        simdgroup_store(acc[0][0], &C[(out_row + 0) * N + (out_col + 0)], N);
        simdgroup_store(acc[0][1], &C[(out_row + 0) * N + (out_col + 8)], N);
        simdgroup_store(acc[1][0], &C[(out_row + 8) * N + (out_col + 0)], N);
        simdgroup_store(acc[1][1], &C[(out_row + 8) * N + (out_col + 8)], N);
    } else {
        threadgroup float staging[BM * BN];
        uint base = sg_row * BN + sg_col;
        simdgroup_store(acc[0][0], &staging[base + 0 * BN + 0], BN);
        simdgroup_store(acc[0][1], &staging[base + 0 * BN + 8], BN);
        simdgroup_store(acc[1][0], &staging[base + 8 * BN + 0], BN);
        simdgroup_store(acc[1][1], &staging[base + 8 * BN + 8], BN);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = lid; idx < BM * BN; idx += 128) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint gr = tile_row + r;
            uint gc = tile_col + c;
            if (gr < M && gc < N) {
                C[gr * N + gc] = staging[r * BN + c];
            }
        }
    }
}

// -----------------------------------------------------------------------
// gelu — element-wise exact GELU using erf approximation
//   y = x * 0.5 * (1 + erf(x / sqrt(2)))
// erf approximation: Abramowitz & Stegun formula 7.1.26 (max error ~1.5e-7)
// Same polynomial used by llama.cpp Metal shaders.
// -----------------------------------------------------------------------
kernel void gelu(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  count  [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    float x = input[tid];
    const float SQRT_2_INV = 0.7071067811865475f;
    float arg = x * SQRT_2_INV;
    // Abramowitz & Stegun erf approximation
    float sign_arg = sign(arg);
    float abs_arg = fabs(arg);
    float t = 1.0f / (1.0f + 0.3275911f * abs_arg);
    float y = 1.0f - (((((1.061405429f * t + -1.453152027f) * t) + 1.421413741f) * t + -0.284496736f) * t + 0.254829592f) * t * exp(-abs_arg * abs_arg);
    float erf_val = sign_arg * y;
    output[tid] = 0.5f * x * (1.0f + erf_val);
}

// -----------------------------------------------------------------------
// add_tensor — element-wise addition  c[i] = a[i] + b[i]
// -----------------------------------------------------------------------
kernel void add_tensor(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device       float* c      [[buffer(2)]],
    constant     uint&  count  [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    c[tid] = a[tid] + b[tid];
}

// -----------------------------------------------------------------------
// add_bias — broadcast row-add:  t[r*cols+c] += bias[c]
// -----------------------------------------------------------------------
kernel void add_bias(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    constant     uint&  rows   [[buffer(3)]],
    constant     uint&  cols   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.y;
    uint c = gid.x;
    if (r >= rows || c >= cols) return;
    output[r * cols + c] = input[r * cols + c] + bias[c];
}

// -----------------------------------------------------------------------
// scale_kernel — out-of-place scalar multiply  output[i] = input[i] * factor
// -----------------------------------------------------------------------
kernel void scale_kernel(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     float& factor [[buffer(2)]],
    constant     uint&  count  [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    output[tid] = input[tid] * factor;
}

// -----------------------------------------------------------------------
// layer_norm — per-row layer normalization
//   out[r,c] = (x[r,c] - mean_r) / sqrt(var_r + eps) * w[c] + b[c]
//   One threadgroup per row; shared-memory reduction for mean and variance.
// -----------------------------------------------------------------------
kernel void layer_norm(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device       float* output  [[buffer(3)]],
    constant     uint&  rows    [[buffer(4)]],
    constant     uint&  cols    [[buffer(5)]],
    constant     float& eps     [[buffer(6)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint row = gid;
    if (row >= rows) return;

    threadgroup float shared_data[256];

    // --- Compute mean ---
    float partial_sum = 0.0f;
    for (uint c = lid; c < cols; c += threads_per_group) {
        partial_sum += input[row * cols + c];
    }
    shared_data[lid] = partial_sum;

    // Tree reduction
    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < stride) {
            shared_data[lid] += shared_data[lid + stride];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = shared_data[0] / float(cols);

    // --- Compute variance ---
    float partial_var = 0.0f;
    for (uint c = lid; c < cols; c += threads_per_group) {
        float diff = input[row * cols + c] - mean;
        partial_var += diff * diff;
    }
    shared_data[lid] = partial_var;

    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < stride) {
            shared_data[lid] += shared_data[lid + stride];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = shared_data[0] / float(cols);
    float inv_std = 1.0f / sqrt(var + eps);

    // --- Normalize ---
    for (uint c = lid; c < cols; c += threads_per_group) {
        uint idx = row * cols + c;
        output[idx] = (input[idx] - mean) * inv_std * weight[c] + bias[c];
    }
}

// -----------------------------------------------------------------------
// softmax_rows — per-row softmax with max subtraction for stability
//   One threadgroup per row; shared-memory reductions for max and sum.
// -----------------------------------------------------------------------
kernel void softmax_rows(
    device const float* input   [[buffer(0)]],
    device       float* output  [[buffer(1)]],
    constant     uint&  rows    [[buffer(2)]],
    constant     uint&  cols    [[buffer(3)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint row = gid;
    if (row >= rows) return;

    threadgroup float shared[256];

    // --- Find row max ---
    float local_max = -INFINITY;
    for (uint c = lid; c < cols; c += threads_per_group) {
        local_max = max(local_max, input[row * cols + c]);
    }
    shared[lid] = local_max;

    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < stride) {
            shared[lid] = max(shared[lid], shared[lid + stride]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = shared[0];

    // --- Compute exp(x - max) and partial sum ---
    float local_sum = 0.0f;
    for (uint c = lid; c < cols; c += threads_per_group) {
        uint idx = row * cols + c;
        float val = exp(input[idx] - row_max);
        output[idx] = val;
        local_sum += val;
    }
    shared[lid] = local_sum;

    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = shared[0];

    // --- Normalize ---
    if (total > 0.0f) {
        float inv_total = 1.0f / total;
        for (uint c = lid; c < cols; c += threads_per_group) {
            output[row * cols + c] *= inv_total;
        }
    }
}

// -----------------------------------------------------------------------
// mean_pool — sum masked rows, divide by count
//   output[c] = sum_over_r(mask[r] ? hidden[r,c] : 0) / count
//   One thread per column.
// -----------------------------------------------------------------------
kernel void mean_pool(
    device const float* hidden  [[buffer(0)]],
    device const uint*  mask    [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  rows    [[buffer(3)]],
    constant     uint&  cols    [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= cols) return;

    float sum = 0.0f;
    float count = 0.0f;
    for (uint r = 0; r < rows; ++r) {
        if (mask[r] == 1u) {
            sum += hidden[r * cols + tid];
            count += 1.0f;
        }
    }
    output[tid] = (count > 0.0f) ? (sum / count) : 0.0f;
}

// =========================================================================
// New kernels for quantized inference
// =========================================================================

// -----------------------------------------------------------------------
// 1. quantized_matmul_q8_0 — Fused Q8_0 dequant + matrix-vector multiply
//
//   Computes output[row] = dot(dequant(weights_row), input) for N rows.
//   weights: Q8_0 quantized (N rows x K cols), input: f32 (K), output: f32 (N).
//
//   Following llama.cpp's approach:
//   - N_R0=2 rows per simdgroup, N_SG=4 simdgroups → 128 threads
//   - Factor scale `d` out of inner loop: compute integer dot sum(qs[j]*y[j])
//     first, multiply by float(d) once at end.
//   - Use simd_sum() for warp-level reduction across the simdgroup.
//
//   Grid: (N + N_R0*N_SG - 1) / (N_R0*N_SG) threadgroups, 128 threads each.
//   Buffer layout: weights is raw bytes, interpreted as block_q8_0 array.
// -----------------------------------------------------------------------
kernel void quantized_matmul_q8_0(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  N       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // Number of Q8_0 blocks per row
    const uint nb = K / QK8_0;

    // Each simdgroup handles N_R0_Q8 consecutive rows
    const uint r0 = (tgpig.x * N_SG_Q8 + sgitg) * N_R0_Q8;

    // Each thread in the simdgroup (32 threads) processes a strided subset
    // of blocks. NQ=8 quants at a time per thread, stepping through blocks.
    constexpr short NQ = 8;

    // Pointers to the rows of quantized weights
    device const block_q8_0 * ax[N_R0_Q8];
    for (short row = 0; row < N_R0_Q8; ++row) {
        // Each row has nb blocks, each block is 34 bytes
        ax[row] = (device const block_q8_0 *)(weights + (r0 + row) * nb * 34);
    }

    float sumf[N_R0_Q8] = { 0.0f };

    // Divide work: 32 threads split into sub-blocks
    // ix = which sub-block within the simdgroup's NQ-block chunk
    // il = element offset within the block
    const short ix = tiisg / (NW / NQ);  // 0..NQ-1 (which group of 4 threads)
    const short il = tiisg % (NW / NQ);  // 0..3 (which sub-chunk within group)

    const int ib0 = ix;

    // Cache for input vector elements
    float yl[NQ];

    device const float * yb = input + ib0 * QK8_0 + il * NQ;

    // Process all blocks for this row — each simdgroup independently
    // covers ALL blocks for its assigned rows (stride = NQ blocks).
    for (int ib = ib0; ib < (int)nb; ib += NQ) {
        // Cache input values
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        // Compute partial dot product for each row
        for (short row = 0; row < N_R0_Q8; ++row) {
            device const int8_t * qs = ax[row][ib].qs + il * NQ;

            float sumq = 0.0f;
            for (short i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }

            // Factor out scale: multiply by d once
            sumf[row] += sumq * float(ax[row][ib].d);
        }

        yb += NQ * QK8_0;
    }

    // Reduce across simdgroup using simd_sum
    for (short row = 0; row < N_R0_Q8; ++row) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && (r0 + row) < N) {
            output[r0 + row] = tot;
        }
    }
}

// -----------------------------------------------------------------------
// 2. quantized_matmul_q4_0 — Fused Q4_0 dequant + matrix-vector multiply
//
//   Q4_0 block format: 18 bytes per 32 values.
//   - 2 bytes f16 scale (d)
//   - 16 bytes packed nibbles: byte[i] holds quants[i] (low nibble) and
//     quants[i+16] (high nibble).
//   Extract nibbles: low = (byte & 0x0F) - 8, high = (byte >> 4) - 8
//   Dequant: value = d * (nibble - 8)
//
//   N_R0=4, N_SG=2 → 64 threads per threadgroup.
//   Grid: (N + N_R0*N_SG - 1) / (N_R0*N_SG) threadgroups, 64 threads each.
// -----------------------------------------------------------------------
kernel void quantized_matmul_q4_0(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  N       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = K / QK4_0;

    // Each simdgroup handles N_R0_Q4 consecutive rows
    const uint r0 = (tgpig.x * N_SG_Q4 + sgitg) * N_R0_Q4;

    // Pointers to the rows of quantized weights
    device const block_q4_0 * ax[N_R0_Q4];
    for (short row = 0; row < N_R0_Q4; ++row) {
        // Each row has nb blocks, each block is 18 bytes
        ax[row] = (device const block_q4_0 *)(weights + (r0 + row) * nb * 18);
    }

    float sumf[N_R0_Q4] = { 0.0f };

    // Each thread processes one block at a time, half the quants per block
    // 32 threads handle 16 blocks at once (stride = 16 blocks)
    constexpr short NQ = 16;
    const short ix = tiisg / (NW / NQ);  // which block within a chunk
    const short il = (tiisg % (NW / NQ)) * 8;  // element offset (0 or 8)

    const int ib0 = ix;
    device const float * yb = input + ib0 * QK4_0 + il;

    // Each thread handles 16 elements (8 low nibbles + 8 high nibbles at offset)
    for (int ib = ib0; ib < (int)nb; ib += NQ) {
        // Load 8 input elements for low nibbles and 8 for high nibbles
        float yl[16];
        float sumy0 = 0.0f;
        float sumy1 = 0.0f;

        for (short i = 0; i < 8; ++i) {
            yl[i]     = yb[i];       // maps to low nibble elements
            yl[i + 8] = yb[i + 16];  // maps to high nibble elements
            sumy0 += yl[i];
            sumy1 += yl[i + 8];
        }

        for (short row = 0; row < N_R0_Q4; ++row) {
            device const uint8_t * qs = ax[row][ib].qs + il / 2;

            float sumq = 0.0f;
            for (short i = 0; i < 8; ++i) {
                uint8_t qbyte = qs[i];
                // Low nibble: quants[il+i], subtract 8 to center
                float q_lo = float(qbyte & 0x0F);
                // High nibble: quants[il+i+16], subtract 8 to center
                float q_hi = float(qbyte >> 4);

                sumq += q_lo * yl[i] + q_hi * yl[i + 8];
            }

            // Subtract bias from centering: sum_of_all_y * 8
            sumf[row] += float(ax[row][ib].d) * (sumq - 8.0f * (sumy0 + sumy1));
        }

        yb += NQ * QK4_0;
    }

    // Reduce across simdgroup
    for (short row = 0; row < N_R0_Q4; ++row) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && (r0 + row) < N) {
            output[r0 + row] = tot;
        }
    }
}

// -----------------------------------------------------------------------
// 3. rms_norm — Per-row RMS normalization with fused weight multiply
//   out[r,c] = x[r,c] * rsqrt(mean(x^2) + eps) * w[c]
//   One threadgroup per row, 256 threads. Shared-memory reduction for
//   sum-of-squares.
// -----------------------------------------------------------------------
kernel void rms_norm(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  rows    [[buffer(3)]],
    constant     uint&  cols    [[buffer(4)]],
    constant     float& eps     [[buffer(5)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint row = gid;
    if (row >= rows) return;

    threadgroup float shared_data[256];

    // --- Compute sum of squares ---
    float partial_ss = 0.0f;
    for (uint c = lid; c < cols; c += threads_per_group) {
        float val = input[row * cols + c];
        partial_ss += val * val;
    }
    shared_data[lid] = partial_ss;

    // Tree reduction
    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < stride) {
            shared_data[lid] += shared_data[lid + stride];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_scale = rsqrt(shared_data[0] / float(cols) + eps);

    // --- Normalize and apply weight ---
    for (uint c = lid; c < cols; c += threads_per_group) {
        uint idx = row * cols + c;
        output[idx] = input[idx] * rms_scale * weight[c];
    }
}

// -----------------------------------------------------------------------
// 4. silu — Element-wise SiLU activation
//   y = x * sigmoid(x) = x / (1 + exp(-x))
//   256 threads per threadgroup.
// -----------------------------------------------------------------------
kernel void silu(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  count  [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    float x = input[tid];
    output[tid] = x / (1.0f + exp(-x));
}

// -----------------------------------------------------------------------
// 5. swiglu — Fused SwiGLU activation (two inputs)
//   y[i] = silu(gate[i]) * up[i]
//   where silu(x) = x / (1 + exp(-x))
//   256 threads per threadgroup.
// -----------------------------------------------------------------------
kernel void swiglu(
    device const float* gate   [[buffer(0)]],
    device const float* up     [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant     uint&  count  [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));
    output[tid] = silu_g * up[tid];
}

// -----------------------------------------------------------------------
// 6. geglu — Fused GeGLU activation (two inputs)
//   y[i] = gelu(gate[i]) * up[i]
//   where gelu uses the fast tanh approximation.
//   256 threads per threadgroup.
// -----------------------------------------------------------------------
kernel void geglu(
    device const float* gate   [[buffer(0)]],
    device const float* up     [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant     uint&  count  [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    float g = gate[tid];
    const float SQRT_2_OVER_PI = 0.7978845608f;
    float inner = SQRT_2_OVER_PI * (g + 0.044715f * g * g * g);
    inner = clamp(inner, -10.0f, 10.0f);
    float gelu_g = 0.5f * g * (1.0f + metal::tanh(inner));
    output[tid] = gelu_g * up[tid];
}

// -----------------------------------------------------------------------
// 7. rope_norm — Rotary position embeddings (interleaved/normal pairing)
//
//   Pairs consecutive elements: (x[2i], x[2i+1])
//   For each pair, apply rotation:
//     out[2i]   = x[2i]   * cos(theta) - x[2i+1] * sin(theta)
//     out[2i+1] = x[2i]   * sin(theta) + x[2i+1] * cos(theta)
//   where theta = (pos_offset + seq_pos) / (freq_base ^ (2i / rope_dim))
//
//   Input layout: [seq_len, n_heads, head_dim]
//   Only the first rope_dim elements of each head are rotated.
//   One thread per pair.
// -----------------------------------------------------------------------
kernel void rope_norm(
    device const float* input      [[buffer(0)]],
    device       float* output     [[buffer(1)]],
    constant     uint&  pos_offset [[buffer(2)]],
    constant     float& freq_base  [[buffer(3)]],
    constant     uint&  head_dim   [[buffer(4)]],
    constant     uint&  rope_dim   [[buffer(5)]],
    constant     uint&  n_heads    [[buffer(6)]],
    constant     uint&  seq_len    [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    // gid.x = pair index within head (0 .. rope_dim/2 - 1)
    // gid.y = head index (0 .. n_heads - 1)
    // gid.z = sequence position (0 .. seq_len - 1)
    uint pair = gid.x;
    uint head = gid.y;
    uint seq  = gid.z;

    if (pair >= rope_dim / 2 || head >= n_heads || seq >= seq_len) return;

    // Compute angle: theta = pos / (freq_base ^ (2*pair / rope_dim))
    float pos = float(pos_offset + seq);
    float freq_exp = float(2 * pair) / float(rope_dim);
    float theta = pos / pow(freq_base, freq_exp);

    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    // Index into the flattened [seq_len, n_heads, head_dim] tensor
    uint base_idx = seq * n_heads * head_dim + head * head_dim;
    uint idx0 = base_idx + 2 * pair;
    uint idx1 = base_idx + 2 * pair + 1;

    float x0 = input[idx0];
    float x1 = input[idx1];

    output[idx0] = x0 * cos_theta - x1 * sin_theta;
    output[idx1] = x0 * sin_theta + x1 * cos_theta;

    // Copy non-rotated dimensions (rope_dim .. head_dim) for this head.
    // Each pair-thread copies one non-rotated element to avoid redundant work.
    uint non_rot_idx = rope_dim + pair;
    if (non_rot_idx < head_dim) {
        uint src = base_idx + non_rot_idx;
        output[src] = input[src];
    }
}

// -----------------------------------------------------------------------
// 8. rope_neox — Rotary position embeddings (half-split/GPT-NeoX pairing)
//
//   Pairs elements at offset n_dims/2: (x[i], x[i + rope_dim/2])
//   For each pair, apply rotation:
//     out[i]              = x[i]              * cos(theta) - x[i+rope_dim/2] * sin(theta)
//     out[i + rope_dim/2] = x[i]              * sin(theta) + x[i+rope_dim/2] * cos(theta)
//   where theta = (pos_offset + seq_pos) / (freq_base ^ (2i / rope_dim))
//
//   Input layout: [seq_len, n_heads, head_dim]
//   Only the first rope_dim elements of each head are rotated.
// -----------------------------------------------------------------------
kernel void rope_neox(
    device const float* input      [[buffer(0)]],
    device       float* output     [[buffer(1)]],
    constant     uint&  pos_offset [[buffer(2)]],
    constant     float& freq_base  [[buffer(3)]],
    constant     uint&  head_dim   [[buffer(4)]],
    constant     uint&  rope_dim   [[buffer(5)]],
    constant     uint&  n_heads    [[buffer(6)]],
    constant     uint&  seq_len    [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    // gid.x = element index within first half of rope_dim (0 .. rope_dim/2 - 1)
    // gid.y = head index (0 .. n_heads - 1)
    // gid.z = sequence position (0 .. seq_len - 1)
    uint i    = gid.x;
    uint head = gid.y;
    uint seq  = gid.z;

    uint half_rope = rope_dim / 2;

    if (i >= half_rope || head >= n_heads || seq >= seq_len) return;

    // Compute angle: theta = pos / (freq_base ^ (2*i / rope_dim))
    float pos = float(pos_offset + seq);
    float freq_exp = float(2 * i) / float(rope_dim);
    float theta = pos / pow(freq_base, freq_exp);

    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    // Index into the flattened [seq_len, n_heads, head_dim] tensor
    uint base_idx = seq * n_heads * head_dim + head * head_dim;
    uint idx_lo = base_idx + i;
    uint idx_hi = base_idx + i + half_rope;

    float x_lo = input[idx_lo];
    float x_hi = input[idx_hi];

    output[idx_lo] = x_lo * cos_theta - x_hi * sin_theta;
    output[idx_hi] = x_lo * sin_theta + x_hi * cos_theta;

    // Copy non-rotated dimensions (rope_dim .. head_dim) for this head.
    // Each thread copies one non-rotated element to avoid redundant work.
    uint non_rot_idx = rope_dim + i;
    if (non_rot_idx < head_dim) {
        uint src = base_idx + non_rot_idx;
        output[src] = input[src];
    }
}

// -----------------------------------------------------------------------
// 9. causal_mask — Set attention scores to -INFINITY for future positions
//   scores[i][j] = -INFINITY where j > i + offset
//   2D grid: (cols, rows). offset allows shifting for KV-cache decoding.
// -----------------------------------------------------------------------
kernel void causal_mask(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  rows   [[buffer(2)]],
    constant     uint&  cols   [[buffer(3)]],
    constant     uint&  offset [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.y;
    uint c = gid.x;
    if (r >= rows || c >= cols) return;
    uint idx = r * cols + c;
    // Mask future tokens: j > i + offset means this position cannot attend
    output[idx] = (c > r + offset) ? -INFINITY : input[idx];
}

// -----------------------------------------------------------------------
// 10. mul_elementwise — Element-wise multiplication with broadcast
//   c[i] = a[i] * b[i]
//   Supports broadcast for [M,N]*[N]: b uses b[tid % cols].
// -----------------------------------------------------------------------
kernel void mul_elementwise(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device       float* c      [[buffer(2)]],
    constant     uint&  count  [[buffer(3)]],
    constant     uint&  b_len  [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    // If b_len == count, no broadcast. If b_len < count, broadcast b.
    c[tid] = a[tid] * b[tid % b_len];
}

// -----------------------------------------------------------------------
// 11. tanh_kernel — Element-wise tanh
// -----------------------------------------------------------------------
kernel void tanh_kernel(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  count  [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    output[tid] = metal::tanh(input[tid]);
}

// -----------------------------------------------------------------------
// 12. l2_normalize — L2 normalization of a vector
//   output[i] = input[i] / sqrt(sum(input[j]^2))
//   One threadgroup processes the entire vector. Two-pass:
//   (1) reduction for squared sum, (2) divide each element.
//   256 threads per threadgroup.
// -----------------------------------------------------------------------
kernel void l2_normalize(
    device const float* input   [[buffer(0)]],
    device       float* output  [[buffer(1)]],
    constant     uint&  count   [[buffer(2)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    // Only one threadgroup expected (gid == 0)
    if (gid != 0) return;

    threadgroup float shared_data[256];

    // Pass 1: compute partial sum of squares
    float partial_ss = 0.0f;
    for (uint i = lid; i < count; i += threads_per_group) {
        float val = input[i];
        partial_ss += val * val;
    }
    shared_data[lid] = partial_ss;

    // Tree reduction
    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < stride) {
            shared_data[lid] += shared_data[lid + stride];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_norm = rsqrt(shared_data[0] + 1e-12f);

    // Pass 2: normalize each element
    for (uint i = lid; i < count; i += threads_per_group) {
        output[i] = input[i] * inv_norm;
    }
}

// -----------------------------------------------------------------------
// 13. embedding_lookup — Gather rows by token IDs from f32 embedding table
//   output[i * hidden + j] = table[ids[i] * hidden + j]
//   2D grid: (hidden, num_tokens)
// -----------------------------------------------------------------------
kernel void embedding_lookup(
    device const float* table   [[buffer(0)]],
    device const uint*  ids     [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  hidden  [[buffer(3)]],
    constant     uint&  num_tokens [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint j = gid.x;  // hidden dimension index
    uint i = gid.y;  // token index

    if (i >= num_tokens || j >= hidden) return;

    uint token_id = ids[i];
    output[i * hidden + j] = table[token_id * hidden + j];
}
"#;
