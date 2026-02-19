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

    // Compute angle matching llama.cpp's formula:
    //   theta = pos * pow(freq_base, -1.0/rope_dim * i0)
    // where i0 = 2*pair. Using negative exponent + multiply instead of
    // positive exponent + divide matches llama.cpp's exact float rounding.
    float theta_base = float(pos_offset + seq);
    float inv_ndims = -1.f / float(rope_dim);
    float theta = theta_base * pow(freq_base, inv_ndims * float(2 * pair));

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
    // Distribute across pair-threads with strided loop.
    uint half_rope = rope_dim / 2;
    for (uint nr = pair; nr < head_dim - rope_dim; nr += half_rope) {
        uint src = base_idx + rope_dim + nr;
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
//   where theta = pos * pow(freq_base, -1.0/rope_dim * 2*i)
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

    // Compute angle matching llama.cpp's formula:
    //   theta = pos * pow(freq_base, -1.0/rope_dim * i0)
    float theta_base = float(pos_offset + seq);
    float inv_ndims = -1.f / float(rope_dim);
    float theta = theta_base * pow(freq_base, inv_ndims * float(2 * i));

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
    // Distribute across threads with strided loop.
    for (uint nr = i; nr < head_dim - rope_dim; nr += half_rope) {
        uint src = base_idx + rope_dim + nr;
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

// -----------------------------------------------------------------------
// 14. grouped_attn_decode — Fused grouped-query attention for single-token decode
//
//   Computes attention for all Q heads in a single kernel dispatch.
//   One threadgroup per Q head, 256 threads each.
//
//   Q: [1, num_heads * head_dim]       (single query token)
//   K: [max_len, num_kv_heads * head_dim]  (KV cache, first total_len rows valid)
//   V: [max_len, num_kv_heads * head_dim]  (KV cache, first total_len rows valid)
//   output: [1, num_heads * head_dim]
//
//   For GQA, kv_head = h * num_kv_heads / num_heads.
//
//   Two-pass tiled approach:
//   Pass 1: threads parallel over positions → compute Q@K scores, find global max
//   Pass 2: process positions in tiles of 256. Within each tile:
//     - All threads compute one score each → store exp(score-max) as prob in shared
//     - Threads switch to parallel over head_dim dimensions, accumulate prob*V
//     - Also accumulate sum_exp for final normalization
//   Final: normalize V accumulation by 1/sum_exp
// -----------------------------------------------------------------------
kernel void grouped_attn_decode(
    device const float* Q       [[buffer(0)]],
    device const float* K       [[buffer(1)]],
    device const float* V       [[buffer(2)]],
    device       float* output  [[buffer(3)]],
    constant     uint&  num_heads    [[buffer(4)]],
    constant     uint&  num_kv_heads [[buffer(5)]],
    constant     uint&  head_dim     [[buffer(6)]],
    constant     uint&  total_len    [[buffer(7)]],
    constant     float& attn_scale   [[buffer(8)]],
    constant     float& softcap      [[buffer(9)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]])
{
    uint h = gid;
    if (h >= num_heads) return;

    uint kv_dim = num_kv_heads * head_dim;
    uint kv_head = h * num_kv_heads / num_heads;
    device const float* q_head = Q + h * head_dim;
    uint kv_off = kv_head * head_dim;

    threadgroup float shared[256];

    // ---- Pass 1: find global max of attention scores ----
    // Threads parallel over positions
    float local_max = -INFINITY;
    for (uint j = lid; j < total_len; j += tpg) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            dot += q_head[d] * K[j * kv_dim + kv_off + d];
        }
        float s = dot * attn_scale;
        if (softcap > 0.0f) {
            s = softcap * precise::tanh(s / softcap);
        }
        local_max = max(local_max, s);
    }

    // Reduce max across threadgroup
    shared[lid] = local_max;
    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < stride) {
            shared[lid] = max(shared[lid], shared[lid + stride]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = shared[0];

    // ---- Pass 2: tiled score + V accumulation ----
    // Process positions in tiles of tpg. Within each tile:
    //   1. Each thread computes score for one position → store weight in shared
    //   2. Threads switch to parallel over head_dim, accumulate weight * V
    //   3. Each thread also accumulates partial sum_exp
    device float* out_head = output + h * head_dim;

    // Each thread owns one output dimension (lid < head_dim) and accumulates in register
    float v_acc = 0.0f;
    float local_sum_exp = 0.0f;

    for (uint tile = 0; tile < total_len; tile += tpg) {
        uint j = tile + lid;

        // Step A: each thread computes exp(score - max) for one position
        float w = 0.0f;
        if (j < total_len) {
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += q_head[d] * K[j * kv_dim + kv_off + d];
            }
            float s = dot * attn_scale;
            if (softcap > 0.0f) {
                s = softcap * precise::tanh(s / softcap);
            }
            w = exp(s - gmax);
        }
        local_sum_exp += w;
        shared[lid] = w;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step B: threads parallel over dimensions, accumulate weight * V
        uint active = min(tpg, total_len - tile);
        if (lid < head_dim) {
            for (uint t = 0; t < active; ++t) {
                v_acc += shared[t] * V[(tile + t) * kv_dim + kv_off + lid];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reduce sum_exp across threadgroup
    shared[lid] = local_sum_exp;
    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = (shared[0] > 0.0f) ? (1.0f / shared[0]) : 0.0f;

    // Write normalized output
    if (lid < head_dim) {
        out_head[lid] = v_acc * inv_sum;
    }
}

// =========================================================================
// K-quant kernels: Q4_K, Q5_K, Q6_K fused dequant + matrix-vector multiply
// =========================================================================

// K-quant constants
constant constexpr uint QK_K = 256;
constant constexpr uint Q4_K_BLOCK_SIZE = 144;
constant constexpr uint Q5_K_BLOCK_SIZE = 176;
constant constexpr uint Q6_K_BLOCK_SIZE = 210;

// Tiling constants for K-quant kernels (same for Q4_K, Q5_K, Q6_K)
constant constexpr short N_R0_KQ = 2;  // rows per simdgroup
constant constexpr short N_SG_KQ = 2;  // simdgroups per threadgroup (64 threads)

// -----------------------------------------------------------------------
// 15. quantized_matmul_q4_k — Fused Q4_K dequant + matrix-vector multiply
//
//   Q4_K block (144 bytes per 256 values):
//     d:      f16  (2B)   super-block scale
//     dmin:   f16  (2B)   super-block min scale
//     scales: u8[12]      8 pairs of 6-bit (scale, min) packed
//     qs:     u8[128]     4-bit quantized values
//
//   Dequant: y = d * scale_j * (q & 0xF) - dmin * min_j
//   8 sub-blocks of 32 values each.
//
//   Ported from llama.cpp's kernel_mul_mv_q4_K_f32_impl.
//   Grid: (N + N_R0_KQ*N_SG_KQ - 1) / (N_R0_KQ*N_SG_KQ) threadgroups, 64 threads.
// -----------------------------------------------------------------------
kernel void quantized_matmul_q4_k(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  N       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = K / QK_K;
    const uint first_row = (tgpig.x * N_SG_KQ + sgitg) * N_R0_KQ;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    const short ix = tiisg / 8;   // 0..3: which block chunk
    const short it = tiisg % 8;
    const short iq = it / 4;      // 0 or 1: first or second 128 values
    const short ir = it % 4;      // 0..3: which 8-element group

    float sumf[N_R0_KQ] = {0.f};

    device const float * y4 = input + ix * QK_K + 64 * iq + 8 * ir;

    ushort sc16[4];
    thread const uchar * sc8 = (thread const uchar *)sc16;

    float yl[16];
    float yh[16];

    for (int ib = ix; ib < (int)nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        for (short row = 0; row < N_R0_KQ; ++row) {
            uint actual_row = first_row + row;
            if (actual_row >= N) continue;

            device const uchar*  bp = (device const uchar*)(weights + actual_row * nb * Q4_K_BLOCK_SIZE + ib * Q4_K_BLOCK_SIZE);
            device const half*   dh = (device const half*)bp;
            device const ushort* sc = (device const ushort*)(bp + 4) + iq;
            device const ushort* q1 = (device const ushort*)(bp + 16) + 16 * iq + 4 * ir;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const ushort * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += float(dh[0]) * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                          (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         float(dh[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y4 += 4 * QK_K;
    }

    for (short row = 0; row < N_R0_KQ; ++row) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && (first_row + row) < N) {
            output[first_row + row] = tot;
        }
    }
}

// -----------------------------------------------------------------------
// 16. quantized_matmul_q5_k — Fused Q5_K dequant + matrix-vector multiply
//
//   Q5_K block (176 bytes per 256 values):
//     d:      f16  (2B), dmin: f16 (2B), scales: u8[12], qh: u8[32], qs: u8[128]
//   Dequant: y = d * scale * ((q & 0xF) + high_bit*16) - dmin * min
//   32 threads × 8 elements = 256 values per block.
// -----------------------------------------------------------------------
kernel void quantized_matmul_q5_k(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  N       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = K / QK_K;
    const uint first_row = (tgpig.x * N_SG_KQ + sgitg) * N_R0_KQ;

    const short elem_offset = tiisg * 8;
    const short sb   = elem_offset / 32;
    const short pos  = elem_offset % 32;
    const short pair = sb / 2;
    const bool  is_high = (sb & 1);
    const uchar qh_bit  = uchar(1 << sb);

    float sumf[N_R0_KQ] = {0.f};

    for (int ib = 0; ib < (int)nb; ib++) {
        device const float* y = input + ib * QK_K + elem_offset;
        float yl[8];
        for (short i = 0; i < 8; i++) yl[i] = y[i];

        for (short row = 0; row < N_R0_KQ; ++row) {
            uint actual_row = first_row + row;
            if (actual_row >= N) continue;

            device const uchar* bp = (device const uchar*)(weights + actual_row * nb * Q5_K_BLOCK_SIZE + ib * Q5_K_BLOCK_SIZE);
            float d_val    = float(*(device const half*)(bp));
            float dmin_val = float(*(device const half*)(bp + 2));
            device const uchar* sa = bp + 4;
            device const uchar* qh_arr = bp + 16;
            device const uchar* qs_arr = bp + 48;

            uchar sc_val, m_val;
            if (sb < 4) {
                sc_val = sa[sb] & 63;
                m_val  = sa[sb + 4] & 63;
            } else {
                sc_val = (sa[sb + 4] & 0xF) | ((sa[sb - 4] >> 6) << 4);
                m_val  = (sa[sb + 4] >> 4)  | ((sa[sb]     >> 6) << 4);
            }
            float sc_f = d_val * float(sc_val);
            float m_f  = dmin_val * float(m_val);

            float partial = 0.f;
            for (short i = 0; i < 8; i++) {
                short l = pos + i;
                uchar q_low  = is_high ? (qs_arr[pair * 32 + l] >> 4)
                                       : (qs_arr[pair * 32 + l] & 0xF);
                uchar q_high = (qh_arr[l] & qh_bit) ? 16 : 0;
                partial += (sc_f * float(q_low + q_high) - m_f) * yl[i];
            }
            sumf[row] += partial;
        }
    }

    for (short row = 0; row < N_R0_KQ; ++row) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && (first_row + row) < N) {
            output[first_row + row] = tot;
        }
    }
}

// -----------------------------------------------------------------------
// 17. quantized_matmul_q6_k — Fused Q6_K dequant + matrix-vector multiply
//
//   Q6_K block (210 bytes per 256 values):
//     ql: u8[128], qh: u8[64], scales: i8[16], d: f16 (2B)
//   Dequant: y = d * scale * (6bit_val - 32)
//   32 threads × 8 elements = 256 values per block.
//   Layout: 2 halves of 128; each half = 4 groups of 32 values.
// -----------------------------------------------------------------------
kernel void quantized_matmul_q6_k(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  N       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = K / QK_K;
    const uint first_row = (tgpig.x * N_SG_KQ + sgitg) * N_R0_KQ;

    const short elem_offset = tiisg * 8;
    const short half_idx    = elem_offset / 128;
    const short within_half = elem_offset % 128;
    const short group32     = within_half / 32;
    const short pos         = within_half % 32;

    const uint  ql_off   = half_idx * 64 + ((group32 & 1) ? 32 : 0);
    const uint  qh_off   = half_idx * 32;
    const short qh_shift = group32 * 2;
    const bool  high_nib = (group32 >= 2);
    const short sc_off   = half_idx * 8 + group32 * 2;

    float sumf[N_R0_KQ] = {0.f};

    for (int ib = 0; ib < (int)nb; ib++) {
        device const float* y = input + ib * QK_K + elem_offset;
        float yl[8];
        for (short i = 0; i < 8; i++) yl[i] = y[i];

        for (short row = 0; row < N_R0_KQ; ++row) {
            uint actual_row = first_row + row;
            if (actual_row >= N) continue;

            device const uchar* bp = (device const uchar*)(weights + actual_row * nb * Q6_K_BLOCK_SIZE + ib * Q6_K_BLOCK_SIZE);
            device const uchar* ql = bp;
            device const uchar* qh = bp + 128;
            device const char*  sc = (device const char*)(bp + 192);
            float d = float(*(device const half*)(bp + 208));

            float partial = 0.f;
            for (short i = 0; i < 8; i++) {
                short l = pos + i;
                uchar ql_byte = ql[ql_off + l];
                uchar qh_byte = qh[qh_off + l];
                uchar q_low   = high_nib ? (ql_byte >> 4) : (ql_byte & 0xF);
                uchar q_high  = (qh_byte >> qh_shift) & 3;
                int   q_val   = int(q_low | (q_high << 4)) - 32;
                char  scale   = sc[sc_off + (l / 16)];
                partial += d * float(scale) * float(q_val) * yl[i];
            }
            sumf[row] += partial;
        }
    }

    for (short row = 0; row < N_R0_KQ; ++row) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && (first_row + row) < N) {
            output[first_row + row] = tot;
        }
    }
}

// 15. copy_buffer — Copy N floats from src to dest at a float offset.
//     Used for appending to GPU KV cache without CPU stalls.
kernel void copy_buffer(
    device const float* src     [[buffer(0)]],
    device       float* dest    [[buffer(1)]],
    constant     uint&  count   [[buffer(2)]],
    constant     uint&  dest_offset [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        dest[dest_offset + gid] = src[gid];
    }
}

// -----------------------------------------------------------------------
// 16. copy_f32_to_f16 — Convert F32 to F16 and copy into destination.
//     Used for writing F32 K/V projections into F16 KV cache.
//     src: f32[count], dest: f16[...], writes at dest[dest_offset..].
// -----------------------------------------------------------------------
kernel void copy_f32_to_f16(
    device const float* src         [[buffer(0)]],
    device       half*  dest        [[buffer(1)]],
    constant     uint&  count       [[buffer(2)]],
    constant     uint&  dest_offset [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        dest[dest_offset + gid] = half(src[gid]);
    }
}

// =========================================================================
// Fused matmul+bias kernels — eliminate separate add_bias dispatch
// =========================================================================

// -----------------------------------------------------------------------
// quantized_matmul_bias_q8_0 — Q8_0 matmul with fused bias addition
//   Same as quantized_matmul_q8_0 but adds bias[row] after reduction.
// -----------------------------------------------------------------------
kernel void quantized_matmul_bias_q8_0(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  N       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    device const float* bias    [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = K / QK8_0;
    const uint r0 = (tgpig.x * N_SG_Q8 + sgitg) * N_R0_Q8;

    constexpr short NQ = 8;

    device const block_q8_0 * ax[N_R0_Q8];
    for (short row = 0; row < N_R0_Q8; ++row) {
        ax[row] = (device const block_q8_0 *)(weights + (r0 + row) * nb * 34);
    }

    float sumf[N_R0_Q8] = { 0.0f };

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);

    const int ib0 = ix;
    float yl[NQ];
    device const float * yb = input + ib0 * QK8_0 + il * NQ;

    for (int ib = ib0; ib < (int)nb; ib += NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }
        for (short row = 0; row < N_R0_Q8; ++row) {
            device const int8_t * qs = ax[row][ib].qs + il * NQ;
            float sumq = 0.0f;
            for (short i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * float(ax[row][ib].d);
        }
        yb += NQ * QK8_0;
    }

    for (short row = 0; row < N_R0_Q8; ++row) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && (r0 + row) < N) {
            output[r0 + row] = tot + bias[r0 + row];
        }
    }
}

// -----------------------------------------------------------------------
// quantized_matmul_bias_q4_0 — Q4_0 matmul with fused bias addition
// -----------------------------------------------------------------------
kernel void quantized_matmul_bias_q4_0(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  N       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    device const float* bias    [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = K / QK4_0;
    const uint r0 = (tgpig.x * N_SG_Q4 + sgitg) * N_R0_Q4;

    device const block_q4_0 * ax[N_R0_Q4];
    for (short row = 0; row < N_R0_Q4; ++row) {
        ax[row] = (device const block_q4_0 *)(weights + (r0 + row) * nb * 18);
    }

    float sumf[N_R0_Q4] = { 0.0f };

    constexpr short NQ = 16;
    const short ix = tiisg / (NW / NQ);
    const short il = (tiisg % (NW / NQ)) * 8;

    const int ib0 = ix;
    device const float * yb = input + ib0 * QK4_0 + il;

    for (int ib = ib0; ib < (int)nb; ib += NQ) {
        float yl[16];
        float sumy0 = 0.0f;
        float sumy1 = 0.0f;
        for (short i = 0; i < 8; ++i) {
            yl[i]     = yb[i];
            yl[i + 8] = yb[i + 16];
            sumy0 += yl[i];
            sumy1 += yl[i + 8];
        }
        for (short row = 0; row < N_R0_Q4; ++row) {
            device const uint8_t * qs = ax[row][ib].qs + il / 2;
            float sumq = 0.0f;
            for (short i = 0; i < 8; ++i) {
                uint8_t qbyte = qs[i];
                float q_lo = float(qbyte & 0x0F);
                float q_hi = float(qbyte >> 4);
                sumq += q_lo * yl[i] + q_hi * yl[i + 8];
            }
            sumf[row] += float(ax[row][ib].d) * (sumq - 8.0f * (sumy0 + sumy1));
        }
        yb += NQ * QK4_0;
    }

    for (short row = 0; row < N_R0_Q4; ++row) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && (r0 + row) < N) {
            output[r0 + row] = tot + bias[r0 + row];
        }
    }
}

// -----------------------------------------------------------------------
// quantized_matmul_bias_q4_k — Q4_K matmul with fused bias addition
// -----------------------------------------------------------------------
kernel void quantized_matmul_bias_q4_k(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  N       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    device const float* bias    [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = K / QK_K;
    const uint first_row = (tgpig.x * N_SG_KQ + sgitg) * N_R0_KQ;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[N_R0_KQ] = {0.f};

    device const float * y4 = input + ix * QK_K + 64 * iq + 8 * ir;

    ushort sc16[4];
    thread const uchar * sc8 = (thread const uchar *)sc16;

    float yl[16];
    float yh[16];

    for (int ib = ix; ib < (int)nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        for (short row = 0; row < N_R0_KQ; ++row) {
            uint actual_row = first_row + row;
            if (actual_row >= N) continue;

            device const uchar*  bp = (device const uchar*)(weights + actual_row * nb * Q4_K_BLOCK_SIZE + ib * Q4_K_BLOCK_SIZE);
            device const half*   dh = (device const half*)bp;
            device const ushort* sc = (device const ushort*)(bp + 4) + iq;
            device const ushort* q1 = (device const ushort*)(bp + 16) + 16 * iq + 4 * ir;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const ushort * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += float(dh[0]) * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                          (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         float(dh[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y4 += 4 * QK_K;
    }

    for (short row = 0; row < N_R0_KQ; ++row) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && (first_row + row) < N) {
            output[first_row + row] = tot + bias[first_row + row];
        }
    }
}

// -----------------------------------------------------------------------
// quantized_matmul_bias_q5_k — Q5_K matmul with fused bias addition
// -----------------------------------------------------------------------
kernel void quantized_matmul_bias_q5_k(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  N       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    device const float* bias    [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = K / QK_K;
    const uint first_row = (tgpig.x * N_SG_KQ + sgitg) * N_R0_KQ;

    const short elem_offset = tiisg * 8;
    const short sb   = elem_offset / 32;
    const short pos  = elem_offset % 32;
    const short pair = sb / 2;
    const bool  is_high = (sb & 1);
    const uchar qh_bit  = uchar(1 << sb);

    float sumf[N_R0_KQ] = {0.f};

    for (int ib = 0; ib < (int)nb; ib++) {
        device const float* y = input + ib * QK_K + elem_offset;
        float yl[8];
        for (short i = 0; i < 8; i++) yl[i] = y[i];

        for (short row = 0; row < N_R0_KQ; ++row) {
            uint actual_row = first_row + row;
            if (actual_row >= N) continue;

            device const uchar* bp = (device const uchar*)(weights + actual_row * nb * Q5_K_BLOCK_SIZE + ib * Q5_K_BLOCK_SIZE);
            float d_val    = float(*(device const half*)(bp));
            float dmin_val = float(*(device const half*)(bp + 2));
            device const uchar* sa = bp + 4;
            device const uchar* qh_arr = bp + 16;
            device const uchar* qs_arr = bp + 48;

            uchar sc_val, m_val;
            if (sb < 4) {
                sc_val = sa[sb] & 63;
                m_val  = sa[sb + 4] & 63;
            } else {
                sc_val = (sa[sb + 4] & 0xF) | ((sa[sb - 4] >> 6) << 4);
                m_val  = (sa[sb + 4] >> 4)  | ((sa[sb]     >> 6) << 4);
            }
            float sc_f = d_val * float(sc_val);
            float m_f  = dmin_val * float(m_val);

            float partial = 0.f;
            for (short i = 0; i < 8; i++) {
                short l = pos + i;
                uchar q_low  = is_high ? (qs_arr[pair * 32 + l] >> 4)
                                       : (qs_arr[pair * 32 + l] & 0xF);
                uchar q_high = (qh_arr[l] & qh_bit) ? 16 : 0;
                partial += (sc_f * float(q_low + q_high) - m_f) * yl[i];
            }
            sumf[row] += partial;
        }
    }

    for (short row = 0; row < N_R0_KQ; ++row) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && (first_row + row) < N) {
            output[first_row + row] = tot + bias[first_row + row];
        }
    }
}

// -----------------------------------------------------------------------
// quantized_matmul_bias_q6_k — Q6_K matmul with fused bias addition
// -----------------------------------------------------------------------
kernel void quantized_matmul_bias_q6_k(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  N       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    device const float* bias    [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = K / QK_K;
    const uint first_row = (tgpig.x * N_SG_KQ + sgitg) * N_R0_KQ;

    const short elem_offset = tiisg * 8;
    const short half_idx    = elem_offset / 128;
    const short within_half = elem_offset % 128;
    const short group32     = within_half / 32;
    const short pos         = within_half % 32;

    const uint  ql_off   = half_idx * 64 + ((group32 & 1) ? 32 : 0);
    const uint  qh_off   = half_idx * 32;
    const short qh_shift = group32 * 2;
    const bool  high_nib = (group32 >= 2);
    const short sc_off   = half_idx * 8 + group32 * 2;

    float sumf[N_R0_KQ] = {0.f};

    for (int ib = 0; ib < (int)nb; ib++) {
        device const float* y = input + ib * QK_K + elem_offset;
        float yl[8];
        for (short i = 0; i < 8; i++) yl[i] = y[i];

        for (short row = 0; row < N_R0_KQ; ++row) {
            uint actual_row = first_row + row;
            if (actual_row >= N) continue;

            device const uchar* bp = (device const uchar*)(weights + actual_row * nb * Q6_K_BLOCK_SIZE + ib * Q6_K_BLOCK_SIZE);
            device const uchar* ql = bp;
            device const uchar* qh = bp + 128;
            device const char*  sc = (device const char*)(bp + 192);
            float d = float(*(device const half*)(bp + 208));

            float partial = 0.f;
            for (short i = 0; i < 8; i++) {
                short l = pos + i;
                uchar ql_byte = ql[ql_off + l];
                uchar qh_byte = qh[qh_off + l];
                uchar q_low   = high_nib ? (ql_byte >> 4) : (ql_byte & 0xF);
                uchar q_high  = (qh_byte >> qh_shift) & 3;
                int   q_val   = int(q_low | (q_high << 4)) - 32;
                char  scale   = sc[sc_off + (l / 16)];
                partial += d * float(scale) * float(q_val) * yl[i];
            }
            sumf[row] += partial;
        }
    }

    for (short row = 0; row < N_R0_KQ; ++row) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && (first_row + row) < N) {
            output[first_row + row] = tot + bias[first_row + row];
        }
    }
}

// -----------------------------------------------------------------------
// gemm_transpose_bias — F32 GEMM with transpose + fused bias addition
//   C[M,N] = A[M,K] * B^T + bias[N]   where B is stored as (N,K)
// -----------------------------------------------------------------------
kernel void gemm_transpose_bias(
    device const float* A       [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device       float* C       [[buffer(2)]],
    constant     uint&  M       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    constant     uint&  N       [[buffer(5)]],
    device const float* bias    [[buffer(6)]],
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

    // Store results with bias addition
    // Use staging to add bias then write
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
            C[gr * N + gc] = staging[r * BN + c] + bias[gc];
        }
    }
}

// -----------------------------------------------------------------------
// grouped_attn_decode_f16 — Online softmax attention reading F16 K/V cache
//
//   Single-pass online softmax (Milakov & Gimelshein 2018):
//   No separate max-finding pass needed. Updates running max, running sum,
//   and V accumulator in a single scan over positions.
//
//   Reads F16 K and V cache (half the bandwidth of F32).
//   Q is still F32 (comes from matmul output).
// -----------------------------------------------------------------------
kernel void grouped_attn_decode_f16(
    device const float* Q       [[buffer(0)]],
    device const half*  K       [[buffer(1)]],
    device const half*  V       [[buffer(2)]],
    device       float* output  [[buffer(3)]],
    constant     uint&  num_heads    [[buffer(4)]],
    constant     uint&  num_kv_heads [[buffer(5)]],
    constant     uint&  head_dim     [[buffer(6)]],
    constant     uint&  total_len    [[buffer(7)]],
    constant     float& attn_scale   [[buffer(8)]],
    constant     float& softcap      [[buffer(9)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]])
{
    uint h = gid;
    if (h >= num_heads) return;

    uint kv_dim = num_kv_heads * head_dim;
    uint kv_head = h * num_kv_heads / num_heads;
    device const float* q_head = Q + h * head_dim;
    uint kv_off = kv_head * head_dim;

    threadgroup float shared[256];

    // ---- Single-pass online softmax ----
    // Each thread processes positions strided by tpg, maintaining its own
    // running_max, running_sum, and v_acc (for its output dimension lid).
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float v_acc = 0.0f;

    for (uint tile = 0; tile < total_len; tile += tpg) {
        uint j = tile + lid;

        // Step A: each thread computes score for position j
        float score = -INFINITY;
        if (j < total_len) {
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += q_head[d] * float(K[j * kv_dim + kv_off + d]);
            }
            score = dot * attn_scale;
            if (softcap > 0.0f) {
                score = softcap * precise::tanh(score / softcap);
            }
        }
        // Store score in shared memory so all threads can access all scores in this tile
        shared[lid] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step B: threads parallel over head_dim dimensions
        // For each position in this tile, update online softmax state
        uint active = min(tpg, total_len - tile);
        if (lid < head_dim) {
            for (uint t = 0; t < active; ++t) {
                float s = shared[t];
                if (s > running_max) {
                    float correction = exp(running_max - s);
                    running_sum = running_sum * correction + 1.0f;
                    v_acc = v_acc * correction + float(V[(tile + t) * kv_dim + kv_off + lid]);
                    running_max = s;
                } else {
                    float w = exp(s - running_max);
                    running_sum += w;
                    v_acc += w * float(V[(tile + t) * kv_dim + kv_off + lid]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write normalized output
    if (lid < head_dim) {
        device float* out_head = output + h * head_dim;
        out_head[lid] = (running_sum > 0.0f) ? (v_acc / running_sum) : 0.0f;
    }
}

// ===========================================================================
// Phase 3: Batched causal attention for prefill
// ===========================================================================

// Multi-token causal attention with online softmax. Reads F32 K/V cache.
// One threadgroup per (head, query_token) pair.
// Dispatch: (num_heads, n_tokens, 1) threadgroups, 256 threads each.
kernel void batched_causal_attention(
    device const float* Q        [[buffer(0)]],   // [n_tokens, num_heads * head_dim]
    device const float* K        [[buffer(1)]],   // [max_seq_len, num_kv_heads * head_dim]
    device const float* V        [[buffer(2)]],   // [max_seq_len, num_kv_heads * head_dim]
    device       float* output   [[buffer(3)]],   // [n_tokens, num_heads * head_dim]
    constant     uint&  num_heads     [[buffer(4)]],
    constant     uint&  num_kv_heads  [[buffer(5)]],
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  n_tokens      [[buffer(7)]],
    constant     uint&  total_len     [[buffer(8)]],
    constant     uint&  pos_offset    [[buffer(9)]],
    constant     float& attn_scale    [[buffer(10)]],
    constant     float& softcap       [[buffer(11)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]])
{
    // Flatten 2D (head, query_token) into 1D: gid = q_idx * num_heads + h
    uint h = gid % num_heads;        // head index
    uint q_idx = gid / num_heads;    // query token index within this batch

    // GQA: map Q head to KV head (same formula as grouped_attn_decode)
    uint kv_head = h * num_kv_heads / num_heads;
    uint kv_dim = num_kv_heads * head_dim;
    uint total_dim = num_heads * head_dim;
    uint kv_off = kv_head * head_dim;

    // Causal mask: this query can attend to positions [0, pos_offset + q_idx]
    uint max_attend = pos_offset + q_idx + 1;
    if (max_attend > total_len) max_attend = total_len;

    // Load Q for this head and query token
    device const float* q_head = Q + q_idx * total_dim + h * head_dim;

    // Shared memory for scores (one per thread in the threadgroup)
    threadgroup float shared[256];

    // Online softmax: accumulate over K/V positions
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float v_acc = 0.0f;

    for (uint tile = 0; tile < max_attend; tile += tpg) {
        uint j = tile + lid;

        // Step A: each thread computes score for position j
        float score = -INFINITY;
        if (j < max_attend) {
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += q_head[d] * K[j * kv_dim + kv_off + d];
            }
            score = dot * attn_scale;
            if (softcap > 0.0f) {
                score = softcap * precise::tanh(score / softcap);
            }
        }
        shared[lid] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step B: threads parallel over head_dim dimensions
        uint active = min(tpg, max_attend - tile);
        if (lid < head_dim) {
            for (uint t = 0; t < active; ++t) {
                float s = shared[t];
                if (s > running_max) {
                    float correction = exp(running_max - s);
                    running_sum = running_sum * correction + 1.0f;
                    v_acc = v_acc * correction + V[(tile + t) * kv_dim + kv_off + lid];
                    running_max = s;
                } else {
                    float w = exp(s - running_max);
                    running_sum += w;
                    v_acc += w * V[(tile + t) * kv_dim + kv_off + lid];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write normalized output
    if (lid < head_dim) {
        device float* out_head = output + q_idx * total_dim + h * head_dim;
        out_head[lid] = (running_sum > 0.0f) ? (v_acc / running_sum) : 0.0f;
    }
}

// ===========================================================================
// Phase 3b: Batched causal attention reading F16 K/V cache
// ===========================================================================

// Same as batched_causal_attention but K and V are stored as half (F16).
// Q and output remain F32.
kernel void batched_causal_attention_f16(
    device const float* Q        [[buffer(0)]],   // [n_tokens, num_heads * head_dim]
    device const half*  K        [[buffer(1)]],   // [max_seq_len, num_kv_heads * head_dim]
    device const half*  V        [[buffer(2)]],   // [max_seq_len, num_kv_heads * head_dim]
    device       float* output   [[buffer(3)]],   // [n_tokens, num_heads * head_dim]
    constant     uint&  num_heads     [[buffer(4)]],
    constant     uint&  num_kv_heads  [[buffer(5)]],
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  n_tokens      [[buffer(7)]],
    constant     uint&  total_len     [[buffer(8)]],
    constant     uint&  pos_offset    [[buffer(9)]],
    constant     float& attn_scale    [[buffer(10)]],
    constant     float& softcap       [[buffer(11)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]])
{
    uint h = gid % num_heads;
    uint q_idx = gid / num_heads;

    uint kv_head = h * num_kv_heads / num_heads;
    uint kv_dim = num_kv_heads * head_dim;
    uint total_dim = num_heads * head_dim;
    uint kv_off = kv_head * head_dim;

    uint max_attend = pos_offset + q_idx + 1;
    if (max_attend > total_len) max_attend = total_len;

    device const float* q_head = Q + q_idx * total_dim + h * head_dim;

    threadgroup float shared[256];

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float v_acc = 0.0f;

    for (uint tile = 0; tile < max_attend; tile += tpg) {
        uint j = tile + lid;

        float score = -INFINITY;
        if (j < max_attend) {
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += q_head[d] * float(K[j * kv_dim + kv_off + d]);
            }
            score = dot * attn_scale;
            if (softcap > 0.0f) {
                score = softcap * precise::tanh(score / softcap);
            }
        }
        shared[lid] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint active = min(tpg, max_attend - tile);
        if (lid < head_dim) {
            for (uint t = 0; t < active; ++t) {
                float s = shared[t];
                if (s > running_max) {
                    float correction = exp(running_max - s);
                    running_sum = running_sum * correction + 1.0f;
                    v_acc = v_acc * correction + float(V[(tile + t) * kv_dim + kv_off + lid]);
                    running_max = s;
                } else {
                    float w = exp(s - running_max);
                    running_sum += w;
                    v_acc += w * float(V[(tile + t) * kv_dim + kv_off + lid]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid < head_dim) {
        device float* out_head = output + q_idx * total_dim + h * head_dim;
        out_head[lid] = (running_sum > 0.0f) ? (v_acc / running_sum) : 0.0f;
    }
}

// =========================================================================
// Phase 4: Batched quantized matmul kernels for prefill
//
// C[M,N] = input[M,K] * weights^T  where weights is [N,K] quantized row-major.
// Same tiling as gemm_transpose (BM=32, BN=32, BK=32, 128 threads, simdgroup 8x8).
// B-tile loading dequantizes from quantized weights instead of copying floats.
// =========================================================================

// -----------------------------------------------------------------------
// batched_matmul_q8_0 — Q8_0 batched GEMM (no bias)
// -----------------------------------------------------------------------
kernel void batched_matmul_q8_0(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
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
    uint row_bytes = (K / QK8_0) * 34;

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_base = kt * BK;

        for (uint idx = lid; idx < BM * BK; idx += 128) {
            uint r = idx / BK;
            uint c = idx % BK;
            uint gr = tile_row + r;
            uint gc = k_base + c;
            tgA[r * BK + c] = (gr < M && gc < K) ? input[gr * K + gc] : 0.0f;
        }

        for (uint idx = lid; idx < BK * BN; idx += 128) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint gk = k_base + r;
            uint gn = tile_col + c;
            float val = 0.0f;
            if (gn < N && gk < K) {
                uint block_idx = gk / QK8_0;
                uint elem = gk % QK8_0;
                device const uchar* bp = (device const uchar*)(weights + gn * row_bytes + block_idx * 34);
                float d = float(*(device const half*)bp);
                int8_t qs = ((device const int8_t*)(bp + 2))[elem];
                val = d * float(qs);
            }
            tgB[r * BN + c] = val;
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
        simdgroup_store(acc[0][0], &output[(out_row + 0) * N + (out_col + 0)], N);
        simdgroup_store(acc[0][1], &output[(out_row + 0) * N + (out_col + 8)], N);
        simdgroup_store(acc[1][0], &output[(out_row + 8) * N + (out_col + 0)], N);
        simdgroup_store(acc[1][1], &output[(out_row + 8) * N + (out_col + 8)], N);
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
                output[gr * N + gc] = staging[r * BN + c];
            }
        }
    }
}

// -----------------------------------------------------------------------
// batched_matmul_bias_q8_0 — Q8_0 batched GEMM with fused bias
// -----------------------------------------------------------------------
kernel void batched_matmul_bias_q8_0(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  M       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    constant     uint&  N       [[buffer(5)]],
    device const float* bias    [[buffer(6)]],
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
    uint row_bytes = (K / QK8_0) * 34;

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_base = kt * BK;

        for (uint idx = lid; idx < BM * BK; idx += 128) {
            uint r = idx / BK;
            uint c = idx % BK;
            uint gr = tile_row + r;
            uint gc = k_base + c;
            tgA[r * BK + c] = (gr < M && gc < K) ? input[gr * K + gc] : 0.0f;
        }

        for (uint idx = lid; idx < BK * BN; idx += 128) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint gk = k_base + r;
            uint gn = tile_col + c;
            float val = 0.0f;
            if (gn < N && gk < K) {
                uint block_idx = gk / QK8_0;
                uint elem = gk % QK8_0;
                device const uchar* bp = (device const uchar*)(weights + gn * row_bytes + block_idx * 34);
                float d = float(*(device const half*)bp);
                int8_t qs = ((device const int8_t*)(bp + 2))[elem];
                val = d * float(qs);
            }
            tgB[r * BN + c] = val;
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
            output[gr * N + gc] = staging[r * BN + c] + bias[gc];
        }
    }
}

// -----------------------------------------------------------------------
// batched_matmul_q4_0 — Q4_0 batched GEMM (no bias)
// -----------------------------------------------------------------------
kernel void batched_matmul_q4_0(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
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
    uint row_bytes = (K / QK4_0) * 18;

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_base = kt * BK;

        for (uint idx = lid; idx < BM * BK; idx += 128) {
            uint r = idx / BK;
            uint c = idx % BK;
            uint gr = tile_row + r;
            uint gc = k_base + c;
            tgA[r * BK + c] = (gr < M && gc < K) ? input[gr * K + gc] : 0.0f;
        }

        for (uint idx = lid; idx < BK * BN; idx += 128) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint gk = k_base + r;
            uint gn = tile_col + c;
            float val = 0.0f;
            if (gn < N && gk < K) {
                uint block_idx = gk / QK4_0;
                uint elem = gk % QK4_0;
                device const uchar* bp = (device const uchar*)(weights + gn * row_bytes + block_idx * 18);
                float d = float(*(device const half*)bp);
                device const uchar* qs = bp + 2;
                float q_val;
                if (elem < 16) {
                    q_val = float(qs[elem] & 0x0F) - 8.0f;
                } else {
                    q_val = float(qs[elem - 16] >> 4) - 8.0f;
                }
                val = d * q_val;
            }
            tgB[r * BN + c] = val;
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
        simdgroup_store(acc[0][0], &output[(out_row + 0) * N + (out_col + 0)], N);
        simdgroup_store(acc[0][1], &output[(out_row + 0) * N + (out_col + 8)], N);
        simdgroup_store(acc[1][0], &output[(out_row + 8) * N + (out_col + 0)], N);
        simdgroup_store(acc[1][1], &output[(out_row + 8) * N + (out_col + 8)], N);
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
                output[gr * N + gc] = staging[r * BN + c];
            }
        }
    }
}

// -----------------------------------------------------------------------
// batched_matmul_bias_q4_0 — Q4_0 batched GEMM with fused bias
// -----------------------------------------------------------------------
kernel void batched_matmul_bias_q4_0(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  M       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    constant     uint&  N       [[buffer(5)]],
    device const float* bias    [[buffer(6)]],
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
    uint row_bytes = (K / QK4_0) * 18;

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_base = kt * BK;

        for (uint idx = lid; idx < BM * BK; idx += 128) {
            uint r = idx / BK;
            uint c = idx % BK;
            uint gr = tile_row + r;
            uint gc = k_base + c;
            tgA[r * BK + c] = (gr < M && gc < K) ? input[gr * K + gc] : 0.0f;
        }

        for (uint idx = lid; idx < BK * BN; idx += 128) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint gk = k_base + r;
            uint gn = tile_col + c;
            float val = 0.0f;
            if (gn < N && gk < K) {
                uint block_idx = gk / QK4_0;
                uint elem = gk % QK4_0;
                device const uchar* bp = (device const uchar*)(weights + gn * row_bytes + block_idx * 18);
                float d = float(*(device const half*)bp);
                device const uchar* qs = bp + 2;
                float q_val;
                if (elem < 16) {
                    q_val = float(qs[elem] & 0x0F) - 8.0f;
                } else {
                    q_val = float(qs[elem - 16] >> 4) - 8.0f;
                }
                val = d * q_val;
            }
            tgB[r * BN + c] = val;
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
            output[gr * N + gc] = staging[r * BN + c] + bias[gc];
        }
    }
}

// -----------------------------------------------------------------------
// batched_matmul_q4_k — Q4_K batched GEMM (no bias)
// -----------------------------------------------------------------------
kernel void batched_matmul_q4_k(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
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
    uint row_bytes = (K / QK_K) * Q4_K_BLOCK_SIZE;

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_base = kt * BK;

        for (uint idx = lid; idx < BM * BK; idx += 128) {
            uint r = idx / BK;
            uint c = idx % BK;
            uint gr = tile_row + r;
            uint gc = k_base + c;
            tgA[r * BK + c] = (gr < M && gc < K) ? input[gr * K + gc] : 0.0f;
        }

        for (uint idx = lid; idx < BK * BN; idx += 128) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint gk = k_base + r;
            uint gn = tile_col + c;
            float val = 0.0f;
            if (gn < N && gk < K) {
                uint sbi = gk / QK_K;
                uint pos = gk % QK_K;
                uint sub_block = pos / 32;
                device const uchar* bp = (device const uchar*)(weights + gn * row_bytes + sbi * Q4_K_BLOCK_SIZE);
                float d_val    = float(*(device const half*)(bp));
                float dmin_val = float(*(device const half*)(bp + 2));
                device const uchar* sa = bp + 4;
                device const uchar* qs_arr = bp + 16;

                uchar sc_val, m_val;
                if (sub_block < 4) {
                    sc_val = sa[sub_block] & 63;
                    m_val  = sa[sub_block + 4] & 63;
                } else {
                    sc_val = (sa[sub_block + 4] & 0xF) | ((sa[sub_block - 4] >> 6) << 4);
                    m_val  = (sa[sub_block + 4] >> 4)  | ((sa[sub_block]     >> 6) << 4);
                }

                uint group64 = pos / 64;
                uint within64 = pos % 64;
                uint byte_idx = group64 * 32 + (within64 < 32 ? within64 : within64 - 32);
                uchar q_byte = qs_arr[byte_idx];
                uchar nibble = (within64 < 32) ? (q_byte & 0xF) : (q_byte >> 4);

                val = d_val * float(sc_val) * float(nibble) - dmin_val * float(m_val);
            }
            tgB[r * BN + c] = val;
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
        simdgroup_store(acc[0][0], &output[(out_row + 0) * N + (out_col + 0)], N);
        simdgroup_store(acc[0][1], &output[(out_row + 0) * N + (out_col + 8)], N);
        simdgroup_store(acc[1][0], &output[(out_row + 8) * N + (out_col + 0)], N);
        simdgroup_store(acc[1][1], &output[(out_row + 8) * N + (out_col + 8)], N);
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
                output[gr * N + gc] = staging[r * BN + c];
            }
        }
    }
}

// -----------------------------------------------------------------------
// batched_matmul_bias_q4_k — Q4_K batched GEMM with fused bias
// -----------------------------------------------------------------------
kernel void batched_matmul_bias_q4_k(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  M       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    constant     uint&  N       [[buffer(5)]],
    device const float* bias    [[buffer(6)]],
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
    uint row_bytes = (K / QK_K) * Q4_K_BLOCK_SIZE;

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_base = kt * BK;

        for (uint idx = lid; idx < BM * BK; idx += 128) {
            uint r = idx / BK;
            uint c = idx % BK;
            uint gr = tile_row + r;
            uint gc = k_base + c;
            tgA[r * BK + c] = (gr < M && gc < K) ? input[gr * K + gc] : 0.0f;
        }

        for (uint idx = lid; idx < BK * BN; idx += 128) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint gk = k_base + r;
            uint gn = tile_col + c;
            float val = 0.0f;
            if (gn < N && gk < K) {
                uint sbi = gk / QK_K;
                uint pos = gk % QK_K;
                uint sub_block = pos / 32;
                device const uchar* bp = (device const uchar*)(weights + gn * row_bytes + sbi * Q4_K_BLOCK_SIZE);
                float d_val    = float(*(device const half*)(bp));
                float dmin_val = float(*(device const half*)(bp + 2));
                device const uchar* sa = bp + 4;
                device const uchar* qs_arr = bp + 16;

                uchar sc_val, m_val;
                if (sub_block < 4) {
                    sc_val = sa[sub_block] & 63;
                    m_val  = sa[sub_block + 4] & 63;
                } else {
                    sc_val = (sa[sub_block + 4] & 0xF) | ((sa[sub_block - 4] >> 6) << 4);
                    m_val  = (sa[sub_block + 4] >> 4)  | ((sa[sub_block]     >> 6) << 4);
                }

                uint group64 = pos / 64;
                uint within64 = pos % 64;
                uint byte_idx = group64 * 32 + (within64 < 32 ? within64 : within64 - 32);
                uchar q_byte = qs_arr[byte_idx];
                uchar nibble = (within64 < 32) ? (q_byte & 0xF) : (q_byte >> 4);

                val = d_val * float(sc_val) * float(nibble) - dmin_val * float(m_val);
            }
            tgB[r * BN + c] = val;
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
            output[gr * N + gc] = staging[r * BN + c] + bias[gc];
        }
    }
}

// -----------------------------------------------------------------------
// batched_matmul_q5_k — Q5_K batched GEMM (no bias)
// -----------------------------------------------------------------------
kernel void batched_matmul_q5_k(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
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
    uint row_bytes = (K / QK_K) * Q5_K_BLOCK_SIZE;

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_base = kt * BK;

        for (uint idx = lid; idx < BM * BK; idx += 128) {
            uint r = idx / BK;
            uint c = idx % BK;
            uint gr = tile_row + r;
            uint gc = k_base + c;
            tgA[r * BK + c] = (gr < M && gc < K) ? input[gr * K + gc] : 0.0f;
        }

        for (uint idx = lid; idx < BK * BN; idx += 128) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint gk = k_base + r;
            uint gn = tile_col + c;
            float val = 0.0f;
            if (gn < N && gk < K) {
                uint sbi = gk / QK_K;
                uint pos = gk % QK_K;
                uint sub_block = pos / 32;
                uint elem = pos % 32;
                device const uchar* bp = (device const uchar*)(weights + gn * row_bytes + sbi * Q5_K_BLOCK_SIZE);
                float d_val    = float(*(device const half*)(bp));
                float dmin_val = float(*(device const half*)(bp + 2));
                device const uchar* sa = bp + 4;
                device const uchar* qh_arr = bp + 16;
                device const uchar* qs_arr = bp + 48;

                uchar sc_val, m_val;
                if (sub_block < 4) {
                    sc_val = sa[sub_block] & 63;
                    m_val  = sa[sub_block + 4] & 63;
                } else {
                    sc_val = (sa[sub_block + 4] & 0xF) | ((sa[sub_block - 4] >> 6) << 4);
                    m_val  = (sa[sub_block + 4] >> 4)  | ((sa[sub_block]     >> 6) << 4);
                }

                uint pair = sub_block / 2;
                bool is_high = (sub_block & 1) != 0;
                uchar q_low = is_high ? (qs_arr[pair * 32 + elem] >> 4)
                                       : (qs_arr[pair * 32 + elem] & 0xF);
                uchar qh_bit = uchar(1 << sub_block);
                uchar q_high = (qh_arr[elem] & qh_bit) ? 16 : 0;

                val = d_val * float(sc_val) * float(q_low + q_high) - dmin_val * float(m_val);
            }
            tgB[r * BN + c] = val;
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
        simdgroup_store(acc[0][0], &output[(out_row + 0) * N + (out_col + 0)], N);
        simdgroup_store(acc[0][1], &output[(out_row + 0) * N + (out_col + 8)], N);
        simdgroup_store(acc[1][0], &output[(out_row + 8) * N + (out_col + 0)], N);
        simdgroup_store(acc[1][1], &output[(out_row + 8) * N + (out_col + 8)], N);
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
                output[gr * N + gc] = staging[r * BN + c];
            }
        }
    }
}

// -----------------------------------------------------------------------
// batched_matmul_bias_q5_k — Q5_K batched GEMM with fused bias
// -----------------------------------------------------------------------
kernel void batched_matmul_bias_q5_k(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  M       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    constant     uint&  N       [[buffer(5)]],
    device const float* bias    [[buffer(6)]],
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
    uint row_bytes = (K / QK_K) * Q5_K_BLOCK_SIZE;

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_base = kt * BK;

        for (uint idx = lid; idx < BM * BK; idx += 128) {
            uint r = idx / BK;
            uint c = idx % BK;
            uint gr = tile_row + r;
            uint gc = k_base + c;
            tgA[r * BK + c] = (gr < M && gc < K) ? input[gr * K + gc] : 0.0f;
        }

        for (uint idx = lid; idx < BK * BN; idx += 128) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint gk = k_base + r;
            uint gn = tile_col + c;
            float val = 0.0f;
            if (gn < N && gk < K) {
                uint sbi = gk / QK_K;
                uint pos = gk % QK_K;
                uint sub_block = pos / 32;
                uint elem = pos % 32;
                device const uchar* bp = (device const uchar*)(weights + gn * row_bytes + sbi * Q5_K_BLOCK_SIZE);
                float d_val    = float(*(device const half*)(bp));
                float dmin_val = float(*(device const half*)(bp + 2));
                device const uchar* sa = bp + 4;
                device const uchar* qh_arr = bp + 16;
                device const uchar* qs_arr = bp + 48;

                uchar sc_val, m_val;
                if (sub_block < 4) {
                    sc_val = sa[sub_block] & 63;
                    m_val  = sa[sub_block + 4] & 63;
                } else {
                    sc_val = (sa[sub_block + 4] & 0xF) | ((sa[sub_block - 4] >> 6) << 4);
                    m_val  = (sa[sub_block + 4] >> 4)  | ((sa[sub_block]     >> 6) << 4);
                }

                uint pair = sub_block / 2;
                bool is_high = (sub_block & 1) != 0;
                uchar q_low = is_high ? (qs_arr[pair * 32 + elem] >> 4)
                                       : (qs_arr[pair * 32 + elem] & 0xF);
                uchar qh_bit = uchar(1 << sub_block);
                uchar q_high = (qh_arr[elem] & qh_bit) ? 16 : 0;

                val = d_val * float(sc_val) * float(q_low + q_high) - dmin_val * float(m_val);
            }
            tgB[r * BN + c] = val;
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
            output[gr * N + gc] = staging[r * BN + c] + bias[gc];
        }
    }
}

// -----------------------------------------------------------------------
// batched_matmul_q6_k — Q6_K batched GEMM (no bias)
// -----------------------------------------------------------------------
kernel void batched_matmul_q6_k(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
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
    uint row_bytes = (K / QK_K) * Q6_K_BLOCK_SIZE;

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_base = kt * BK;

        for (uint idx = lid; idx < BM * BK; idx += 128) {
            uint r = idx / BK;
            uint c = idx % BK;
            uint gr = tile_row + r;
            uint gc = k_base + c;
            tgA[r * BK + c] = (gr < M && gc < K) ? input[gr * K + gc] : 0.0f;
        }

        for (uint idx = lid; idx < BK * BN; idx += 128) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint gk = k_base + r;
            uint gn = tile_col + c;
            float val = 0.0f;
            if (gn < N && gk < K) {
                uint sbi = gk / QK_K;
                uint pos = gk % QK_K;
                device const uchar* bp = (device const uchar*)(weights + gn * row_bytes + sbi * Q6_K_BLOCK_SIZE);
                device const uchar* ql = bp;
                device const uchar* qh = bp + 128;
                device const char*  sc = (device const char*)(bp + 192);
                float d = float(*(device const half*)(bp + 208));

                uint half_idx    = pos / 128;
                uint within_half = pos % 128;
                uint group32     = within_half / 32;
                uint pos_in_grp  = within_half % 32;

                uint ql_off   = half_idx * 64 + ((group32 & 1) ? 32 : 0);
                uint qh_off   = half_idx * 32;
                short qh_shift = short(group32 * 2);
                bool high_nib  = (group32 >= 2);
                short sc_off   = short(half_idx * 8 + group32 * 2);

                uchar ql_byte = ql[ql_off + pos_in_grp];
                uchar qh_byte = qh[qh_off + pos_in_grp];
                uchar q_low   = high_nib ? (ql_byte >> 4) : (ql_byte & 0xF);
                uchar q_high  = (qh_byte >> qh_shift) & 3;
                int   q_val   = int(q_low | (q_high << 4)) - 32;
                char  scale   = sc[sc_off + short(pos_in_grp / 16)];

                val = d * float(scale) * float(q_val);
            }
            tgB[r * BN + c] = val;
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
        simdgroup_store(acc[0][0], &output[(out_row + 0) * N + (out_col + 0)], N);
        simdgroup_store(acc[0][1], &output[(out_row + 0) * N + (out_col + 8)], N);
        simdgroup_store(acc[1][0], &output[(out_row + 8) * N + (out_col + 0)], N);
        simdgroup_store(acc[1][1], &output[(out_row + 8) * N + (out_col + 8)], N);
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
                output[gr * N + gc] = staging[r * BN + c];
            }
        }
    }
}

// -----------------------------------------------------------------------
// batched_matmul_bias_q6_k — Q6_K batched GEMM with fused bias
// -----------------------------------------------------------------------
kernel void batched_matmul_bias_q6_k(
    device const char*  weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  M       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    constant     uint&  N       [[buffer(5)]],
    device const float* bias    [[buffer(6)]],
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
    uint row_bytes = (K / QK_K) * Q6_K_BLOCK_SIZE;

    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        uint k_base = kt * BK;

        for (uint idx = lid; idx < BM * BK; idx += 128) {
            uint r = idx / BK;
            uint c = idx % BK;
            uint gr = tile_row + r;
            uint gc = k_base + c;
            tgA[r * BK + c] = (gr < M && gc < K) ? input[gr * K + gc] : 0.0f;
        }

        for (uint idx = lid; idx < BK * BN; idx += 128) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint gk = k_base + r;
            uint gn = tile_col + c;
            float val = 0.0f;
            if (gn < N && gk < K) {
                uint sbi = gk / QK_K;
                uint pos = gk % QK_K;
                device const uchar* bp = (device const uchar*)(weights + gn * row_bytes + sbi * Q6_K_BLOCK_SIZE);
                device const uchar* ql = bp;
                device const uchar* qh = bp + 128;
                device const char*  sc = (device const char*)(bp + 192);
                float d = float(*(device const half*)(bp + 208));

                uint half_idx    = pos / 128;
                uint within_half = pos % 128;
                uint group32     = within_half / 32;
                uint pos_in_grp  = within_half % 32;

                uint ql_off   = half_idx * 64 + ((group32 & 1) ? 32 : 0);
                uint qh_off   = half_idx * 32;
                short qh_shift = short(group32 * 2);
                bool high_nib  = (group32 >= 2);
                short sc_off   = short(half_idx * 8 + group32 * 2);

                uchar ql_byte = ql[ql_off + pos_in_grp];
                uchar qh_byte = qh[qh_off + pos_in_grp];
                uchar q_low   = high_nib ? (ql_byte >> 4) : (ql_byte & 0xF);
                uchar q_high  = (qh_byte >> qh_shift) & 3;
                int   q_val   = int(q_low | (q_high << 4)) - 32;
                char  scale   = sc[sc_off + short(pos_in_grp / 16)];

                val = d * float(scale) * float(q_val);
            }
            tgB[r * BN + c] = val;
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
            output[gr * N + gc] = staging[r * BN + c] + bias[gc];
        }
    }
}
"#;
