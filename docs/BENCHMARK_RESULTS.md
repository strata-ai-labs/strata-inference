# strata vs llama.cpp Performance Benchmark

**Date:** 2026-02-19
**Machine:** Apple M1 Pro
**Prompt:** ~90 tokens | Greedy decoding (temp=0)
**Context size:** 4096
**Commit:** ca6c7eb (Fix Phi-3.5 LongRoPE bugs + per-kernel Metal GPU profiling)

## Metal Backend (100 generated tokens)

| Model | strata pp | strata tg | llama.cpp pp | llama.cpp tg | pp ratio | tg ratio |
|-------|----------:|----------:|-------------:|-------------:|---------:|---------:|
| GPT-2 Q8_0 | 1257.6 | 365.2 | 6564.3 | 297.2 | 0.19x | **1.23x** |
| TinyLlama Q4_K_M | 167.7 | 108.6 | 1210.3 | 143.5 | 0.14x | 0.76x |
| Qwen3-1.7B Q8_0 | 76.2 | 59.3 | 1052.5 | 54.3 | 0.07x | **1.09x** |
| Phi-3.1-mini Q4_K_M | 45.6 | 33.3 | 310.7 | 41.1 | 0.15x | 0.81x |
| Phi-3.5-mini Q4_K_M | 46.1 | 33.9 | 327.5 | 40.7 | 0.14x | 0.83x |
| Gemma3-1B Q4_K_M | err | err | 1368.4 | 80.0 | — | — |

pp = prefill tok/s, tg = decode tok/s. Higher is better. Ratio = strata / llama.cpp.

## CPU Backend (20 generated tokens)

| Model | strata pp | strata tg | llama.cpp pp | llama.cpp tg | pp ratio | tg ratio |
|-------|----------:|----------:|-------------:|-------------:|---------:|---------:|
| GPT-2 Q8_0 | 11.7 | 6.9 | 3140.1 | 104.7 | 0.004x | 0.07x |
| TinyLlama Q4_K_M | 1.0 | 0.9 | 242.0 | 59.1 | 0.004x | 0.015x |
| Qwen3-1.7B Q8_0 | 0.7 | 0.5 | 259.7 | 34.6 | 0.003x | 0.014x |
| Phi-3.1-mini Q4_K_M | 0.3 | 0.2 | 74.4 | 24.7 | 0.004x | 0.008x |
| Phi-3.5-mini Q4_K_M | 0.3 | 0.2 | 63.6 | 14.5 | 0.005x | 0.014x |
| Gemma3-1B Q4_K_M | 1.3 | 0.8 | 156.4 | 47.0 | 0.008x | 0.017x |

## Analysis

### Metal decode is competitive

strata achieves 0.76–1.23x of llama.cpp decode throughput on Metal. GPT-2 (1.23x) and Qwen3-1.7B (1.09x) are actually faster than llama.cpp. The decode path uses a single command buffer with fused kernels per token, which keeps GPU utilization high for the memory-bound single-token forward pass.

### Metal prefill is 5–14x slower

The prefill gap is the largest bottleneck. llama.cpp uses heavily optimized batch GEMM with tiled SIMD group operations and multi-pass accumulation. Our prefill matmul kernels (64x32 tiling, introduced in 1d7a79d) are functional but not yet competitive with llama.cpp's years of Metal shader optimization.

### CPU is ~60–250x slower

The CPU backend has no SIMD intrinsics and no multi-threading. llama.cpp uses NEON (ARM) and AVX (x86) intrinsics with multi-threaded GGML tensor operations. This gap is expected and not a priority — Metal is the primary backend on macOS.

### Gemma3-1B fails on Metal

Gemma3 uses Interleaved Sliding Window Attention (ISWA), which is not yet implemented. This is a known gap tracked separately.

## Optimization Priorities

1. **Metal prefill GEMM** — Largest impact. Investigate llama.cpp's `kernel_mul_mm` for tiling strategy, SIMD group operations, and shared memory usage patterns.
2. **Metal decode for Q4_K_M models** — TinyLlama and Phi models are 17–24% slower than llama.cpp on decode. Profile with `STRATA_PROFILE=1` to identify bottleneck kernels.
3. **ISWA for Gemma3** — Required for Gemma3 family support on Metal.
4. **CPU SIMD** — Low priority unless CPU becomes a deployment target.

## Reproduction

```bash
# Build
cargo build --release --bin strata-generate --features cli,metal

# Run all benchmarks
bash bench_compare.sh

# Run Metal only
bash bench_compare.sh metal

# Run CPU only
bash bench_compare.sh cpu
```
