# CLAUDE.md — strata-inference

## Project Overview

strata-inference is a **pure Rust inference engine** for transformer models. It loads GGUF quantized models and runs inference on CPU, Metal (macOS), or CUDA (NVIDIA) — with zero C/C++ dependencies. Everything is built from scratch.

This is part of the [Strata](https://github.com/strata-ai-labs) project. The primary consumer is `strata-core`'s intelligence crate, which currently uses a hardcoded MiniLM-L6-v2 BERT model. strata-inference replaces that with a general-purpose engine supporting any GGUF model.

**Target model:** EmbeddingGemma-300M-Q8_0 (308M params, 24 layers, 768-dim, Gemma3 architecture, SentencePiece BPE, RoPE, RMSNorm, SwiGLU)

## Guiding Principles

1. **Completely owned stack** — No runtime dependencies on llama.cpp, ONNX, PyTorch, or any C library. Metal and CUDA are loaded dynamically via FFI at runtime (no link-time dependency).
2. **Correctness first** — Validate every operation against HuggingFace reference outputs. Cosine similarity > 0.99 for embeddings.
3. **Fused quantized operations** — Never dequantize full weight matrices to f32. Dequantization happens inside GPU matmul kernels (fused dequant+dot product).
4. **Architecture-driven** — Model architecture is determined entirely from GGUF metadata at runtime. No hardcoded dimensions, layer counts, or vocab sizes.

## Repository Structure

```
strata-inference/
├── src/
│   ├── gguf/           # GGUF v3 binary parser, quantization blocks
│   │   ├── mod.rs      # GgufFile: header, KV metadata, tensor info
│   │   ├── quant.rs    # BlockQ8_0, BlockQ4_0, dequantization
│   │   └── tensor.rs   # Tensor data loading, mmap
│   ├── tokenizer/      # Tokenizer trait + implementations
│   │   ├── mod.rs      # Tokenizer trait, from_gguf() factory
│   │   ├── bpe.rs      # BPE with priority queue merges
│   │   └── wordpiece.rs # WordPiece (ported from strata-core)
│   ├── tensor/         # Tensor types
│   │   └── mod.rs      # N-dim tensor with dtype, TensorStorage
│   ├── backend/        # Compute backends
│   │   ├── mod.rs      # ComputeBackend trait (ported from strata-core)
│   │   ├── cpu.rs      # CPU backend (ported from strata-core)
│   │   ├── metal/      # Metal backend (ported + new kernels)
│   │   └── cuda/       # CUDA backend (ported + new kernels)
│   ├── model/          # Model loading and transformer runtime
│   │   ├── config.rs   # ModelConfig from GGUF metadata
│   │   ├── weights.rs  # Load tensors by name into LayerWeights
│   │   └── layer.rs    # Transformer layer forward pass
│   └── engine/         # High-level APIs
│       ├── embed.rs    # EmbeddingEngine: text → Vec<f32>
│       └── generate.rs # GenerationEngine: prompt → text (M6)
├── tests/
├── Cargo.toml
├── CLAUDE.md
└── README.md
```

## What to Port from strata-core

The `strata-core/crates/intelligence/src/` directory has 16K LOC, of which **85% is directly portable**:

| Source (strata-core) | Destination (strata-inference) | Action |
|---|---|---|
| `runtime/backend.rs` (261 LOC) | `src/backend/mod.rs` | Port as-is, add new ops |
| `runtime/tensor.rs` (450 LOC) | `src/tensor/mod.rs` | Adapt for N-dim + dtypes |
| `runtime/cpu_backend.rs` (1,245 LOC) | `src/backend/cpu.rs` | Port as-is, add new ops |
| `runtime/metal/mod.rs` (2,094 LOC) | `src/backend/metal/mod.rs` | Port as-is |
| `runtime/metal/ffi.rs` (377 LOC) | `src/backend/metal/ffi.rs` | Port as-is |
| `runtime/metal/kernels.rs` (1,666 LOC) | `src/backend/metal/kernels.rs` | Port + add new kernels |
| `runtime/cuda/mod.rs` (1,812 LOC) | `src/backend/cuda/mod.rs` | Port as-is |
| `runtime/cuda/ffi.rs` (699 LOC) | `src/backend/cuda/ffi.rs` | Port as-is |
| `runtime/cuda/kernels.rs` (1,928 LOC) | `src/backend/cuda/kernels.rs` | Port + add new kernels |
| `embed/tokenizer.rs` (594 LOC) | `src/tokenizer/wordpiece.rs` | Port, adapt for GGUF vocab |
| `embed/model.rs` (1,074 LOC) | **Do NOT port** | Rewrite as config-driven |

**Do NOT port:** `embed/model.rs` — this has hardcoded BERT constants. The new `model/` directory replaces it entirely with architecture-driven code.

## New Operations to Add (beyond strata-core)

These operations exist in strata-core's backends but need new counterparts:

| Operation | CPU | Metal Kernel | CUDA Kernel | Used By |
|---|---|---|---|---|
| `quantized_matmul` (Q8_0 × f32 → f32) | Dequant + matmul | Fused kernel | Fused PTX | All quantized models |
| `rms_norm` (x × rsqrt(mean(x²)+ε) × w) | New | New kernel | New PTX | Gemma, LLaMA |
| `silu` (x × σ(x)) | New | New kernel | New PTX | Gemma, LLaMA |
| `swiglu` (silu(gate) × up) | New | New kernel | New PTX | Gemma, LLaMA |
| `rope` (rotary position embeddings) | New | New kernel | New PTX | Gemma, LLaMA |

## GGUF Format — Key Facts

- **Magic:** `0x46554747` ("GGUF"), little-endian
- **Version:** 3 (current)
- **Alignment:** 32 bytes default (from `general.alignment` key)
- **Strings:** `u64 length` + raw bytes (no null terminator)
- **Arrays:** `element_type: i32` + `count: u64` + data

### Q8_0 Block Format (Primary Target)
```
struct BlockQ8_0 {     // 34 bytes per block of 32 values
    d: f16,            // scale factor (2 bytes)
    qs: [i8; 32],      // quantized values (32 bytes)
}
// Dequant: y[i] = d * qs[i]
```

### Architecture Metadata Keys
```
general.architecture         → "gemma3", "llama", "bert"
{arch}.embedding_length      → hidden size (768 for Gemma-300M)
{arch}.block_count           → num layers (24 for Gemma-300M)
{arch}.attention.head_count  → num heads (16 for Gemma-300M)
{arch}.attention.head_count_kv → KV heads (for GQA)
{arch}.feed_forward_length   → FFN hidden dim
{arch}.attention.layer_norm_rms_epsilon → RMSNorm ε
{arch}.rope.freq_base        → RoPE base frequency
{arch}.attention.causal      → causal vs bidirectional
{arch}.pooling_type          → 0=NONE, 1=MEAN, 2=CLS, 3=LAST
```

### Tensor Name Patterns
```
token_embd.weight              → word embeddings
output_norm.weight             → final normalization
blk.{i}.attn_norm.weight       → pre-attention norm
blk.{i}.attn_q.weight          → Q projection
blk.{i}.attn_k.weight          → K projection
blk.{i}.attn_v.weight          → V projection
blk.{i}.attn_output.weight     → output projection
blk.{i}.ffn_norm.weight        → pre-FFN norm
blk.{i}.ffn_gate.weight        → gate (SwiGLU only)
blk.{i}.ffn_up.weight          → up projection
blk.{i}.ffn_down.weight        → down projection
```

## Architecture Differences Cheat Sheet

| Feature | BERT | Gemma/LLaMA |
|---|---|---|
| Attention | Bidirectional | Causal |
| Normalization | LayerNorm (w + b) | RMSNorm (w only) |
| Activation | GELU | SwiGLU (gate × silu(up)) |
| Position encoding | Learned embeddings | RoPE (rotary) |
| FFN gate tensor | No (`ffn_gate` absent) | Yes (`ffn_gate.weight`) |
| Norm bias | Yes | No |
| Pooling | Mean or CLS | Mean (for embedding) |

## Code Conventions

- **Error handling:** Use `thiserror` for library errors. Return `Result<T, InferenceError>` from public APIs.
- **Logging:** Use `tracing` (not `log`). `info!` for model loading milestones, `debug!` for tensor shapes, `trace!` for per-layer timing.
- **Testing:** Every module needs unit tests. GPU tests should be gated behind feature flags (`#[cfg(feature = "metal")]`).
- **No `unsafe` outside FFI:** All `unsafe` code should be confined to `metal/ffi.rs`, `cuda/ffi.rs`, and mmap operations. Wrap in safe abstractions.
- **Feature flags:** `metal`, `cuda` — both optional. CPU always available.
- **Naming:** Follow Rust conventions. Types are `PascalCase`, functions are `snake_case`. Keep names close to the ML terminology (e.g., `rms_norm`, `swiglu`, `rope`).

## Build & Test

```bash
# CPU only
cargo test

# With Metal (macOS)
cargo test --features metal

# With CUDA (Linux with NVIDIA GPU)
cargo test --features cuda

# All features
cargo test --all-features
```

## Reference Material

The llama.cpp codebase is cloned at `/Users/aniruddhajoshi/Documents/GitHub/llama.cpp/` for reference. Key files:

| What | File |
|---|---|
| GGUF format spec | `ggml/include/gguf.h` (lines 1-46) |
| GGUF parser | `ggml/src/gguf.cpp` (lines 319-739) |
| Quantization blocks | `ggml/src/ggml-common.h` (lines 170-400) |
| Dequantization | `ggml/src/ggml-quants.c` (lines 307-415) |
| Architecture enum | `src/llama-arch.h` (lines 12-534) |
| BERT builder | `src/models/bert.cpp` |
| LLaMA builder | `src/models/llama.cpp` |
| Metal shaders | `ggml/src/ggml-metal/ggml-metal.metal` (8000+ lines) |
| BPE tokenizer | `src/llama-vocab.cpp` (lines 241-671) |
| Metadata keys | `gguf-py/gguf/constants.py` |

## Milestone Order

```
M1 (GGUF Parser) → M2 (Tokenizer) ─┐
                                     ├→ M4 (Transformer) → M5 (Embedding) → M7 (Integration)
M1 (GGUF Parser) → M3 (Backend) ───┘                  └→ M6 (Generation)
```

M1 and M3 can progress in parallel. M2 depends on M1 (reads vocab from GGUF). M4 depends on both M2 and M3. M5 is the primary deliverable. M6 is a stretch goal. M7 wires it into strata-core.

## Common Pitfalls

1. **GGUF strings have no null terminator.** Read exactly `length` bytes.
2. **Tensor offsets are relative to the data section start**, not the file start. Calculate `data_start = ALIGN(end_of_metadata, alignment)` first.
3. **Q8_0 scale is f16, not f32.** Use proper half-float conversion.
4. **RoPE applies to Q and K only**, not V. And only to the first `n_rot` dimensions of each head (often all of them, but check `{arch}.rope.dimension_count`).
5. **SwiGLU doubles the FFN up-projection width.** `ffn_gate` and `ffn_up` have the same shape; the output is `silu(gate(x)) * up(x)`, then projected down.
6. **GQA (Grouped Query Attention):** When `head_count_kv < head_count`, K and V are shared across groups. Repeat K/V heads before computing attention scores.
7. **Metal kernel threadgroup sizes matter.** Q8_0 uses N_R0=2, N_SG=4 (128 threads). Getting this wrong causes incorrect results, not just slowness.
8. **Mean pooling must exclude padding tokens.** Multiply hidden states by attention mask before averaging, then divide by the sum of the mask (not sequence length).
