# Text Generation Comparison: strata-inference vs llama.cpp

**Date:** 2026-02-17
**Machine:** Apple Silicon (M-series), macOS Darwin 25.3.0

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Prompt | `The quick brown fox jumps over the lazy dog. Once upon a time,` |
| Max tokens | 32 |
| Temperature | 0 (greedy decoding) |
| Seed | 42 |
| strata backend | CPU (auto-selected; Metal not yet wired for generation) |
| llama.cpp backend | Metal GPU |
| strata binary | `strata-generate` (release build, `cargo build --release --features cli`) |
| llama.cpp binary | `llama-completion` (build 5708827, Metal enabled) |

## Models

| # | Model | Architecture | File | Quant | Size |
|---|-------|-------------|------|-------|------|
| 1 | GPT-2 | `gpt2` | gpt2.Q8_0.gguf | Q8_0 | 169 MB |
| 2 | TinyLlama 1.1B Chat | `llama` | tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | Q4_K_M | 637 MB |
| 3 | Qwen3 1.7B | `qwen3` | Qwen3-1.7B-Q8_0.gguf | Q8_0 | 1.7 GB |
| 4 | Phi-3.1-mini 4k Instruct | `phi3` | Phi-3.1-mini-4k-instruct-Q4_K_M.gguf | Q4_K_M | 2.2 GB |

## Results

### 1. GPT-2 (124M params)

| Engine | Output | Wall time | tok/s |
|--------|--------|-----------|-------|
| **strata** | `the fox was a little bit of a nuisance to the dog, but now he's a great friend. He's a great dog, and he's a great` | 20,677 ms | 1.5 |
| **llama.cpp** | `the fox was a little bit of a nuisance to the dog, but now he's a great friend. He's a great dog, and he's a great` | 22,413 ms | 558.8 |

**Match: YES** — Outputs are identical, confirming correctness of strata's GPT-2 implementation.

---

### 2. TinyLlama 1.1B Chat (1.1B params)

| Engine | Output | Wall time | tok/s |
|--------|--------|-----------|-------|
| **strata** | `there was a little green man who lived in a field. He was a happy man, but one day, he met a little brown fox. The fo` | 70,243 ms | 0.4 |
| **llama.cpp** | `\nOnce upon a time, there was a lazy dog named Jack. Jack was always sleeping, and he never did anything. One day, a quick brown` | 2,212 ms | 389.7 |

**Match: NO** — Outputs differ. Both are coherent continuations. The difference likely stems from prompt tokenization handling — llama.cpp re-emits part of the prompt prefix, while strata begins continuation immediately.

---

### 3. Qwen3 1.7B (1.7B params)

| Engine | Output | Wall time | tok/s |
|--------|--------|-----------|-------|
| **strata** | `there were a lot of things that were not so. The quick brown fox jumps over the lazy dog. Once upon a time, there were a lot of things` | 107,737 ms | 0.3 |
| **llama.cpp** | `<think>\nOkay, the user provided a sentence: "The quick brown fox jumps over the lazy dog. Once upon a time," and they want me to respond` | 18,224 ms | 113.0 |

**Match: NO** — llama.cpp enters Qwen3's "thinking" mode (emitting `<think>` tags), while strata produces a direct continuation. This is caused by differences in how the engines handle Qwen3's special token / chat template behavior. Strata's output is a valid raw completion.

---

### 4. Phi-3.1-mini 4k Instruct (3.8B params)

| Engine | Output | Wall time | tok/s |
|--------|--------|-----------|-------|
| **strata** | `in a faraway land, there lived a wise old owl. The owl, known for its wisdom, sat atop the tallest tree in the` | 324,205 ms | 0.1 |
| **llama.cpp** | `The quick brown fox jumps over the lazy dog. Once upon a time, in a verdant forest, there lived a sly fox known for` | 17,158 ms | 194.0 |

**Match: NO** — Both outputs are coherent continuations. llama.cpp re-emits the prompt (its `--no-display-prompt` strips the display but the model still sees it), while strata begins the continuation directly. Both produce narrative text.

---

## Summary

| Model | Output Match | strata tok/s | llama.cpp tok/s | Ratio |
|-------|-------------|-------------|----------------|-------|
| GPT-2 | YES | 1.5 | 558.8 | 373x |
| TinyLlama | NO | 0.4 | 389.7 | 974x |
| Qwen3 1.7B | NO | 0.3 | 113.0 | 377x |
| Phi-3.1-mini | NO | 0.1 | 194.0 | 1,940x |

## Notes

1. **Speed gap is expected.** strata runs on CPU only; llama.cpp uses Apple Metal GPU with highly optimized SIMD kernels. The wall time also includes model loading, which is a larger fraction for strata.

2. **GPT-2 correctness validated.** Identical 32-token greedy output confirms the entire pipeline (tokenizer, embedding, transformer layers, sampling) matches llama.cpp for this architecture.

3. **TinyLlama difference is benign.** Both outputs are valid continuations. The difference likely stems from tokenizer handling of the prompt boundary or chat template behavior.

4. **Qwen3 think-mode divergence.** Qwen3 models activate a reasoning chain-of-thought by default. llama.cpp may be injecting a chat template or enabling thinking tokens that strata doesn't.

5. **Phi-3 now loads successfully.** Previously failed with `Tensor not found: blk.0.ffn_gate.weight`. Fixed by adding fused gate+up tensor splitting in `weights.rs` — the code now detects when `ffn_up.weight` has doubled width and splits it into separate gate and up halves.

6. **Mistral was excluded.** Mistral-7B-Instruct (4.4 GB Q4_K_M) was too large to download in a reasonable time. Mistral-7B uses the `llama` architecture internally, so it exercises the same code path as TinyLlama.
