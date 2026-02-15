# strata-inference

A pure Rust inference engine for transformer models. Built from scratch with zero C/C++ dependencies.

## Goals

- Load and run GGUF quantized models (Q8_0, Q4_0, F16, F32)
- Support both embedding (BERT, Gemma) and generation (LLaMA, Gemma) architectures
- GPU acceleration via Metal (macOS) and CUDA (NVIDIA)
- Completely owned stack — no external inference dependencies

## Architecture

```
strata-inference/
├── src/
│   ├── gguf/          # GGUF v3 binary format parser
│   ├── tokenizer/     # BPE, WordPiece tokenizers (loaded from GGUF)
│   ├── tensor/        # Tensor types, quantization block formats
│   ├── backend/       # ComputeBackend trait + CPU/Metal/CUDA
│   ├── model/         # Transformer block builders (BERT, LLaMA, Gemma)
│   └── engine/        # High-level embed() and generate() APIs
├── Cargo.toml
└── README.md
```

## Status

Under active development. See [GitHub Issues](https://github.com/strata-ai-labs/strata-inference/issues) for the roadmap.

## License

MIT
