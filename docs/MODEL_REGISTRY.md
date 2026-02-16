# Model Registry & Auto-Download

## Problem

strata-inference requires GGUF model files on disk. Users of Strata (the embedded database) shouldn't need to manually find, download, and place GGUF files from HuggingFace. The engine needs a way to:

1. Map friendly model names to specific GGUF files
2. Download models on first use (or ahead of time)
3. Store them in a known, managed location
4. Verify integrity via checksums

## Design

### Architecture

```
strata-inference/src/
├── registry/
│   ├── mod.rs          # ModelRegistry: lookup, ensure, list
│   ├── catalog.rs      # Static catalog of known models
│   └── download.rs     # HTTP download + progress + checksum
├── bin/
│   └── model.rs        # strata-model CLI binary
```

### Model Catalog

A static, hardcoded table of known-good models. No external config files or network-dependent discovery. Each entry:

```rust
pub struct CatalogEntry {
    /// Friendly short name: "minilm", "embeddinggemma", "bge-m3"
    pub name: &'static str,
    /// Human description
    pub description: &'static str,
    /// HuggingFace repo: "sentence-transformers/all-MiniLM-L6-v2-GGUF"
    pub hf_repo: &'static str,
    /// Filename within the repo: "all-MiniLM-L6-v2.Q8_0.gguf"
    pub hf_filename: &'static str,
    /// SHA-256 of the file (hex string)
    pub sha256: &'static str,
    /// File size in bytes (for progress reporting)
    pub size_bytes: u64,
    /// Model architecture (for display/filtering)
    pub arch: &'static str,
    /// Embedding dimension (0 for generative models)
    pub embedding_dim: u32,
    /// Default quantization
    pub quant: &'static str,
    /// Task type
    pub task: ModelTask,
}

pub enum ModelTask {
    Embedding,
    Generation,
}
```

### Initial Catalog (5 embedding models)

| Name | HF Repo | File | Arch | Dim | Size |
|------|---------|------|------|-----|------|
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2-GGUF` | `all-MiniLM-L6-v2.Q8_0.gguf` | bert | 384 | 24MB |
| `embeddinggemma` | `lmstudio-community/gemma-3-embedding-model-300M-GGUF` | `embeddinggemma-300M-Q8_0.gguf` | gemma3 | 768 | 313MB |
| `nomic-embed` | `nomic-ai/nomic-embed-text-v1.5-GGUF` | `nomic-embed-text-v1.5.Q8_0.gguf` | nomic-bert | 768 | 146MB |
| `bge-m3` | `gpustack/bge-m3-GGUF` | `bge-m3-Q8_0.gguf` | bert | 1024 | 635MB |
| `qwen3-embed` | `Qwen/Qwen3-Embedding-8B-GGUF` | `Qwen3-Embedding-8B-Q8_0.gguf` | qwen3 | 4096 | 8.05GB |

Note: `nomic-embed` (nomic-bert arch) and `qwen3-embed` (qwen3 arch) are not yet supported by strata-inference's model runner. They are included in the catalog for forward compatibility — the registry can download them, but `EmbeddingEngine` will fail at load time until those architectures are implemented.

### Storage Layout

```
~/.strata/models/
├── all-MiniLM-L6-v2.Q8_0.gguf
├── embeddinggemma-300M-Q8_0.gguf
├── nomic-embed-text-v1.5.Q8_0.gguf
├── bge-m3-Q8_0.gguf
└── Qwen3-Embedding-8B-Q8_0.gguf
```

`STRATA_MODELS_DIR` env var overrides the default `~/.strata/models/`. This is useful for:
- CI/CD (use a shared cache directory)
- Docker (mount a volume)
- Custom installs

### Library API

```rust
use strata_inference::registry::ModelRegistry;

// List available models
let models = ModelRegistry::list();
for m in &models {
    println!("{}: {} ({})", m.name, m.description, m.arch);
}

// Get path to a model, downloading if needed
let path: PathBuf = ModelRegistry::ensure("minilm")?;
// Returns ~/.strata/models/all-MiniLM-L6-v2.Q8_0.gguf
// Downloads from HF if not already present
// Verifies SHA-256 after download

// Check if a model is already downloaded
let available: bool = ModelRegistry::is_downloaded("minilm");

// Get path without downloading (returns None if not present)
let path: Option<PathBuf> = ModelRegistry::path("minilm");

// Delete a downloaded model
ModelRegistry::remove("minilm")?;
```

### Integration with EmbeddingEngine

```rust
// Before (user must know the path):
let engine = EmbeddingEngine::from_gguf("path/to/model.gguf", backend)?;

// After (by name — downloads automatically):
let engine = EmbeddingEngine::from_registry("minilm", backend)?;

// Which is equivalent to:
let path = ModelRegistry::ensure("minilm")?;
let engine = EmbeddingEngine::from_gguf(&path, backend)?;
```

### CLI: `strata-model`

```
strata-model list                    # Show all catalog models + download status
strata-model download <name>         # Download a specific model
strata-model download --all          # Download all models
strata-model path <name>             # Print the local path (for scripts)
strata-model remove <name>           # Delete a downloaded model
strata-model info <name>             # Show model details (arch, dim, size, etc.)
```

Example output of `strata-model list`:

```
Available models:

  NAME              ARCH     DIM   SIZE    STATUS
  minilm            bert     384   24MB    downloaded
  embeddinggemma    gemma3   768   313MB   downloaded
  nomic-embed       nomic    768   146MB   not downloaded
  bge-m3            bert     1024  635MB   not downloaded
  qwen3-embed       qwen3    4096  8.1GB   not downloaded

Models directory: ~/.strata/models/
```

### Download Implementation

Requirements:
- HTTP GET from HuggingFace CDN (`https://huggingface.co/{repo}/resolve/main/{filename}`)
- Progress bar (bytes downloaded / total, speed, ETA)
- Resume interrupted downloads (HTTP Range header)
- SHA-256 verification after download
- Atomic writes: download to `.tmp` file, rename on success
- No dependency on `huggingface_hub` Python — pure Rust HTTP

Dependencies:
- `ureq` (or `reqwest` with blocking) for HTTP — minimal, no async runtime needed
- `sha2` for SHA-256
- `indicatif` for progress bars (CLI only, behind `cli` feature flag)

```rust
// download.rs sketch
pub fn download_model(entry: &CatalogEntry, dest_dir: &Path) -> Result<PathBuf> {
    let dest = dest_dir.join(entry.hf_filename);
    if dest.exists() {
        // Verify checksum of existing file
        if verify_sha256(&dest, entry.sha256)? {
            return Ok(dest);
        }
        // Corrupted — re-download
        std::fs::remove_file(&dest)?;
    }

    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        entry.hf_repo, entry.hf_filename
    );

    let tmp = dest.with_extension("gguf.tmp");
    download_with_progress(&url, &tmp, entry.size_bytes)?;

    // Verify
    if !verify_sha256(&tmp, entry.sha256)? {
        std::fs::remove_file(&tmp)?;
        return Err(InferenceError::ChecksumMismatch { ... });
    }

    std::fs::rename(&tmp, &dest)?;
    Ok(dest)
}
```

### Integration with strata-core

strata-core's intelligence crate currently uses a hardcoded MiniLM path. With the registry:

```rust
// In strata-core intelligence crate
use strata_inference::registry::ModelRegistry;
use strata_inference::engine::EmbeddingEngine;

pub fn init_embedding_engine() -> Result<EmbeddingEngine> {
    let model_path = ModelRegistry::ensure("minilm")?;
    let backend = strata_inference::backend::select_backend();
    EmbeddingEngine::from_gguf(&model_path, backend)
}
```

First call downloads MiniLM (~24MB, takes ~2 seconds). Subsequent calls are instant.

### Error Handling

```rust
pub enum RegistryError {
    /// Model name not found in catalog
    UnknownModel(String),
    /// Network error during download
    DownloadFailed { url: String, source: Box<dyn Error> },
    /// SHA-256 mismatch after download (corrupted or tampered)
    ChecksumMismatch { expected: String, actual: String },
    /// Cannot create models directory
    StorageError(std::io::Error),
    /// Model downloaded but architecture not yet supported
    UnsupportedArch { model: String, arch: String },
}
```

### Feature Flags

```toml
[features]
default = []
registry = ["ureq", "sha2"]        # Model download support
cli = ["clap", "registry", "indicatif"]  # CLI binaries
```

The `registry` feature is opt-in. If strata-core only needs to use pre-downloaded models (e.g., bundled in a Docker image), it can skip the download dependency entirely and just use `ModelRegistry::path()`.

### Security Considerations

- SHA-256 verification prevents tampering and corrupted downloads
- HTTPS only — no HTTP fallback
- No arbitrary URL downloads — only from the hardcoded HuggingFace repos in the catalog
- Models directory permissions: created with 0755, files with 0644
- `.tmp` files cleaned up on failure (no partial files left behind)

### Future Extensions

1. **Multiple quantizations**: `ModelRegistry::ensure_with_quant("minilm", Quant::Q4_0)` — catalog entries for Q4_0, Q8_0, F16 variants
2. **Custom models**: `ModelRegistry::register(name, url, sha256)` — user-added models stored in a local JSON config
3. **Model aliases**: `"default-embed" -> "minilm"` — configurable default model
4. **Disk space management**: `ModelRegistry::prune(max_bytes)` — remove least-recently-used models to stay within budget
5. **Version pinning**: Catalog entries include model version, allowing safe updates

### Implementation Order

1. `catalog.rs` — static catalog table with the 5 embedding models
2. `mod.rs` — `list()`, `path()`, `is_downloaded()` (no network, just filesystem)
3. `download.rs` — HTTP download with progress + SHA-256 verification
4. `mod.rs` — `ensure()`, `remove()`
5. `src/bin/model.rs` — CLI binary
6. `engine/embed.rs` — `from_registry()` convenience method
7. Wire into strata-core
