use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("GGUF parse error: {0}")]
    GgufParse(String),

    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    #[error("Invalid magic number: expected 0x46554747, got 0x{0:08X}")]
    InvalidMagic(u32),

    #[error("Missing metadata key: {0}")]
    MissingKey(String),

    #[error("Type mismatch for key '{key}': expected {expected}, got {actual}")]
    TypeMismatch {
        key: String,
        expected: String,
        actual: String,
    },

    #[error("Unknown tensor type: {0}")]
    UnknownTensorType(u32),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Model error: {0}")]
    Model(String),
}
