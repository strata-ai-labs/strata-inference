//! Model resolution from CLI `--model` argument.
//!
//! Detects whether the argument is a file path or a registry name,
//! and returns a validated `PathBuf` to a local GGUF file.

use std::path::PathBuf;

use crate::error::InferenceError;

/// Resolve a `--model` argument to a local GGUF file path.
///
/// Detection heuristic:
/// - Contains `/` or `\` → treat as file path
/// - Ends with `.gguf` (case-insensitive) → treat as file path
/// - Otherwise → treat as registry name
pub fn resolve_model(input: &str) -> Result<PathBuf, InferenceError> {
    let path = PathBuf::from(input);

    // Looks like a file path
    let lower = input.to_ascii_lowercase();
    if input.contains('/') || input.contains('\\') || lower.ends_with(".gguf") {
        if !path.exists() {
            return Err(InferenceError::Model(format!(
                "Model file not found: {}",
                path.display()
            )));
        }
        return Ok(path);
    }

    // Try registry
    let registry = crate::registry::ModelRegistry::new();
    registry.resolve(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Mutex;
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    // ===== File-path detection branch =====

    #[test]
    fn test_path_with_slash_returns_model_error() {
        let err = resolve_model("/nonexistent/model.gguf").unwrap_err();
        assert!(
            matches!(err, InferenceError::Model(_)),
            "Expected InferenceError::Model, got: {:?}",
            err
        );
        let msg = format!("{}", err);
        assert!(msg.contains("Model file not found"), "Error: {}", msg);
        assert!(msg.contains("/nonexistent/model.gguf"), "Error: {}", msg);
    }

    #[test]
    fn test_path_with_backslash_returns_model_error() {
        let err = resolve_model("C:\\models\\model.gguf").unwrap_err();
        assert!(matches!(err, InferenceError::Model(_)));
    }

    #[test]
    fn test_gguf_extension_lowercase_returns_model_error() {
        let err = resolve_model("model.gguf").unwrap_err();
        assert!(matches!(err, InferenceError::Model(_)));
    }

    #[test]
    fn test_gguf_extension_uppercase_returns_model_error() {
        // Users on case-insensitive filesystems might type .GGUF
        let err = resolve_model("model.GGUF").unwrap_err();
        assert!(matches!(err, InferenceError::Model(_)));
    }

    #[test]
    fn test_gguf_extension_mixed_case_returns_model_error() {
        let err = resolve_model("model.Gguf").unwrap_err();
        assert!(matches!(err, InferenceError::Model(_)));
    }

    #[test]
    fn test_relative_path_with_slash_returns_model_error() {
        // "./model.gguf" contains '/' → file-path branch
        let err = resolve_model("./model.gguf").unwrap_err();
        assert!(matches!(err, InferenceError::Model(_)));
    }

    #[test]
    fn test_existing_file_path_resolves_successfully() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test-model.gguf");
        std::fs::write(&file_path, b"fake gguf").unwrap();

        let result = resolve_model(file_path.to_str().unwrap()).unwrap();
        assert_eq!(result, file_path);
    }

    #[test]
    fn test_existing_file_without_gguf_extension_resolves_via_slash() {
        // A path like "/tmp/xxx/mymodel" has a '/' so takes the file-path branch
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("mymodel");
        std::fs::write(&file_path, b"fake").unwrap();

        let result = resolve_model(file_path.to_str().unwrap()).unwrap();
        assert_eq!(result, file_path);
    }

    // ===== Registry branch =====

    #[test]
    fn test_unknown_name_returns_registry_error() {
        // A bare name without / or .gguf goes through the registry
        let err = resolve_model("nonexistent-model-xyz").unwrap_err();
        assert!(
            matches!(err, InferenceError::Registry(_)),
            "Expected InferenceError::Registry, got: {:?}",
            err
        );
        let msg = format!("{}", err);
        assert!(msg.contains("Unknown model"), "Error: {}", msg);
    }

    #[test]
    fn test_known_name_not_downloaded_returns_registry_not_found() {
        // "miniLM" is in the catalog but not downloaded in the default dir.
        // This MUST produce "not found locally", NOT "Unknown model".
        let err = resolve_model("miniLM").unwrap_err();
        assert!(matches!(err, InferenceError::Registry(_)));
        let msg = format!("{}", err);
        assert!(
            msg.contains("not found locally"),
            "Expected 'not found locally' for known model, got: {}",
            msg
        );
    }

    #[test]
    fn test_colon_name_not_downloaded_returns_registry_not_found() {
        // "qwen3:8b" has colons but no '/' or '.gguf' → registry branch
        let err = resolve_model("qwen3:8b").unwrap_err();
        assert!(matches!(err, InferenceError::Registry(_)));
        let msg = format!("{}", err);
        assert!(
            msg.contains("not found locally"),
            "Expected 'not found locally' for known model, got: {}",
            msg
        );
    }

    #[test]
    fn test_registry_name_resolves_when_file_exists() {
        let _lock = ENV_MUTEX.lock().unwrap();

        // Point STRATA_MODELS_DIR at a temp dir with the expected file
        let dir = tempfile::tempdir().unwrap();
        let original_env = std::env::var("STRATA_MODELS_DIR").ok();
        std::env::set_var("STRATA_MODELS_DIR", dir.path());

        // Find the expected filename for "miniLM" from the catalog
        let expected_file = {
            let entry = crate::registry::catalog::find_entry("miniLM").unwrap();
            entry.variants[0].hf_file
        };

        // Place the file
        std::fs::write(dir.path().join(expected_file), b"fake gguf").unwrap();

        // Now resolve_model("miniLM") should succeed
        let result = resolve_model("miniLM");
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);
        let path = result.unwrap();
        assert_eq!(path.file_name().unwrap(), expected_file);
        assert!(path.exists());

        // Restore env
        match original_env {
            Some(val) => std::env::set_var("STRATA_MODELS_DIR", val),
            None => std::env::remove_var("STRATA_MODELS_DIR"),
        }
    }

    #[test]
    fn test_registry_colon_name_resolves_when_file_exists() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let dir = tempfile::tempdir().unwrap();
        let original_env = std::env::var("STRATA_MODELS_DIR").ok();
        std::env::set_var("STRATA_MODELS_DIR", dir.path());

        let entry = crate::registry::catalog::find_entry("qwen3:8b").unwrap();
        let expected_file = entry
            .variants
            .iter()
            .find(|v| v.name == entry.default_quant)
            .unwrap()
            .hf_file;

        std::fs::write(dir.path().join(expected_file), b"fake gguf").unwrap();

        let result = resolve_model("qwen3:8b");
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);
        let path = result.unwrap();
        assert_eq!(path.file_name().unwrap(), expected_file);

        match original_env {
            Some(val) => std::env::set_var("STRATA_MODELS_DIR", val),
            None => std::env::remove_var("STRATA_MODELS_DIR"),
        }
    }

    #[test]
    fn test_alias_resolves_when_file_exists() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let dir = tempfile::tempdir().unwrap();
        let original_env = std::env::var("STRATA_MODELS_DIR").ok();
        std::env::set_var("STRATA_MODELS_DIR", dir.path());

        // "tiny-llama" is an alias for "tinyllama"
        let entry = crate::registry::catalog::find_entry("tinyllama").unwrap();
        let expected_file = entry.variants[0].hf_file;
        std::fs::write(dir.path().join(expected_file), b"fake").unwrap();

        let result = resolve_model("tiny-llama");
        assert!(result.is_ok(), "Expected Ok for alias, got: {:?}", result);
        assert_eq!(result.unwrap().file_name().unwrap(), expected_file);

        match original_env {
            Some(val) => std::env::set_var("STRATA_MODELS_DIR", val),
            None => std::env::remove_var("STRATA_MODELS_DIR"),
        }
    }

    // ===== Edge cases =====

    #[test]
    fn test_empty_string_returns_registry_error() {
        let err = resolve_model("").unwrap_err();
        assert!(matches!(err, InferenceError::Registry(_)));
    }

    #[test]
    fn test_whitespace_only_returns_registry_error() {
        let err = resolve_model("  ").unwrap_err();
        assert!(matches!(err, InferenceError::Registry(_)));
    }

    #[test]
    fn test_dot_gguf_alone_is_file_path() {
        // ".gguf" ends with .gguf → file-path branch (not registry)
        let err = resolve_model(".gguf").unwrap_err();
        assert!(matches!(err, InferenceError::Model(_)));
    }
}
