//! Shared CLI utilities for strata-inference binary tools.

pub mod backend;

use std::io::Read;
use std::path::Path;

/// Initialize tracing/logging to stderr.
///
/// If `disable` is true, no output is produced.
/// Otherwise respects `RUST_LOG` env var, defaulting to WARN.
pub fn init_logging(disable: bool) {
    use tracing_subscriber::EnvFilter;

    if disable {
        return;
    }

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}

/// Read input text from one of: prompt string, file path, or stdin.
///
/// Returns an error message string if no input source is provided.
pub fn read_input(
    prompt: Option<&str>,
    file: Option<&Path>,
    use_stdin: bool,
) -> Result<String, String> {
    if let Some(text) = prompt {
        return Ok(text.to_string());
    }

    if let Some(path) = file {
        return std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file '{}': {}", path.display(), e));
    }

    if use_stdin {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("Failed to read stdin: {}", e))?;
        return Ok(buf);
    }

    Err("No input provided. Use --prompt, --file, or --stdin".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_read_input_from_prompt() {
        let result = read_input(Some("hello world"), None, false);
        assert_eq!(result.unwrap(), "hello world");
    }

    #[test]
    fn test_read_input_from_prompt_empty() {
        let result = read_input(Some(""), None, false);
        assert_eq!(result.unwrap(), "");
    }

    #[test]
    fn test_read_input_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test_input.txt");
        std::fs::write(&file_path, "file content here").unwrap();

        let result = read_input(None, Some(&file_path), false);
        assert_eq!(result.unwrap(), "file content here");
    }

    #[test]
    fn test_read_input_from_file_not_found() {
        let result = read_input(None, Some(Path::new("/nonexistent/file.txt")), false);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to read file"));
    }

    #[test]
    fn test_read_input_no_source() {
        let result = read_input(None, None, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No input provided"));
    }

    #[test]
    fn test_read_input_prompt_takes_priority_over_file() {
        // If prompt is provided, file is ignored (clap prevents both,
        // but read_input checks prompt first)
        let result = read_input(Some("from prompt"), Some(Path::new("/nonexistent")), false);
        assert_eq!(result.unwrap(), "from prompt");
    }

    #[test]
    fn test_read_input_from_file_with_unicode() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("unicode.txt");
        let mut f = std::fs::File::create(&file_path).unwrap();
        f.write_all("你好世界 🌍".as_bytes()).unwrap();

        let result = read_input(None, Some(&file_path), false);
        assert_eq!(result.unwrap(), "你好世界 🌍");
    }

    #[test]
    fn test_read_input_from_file_with_newlines() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("multiline.txt");
        std::fs::write(&file_path, "line 1\nline 2\nline 3").unwrap();

        let result = read_input(None, Some(&file_path), false);
        assert_eq!(result.unwrap(), "line 1\nline 2\nline 3");
    }

    #[test]
    fn test_init_logging_disabled_does_not_panic() {
        // Just smoke-test: calling with disable=true should not panic
        init_logging(true);
    }
}
