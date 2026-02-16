//! Backend selection from CLI --backend flag.

use std::sync::Arc;

use crate::backend::ComputeBackend;
use crate::error::InferenceError;

/// Resolve a compute backend from the --backend CLI flag.
///
/// Accepted values: "auto" (default), "cpu", "metal", "cuda".
pub fn resolve_backend(name: Option<&str>) -> Result<Arc<dyn ComputeBackend>, InferenceError> {
    match name.unwrap_or("auto") {
        "auto" => Ok(crate::backend::select_backend()),

        "cpu" => Ok(Arc::new(crate::backend::cpu::CpuBackend::new())),

        "metal" => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                Ok(Arc::new(
                    crate::backend::metal::MetalBackend::try_new()?,
                ))
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                Err(InferenceError::Backend(
                    "Metal backend not available (compile with --features metal on macOS)"
                        .to_string(),
                ))
            }
        }

        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Ok(Arc::new(crate::backend::cuda::CudaBackend::try_new()?))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(InferenceError::Backend(
                    "CUDA backend not available (compile with --features cuda)".to_string(),
                ))
            }
        }

        other => Err(InferenceError::Backend(format!(
            "Unknown backend '{}'. Options: auto, cpu, metal, cuda",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_backend_auto() {
        // auto should always succeed (falls back to CPU)
        let result = resolve_backend(Some("auto"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_backend_none_defaults_to_auto() {
        let result = resolve_backend(None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_backend_cpu() {
        let result = resolve_backend(Some("cpu"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_backend_unknown() {
        let result = resolve_backend(Some("tpu"));
        match result {
            Err(InferenceError::Backend(msg)) => {
                assert!(msg.contains("Unknown backend 'tpu'"), "Error: {}", msg);
            }
            Err(other) => panic!("Expected Backend error, got: {:?}", other),
            Ok(_) => panic!("Expected error for unknown backend"),
        }
    }

    #[test]
    fn test_resolve_backend_empty_string() {
        let result = resolve_backend(Some(""));
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_backend_metal_without_feature() {
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            match resolve_backend(Some("metal")) {
                Err(InferenceError::Backend(msg)) => {
                    assert!(msg.contains("Metal backend not available"), "Error: {}", msg);
                }
                Err(other) => panic!("Expected Backend error, got: {:?}", other),
                Ok(_) => panic!("Expected error for unavailable metal backend"),
            }
        }
    }

    #[test]
    fn test_resolve_backend_cuda_without_feature() {
        #[cfg(not(feature = "cuda"))]
        {
            match resolve_backend(Some("cuda")) {
                Err(InferenceError::Backend(msg)) => {
                    assert!(msg.contains("CUDA backend not available"), "Error: {}", msg);
                }
                Err(other) => panic!("Expected Backend error, got: {:?}", other),
                Ok(_) => panic!("Expected error for unavailable cuda backend"),
            }
        }
    }
}
