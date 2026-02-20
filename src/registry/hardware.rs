//! Hardware detection for automatic backend selection.

/// Information about available compute hardware.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub cuda_available: bool,
    pub metal_available: bool,
    pub cpu_available: bool,
    pub recommended_backend: String,
}

impl HardwareInfo {
    /// Detect available hardware backends.
    ///
    /// Probes CUDA > Metal > CPU (same priority as `select_backend`).
    /// Never panics — if probing fails, that backend is marked unavailable.
    pub fn detect() -> Self {
        let cuda_available = Self::probe_cuda();
        let metal_available = Self::probe_metal();

        let recommended_backend = if cuda_available {
            "cuda"
        } else if metal_available {
            "metal"
        } else {
            "cpu"
        }
        .to_string();

        Self {
            cuda_available,
            metal_available,
            cpu_available: true,
            recommended_backend,
        }
    }

    fn probe_cuda() -> bool {
        #[cfg(feature = "cuda")]
        {
            crate::backend::cuda::CudaBackend::try_new().is_ok()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    fn probe_metal() -> bool {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            crate::backend::metal::MetalBackend::try_new().is_ok()
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_does_not_panic() {
        let info = HardwareInfo::detect();
        assert!(info.cpu_available);
        assert!(!info.recommended_backend.is_empty());
    }

    #[test]
    fn test_recommended_backend_is_valid() {
        let info = HardwareInfo::detect();
        assert!(
            ["cuda", "metal", "cpu"].contains(&info.recommended_backend.as_str()),
            "Unexpected backend: {}",
            info.recommended_backend
        );
    }
}
