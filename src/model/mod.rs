//! Model loading and transformer runtime.
//!
//! This module provides:
//! - [`config::ModelConfig`] — architecture configuration extracted from GGUF metadata
//! - [`weights::ModelWeights`] — weight tensors loaded from GGUF onto a compute device
//! - [`layer`] — transformer layer forward pass with architecture dispatch

pub mod config;
pub mod weights;
pub mod layer;
pub mod cache;

pub use config::{ModelConfig, ModelArch, NormType, Activation, PositionType, PoolingType};
pub use weights::{ModelWeights, LayerWeights};
pub use cache::KvCache;
