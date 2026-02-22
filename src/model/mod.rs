//! Model loading and transformer runtime.
//!
//! This module provides:
//! - [`config::ModelConfig`] — architecture configuration extracted from GGUF metadata
//! - [`weights::ModelWeights`] — weight tensors loaded from GGUF onto a compute device
//! - [`layer`] — transformer layer forward pass with architecture dispatch

pub mod cache;
pub mod config;
pub mod layer;
pub mod weights;

pub use cache::KvCache;
pub use config::{Activation, ModelArch, ModelConfig, NormType, PoolingType, PositionType};
pub use weights::{LayerWeights, ModelWeights};
