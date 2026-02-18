//! Graph-based Metal generation engine for high-performance single-token decode.
//!
//! This module builds the entire decode operation list at init time, pre-allocates
//! all intermediate Metal buffers, and encodes ~245 operations into a single
//! command buffer per token — eliminating per-operation allocation, mutex contention,
//! and redundant memory barriers.
//!
//! # Architecture
//!
//! - [`pool`]: Pre-allocated Metal buffer pool (zero allocation during decode)
//! - [`graph`]: Decode operation list builder + static barrier analysis
//! - [`exec`]: Tight Metal encoding loop (the hot path)
//! - [`engine`]: High-level `MetalGenerationEngine` API

mod pool;
mod graph;
mod exec;
mod engine;

pub use engine::MetalGenerationEngine;
