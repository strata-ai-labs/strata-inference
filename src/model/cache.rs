// M6: KV cache for autoregressive generation.
//
// Stores cached K and V projections for each layer, enabling efficient
// single-token decode steps without recomputing past K/V values.
//
// Supports two storage modes:
// - CPU-resident (Vec<f32>) — used with CpuBackend
// - GPU-resident (DeviceTensor) — used with Metal/CUDA for zero-copy attention

use crate::backend::{ComputeBackend, DeviceTensor};
use crate::error::InferenceError;
use crate::tensor::{Tensor, TensorDtype};
use super::config::ModelConfig;

/// Per-layer KV cache for autoregressive generation.
///
/// Pre-allocates buffers for `max_seq_len` positions across all layers.
/// During generation, new K/V values are appended after each forward step,
/// and the full cached K/V is read for attention computation.
///
/// When created with `new_gpu()`, buffers live on the GPU device and attention
/// can be computed without any CPU-GPU data transfers.
pub struct KvCache {
    /// Per-layer K cache: each Vec has capacity for max_seq_len * kv_dim.
    k_cache: Vec<Vec<f32>>,
    /// Per-layer V cache: each Vec has capacity for max_seq_len * kv_dim.
    v_cache: Vec<Vec<f32>>,
    /// GPU K cache buffers: pre-allocated [max_seq_len, kv_dim] per layer.
    /// `None` for CPU-only mode.
    k_gpu: Option<Vec<DeviceTensor>>,
    /// GPU V cache buffers: pre-allocated [max_seq_len, kv_dim] per layer.
    v_gpu: Option<Vec<DeviceTensor>>,
    /// Number of positions filled so far (same across all layers).
    pos: usize,
    /// Maximum sequence length this cache supports.
    max_seq_len: usize,
    /// KV dimension per position: num_kv_heads * head_dim.
    kv_dim: usize,
    /// Number of transformer layers.
    num_layers: usize,
    /// Data type of the GPU KV cache buffers (F32 or F16).
    kv_dtype: TensorDtype,
}

impl KvCache {
    /// Create a new CPU-resident KV cache pre-allocated for the given model configuration.
    pub fn new(config: &ModelConfig) -> Self {
        let kv_dim = config.num_kv_heads * config.head_dim;
        let num_layers = config.num_layers;
        let max_seq_len = config.max_seq_len;
        let capacity = max_seq_len * kv_dim;

        let k_cache = (0..num_layers).map(|_| Vec::with_capacity(capacity)).collect();
        let v_cache = (0..num_layers).map(|_| Vec::with_capacity(capacity)).collect();

        Self {
            k_cache,
            v_cache,
            k_gpu: None,
            v_gpu: None,
            pos: 0,
            max_seq_len,
            kv_dim,
            num_layers,
            kv_dtype: TensorDtype::F32,
        }
    }

    /// Create a GPU-resident KV cache with pre-allocated F32 device buffers.
    ///
    /// Each layer gets a `[max_seq_len, kv_dim]` buffer for K and V.
    /// CPU vecs are still maintained as a fallback for prefill.
    pub fn new_gpu(config: &ModelConfig, backend: &dyn ComputeBackend) -> Self {
        let kv_dim = config.num_kv_heads * config.head_dim;
        let num_layers = config.num_layers;
        let max_seq_len = config.max_seq_len;
        let capacity = max_seq_len * kv_dim;

        let k_cache: Vec<Vec<f32>> = (0..num_layers).map(|_| Vec::with_capacity(capacity)).collect();
        let v_cache: Vec<Vec<f32>> = (0..num_layers).map(|_| Vec::with_capacity(capacity)).collect();

        // Pre-allocate GPU buffers (zero-filled)
        let zeros = vec![0.0f32; max_seq_len * kv_dim];
        let zero_tensor = Tensor::new(vec![max_seq_len, kv_dim], zeros);

        let k_gpu: Vec<DeviceTensor> = (0..num_layers)
            .map(|_| backend.upload(&zero_tensor))
            .collect();
        let v_gpu: Vec<DeviceTensor> = (0..num_layers)
            .map(|_| backend.upload(&zero_tensor))
            .collect();

        Self {
            k_cache,
            v_cache,
            k_gpu: Some(k_gpu),
            v_gpu: Some(v_gpu),
            pos: 0,
            max_seq_len,
            kv_dim,
            num_layers,
            kv_dtype: TensorDtype::F32,
        }
    }

    /// Create a GPU-resident KV cache with pre-allocated F16 device buffers.
    ///
    /// Halves memory bandwidth for attention compared to F32 cache.
    /// Each layer gets a `[max_seq_len, kv_dim]` buffer at 2 bytes per element.
    /// CPU vecs are still maintained as a fallback for prefill.
    ///
    /// # Safety
    /// Requires the backend to support `create_buffer_empty()` for raw byte allocation.
    /// The graph must use `copy_f32_to_f16` kernel instead of `copy_buffer` for cache writes,
    /// and `grouped_attn_decode_f16` for attention reads.
    pub fn new_gpu_f16(config: &ModelConfig, backend: &dyn ComputeBackend) -> Self {
        let kv_dim = config.num_kv_heads * config.head_dim;
        let num_layers = config.num_layers;
        let max_seq_len = config.max_seq_len;
        let capacity = max_seq_len * kv_dim;

        let k_cache: Vec<Vec<f32>> = (0..num_layers).map(|_| Vec::with_capacity(capacity)).collect();
        let v_cache: Vec<Vec<f32>> = (0..num_layers).map(|_| Vec::with_capacity(capacity)).collect();

        // Pre-allocate F16 GPU buffers: 2 bytes per element instead of 4
        let f16_bytes = max_seq_len * kv_dim * 2;
        let k_gpu: Vec<DeviceTensor> = (0..num_layers)
            .map(|_| backend.create_buffer_empty(f16_bytes, vec![max_seq_len, kv_dim], TensorDtype::F16))
            .collect();
        let v_gpu: Vec<DeviceTensor> = (0..num_layers)
            .map(|_| backend.create_buffer_empty(f16_bytes, vec![max_seq_len, kv_dim], TensorDtype::F16))
            .collect();

        Self {
            k_cache,
            v_cache,
            k_gpu: Some(k_gpu),
            v_gpu: Some(v_gpu),
            pos: 0,
            max_seq_len,
            kv_dim,
            num_layers,
            kv_dtype: TensorDtype::F16,
        }
    }

    /// Whether this cache has GPU-resident buffers.
    pub fn is_gpu(&self) -> bool {
        self.k_gpu.is_some()
    }

    /// Append new K/V values for a single layer (CPU path).
    ///
    /// `k_new` and `v_new` should each contain `n_tokens * kv_dim` elements.
    pub fn append(
        &mut self,
        layer: usize,
        k_new: &[f32],
        v_new: &[f32],
        n_tokens: usize,
    ) -> Result<(), InferenceError> {
        if layer >= self.num_layers {
            return Err(InferenceError::Generation(format!(
                "KV cache layer index {} out of bounds (num_layers={})",
                layer, self.num_layers
            )));
        }
        let expected_len = n_tokens * self.kv_dim;
        if k_new.len() != expected_len || v_new.len() != expected_len {
            return Err(InferenceError::Generation(format!(
                "KV cache append: expected {} elements ({}*{}), got k={} v={}",
                expected_len, n_tokens, self.kv_dim, k_new.len(), v_new.len()
            )));
        }
        if self.pos + n_tokens > self.max_seq_len {
            return Err(InferenceError::Generation(format!(
                "KV cache overflow: pos={} + n_tokens={} > max_seq_len={}",
                self.pos, n_tokens, self.max_seq_len
            )));
        }
        self.k_cache[layer].extend_from_slice(k_new);
        self.v_cache[layer].extend_from_slice(v_new);
        Ok(())
    }

    /// Append new K/V device tensors to GPU cache for a single layer.
    ///
    /// `k_new` and `v_new` should be `[n_tokens, kv_dim]` on the GPU.
    /// Uses `backend.copy_rows_into()` to write into the pre-allocated buffers.
    pub fn append_gpu(
        &mut self,
        layer: usize,
        k_new: &DeviceTensor,
        v_new: &DeviceTensor,
        n_tokens: usize,
        backend: &dyn ComputeBackend,
    ) -> Result<(), InferenceError> {
        if layer >= self.num_layers {
            return Err(InferenceError::Generation(format!(
                "KV cache layer index {} out of bounds (num_layers={})",
                layer, self.num_layers
            )));
        }
        if self.pos + n_tokens > self.max_seq_len {
            return Err(InferenceError::Generation(format!(
                "KV cache overflow: pos={} + n_tokens={} > max_seq_len={}",
                self.pos, n_tokens, self.max_seq_len
            )));
        }

        let k_bufs = self.k_gpu.as_ref().expect("append_gpu requires GPU KV cache");
        let v_bufs = self.v_gpu.as_ref().expect("append_gpu requires GPU KV cache");

        backend.copy_rows_into(&k_bufs[layer], k_new, self.pos);
        backend.copy_rows_into(&v_bufs[layer], v_new, self.pos);

        Ok(())
    }

    /// Get the GPU K cache buffer for a layer.
    ///
    /// The returned tensor is `[max_seq_len, kv_dim]` — the caller must
    /// only read the first `self.len() + n_new` rows.
    pub fn get_k_gpu(&self, layer: usize) -> &DeviceTensor {
        &self.k_gpu.as_ref().expect("get_k_gpu requires GPU KV cache")[layer]
    }

    /// Get the GPU V cache buffer for a layer.
    pub fn get_v_gpu(&self, layer: usize) -> &DeviceTensor {
        &self.v_gpu.as_ref().expect("get_v_gpu requires GPU KV cache")[layer]
    }

    /// Read cached K values for a layer.
    ///
    /// The returned slice contains all appended K data for this layer.
    ///
    /// # Panics
    ///
    /// Panics if `layer >= num_layers`. Use `num_layers()` to check bounds.
    pub fn get_k(&self, layer: usize) -> &[f32] {
        &self.k_cache[layer]
    }

    /// Read cached V values for a layer.
    ///
    /// # Panics
    ///
    /// Panics if `layer >= num_layers`. Use `num_layers()` to check bounds.
    pub fn get_v(&self, layer: usize) -> &[f32] {
        &self.v_cache[layer]
    }

    /// Number of positions filled so far (before the current step's tokens).
    pub fn len(&self) -> usize {
        self.pos
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.pos == 0
    }

    /// Advance the position counter after all layers have processed a step.
    pub fn advance(&mut self, n_tokens: usize) {
        self.pos += n_tokens;
    }

    /// Reset the cache for a new sequence.
    pub fn clear(&mut self) {
        self.pos = 0;
        for k in &mut self.k_cache {
            k.clear();
        }
        for v in &mut self.v_cache {
            v.clear();
        }
        // GPU buffers don't need clearing — we track via `pos` how much is valid
    }

    /// The KV dimension (num_kv_heads * head_dim).
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// The maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// The number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// The data type of GPU KV cache buffers (F32 or F16).
    pub fn kv_dtype(&self) -> TensorDtype {
        self.kv_dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::*;

    fn test_config(num_layers: usize) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Gemma3,
            arch_name: "gemma3".to_string(),
            hidden_size: 8,
            num_layers,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            ffn_hidden: 32,
            vocab_size: 16,
            max_seq_len: 64,
            norm_type: NormType::RMSNorm,
            norm_eps: 1e-6,
            activation: Activation::SwiGLU,
            position_type: PositionType::RoPE,
            rope_freq_base: 10000.0,
            rope_dim: 4,
            rope_neox: false,
            causal: true,
            attn_logit_softcap: 0.0,
            attn_scale: None,
            embedding_scale: 1.0,
            pooling_type: PoolingType::None,
            has_ffn_gate: true,
            has_bias: false,
            pre_norm: true,
        }
    }

    #[test]
    fn test_new_cache() {
        let config = test_config(2);
        let cache = KvCache::new(&config);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.kv_dim(), 8); // 2 heads * 4 dim
        assert_eq!(cache.num_layers(), 2);
        assert_eq!(cache.max_seq_len(), 64);
        assert!(!cache.is_gpu());
    }

    #[test]
    fn test_append_and_get() {
        let config = test_config(1);
        let mut cache = KvCache::new(&config);
        let kv_dim = cache.kv_dim(); // 8

        // Append 2 tokens for layer 0
        let k = vec![1.0f32; 2 * kv_dim];
        let v = vec![2.0f32; 2 * kv_dim];
        cache.append(0, &k, &v, 2).unwrap();
        cache.advance(2);

        assert_eq!(cache.len(), 2);
        assert!(!cache.is_empty());
        assert_eq!(cache.get_k(0).len(), 2 * kv_dim);
        assert_eq!(cache.get_v(0).len(), 2 * kv_dim);
        assert!(cache.get_k(0).iter().all(|&x| x == 1.0));
        assert!(cache.get_v(0).iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_append_sequential() {
        let config = test_config(1);
        let mut cache = KvCache::new(&config);
        let kv_dim = cache.kv_dim();

        // First append: 1 token
        let k1 = vec![1.0f32; kv_dim];
        let v1 = vec![10.0f32; kv_dim];
        cache.append(0, &k1, &v1, 1).unwrap();
        cache.advance(1);
        assert_eq!(cache.len(), 1);

        // Second append: 1 token
        let k2 = vec![2.0f32; kv_dim];
        let v2 = vec![20.0f32; kv_dim];
        cache.append(0, &k2, &v2, 1).unwrap();
        cache.advance(1);
        assert_eq!(cache.len(), 2);

        // Verify combined cache
        let k_all = cache.get_k(0);
        assert_eq!(k_all.len(), 2 * kv_dim);
        assert!(k_all[..kv_dim].iter().all(|&x| x == 1.0));
        assert!(k_all[kv_dim..].iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_clear() {
        let config = test_config(1);
        let mut cache = KvCache::new(&config);
        let kv_dim = cache.kv_dim();

        cache.append(0, &vec![1.0; kv_dim], &vec![2.0; kv_dim], 1).unwrap();
        cache.advance(1);
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.get_k(0).len(), 0);
        assert_eq!(cache.get_v(0).len(), 0);
    }

    #[test]
    fn test_multi_layer() {
        let config = test_config(3);
        let mut cache = KvCache::new(&config);
        let kv_dim = cache.kv_dim();

        // Append to each layer with different values
        for layer in 0..3 {
            let k = vec![(layer + 1) as f32; kv_dim];
            let v = vec![(layer + 10) as f32; kv_dim];
            cache.append(layer, &k, &v, 1).unwrap();
        }
        cache.advance(1);

        // Verify each layer has its own data
        assert!(cache.get_k(0).iter().all(|&x| x == 1.0));
        assert!(cache.get_k(1).iter().all(|&x| x == 2.0));
        assert!(cache.get_k(2).iter().all(|&x| x == 3.0));
        assert!(cache.get_v(0).iter().all(|&x| x == 10.0));
        assert!(cache.get_v(1).iter().all(|&x| x == 11.0));
        assert!(cache.get_v(2).iter().all(|&x| x == 12.0));
    }

    #[test]
    fn test_overflow_returns_error() {
        let mut config = test_config(1);
        config.max_seq_len = 2;
        let mut cache = KvCache::new(&config);
        let kv_dim = cache.kv_dim();

        // Fill to capacity
        cache.append(0, &vec![1.0; 2 * kv_dim], &vec![2.0; 2 * kv_dim], 2).unwrap();
        cache.advance(2);

        // This should fail
        let result = cache.append(0, &vec![3.0; kv_dim], &vec![4.0; kv_dim], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_size_returns_error() {
        let config = test_config(1);
        let mut cache = KvCache::new(&config);

        // Wrong number of elements
        let result = cache.append(0, &[1.0, 2.0], &[3.0, 4.0], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_out_of_bounds_returns_error() {
        let config = test_config(2);
        let mut cache = KvCache::new(&config);
        let kv_dim = cache.kv_dim();

        // Layer index 2 is out of bounds for 2-layer cache (valid: 0, 1)
        let result = cache.append(2, &vec![1.0; kv_dim], &vec![2.0; kv_dim], 1);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("out of bounds"), "Error: {}", err_msg);
    }

    #[test]
    fn test_clear_then_reuse() {
        let config = test_config(1);
        let mut cache = KvCache::new(&config);
        let kv_dim = cache.kv_dim();

        // Fill with some data
        cache.append(0, &vec![1.0; 2 * kv_dim], &vec![2.0; 2 * kv_dim], 2).unwrap();
        cache.advance(2);
        assert_eq!(cache.len(), 2);

        // Clear
        cache.clear();
        assert_eq!(cache.len(), 0);

        // Reuse — should work and contain only the new data
        cache.append(0, &vec![3.0; kv_dim], &vec![4.0; kv_dim], 1).unwrap();
        cache.advance(1);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get_k(0).len(), kv_dim);
        assert!(cache.get_k(0).iter().all(|&x| x == 3.0));
        assert!(cache.get_v(0).iter().all(|&x| x == 4.0));
    }
}
