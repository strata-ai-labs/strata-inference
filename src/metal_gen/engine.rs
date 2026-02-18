//! MetalGenerationEngine — high-level API for graph-based Metal generation.
//!
//! Uses graph-based fast path for both prefill (single command buffer) and
//! single-token decode.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use tracing::info;

use crate::backend::metal::ffi::*;
use crate::backend::metal::MetalBackend;
use crate::backend::metal::MetalBuffer;
use crate::backend::{ComputeBackend, DeviceTensor};
use crate::engine::generate::{GenerationConfig, GenerationOutput, StopReason};
use crate::engine::sampler::{XorShiftRng, sample_token};
use crate::error::InferenceError;
use crate::gguf::GgufFile;
use crate::model::cache::KvCache;
use crate::model::config::ModelConfig;
use crate::model::weights::ModelWeights;
use crate::tensor::TensorDtype;
use crate::tokenizer::{Tokenizer, create_tokenizer_from_gguf};

use super::exec::{DynamicParams, MetalResources, PrefillParams, encode_decode_token, encode_prefill};
use super::graph::{BufferRef, DecodeGraph, PrefillGraph, patch_ops, weight_walk_order};
use super::pool::BufferPool;

/// High-level generation engine using graph-based Metal execution.
///
/// Both prefill and decode use operation graphs encoded into single command
/// buffers. The prefill graph is built per-prompt (sized for M tokens), while
/// the decode graph is built once at init (always M=1).
pub struct MetalGenerationEngine {
    config: ModelConfig,
    weights: ModelWeights,
    tokenizer: Box<dyn Tokenizer>,
    backend: Arc<MetalBackend>,
    metal_res: MetalResources,
    decode_graph: DecodeGraph,
    buffer_pool: BufferPool,
    weight_buf_ids: Vec<Id>,
    kv_dim: usize,
    /// Pre-allocated Metal buffer for the token ID (single u32).
    token_id_buf: Id,
    /// Pre-allocated Metal buffer for the position ID (single u32).
    position_id_buf: Id,
    /// Pre-converted F32 embedding tensors kept alive to prevent buffer deallocation.
    /// The embedding_lookup kernel expects F32 data, but quantized models store
    /// embeddings in Q8_0/Q4_K/etc. We dequantize once at init and patch the graph.
    _f32_emb_tensors: Vec<DeviceTensor>,
    /// F32 token embedding weight index (if token_emb was pre-converted).
    f32_token_emb_idx: Option<u16>,
    /// F32 position embedding weight index (if pos_emb was pre-converted).
    f32_pos_emb_idx: Option<u16>,
}

// MetalGenerationEngine holds Metal pointers that are safe to share.
unsafe impl Send for MetalGenerationEngine {}
unsafe impl Sync for MetalGenerationEngine {}

impl MetalGenerationEngine {
    /// Load a generation engine from a GGUF file path.
    ///
    /// Creates Metal resources, builds the decode graph, and pre-allocates
    /// all intermediate buffers.
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Self, InferenceError> {
        let path = path.as_ref();
        info!(path = %path.display(), "Loading MetalGenerationEngine from GGUF");

        let gguf = GgufFile::open(path)?;
        let config = ModelConfig::from_gguf(&gguf)?;

        if !config.causal {
            return Err(InferenceError::Generation(
                "generation requires a causal model (not bidirectional)".to_string(),
            ));
        }

        // Create Metal backend (also compiles MSL and creates PSOs)
        let backend = Arc::new(MetalBackend::try_new()?);

        // Load weights using the standard path
        let weights = ModelWeights::from_gguf(&gguf, &config, backend.as_ref())?;
        let tokenizer = create_tokenizer_from_gguf(&gguf)?;

        info!(
            arch = %config.arch_name,
            hidden_size = config.hidden_size,
            num_layers = config.num_layers,
            vocab_size = config.vocab_size,
            token_emb_dtype = ?weights.token_embedding.dtype(),
            pos_emb_dtype = ?weights.position_embedding.as_ref().map(|w| w.dtype()),
            layer0_q_dtype = ?weights.layers[0].attn_q.dtype(),
            "Model loaded, building decode graph"
        );

        // Build decode graph with F16 KV cache (mutable — we may patch embedding ops below)
        let mut decode_graph = DecodeGraph::build(&config, &weights, true);
        info!(
            ops = decode_graph.ops.len(),
            barriers = decode_graph.barriers.len(),
            slots = decode_graph.num_slots,
            "Decode graph built"
        );

        // Create Metal resources (device, queue, PSOs) for the fast path.
        // We create a separate set of resources so the fast path doesn't
        // interfere with the backend's command buffer.
        let (device, command_queue) = unsafe {
            let device = MTLCreateSystemDefaultDevice();
            if device == NIL {
                return Err(InferenceError::Backend("No Metal device available".into()));
            }
            msg_send_void(device, sel_registerName(b"retain\0".as_ptr() as _));
            let sels = Selectors::new();
            let queue = msg_send_id(device, sels.new_command_queue);
            if queue == NIL {
                msg_send_void(device, sels.release);
                return Err(InferenceError::Backend("Failed to create command queue".into()));
            }
            (device, queue)
        };

        let metal_res = unsafe { MetalResources::new(device, command_queue)? };

        // Pre-allocate buffer pool
        let buffer_pool = unsafe {
            BufferPool::new(device, &metal_res.sels, &decode_graph.slot_sizes)
        };

        // Extract raw buffer Ids from model weights
        let walked = weight_walk_order(&weights);
        let mut weight_buf_ids: Vec<Id> = walked
            .iter()
            .map(|dt| extract_buffer_id(dt))
            .collect();

        // Pre-convert non-F32 embedding tables to F32.
        // The embedding_lookup kernel reads `device const float*`, so if the
        // embedding table is quantized (Q8_0, etc.) we must dequantize it once
        // at init. We append the F32 buffer as a new weight entry and patch the
        // graph's EmbeddingLookup ops to reference it — the original quantized
        // buffer stays at its original index for the logits projection (tied
        // embeddings) which uses quantized_matmul.
        let mut f32_emb_tensors: Vec<DeviceTensor> = Vec::new();
        let mut f32_token_emb_idx: Option<u16> = None;
        let mut f32_pos_emb_idx: Option<u16> = None;

        // Weight(0) is always token_embedding
        if weights.token_embedding.dtype() != TensorDtype::F32 {
            let f32_tensor = backend.download(&weights.token_embedding).to_f32();
            let f32_dt = backend.upload(&f32_tensor);
            let f32_buf = extract_buffer_id(&f32_dt);
            let new_idx = weight_buf_ids.len() as u16;
            weight_buf_ids.push(f32_buf);
            f32_emb_tensors.push(f32_dt);
            patch_ops(&mut decode_graph.ops, BufferRef::Weight(0), BufferRef::Weight(new_idx));
            f32_token_emb_idx = Some(new_idx);
            info!(
                original_dtype = ?weights.token_embedding.dtype(),
                new_weight_idx = new_idx,
                "Pre-converted token embedding to F32 for graph"
            );
        }

        // Weight(1) is position_embedding (if present)
        if let Some(ref pos_emb) = weights.position_embedding {
            if pos_emb.dtype() != TensorDtype::F32 {
                let f32_tensor = backend.download(pos_emb).to_f32();
                let f32_dt = backend.upload(&f32_tensor);
                let f32_buf = extract_buffer_id(&f32_dt);
                let new_idx = weight_buf_ids.len() as u16;
                weight_buf_ids.push(f32_buf);
                f32_emb_tensors.push(f32_dt);
                patch_ops(&mut decode_graph.ops, BufferRef::Weight(1), BufferRef::Weight(new_idx));
                f32_pos_emb_idx = Some(new_idx);
                info!(
                    original_dtype = ?pos_emb.dtype(),
                    new_weight_idx = new_idx,
                    "Pre-converted position embedding to F32 for graph"
                );
            }
        }

        let kv_dim = config.num_kv_heads * config.head_dim;

        // Pre-allocate tiny buffers for the token ID and position ID
        let token_id_buf = unsafe {
            msg_send_new_buffer_length(
                device,
                metal_res.sels.new_buffer_with_length,
                std::mem::size_of::<u32>(),
                MTL_RESOURCE_STORAGE_MODE_SHARED,
            )
        };
        let position_id_buf = unsafe {
            msg_send_new_buffer_length(
                device,
                metal_res.sels.new_buffer_with_length,
                std::mem::size_of::<u32>(),
                MTL_RESOURCE_STORAGE_MODE_SHARED,
            )
        };

        info!("MetalGenerationEngine initialized");

        Ok(Self {
            config,
            weights,
            tokenizer,
            backend,
            metal_res,
            decode_graph,
            buffer_pool,
            weight_buf_ids,
            kv_dim,
            token_id_buf,
            position_id_buf,
            _f32_emb_tensors: f32_emb_tensors,
            f32_token_emb_idx,
            f32_pos_emb_idx,
        })
    }

    /// Generate text from a prompt.
    pub fn generate(
        &mut self,
        prompt: &str,
        gen_config: &GenerationConfig,
    ) -> Result<String, InferenceError> {
        let output = self.generate_full(prompt, gen_config)?;
        Ok(self.tokenizer.decode(&output.token_ids))
    }

    /// Generate text with full metadata.
    pub fn generate_full(
        &mut self,
        prompt: &str,
        gen_config: &GenerationConfig,
    ) -> Result<GenerationOutput, InferenceError> {
        self.generate_stream_full(prompt, gen_config, |_| true)
    }

    /// Generate with streaming: invokes callback for each generated token.
    pub fn generate_stream_full(
        &mut self,
        prompt: &str,
        gen_config: &GenerationConfig,
        mut callback: impl FnMut(u32) -> bool,
    ) -> Result<GenerationOutput, InferenceError> {
        let profiling = std::env::var("STRATA_PROFILE").map_or(false, |v| v == "1");

        let prompt_ids = self.tokenizer.encode(prompt, true);
        let prompt_tokens = prompt_ids.len();

        if prompt_ids.is_empty() {
            return Err(InferenceError::Generation(
                "prompt tokenized to empty sequence".to_string(),
            ));
        }

        if prompt_ids.len() >= self.config.max_seq_len {
            return Err(InferenceError::Generation(format!(
                "prompt length ({}) exceeds max_seq_len ({})",
                prompt_ids.len(),
                self.config.max_seq_len
            )));
        }

        let mut rng = XorShiftRng::new(gen_config.sampling.seed.unwrap_or(42));
        let mut generated_ids: Vec<u32> = Vec::new();

        let mut stop_tokens = gen_config.stop_tokens.clone();
        if let Some(eos) = self.tokenizer.eos_token_id() {
            if !stop_tokens.contains(&eos) {
                stop_tokens.push(eos);
            }
        }

        // ===================================================================
        // PREFILL (graph-based fast path — single command buffer)
        // ===================================================================
        let prefill_start = Instant::now();

        // Build prefill graph for this prompt's token count.
        // Use the same kv_f16 setting as decode so prefill writes directly
        // into the format decode expects (no F32→F16 conversion needed).
        let kv_f16 = self.decode_graph.kv_f16;
        let mut prefill_graph = PrefillGraph::build(
            &self.config, &self.weights, prompt_ids.len(), kv_f16,
        );

        // Apply F32 embedding patches (same indices as decode graph)
        if let Some(idx) = self.f32_token_emb_idx {
            patch_ops(&mut prefill_graph.ops, BufferRef::Weight(0), BufferRef::Weight(idx));
        }
        if let Some(idx) = self.f32_pos_emb_idx {
            patch_ops(&mut prefill_graph.ops, BufferRef::Weight(1), BufferRef::Weight(idx));
        }

        // Recompute barriers after patching
        prefill_graph.barriers = super::graph::compute_barriers(&prefill_graph.ops);

        // Allocate temporary prefill buffer pool
        let prefill_pool = unsafe {
            BufferPool::new(
                self.metal_res.device,
                &self.metal_res.sels,
                &prefill_graph.slot_sizes,
            )
        };

        // Allocate and fill token IDs buffer (M u32s)
        let token_ids_buf = unsafe {
            let buf = msg_send_new_buffer_length(
                self.metal_res.device,
                self.metal_res.sels.new_buffer_with_length,
                prompt_ids.len() * std::mem::size_of::<u32>(),
                MTL_RESOURCE_STORAGE_MODE_SHARED,
            );
            let ptr = msg_send_ptr(buf, self.metal_res.sels.contents) as *mut u32;
            for (i, &tid) in prompt_ids.iter().enumerate() {
                *ptr.add(i) = tid;
            }
            buf
        };

        // Allocate and fill position IDs buffer [0, 1, ..., M-1]
        let position_ids_buf = unsafe {
            let buf = msg_send_new_buffer_length(
                self.metal_res.device,
                self.metal_res.sels.new_buffer_with_length,
                prompt_ids.len() * std::mem::size_of::<u32>(),
                MTL_RESOURCE_STORAGE_MODE_SHARED,
            );
            let ptr = msg_send_ptr(buf, self.metal_res.sels.contents) as *mut u32;
            for i in 0..prompt_ids.len() {
                *ptr.add(i) = i as u32;
            }
            buf
        };

        // Allocate KV cache (F16 or F32, matching prefill graph)
        let mut cache = if kv_f16 {
            KvCache::new_gpu_f16(&self.config, self.backend.as_ref())
        } else {
            KvCache::new_gpu(&self.config, self.backend.as_ref())
        };
        let kv_buf_ids = self.extract_kv_buf_ids(&cache);

        // Encode and execute (single command buffer!)
        let logits = unsafe {
            encode_prefill(
                &prefill_graph,
                &prefill_pool,
                &self.metal_res,
                &self.weight_buf_ids,
                &kv_buf_ids,
                &PrefillParams {
                    n_tokens: prompt_ids.len(),
                    pos_offset: 0,
                },
                token_ids_buf,
                position_ids_buf,
            )
        };

        // Advance cache position to match prefilled tokens
        cache.advance(prompt_ids.len());

        let mut next_token = sample_token(&logits, &gen_config.sampling, &mut rng);
        let prefill_duration = prefill_start.elapsed();

        if profiling {
            let prefill_ms = prefill_duration.as_secs_f64() * 1000.0;
            eprintln!(
                "[profile] prefill: {:.1}ms ({} tokens, {} ops, 1 cmd buffer)",
                prefill_ms, prompt_tokens, prefill_graph.ops.len(),
            );
        }

        // Release temporary prefill buffers
        unsafe {
            msg_send_void(token_ids_buf, self.metal_res.sels.release);
            msg_send_void(position_ids_buf, self.metal_res.sels.release);
            // prefill_pool is dropped here (its Drop impl releases buffers)
        }

        // ===================================================================
        // DECODE LOOP (graph-based fast path)
        // ===================================================================

        let mut stop_reason = StopReason::MaxTokens;

        for step in 0..gen_config.max_tokens {
            if stop_tokens.contains(&next_token) {
                stop_reason = StopReason::StopToken;
                break;
            }

            generated_ids.push(next_token);

            if !callback(next_token) {
                stop_reason = StopReason::Cancelled;
                break;
            }

            if cache.len() >= self.config.max_seq_len {
                stop_reason = StopReason::ContextLength;
                break;
            }

            let tok_start = if profiling { Some(Instant::now()) } else { None };

            // Fast path: encode all ops into one command buffer
            let dynamic = DynamicParams {
                pos: cache.len(),
                token_id: next_token,
                kv_dim: self.kv_dim,
            };

            let logits = unsafe {
                encode_decode_token(
                    &self.decode_graph,
                    &self.buffer_pool,
                    &self.metal_res,
                    &self.weight_buf_ids,
                    &kv_buf_ids,
                    &dynamic,
                    self.token_id_buf,
                    self.position_id_buf,
                )
            };

            // Advance cache (the graph has already written K/V into the cache buffers)
            cache.advance(1);

            next_token = sample_token(&logits, &gen_config.sampling, &mut rng);

            if let Some(start) = tok_start {
                let total_ms = start.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "[profile] tok {}: {:.1}ms (graph fast path)",
                    step + 1, total_ms,
                );
            }
        }

        Ok(GenerationOutput {
            token_ids: generated_ids,
            stop_reason,
            prompt_tokens,
            prefill_duration,
        })
    }

    /// Extract raw MTLBuffer pointers from KV cache GPU tensors.
    fn extract_kv_buf_ids(&self, cache: &KvCache) -> Vec<Id> {
        let mut ids = Vec::with_capacity(self.config.num_layers * 2);
        for layer in 0..self.config.num_layers {
            ids.push(extract_buffer_id(cache.get_k_gpu(layer)));
            ids.push(extract_buffer_id(cache.get_v_gpu(layer)));
        }
        ids
    }

    /// The model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

impl Drop for MetalGenerationEngine {
    fn drop(&mut self) {
        unsafe {
            let rel = sel_registerName(b"release\0".as_ptr() as _);
            // Release the token_id and position_id buffers
            if self.token_id_buf != NIL {
                msg_send_void(self.token_id_buf, rel);
            }
            if self.position_id_buf != NIL {
                msg_send_void(self.position_id_buf, rel);
            }
            // Release device and command queue owned by metal_res
            if self.metal_res.command_queue != NIL {
                msg_send_void(self.metal_res.command_queue, rel);
            }
            if self.metal_res.device != NIL {
                msg_send_void(self.metal_res.device, rel);
            }
        }
    }
}

/// Extract the raw MTLBuffer Id from a GPU-resident DeviceTensor.
fn extract_buffer_id(dt: &DeviceTensor) -> Id {
    dt.gpu_inner::<MetalBuffer>()
        .expect("expected GPU-resident MetalBuffer in DeviceTensor")
        .buffer
}

