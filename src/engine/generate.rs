//! Generation engine: prompt → text via autoregressive decoding.
//!
//! [`GenerationEngine`] provides a high-level API for text generation from
//! any supported causal GGUF model (Gemma, LLaMA, etc.) using a KV cache
//! for efficient token-by-token decoding.
//!
//! On Metal, the engine uses graph-based execution (single command buffer per
//! token) for maximum throughput. On CPU, it falls back to per-op dispatch
//! via `model_forward_step`.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::info;

use crate::backend::{ComputeBackend, DeviceTensor};
use crate::error::InferenceError;
use crate::gguf::GgufFile;
use crate::model::cache::KvCache;
use crate::model::config::ModelConfig;
use crate::model::layer::{linear_forward, model_forward_step};
use crate::model::weights::ModelWeights;
use crate::tensor::Tensor;
use crate::tokenizer::{create_tokenizer_from_gguf, Tokenizer};

use super::sampler::{SamplingConfig, XorShiftRng, sample_token};

#[cfg(all(feature = "metal", target_os = "macos"))]
use super::graph::{BufferRef, DecodeGraph, PrefillGraph, compute_barriers, patch_ops, weight_walk_order};
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::backend::metal::ffi::*;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::backend::metal::{MetalBackend, MetalBuffer};
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::tensor::TensorDtype;
#[cfg(all(feature = "metal", target_os = "macos"))]
use super::exec_metal::{
    DynamicParams, MetalBufferPool, MetalResources, PrefillParams,
    encode_decode_token, encode_prefill,
    encode_decode_token_profiled, encode_prefill_profiled,
    format_kernel_breakdown,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
use super::graph::PsoRef;

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Generated an end-of-sequence or explicit stop token.
    StopToken,
    /// Reached the `max_tokens` limit.
    MaxTokens,
    /// Filled the model's context window.
    ContextLength,
    /// Streaming callback returned `false`.
    Cancelled,
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopReason::StopToken => write!(f, "eos"),
            StopReason::MaxTokens => write!(f, "max_tokens"),
            StopReason::ContextLength => write!(f, "context_length"),
            StopReason::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Output from text generation, including metadata.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    /// Generated token IDs (not including the prompt).
    pub token_ids: Vec<u32>,
    /// Why generation stopped.
    pub stop_reason: StopReason,
    /// Number of prompt tokens (for timing calculations).
    pub prompt_tokens: usize,
    /// Time spent on the prefill forward pass (excludes tokenization and setup).
    pub prefill_duration: Duration,
}

/// Configuration for text generation.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate (excluding the prompt).
    pub max_tokens: usize,
    /// Stop generation when any of these token IDs are produced.
    pub stop_tokens: Vec<u32>,
    /// Sampling parameters (temperature, top-k, top-p).
    pub sampling: SamplingConfig,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            stop_tokens: Vec::new(),
            sampling: SamplingConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Executor — backend-specific execution strategy
// ---------------------------------------------------------------------------

/// Metal graph-based executor.
#[cfg(all(feature = "metal", target_os = "macos"))]
struct MetalExecutor {
    backend: Arc<MetalBackend>,
    metal_res: MetalResources,
    buffer_pool: MetalBufferPool,
    weight_buf_ids: Vec<Id>,
    kv_dim: usize,
    token_id_buf: Id,
    position_id_buf: Id,
    /// Pre-converted F32 embedding tensors kept alive to prevent buffer deallocation.
    _f32_emb_tensors: Vec<DeviceTensor>,
    f32_token_emb_idx: Option<u16>,
    f32_pos_emb_idx: Option<u16>,
    /// LongRoPE runtime factor swapping: index of rope_factors_short in weight_buf_ids.
    rope_short_weight_idx: Option<u16>,
    /// Metal buffer ID for rope_factors_long (to swap in when seq_len > original_ctx).
    rope_long_buf_id: Option<Id>,
    /// Original Metal buffer ID for rope_factors_short (to restore when seq_len < original_ctx).
    rope_short_buf_id: Option<Id>,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
unsafe impl Send for MetalExecutor {}
#[cfg(all(feature = "metal", target_os = "macos"))]
unsafe impl Sync for MetalExecutor {}

/// CPU executor using per-op dispatch via ComputeBackend.
struct CpuExecutor {
    backend: Arc<dyn ComputeBackend>,
}

/// Backend-specific execution strategy.
enum Executor {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(MetalExecutor),
    Cpu(CpuExecutor),
}

// ---------------------------------------------------------------------------
// GenerationEngine
// ---------------------------------------------------------------------------

/// High-level generation engine for autoregressive text generation.
///
/// Wraps a GGUF model with its tokenizer, compute backend, and KV cache
/// to provide `generate(prompt) -> String` and streaming APIs.
///
/// On Metal, uses graph-based execution (single command buffer per token)
/// for maximum throughput. On CPU, falls back to per-op `model_forward_step`.
pub struct GenerationEngine {
    config: ModelConfig,
    weights: ModelWeights,
    tokenizer: Box<dyn Tokenizer>,
    executor: Executor,
    /// Decode graph (Metal path only).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    decode_graph: Option<DecodeGraph>,
    /// Whether KV cache uses F16 (Metal path only).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    kv_f16: bool,
}

impl GenerationEngine {
    /// Load a generation engine from a GGUF file, auto-selecting the best backend.
    ///
    /// Prefers Metal > CUDA > CPU.
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Self, InferenceError> {
        // Auto-select: try Metal first, then fall back to CPU
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            match Self::from_gguf_metal(path.as_ref()) {
                Ok(engine) => return Ok(engine),
                Err(e) => {
                    info!(error = %e, "Metal init failed, falling back to CPU");
                }
            }
        }

        Self::from_gguf_cpu(path.as_ref())
    }

    /// Load with an explicit backend selection.
    pub fn from_gguf_with_backend(
        path: impl AsRef<Path>,
        backend: &str,
    ) -> Result<Self, InferenceError> {
        Self::from_gguf_with_options(path, backend, None)
    }

    /// Load with an explicit backend selection and optional context length override.
    pub fn from_gguf_with_options(
        path: impl AsRef<Path>,
        backend: &str,
        ctx_size: Option<usize>,
    ) -> Result<Self, InferenceError> {
        let path = path.as_ref();
        match backend {
            "auto" => Self::from_gguf(path),
            "cpu" => Self::from_gguf_cpu(path),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            "metal" => Self::from_gguf_metal_with_ctx(path, ctx_size),
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            "metal" => Err(InferenceError::Backend(
                "Metal backend not available (compile with --features metal on macOS)".to_string(),
            )),
            other => Err(InferenceError::Backend(format!(
                "Unknown backend '{}'. Options: auto, cpu, metal", other
            ))),
        }
    }

    /// Build a CPU-backed generation engine.
    fn from_gguf_cpu(path: &Path) -> Result<Self, InferenceError> {
        info!(path = %path.display(), "Loading generation engine (CPU) from GGUF");

        let gguf = GgufFile::open(path)?;
        let config = ModelConfig::from_gguf(&gguf)?;

        if !config.causal {
            return Err(InferenceError::Generation(
                "generation requires a causal model (not bidirectional)".to_string(),
            ));
        }

        let backend: Arc<dyn ComputeBackend> =
            Arc::new(crate::backend::cpu::CpuBackend::new());
        let weights = ModelWeights::from_gguf(&gguf, &config, backend.as_ref())?;
        let tokenizer = create_tokenizer_from_gguf(&gguf)?;

        info!(
            arch = %config.arch_name,
            hidden_size = config.hidden_size,
            vocab_size = config.vocab_size,
            max_seq_len = config.max_seq_len,
            "CPU generation engine loaded"
        );

        Ok(Self {
            config,
            weights,
            tokenizer,
            executor: Executor::Cpu(CpuExecutor { backend }),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            decode_graph: None,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            kv_f16: false,
        })
    }

    /// Build a Metal graph-based generation engine.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn from_gguf_metal(path: &Path) -> Result<Self, InferenceError> {
        Self::from_gguf_metal_with_ctx(path, None)
    }

    /// Build a Metal graph-based generation engine with an optional context length override.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn from_gguf_metal_with_ctx(path: &Path, ctx_override: Option<usize>) -> Result<Self, InferenceError> {
        info!(path = %path.display(), "Loading generation engine (Metal) from GGUF");

        let gguf = GgufFile::open(path)?;
        let mut config = ModelConfig::from_gguf(&gguf)?;

        // Cap max_seq_len to avoid OOM from huge KV caches (e.g., Phi-3.5 has 131072)
        let default_ctx = 4096;
        if let Some(ctx) = ctx_override {
            if ctx < config.max_seq_len {
                info!(
                    original = config.max_seq_len,
                    capped = ctx,
                    "Capping max_seq_len to user-specified context size"
                );
                config.max_seq_len = ctx;
            }
        } else if config.max_seq_len > default_ctx {
            info!(
                original = config.max_seq_len,
                capped = default_ctx,
                "Capping max_seq_len to default (use --ctx to override)"
            );
            config.max_seq_len = default_ctx;
        }

        if !config.causal {
            return Err(InferenceError::Generation(
                "generation requires a causal model (not bidirectional)".to_string(),
            ));
        }

        let backend = Arc::new(MetalBackend::try_new()?);
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

        // Build decode graph with F16 KV cache
        let kv_f16 = true;
        let mut decode_graph = DecodeGraph::build(&config, &weights, kv_f16);
        info!(
            ops = decode_graph.ops.len(),
            barriers = decode_graph.barriers.len(),
            slots = decode_graph.num_slots,
            "Decode graph built"
        );

        // Create Metal resources (device, queue, PSOs)
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
            MetalBufferPool::new(device, &metal_res.sels, &decode_graph.slot_sizes)
        };

        // Extract raw buffer Ids from model weights
        let walked = weight_walk_order(&weights);
        let mut weight_buf_ids: Vec<Id> = walked
            .iter()
            .map(|dt| extract_buffer_id(dt))
            .collect();

        // Pre-convert non-F32 embedding tables to F32
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

        // Compute LongRoPE factor indices in weight_buf_ids
        let (rope_short_weight_idx, rope_long_buf_id, rope_short_buf_id) = {
            if weights.rope_factors_short.is_some() {
                // Compute index of rope_factors_short in walk order:
                // token_embedding(1) + position_embedding? + output_norm_w? + output_norm_b?
                // + output_projection? = offset for rope_factors_short
                let mut idx: u16 = 1; // token_embedding always at 0
                if weights.position_embedding.is_some() { idx += 1; }
                if weights.output_norm_w.is_some() { idx += 1; }
                if weights.output_norm_b.is_some() { idx += 1; }
                if weights.output_projection.is_some() { idx += 1; }
                let short_buf = weight_buf_ids[idx as usize];
                let long_buf = if weights.rope_factors_long.is_some() {
                    Some(weight_buf_ids[(idx + 1) as usize])
                } else {
                    None
                };
                (Some(idx), long_buf, Some(short_buf))
            } else {
                (None, None, None)
            }
        };

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

        info!("Metal generation engine initialized");

        Ok(Self {
            config,
            weights,
            tokenizer,
            executor: Executor::Metal(MetalExecutor {
                backend,
                metal_res,
                buffer_pool,
                weight_buf_ids,
                kv_dim,
                token_id_buf,
                position_id_buf,
                _f32_emb_tensors: f32_emb_tensors,
                f32_token_emb_idx,
                f32_pos_emb_idx,
                rope_short_weight_idx,
                rope_long_buf_id,
                rope_short_buf_id,
            }),
            decode_graph: Some(decode_graph),
            kv_f16,
        })
    }

    /// Crate-internal constructor for testing with synthetic components.
    #[cfg(test)]
    pub(crate) fn new(
        config: ModelConfig,
        weights: ModelWeights,
        tokenizer: Box<dyn Tokenizer>,
        backend: Arc<dyn ComputeBackend>,
    ) -> Self {
        Self {
            config,
            weights,
            tokenizer,
            executor: Executor::Cpu(CpuExecutor { backend }),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            decode_graph: None,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            kv_f16: false,
        }
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

    /// Generate text with full metadata (stop reason, prompt token count).
    pub fn generate_full(
        &mut self,
        prompt: &str,
        gen_config: &GenerationConfig,
    ) -> Result<GenerationOutput, InferenceError> {
        self.generate_stream_full(prompt, gen_config, |_| true)
    }

    /// Generate text with streaming: invokes callback for each generated token.
    ///
    /// The callback receives each token ID as it's generated. Return `false`
    /// from the callback to stop generation early.
    pub fn generate_stream(
        &mut self,
        prompt: &str,
        gen_config: &GenerationConfig,
        callback: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>, InferenceError> {
        let output = self.generate_stream_full(prompt, gen_config, callback)?;
        Ok(output.token_ids)
    }

    /// Generate with streaming and full metadata.
    pub fn generate_stream_full(
        &mut self,
        prompt: &str,
        gen_config: &GenerationConfig,
        callback: impl FnMut(u32) -> bool,
    ) -> Result<GenerationOutput, InferenceError> {
        match &self.executor {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Executor::Metal(_) => self.generate_stream_full_metal(prompt, gen_config, callback),
            Executor::Cpu(_) => self.generate_stream_full_cpu(prompt, gen_config, callback),
        }
    }

    /// CPU generation path using model_forward_step.
    fn generate_stream_full_cpu(
        &self,
        prompt: &str,
        gen_config: &GenerationConfig,
        mut callback: impl FnMut(u32) -> bool,
    ) -> Result<GenerationOutput, InferenceError> {
        let backend = match &self.executor {
            Executor::Cpu(cpu) => &cpu.backend,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            _ => unreachable!(),
        };

        if !self.config.causal {
            return Err(InferenceError::Generation(
                "generation requires a causal model (not bidirectional)".to_string(),
            ));
        }

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
        let mut cache = if backend.is_gpu() {
            KvCache::new_gpu(&self.config, backend.as_ref())
        } else {
            KvCache::new(&self.config)
        };
        let mut generated_ids: Vec<u32> = Vec::new();

        let mut stop_tokens = gen_config.stop_tokens.clone();
        if let Some(eos) = self.tokenizer.eos_token_id() {
            if !stop_tokens.contains(&eos) {
                stop_tokens.push(eos);
            }
        }

        // Prefill
        if profiling {
            backend.reset_profile();
        }
        let prefill_start = Instant::now();
        let hidden = model_forward_step(
            &prompt_ids,
            &self.weights,
            &self.config,
            backend.as_ref(),
            &mut cache,
        )?;

        let logits = self.project_to_logits_cpu(&hidden, prompt_ids.len() - 1, backend.as_ref());
        let mut next_token = sample_token(&logits, &gen_config.sampling, &mut rng);
        let prefill_duration = prefill_start.elapsed();

        if profiling {
            let prefill_ms = prefill_duration.as_secs_f64() * 1000.0;
            eprintln!(
                "[profile] prefill: {:.1}ms ({} tokens) | {}",
                prefill_ms, prompt_tokens, backend.profile_summary(),
            );
        }

        // Decode loop
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

            if profiling {
                backend.reset_profile();
            }
            let tok_start = Instant::now();

            let hidden = model_forward_step(
                &[next_token],
                &self.weights,
                &self.config,
                backend.as_ref(),
                &mut cache,
            )?;
            let fwd_ms = tok_start.elapsed().as_secs_f64() * 1000.0;

            let logits_start = Instant::now();
            let logits = self.project_to_logits_cpu(&hidden, 0, backend.as_ref());
            let logits_ms = logits_start.elapsed().as_secs_f64() * 1000.0;

            let sample_start = Instant::now();
            next_token = sample_token(&logits, &gen_config.sampling, &mut rng);
            let sample_ms = sample_start.elapsed().as_secs_f64() * 1000.0;

            if profiling {
                let total_ms = tok_start.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "[profile] tok {}: {:.1}ms (fwd={:.1}ms logits={:.1}ms sample={:.1}ms) | {}",
                    step + 1, total_ms, fwd_ms, logits_ms, sample_ms,
                    backend.profile_summary(),
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

    /// Metal generation path using graph-based execution.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn generate_stream_full_metal(
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

        // Build prefill graph for this prompt's token count
        let kv_f16 = self.kv_f16;
        let mut prefill_graph = PrefillGraph::build(
            &self.config, &self.weights, prompt_ids.len(), kv_f16,
        );
        if profiling {
            eprintln!("[profile-detail] graph build: {:.1}ms", prefill_start.elapsed().as_secs_f64() * 1000.0);
        }

        // LongRoPE: select long vs short factors based on configured context size (n_ctx).
        // llama.cpp uses the configured n_ctx, not the current sequence position.
        if self.config.rope_scaling_original_ctx > 0 {
            if let Executor::Metal(ref mut m) = self.executor {
                if let (Some(si), Some(li), Some(sbi)) =
                    (m.rope_short_weight_idx, m.rope_long_buf_id, m.rope_short_buf_id)
                {
                    if self.config.max_seq_len > self.config.rope_scaling_original_ctx {
                        m.weight_buf_ids[si as usize] = li;
                    } else {
                        m.weight_buf_ids[si as usize] = sbi;
                    }
                }
            }
        }

        // Borrow executor fields
        let metal = match &self.executor {
            Executor::Metal(m) => m,
            _ => unreachable!(),
        };

        // Apply F32 embedding patches (same indices as decode graph)
        if let Some(idx) = metal.f32_token_emb_idx {
            patch_ops(&mut prefill_graph.ops, BufferRef::Weight(0), BufferRef::Weight(idx));
        }
        if let Some(idx) = metal.f32_pos_emb_idx {
            patch_ops(&mut prefill_graph.ops, BufferRef::Weight(1), BufferRef::Weight(idx));
        }

        // Recompute barriers after patching
        prefill_graph.barriers = compute_barriers(&prefill_graph.ops);
        if profiling {
            eprintln!("[profile-detail] patch+barrier: {:.1}ms", prefill_start.elapsed().as_secs_f64() * 1000.0);
        }

        // Allocate temporary prefill buffer pool
        let prefill_pool = unsafe {
            MetalBufferPool::new(
                metal.metal_res.device,
                &metal.metal_res.sels,
                &prefill_graph.slot_sizes,
            )
        };

        // Allocate and fill token IDs buffer (M u32s)
        let token_ids_buf = unsafe {
            let buf = msg_send_new_buffer_length(
                metal.metal_res.device,
                metal.metal_res.sels.new_buffer_with_length,
                prompt_ids.len() * std::mem::size_of::<u32>(),
                MTL_RESOURCE_STORAGE_MODE_SHARED,
            );
            let ptr = msg_send_ptr(buf, metal.metal_res.sels.contents) as *mut u32;
            for (i, &tid) in prompt_ids.iter().enumerate() {
                *ptr.add(i) = tid;
            }
            buf
        };

        // Allocate and fill position IDs buffer [0, 1, ..., M-1]
        let position_ids_buf = unsafe {
            let buf = msg_send_new_buffer_length(
                metal.metal_res.device,
                metal.metal_res.sels.new_buffer_with_length,
                prompt_ids.len() * std::mem::size_of::<u32>(),
                MTL_RESOURCE_STORAGE_MODE_SHARED,
            );
            let ptr = msg_send_ptr(buf, metal.metal_res.sels.contents) as *mut u32;
            for i in 0..prompt_ids.len() {
                *ptr.add(i) = i as u32;
            }
            buf
        };

        // Allocate KV cache (F16 or F32, matching prefill graph)
        let mut cache = if kv_f16 {
            KvCache::new_gpu_f16(&self.config, metal.backend.as_ref())
        } else {
            KvCache::new_gpu(&self.config, metal.backend.as_ref())
        };
        let kv_buf_ids = Self::extract_kv_buf_ids_metal(&self.config, &cache);
        if profiling {
            eprintln!("[profile-detail] alloc (pool+tokens+kv): {:.1}ms", prefill_start.elapsed().as_secs_f64() * 1000.0);
        }

        // Encode and execute
        let gpu_start = Instant::now();
        let prefill_params = PrefillParams {
            n_tokens: prompt_ids.len(),
            pos_offset: 0,
        };

        let logits = if profiling {
            // Profiled path: per-op command buffers with GPU timestamps
            let (logits, timings) = unsafe {
                encode_prefill_profiled(
                    &prefill_graph,
                    &prefill_pool,
                    &metal.metal_res,
                    &metal.weight_buf_ids,
                    &kv_buf_ids,
                    &prefill_params,
                    token_ids_buf,
                    position_ids_buf,
                )
            };
            let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;
            let (breakdown, _total_gpu) = format_kernel_breakdown(&timings);
            eprintln!(
                "[profile] prefill GPU kernel breakdown ({} tok, {} ops, {:.1}ms wall):",
                prompt_tokens, prefill_graph.ops.len(), gpu_ms,
            );
            eprint!("{}", breakdown);
            logits
        } else {
            // Fast path: single command buffer
            unsafe {
                encode_prefill(
                    &prefill_graph,
                    &prefill_pool,
                    &metal.metal_res,
                    &metal.weight_buf_ids,
                    &kv_buf_ids,
                    &prefill_params,
                    token_ids_buf,
                    position_ids_buf,
                )
            }
        };

        // Advance cache position to match prefilled tokens
        cache.advance(prompt_ids.len());

        let mut next_token = sample_token(&logits, &gen_config.sampling, &mut rng);
        let prefill_duration = prefill_start.elapsed();

        if profiling {
            let prefill_ms = prefill_duration.as_secs_f64() * 1000.0;
            eprintln!(
                "[profile] prefill total: {:.1}ms ({} tokens)",
                prefill_ms, prompt_tokens,
            );
        }

        // Release temporary prefill buffers
        unsafe {
            msg_send_void(token_ids_buf, metal.metal_res.sels.release);
            msg_send_void(position_ids_buf, metal.metal_res.sels.release);
        }

        // ===================================================================
        // DECODE LOOP (graph-based fast path)
        // ===================================================================

        let mut stop_reason = StopReason::MaxTokens;
        // Accumulate per-token profiling for aggregate decode summary
        let mut decode_timings_accum: Vec<(PsoRef, f64)> = Vec::new();
        let mut decode_tokens_profiled: usize = 0;

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

            // LongRoPE factors were selected once before prefill based on max_seq_len
            // (matching llama.cpp's n_ctx-based selection), so no per-token swap needed.

            let metal = match &self.executor {
                Executor::Metal(m) => m,
                _ => unreachable!(),
            };

            let dynamic = DynamicParams {
                pos: cache.len(),
                token_id: next_token,
                kv_dim: metal.kv_dim,
            };

            let decode_graph = self.decode_graph.as_ref()
                .expect("Metal executor requires decode_graph");

            let logits = if profiling {
                let (logits, timings) = unsafe {
                    encode_decode_token_profiled(
                        decode_graph,
                        &metal.buffer_pool,
                        &metal.metal_res,
                        &metal.weight_buf_ids,
                        &kv_buf_ids,
                        &dynamic,
                        metal.token_id_buf,
                        metal.position_id_buf,
                    )
                };
                decode_timings_accum.extend_from_slice(&timings);
                decode_tokens_profiled += 1;
                logits
            } else {
                unsafe {
                    encode_decode_token(
                        decode_graph,
                        &metal.buffer_pool,
                        &metal.metal_res,
                        &metal.weight_buf_ids,
                        &kv_buf_ids,
                        &dynamic,
                        metal.token_id_buf,
                        metal.position_id_buf,
                    )
                }
            };

            cache.advance(1);

            next_token = sample_token(&logits, &gen_config.sampling, &mut rng);

            if let Some(start) = tok_start {
                let total_ms = start.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "[profile] tok {}: {:.1}ms",
                    step + 1, total_ms,
                );
            }
        }

        // Print aggregate decode kernel breakdown
        if profiling && decode_tokens_profiled > 0 {
            let (breakdown, _total_gpu) = format_kernel_breakdown(&decode_timings_accum);
            eprintln!(
                "[profile] decode GPU kernel breakdown ({} tok, {} ops total):",
                decode_tokens_profiled, decode_timings_accum.len(),
            );
            eprint!("{}", breakdown);
        }

        Ok(GenerationOutput {
            token_ids: generated_ids,
            stop_reason,
            prompt_tokens,
            prefill_duration,
        })
    }

    /// Project a hidden state row to vocabulary logits (CPU path).
    fn project_to_logits_cpu(
        &self,
        hidden: &DeviceTensor,
        row: usize,
        backend: &dyn ComputeBackend,
    ) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let n_rows = hidden.shape()[0];

        let proj_weight = self.weights.output_projection.as_ref()
            .unwrap_or(&self.weights.token_embedding);

        if n_rows == 1 {
            let logits_tensor = linear_forward(hidden, proj_weight, None, backend);
            let logits_host = backend.download(&logits_tensor);
            logits_host.as_f32().to_vec()
        } else {
            let hidden_host = backend.download(hidden);
            let hidden_data = hidden_host.as_f32();
            let start = row * hidden_size;
            let row_data = &hidden_data[start..start + hidden_size];
            let row_tensor = backend.upload(
                &Tensor::new(vec![1, hidden_size], row_data.to_vec()),
            );
            let logits_tensor = linear_forward(&row_tensor, proj_weight, None, backend);
            let logits_host = backend.download(&logits_tensor);
            logits_host.as_f32().to_vec()
        }
    }

    /// Extract raw MTLBuffer pointers from KV cache GPU tensors.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn extract_kv_buf_ids_metal(config: &ModelConfig, cache: &KvCache) -> Vec<Id> {
        let mut ids = Vec::with_capacity(config.num_layers * 2);
        for layer in 0..config.num_layers {
            ids.push(extract_buffer_id(cache.get_k_gpu(layer)));
            ids.push(extract_buffer_id(cache.get_v_gpu(layer)));
        }
        ids
    }

    /// The model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The vocabulary size of the loaded model.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids)
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl Drop for GenerationEngine {
    fn drop(&mut self) {
        if let Executor::Metal(ref metal) = self.executor {
            unsafe {
                let rel = sel_registerName(b"release\0".as_ptr() as _);
                if metal.token_id_buf != NIL {
                    msg_send_void(metal.token_id_buf, rel);
                }
                if metal.position_id_buf != NIL {
                    msg_send_void(metal.position_id_buf, rel);
                }
                if metal.metal_res.command_queue != NIL {
                    msg_send_void(metal.metal_res.command_queue, rel);
                }
                if metal.metal_res.device != NIL {
                    msg_send_void(metal.metal_res.device, rel);
                }
            }
        }
    }
}

/// Extract the raw MTLBuffer Id from a GPU-resident DeviceTensor.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn extract_buffer_id(dt: &DeviceTensor) -> Id {
    dt.gpu_inner::<MetalBuffer>()
        .expect("expected GPU-resident MetalBuffer in DeviceTensor")
        .buffer
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::backend::DeviceTensor;
    use crate::model::config::*;
    use crate::model::weights::{LayerWeights, ModelWeights};
    use crate::tensor::Tensor;
    use crate::tokenizer::Tokenizer;

    /// A mock tokenizer for generation tests.
    struct MockTokenizer {
        vocab: Vec<String>,
        eos_id: u32,
    }

    impl MockTokenizer {
        fn new() -> Self {
            Self {
                vocab: vec![
                    "<pad>".to_string(),  // 0
                    "<bos>".to_string(),  // 1
                    "<eos>".to_string(),  // 2
                    "hello".to_string(),  // 3
                    "world".to_string(),  // 4
                    "foo".to_string(),    // 5
                    "bar".to_string(),    // 6
                    "baz".to_string(),    // 7
                ],
                eos_id: 2,
            }
        }
    }

    impl Tokenizer for MockTokenizer {
        fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
            let mut ids = Vec::new();
            if add_special_tokens {
                ids.push(1); // BOS
            }
            for word in text.split_whitespace() {
                let id = self
                    .vocab
                    .iter()
                    .position(|v| v == word)
                    .map(|i| i as u32)
                    .unwrap_or(0);
                ids.push(id);
            }
            ids
        }

        fn decode(&self, ids: &[u32]) -> String {
            ids.iter()
                .filter_map(|&id| self.vocab.get(id as usize))
                .cloned()
                .collect::<Vec<_>>()
                .join(" ")
        }

        fn vocab_size(&self) -> usize {
            self.vocab.len()
        }

        fn bos_token_id(&self) -> Option<u32> {
            Some(1)
        }

        fn eos_token_id(&self) -> Option<u32> {
            Some(self.eos_id)
        }

        fn pad_token_id(&self) -> Option<u32> {
            Some(0)
        }
    }

    /// A mock tokenizer with no EOS token.
    struct NoEosTokenizer;

    impl NoEosTokenizer {
        fn new() -> Self {
            Self
        }
    }

    impl Tokenizer for NoEosTokenizer {
        fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
            let mut ids = Vec::new();
            if add_special_tokens {
                ids.push(1); // BOS
            }
            for word in text.split_whitespace() {
                let id = 3 + (word.len() as u32 % 5);
                ids.push(id);
            }
            ids
        }

        fn decode(&self, ids: &[u32]) -> String {
            ids.iter().map(|id| format!("t{}", id)).collect::<Vec<_>>().join(" ")
        }

        fn vocab_size(&self) -> usize { 8 }
        fn bos_token_id(&self) -> Option<u32> { Some(1) }
        fn eos_token_id(&self) -> Option<u32> { None }
        fn pad_token_id(&self) -> Option<u32> { Some(0) }
    }

    fn dt(tensor: Tensor) -> DeviceTensor {
        DeviceTensor::new(tensor)
    }

    fn identity_weight(size: usize) -> DeviceTensor {
        let mut data = vec![0.0f32; size * size];
        for i in 0..size {
            data[i * size + i] = 1.0;
        }
        dt(Tensor::new(vec![size, size], data))
    }

    fn zero_weight(out_dim: usize, in_dim: usize) -> DeviceTensor {
        dt(Tensor::new(
            vec![out_dim, in_dim],
            vec![0.0f32; out_dim * in_dim],
        ))
    }

    fn ones_weight(dim: usize) -> DeviceTensor {
        dt(Tensor::new(vec![dim], vec![1.0f32; dim]))
    }

    fn gen_config(hidden_size: usize) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Gemma3,
            arch_name: "gemma3".to_string(),
            hidden_size,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: hidden_size,
            ffn_hidden: hidden_size * 4,
            vocab_size: 8,
            max_seq_len: 64,
            norm_type: NormType::RMSNorm,
            norm_eps: 1e-6,
            activation: Activation::SwiGLU,
            position_type: PositionType::RoPE,
            rope_freq_base: 10000.0,
            rope_dim: hidden_size,
            rope_neox: false,
            rope_scaling_original_ctx: 0,
            rope_scaling_attn_factor: 1.0,
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

    fn gen_weights(config: &ModelConfig) -> ModelWeights {
        let h = config.hidden_size;
        let ffn = config.ffn_hidden;
        let v = config.vocab_size;

        let mut emb_data = vec![0.0f32; v * h];
        for i in 0..v {
            for j in 0..h {
                emb_data[i * h + j] = ((i * h + j) as f32) * 0.1;
            }
        }

        let layer = LayerWeights {
            attn_norm_w: Some(ones_weight(h)),
            attn_norm_b: None,
            ffn_norm_w: Some(ones_weight(h)),
            ffn_norm_b: None,
            attn_output_norm_w: None,
            attn_output_norm_b: None,
            ffn_output_norm_w: None,
            ffn_output_norm_b: None,
            attn_q: identity_weight(h),
            attn_k: identity_weight(h),
            attn_v: identity_weight(h),
            attn_output: identity_weight(h),
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            attn_output_bias: None,
            ffn_up: zero_weight(ffn, h),
            ffn_down: zero_weight(h, ffn),
            ffn_gate: Some(zero_weight(ffn, h)),
            ffn_up_bias: None,
            ffn_down_bias: None,
            attn_q_norm_w: None,
            attn_k_norm_w: None,
            attn_post_norm_w: None,
            ffn_post_norm_w: None,
        };

        ModelWeights {
            token_embedding: dt(Tensor::new(vec![v, h], emb_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![layer],
            output_norm_w: Some(ones_weight(h)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        }
    }

    fn build_gen_engine(config: ModelConfig) -> GenerationEngine {
        let weights = gen_weights(&config);
        let tokenizer: Box<dyn Tokenizer> = Box::new(MockTokenizer::new());
        let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
        GenerationEngine::new(config, weights, tokenizer, backend)
    }

    #[test]
    fn test_generate_greedy_deterministic() {
        let config = gen_config(4);
        let mut engine = build_gen_engine(config);
        let gen_cfg = GenerationConfig {
            max_tokens: 5,
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let result1 = engine.generate("hello", &gen_cfg).unwrap();
        let result2 = engine.generate("hello", &gen_cfg).unwrap();
        assert_eq!(result1, result2, "Greedy generation should be deterministic");
    }

    #[test]
    fn test_generate_stops_at_max_tokens() {
        let config = gen_config(4);
        let weights = gen_weights(&config);
        let tokenizer: Box<dyn Tokenizer> = Box::new(NoEosTokenizer::new());
        let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
        let mut engine = GenerationEngine::new(config, weights, tokenizer, backend);

        let gen_cfg = GenerationConfig {
            max_tokens: 3,
            stop_tokens: vec![],
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
        };

        let mut token_count = 0;
        let result = engine.generate_stream("hello", &gen_cfg, |_| {
            token_count += 1;
            true
        });
        assert!(result.is_ok());
        let generated = result.unwrap();
        assert!(
            generated.len() <= 3,
            "Expected at most 3 tokens, got {}",
            generated.len()
        );
        assert_eq!(generated.len(), token_count);
    }

    #[test]
    fn test_generate_stops_at_eos() {
        let config = gen_config(4);
        let mut engine = build_gen_engine(config);

        let gen_cfg = GenerationConfig {
            max_tokens: 100,
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut token_count = 0;
        let result = engine.generate_stream("hello", &gen_cfg, |_| {
            token_count += 1;
            true
        });
        assert!(result.is_ok());
        let generated = result.unwrap();
        assert!(
            generated.len() < 100,
            "Expected EOS to stop generation before 100 tokens, got {}",
            generated.len()
        );
    }

    #[test]
    fn test_generate_stream_callback() {
        let config = gen_config(4);
        let mut engine = build_gen_engine(config);
        let gen_cfg = GenerationConfig {
            max_tokens: 10,
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut received_tokens = Vec::new();
        let result = engine.generate_stream("hello", &gen_cfg, |token_id| {
            received_tokens.push(token_id);
            received_tokens.len() < 3
        });

        assert!(result.is_ok());
        let generated = result.unwrap();
        assert!(generated.len() <= 3);
        assert_eq!(generated.len(), received_tokens.len());
        assert_eq!(
            generated, received_tokens,
            "Stream callback tokens should match returned tokens"
        );
    }

    #[test]
    fn test_generate_rejects_non_causal_model() {
        let mut config = gen_config(4);
        config.causal = false;
        config.arch = ModelArch::Bert;

        let weights = gen_weights(&config);
        let tokenizer: Box<dyn Tokenizer> = Box::new(MockTokenizer::new());
        let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
        let mut engine = GenerationEngine::new(config, weights, tokenizer, backend);

        let gen_cfg = GenerationConfig::default();
        let result = engine.generate("hello", &gen_cfg);
        assert!(result.is_err(), "generate() should reject non-causal model");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("causal"), "Error should mention causal: {}", err_msg);

        let stream_result = engine.generate_stream("hello", &gen_cfg, |_| true);
        assert!(stream_result.is_err(), "generate_stream() should reject non-causal model");
    }

    #[test]
    fn test_generate_empty_prompt() {
        let config = gen_config(4);
        let mut engine = build_gen_engine(config);
        let gen_cfg = GenerationConfig::default();

        let result = engine.generate("", &gen_cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_prompt_exceeds_context() {
        let mut config = gen_config(4);
        config.max_seq_len = 3;
        let mut engine = build_gen_engine(config);
        let gen_cfg = GenerationConfig::default();

        let result = engine.generate("hello world foo", &gen_cfg);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("exceeds max_seq_len"), "Error: {}", err_msg);
    }

    #[test]
    fn test_generate_with_explicit_stop_tokens() {
        let config = gen_config(4);
        let mut engine = build_gen_engine(config);
        let gen_cfg = GenerationConfig {
            max_tokens: 100,
            stop_tokens: vec![3, 4, 5, 6, 7],
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
        };

        let result = engine.generate("hello", &gen_cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_accessors() {
        let config = gen_config(4);
        let engine = build_gen_engine(config);
        assert_eq!(engine.vocab_size(), 8);
        assert_eq!(engine.config().arch, ModelArch::Gemma3);
        assert!(engine.config().causal);
    }

    #[test]
    fn test_generate_max_tokens_zero() {
        let config = gen_config(4);
        let mut engine = build_gen_engine(config);
        let gen_cfg = GenerationConfig {
            max_tokens: 0,
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = engine.generate("hello", &gen_cfg).unwrap();
        assert!(result.is_empty(), "max_tokens=0 should produce empty output, got: {:?}", result);
    }

    #[test]
    fn test_generate_and_stream_consistency() {
        let config = gen_config(4);

        let mut engine1 = build_gen_engine(config.clone());
        let mut engine2 = build_gen_engine(config);

        let gen_cfg = GenerationConfig {
            max_tokens: 5,
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let text_result = engine1.generate("hello", &gen_cfg).unwrap();
        let stream_ids = engine2.generate_stream("hello", &gen_cfg, |_| true).unwrap();

        let mock_tok = MockTokenizer::new();
        let stream_text = mock_tok.decode(&stream_ids);
        assert_eq!(
            text_result, stream_text,
            "generate() and generate_stream() should produce identical output"
        );
    }

    #[test]
    fn test_stop_reason_display() {
        assert_eq!(StopReason::StopToken.to_string(), "eos");
        assert_eq!(StopReason::MaxTokens.to_string(), "max_tokens");
        assert_eq!(StopReason::ContextLength.to_string(), "context_length");
        assert_eq!(StopReason::Cancelled.to_string(), "cancelled");
    }

    #[test]
    fn test_generate_full_returns_metadata() {
        let config = gen_config(4);
        let mut engine = build_gen_engine(config);
        let gen_cfg = GenerationConfig {
            max_tokens: 5,
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let output = engine.generate_full("hello", &gen_cfg).unwrap();
        assert_eq!(output.prompt_tokens, 2);
        assert!(!output.token_ids.is_empty() || output.stop_reason == StopReason::StopToken);
        assert!(
            output.prefill_duration.as_nanos() > 0,
            "prefill_duration should be non-zero"
        );
    }

    #[test]
    fn test_generate_stream_full_cancelled() {
        let config = gen_config(4);
        let weights = gen_weights(&config);
        let tokenizer: Box<dyn Tokenizer> = Box::new(NoEosTokenizer::new());
        let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
        let mut engine = GenerationEngine::new(config, weights, tokenizer, backend);

        let gen_cfg = GenerationConfig {
            max_tokens: 100,
            stop_tokens: vec![],
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
        };

        let output = engine.generate_stream_full("hello", &gen_cfg, |_| false).unwrap();
        assert_eq!(output.stop_reason, StopReason::Cancelled);
        assert_eq!(output.token_ids.len(), 1);
    }

    #[test]
    fn test_generate_full_max_tokens_stop_reason() {
        let config = gen_config(4);
        let weights = gen_weights(&config);
        let tokenizer: Box<dyn Tokenizer> = Box::new(NoEosTokenizer::new());
        let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
        let mut engine = GenerationEngine::new(config, weights, tokenizer, backend);

        let gen_cfg = GenerationConfig {
            max_tokens: 3,
            stop_tokens: vec![],
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
        };

        let output = engine.generate_full("hello", &gen_cfg).unwrap();
        assert_eq!(output.stop_reason, StopReason::MaxTokens);
        assert_eq!(output.token_ids.len(), 3);
    }

    #[test]
    fn test_generate_full_stop_token_via_explicit_stop() {
        let config = gen_config(4);
        let mut engine = build_gen_engine(config);

        let probe = engine.generate_full("hello", &GenerationConfig {
            max_tokens: 1,
            stop_tokens: vec![],
            sampling: SamplingConfig { temperature: 0.0, ..Default::default() },
        }).unwrap();

        assert!(!probe.token_ids.is_empty(), "Need at least one token to test stop");
        let first_token = probe.token_ids[0];

        let output = engine.generate_full("hello", &GenerationConfig {
            max_tokens: 100,
            stop_tokens: vec![first_token],
            sampling: SamplingConfig { temperature: 0.0, ..Default::default() },
        }).unwrap();

        assert_eq!(output.stop_reason, StopReason::StopToken);
        assert!(output.token_ids.is_empty());
    }

    #[test]
    fn test_generate_full_context_length_stop_reason() {
        let mut config = gen_config(4);
        config.max_seq_len = 5;
        let weights = gen_weights(&config);
        let tokenizer: Box<dyn Tokenizer> = Box::new(NoEosTokenizer::new());
        let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
        let mut engine = GenerationEngine::new(config, weights, tokenizer, backend);

        let gen_cfg = GenerationConfig {
            max_tokens: 100,
            stop_tokens: vec![],
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
        };

        let output = engine.generate_full("hello", &gen_cfg).unwrap();
        assert_eq!(output.stop_reason, StopReason::ContextLength);
        assert_eq!(
            output.token_ids.len(),
            4,
            "Context-length stop should preserve the final sampled token"
        );
    }

    #[test]
    fn test_generate_context_length_preserves_last_token() {
        let mut config = gen_config(4);
        config.max_seq_len = 5;
        let weights = gen_weights(&config);
        let tokenizer: Box<dyn Tokenizer> = Box::new(NoEosTokenizer::new());
        let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
        let mut engine = GenerationEngine::new(config, weights, tokenizer, backend);

        let gen_cfg = GenerationConfig {
            max_tokens: 100,
            stop_tokens: vec![],
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
        };

        let text = engine.generate("hello", &gen_cfg).unwrap();
        assert!(!text.is_empty(), "Should produce output at context limit");
    }

    #[test]
    fn test_generate_stream_context_length_fires_callback() {
        let mut config = gen_config(4);
        config.max_seq_len = 5;
        let weights = gen_weights(&config);
        let tokenizer: Box<dyn Tokenizer> = Box::new(NoEosTokenizer::new());
        let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
        let mut engine = GenerationEngine::new(config, weights, tokenizer, backend);

        let gen_cfg = GenerationConfig {
            max_tokens: 100,
            stop_tokens: vec![],
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
        };

        let mut streamed_tokens = Vec::new();
        let output = engine
            .generate_stream_full("hello", &gen_cfg, |tok| {
                streamed_tokens.push(tok);
                true
            })
            .unwrap();

        assert_eq!(output.stop_reason, StopReason::ContextLength);
        assert_eq!(output.token_ids.len(), 4);
        assert_eq!(
            output.token_ids, streamed_tokens,
            "Streamed tokens should match token_ids exactly"
        );
    }

    #[test]
    fn test_generate_stream_full_tokens_match_stream() {
        let config = gen_config(4);
        let mut engine1 = build_gen_engine(config.clone());
        let mut engine2 = build_gen_engine(config);
        let gen_cfg = GenerationConfig {
            max_tokens: 5,
            sampling: SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let stream_ids = engine1.generate_stream("hello", &gen_cfg, |_| true).unwrap();
        let full_output = engine2.generate_stream_full("hello", &gen_cfg, |_| true).unwrap();
        assert_eq!(stream_ids, full_output.token_ids);
    }

    #[test]
    fn test_decode_method() {
        let config = gen_config(4);
        let engine = build_gen_engine(config);
        assert_eq!(engine.decode(&[3, 4]), "hello world");
        assert_eq!(engine.decode(&[]), "");
    }

    // ------------------------------------------------------------------
    // Backend selection tests
    // ------------------------------------------------------------------

    #[test]
    fn test_from_gguf_with_backend_unknown_returns_error() {
        let result = GenerationEngine::from_gguf_with_backend("/nonexistent.gguf", "cuda");
        match result {
            Err(e) => {
                let err = format!("{}", e);
                assert!(err.contains("Unknown backend"), "Error: {}", err);
            }
            Ok(_) => panic!("Expected error for unknown backend"),
        }
    }

    #[test]
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    fn test_from_gguf_with_backend_metal_unavailable() {
        let result = GenerationEngine::from_gguf_with_backend("/nonexistent.gguf", "metal");
        match result {
            Err(e) => {
                let err = format!("{}", e);
                assert!(err.contains("Metal backend not available"), "Error: {}", err);
            }
            Ok(_) => panic!("Expected error for Metal on non-Metal build"),
        }
    }

    #[test]
    fn test_from_gguf_with_backend_cpu_bad_path() {
        let result = GenerationEngine::from_gguf_with_backend("/nonexistent.gguf", "cpu");
        assert!(result.is_err(), "CPU backend with bad path should fail");
    }

    #[test]
    fn test_from_gguf_with_backend_auto_bad_path() {
        let result = GenerationEngine::from_gguf_with_backend("/nonexistent.gguf", "auto");
        assert!(result.is_err(), "Auto backend with bad path should fail");
    }
}
