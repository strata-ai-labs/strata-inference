// M4.3: Transformer layer forward pass.
//
// Implements the forward pass for a single transformer layer and the full model,
// parameterized by ModelConfig to handle Gemma3, LLaMA, and BERT architectures.

use tracing::{debug, trace};

use crate::backend::{ComputeBackend, DeviceTensor};
use crate::error::InferenceError;
use crate::tensor::{Tensor, TensorDtype};

use super::config::{Activation, ModelArch, ModelConfig, NormType, PositionType};
use super::weights::{LayerWeights, ModelWeights};

/// Linear (matrix multiply) forward pass, dispatching on quantized vs F32 weights.
///
/// Weights are stored as [out_features, in_features]:
/// - For Q8_0/Q4_0: uses `backend.quantized_matmul(weight, input)` which does fused dequant.
/// - For F32/F16: uses `backend.matmul_transpose(input, weight)` since weight is [out, in].
///
/// If bias is provided, adds it to each row of the result.
pub(crate) fn linear_forward(
    input: &DeviceTensor,
    weight: &DeviceTensor,
    bias: Option<&DeviceTensor>,
    backend: &dyn ComputeBackend,
) -> DeviceTensor {
    let result = match weight.dtype() {
        TensorDtype::F32 | TensorDtype::F16 => {
            // matmul_transpose: input [M, K] x weight [N, K]^T -> [M, N]
            backend.matmul_transpose(input, weight)
        }
        _ => {
            // quantized_matmul: Q weights [N, K] x f32 input [M, K] -> [M, N]
            backend.quantized_matmul(weight, input)
        }
    };

    match bias {
        Some(b) => backend.add_bias(&result, b),
        None => result,
    }
}

/// Apply normalization (LayerNorm or RMSNorm) based on config.
fn normalize(
    input: &DeviceTensor,
    weight: &DeviceTensor,
    bias: Option<&DeviceTensor>,
    config: &ModelConfig,
    backend: &dyn ComputeBackend,
) -> DeviceTensor {
    match config.norm_type {
        NormType::LayerNorm => {
            let b = bias.expect("LayerNorm requires a bias tensor");
            backend.layer_norm(input, weight, b, config.norm_eps)
        }
        NormType::RMSNorm => backend.rms_norm(input, weight, config.norm_eps),
    }
}

/// Extract a per-head slice from a [seq_len, total_dim] tensor.
///
/// Returns a [seq_len, head_dim] tensor for the given head index.
fn extract_head(data: &[f32], seq_len: usize, total_dim: usize, head: usize, head_dim: usize) -> Tensor {
    let mut head_data = vec![0.0f32; seq_len * head_dim];
    for pos in 0..seq_len {
        let src_offset = pos * total_dim + head * head_dim;
        let dst_offset = pos * head_dim;
        head_data[dst_offset..dst_offset + head_dim]
            .copy_from_slice(&data[src_offset..src_offset + head_dim]);
    }
    Tensor::new(vec![seq_len, head_dim], head_data)
}

/// Reassemble per-head outputs into a single [seq_len, num_heads * head_dim] tensor.
fn assemble_heads(head_outputs: &[Vec<f32>], seq_len: usize, num_heads: usize, head_dim: usize) -> Tensor {
    let total_dim = num_heads * head_dim;
    let mut result = vec![0.0f32; seq_len * total_dim];
    for head in 0..num_heads {
        let head_data = &head_outputs[head];
        for pos in 0..seq_len {
            let src_offset = pos * head_dim;
            let dst_offset = pos * total_dim + head * head_dim;
            result[dst_offset..dst_offset + head_dim]
                .copy_from_slice(&head_data[src_offset..src_offset + head_dim]);
        }
    }
    Tensor::new(vec![seq_len, total_dim], result)
}

/// Repeat K/V heads for Grouped Query Attention (GQA).
///
/// When num_kv_heads < num_heads, each KV head serves (num_heads / num_kv_heads) Q heads.
/// This function expands [seq_len, num_kv_heads * head_dim] to [seq_len, num_heads * head_dim]
/// by repeating each KV head the appropriate number of times.
fn repeat_kv_heads(
    data: &[f32],
    seq_len: usize,
    num_kv_heads: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let repeats = num_heads / num_kv_heads;
    let kv_total_dim = num_kv_heads * head_dim;
    let out_total_dim = num_heads * head_dim;
    let mut result = vec![0.0f32; seq_len * out_total_dim];

    for pos in 0..seq_len {
        for kv_head in 0..num_kv_heads {
            let src_offset = pos * kv_total_dim + kv_head * head_dim;
            for r in 0..repeats {
                let out_head = kv_head * repeats + r;
                let dst_offset = pos * out_total_dim + out_head * head_dim;
                result[dst_offset..dst_offset + head_dim]
                    .copy_from_slice(&data[src_offset..src_offset + head_dim]);
            }
        }
    }

    result
}

/// Multi-head attention mechanism.
///
/// Handles:
/// - RoPE (if config.position_type == RoPE)
/// - Grouped Query Attention (GQA) when num_kv_heads < num_heads
/// - Attention mask (padding exclusion)
/// - Per-head score computation, scaling, optional softcap, optional causal mask, softmax
/// - Weighted sum of values and head concatenation
fn multi_head_attention(
    q: DeviceTensor,
    k: DeviceTensor,
    v: DeviceTensor,
    config: &ModelConfig,
    backend: &dyn ComputeBackend,
    pos_offset: usize,
    attention_mask: &[f32],
    longrope: Option<&LongRopeParams>,
) -> DeviceTensor {
    let seq_len = q.shape()[0];
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;

    trace!(seq_len, num_heads, num_kv_heads, head_dim, "multi_head_attention");

    // Step a: Apply RoPE if configured (RoPE applies to Q and K only, not V)
    let (q_proc, k_proc) = if config.position_type == PositionType::RoPE {
        if let Some(params) = longrope {
            rope_neox_with_factors(
                &q, &k, pos_offset, config.rope_freq_base,
                head_dim, config.rope_dim, params.factors, params.mscale, backend,
            )
        } else if config.rope_neox {
            backend.rope_neox(
                &q, &k, pos_offset, config.rope_freq_base, head_dim, config.rope_dim,
            )
        } else {
            backend.rope(
                &q, &k, pos_offset, config.rope_freq_base, head_dim, config.rope_dim,
            )
        }
    } else {
        (q, k)
    };

    // Step b: Download Q, K, V for head extraction (CPU-side operation)
    let q_host = backend.download(&q_proc);
    let k_host = backend.download(&k_proc);
    let v_host = backend.download(&v);
    let q_data = q_host.as_f32();
    let k_data_raw = k_host.as_f32();
    let v_data_raw = v_host.as_f32();

    // GQA - repeat K/V heads if needed
    let (k_expanded_data, v_expanded_data) = if num_kv_heads < num_heads {
        let k_exp = repeat_kv_heads(k_data_raw, seq_len, num_kv_heads, num_heads, head_dim);
        let v_exp = repeat_kv_heads(v_data_raw, seq_len, num_kv_heads, num_heads, head_dim);
        (k_exp, v_exp)
    } else {
        (k_data_raw.to_vec(), v_data_raw.to_vec())
    };

    let total_dim = num_heads * head_dim;

    // Compute attention scale
    let attn_scale = config.attn_scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

    // Step c-j: Per-head attention computation
    let mut head_outputs: Vec<Vec<f32>> = Vec::with_capacity(num_heads);

    for h in 0..num_heads {
        // Extract per-head Q, K, V: [seq_len, head_dim]
        let q_head = extract_head(q_data, seq_len, total_dim, h, head_dim);
        let k_head = extract_head(&k_expanded_data, seq_len, total_dim, h, head_dim);
        let v_head = extract_head(&v_expanded_data, seq_len, total_dim, h, head_dim);

        // Step d: scores = Q @ K^T: [seq_len, head_dim] x [head_dim, seq_len] -> [seq_len, seq_len]
        let q_dev = backend.upload(&q_head);
        let k_dev = backend.upload(&k_head);
        let v_dev = backend.upload(&v_head);

        let scores = backend.matmul_transpose(&q_dev, &k_dev);

        // Step e: Scale
        let scores = backend.scale(&scores, attn_scale);

        // Step f: Softcap (if attn_logit_softcap > 0)
        let scores = if config.attn_logit_softcap > 0.0 {
            let cap = config.attn_logit_softcap;
            // scores = cap * tanh(scores / cap)
            let scaled_down = backend.scale(&scores, 1.0 / cap);
            let tanh_tensor = backend.tanh(&scaled_down);
            backend.scale(&tanh_tensor, cap)
        } else {
            scores
        };

        // Step g1: Apply attention mask — where mask[j] == 0, set scores[i][j] = -inf
        let scores = {
            assert_eq!(
                attention_mask.len(),
                seq_len,
                "attention_mask length ({}) must equal seq_len ({})",
                attention_mask.len(),
                seq_len,
            );
            let needs_masking = attention_mask.iter().any(|&m| m == 0.0);
            if needs_masking {
                let scores_tensor = backend.download(&scores);
                let mut scores_data = scores_tensor.as_f32().to_vec();
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        if attention_mask[j] == 0.0 {
                            scores_data[i * seq_len + j] = f32::NEG_INFINITY;
                        }
                    }
                }
                backend.upload(&Tensor::new(vec![seq_len, seq_len], scores_data))
            } else {
                scores
            }
        };

        // Step g2: Apply causal mask if configured
        let scores = if config.causal {
            backend.apply_causal_mask(&scores, seq_len)
        } else {
            scores
        };

        // Step h: Softmax
        let probs = backend.softmax(&scores);

        // Step i: Weighted sum: output = probs @ V: [seq_len, seq_len] x [seq_len, head_dim] -> [seq_len, head_dim]
        let head_out = backend.matmul(&probs, &v_dev);
        let head_out_host = backend.download(&head_out);
        head_outputs.push(head_out_host.as_f32().to_vec());
    }

    // Step j: Concatenate heads back to [seq_len, num_heads * head_dim]
    let result = assemble_heads(&head_outputs, seq_len, num_heads, head_dim);
    backend.upload(&result)
}

/// Single transformer layer forward pass.
///
/// Dispatches between pre-norm (Gemma/LLaMA) and post-norm (BERT) architectures
/// based on `config.pre_norm`.
pub(crate) fn transformer_layer_forward(
    input: &DeviceTensor,
    layer: &LayerWeights,
    config: &ModelConfig,
    backend: &dyn ComputeBackend,
    pos_offset: usize,
    attention_mask: &[f32],
    longrope: Option<&LongRopeParams>,
) -> DeviceTensor {
    trace!(pos_offset, pre_norm = config.pre_norm, "transformer_layer_forward");

    match config.arch {
        ModelArch::GemmaEmbedding => {
            // GemmaEmbedding doesn't use LongRoPE
            gemma_embedding_layer_forward(input, layer, config, backend, pos_offset, attention_mask)
        }
        _ if config.pre_norm => {
            pre_norm_layer_forward(input, layer, config, backend, pos_offset, attention_mask, longrope)
        }
        _ => {
            post_norm_layer_forward(input, layer, config, backend, pos_offset, attention_mask, longrope)
        }
    }
}

/// Pre-norm layer forward pass (Gemma/LLaMA — matches llama.cpp).
///
/// 1. normed = rms_norm(input, attn_norm_w)
/// 2. q, k, v = project(normed)
/// 3. attn_out = multi_head_attention(q, k, v, mask)
/// 4. projected = output_project(attn_out)
/// 5. residual = input + projected
/// 6. normed2 = rms_norm(residual, ffn_norm_w)
/// 7. ffn_out = swiglu_ffn(normed2)
/// 8. output = residual + ffn_out
fn pre_norm_layer_forward(
    input: &DeviceTensor,
    layer: &LayerWeights,
    config: &ModelConfig,
    backend: &dyn ComputeBackend,
    pos_offset: usize,
    attention_mask: &[f32],
    longrope: Option<&LongRopeParams>,
) -> DeviceTensor {
    // 1. Pre-attention normalization
    let attn_norm_w = layer.attn_norm_w.as_ref()
        .expect("pre_norm layer requires attn_norm_w");
    let normed = normalize(
        input,
        attn_norm_w,
        layer.attn_norm_b.as_ref(),
        config,
        backend,
    );

    // 2. QKV projections
    let q = linear_forward(&normed, &layer.attn_q, layer.attn_q_bias.as_ref(), backend);
    let k = linear_forward(&normed, &layer.attn_k, layer.attn_k_bias.as_ref(), backend);
    let v = linear_forward(&normed, &layer.attn_v, layer.attn_v_bias.as_ref(), backend);

    // Per-head Q/K RMSNorm (Qwen3)
    let q = if let Some(ref qn_w) = layer.attn_q_norm_w {
        rms_norm_per_head(q, qn_w, config.num_heads, config.head_dim, config.norm_eps, backend)
    } else { q };
    let k = if let Some(ref kn_w) = layer.attn_k_norm_w {
        rms_norm_per_head(k, kn_w, config.num_kv_heads, config.head_dim, config.norm_eps, backend)
    } else { k };

    // 3. Multi-head attention
    let attn_out = multi_head_attention(q, k, v, config, backend, pos_offset, attention_mask, longrope);

    // 4. Output projection
    let projected = linear_forward(
        &attn_out,
        &layer.attn_output,
        layer.attn_output_bias.as_ref(),
        backend,
    );

    // 4b. Post-attention norm (Gemma3: before residual add)
    let projected = if let Some(ref norm_w) = layer.attn_post_norm_w {
        backend.rms_norm(&projected, norm_w, config.norm_eps)
    } else {
        projected
    };

    let residual = backend.add(input, &projected);

    // 5. Pre-FFN normalization
    let ffn_norm_w = layer.ffn_norm_w.as_ref()
        .expect("pre_norm layer requires ffn_norm_w");
    let normed2 = normalize(
        &residual,
        ffn_norm_w,
        layer.ffn_norm_b.as_ref(),
        config,
        backend,
    );

    // 6. FFN
    let ffn_out = ffn_forward(&normed2, layer, config, backend);

    // 6b. Post-FFN norm (Gemma3: before residual add)
    let ffn_out = if let Some(ref norm_w) = layer.ffn_post_norm_w {
        backend.rms_norm(&ffn_out, norm_w, config.norm_eps)
    } else {
        ffn_out
    };

    // 7. Residual connection
    backend.add(&residual, &ffn_out)
}

/// Post-norm layer forward pass (BERT — matches llama.cpp bert.cpp).
///
/// 1. q, k, v = project(input)              // NO pre-norm
/// 2. attn_out = multi_head_attention(q, k, v, mask)
/// 3. projected = output_project(attn_out)
/// 4. residual = input + projected
/// 5. residual = layer_norm(residual, attn_output_norm_w, b)  // POST-norm
/// 6. ffn_out = gelu_ffn(residual)           // NO pre-FFN norm
/// 7. output = residual + ffn_out
/// 8. output = layer_norm(output, ffn_output_norm_w, b)       // POST-norm
fn post_norm_layer_forward(
    input: &DeviceTensor,
    layer: &LayerWeights,
    config: &ModelConfig,
    backend: &dyn ComputeBackend,
    pos_offset: usize,
    attention_mask: &[f32],
    longrope: Option<&LongRopeParams>,
) -> DeviceTensor {
    // 1. QKV projections (NO pre-norm)
    let q = linear_forward(input, &layer.attn_q, layer.attn_q_bias.as_ref(), backend);
    let k = linear_forward(input, &layer.attn_k, layer.attn_k_bias.as_ref(), backend);
    let v = linear_forward(input, &layer.attn_v, layer.attn_v_bias.as_ref(), backend);

    // 2. Multi-head attention
    let attn_out = multi_head_attention(q, k, v, config, backend, pos_offset, attention_mask, longrope);

    // 3. Output projection + residual
    let projected = linear_forward(
        &attn_out,
        &layer.attn_output,
        layer.attn_output_bias.as_ref(),
        backend,
    );
    let mut residual = backend.add(input, &projected);

    // 4. Post-attention norm
    if let Some(ref norm_w) = layer.attn_output_norm_w {
        residual = normalize(
            &residual,
            norm_w,
            layer.attn_output_norm_b.as_ref(),
            config,
            backend,
        );
    }

    // 5. FFN (NO pre-FFN norm)
    let ffn_out = ffn_forward(&residual, layer, config, backend);

    // 6. Residual connection
    let mut output = backend.add(&residual, &ffn_out);

    // 7. Post-FFN norm
    if let Some(ref norm_w) = layer.ffn_output_norm_w {
        output = normalize(
            &output,
            norm_w,
            layer.ffn_output_norm_b.as_ref(),
            config,
            backend,
        );
    }

    output
}

/// Apply RMSNorm per-head to a [seq_len, num_heads * head_dim] tensor.
///
/// The weight tensor is [head_dim], shared across all heads.
/// For each head slice [seq_len, head_dim], we compute RMSNorm independently
/// (since RMSNorm normalizes over the last dimension, we must split by head
/// to get the correct mean-of-squares over head_dim, not total_dim).
///
/// Implementation: reshape [seq_len, num_heads * head_dim] to
/// [seq_len * num_heads, head_dim], apply rms_norm (which operates on last dim
/// per row), then reshape back. Single GPU dispatch, no CPU roundtrip.
fn rms_norm_per_head(
    mut input: DeviceTensor,
    weight: &DeviceTensor,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
    backend: &dyn ComputeBackend,
) -> DeviceTensor {
    let seq_len = input.shape()[0];
    let total_dim = num_heads * head_dim;
    debug_assert_eq!(input.shape()[1], total_dim);

    // Reshape to [seq_len * num_heads, head_dim] so rms_norm treats each head
    // as a separate row, normalizing over head_dim independently.
    input.reshape(vec![seq_len * num_heads, head_dim]);
    let mut normed = backend.rms_norm(&input, weight, eps);
    normed.reshape(vec![seq_len, total_dim]);
    normed
}

/// GemmaEmbedding layer forward pass.
///
/// This architecture has per-head Q/K norms and post-projection norms:
///
/// 1.  normed = rms_norm(input, attn_norm_w)
/// 2.  q, k, v = project(normed)
/// 3.  q = rms_norm_per_head(q, attn_q_norm_w)
/// 4.  k = rms_norm_per_head(k, attn_k_norm_w)
/// 5.  (q, k) = rope(q, k)
/// 6.  attn_out = multi_head_attention(q, k, v)  (bidirectional)
/// 7.  projected = output_project(attn_out)
/// 8.  projected = rms_norm(projected, attn_post_norm_w)
/// 9.  residual = input + projected
/// 10. normed2 = rms_norm(residual, ffn_norm_w)
/// 11. ffn_out = geglu_ffn(normed2)
/// 12. ffn_out = rms_norm(ffn_out, ffn_post_norm_w)
/// 13. output = residual + ffn_out
fn gemma_embedding_layer_forward(
    input: &DeviceTensor,
    layer: &LayerWeights,
    config: &ModelConfig,
    backend: &dyn ComputeBackend,
    pos_offset: usize,
    attention_mask: &[f32],
) -> DeviceTensor {
    // 1. Pre-attention normalization
    let attn_norm_w = layer.attn_norm_w.as_ref()
        .expect("GemmaEmbedding layer requires attn_norm_w");
    let normed = normalize(input, attn_norm_w, None, config, backend);

    // 2. QKV projections
    let q = linear_forward(&normed, &layer.attn_q, None, backend);
    let k = linear_forward(&normed, &layer.attn_k, None, backend);
    let v = linear_forward(&normed, &layer.attn_v, None, backend);

    // 3-4. Per-head Q/K RMSNorm
    let q = if let Some(ref qn_w) = layer.attn_q_norm_w {
        rms_norm_per_head(q, qn_w, config.num_heads, config.head_dim, config.norm_eps, backend)
    } else {
        q
    };
    let k = if let Some(ref kn_w) = layer.attn_k_norm_w {
        rms_norm_per_head(k, kn_w, config.num_kv_heads, config.head_dim, config.norm_eps, backend)
    } else {
        k
    };

    // 5-6. Multi-head attention (RoPE is applied inside, causal=false for bidirectional)
    // GemmaEmbedding doesn't use LongRoPE.
    let attn_out = multi_head_attention(q, k, v, config, backend, pos_offset, attention_mask, None);

    // 7. Output projection
    let projected = linear_forward(&attn_out, &layer.attn_output, None, backend);

    // 8. Post-attention norm (before residual)
    let projected = if let Some(ref norm_w) = layer.attn_post_norm_w {
        backend.rms_norm(&projected, norm_w, config.norm_eps)
    } else {
        projected
    };

    // 9. Residual
    let residual = backend.add(input, &projected);

    // 10. Pre-FFN normalization
    let ffn_norm_w = layer.ffn_norm_w.as_ref()
        .expect("GemmaEmbedding layer requires ffn_norm_w");
    let normed2 = normalize(&residual, ffn_norm_w, None, config, backend);

    // 11. FFN (GeGLU path)
    let ffn_out = ffn_forward(&normed2, layer, config, backend);

    // 12. Post-FFN norm (before residual)
    let ffn_out = if let Some(ref norm_w) = layer.ffn_post_norm_w {
        backend.rms_norm(&ffn_out, norm_w, config.norm_eps)
    } else {
        ffn_out
    };

    // 13. Residual connection
    backend.add(&residual, &ffn_out)
}

/// FFN forward pass, dispatching on activation type.
fn ffn_forward(
    input: &DeviceTensor,
    layer: &LayerWeights,
    config: &ModelConfig,
    backend: &dyn ComputeBackend,
) -> DeviceTensor {
    match config.activation {
        Activation::SwiGLU => {
            let gate_weight = layer
                .ffn_gate
                .as_ref()
                .expect("SwiGLU activation requires ffn_gate weight");
            let gate = linear_forward(input, gate_weight, None, backend);
            let up = linear_forward(input, &layer.ffn_up, layer.ffn_up_bias.as_ref(), backend);
            let activated = backend.swiglu(&gate, &up);
            linear_forward(&activated, &layer.ffn_down, layer.ffn_down_bias.as_ref(), backend)
        }
        Activation::GeGLU => {
            let gate_weight = layer
                .ffn_gate
                .as_ref()
                .expect("GeGLU activation requires ffn_gate weight");
            let gate = linear_forward(input, gate_weight, None, backend);
            let up = linear_forward(input, &layer.ffn_up, layer.ffn_up_bias.as_ref(), backend);
            let activated = backend.geglu(&gate, &up);
            linear_forward(&activated, &layer.ffn_down, layer.ffn_down_bias.as_ref(), backend)
        }
        Activation::GELU => {
            let up = linear_forward(input, &layer.ffn_up, layer.ffn_up_bias.as_ref(), backend);
            let activated = backend.gelu(&up);
            linear_forward(&activated, &layer.ffn_down, layer.ffn_down_bias.as_ref(), backend)
        }
    }
}

/// Full model forward pass.
///
/// Steps:
/// 1. Validate input_ids
/// 2. Embedding lookup
/// 3. Embedding scaling (if embedding_scale != 1.0)
/// 4. Add position embeddings (if Learned position type)
/// 5. Embedding normalization (BERT only)
/// 6. Run through all transformer layers
/// 7. Final output normalization
/// 8. Return hidden states [seq_len, hidden_size]
pub fn model_forward(
    input_ids: &[u32],
    attention_mask: &[f32],
    weights: &ModelWeights,
    config: &ModelConfig,
    backend: &dyn ComputeBackend,
    pos_offset: usize,
) -> Result<DeviceTensor, InferenceError> {
    let seq_len = input_ids.len();
    debug!(seq_len, num_layers = config.num_layers, pos_offset, "model_forward");

    // 1. Validate input_ids against vocab_size
    for &id in input_ids {
        if (id as usize) >= config.vocab_size {
            return Err(InferenceError::Model(format!(
                "input token ID {} exceeds vocab_size {}",
                id, config.vocab_size
            )));
        }
    }

    // 2. Embedding lookup: [seq_len, hidden_size]
    let mut hidden = backend.embedding_lookup(&weights.token_embedding, input_ids);

    // 3. Embedding scaling
    if (config.embedding_scale - 1.0).abs() > f32::EPSILON {
        hidden = backend.scale(&hidden, config.embedding_scale);
    }

    // 4. Add position embeddings (Learned, e.g., BERT)
    if config.position_type == PositionType::Learned {
        if let Some(ref pos_emb) = weights.position_embedding {
            // Create position IDs [0, 1, 2, ..., seq_len-1]
            let pos_ids: Vec<u32> = (0..seq_len as u32).collect();
            let pos_embeddings = backend.embedding_lookup(pos_emb, &pos_ids);
            hidden = backend.add(&hidden, &pos_embeddings);
        }
    }

    // 4b. Add token type embeddings (BERT only — hardcoded type 0 = "Sentence A")
    if let Some(ref type_emb) = weights.token_type_embedding {
        let type_shape = type_emb.shape();
        if type_shape.len() == 1 {
            // 1D tensor (e.g., BGE-M3 with token_type_count=1): broadcast-add
            // the single vector to every position via add_bias [M,N] + [N].
            hidden = backend.add_bias(&hidden, type_emb);
        } else {
            // 2D tensor: look up row 0 for every token (all same type)
            let type_ids = vec![0u32; seq_len];
            let type_embeddings = backend.embedding_lookup(type_emb, &type_ids);
            hidden = backend.add(&hidden, &type_embeddings);
        }
    }

    // 5. Embedding normalization (BERT only)
    if let Some(ref norm_w) = weights.embedding_norm_w {
        hidden = normalize(
            &hidden,
            norm_w,
            weights.embedding_norm_b.as_ref(),
            config,
            backend,
        );
    }

    // 6. Determine LongRoPE frequency factors and mscale (Phi-3.5 etc.)
    // Same logic as model_forward_step: llama.cpp selects short vs long factors
    // based on configured n_ctx (max_seq_len), not the current sequence position.
    let rope_factors_data: Option<Vec<f32>> = if config.rope_scaling_original_ctx > 0 {
        let factors_tensor = if config.max_seq_len > config.rope_scaling_original_ctx {
            weights.rope_factors_long.as_ref()
        } else {
            weights.rope_factors_short.as_ref()
        };
        factors_tensor.map(|t| {
            let host = backend.download(t);
            host.as_f32().to_vec()
        })
    } else {
        None
    };
    let longrope_params = rope_factors_data.as_ref().map(|factors| LongRopeParams {
        factors: factors.as_slice(),
        mscale: config.rope_scaling_attn_factor,
    });

    // 7. Run through all transformer layers
    for (i, layer) in weights.layers.iter().enumerate() {
        trace!(layer = i, "Running layer");
        hidden = transformer_layer_forward(
            &hidden, layer, config, backend, pos_offset, attention_mask,
            longrope_params.as_ref(),
        );
    }

    // 8. Final output normalization (absent for BERT which uses per-layer post-norm)
    if let Some(ref norm_w) = weights.output_norm_w {
        hidden = normalize(
            &hidden,
            norm_w,
            weights.output_norm_b.as_ref(),
            config,
            backend,
        );
    }

    // Return hidden states [seq_len, hidden_size]
    Ok(hidden)
}

// =========================================================================
// Cached forward pass for autoregressive generation (M6)
// =========================================================================

use super::cache::KvCache;

/// NeoX-style RoPE with per-dimension LongRoPE frequency factors and mscale.
///
/// Same as `rope_neox` but divides theta by `factors[i]` and scales cos/sin by `mscale`:
///   theta[i] = pos * freq_base^(-2i/rope_dim) / factors[i]
///   cos_theta = cos(theta) * mscale
///   sin_theta = sin(theta) * mscale
fn rope_neox_with_factors(
    q: &DeviceTensor,
    k: &DeviceTensor,
    pos_offset: usize,
    freq_base: f32,
    head_dim: usize,
    rope_dim: usize,
    factors: &[f32],
    mscale: f32,
    backend: &dyn ComputeBackend,
) -> (DeviceTensor, DeviceTensor) {
    let half_rope_dim = rope_dim / 2;
    assert_eq!(factors.len(), half_rope_dim,
        "rope factors length ({}) must equal rope_dim/2 ({})", factors.len(), half_rope_dim);

    let q_host = backend.download(q);
    let k_host = backend.download(k);
    let q_data = q_host.as_f32();
    let k_data = k_host.as_f32();
    let q_shape = q.shape();
    let k_shape = k.shape();

    let seq_len = q_shape[0];
    let total_dim = q_shape[1];
    let n_heads = total_dim / head_dim;
    let k_total_dim = k_shape[1];
    let k_n_heads = k_total_dim / head_dim;

    // Precompute frequencies with factors (factors are divisors, matching llama.cpp: theta/ff)
    let inv_ndims = -1.0f32 / rope_dim as f32;
    let mut freqs = vec![0.0f32; half_rope_dim];
    for i in 0..half_rope_dim {
        freqs[i] = freq_base.powf(inv_ndims * (2 * i) as f32) / factors[i];
    }

    let mut q_rot = q_data.to_vec();
    let mut k_rot = k_data.to_vec();

    for pos in 0..seq_len {
        let abs_pos = (pos + pos_offset) as f32;

        for head in 0..n_heads {
            let offset = pos * total_dim + head * head_dim;
            for i in 0..half_rope_dim {
                let theta = abs_pos * freqs[i];
                let cos_theta = theta.cos() * mscale;
                let sin_theta = theta.sin() * mscale;

                let q0 = q_data[offset + i];
                let q1 = q_data[offset + i + half_rope_dim];
                q_rot[offset + i] = q0 * cos_theta - q1 * sin_theta;
                q_rot[offset + i + half_rope_dim] = q0 * sin_theta + q1 * cos_theta;
            }
        }

        for head in 0..k_n_heads {
            let offset = pos * k_total_dim + head * head_dim;
            for i in 0..half_rope_dim {
                let theta = abs_pos * freqs[i];
                let cos_theta = theta.cos() * mscale;
                let sin_theta = theta.sin() * mscale;

                let k0 = k_data[offset + i];
                let k1 = k_data[offset + i + half_rope_dim];
                k_rot[offset + i] = k0 * cos_theta - k1 * sin_theta;
                k_rot[offset + i + half_rope_dim] = k0 * sin_theta + k1 * cos_theta;
            }
        }
    }

    (
        DeviceTensor::new(Tensor::new(q_shape.to_vec(), q_rot)),
        DeviceTensor::new(Tensor::new(k_shape.to_vec(), k_rot)),
    )
}

/// LongRoPE parameters: per-dimension frequency factors and magnitude scale.
pub(crate) struct LongRopeParams<'a> {
    pub(crate) factors: &'a [f32],
    pub(crate) mscale: f32,
}

/// Multi-head attention with KV cache for autoregressive generation.
///
/// Only new tokens' Q/K/V are computed; previous K/V are read from cache.
/// RoPE positions are offset by the cache length.
fn multi_head_attention_cached(
    q: DeviceTensor,
    k_new: DeviceTensor,
    v_new: DeviceTensor,
    cache: &mut KvCache,
    layer_idx: usize,
    config: &ModelConfig,
    backend: &dyn ComputeBackend,
    longrope: Option<&LongRopeParams>,
    rope_freq_base: f32,
    swa_window: usize,
) -> Result<DeviceTensor, InferenceError> {
    let n_new = q.shape()[0];
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let pos_offset = cache.len();

    trace!(n_new, pos_offset, layer_idx, "multi_head_attention_cached");

    // Step a: Apply RoPE to Q and K_new with pos_offset
    let (q_proc, k_proc) = if config.position_type == PositionType::RoPE {
        if let Some(params) = longrope {
            // LongRoPE with per-dimension frequency factors and mscale
            rope_neox_with_factors(
                &q, &k_new, pos_offset, rope_freq_base,
                head_dim, config.rope_dim, params.factors, params.mscale, backend,
            )
        } else if config.rope_neox {
            backend.rope_neox(
                &q, &k_new, pos_offset, rope_freq_base, head_dim, config.rope_dim,
            )
        } else {
            backend.rope(
                &q, &k_new, pos_offset, rope_freq_base, head_dim, config.rope_dim,
            )
        }
    } else {
        (q, k_new)
    };

    // =====================================================================
    // GPU FAST PATH: single-token decode with GPU-resident KV cache
    // (skipped when SWA is active — the graph path handles SWA on Metal)
    // =====================================================================
    if n_new == 1 && cache.is_gpu() && swa_window == 0 {
        let total_len = pos_offset + n_new;
        let attn_scale = config.attn_scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

        // Append K_new/V_new to GPU cache (no CPU roundtrip)
        cache.append_gpu(layer_idx, &k_proc, &v_new, n_new, backend)?;

        // Single GPU kernel: fused Q@K^T + softmax + @V across all heads
        let k_full = cache.get_k_gpu(layer_idx);
        let v_full = cache.get_v_gpu(layer_idx);

        return Ok(backend.grouped_attention_decode(
            &q_proc, k_full, v_full, total_len,
            num_heads, num_kv_heads, head_dim,
            attn_scale, config.attn_logit_softcap,
        ));
    }

    // =====================================================================
    // GPU FAST PATH: multi-token prefill with GPU-resident KV cache
    // (skipped when SWA is active — the graph path handles SWA on Metal)
    // =====================================================================
    if n_new > 1 && cache.is_gpu() && swa_window == 0 {
        let total_len = pos_offset + n_new;
        let attn_scale = config.attn_scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

        // Append K_new/V_new to GPU cache (no CPU roundtrip)
        cache.append_gpu(layer_idx, &k_proc, &v_new, n_new, backend)?;

        // Single GPU kernel: batched causal attention across all heads and tokens
        let k_full = cache.get_k_gpu(layer_idx);
        let v_full = cache.get_v_gpu(layer_idx);

        return Ok(backend.batched_causal_attention(
            &q_proc, k_full, v_full,
            n_new, total_len, pos_offset,
            num_heads, num_kv_heads, head_dim,
            attn_scale, config.attn_logit_softcap,
        ));
    }

    // =====================================================================
    // SLOW PATH: prefill without GPU cache, or CPU-resident cache
    // =====================================================================

    // Step b: Download K_new and V_new, append to CPU cache
    let k_host = backend.download(&k_proc);
    let v_host = backend.download(&v_new);
    cache.append(layer_idx, k_host.as_f32(), v_host.as_f32(), n_new)?;

    // Also update GPU cache if present (for subsequent decode steps)
    if cache.is_gpu() {
        cache.append_gpu(layer_idx, &k_proc, &v_new, n_new, backend)?;
    }

    // Step c: Read full cached K/V
    let total_len = pos_offset + n_new;
    let k_full_data = cache.get_k(layer_idx);
    let v_full_data = cache.get_v(layer_idx);

    // Step d: Expand K/V for GQA if needed
    let (k_expanded, v_expanded) = if num_kv_heads < num_heads {
        let k_exp = repeat_kv_heads(k_full_data, total_len, num_kv_heads, num_heads, head_dim);
        let v_exp = repeat_kv_heads(v_full_data, total_len, num_kv_heads, num_heads, head_dim);
        (k_exp, v_exp)
    } else {
        (k_full_data.to_vec(), v_full_data.to_vec())
    };

    let q_host = backend.download(&q_proc);
    let q_data = q_host.as_f32();
    let total_dim = num_heads * head_dim;

    let attn_scale = config.attn_scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

    // Per-head attention
    let mut head_outputs: Vec<Vec<f32>> = Vec::with_capacity(num_heads);

    for h in 0..num_heads {
        // Q: [n_new, head_dim], K: [total_len, head_dim], V: [total_len, head_dim]
        let q_head = extract_head(q_data, n_new, total_dim, h, head_dim);
        let k_head = extract_head(&k_expanded, total_len, total_dim, h, head_dim);
        let v_head = extract_head(&v_expanded, total_len, total_dim, h, head_dim);

        let q_dev = backend.upload(&q_head);
        let k_dev = backend.upload(&k_head);
        let v_dev = backend.upload(&v_head);

        // scores = Q @ K^T: [n_new, head_dim] x [head_dim, total_len] -> [n_new, total_len]
        let scores = backend.matmul_transpose(&q_dev, &k_dev);
        let scores = backend.scale(&scores, attn_scale);

        // Softcap
        let scores = if config.attn_logit_softcap > 0.0 {
            let cap = config.attn_logit_softcap;
            let scaled_down = backend.scale(&scores, 1.0 / cap);
            let tanh_tensor = backend.tanh(&scaled_down);
            backend.scale(&tanh_tensor, cap)
        } else {
            scores
        };

        // Causal + SWA mask: needed during prefill (n_new > 1) or when SWA is active
        let needs_mask = (config.causal && n_new > 1) || swa_window > 0;
        let scores = if needs_mask {
            let scores_tensor = backend.download(&scores);
            let mut scores_data = scores_tensor.as_f32().to_vec();
            for i in 0..n_new {
                let abs_pos = pos_offset + i;
                for j in 0..total_len {
                    let causal_mask = config.causal && j > abs_pos;
                    let swa_mask = swa_window > 0 && abs_pos >= j + swa_window;
                    if causal_mask || swa_mask {
                        scores_data[i * total_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            backend.upload(&Tensor::new(vec![n_new, total_len], scores_data))
        } else {
            scores
        };

        // Softmax
        let probs = backend.softmax(&scores);

        // Weighted sum: [n_new, total_len] x [total_len, head_dim] -> [n_new, head_dim]
        let head_out = backend.matmul(&probs, &v_dev);
        let head_out_host = backend.download(&head_out);
        head_outputs.push(head_out_host.as_f32().to_vec());
    }

    let result = assemble_heads(&head_outputs, n_new, num_heads, head_dim);
    Ok(backend.upload(&result))
}

/// Cached transformer layer forward pass for autoregressive generation.
///
/// Same structure as `pre_norm_layer_forward()` but uses cached attention.
/// Only supports pre-norm architectures (Gemma/LLaMA) since generation is
/// only valid for causal models.
fn transformer_layer_forward_cached(
    input: &DeviceTensor,
    layer: &LayerWeights,
    layer_idx: usize,
    config: &ModelConfig,
    backend: &dyn ComputeBackend,
    cache: &mut KvCache,
    longrope: Option<&LongRopeParams>,
    rope_freq_base: f32,
    swa_window: usize,
) -> Result<DeviceTensor, InferenceError> {
    trace!(layer = layer_idx, "transformer_layer_forward_cached");

    if config.pre_norm {
        // Pre-norm path (Gemma/LLaMA)
        let attn_norm_w = layer.attn_norm_w.as_ref()
            .expect("pre_norm layer requires attn_norm_w");
        let normed = normalize(input, attn_norm_w, layer.attn_norm_b.as_ref(), config, backend);

        let q = linear_forward(&normed, &layer.attn_q, layer.attn_q_bias.as_ref(), backend);
        let k = linear_forward(&normed, &layer.attn_k, layer.attn_k_bias.as_ref(), backend);
        let v = linear_forward(&normed, &layer.attn_v, layer.attn_v_bias.as_ref(), backend);

        // Per-head Q/K RMSNorm (Qwen3)
        let q = if let Some(ref qn_w) = layer.attn_q_norm_w {
            rms_norm_per_head(q, qn_w, config.num_heads, config.head_dim, config.norm_eps, backend)
        } else { q };
        let k = if let Some(ref kn_w) = layer.attn_k_norm_w {
            rms_norm_per_head(k, kn_w, config.num_kv_heads, config.head_dim, config.norm_eps, backend)
        } else { k };

        let attn_out = multi_head_attention_cached(q, k, v, cache, layer_idx, config, backend, longrope, rope_freq_base, swa_window)?;

        let projected = linear_forward(
            &attn_out, &layer.attn_output, layer.attn_output_bias.as_ref(), backend,
        );

        // Post-attention norm (Gemma3: applied to projection BEFORE residual add)
        let projected = if let Some(ref norm_w) = layer.attn_post_norm_w {
            backend.rms_norm(&projected, norm_w, config.norm_eps)
        } else {
            projected
        };

        let residual = backend.add(input, &projected);

        let ffn_norm_w = layer.ffn_norm_w.as_ref()
            .expect("pre_norm layer requires ffn_norm_w");
        let normed2 = normalize(&residual, ffn_norm_w, layer.ffn_norm_b.as_ref(), config, backend);
        let ffn_out = ffn_forward(&normed2, layer, config, backend);

        // Post-FFN norm (Gemma3: applied to FFN output BEFORE residual add)
        let ffn_out = if let Some(ref norm_w) = layer.ffn_post_norm_w {
            backend.rms_norm(&ffn_out, norm_w, config.norm_eps)
        } else {
            ffn_out
        };

        Ok(backend.add(&residual, &ffn_out))
    } else {
        // Post-norm path (BERT) — shouldn't be used for generation,
        // but support it for completeness
        let q = linear_forward(input, &layer.attn_q, layer.attn_q_bias.as_ref(), backend);
        let k = linear_forward(input, &layer.attn_k, layer.attn_k_bias.as_ref(), backend);
        let v = linear_forward(input, &layer.attn_v, layer.attn_v_bias.as_ref(), backend);

        let attn_out = multi_head_attention_cached(q, k, v, cache, layer_idx, config, backend, longrope, rope_freq_base, swa_window)?;

        let projected = linear_forward(
            &attn_out, &layer.attn_output, layer.attn_output_bias.as_ref(), backend,
        );
        let mut residual = backend.add(input, &projected);

        if let Some(ref norm_w) = layer.attn_output_norm_w {
            residual = normalize(&residual, norm_w, layer.attn_output_norm_b.as_ref(), config, backend);
        }

        let ffn_out = ffn_forward(&residual, layer, config, backend);
        let mut output = backend.add(&residual, &ffn_out);

        if let Some(ref norm_w) = layer.ffn_output_norm_w {
            output = normalize(&output, norm_w, layer.ffn_output_norm_b.as_ref(), config, backend);
        }

        Ok(output)
    }
}

/// Model forward pass with KV cache for autoregressive generation.
///
/// Same pipeline as `model_forward()` but:
/// - Uses `cache.len()` as `pos_offset` for learned position embeddings
/// - Calls `transformer_layer_forward_cached()` per layer
/// - Calls `cache.advance(n_tokens)` after all layers
/// - Returns `[n_tokens, hidden_size]`
pub fn model_forward_step(
    input_ids: &[u32],
    weights: &ModelWeights,
    config: &ModelConfig,
    backend: &dyn ComputeBackend,
    cache: &mut KvCache,
) -> Result<DeviceTensor, InferenceError> {
    let n_tokens = input_ids.len();
    let pos_offset = cache.len();
    debug!(n_tokens, pos_offset, "model_forward_step");

    // 1. Validate input_ids
    for &id in input_ids {
        if (id as usize) >= config.vocab_size {
            return Err(InferenceError::Model(format!(
                "input token ID {} exceeds vocab_size {}",
                id, config.vocab_size
            )));
        }
    }

    // 2. Embedding lookup
    let mut hidden = backend.embedding_lookup(&weights.token_embedding, input_ids);

    // 3. Embedding scaling
    if (config.embedding_scale - 1.0).abs() > f32::EPSILON {
        hidden = backend.scale(&hidden, config.embedding_scale);
    }

    // 4. Add position embeddings (Learned, e.g., BERT)
    if config.position_type == PositionType::Learned {
        if let Some(ref pos_emb) = weights.position_embedding {
            let pos_ids: Vec<u32> = (pos_offset..pos_offset + n_tokens)
                .map(|p| p as u32)
                .collect();
            let pos_embeddings = backend.embedding_lookup(pos_emb, &pos_ids);
            hidden = backend.add(&hidden, &pos_embeddings);
        }
    }

    // 5. Embedding normalization (BERT only)
    if let Some(ref norm_w) = weights.embedding_norm_w {
        hidden = normalize(
            &hidden, norm_w, weights.embedding_norm_b.as_ref(), config, backend,
        );
    }

    // 6. Determine LongRoPE frequency factors and mscale (Phi-3.5 etc.)
    // llama.cpp selects short vs long factors based on configured n_ctx (max_seq_len),
    // not the current sequence position.
    // mscale = rope_scaling_attn_factor (applied to cos/sin in RoPE).
    let rope_factors_data: Option<Vec<f32>> = if config.rope_scaling_original_ctx > 0 {
        let factors_tensor = if config.max_seq_len > config.rope_scaling_original_ctx {
            weights.rope_factors_long.as_ref()
        } else {
            weights.rope_factors_short.as_ref()
        };
        factors_tensor.map(|t| {
            let host = backend.download(t);
            host.as_f32().to_vec()
        })
    } else {
        None
    };
    let longrope_params = rope_factors_data.as_ref().map(|factors| LongRopeParams {
        factors: factors.as_slice(),
        mscale: config.rope_scaling_attn_factor,
    });

    // 7. Run through all transformer layers with cache
    for (i, layer) in weights.layers.iter().enumerate() {
        trace!(layer = i, "Running cached layer");
        let layer_rope_base = if config.swa_layers.get(i).copied().unwrap_or(false) {
            config.rope_freq_base_swa
        } else {
            config.rope_freq_base
        };
        let layer_swa_window = if config.swa_layers.get(i).copied().unwrap_or(false) {
            config.swa_window
        } else {
            0
        };
        hidden = transformer_layer_forward_cached(
            &hidden, layer, i, config, backend, cache, longrope_params.as_ref(),
            layer_rope_base, layer_swa_window,
        )?;
    }

    // 8. Advance cache position after all layers processed
    cache.advance(n_tokens);

    // 9. Final output normalization (absent for BERT)
    if let Some(ref norm_w) = weights.output_norm_w {
        hidden = normalize(
            &hidden,
            norm_w,
            weights.output_norm_b.as_ref(),
            config,
            backend,
        );
    }

    Ok(hidden)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::model::config::*;

    fn cpu() -> CpuBackend {
        CpuBackend::default()
    }

    fn dt(tensor: Tensor) -> DeviceTensor {
        DeviceTensor::new(tensor)
    }

    /// Create a Gemma-style config for testing.
    fn gemma_config(hidden_size: usize, num_heads: usize, head_dim: usize, ffn_hidden: usize) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Gemma3,
            arch_name: "gemma3".to_string(),
            hidden_size,
            num_layers: 1,
            num_heads,
            num_kv_heads: num_heads, // MHA by default
            head_dim,
            ffn_hidden,
            vocab_size: 16,
            max_seq_len: 128,
            norm_type: NormType::RMSNorm,
            norm_eps: 1e-6,
            activation: Activation::SwiGLU,
            position_type: PositionType::RoPE,
            rope_freq_base: 10000.0,
            rope_dim: head_dim,
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
            swa_window: 0,
            swa_layers: vec![],
            rope_freq_base_swa: 10000.0,
        }
    }

    /// Create a BERT-style config for testing.
    fn bert_config(hidden_size: usize, num_heads: usize, head_dim: usize, ffn_hidden: usize) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Bert,
            arch_name: "bert".to_string(),
            hidden_size,
            num_layers: 1,
            num_heads,
            num_kv_heads: num_heads,
            head_dim,
            ffn_hidden,
            vocab_size: 16,
            max_seq_len: 128,
            norm_type: NormType::LayerNorm,
            norm_eps: 1e-12,
            activation: Activation::GELU,
            position_type: PositionType::Learned,
            rope_freq_base: 10000.0,
            rope_dim: head_dim,
            rope_neox: false,
            rope_scaling_original_ctx: 0,
            rope_scaling_attn_factor: 1.0,
            causal: false,
            attn_logit_softcap: 0.0,
            attn_scale: None,
            embedding_scale: 1.0,
            pooling_type: PoolingType::Mean,
            has_ffn_gate: false,
            has_bias: true,
            pre_norm: false,
            swa_window: 0,
            swa_layers: vec![],
            rope_freq_base_swa: 10000.0,
        }
    }

    /// Create identity-like weight matrix [out, in] for F32.
    fn identity_weight(size: usize) -> DeviceTensor {
        let mut data = vec![0.0f32; size * size];
        for i in 0..size {
            data[i * size + i] = 1.0;
        }
        dt(Tensor::new(vec![size, size], data))
    }

    /// Create a zero weight matrix [out, in].
    fn zero_weight(out_dim: usize, in_dim: usize) -> DeviceTensor {
        dt(Tensor::new(vec![out_dim, in_dim], vec![0.0f32; out_dim * in_dim]))
    }

    /// Create a ones weight vector [dim].
    fn ones_weight(dim: usize) -> DeviceTensor {
        dt(Tensor::new(vec![dim], vec![1.0f32; dim]))
    }

    /// Create a zeros bias vector [dim].
    fn zeros_bias(dim: usize) -> DeviceTensor {
        dt(Tensor::new(vec![dim], vec![0.0f32; dim]))
    }

    /// Create minimal LayerWeights for a Gemma-style (SwiGLU, RMSNorm, pre-norm) layer.
    fn gemma_layer_weights(hidden_size: usize, ffn_hidden: usize) -> LayerWeights {
        LayerWeights {
            attn_norm_w: Some(ones_weight(hidden_size)),
            attn_norm_b: None,
            ffn_norm_w: Some(ones_weight(hidden_size)),
            ffn_norm_b: None,
            attn_output_norm_w: None,
            attn_output_norm_b: None,
            ffn_output_norm_w: None,
            ffn_output_norm_b: None,
            attn_q: identity_weight(hidden_size),
            attn_k: identity_weight(hidden_size),
            attn_v: identity_weight(hidden_size),
            attn_output: identity_weight(hidden_size),
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            attn_output_bias: None,
            ffn_up: zero_weight(ffn_hidden, hidden_size),
            ffn_down: zero_weight(hidden_size, ffn_hidden),
            ffn_gate: Some(zero_weight(ffn_hidden, hidden_size)),
            ffn_up_bias: None,
            ffn_down_bias: None,
            attn_q_norm_w: None,
            attn_k_norm_w: None,
            attn_post_norm_w: None,
            ffn_post_norm_w: None,
        }
    }

    /// Create minimal LayerWeights for a BERT-style (GELU, LayerNorm, post-norm) layer.
    fn bert_layer_weights(hidden_size: usize, ffn_hidden: usize) -> LayerWeights {
        LayerWeights {
            attn_norm_w: None,
            attn_norm_b: None,
            ffn_norm_w: None,
            ffn_norm_b: None,
            attn_output_norm_w: Some(ones_weight(hidden_size)),
            attn_output_norm_b: Some(zeros_bias(hidden_size)),
            ffn_output_norm_w: Some(ones_weight(hidden_size)),
            ffn_output_norm_b: Some(zeros_bias(hidden_size)),
            attn_q: identity_weight(hidden_size),
            attn_k: identity_weight(hidden_size),
            attn_v: identity_weight(hidden_size),
            attn_output: identity_weight(hidden_size),
            attn_q_bias: Some(zeros_bias(hidden_size)),
            attn_k_bias: Some(zeros_bias(hidden_size)),
            attn_v_bias: Some(zeros_bias(hidden_size)),
            attn_output_bias: Some(zeros_bias(hidden_size)),
            ffn_up: zero_weight(ffn_hidden, hidden_size),
            ffn_down: zero_weight(hidden_size, ffn_hidden),
            ffn_gate: None,
            ffn_up_bias: Some(zeros_bias(ffn_hidden)),
            ffn_down_bias: Some(zeros_bias(hidden_size)),
            attn_q_norm_w: None,
            attn_k_norm_w: None,
            attn_post_norm_w: None,
            ffn_post_norm_w: None,
        }
    }

    // ====================================================================
    // linear_forward tests
    // ====================================================================

    #[test]
    fn test_linear_forward_f32_identity() {
        let b = cpu();
        let input = dt(Tensor::new(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
        let weight = identity_weight(4);
        let result = linear_forward(&input, &weight, None, &b);
        assert_eq!(result.shape(), &[2, 4]);
        let data = result.as_tensor().as_f32();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[7] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_forward_with_bias() {
        let b = cpu();
        let input = dt(Tensor::new(vec![1, 4], vec![1.0, 0.0, 0.0, 0.0]));
        let weight = identity_weight(4);
        let bias = dt(Tensor::new(vec![4], vec![10.0, 20.0, 30.0, 40.0]));
        let result = linear_forward(&input, &weight, Some(&bias), &b);
        let data = result.as_tensor().as_f32();
        assert!((data[0] - 11.0).abs() < 1e-5);
        assert!((data[1] - 20.0).abs() < 1e-5);
        assert!((data[2] - 30.0).abs() < 1e-5);
        assert!((data[3] - 40.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_forward_quantized() {
        let b = cpu();
        // Create a 1x32 input and 1x32 Q8_0 weight
        let input = dt(Tensor::new(vec![1, 32], vec![1.0f32; 32]));
        let scale = half::f16::from_f32(1.0 / 127.0);
        let mut raw = Vec::new();
        raw.extend_from_slice(&scale.to_bits().to_le_bytes());
        for _ in 0..32 {
            raw.push(127u8);
        }
        let weight = dt(Tensor::from_quantized(vec![1, 32], TensorDtype::Q8_0, raw));
        let result = linear_forward(&input, &weight, None, &b);
        assert_eq!(result.shape(), &[1, 1]);
        let val = result.as_tensor().as_f32()[0];
        assert!((val - 32.0).abs() < 0.5, "Q8_0 linear_forward: got {}", val);
    }

    // ====================================================================
    // extract_head and assemble_heads tests
    // ====================================================================

    #[test]
    fn test_extract_and_assemble_heads() {
        // seq_len=2, num_heads=2, head_dim=3, total_dim=6
        let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        // [1, 2, 3, 4, 5, 6,   7, 8, 9, 10, 11, 12]
        // pos 0: head0=[1,2,3] head1=[4,5,6]
        // pos 1: head0=[7,8,9] head1=[10,11,12]

        let head0 = extract_head(&data, 2, 6, 0, 3);
        assert_eq!(head0.as_f32(), &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);

        let head1 = extract_head(&data, 2, 6, 1, 3);
        assert_eq!(head1.as_f32(), &[4.0, 5.0, 6.0, 10.0, 11.0, 12.0]);

        let head_outputs = vec![
            head0.as_f32().to_vec(),
            head1.as_f32().to_vec(),
        ];
        let assembled = assemble_heads(&head_outputs, 2, 2, 3);
        assert_eq!(assembled.as_f32(), &data[..]);
    }

    // ====================================================================
    // repeat_kv_heads tests
    // ====================================================================

    #[test]
    fn test_repeat_kv_heads() {
        // 1 KV head -> 2 Q heads, head_dim=2, seq_len=1
        let data = vec![1.0, 2.0]; // [seq_len=1, kv_heads=1 * head_dim=2]
        let expanded = repeat_kv_heads(&data, 1, 1, 2, 2);
        // Should be [1.0, 2.0, 1.0, 2.0]
        assert_eq!(expanded, vec![1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_repeat_kv_heads_2_to_4() {
        // 2 KV heads -> 4 Q heads, head_dim=2, seq_len=1
        let data = vec![1.0, 2.0, 3.0, 4.0]; // kv0=[1,2], kv1=[3,4]
        let expanded = repeat_kv_heads(&data, 1, 2, 4, 2);
        // kv0 -> q0, q1: [1,2,1,2], kv1 -> q2, q3: [3,4,3,4]
        assert_eq!(expanded, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn test_repeat_kv_heads_noop() {
        // Same number of KV heads as Q heads -> no change
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let expanded = repeat_kv_heads(&data, 1, 2, 2, 2);
        assert_eq!(expanded, data);
    }

    // ====================================================================
    // transformer_layer_forward tests
    // ====================================================================

    #[test]
    fn test_transformer_layer_output_shape_gemma() {
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let config = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);
        let layer = gemma_layer_weights(hidden_size, ffn_hidden);

        let seq_len = 3;
        let input_data: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32) * 0.1).collect();
        let input = dt(Tensor::new(vec![seq_len, hidden_size], input_data));
        let mask = vec![1.0f32; seq_len];

        let output = transformer_layer_forward(&input, &layer, &config, &b, 0, &mask, None);
        assert_eq!(output.shape(), &[seq_len, hidden_size]);

        // Verify all values are finite
        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "Output contains non-finite value: {}", v);
        }
    }

    #[test]
    fn test_transformer_layer_output_shape_bert() {
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let config = bert_config(hidden_size, num_heads, head_dim, ffn_hidden);
        let layer = bert_layer_weights(hidden_size, ffn_hidden);

        let seq_len = 3;
        let input_data: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32) * 0.1).collect();
        let input = dt(Tensor::new(vec![seq_len, hidden_size], input_data));
        let mask = vec![1.0f32; seq_len];

        let output = transformer_layer_forward(&input, &layer, &config, &b, 0, &mask, None);
        assert_eq!(output.shape(), &[seq_len, hidden_size]);

        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "Output contains non-finite value: {}", v);
        }
    }

    #[test]
    fn test_residual_with_zero_ffn() {
        // With zero FFN weights and identity attention weights, the residual
        // connections should dominate the output.
        let b = cpu();
        let hidden_size = 4;
        let num_heads = 1;
        let head_dim = 4;
        let ffn_hidden = 8;

        let mut config = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);
        config.causal = false; // bidirectional for simpler testing

        let layer = gemma_layer_weights(hidden_size, ffn_hidden);

        let input_data = vec![1.0, 2.0, 3.0, 4.0]; // seq_len=1
        let input = dt(Tensor::new(vec![1, hidden_size], input_data.clone()));
        let mask = vec![1.0f32; 1];

        let output = transformer_layer_forward(&input, &layer, &config, &b, 0, &mask, None);
        assert_eq!(output.shape(), &[1, hidden_size]);

        // With identity attention weights and zero FFN, the attention output should be
        // the same as the normalized input (which then gets added to the original input).
        // The output should be finite and reasonably close to the input
        // (since FFN contributes zero, so output = residual1 + 0 = input + attn_projection(attn_output))
        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "Non-finite value in output");
        }
    }

    #[test]
    fn test_rope_changes_qk_at_different_positions() {
        // RoPE is designed so attention scores depend only on relative positions,
        // meaning a uniform pos_offset change won't alter scores. Instead, we
        // verify that RoPE actually rotates Q and K differently at different positions
        // by directly testing the rope function.
        let b = cpu();
        let head_dim = 4;

        let q = dt(Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]));
        let k = dt(Tensor::new(vec![1, 4], vec![5.0, 6.0, 7.0, 8.0]));

        let (q_rot0, k_rot0) = b.rope(&q, &k, 0, 10000.0, head_dim, head_dim);
        let (q_rot5, k_rot5) = b.rope(&q, &k, 5, 10000.0, head_dim, head_dim);

        // At position 0, cos(0)=1, sin(0)=0, so rotation is identity
        let q0 = q_rot0.as_tensor().as_f32();
        assert!((q0[0] - 1.0).abs() < 1e-5, "RoPE at pos 0 should be identity for Q");

        // At position 5, the rotation should be different
        let q5 = q_rot5.as_tensor().as_f32();
        let mut any_different = false;
        for i in 0..head_dim {
            if (q0[i] - q5[i]).abs() > 1e-6 {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "RoPE at different positions should produce different Q vectors");

        // Also verify K is rotated differently
        let k0 = k_rot0.as_tensor().as_f32();
        let k5 = k_rot5.as_tensor().as_f32();
        let mut k_different = false;
        for i in 0..head_dim {
            if (k0[i] - k5[i]).abs() > 1e-6 {
                k_different = true;
                break;
            }
        }
        assert!(k_different, "RoPE at different positions should produce different K vectors");
    }

    #[test]
    fn test_rope_vs_no_rope_produces_different_outputs() {
        // Verify that enabling RoPE vs using learned positions (no rotation)
        // produces different attention outputs for the same input.
        let b = cpu();
        let hidden_size = 4;
        let num_heads = 1;
        let head_dim = 4;
        let ffn_hidden = 8;

        let config_rope = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);
        let mut config_no_rope = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);
        config_no_rope.position_type = PositionType::Learned;

        let layer = gemma_layer_weights(hidden_size, ffn_hidden);

        let seq_len = 2;
        let input_data: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let input = dt(Tensor::new(vec![seq_len, hidden_size], input_data));
        let mask = vec![1.0f32; seq_len];

        let output_rope = transformer_layer_forward(&input, &layer, &config_rope, &b, 0, &mask, None);
        let output_no_rope = transformer_layer_forward(&input, &layer, &config_no_rope, &b, 0, &mask, None);

        let data_r = output_rope.as_tensor().as_f32();
        let data_nr = output_no_rope.as_tensor().as_f32();

        let mut any_different = false;
        for i in 0..data_r.len() {
            if (data_r[i] - data_nr[i]).abs() > 1e-6 {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "RoPE should change outputs compared to no-RoPE");
    }

    #[test]
    fn test_causal_mask_prevents_future_attention() {
        let b = cpu();
        let hidden_size = 4;
        let num_heads = 1;
        let head_dim = 4;
        let ffn_hidden = 8;

        // Causal config
        let config_causal = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);

        // Non-causal config
        let mut config_bidir = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);
        config_bidir.causal = false;

        let layer = gemma_layer_weights(hidden_size, ffn_hidden);

        let seq_len = 3;
        let input_data: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32) * 0.1).collect();
        let input = dt(Tensor::new(vec![seq_len, hidden_size], input_data));
        let mask = vec![1.0f32; seq_len];

        let output_causal = transformer_layer_forward(&input, &layer, &config_causal, &b, 0, &mask, None);
        let output_bidir = transformer_layer_forward(&input, &layer, &config_bidir, &b, 0, &mask, None);

        // Outputs should differ because causal mask restricts attention
        let data_c = output_causal.as_tensor().as_f32();
        let data_b = output_bidir.as_tensor().as_f32();

        let mut any_different = false;
        for i in 0..data_c.len() {
            if (data_c[i] - data_b[i]).abs() > 1e-6 {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "Causal and bidirectional should produce different outputs for seq_len > 1");
    }

    #[test]
    fn test_gelu_ffn_path() {
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let config = bert_config(hidden_size, num_heads, head_dim, ffn_hidden);

        // Create layer with non-zero FFN weights to exercise the GELU path
        let mut layer = bert_layer_weights(hidden_size, ffn_hidden);
        // Set ffn_up to have some non-zero values
        let up_data: Vec<f32> = (0..ffn_hidden * hidden_size).map(|i| (i as f32) * 0.01).collect();
        layer.ffn_up = dt(Tensor::new(vec![ffn_hidden, hidden_size], up_data));

        let input_data: Vec<f32> = (0..hidden_size).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let input = dt(Tensor::new(vec![1, hidden_size], input_data));
        let mask = vec![1.0f32; 1];

        let output = transformer_layer_forward(&input, &layer, &config, &b, 0, &mask, None);
        assert_eq!(output.shape(), &[1, hidden_size]);
        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "GELU FFN path produced non-finite value");
        }
    }

    #[test]
    fn test_swiglu_ffn_path() {
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let config = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);

        // Create layer with non-zero FFN weights to exercise the SwiGLU path
        let mut layer = gemma_layer_weights(hidden_size, ffn_hidden);
        let gate_data: Vec<f32> = (0..ffn_hidden * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let up_data: Vec<f32> = (0..ffn_hidden * hidden_size).map(|i| (i as f32) * 0.01).collect();
        layer.ffn_gate = Some(dt(Tensor::new(vec![ffn_hidden, hidden_size], gate_data)));
        layer.ffn_up = dt(Tensor::new(vec![ffn_hidden, hidden_size], up_data));

        let input_data: Vec<f32> = (0..hidden_size).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let input = dt(Tensor::new(vec![1, hidden_size], input_data));
        let mask = vec![1.0f32; 1];

        let output = transformer_layer_forward(&input, &layer, &config, &b, 0, &mask, None);
        assert_eq!(output.shape(), &[1, hidden_size]);
        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "SwiGLU FFN path produced non-finite value");
        }
    }

    #[test]
    fn test_gqa_layer_forward() {
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 2;
        let ffn_hidden = 16;

        let mut config = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);
        config.num_kv_heads = num_kv_heads;

        // Need K and V projection weights with correct output dim for kv_heads
        let kv_dim = num_kv_heads * head_dim; // 4
        let layer = LayerWeights {
            attn_norm_w: Some(ones_weight(hidden_size)),
            attn_norm_b: None,
            ffn_norm_w: Some(ones_weight(hidden_size)),
            ffn_norm_b: None,
            attn_output_norm_w: None,
            attn_output_norm_b: None,
            ffn_output_norm_w: None,
            ffn_output_norm_b: None,
            attn_q: identity_weight(hidden_size), // [8, 8] -> outputs [seq, 8]
            attn_k: zero_weight(kv_dim, hidden_size), // [4, 8] -> outputs [seq, 4]
            attn_v: zero_weight(kv_dim, hidden_size), // [4, 8] -> outputs [seq, 4]
            attn_output: identity_weight(hidden_size),
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            attn_output_bias: None,
            ffn_up: zero_weight(ffn_hidden, hidden_size),
            ffn_down: zero_weight(hidden_size, ffn_hidden),
            ffn_gate: Some(zero_weight(ffn_hidden, hidden_size)),
            ffn_up_bias: None,
            ffn_down_bias: None,
            attn_q_norm_w: None,
            attn_k_norm_w: None,
            attn_post_norm_w: None,
            ffn_post_norm_w: None,
        };

        let seq_len = 2;
        let input_data: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32) * 0.1).collect();
        let input = dt(Tensor::new(vec![seq_len, hidden_size], input_data));
        let mask = vec![1.0f32; seq_len];

        let output = transformer_layer_forward(&input, &layer, &config, &b, 0, &mask, None);
        assert_eq!(output.shape(), &[seq_len, hidden_size]);
        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "GQA forward pass produced non-finite value: {}", v);
        }
    }

    // ====================================================================
    // model_forward tests
    // ====================================================================

    #[test]
    fn test_model_forward_output_shape_gemma() {
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let vocab_size = 16;
        let config = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let weights = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![gemma_layer_weights(hidden_size, ffn_hidden)],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let input_ids = &[1u32, 3, 5];
        let attention_mask = &[1.0f32, 1.0, 1.0];

        let output = model_forward(input_ids, attention_mask, &weights, &config, &b, 0).unwrap();
        assert_eq!(output.shape(), &[3, hidden_size]);
        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "model_forward produced non-finite value");
        }
    }

    #[test]
    fn test_model_forward_output_shape_bert() {
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let vocab_size = 16;
        let max_seq_len = 128;
        let config = bert_config(hidden_size, num_heads, head_dim, ffn_hidden);

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let pos_emb_data: Vec<f32> = (0..max_seq_len * hidden_size).map(|i| (i as f32) * 0.001).collect();

        let weights = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: Some(dt(Tensor::new(vec![max_seq_len, hidden_size], pos_emb_data))),
            token_type_embedding: None,
            embedding_norm_w: Some(ones_weight(hidden_size)),
            embedding_norm_b: Some(zeros_bias(hidden_size)),
            layers: vec![bert_layer_weights(hidden_size, ffn_hidden)],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: Some(zeros_bias(hidden_size)),
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let input_ids = &[0u32, 2, 4, 6];
        let attention_mask = &[1.0f32, 1.0, 1.0, 1.0];

        let output = model_forward(input_ids, attention_mask, &weights, &config, &b, 0).unwrap();
        assert_eq!(output.shape(), &[4, hidden_size]);
        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "BERT model_forward produced non-finite value");
        }
    }

    #[test]
    fn test_model_forward_embedding_scale() {
        let b = cpu();
        let hidden_size = 4;
        let num_heads = 1;
        let head_dim = 4;
        let ffn_hidden = 8;
        let vocab_size = 4;

        let mut config = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);
        config.embedding_scale = 2.0;
        config.vocab_size = vocab_size;

        let embedding_data = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let weights = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![gemma_layer_weights(hidden_size, ffn_hidden)],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let input_ids = &[0u32];
        let attention_mask = &[1.0f32];

        let output = model_forward(input_ids, attention_mask, &weights, &config, &b, 0).unwrap();
        assert_eq!(output.shape(), &[1, hidden_size]);
        // With scale=2.0, the embedding is doubled before going through the layers.
        // Just verify the output is finite and the function runs without error.
        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "Scaled embedding model_forward produced non-finite value");
        }
    }

    #[test]
    fn test_model_forward_multiple_layers() {
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let vocab_size = 16;

        let mut config = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);
        config.num_layers = 3;

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let weights = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![
                gemma_layer_weights(hidden_size, ffn_hidden),
                gemma_layer_weights(hidden_size, ffn_hidden),
                gemma_layer_weights(hidden_size, ffn_hidden),
            ],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let input_ids = &[1u32, 5];
        let attention_mask = &[1.0f32, 1.0];

        let output = model_forward(input_ids, attention_mask, &weights, &config, &b, 0).unwrap();
        assert_eq!(output.shape(), &[2, hidden_size]);
        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "Multi-layer model_forward produced non-finite value");
        }
    }

    #[test]
    fn test_model_forward_single_token() {
        let b = cpu();
        let hidden_size = 4;
        let num_heads = 1;
        let head_dim = 4;
        let ffn_hidden = 8;
        let vocab_size = 8;
        let config = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size).map(|i| (i as f32) * 0.1).collect();
        let weights = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![gemma_layer_weights(hidden_size, ffn_hidden)],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let input_ids = &[3u32];
        let attention_mask = &[1.0f32];

        let output = model_forward(input_ids, attention_mask, &weights, &config, &b, 0).unwrap();
        assert_eq!(output.shape(), &[1, hidden_size]);
        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "Single token model_forward produced non-finite value");
        }
    }

    #[test]
    fn test_softcap_changes_output() {
        let b = cpu();
        let hidden_size = 4;
        let num_heads = 1;
        let head_dim = 4;
        let ffn_hidden = 8;

        let mut config_no_cap = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);
        config_no_cap.attn_logit_softcap = 0.0;
        config_no_cap.causal = false;

        let mut config_with_cap = config_no_cap.clone();
        config_with_cap.attn_logit_softcap = 50.0;

        let layer = gemma_layer_weights(hidden_size, ffn_hidden);

        let seq_len = 2;
        let input_data: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32 + 1.0) * 5.0).collect();
        let input = dt(Tensor::new(vec![seq_len, hidden_size], input_data));
        let mask = vec![1.0f32; seq_len];

        let output_no_cap = transformer_layer_forward(&input, &layer, &config_no_cap, &b, 0, &mask, None);
        let output_with_cap = transformer_layer_forward(&input, &layer, &config_with_cap, &b, 0, &mask, None);

        // Softcap should change outputs (unless scores are very small)
        let data_no = output_no_cap.as_tensor().as_f32();
        let data_cap = output_with_cap.as_tensor().as_f32();
        let mut any_diff = false;
        for i in 0..data_no.len() {
            if (data_no[i] - data_cap[i]).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "Softcap should change transformer outputs");
    }

    // ====================================================================
    // multi_head_attention direct tests
    // ====================================================================

    #[test]
    fn test_multi_head_attention_output_shape() {
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let head_dim = 4;

        let mut config = gemma_config(hidden_size, num_heads, head_dim, 16);
        config.causal = false;
        config.position_type = PositionType::Learned; // skip RoPE for this test

        let seq_len = 3;
        let q = dt(Tensor::new(vec![seq_len, hidden_size], vec![0.1; seq_len * hidden_size]));
        let k = dt(Tensor::new(vec![seq_len, hidden_size], vec![0.1; seq_len * hidden_size]));
        let v = dt(Tensor::new(vec![seq_len, hidden_size], vec![0.1; seq_len * hidden_size]));
        let mask = vec![1.0f32; seq_len];

        let output = multi_head_attention(q, k, v, &config, &b, 0, &mask, None);
        assert_eq!(output.shape(), &[seq_len, hidden_size]);
    }

    #[test]
    fn test_multi_head_attention_uniform_produces_uniform() {
        // When Q, K, V are all the same uniform value, all attention probs
        // should be equal (1/seq_len), and the output should equal the
        // average of V (which is the same uniform value).
        let b = cpu();
        let hidden_size = 4;
        let num_heads = 1;
        let head_dim = 4;

        let mut config = gemma_config(hidden_size, num_heads, head_dim, 8);
        config.causal = false;
        config.position_type = PositionType::Learned;

        let seq_len = 3;
        let val = 0.5f32;
        let q = dt(Tensor::new(vec![seq_len, hidden_size], vec![val; seq_len * hidden_size]));
        let k = dt(Tensor::new(vec![seq_len, hidden_size], vec![val; seq_len * hidden_size]));
        let v = dt(Tensor::new(vec![seq_len, hidden_size], vec![val; seq_len * hidden_size]));
        let mask = vec![1.0f32; seq_len];

        let output = multi_head_attention(q, k, v, &config, &b, 0, &mask, None);
        let data = output.as_tensor().as_f32();

        // All outputs should be close to val
        for &v_out in data {
            assert!(
                (v_out - val).abs() < 1e-4,
                "Uniform attention should produce uniform output, got {}",
                v_out
            );
        }
    }

    #[test]
    fn test_attention_mask_excludes_padding() {
        // Verify that masked positions (mask=0) get zero attention weight.
        let b = cpu();
        let hidden_size = 4;
        let num_heads = 1;
        let head_dim = 4;

        let mut config = gemma_config(hidden_size, num_heads, head_dim, 8);
        config.causal = false;
        config.position_type = PositionType::Learned;

        // seq_len = 3, but last token is padding
        let seq_len = 3;
        let mask_all = vec![1.0f32; seq_len];
        let mask_pad = vec![1.0f32, 1.0, 0.0]; // last token is padding

        // Use distinct V values so the output differs when padding is excluded
        let q_data: Vec<f32> = vec![0.1; seq_len * hidden_size];
        let k_data: Vec<f32> = vec![0.1; seq_len * hidden_size];
        let mut v_data = vec![1.0f32; seq_len * hidden_size];
        // Make the padding token's V values large and distinct
        for i in 0..hidden_size {
            v_data[2 * hidden_size + i] = 100.0;
        }

        let q1 = dt(Tensor::new(vec![seq_len, hidden_size], q_data.clone()));
        let k1 = dt(Tensor::new(vec![seq_len, hidden_size], k_data.clone()));
        let v1 = dt(Tensor::new(vec![seq_len, hidden_size], v_data.clone()));
        let out_all = multi_head_attention(q1, k1, v1, &config, &b, 0, &mask_all, None);

        let q2 = dt(Tensor::new(vec![seq_len, hidden_size], q_data));
        let k2 = dt(Tensor::new(vec![seq_len, hidden_size], k_data));
        let v2 = dt(Tensor::new(vec![seq_len, hidden_size], v_data));
        let out_pad = multi_head_attention(q2, k2, v2, &config, &b, 0, &mask_pad, None);

        // The outputs should differ because the masked version excludes the padding token
        let data_all = out_all.as_tensor().as_f32();
        let data_pad = out_pad.as_tensor().as_f32();
        let mut any_diff = false;
        for i in 0..data_all.len() {
            if (data_all[i] - data_pad[i]).abs() > 1e-4 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "Attention mask should change outputs when padding token has different V values");
    }

    #[test]
    fn test_model_forward_input_id_out_of_range() {
        let b = cpu();
        let hidden_size = 4;
        let num_heads = 1;
        let head_dim = 4;
        let ffn_hidden = 8;
        let vocab_size = 8;
        let config = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size).map(|i| (i as f32) * 0.1).collect();
        let weights = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![gemma_layer_weights(hidden_size, ffn_hidden)],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        // Token ID 100 exceeds vocab_size=8 (note: config has vocab_size=16, override)
        let mut config = config;
        config.vocab_size = vocab_size;
        let input_ids = &[100u32]; // out of range
        let attention_mask = &[1.0f32];

        let result = model_forward(input_ids, attention_mask, &weights, &config, &b, 0);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("exceeds vocab_size"), "Error: {}", err_msg);
    }

    // ====================================================================
    // Cached forward pass tests (model_forward_step)
    // ====================================================================

    use crate::model::cache::KvCache;

    #[test]
    fn test_model_forward_step_output_shape() {
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let vocab_size = 16;
        let config = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let weights = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![gemma_layer_weights(hidden_size, ffn_hidden)],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let mut cache = KvCache::new(&config);
        let input_ids = &[1u32, 3, 5];
        let output = model_forward_step(input_ids, &weights, &config, &b, &mut cache).unwrap();
        assert_eq!(output.shape(), &[3, hidden_size]);
        assert_eq!(cache.len(), 3);

        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "model_forward_step produced non-finite value");
        }
    }

    #[test]
    fn test_model_forward_step_single_token_decode() {
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let vocab_size = 16;
        let config = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let weights = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![gemma_layer_weights(hidden_size, ffn_hidden)],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let mut cache = KvCache::new(&config);

        // Prefill with 3 tokens
        let _ = model_forward_step(&[1, 3, 5], &weights, &config, &b, &mut cache).unwrap();
        assert_eq!(cache.len(), 3);

        // Decode 1 token
        let output = model_forward_step(&[7], &weights, &config, &b, &mut cache).unwrap();
        assert_eq!(output.shape(), &[1, hidden_size]);
        assert_eq!(cache.len(), 4);

        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "Single token decode produced non-finite value");
        }
    }

    #[test]
    fn test_cached_vs_uncached_prefill_match() {
        // Full sequence through model_forward() should match
        // the same sequence through model_forward_step() (prefill).
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let vocab_size = 16;
        let config = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let weights = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![gemma_layer_weights(hidden_size, ffn_hidden)],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let input_ids = &[1u32, 3, 5];
        let attention_mask = &[1.0f32, 1.0, 1.0];

        // Uncached
        let uncached = model_forward(input_ids, attention_mask, &weights, &config, &b, 0).unwrap();

        // Cached (prefill)
        let mut cache = KvCache::new(&config);
        let cached = model_forward_step(input_ids, &weights, &config, &b, &mut cache).unwrap();

        let uncached_data = uncached.as_tensor().as_f32();
        let cached_data = cached.as_tensor().as_f32();
        assert_eq!(uncached_data.len(), cached_data.len());
        for i in 0..uncached_data.len() {
            assert!(
                (uncached_data[i] - cached_data[i]).abs() < 1e-4,
                "Mismatch at index {}: uncached={} cached={}",
                i, uncached_data[i], cached_data[i]
            );
        }
    }

    #[test]
    fn test_model_forward_step_multiple_layers() {
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let vocab_size = 16;

        let mut config = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);
        config.num_layers = 2;

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let weights = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![
                gemma_layer_weights(hidden_size, ffn_hidden),
                gemma_layer_weights(hidden_size, ffn_hidden),
            ],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let mut cache = KvCache::new(&config);

        // Prefill
        let out1 = model_forward_step(&[1, 3], &weights, &config, &b, &mut cache).unwrap();
        assert_eq!(out1.shape(), &[2, hidden_size]);
        assert_eq!(cache.len(), 2);

        // Decode
        let out2 = model_forward_step(&[5], &weights, &config, &b, &mut cache).unwrap();
        assert_eq!(out2.shape(), &[1, hidden_size]);
        assert_eq!(cache.len(), 3);

        for &v in out2.as_tensor().as_f32() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_model_forward_step_gqa() {
        // Test cached forward pass with GQA (num_kv_heads < num_heads)
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 2; // 4 heads * 2 dim = 8 = hidden_size
        let ffn_hidden = 16;
        let vocab_size = 16;

        let mut config = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);
        config.num_kv_heads = num_kv_heads;

        let kv_dim = num_kv_heads * head_dim; // 4
        let layer = LayerWeights {
            attn_norm_w: Some(ones_weight(hidden_size)),
            attn_norm_b: None,
            ffn_norm_w: Some(ones_weight(hidden_size)),
            ffn_norm_b: None,
            attn_output_norm_w: None,
            attn_output_norm_b: None,
            ffn_output_norm_w: None,
            ffn_output_norm_b: None,
            attn_q: identity_weight(hidden_size),
            attn_k: zero_weight(kv_dim, hidden_size),
            attn_v: zero_weight(kv_dim, hidden_size),
            attn_output: identity_weight(hidden_size),
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            attn_output_bias: None,
            ffn_up: zero_weight(ffn_hidden, hidden_size),
            ffn_down: zero_weight(hidden_size, ffn_hidden),
            ffn_gate: Some(zero_weight(ffn_hidden, hidden_size)),
            ffn_up_bias: None,
            ffn_down_bias: None,
            attn_q_norm_w: None,
            attn_k_norm_w: None,
            attn_post_norm_w: None,
            ffn_post_norm_w: None,
        };

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let weights = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![layer],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let mut cache = KvCache::new(&config);

        // Prefill with 2 tokens
        let out1 = model_forward_step(&[1, 3], &weights, &config, &b, &mut cache).unwrap();
        assert_eq!(out1.shape(), &[2, hidden_size]);
        assert_eq!(cache.len(), 2);

        // Decode 1 token
        let out2 = model_forward_step(&[5], &weights, &config, &b, &mut cache).unwrap();
        assert_eq!(out2.shape(), &[1, hidden_size]);
        assert_eq!(cache.len(), 3);

        for &v in out2.as_tensor().as_f32() {
            assert!(v.is_finite(), "GQA cached forward pass produced non-finite value: {}", v);
        }
    }

    // ====================================================================
    // rms_norm_per_head tests
    // ====================================================================

    #[test]
    fn test_rms_norm_per_head_single_head() {
        // With 1 head, per-head norm should equal regular rms_norm
        let b = cpu();
        let head_dim = 4;
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // seq_len=2, dim=4
        let input = dt(Tensor::new(vec![2, head_dim], input_data));
        let weight = ones_weight(head_dim);

        let result = rms_norm_per_head(input.clone(), &weight, 1, head_dim, 1e-6, &b);
        let expected = b.rms_norm(&input, &weight, 1e-6);

        assert_eq!(result.shape(), expected.shape());
        let r = result.as_tensor().as_f32();
        let e = expected.as_tensor().as_f32();
        for i in 0..r.len() {
            assert!((r[i] - e[i]).abs() < 1e-5, "mismatch at {}: {} vs {}", i, r[i], e[i]);
        }
    }

    #[test]
    fn test_rms_norm_per_head_multi_head_differs_from_full() {
        // With 2 heads of dim 2, per-head norm normalizes each head's [2]
        // independently. This should differ from normalizing the full [4] dim.
        let b = cpu();
        let num_heads = 2;
        let head_dim = 2;
        let total_dim = num_heads * head_dim; // 4
        // seq_len=1: [3.0, 4.0, 0.1, 0.1]
        // Head 0: [3.0, 4.0], Head 1: [0.1, 0.1]
        // RMS of head 0 = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        // RMS of head 1 = sqrt((0.01+0.01)/2) = sqrt(0.01) = 0.1
        // Per-head norm produces very different scales for the two heads,
        // whereas full-dim norm would use a single RMS over all 4 values.
        let input = dt(Tensor::new(vec![1, total_dim], vec![3.0, 4.0, 0.1, 0.1]));
        let weight = ones_weight(head_dim);

        let per_head = rms_norm_per_head(input.clone(), &weight, num_heads, head_dim, 1e-6, &b);

        // Full rms_norm over [4] dims (would use RMS of [3.0, 4.0, 0.1, 0.1])
        let full_weight = ones_weight(total_dim);
        let full_norm = b.rms_norm(&input, &full_weight, 1e-6);

        let ph = per_head.as_tensor().as_f32();
        let fn_ = full_norm.as_tensor().as_f32();

        // They should be different because per-head normalizes independently
        let mut any_different = false;
        for i in 0..ph.len() {
            if (ph[i] - fn_[i]).abs() > 1e-4 {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "per-head norm should differ from full-dim norm when heads have different magnitudes");

        // Per-head: head 0 values should be roughly [3/3.536, 4/3.536] ≈ [0.849, 1.131]
        // Per-head: head 1 values should be roughly [0.1/0.1, 0.1/0.1] = [1.0, 1.0]
        assert!((ph[0] - 3.0 / (12.5f32).sqrt()).abs() < 1e-3,
            "head 0 elem 0: expected ~{}, got {}", 3.0 / (12.5f32).sqrt(), ph[0]);
        assert!((ph[2] - 1.0).abs() < 1e-3,
            "head 1 elem 0: expected ~1.0, got {}", ph[2]);
    }

    #[test]
    fn test_rms_norm_per_head_preserves_shape() {
        let b = cpu();
        let num_heads = 3;
        let head_dim = 4;
        let seq_len = 5;
        let total_dim = num_heads * head_dim; // 12

        let input_data: Vec<f32> = (0..seq_len * total_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let input = dt(Tensor::new(vec![seq_len, total_dim], input_data));
        let weight = ones_weight(head_dim);

        let result = rms_norm_per_head(input, &weight, num_heads, head_dim, 1e-6, &b);
        assert_eq!(result.shape(), &[seq_len, total_dim]);

        // All values should be finite
        for &v in result.as_tensor().as_f32() {
            assert!(v.is_finite(), "rms_norm_per_head produced non-finite value: {}", v);
        }
    }

    // ====================================================================
    // gemma_embedding_layer_forward tests
    // ====================================================================

    /// Create a GemmaEmbedding-style config for testing.
    fn gemma_embedding_config(
        hidden_size: usize, num_heads: usize, num_kv_heads: usize,
        head_dim: usize, ffn_hidden: usize,
    ) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::GemmaEmbedding,
            arch_name: "gemma-embedding".to_string(),
            hidden_size,
            num_layers: 1,
            num_heads,
            num_kv_heads,
            head_dim,
            ffn_hidden,
            vocab_size: 16,
            max_seq_len: 128,
            norm_type: NormType::RMSNorm,
            norm_eps: 1e-6,
            activation: Activation::GeGLU,
            position_type: PositionType::RoPE,
            rope_freq_base: 10000.0,
            rope_dim: head_dim,
            rope_neox: false,
            rope_scaling_original_ctx: 0,
            rope_scaling_attn_factor: 1.0,
            causal: false,
            attn_logit_softcap: 0.0,
            attn_scale: None,
            embedding_scale: (hidden_size as f32).sqrt(),
            pooling_type: PoolingType::Mean,
            has_ffn_gate: true,
            has_bias: false,
            pre_norm: true,
            swa_window: 0,
            swa_layers: vec![],
            rope_freq_base_swa: 10000.0,
        }
    }

    /// Create minimal LayerWeights for GemmaEmbedding with per-head norms and post-projection norms.
    fn gemma_embedding_layer_weights(
        hidden_size: usize, num_kv_heads: usize, head_dim: usize, ffn_hidden: usize,
    ) -> LayerWeights {
        let kv_dim = num_kv_heads * head_dim;
        LayerWeights {
            attn_norm_w: Some(ones_weight(hidden_size)),
            attn_norm_b: None,
            ffn_norm_w: Some(ones_weight(hidden_size)),
            ffn_norm_b: None,
            attn_output_norm_w: None,
            attn_output_norm_b: None,
            ffn_output_norm_w: None,
            ffn_output_norm_b: None,
            attn_q: identity_weight(hidden_size),
            attn_k: zero_weight(kv_dim, hidden_size),
            attn_v: zero_weight(kv_dim, hidden_size),
            attn_output: identity_weight(hidden_size),
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            attn_output_bias: None,
            ffn_up: zero_weight(ffn_hidden, hidden_size),
            ffn_down: zero_weight(hidden_size, ffn_hidden),
            ffn_gate: Some(zero_weight(ffn_hidden, hidden_size)),
            ffn_up_bias: None,
            ffn_down_bias: None,
            attn_q_norm_w: Some(ones_weight(head_dim)),
            attn_k_norm_w: Some(ones_weight(head_dim)),
            attn_post_norm_w: Some(ones_weight(hidden_size)),
            ffn_post_norm_w: Some(ones_weight(hidden_size)),
        }
    }

    #[test]
    fn test_gemma_embedding_layer_output_shape() {
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let ffn_hidden = 16;

        let config = gemma_embedding_config(hidden_size, num_heads, num_kv_heads, head_dim, ffn_hidden);
        let layer = gemma_embedding_layer_weights(hidden_size, num_kv_heads, head_dim, ffn_hidden);

        let seq_len = 3;
        let input_data: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32) * 0.1).collect();
        let input = dt(Tensor::new(vec![seq_len, hidden_size], input_data));
        let mask = vec![1.0f32; seq_len];

        let output = transformer_layer_forward(&input, &layer, &config, &b, 0, &mask, None);
        assert_eq!(output.shape(), &[seq_len, hidden_size]);

        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "GemmaEmbedding layer produced non-finite value: {}", v);
        }
    }

    #[test]
    fn test_gemma_embedding_layer_with_gqa() {
        // Test the real GemmaEmbedding architecture: 3 heads, 1 KV head, head_dim=4
        // (scaled down from the actual 256 for test speed)
        let b = cpu();
        let hidden_size = 12; // 3 heads * 4 head_dim
        let num_heads = 3;
        let num_kv_heads = 1;
        let head_dim = 4;
        let ffn_hidden = 8;

        let config = gemma_embedding_config(hidden_size, num_heads, num_kv_heads, head_dim, ffn_hidden);
        let layer = gemma_embedding_layer_weights(hidden_size, num_kv_heads, head_dim, ffn_hidden);

        let seq_len = 2;
        let input_data: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let input = dt(Tensor::new(vec![seq_len, hidden_size], input_data));
        let mask = vec![1.0f32; seq_len];

        let output = transformer_layer_forward(&input, &layer, &config, &b, 0, &mask, None);
        assert_eq!(output.shape(), &[seq_len, hidden_size]);

        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "GemmaEmbedding GQA layer produced non-finite value: {}", v);
        }
    }

    #[test]
    fn test_gemma_embedding_post_norms_affect_output() {
        // Verify that post-attention and post-FFN norms actually change the output
        // by comparing with/without them.
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;

        let config = gemma_embedding_config(hidden_size, num_heads, num_kv_heads, head_dim, ffn_hidden);

        // Layer WITH post-norms (use non-trivial weights to see a difference)
        let mut layer_with = gemma_embedding_layer_weights(hidden_size, num_kv_heads, head_dim, ffn_hidden);
        // Use non-identity attn weights so attention actually does something
        let attn_data: Vec<f32> = (0..hidden_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
        layer_with.attn_q = dt(Tensor::new(vec![hidden_size, hidden_size], attn_data.clone()));
        layer_with.attn_k = dt(Tensor::new(vec![hidden_size, hidden_size], attn_data.clone()));
        layer_with.attn_v = dt(Tensor::new(vec![hidden_size, hidden_size], attn_data.clone()));

        // Layer WITHOUT post-norms (but otherwise identical)
        let mut layer_without = gemma_embedding_layer_weights(hidden_size, num_kv_heads, head_dim, ffn_hidden);
        layer_without.attn_q = dt(Tensor::new(vec![hidden_size, hidden_size], attn_data.clone()));
        layer_without.attn_k = dt(Tensor::new(vec![hidden_size, hidden_size], attn_data.clone()));
        layer_without.attn_v = dt(Tensor::new(vec![hidden_size, hidden_size], attn_data));
        layer_without.attn_post_norm_w = None;
        layer_without.ffn_post_norm_w = None;

        let seq_len = 2;
        let input_data: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let input = dt(Tensor::new(vec![seq_len, hidden_size], input_data));
        let mask = vec![1.0f32; seq_len];

        let output_with = transformer_layer_forward(&input, &layer_with, &config, &b, 0, &mask, None);
        let output_without = transformer_layer_forward(&input, &layer_without, &config, &b, 0, &mask, None);

        let data_w = output_with.as_tensor().as_f32();
        let data_wo = output_without.as_tensor().as_f32();

        let mut any_different = false;
        for i in 0..data_w.len() {
            if (data_w[i] - data_wo[i]).abs() > 1e-6 {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "Post-norms should change the output when attention weights are non-zero");
    }

    #[test]
    fn test_gemma_embedding_per_head_norms_affect_output() {
        // Verify that per-head Q/K norms actually change the output
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;

        let config = gemma_embedding_config(hidden_size, num_heads, num_kv_heads, head_dim, ffn_hidden);

        // Use non-zero K and V weights so attention actually does something
        let attn_data: Vec<f32> = (0..hidden_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let kv_dim = num_kv_heads * head_dim;
        let kv_data: Vec<f32> = (0..kv_dim * hidden_size).map(|i| (i as f32) * 0.01).collect();

        // Layer WITH per-head Q/K norms (using weight 2.0 so it's different from 1.0)
        let mut layer_with = gemma_embedding_layer_weights(hidden_size, num_kv_heads, head_dim, ffn_hidden);
        layer_with.attn_q = dt(Tensor::new(vec![hidden_size, hidden_size], attn_data.clone()));
        layer_with.attn_k = dt(Tensor::new(vec![kv_dim, hidden_size], kv_data.clone()));
        layer_with.attn_v = dt(Tensor::new(vec![kv_dim, hidden_size], kv_data.clone()));
        layer_with.attn_q_norm_w = Some(dt(Tensor::new(vec![head_dim], vec![2.0f32; head_dim])));
        layer_with.attn_k_norm_w = Some(dt(Tensor::new(vec![head_dim], vec![2.0f32; head_dim])));

        // Layer WITHOUT per-head norms (but same attention weights)
        let mut layer_without = gemma_embedding_layer_weights(hidden_size, num_kv_heads, head_dim, ffn_hidden);
        layer_without.attn_q = dt(Tensor::new(vec![hidden_size, hidden_size], attn_data));
        layer_without.attn_k = dt(Tensor::new(vec![kv_dim, hidden_size], kv_data.clone()));
        layer_without.attn_v = dt(Tensor::new(vec![kv_dim, hidden_size], kv_data));
        layer_without.attn_q_norm_w = None;
        layer_without.attn_k_norm_w = None;

        let seq_len = 2;
        let input_data: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let input = dt(Tensor::new(vec![seq_len, hidden_size], input_data));
        let mask = vec![1.0f32; seq_len];

        let output_with = transformer_layer_forward(&input, &layer_with, &config, &b, 0, &mask, None);
        let output_without = transformer_layer_forward(&input, &layer_without, &config, &b, 0, &mask, None);

        let data_w = output_with.as_tensor().as_f32();
        let data_wo = output_without.as_tensor().as_f32();

        let mut any_different = false;
        for i in 0..data_w.len() {
            if (data_w[i] - data_wo[i]).abs() > 1e-6 {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "Per-head Q/K norms with weight=2.0 should produce different output than no norms");
    }

    #[test]
    fn test_model_forward_bert_no_output_norm() {
        // BERT model without output_norm_w should still produce valid output
        // (BERT does per-layer post-norm, so no global final norm is needed)
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let vocab_size = 16;
        let max_seq_len = 128;
        let config = bert_config(hidden_size, num_heads, head_dim, ffn_hidden);

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let pos_emb_data: Vec<f32> = (0..max_seq_len * hidden_size).map(|i| (i as f32) * 0.001).collect();

        let weights = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: Some(dt(Tensor::new(vec![max_seq_len, hidden_size], pos_emb_data))),
            token_type_embedding: None,
            embedding_norm_w: Some(ones_weight(hidden_size)),
            embedding_norm_b: Some(zeros_bias(hidden_size)),
            layers: vec![bert_layer_weights(hidden_size, ffn_hidden)],
            output_norm_w: None,
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let input_ids = &[0u32, 2, 4, 6];
        let attention_mask = &[1.0f32, 1.0, 1.0, 1.0];

        let output = model_forward(input_ids, attention_mask, &weights, &config, &b, 0).unwrap();
        assert_eq!(output.shape(), &[4, hidden_size]);
        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "BERT model_forward without output_norm produced non-finite value");
        }
    }

    #[test]
    fn test_model_forward_gemma_embedding_full() {
        // End-to-end model_forward with GemmaEmbedding architecture
        let b = cpu();
        let hidden_size = 12; // 3 heads * 4 head_dim
        let num_heads = 3;
        let num_kv_heads = 1;
        let head_dim = 4;
        let ffn_hidden = 8;
        let vocab_size = 16;

        let mut config = gemma_embedding_config(hidden_size, num_heads, num_kv_heads, head_dim, ffn_hidden);
        config.vocab_size = vocab_size;

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let layer = gemma_embedding_layer_weights(hidden_size, num_kv_heads, head_dim, ffn_hidden);

        let weights = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![layer],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let input_ids = &[1u32, 3, 5];
        let attention_mask = &[1.0f32, 1.0, 1.0];

        let output = model_forward(input_ids, attention_mask, &weights, &config, &b, 0).unwrap();
        assert_eq!(output.shape(), &[3, hidden_size]);
        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "GemmaEmbedding model_forward produced non-finite value: {}", v);
        }

        // Verify embedding scale is applied (output should differ from unscaled)
        let mut config_no_scale = config.clone();
        config_no_scale.embedding_scale = 1.0;
        let embedding_data2: Vec<f32> = (0..vocab_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let layer2 = gemma_embedding_layer_weights(hidden_size, num_kv_heads, head_dim, ffn_hidden);
        let weights2 = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data2)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![layer2],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };
        let output_no_scale = model_forward(input_ids, attention_mask, &weights2, &config_no_scale, &b, 0).unwrap();
        let d1 = output.as_tensor().as_f32();
        let d2 = output_no_scale.as_tensor().as_f32();
        let mut any_diff = false;
        for i in 0..d1.len() {
            if (d1[i] - d2[i]).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "Embedding scale should produce different outputs");
    }

    #[test]
    fn test_gemma_embedding_is_bidirectional() {
        // GemmaEmbedding should be bidirectional (causal=false).
        // Verify that changing token at position 2 affects output at position 0.
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let vocab_size = 16;

        let mut config = gemma_embedding_config(hidden_size, num_heads, num_kv_heads, head_dim, ffn_hidden);
        config.vocab_size = vocab_size;
        assert!(!config.causal, "GemmaEmbedding should be bidirectional");

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size).map(|i| (i as f32) * 0.05).collect();

        // Use non-zero K and V weights so attention actually attends across positions
        let mut layer = gemma_embedding_layer_weights(hidden_size, num_kv_heads, head_dim, ffn_hidden);
        let attn_data: Vec<f32> = (0..hidden_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let kv_dim = num_kv_heads * head_dim;
        let kv_data: Vec<f32> = (0..kv_dim * hidden_size).map(|i| (i as f32) * 0.01).collect();
        layer.attn_q = dt(Tensor::new(vec![hidden_size, hidden_size], attn_data));
        layer.attn_k = dt(Tensor::new(vec![kv_dim, hidden_size], kv_data.clone()));
        layer.attn_v = dt(Tensor::new(vec![kv_dim, hidden_size], kv_data));

        let weights = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![layer],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let mask = &[1.0f32, 1.0, 1.0];

        let out1 = model_forward(&[1, 2, 3], mask, &weights, &config, &b, 0).unwrap();
        let out2 = model_forward(&[1, 2, 5], mask, &weights, &config, &b, 0).unwrap();

        // Position 0 output should differ because token at position 2 changed
        // (bidirectional attention allows position 0 to see position 2)
        let d1 = out1.as_tensor().as_f32();
        let d2 = out2.as_tensor().as_f32();
        let pos0_differs = (0..hidden_size).any(|j| (d1[j] - d2[j]).abs() > 1e-6);
        assert!(pos0_differs, "Bidirectional: changing token at pos 2 should affect output at pos 0");
    }

    // ====================================================================
    // Qwen3 (pre-norm with per-head Q/K norms) tests
    // ====================================================================

    /// Create a Qwen3-style config for testing.
    fn qwen3_config(
        hidden_size: usize, num_heads: usize, num_kv_heads: usize,
        head_dim: usize, ffn_hidden: usize,
    ) -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Qwen3,
            arch_name: "qwen3".to_string(),
            hidden_size,
            num_layers: 1,
            num_heads,
            num_kv_heads,
            head_dim,
            ffn_hidden,
            vocab_size: 16,
            max_seq_len: 128,
            norm_type: NormType::RMSNorm,
            norm_eps: 1e-6,
            activation: Activation::SwiGLU,
            position_type: PositionType::RoPE,
            rope_freq_base: 10000.0,
            rope_dim: head_dim,
            rope_neox: true,
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
            swa_window: 0,
            swa_layers: vec![],
            rope_freq_base_swa: 10000.0,
        }
    }

    /// Create minimal LayerWeights for Qwen3 with per-head Q/K norms.
    fn qwen3_layer_weights(
        hidden_size: usize, num_kv_heads: usize, head_dim: usize, ffn_hidden: usize,
    ) -> LayerWeights {
        let kv_dim = num_kv_heads * head_dim;
        LayerWeights {
            attn_norm_w: Some(ones_weight(hidden_size)),
            attn_norm_b: None,
            ffn_norm_w: Some(ones_weight(hidden_size)),
            ffn_norm_b: None,
            attn_output_norm_w: None,
            attn_output_norm_b: None,
            ffn_output_norm_w: None,
            ffn_output_norm_b: None,
            attn_q: identity_weight(hidden_size),
            attn_k: zero_weight(kv_dim, hidden_size),
            attn_v: zero_weight(kv_dim, hidden_size),
            attn_output: identity_weight(hidden_size),
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            attn_output_bias: None,
            ffn_up: zero_weight(ffn_hidden, hidden_size),
            ffn_down: zero_weight(hidden_size, ffn_hidden),
            ffn_gate: Some(zero_weight(ffn_hidden, hidden_size)),
            ffn_up_bias: None,
            ffn_down_bias: None,
            attn_q_norm_w: Some(ones_weight(head_dim)),
            attn_k_norm_w: Some(ones_weight(head_dim)),
            attn_post_norm_w: None,
            ffn_post_norm_w: None,
        }
    }

    #[test]
    fn test_qwen3_layer_output_shape() {
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;

        let config = qwen3_config(hidden_size, num_heads, num_kv_heads, head_dim, ffn_hidden);
        let layer = qwen3_layer_weights(hidden_size, num_kv_heads, head_dim, ffn_hidden);

        let seq_len = 3;
        let input_data: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32) * 0.1).collect();
        let input = dt(Tensor::new(vec![seq_len, hidden_size], input_data));
        let mask = vec![1.0f32; seq_len];

        let output = transformer_layer_forward(&input, &layer, &config, &b, 0, &mask, None);
        assert_eq!(output.shape(), &[seq_len, hidden_size]);

        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "Qwen3 layer produced non-finite value: {}", v);
        }
    }

    #[test]
    fn test_qwen3_per_head_norms_affect_output() {
        // Verify Q/K norms in pre_norm_layer_forward actually change the output
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;

        let config = qwen3_config(hidden_size, num_heads, num_kv_heads, head_dim, ffn_hidden);

        let kv_dim = num_kv_heads * head_dim;
        let kv_data: Vec<f32> = (0..kv_dim * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let attn_data: Vec<f32> = (0..hidden_size * hidden_size).map(|i| (i as f32) * 0.01).collect();

        // Layer WITH per-head Q/K norms (weight=2.0)
        let mut layer_with = qwen3_layer_weights(hidden_size, num_kv_heads, head_dim, ffn_hidden);
        layer_with.attn_q = dt(Tensor::new(vec![hidden_size, hidden_size], attn_data.clone()));
        layer_with.attn_k = dt(Tensor::new(vec![kv_dim, hidden_size], kv_data.clone()));
        layer_with.attn_v = dt(Tensor::new(vec![kv_dim, hidden_size], kv_data.clone()));
        layer_with.attn_q_norm_w = Some(dt(Tensor::new(vec![head_dim], vec![2.0f32; head_dim])));
        layer_with.attn_k_norm_w = Some(dt(Tensor::new(vec![head_dim], vec![2.0f32; head_dim])));

        // Layer WITHOUT per-head norms (same attention weights)
        let mut layer_without = qwen3_layer_weights(hidden_size, num_kv_heads, head_dim, ffn_hidden);
        layer_without.attn_q = dt(Tensor::new(vec![hidden_size, hidden_size], attn_data));
        layer_without.attn_k = dt(Tensor::new(vec![kv_dim, hidden_size], kv_data.clone()));
        layer_without.attn_v = dt(Tensor::new(vec![kv_dim, hidden_size], kv_data));
        layer_without.attn_q_norm_w = None;
        layer_without.attn_k_norm_w = None;

        let seq_len = 2;
        let input_data: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let input = dt(Tensor::new(vec![seq_len, hidden_size], input_data));
        let mask = vec![1.0f32; seq_len];

        let output_with = transformer_layer_forward(&input, &layer_with, &config, &b, 0, &mask, None);
        let output_without = transformer_layer_forward(&input, &layer_without, &config, &b, 0, &mask, None);

        let data_w = output_with.as_tensor().as_f32();
        let data_wo = output_without.as_tensor().as_f32();

        let any_different = (0..data_w.len()).any(|i| (data_w[i] - data_wo[i]).abs() > 1e-6);
        assert!(any_different,
            "Per-head Q/K norms (weight=2.0) in pre_norm path should produce different output than no norms");
    }

    #[test]
    fn test_qwen3_gqa_with_per_head_norms() {
        // Qwen3-1.7B has 16 heads, 4 KV heads. Test the Q/K norm path with GQA.
        let b = cpu();
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let hidden_size = num_heads * head_dim; // 16
        let ffn_hidden = 32;

        let config = qwen3_config(hidden_size, num_heads, num_kv_heads, head_dim, ffn_hidden);
        let layer = qwen3_layer_weights(hidden_size, num_kv_heads, head_dim, ffn_hidden);

        let seq_len = 3;
        let input_data: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let input = dt(Tensor::new(vec![seq_len, hidden_size], input_data));
        let mask = vec![1.0f32; seq_len];

        let output = transformer_layer_forward(&input, &layer, &config, &b, 0, &mask, None);
        assert_eq!(output.shape(), &[seq_len, hidden_size]);

        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "Qwen3 GQA with per-head norms produced non-finite value: {}", v);
        }
    }

    #[test]
    fn test_qwen3_cached_forward_with_per_head_norms() {
        // Test Q/K norms in the cached (generation) path
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let vocab_size = 16;

        let mut config = qwen3_config(hidden_size, num_heads, num_kv_heads, head_dim, ffn_hidden);
        config.num_layers = 1;

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let weights = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![qwen3_layer_weights(hidden_size, num_kv_heads, head_dim, ffn_hidden)],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let mut cache = KvCache::new(&config);

        // Prefill
        let output = model_forward_step(&[1, 3, 5], &weights, &config, &b, &mut cache).unwrap();
        assert_eq!(output.shape(), &[3, hidden_size]);
        assert_eq!(cache.len(), 3);

        // Decode single token
        let output = model_forward_step(&[7], &weights, &config, &b, &mut cache).unwrap();
        assert_eq!(output.shape(), &[1, hidden_size]);
        assert_eq!(cache.len(), 4);

        for &v in output.as_tensor().as_f32() {
            assert!(v.is_finite(), "Qwen3 cached decode with per-head norms produced non-finite value");
        }
    }

    #[test]
    fn test_qwen3_cached_vs_uncached_prefill_match() {
        // Verify cached and uncached paths produce the same result for Qwen3
        // (both paths now have Q/K norm code — this catches mismatches)
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let vocab_size = 16;

        let mut config = qwen3_config(hidden_size, num_heads, num_kv_heads, head_dim, ffn_hidden);
        config.num_layers = 1;

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let weights = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![qwen3_layer_weights(hidden_size, num_kv_heads, head_dim, ffn_hidden)],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let input_ids = &[1u32, 3, 5];
        let attention_mask = &[1.0f32, 1.0, 1.0];

        // Uncached
        let uncached = model_forward(input_ids, attention_mask, &weights, &config, &b, 0).unwrap();

        // Cached (prefill)
        let mut cache = KvCache::new(&config);
        let cached = model_forward_step(input_ids, &weights, &config, &b, &mut cache).unwrap();

        let uncached_data = uncached.as_tensor().as_f32();
        let cached_data = cached.as_tensor().as_f32();
        assert_eq!(uncached_data.len(), cached_data.len());
        for i in 0..uncached_data.len() {
            assert!(
                (uncached_data[i] - cached_data[i]).abs() < 1e-4,
                "Qwen3 cached vs uncached mismatch at {}: uncached={} cached={}",
                i, uncached_data[i], cached_data[i]
            );
        }
    }

    #[test]
    fn test_rope_neox_with_factors_correctness() {
        // Verify rope_neox_with_factors computes:
        //   freq[i] = freq_base^(-2i/rope_dim) / factors[i]
        //   theta = pos * freq[i]
        //   out_lo = (x_lo * cos(theta) - x_hi * sin(theta)) * mscale
        //   out_hi = (x_lo * sin(theta) + x_hi * cos(theta)) * mscale
        let b = cpu();

        // 1 token, 1 head, head_dim=4, rope_dim=4 → half_rope=2
        let head_dim = 4;
        let rope_dim = 4;
        let freq_base = 10000.0f32;
        let pos_offset = 3;
        let mscale = 1.19f32;
        let factors = vec![2.0f32, 0.5]; // 2 factors for half_rope=2

        let q_data = vec![1.0f32, 2.0, 3.0, 4.0]; // [lo0, lo1, hi0, hi1]
        let k_data = vec![5.0f32, 6.0, 7.0, 8.0];
        let q = dt(Tensor::new(vec![1, 4], q_data.clone()));
        let k = dt(Tensor::new(vec![1, 4], k_data.clone()));

        let (q_rot, k_rot) = rope_neox_with_factors(
            &q, &k, pos_offset, freq_base, head_dim, rope_dim,
            &factors, mscale, &b,
        );

        // Manual computation
        let inv_ndims = -1.0f32 / rope_dim as f32;
        let pos = pos_offset as f32;

        // freq[0] = freq_base^(-0/4) / 2.0 = 1.0 / 2.0 = 0.5
        let freq0 = freq_base.powf(inv_ndims * 0.0) / factors[0];
        // freq[1] = freq_base^(-2/4) / 0.5 = freq_base^(-0.5) / 0.5
        let freq1 = freq_base.powf(inv_ndims * 2.0) / factors[1];

        let theta0 = pos * freq0;
        let theta1 = pos * freq1;

        let expected_q = [
            (q_data[0] * theta0.cos() - q_data[2] * theta0.sin()) * mscale,
            (q_data[1] * theta1.cos() - q_data[3] * theta1.sin()) * mscale,
            (q_data[0] * theta0.sin() + q_data[2] * theta0.cos()) * mscale,
            (q_data[1] * theta1.sin() + q_data[3] * theta1.cos()) * mscale,
        ];
        let expected_k = [
            (k_data[0] * theta0.cos() - k_data[2] * theta0.sin()) * mscale,
            (k_data[1] * theta1.cos() - k_data[3] * theta1.sin()) * mscale,
            (k_data[0] * theta0.sin() + k_data[2] * theta0.cos()) * mscale,
            (k_data[1] * theta1.sin() + k_data[3] * theta1.cos()) * mscale,
        ];

        let q_out = q_rot.as_tensor().as_f32();
        let k_out = k_rot.as_tensor().as_f32();

        for i in 0..4 {
            assert!(
                (q_out[i] - expected_q[i]).abs() < 1e-5,
                "Q mismatch at {}: got {} expected {}", i, q_out[i], expected_q[i]
            );
            assert!(
                (k_out[i] - expected_k[i]).abs() < 1e-5,
                "K mismatch at {}: got {} expected {}", i, k_out[i], expected_k[i]
            );
        }

        // Verify mscale != 1.0 actually changes the output magnitude
        let (q_no_scale, _) = rope_neox_with_factors(
            &q, &k, pos_offset, freq_base, head_dim, rope_dim,
            &factors, 1.0, &b,
        );
        let q_ns = q_no_scale.as_tensor().as_f32();
        for i in 0..4 {
            assert!(
                (q_out[i] - q_ns[i] * mscale).abs() < 1e-5,
                "mscale should linearly scale output at {}", i
            );
        }
    }

    #[test]
    fn test_rope_neox_with_factors_divides_not_multiplies() {
        // Regression test: factors are DIVISORS of the base frequency, not multipliers.
        // theta = pos * freq_base^(-2i/rope_dim) / factor
        // So factor=2 at pos=10 should give the same result as factor=1 at pos=5.
        let b = cpu();
        let head_dim = 4;
        let rope_dim = 4;
        let freq_base = 10000.0f32;

        let q = dt(Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]));
        let k = dt(Tensor::new(vec![1, 4], vec![5.0, 6.0, 7.0, 8.0]));

        // factor=2 at pos=10 should equal factor=1 at pos=5
        // because theta = 10 * freq / 2 = 5 * freq / 1
        let factors_1 = vec![1.0f32, 1.0];
        let factors_2 = vec![2.0f32, 2.0];

        let (q_f1_p5, _) = rope_neox_with_factors(
            &q, &k, 5, freq_base, head_dim, rope_dim, &factors_1, 1.0, &b,
        );
        let (q_f2_p10, _) = rope_neox_with_factors(
            &q, &k, 10, freq_base, head_dim, rope_dim, &factors_2, 1.0, &b,
        );

        let q1_data = q_f1_p5.as_tensor().as_f32();
        let q2_data = q_f2_p10.as_tensor().as_f32();

        for i in 0..4 {
            assert!(
                (q1_data[i] - q2_data[i]).abs() < 1e-5,
                "factor=2 at pos=10 should match factor=1 at pos=5 (idx {}): {} vs {}",
                i, q2_data[i], q1_data[i]
            );
        }

        // Sanity: factor=2 at pos=10 should NOT match factor=2 at pos=5
        let (q_f2_p5, _) = rope_neox_with_factors(
            &q, &k, 5, freq_base, head_dim, rope_dim, &factors_2, 1.0, &b,
        );
        let q3_data = q_f2_p5.as_tensor().as_f32();
        let mut different = false;
        for i in 0..4 {
            if (q2_data[i] - q3_data[i]).abs() > 1e-6 {
                different = true;
                break;
            }
        }
        assert!(different, "Different positions should produce different rotations");
    }

    #[test]
    fn test_cached_forward_with_longrope_factors() {
        // Test that model_forward_step correctly uses LongRoPE factors,
        // and that factored RoPE produces different outputs vs unfactored.
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let vocab_size = 16;

        // Config with LongRoPE enabled (NeoX-style, original_ctx=64, mscale=1.19)
        let mut config = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);
        config.rope_neox = true;
        config.rope_scaling_original_ctx = 64;
        config.rope_scaling_attn_factor = 1.19;
        // max_seq_len < original_ctx → should pick short factors
        config.max_seq_len = 32;

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size)
            .map(|i| (i as f32) * 0.01)
            .collect();

        // Short factors: halve the base frequency
        let short_factors = vec![2.0f32; head_dim / 2];
        // Long factors: double the base frequency
        let long_factors = vec![0.5f32; head_dim / 2];

        let weights_with_factors = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data.clone())),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![gemma_layer_weights(hidden_size, ffn_hidden)],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: Some(dt(Tensor::new(vec![head_dim / 2], short_factors.clone()))),
            rope_factors_long: Some(dt(Tensor::new(vec![head_dim / 2], long_factors.clone()))),
        };

        let weights_no_factors = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![gemma_layer_weights(hidden_size, ffn_hidden)],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let config_no_longrope = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);

        let input_ids = &[1u32, 3, 5];

        // Run with LongRoPE factors (max_seq_len=32 < original_ctx=64 → uses short factors)
        let mut cache1 = KvCache::new(&config);
        let out_factored = model_forward_step(input_ids, &weights_with_factors, &config, &b, &mut cache1).unwrap();

        // Run without LongRoPE factors
        let mut cache2 = KvCache::new(&config_no_longrope);
        let out_unfactored = model_forward_step(input_ids, &weights_no_factors, &config_no_longrope, &b, &mut cache2).unwrap();

        // Outputs should differ because factored RoPE changes rotation frequencies
        let data_f = out_factored.as_tensor().as_f32();
        let data_u = out_unfactored.as_tensor().as_f32();
        assert_eq!(data_f.len(), data_u.len());

        let mut any_different = false;
        for i in 0..data_f.len() {
            if (data_f[i] - data_u[i]).abs() > 1e-5 {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "LongRoPE factors should change outputs vs unfactored RoPE");

        // Now test long factor selection: max_seq_len > original_ctx → uses long factors
        let mut config_long = config.clone();
        config_long.max_seq_len = 128; // > original_ctx=64

        let mut cache3 = KvCache::new(&config_long);
        let out_long = model_forward_step(input_ids, &weights_with_factors, &config_long, &b, &mut cache3).unwrap();

        let data_l = out_long.as_tensor().as_f32();

        // Long vs short factors should produce different outputs
        let mut long_vs_short_differ = false;
        for i in 0..data_f.len() {
            if (data_f[i] - data_l[i]).abs() > 1e-5 {
                long_vs_short_differ = true;
                break;
            }
        }
        assert!(long_vs_short_differ, "Long vs short factors should produce different outputs");

        // All outputs should be finite
        for &v in data_f.iter().chain(data_l.iter()) {
            assert!(v.is_finite(), "Non-finite value in LongRoPE output");
        }
    }

    #[test]
    fn test_model_forward_noncached_with_longrope_factors() {
        // Regression test: model_forward (non-cached path, used by EmbeddingEngine)
        // must apply LongRoPE factors. Previously this was a bug where factors
        // were only used in model_forward_step (cached path).
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let vocab_size = 16;

        // Config with LongRoPE enabled
        let mut config = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);
        config.rope_neox = true;
        config.rope_scaling_original_ctx = 64;
        config.rope_scaling_attn_factor = 1.19;
        config.max_seq_len = 32; // < original_ctx → uses short factors

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size)
            .map(|i| (i as f32) * 0.01)
            .collect();

        let short_factors = vec![2.0f32; head_dim / 2];
        let long_factors = vec![0.5f32; head_dim / 2];

        let weights_with_factors = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data.clone())),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![gemma_layer_weights(hidden_size, ffn_hidden)],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: Some(dt(Tensor::new(vec![head_dim / 2], short_factors))),
            rope_factors_long: Some(dt(Tensor::new(vec![head_dim / 2], long_factors))),
        };

        let weights_no_factors = ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data)),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![gemma_layer_weights(hidden_size, ffn_hidden)],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: None,
            rope_factors_long: None,
        };

        let config_no_longrope = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);

        let input_ids = &[1u32, 3, 5];
        let mask = &[1.0f32, 1.0, 1.0];

        // Run model_forward WITH LongRoPE factors
        let out_factored = model_forward(input_ids, mask, &weights_with_factors, &config, &b, 0).unwrap();

        // Run model_forward WITHOUT LongRoPE factors
        let out_unfactored = model_forward(input_ids, mask, &weights_no_factors, &config_no_longrope, &b, 0).unwrap();

        let data_f = out_factored.as_tensor().as_f32();
        let data_u = out_unfactored.as_tensor().as_f32();
        assert_eq!(data_f.len(), data_u.len());
        assert_eq!(out_factored.shape(), &[3, hidden_size]);

        // Outputs MUST differ — this was the bug: model_forward ignored LongRoPE factors
        let mut any_different = false;
        for i in 0..data_f.len() {
            if (data_f[i] - data_u[i]).abs() > 1e-5 {
                any_different = true;
                break;
            }
        }
        assert!(any_different,
            "model_forward with LongRoPE factors must produce different output than without");

        // All outputs should be finite
        for &v in data_f.iter() {
            assert!(v.is_finite(), "Non-finite value in model_forward LongRoPE output");
        }
    }

    #[test]
    fn test_model_forward_noncached_longrope_factor_selection_boundary() {
        // Test the boundary condition: max_seq_len == rope_scaling_original_ctx
        // should select SHORT factors (> comparison, matching llama.cpp).
        // Only max_seq_len > original_ctx selects LONG factors.
        let b = cpu();
        let hidden_size = 8;
        let num_heads = 2;
        let head_dim = 4;
        let ffn_hidden = 16;
        let vocab_size = 16;

        let embedding_data: Vec<f32> = (0..vocab_size * hidden_size)
            .map(|i| (i as f32) * 0.01)
            .collect();

        // Short and long factors are deliberately different
        let short_factors = vec![2.0f32; head_dim / 2];
        let long_factors = vec![0.5f32; head_dim / 2];

        let make_weights = |short: Vec<f32>, long: Vec<f32>| ModelWeights {
            token_embedding: dt(Tensor::new(vec![vocab_size, hidden_size], embedding_data.clone())),
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_w: None,
            embedding_norm_b: None,
            layers: vec![gemma_layer_weights(hidden_size, ffn_hidden)],
            output_norm_w: Some(ones_weight(hidden_size)),
            output_norm_b: None,
            output_projection: None,
            rope_factors_short: Some(dt(Tensor::new(vec![head_dim / 2], short))),
            rope_factors_long: Some(dt(Tensor::new(vec![head_dim / 2], long))),
        };

        let mut config_eq = gemma_config(hidden_size, num_heads, head_dim, ffn_hidden);
        config_eq.rope_neox = true;
        config_eq.rope_scaling_original_ctx = 64;
        config_eq.rope_scaling_attn_factor = 1.19;
        config_eq.max_seq_len = 64; // == original_ctx → should pick SHORT (matching llama.cpp)

        let mut config_below = config_eq.clone();
        config_below.max_seq_len = 32; // clearly < original_ctx → definitely SHORT

        let mut config_above = config_eq.clone();
        config_above.max_seq_len = 65; // > original_ctx → should pick LONG

        let input_ids = &[1u32, 3, 5];
        let mask = &[1.0f32, 1.0, 1.0];

        let weights = make_weights(short_factors, long_factors);

        let out_eq = model_forward(input_ids, mask, &weights, &config_eq, &b, 0).unwrap();
        let out_below = model_forward(input_ids, mask, &weights, &config_below, &b, 0).unwrap();
        let out_above = model_forward(input_ids, mask, &weights, &config_above, &b, 0).unwrap();

        let data_eq = out_eq.as_tensor().as_f32();
        let data_below = out_below.as_tensor().as_f32();
        let data_above = out_above.as_tensor().as_f32();

        // 1) max_seq_len == original_ctx must match clearly-short (both pick short factors)
        for i in 0..data_eq.len() {
            assert!((data_eq[i] - data_below[i]).abs() < 1e-5,
                "Boundary: == original_ctx should pick SHORT, but differs from clearly-short at [{}]: {} vs {}",
                i, data_eq[i], data_below[i]);
        }

        // 2) max_seq_len > original_ctx must differ from short (picks long factors)
        let mut any_different = false;
        for i in 0..data_eq.len() {
            if (data_eq[i] - data_above[i]).abs() > 1e-5 {
                any_different = true;
                break;
            }
        }
        assert!(any_different,
            "Boundary: max_seq_len > original_ctx (long) should differ from == (short)");
    }
}
