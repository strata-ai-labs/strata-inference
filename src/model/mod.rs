// M4: Model loading and transformer runtime (future milestone)

#[cfg(test)]
mod tests {
    // ====================================================================
    // Forward-looking tests for M4: Model config, weight loading, and
    // transformer layer forward pass.
    //
    // These tests define the expected API surface and behavior for
    // upcoming milestone M4. They are marked #[ignore] so they compile
    // but do not run until the implementation exists.
    // ====================================================================

    // ------------------------------------------------------------------
    // M4.1: ModelConfig from GGUF metadata
    // ------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_model_config_from_gemma3_gguf() {
        // Loading a Gemma3 GGUF file should extract the correct config:
        //   architecture = "gemma3"
        //   embedding_length (hidden_size) = 768
        //   block_count (num_layers) = 24
        //   head_count = 16
        //   head_count_kv = (could be < head_count for GQA)
        //   feed_forward_length
        //   layer_norm_rms_epsilon
        //   rope.freq_base
        //   attention.causal
        //   pooling_type
        //
        // This test will need a small test fixture or a real GGUF file.
        // For now, it documents the expected API:
        //
        //   let gguf = GgufFile::open("path/to/gemma3.gguf").unwrap();
        //   let config = ModelConfig::from_gguf(&gguf).unwrap();
        //   assert_eq!(config.architecture, "gemma3");
        //   assert_eq!(config.hidden_size, 768);
        //   assert_eq!(config.num_layers, 24);
        //   assert_eq!(config.num_heads, 16);
        //   assert!(config.rms_norm_eps > 0.0);
        //   assert!(config.rope_freq_base > 0.0);
        todo!("Implement ModelConfig::from_gguf");
    }

    #[test]
    #[ignore]
    fn test_model_config_detects_architecture() {
        // ModelConfig should read `general.architecture` and dispatch
        // accordingly. Supported architectures: gemma3, llama, bert.
        // Unsupported architecture should return InferenceError::UnsupportedArchitecture.
        //
        //   let gguf = make_gguf_with_arch("unsupported_arch");
        //   let result = ModelConfig::from_gguf(&gguf);
        //   assert!(matches!(result, Err(InferenceError::UnsupportedArchitecture(_))));
        todo!("Implement architecture detection in ModelConfig");
    }

    #[test]
    #[ignore]
    fn test_model_config_missing_required_key() {
        // If a required key (e.g., embedding_length) is missing from the
        // GGUF metadata, ModelConfig::from_gguf should return
        // InferenceError::MissingKey.
        //
        //   let gguf = make_gguf_missing_key("gemma3.embedding_length");
        //   let result = ModelConfig::from_gguf(&gguf);
        //   assert!(matches!(result, Err(InferenceError::MissingKey(_))));
        todo!("Implement required-key validation in ModelConfig");
    }

    #[test]
    #[ignore]
    fn test_model_config_bert_vs_gemma_differences() {
        // BERT uses LayerNorm (weight + bias), no gate tensor, GELU, learned positions.
        // Gemma uses RMSNorm (weight only), has gate tensor, SwiGLU, RoPE.
        // ModelConfig should capture these differences:
        //
        //   let bert_config = ModelConfig::from_gguf(&bert_gguf).unwrap();
        //   assert!(!bert_config.use_rms_norm);     // BERT uses LayerNorm
        //   assert!(!bert_config.has_ffn_gate);      // No SwiGLU gate
        //   assert!(!bert_config.use_rope);          // Learned positions
        //
        //   let gemma_config = ModelConfig::from_gguf(&gemma_gguf).unwrap();
        //   assert!(gemma_config.use_rms_norm);
        //   assert!(gemma_config.has_ffn_gate);
        //   assert!(gemma_config.use_rope);
        todo!("Implement architecture-specific config fields");
    }

    #[test]
    #[ignore]
    fn test_model_config_pooling_type() {
        // pooling_type metadata: 0=NONE, 1=MEAN, 2=CLS, 3=LAST
        // Should be parsed correctly.
        //
        //   let config = ModelConfig::from_gguf(&gguf).unwrap();
        //   assert_eq!(config.pooling_type, PoolingType::Mean);
        todo!("Implement pooling_type parsing");
    }

    #[test]
    #[ignore]
    fn test_model_config_gqa_head_counts() {
        // When head_count_kv < head_count, the model uses Grouped Query
        // Attention. Config should expose both values correctly.
        //
        //   let config = ModelConfig::from_gguf(&gqa_gguf).unwrap();
        //   assert!(config.num_kv_heads < config.num_heads);
        //   assert_eq!(config.num_heads % config.num_kv_heads, 0);
        todo!("Implement GQA config parsing");
    }

    // ------------------------------------------------------------------
    // M4.2: Weight loading from GGUF tensors
    // ------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_load_embedding_weights() {
        // Load `token_embd.weight` from the GGUF file.
        // Should have shape [vocab_size, hidden_size].
        //
        //   let gguf = GgufFile::open("model.gguf").unwrap();
        //   let config = ModelConfig::from_gguf(&gguf).unwrap();
        //   let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();
        //   assert_eq!(weights.token_embedding.shape(), &[config.vocab_size, config.hidden_size]);
        todo!("Implement ModelWeights::from_gguf with embedding weight loading");
    }

    #[test]
    #[ignore]
    fn test_load_layer_weights_all_present() {
        // For each layer i in 0..num_layers, the following tensors
        // should be loaded:
        //   blk.{i}.attn_norm.weight
        //   blk.{i}.attn_q.weight
        //   blk.{i}.attn_k.weight
        //   blk.{i}.attn_v.weight
        //   blk.{i}.attn_output.weight
        //   blk.{i}.ffn_norm.weight
        //   blk.{i}.ffn_up.weight
        //   blk.{i}.ffn_down.weight
        //   blk.{i}.ffn_gate.weight (Gemma/LLaMA only, absent for BERT)
        //
        //   let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();
        //   assert_eq!(weights.layers.len(), config.num_layers);
        //   for layer in &weights.layers {
        //       assert!(layer.attn_q.shape()[0] > 0);
        //       assert!(layer.attn_k.shape()[0] > 0);
        //   }
        todo!("Implement per-layer weight loading");
    }

    #[test]
    #[ignore]
    fn test_load_weights_missing_tensor_error() {
        // If a required tensor (e.g., blk.0.attn_q.weight) is not found
        // in the GGUF file, weight loading should return
        // InferenceError::TensorNotFound.
        //
        //   let result = ModelWeights::from_gguf(&incomplete_gguf, &config, &backend);
        //   assert!(matches!(result, Err(InferenceError::TensorNotFound(_))));
        todo!("Implement missing tensor error handling");
    }

    #[test]
    #[ignore]
    fn test_load_output_norm_weight() {
        // The final normalization weight `output_norm.weight` should be loaded.
        // Shape: [hidden_size].
        //
        //   let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();
        //   assert_eq!(weights.output_norm.shape(), &[config.hidden_size]);
        todo!("Implement output_norm weight loading");
    }

    #[test]
    #[ignore]
    fn test_weight_shapes_match_config() {
        // All loaded weight shapes should be consistent with ModelConfig:
        //   attn_q: [num_heads * head_dim, hidden_size]
        //   attn_k: [num_kv_heads * head_dim, hidden_size]
        //   ffn_up: [ffn_hidden_dim, hidden_size]
        //   ffn_down: [hidden_size, ffn_hidden_dim]
        //
        //   let weights = ModelWeights::from_gguf(&gguf, &config, &backend).unwrap();
        //   let layer = &weights.layers[0];
        //   assert_eq!(layer.ffn_down.shape()[0], config.hidden_size);
        //   assert_eq!(layer.ffn_down.shape()[1], config.ffn_hidden_dim);
        todo!("Implement weight shape validation");
    }

    // ------------------------------------------------------------------
    // M4.3: Transformer layer forward pass
    // ------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_single_transformer_layer_output_shape() {
        // A single transformer layer should preserve the input shape.
        // Input: [seq_len, hidden_size] -> Output: [seq_len, hidden_size]
        //
        //   let output = layer.forward(&input, &mask, pos_offset, &backend);
        //   assert_eq!(output.shape(), input.shape());
        todo!("Implement transformer layer forward pass");
    }

    #[test]
    #[ignore]
    fn test_transformer_layer_with_causal_mask() {
        // With causal attention, the output at position i should NOT
        // depend on tokens at positions j > i.
        // Verify by running the same prefix with different suffixes
        // and checking that prefix outputs are identical.
        //
        //   let out_short = model.forward(&input_3_tokens, &mask_3, 0, &backend);
        //   let out_long = model.forward(&input_5_tokens, &mask_5, 0, &backend);
        //   // First 3 positions should produce identical outputs (causal)
        //   assert_tensors_close(&out_short, &out_long.slice(0, 3), 1e-5);
        todo!("Implement causal mask verification test");
    }

    #[test]
    #[ignore]
    fn test_transformer_layer_bidirectional() {
        // For BERT-style models (non-causal), all positions attend to all
        // other positions. Changing a later token SHOULD affect earlier
        // positions' outputs.
        //
        //   let out_a = model.forward(&input_a, &mask, 0, &backend);
        //   let out_b = model.forward(&input_b_different_last, &mask, 0, &backend);
        //   // First position output should differ (bidirectional attention)
        //   assert_ne!(out_a[0], out_b[0]);
        todo!("Implement bidirectional attention test");
    }

    #[test]
    #[ignore]
    fn test_rope_position_encoding_applied() {
        // RoPE should make the same token at different positions produce
        // different Q/K representations. Run the same input at pos_offset=0
        // and pos_offset=100; outputs should differ.
        //
        //   let out_0 = layer.forward(&input, &mask, /*pos_offset=*/0, &backend);
        //   let out_100 = layer.forward(&input, &mask, /*pos_offset=*/100, &backend);
        //   // Same input, different positions -> different outputs
        //   assert_ne!(out_0, out_100);
        todo!("Implement RoPE position offset test");
    }

    #[test]
    #[ignore]
    fn test_rms_norm_in_transformer_layer() {
        // Gemma/LLaMA layers should use RMSNorm before attention and FFN.
        // Verify the layer output is not NaN/Inf (sanity check for norm stability).
        //
        //   let output = layer.forward(&input, &mask, 0, &backend);
        //   let data = backend.download(&output);
        //   for &v in data.as_f32() {
        //       assert!(v.is_finite(), "RMSNorm produced non-finite value");
        //   }
        todo!("Implement RMSNorm stability test");
    }

    #[test]
    #[ignore]
    fn test_swiglu_ffn_in_transformer_layer() {
        // Gemma/LLaMA use SwiGLU in the FFN block:
        //   hidden = swiglu(gate_proj(x), up_proj(x))
        //   output = down_proj(hidden)
        // The gate and up projections should have the same shape,
        // and the output should be projected back to hidden_size.
        //
        //   let output = layer.forward(&input, &mask, 0, &backend);
        //   assert_eq!(output.shape(), &[seq_len, hidden_size]);
        todo!("Implement SwiGLU FFN forward test");
    }

    #[test]
    #[ignore]
    fn test_gqa_multi_head_attention() {
        // When head_count_kv < head_count (Grouped Query Attention),
        // K/V heads are repeated across groups. The forward pass should
        // handle this correctly without shape errors.
        //
        //   let config = ModelConfig { num_heads: 16, num_kv_heads: 4, .. };
        //   let output = layer.forward(&input, &mask, 0, &backend);
        //   assert_eq!(output.shape(), input.shape());
        todo!("Implement GQA forward pass test");
    }

    #[test]
    #[ignore]
    fn test_full_model_forward_output_shape() {
        // Running all layers should produce output of shape
        // [seq_len, hidden_size] after the final norm.
        //
        //   let model = TransformerModel::from_gguf(&gguf, &backend).unwrap();
        //   let output = model.forward(&input_ids, &attention_mask, &backend);
        //   assert_eq!(output.shape(), &[seq_len, hidden_size]);
        todo!("Implement full model forward pass");
    }

    #[test]
    #[ignore]
    fn test_residual_connections_preserved() {
        // Transformer layers use residual connections:
        //   output = input + attention(norm(input))
        // With zero-initialized weights, the output should equal the input
        // (since attention output would be zero).
        // This verifies the residual path is wired correctly.
        todo!("Implement residual connection verification");
    }

    #[test]
    #[ignore]
    fn test_quantized_weights_in_forward_pass() {
        // Forward pass with Q8_0 quantized weights should use
        // quantized_matmul (fused dequant) rather than dequantizing
        // the full weight matrix first.
        // The output should be close to the f32 reference (cosine sim > 0.99).
        todo!("Implement quantized forward pass comparison");
    }
}
