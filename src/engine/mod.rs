//! High-level inference APIs.
//!
//! - [`EmbeddingEngine`]: text → dense vector embedding (M5)
//! - [`GenerationEngine`]: prompt → text via autoregressive decoding (M6)

pub mod embed;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) mod exec_metal;
pub mod generate;
#[allow(dead_code)]
pub(crate) mod graph;
pub mod sampler;

pub use embed::EmbeddingEngine;
pub use generate::GenerationEngine;

#[cfg(test)]
mod tests {
    // ====================================================================
    // M5: Integration tests requiring real GGUF model files.
    //
    // These tests remain #[ignore] until a test model artifact is available.
    // They correspond to issue #15 (accuracy validation).
    // ====================================================================

    #[test]
    #[ignore]
    fn test_full_pipeline_gguf_to_embedding() {
        // End-to-end: open GGUF file, parse metadata, load tokenizer,
        // load model weights, tokenize input, run forward pass, pool,
        // normalize, return embedding.
        //
        //   let engine = EmbeddingEngine::from_gguf("model.gguf", backend).unwrap();
        //   let embedding = engine.embed("Hello, world!").unwrap();
        //   assert_eq!(embedding.len(), engine.embedding_dim());
        //   let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        //   assert!((norm - 1.0).abs() < 1e-5);
        todo!("Requires real GGUF model file");
    }

    #[test]
    #[ignore]
    fn test_full_pipeline_tokenizer_from_gguf_vocab() {
        // The tokenizer should be constructed from GGUF vocabulary metadata.
        //
        //   let gguf = GgufFile::open("model.gguf").unwrap();
        //   let tokenizer = create_tokenizer_from_gguf(&gguf).unwrap();
        //   assert!(tokenizer.vocab_size() > 0);
        todo!("Requires real GGUF model file");
    }

    #[test]
    #[ignore]
    fn test_full_pipeline_quantized_matches_f32() {
        // Q8_0 quantized model should produce embeddings very close to
        // the f32 reference (cosine similarity > 0.99).
        todo!("Requires both Q8_0 and F32 model files");
    }

    #[test]
    #[ignore]
    fn test_embedding_cosine_similarity_with_huggingface() {
        // Compare embedding output to HuggingFace reference output.
        // Cosine similarity should be > 0.99.
        todo!("Requires real GGUF model file and HF reference embeddings");
    }

    // ------------------------------------------------------------------
    // M6: GenerationEngine integration tests
    //
    // These require a real causal GGUF model file (e.g., Gemma, LLaMA).
    // Unit tests for GenerationEngine are in generate.rs.
    // ------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_generation_engine_greedy_decode() {
        // End-to-end: load real GGUF, generate with greedy decoding,
        // verify output is deterministic and coherent.
        //
        //   let engine = GenerationEngine::from_gguf("model.gguf").unwrap();
        //   let result = engine.generate("Once upon a time", &gen_config).unwrap();
        //   assert!(!result.is_empty());
        todo!("Requires real causal GGUF model file");
    }

    #[test]
    #[ignore]
    fn test_generation_engine_max_tokens() {
        // Verify max_tokens limit is respected with real model.
        todo!("Requires real causal GGUF model file");
    }

    #[test]
    #[ignore]
    fn test_generation_engine_stop_at_eos() {
        // Verify EOS stopping with real model.
        todo!("Requires real causal GGUF model file");
    }

    #[test]
    #[ignore]
    fn test_generation_engine_kv_cache_consistency() {
        // Verify KV cache produces same results as full recomputation.
        todo!("Requires real causal GGUF model file");
    }
}
