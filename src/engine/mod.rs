//! High-level inference APIs.
//!
//! - [`EmbeddingEngine`]: text → dense vector embedding (M5)
//! - Generation engine: prompt → text (M6, future)

pub mod embed;

pub use embed::EmbeddingEngine;

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
    // M6: GenerationEngine (stretch goal)
    // ------------------------------------------------------------------

    #[test]
    #[ignore]
    fn test_generation_engine_greedy_decode() {
        todo!("Implement greedy generation (M6)");
    }

    #[test]
    #[ignore]
    fn test_generation_engine_max_tokens() {
        todo!("Implement max_tokens limit (M6)");
    }

    #[test]
    #[ignore]
    fn test_generation_engine_stop_at_eos() {
        todo!("Implement EOS stopping (M6)");
    }

    #[test]
    #[ignore]
    fn test_generation_engine_kv_cache() {
        todo!("Implement KV cache test (M6)");
    }
}
