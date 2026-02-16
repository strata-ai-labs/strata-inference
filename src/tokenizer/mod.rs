//! Tokenizer trait and utilities for encoding text to token IDs.
//!
//! This module defines the [`Tokenizer`] trait implemented by both [`BpeTokenizer`]
//! and [`WordPieceTokenizer`], along with batch encoding and padding utilities.

pub mod bpe;
pub mod wordpiece;

pub use bpe::BpeTokenizer;
pub use wordpiece::WordPieceTokenizer;

use tracing::warn;

use crate::error::InferenceError;
use crate::gguf::{GgufFile, GgufValue};

/// A tokenizer that converts text to token IDs and back.
///
/// All implementations must be thread-safe (`Send + Sync`) for concurrent use
/// across multiple threads.
pub trait Tokenizer: Send + Sync {
    /// Encode text into a sequence of token IDs.
    ///
    /// When `add_special_tokens` is true, implementation-specific special tokens
    /// are added (e.g., BOS/EOS for BPE, CLS/SEP for WordPiece).
    fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<u32>;

    /// Decode a sequence of token IDs back into text.
    fn decode(&self, ids: &[u32]) -> String;

    /// Return the total vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Beginning-of-sequence token ID, if applicable.
    fn bos_token_id(&self) -> Option<u32>;

    /// End-of-sequence token ID, if applicable.
    fn eos_token_id(&self) -> Option<u32>;

    /// Padding token ID, if applicable.
    fn pad_token_id(&self) -> Option<u32>;
}

/// Encode a batch of texts using the given tokenizer.
///
/// Each text is encoded independently. Returns one `Vec<u32>` per input text.
pub fn encode_batch(
    tokenizer: &dyn Tokenizer,
    texts: &[&str],
    add_special_tokens: bool,
) -> Vec<Vec<u32>> {
    texts
        .iter()
        .map(|text| tokenizer.encode(text, add_special_tokens))
        .collect()
}

/// Pad a set of token ID sequences to equal length.
///
/// Returns a tuple of `(padded_sequences, attention_masks)`:
/// - `padded_sequences`: each inner `Vec<u32>` is padded with `pad_id` to match
///   the length of the longest sequence.
/// - `attention_masks`: each inner `Vec<u8>` has `1` for real tokens and `0` for
///   padding positions.
///
/// If `sequences` is empty, returns empty vectors.
pub fn pad_sequences(
    sequences: &[Vec<u32>],
    pad_id: u32,
) -> (Vec<Vec<u32>>, Vec<Vec<u8>>) {
    if sequences.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);

    let mut padded = Vec::with_capacity(sequences.len());
    let mut masks = Vec::with_capacity(sequences.len());

    for seq in sequences {
        let real_len = seq.len();

        // Build padded token sequence
        let mut padded_seq = Vec::with_capacity(max_len);
        padded_seq.extend_from_slice(seq);
        padded_seq.resize(max_len, pad_id);

        // Build attention mask
        let mut mask = Vec::with_capacity(max_len);
        mask.resize(real_len, 1u8);
        mask.resize(max_len, 0u8);

        padded.push(padded_seq);
        masks.push(mask);
    }

    (padded, masks)
}

/// Create a tokenizer from GGUF vocabulary metadata.
///
/// Reads `tokenizer.ggml.model` to determine the tokenizer type:
/// - `"llama"`, `"gpt2"`, or other non-`"bert"` values → [`BpeTokenizer`]
/// - `"bert"` → [`WordPieceTokenizer`]
///
/// Requires the `tokenizer.ggml.tokens` string array to be present. Optional
/// keys (`tokenizer.ggml.scores`, `tokenizer.ggml.merges`, etc.) are read when
/// available and default to empty/absent otherwise.
pub fn create_tokenizer_from_gguf(
    gguf: &GgufFile,
) -> Result<Box<dyn Tokenizer>, InferenceError> {
    // Read the token list (required)
    let tokens: Vec<String> = gguf
        .get_str_array("tokenizer.ggml.tokens")
        .ok_or_else(|| {
            InferenceError::Tokenizer(
                "missing required key: tokenizer.ggml.tokens".to_string(),
            )
        })?
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    if tokens.is_empty() {
        return Err(InferenceError::Tokenizer(
            "tokenizer.ggml.tokens is empty".to_string(),
        ));
    }

    // Read tokenizer model type (default to "llama" i.e. BPE if absent)
    let model_type = gguf.get_str("tokenizer.ggml.model").unwrap_or("llama");

    match model_type {
        "llama" | "gpt2" | "bert" => {}
        other => {
            warn!(
                model_type = other,
                "Unknown tokenizer model type, falling back to BPE"
            );
        }
    }

    // Read common special token IDs
    let bos_id = gguf.get_u32("tokenizer.ggml.bos_token_id");
    let eos_id = gguf.get_u32("tokenizer.ggml.eos_token_id");
    let pad_id = gguf.get_u32("tokenizer.ggml.padding_token_id");

    if model_type == "bert" {
        // WordPiece path
        let cls_id = find_token_id(&tokens, "[CLS]").unwrap_or(101);
        let sep_id = find_token_id(&tokens, "[SEP]").unwrap_or(102);
        let unk_id = find_token_id(&tokens, "[UNK]").unwrap_or(100);
        let wp_pad_id = find_token_id(&tokens, "[PAD]").unwrap_or(pad_id.unwrap_or(0));

        Ok(Box::new(WordPieceTokenizer::new(
            tokens, cls_id, sep_id, unk_id, wp_pad_id,
        )))
    } else {
        // BPE path (llama, gpt2, and all other model types)
        let scores = gguf
            .get_f32_array("tokenizer.ggml.scores")
            .unwrap_or_default();

        let token_types = match gguf.get("tokenizer.ggml.token_type") {
            Some(GgufValue::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    GgufValue::U32(n) => *n,
                    GgufValue::I32(n) => *n as u32,
                    _ => 0,
                })
                .collect(),
            _ => Vec::new(),
        };

        let merges: Vec<String> = gguf
            .get_str_array("tokenizer.ggml.merges")
            .map(|v| v.into_iter().map(|s| s.to_string()).collect())
            .unwrap_or_default();

        let add_bos = gguf.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(false);
        let add_eos = gguf.get_bool("tokenizer.ggml.add_eos_token").unwrap_or(false);

        Ok(Box::new(BpeTokenizer::new(
            tokens,
            scores,
            token_types,
            merges,
            bos_id,
            eos_id,
            pad_id,
            add_bos,
            add_eos,
        )))
    }
}

/// Find a token's ID by its string value.
fn find_token_id(tokens: &[String], target: &str) -> Option<u32> {
    tokens.iter().position(|t| t == target).map(|i| i as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal tokenizer for testing the trait utilities.
    struct MockTokenizer {
        vocab: Vec<String>,
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
                ],
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
            if add_special_tokens {
                ids.push(2); // EOS
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
            Some(2)
        }

        fn pad_token_id(&self) -> Option<u32> {
            Some(0)
        }
    }

    #[test]
    fn test_encode_batch_basic() {
        let tok = MockTokenizer::new();
        let results = encode_batch(&tok, &["hello world", "foo"], true);
        assert_eq!(results.len(), 2);
        // "hello world" -> [BOS=1, hello=3, world=4, EOS=2]
        assert_eq!(results[0], vec![1, 3, 4, 2]);
        // "foo" -> [BOS=1, foo=5, EOS=2]
        assert_eq!(results[1], vec![1, 5, 2]);
    }

    #[test]
    fn test_encode_batch_no_special() {
        let tok = MockTokenizer::new();
        let results = encode_batch(&tok, &["hello"], false);
        assert_eq!(results[0], vec![3]);
    }

    #[test]
    fn test_encode_batch_empty() {
        let tok = MockTokenizer::new();
        let results = encode_batch(&tok, &[], true);
        assert!(results.is_empty());
    }

    #[test]
    fn test_pad_sequences_basic() {
        let sequences = vec![
            vec![1, 3, 4, 2],  // length 4
            vec![1, 5, 2],     // length 3
        ];
        let (padded, masks) = pad_sequences(&sequences, 0);

        assert_eq!(padded.len(), 2);
        assert_eq!(masks.len(), 2);

        // Both padded to length 4
        assert_eq!(padded[0], vec![1, 3, 4, 2]);
        assert_eq!(padded[1], vec![1, 5, 2, 0]);

        assert_eq!(masks[0], vec![1, 1, 1, 1]);
        assert_eq!(masks[1], vec![1, 1, 1, 0]);
    }

    #[test]
    fn test_pad_sequences_same_length() {
        let sequences = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let (padded, masks) = pad_sequences(&sequences, 0);

        // No padding needed
        assert_eq!(padded[0], vec![1, 2, 3]);
        assert_eq!(padded[1], vec![4, 5, 6]);
        assert_eq!(masks[0], vec![1, 1, 1]);
        assert_eq!(masks[1], vec![1, 1, 1]);
    }

    #[test]
    fn test_pad_sequences_empty() {
        let sequences: Vec<Vec<u32>> = vec![];
        let (padded, masks) = pad_sequences(&sequences, 0);
        assert!(padded.is_empty());
        assert!(masks.is_empty());
    }

    #[test]
    fn test_pad_sequences_single() {
        let sequences = vec![vec![10, 20]];
        let (padded, masks) = pad_sequences(&sequences, 99);
        assert_eq!(padded[0], vec![10, 20]);
        assert_eq!(masks[0], vec![1, 1]);
    }

    #[test]
    fn test_pad_sequences_varied_lengths() {
        let sequences = vec![
            vec![1],           // length 1
            vec![1, 2, 3, 4],  // length 4
            vec![1, 2],        // length 2
        ];
        let (padded, masks) = pad_sequences(&sequences, 0);

        assert_eq!(padded[0], vec![1, 0, 0, 0]);
        assert_eq!(padded[1], vec![1, 2, 3, 4]);
        assert_eq!(padded[2], vec![1, 2, 0, 0]);

        assert_eq!(masks[0], vec![1, 0, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 1, 1]);
        assert_eq!(masks[2], vec![1, 1, 0, 0]);
    }

    #[test]
    fn test_pad_sequences_with_empty_sequence() {
        let sequences = vec![vec![], vec![1, 2]];
        let (padded, masks) = pad_sequences(&sequences, 0);

        assert_eq!(padded[0], vec![0, 0]);
        assert_eq!(padded[1], vec![1, 2]);

        assert_eq!(masks[0], vec![0, 0]);
        assert_eq!(masks[1], vec![1, 1]);
    }

    #[test]
    fn test_trait_object_dispatch() {
        // Ensure Tokenizer can be used as a trait object
        let tok: Box<dyn Tokenizer> = Box::new(MockTokenizer::new());
        let ids = tok.encode("hello", true);
        assert_eq!(ids, vec![1, 3, 2]);
        assert_eq!(tok.vocab_size(), 6);
        assert_eq!(tok.bos_token_id(), Some(1));
        assert_eq!(tok.eos_token_id(), Some(2));
        assert_eq!(tok.pad_token_id(), Some(0));
    }

    #[test]
    fn test_decode() {
        let tok = MockTokenizer::new();
        let text = tok.decode(&[3, 4]);
        assert_eq!(text, "hello world");
    }

    // ====================================================================
    // NEW TESTS: Tokenizer trait and padding edge cases
    // ====================================================================

    #[test]
    fn test_encode_batch_single_element() {
        let tok = MockTokenizer::new();
        let results = encode_batch(&tok, &["hello"], true);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], vec![1, 3, 2]);
    }

    #[test]
    fn test_pad_sequences_all_empty() {
        // All sequences are empty.
        let sequences = vec![vec![], vec![], vec![]];
        let (padded, masks) = pad_sequences(&sequences, 0);
        assert_eq!(padded.len(), 3);
        // Max length is 0, so all padded sequences and masks are empty.
        for p in &padded {
            assert!(p.is_empty());
        }
        for m in &masks {
            assert!(m.is_empty());
        }
    }

    #[test]
    fn test_pad_sequences_single_empty_sequence() {
        let sequences = vec![vec![]];
        let (padded, masks) = pad_sequences(&sequences, 99);
        assert_eq!(padded.len(), 1);
        assert!(padded[0].is_empty());
        assert!(masks[0].is_empty());
    }

    #[test]
    fn test_pad_sequences_large_pad_id() {
        // Verify that a large pad_id value works correctly.
        let sequences = vec![vec![1], vec![2, 3]];
        let (padded, masks) = pad_sequences(&sequences, u32::MAX);
        assert_eq!(padded[0], vec![1, u32::MAX]);
        assert_eq!(masks[0], vec![1, 0]);
    }

    #[test]
    fn test_pad_sequences_preserves_order() {
        let sequences = vec![vec![10, 20, 30], vec![40], vec![50, 60]];
        let (padded, masks) = pad_sequences(&sequences, 0);
        assert_eq!(padded[0], vec![10, 20, 30]);
        assert_eq!(padded[1], vec![40, 0, 0]);
        assert_eq!(padded[2], vec![50, 60, 0]);
        assert_eq!(masks[0], vec![1, 1, 1]);
        assert_eq!(masks[1], vec![1, 0, 0]);
        assert_eq!(masks[2], vec![1, 1, 0]);
    }

    #[test]
    fn test_encode_batch_with_padding() {
        // Full pipeline: batch encode then pad.
        let tok = MockTokenizer::new();
        let encoded = encode_batch(&tok, &["hello world", "foo"], true);
        let (padded, masks) = pad_sequences(&encoded, tok.pad_token_id().unwrap());

        assert_eq!(padded.len(), 2);
        // Longest is 4, shorter gets padded
        assert_eq!(padded[0].len(), 4);
        assert_eq!(padded[1].len(), 4);
        assert_eq!(masks[0], vec![1, 1, 1, 1]);
        assert_eq!(masks[1], vec![1, 1, 1, 0]);
    }

    #[test]
    fn test_mock_tokenizer_decode_empty() {
        let tok = MockTokenizer::new();
        let text = tok.decode(&[]);
        assert_eq!(text, "");
    }

    #[test]
    fn test_mock_tokenizer_unknown_word() {
        let tok = MockTokenizer::new();
        // "unknown" is not in the mock vocab, falls back to id 0 ("<pad>")
        let ids = tok.encode("unknown", false);
        assert_eq!(ids, vec![0]);
    }

    // ====================================================================
    // create_tokenizer_from_gguf tests with synthetic GGUF data
    // ====================================================================

    use std::io::Write;
    use tempfile::NamedTempFile;
    use crate::gguf::GgufFile;

    const GGUF_MAGIC: u32 = 0x4655_4747;

    fn align_offset(offset: u64, alignment: u64) -> u64 {
        let remainder = offset % alignment;
        if remainder == 0 { offset } else { offset + (alignment - remainder) }
    }

    fn build_gguf_bytes(kv_pairs: &[(&str, &[u8])]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&(kv_pairs.len() as u64).to_le_bytes());
        for &(key, value_bytes) in kv_pairs {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(value_bytes);
        }
        let pos = buf.len();
        let aligned = align_offset(pos as u64, 32) as usize;
        buf.resize(aligned, 0);
        buf
    }

    fn kv_string(val: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val.as_bytes());
        buf
    }

    fn kv_u32(val: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&val.to_le_bytes());
        buf
    }

    fn kv_bool(val: bool) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&7u32.to_le_bytes());
        buf.push(if val { 1 } else { 0 });
        buf
    }

    fn kv_str_array(vals: &[&str]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
        buf.extend_from_slice(&8u32.to_le_bytes()); // element type: String
        buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for s in vals {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        buf
    }

    fn kv_f32_array(vals: &[f32]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
        buf.extend_from_slice(&6u32.to_le_bytes()); // element type: F32
        buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    fn write_temp_gguf(bytes: &[u8]) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("failed to create temp file");
        file.write_all(bytes).expect("failed to write temp file");
        file.flush().expect("failed to flush temp file");
        file
    }

    fn open_gguf_from_kv(kv_pairs: &[(&str, Vec<u8>)]) -> (GgufFile, NamedTempFile) {
        let refs: Vec<(&str, &[u8])> = kv_pairs
            .iter()
            .map(|(k, v)| (*k, v.as_slice()))
            .collect();
        let bytes = build_gguf_bytes(&refs);
        let file = write_temp_gguf(&bytes);
        let gguf = GgufFile::open(file.path()).expect("failed to open temp GGUF");
        (gguf, file)
    }

    #[test]
    fn test_create_tokenizer_from_gguf_bpe() {
        let kv = vec![
            ("tokenizer.ggml.model", kv_string("llama")),
            ("tokenizer.ggml.tokens", kv_str_array(&[
                "<pad>", "<bos>", "<eos>", "\u{2581}hello", "\u{2581}world",
            ])),
            ("tokenizer.ggml.scores", kv_f32_array(&[0.0, 0.0, 0.0, -1.0, -2.0])),
            ("tokenizer.ggml.bos_token_id", kv_u32(1)),
            ("tokenizer.ggml.eos_token_id", kv_u32(2)),
            ("tokenizer.ggml.add_bos_token", kv_bool(true)),
            ("tokenizer.ggml.add_eos_token", kv_bool(false)),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let tok = create_tokenizer_from_gguf(&gguf).unwrap();
        assert_eq!(tok.vocab_size(), 5);
        assert_eq!(tok.bos_token_id(), Some(1));
        assert_eq!(tok.eos_token_id(), Some(2));
    }

    #[test]
    fn test_create_tokenizer_from_gguf_wordpiece() {
        let kv = vec![
            ("tokenizer.ggml.model", kv_string("bert")),
            ("tokenizer.ggml.tokens", kv_str_array(&[
                "[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world",
            ])),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let tok = create_tokenizer_from_gguf(&gguf).unwrap();
        assert_eq!(tok.vocab_size(), 6);
        // CLS=2, SEP=3 based on token positions in the vocab
        assert_eq!(tok.bos_token_id(), Some(2)); // WordPiece returns CLS as BOS
        assert_eq!(tok.eos_token_id(), Some(3)); // WordPiece returns SEP as EOS
        assert_eq!(tok.pad_token_id(), Some(0));
    }

    #[test]
    fn test_create_tokenizer_from_gguf_missing_tokens() {
        let kv: Vec<(&str, Vec<u8>)> = vec![
            ("tokenizer.ggml.model", kv_string("llama")),
            // No tokenizer.ggml.tokens key
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let result = create_tokenizer_from_gguf(&gguf);
        match result {
            Err(InferenceError::Tokenizer(msg)) => {
                assert!(msg.contains("tokenizer.ggml.tokens"), "Error: {}", msg);
            }
            Err(e) => panic!("expected Tokenizer error, got {:?}", e),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn test_create_tokenizer_from_gguf_empty_tokens() {
        let kv = vec![
            ("tokenizer.ggml.model", kv_string("llama")),
            ("tokenizer.ggml.tokens", kv_str_array(&[])),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let result = create_tokenizer_from_gguf(&gguf);
        match result {
            Err(InferenceError::Tokenizer(msg)) => {
                assert!(msg.contains("empty"), "Error: {}", msg);
            }
            Err(e) => panic!("expected Tokenizer error for empty tokens, got {:?}", e),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn test_create_tokenizer_from_gguf_defaults_to_bpe() {
        // No tokenizer.ggml.model key → defaults to "llama" (BPE)
        let kv = vec![
            ("tokenizer.ggml.tokens", kv_str_array(&[
                "<pad>", "<bos>", "<eos>", "a", "b",
            ])),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let tok = create_tokenizer_from_gguf(&gguf).unwrap();
        assert_eq!(tok.vocab_size(), 5);
        // Default BPE with add_bos=true should add BOS
        assert_eq!(tok.bos_token_id(), None); // No bos_token_id key in metadata
    }

    #[test]
    fn test_create_tokenizer_from_gguf_bert_finds_special_tokens() {
        // BERT tokens in non-standard positions
        let kv = vec![
            ("tokenizer.ggml.model", kv_string("bert")),
            ("tokenizer.ggml.tokens", kv_str_array(&[
                "a", "b", "[UNK]", "[PAD]", "[CLS]", "[SEP]", "hello",
            ])),
        ];
        let (gguf, _tmp) = open_gguf_from_kv(&kv);

        let tok = create_tokenizer_from_gguf(&gguf).unwrap();
        assert_eq!(tok.vocab_size(), 7);
        assert_eq!(tok.pad_token_id(), Some(3)); // [PAD] is at position 3
    }
}
