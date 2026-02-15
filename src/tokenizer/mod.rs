//! Tokenizer trait and utilities for encoding text to token IDs.
//!
//! This module defines the [`Tokenizer`] trait implemented by both [`BpeTokenizer`]
//! and [`WordPieceTokenizer`], along with batch encoding and padding utilities.

pub mod bpe;
pub mod wordpiece;

pub use bpe::BpeTokenizer;
pub use wordpiece::WordPieceTokenizer;

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
}
