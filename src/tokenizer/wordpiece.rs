//! WordPiece tokenizer for BERT-family models.
//!
//! Implements the WordPiece subword tokenization algorithm used by BERT,
//! DistilBERT, MiniLM, and similar models. The algorithm:
//!
//! 1. Normalizes text: lowercase, NFD decomposition, strip accents.
//! 2. Splits on whitespace and punctuation.
//! 3. For each word, greedily finds the longest prefix match in the vocabulary.
//! 4. Continues with the remainder using the `##` continuation prefix.
//! 5. If no match is found, emits the UNK token.
//! 6. Wraps the result with [CLS] and [SEP] special tokens.
//!
//! Ported and adapted from `strata-core/crates/intelligence/src/embed/tokenizer.rs`.

use std::collections::HashMap;

use tracing::debug;
use unicode_normalization::UnicodeNormalization;

use super::Tokenizer;

/// The continuation prefix for WordPiece subwords.
const CONTINUATION_PREFIX: &str = "##";

/// Maximum sequence length (including special tokens).
const MAX_SEQ_LEN: usize = 512;

/// WordPiece tokenizer compatible with BERT-family vocabularies.
///
/// # Construction
///
/// Use [`WordPieceTokenizer::new`] with a vocabulary extracted from a GGUF file
/// or a vocab.txt file (one token per line).
pub struct WordPieceTokenizer {
    /// Token string -> token ID lookup.
    vocab: HashMap<String, u32>,
    /// Token ID -> token string (for decoding).
    id_to_token: Vec<String>,
    /// [CLS] token ID (beginning of sentence).
    cls_id: u32,
    /// [SEP] token ID (separator / end of sentence).
    sep_id: u32,
    /// [UNK] token ID (unknown token).
    unk_id: u32,
    /// [PAD] token ID (padding).
    pad_id: u32,
}

impl WordPieceTokenizer {
    /// Create a new WordPiece tokenizer from vocabulary data.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Vocabulary tokens, indexed by token ID. The order determines
    ///   the token-to-ID mapping.
    /// * `cls_id` - Token ID for [CLS] (classification / start token).
    /// * `sep_id` - Token ID for [SEP] (separator / end token).
    /// * `unk_id` - Token ID for [UNK] (unknown / out-of-vocabulary token).
    /// * `pad_id` - Token ID for [PAD] (padding token).
    pub fn new(
        tokens: Vec<String>,
        cls_id: u32,
        sep_id: u32,
        unk_id: u32,
        pad_id: u32,
    ) -> Self {
        let mut vocab = HashMap::with_capacity(tokens.len());
        for (i, tok) in tokens.iter().enumerate() {
            vocab.insert(tok.clone(), i as u32);
        }

        debug!(
            vocab_size = tokens.len(),
            cls_id,
            sep_id,
            unk_id,
            pad_id,
            "WordPiece tokenizer initialized"
        );

        Self {
            vocab,
            id_to_token: tokens,
            cls_id,
            sep_id,
            unk_id,
            pad_id,
        }
    }

    /// Convenience constructor from a `vocab.txt` file content (one token per line).
    ///
    /// Assumes standard BERT token IDs:
    /// - [PAD] = 0
    /// - [UNK] = 100
    /// - [CLS] = 101
    /// - [SEP] = 102
    pub fn from_vocab_text(vocab_text: &str) -> Self {
        let tokens: Vec<String> = vocab_text.lines().map(|l| l.to_string()).collect();

        // Look up special token IDs, falling back to standard BERT positions.
        let find_id = |name: &str, fallback: u32| -> u32 {
            tokens
                .iter()
                .position(|t| t == name)
                .map(|i| i as u32)
                .unwrap_or(fallback)
        };

        let cls_id = find_id("[CLS]", 101);
        let sep_id = find_id("[SEP]", 102);
        let unk_id = find_id("[UNK]", 100);
        let pad_id = find_id("[PAD]", 0);

        Self::new(tokens, cls_id, sep_id, unk_id, pad_id)
    }

    /// Normalize a text string for tokenization.
    ///
    /// - Convert to lowercase.
    /// - Apply NFD (canonical decomposition) Unicode normalization.
    /// - Strip combining marks (accents, diacritics).
    fn normalize(&self, text: &str) -> String {
        text.to_lowercase()
            .nfd()
            .filter(|c| !is_combining_mark(*c))
            .collect()
    }

    /// Split text on whitespace and punctuation into individual words.
    ///
    /// Punctuation characters become their own tokens. Multiple whitespace
    /// characters are collapsed.
    fn basic_tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            if ch.is_whitespace() {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            } else if is_punctuation(ch) {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                tokens.push(ch.to_string());
            } else {
                current.push(ch);
            }
        }

        if !current.is_empty() {
            tokens.push(current);
        }

        tokens
    }

    /// Apply WordPiece subword tokenization to a single word.
    ///
    /// Greedily finds the longest prefix match in the vocabulary, then continues
    /// with the remainder using the `##` continuation prefix. If no match is found
    /// for any character, emits UNK.
    fn wordpiece_tokenize(&self, word: &str, output: &mut Vec<u32>) {
        if word.is_empty() {
            return;
        }

        // Try whole word first (common case for short words).
        if let Some(&id) = self.vocab.get(word) {
            output.push(id);
            return;
        }

        let chars: Vec<char> = word.chars().collect();
        let mut start = 0;
        let mut is_first = true;

        while start < chars.len() {
            let mut end = chars.len();
            let mut found = false;

            while start < end {
                let substr: String = if is_first {
                    chars[start..end].iter().collect()
                } else {
                    format!(
                        "{}{}",
                        CONTINUATION_PREFIX,
                        chars[start..end].iter().collect::<String>()
                    )
                };

                if let Some(&id) = self.vocab.get(&substr) {
                    output.push(id);
                    found = true;
                    start = end;
                    is_first = false;
                    break;
                }

                end -= 1;
            }

            if !found {
                // No match for this character at all; emit UNK and move on.
                output.push(self.unk_id);
                start += 1;
                is_first = false;
            }
        }
    }
}

impl Tokenizer for WordPieceTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
        let normalized = self.normalize(text);
        let words = self.basic_tokenize(&normalized);

        let max_content_len = if add_special_tokens {
            MAX_SEQ_LEN - 2 // Reserve space for [CLS] and [SEP]
        } else {
            MAX_SEQ_LEN
        };

        let mut tokens = Vec::new();

        for word in &words {
            self.wordpiece_tokenize(word, &mut tokens);
            if tokens.len() >= max_content_len {
                tokens.truncate(max_content_len);
                break;
            }
        }

        if add_special_tokens {
            // Wrap with [CLS] ... [SEP]
            let mut result = Vec::with_capacity(tokens.len() + 2);
            result.push(self.cls_id);
            result.extend_from_slice(&tokens);
            result.push(self.sep_id);
            result
        } else {
            tokens
        }
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut pieces = Vec::new();
        for &id in ids {
            // Skip special tokens.
            if id == self.cls_id || id == self.sep_id || id == self.pad_id {
                continue;
            }
            if let Some(token) = self.id_to_token.get(id as usize) {
                if let Some(suffix) = token.strip_prefix(CONTINUATION_PREFIX) {
                    // Continuation token: append directly to the previous word
                    // (no space).
                    if let Some(last) = pieces.last_mut() {
                        let s: &mut String = last;
                        s.push_str(suffix);
                    } else {
                        pieces.push(suffix.to_string());
                    }
                } else {
                    pieces.push(token.clone());
                }
            } else if id == self.unk_id {
                pieces.push("[UNK]".to_string());
            }
        }
        pieces.join(" ")
    }

    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    fn bos_token_id(&self) -> Option<u32> {
        Some(self.cls_id)
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(self.sep_id)
    }

    fn pad_token_id(&self) -> Option<u32> {
        Some(self.pad_id)
    }
}

// ---------------------------------------------------------------------------
// Character classification helpers
// ---------------------------------------------------------------------------

/// Check if a character is a Unicode combining mark (category M).
///
/// Combining marks include accents, diacritics, and other marks that combine
/// with the preceding character. Stripping these gives us "unaccented" text.
fn is_combining_mark(c: char) -> bool {
    use std::ops::RangeInclusive;

    // Unicode General Category "M" (Mark) ranges.
    // This covers Mn (nonspacing), Mc (spacing combining), and Me (enclosing).
    const COMBINING_RANGES: &[RangeInclusive<u32>] = &[
        0x0300..=0x036F, // Combining Diacritical Marks
        0x0483..=0x0489, // Combining Cyrillic
        0x0591..=0x05BD, // Hebrew
        0x05BF..=0x05BF,
        0x05C1..=0x05C2,
        0x05C4..=0x05C5,
        0x05C7..=0x05C7,
        0x0610..=0x061A, // Arabic
        0x064B..=0x065F,
        0x0670..=0x0670,
        0x06D6..=0x06DC,
        0x06DF..=0x06E4,
        0x06E7..=0x06E8,
        0x06EA..=0x06ED,
        0x0711..=0x0711, // Syriac
        0x0730..=0x074A,
        0x07A6..=0x07B0, // Thaana
        0x07EB..=0x07F3, // NKo
        0x07FD..=0x07FD,
        0x0816..=0x0819, // Samaritan
        0x081B..=0x0823,
        0x0825..=0x0827,
        0x0829..=0x082D,
        0x0859..=0x085B, // Mandaic
        0x0898..=0x089F, // Arabic Extended-B
        0x08CA..=0x08E1, // Arabic Extended-A
        0x08E3..=0x0903,
        0x093A..=0x093C, // Devanagari
        0x093E..=0x094F,
        0x0951..=0x0957,
        0x0962..=0x0963,
        0x0981..=0x0983, // Bengali
        0x09BC..=0x09BC,
        0x09BE..=0x09C4,
        0x09C7..=0x09C8,
        0x09CB..=0x09CD,
        0x09D7..=0x09D7,
        0x09E2..=0x09E3,
        0x09FE..=0x09FE,
        0x0A01..=0x0A03, // Gurmukhi
        0x0A3C..=0x0A3C,
        0x0A3E..=0x0A42,
        0x0A47..=0x0A48,
        0x0A4B..=0x0A4D,
        0x0A51..=0x0A51,
        0x0A70..=0x0A71,
        0x0A75..=0x0A75,
        0x0A81..=0x0A83, // Gujarati
        0x0ABC..=0x0ABC,
        0x0ABE..=0x0AC5,
        0x0AC7..=0x0AC9,
        0x0ACB..=0x0ACD,
        0x0AE2..=0x0AE3,
        0x0AFA..=0x0AFF,
        0x0B01..=0x0B03, // Oriya
        0x0B3C..=0x0B3C,
        0x0B3E..=0x0B44,
        0x0B47..=0x0B48,
        0x0B4B..=0x0B4D,
        0x0B55..=0x0B57,
        0x0B62..=0x0B63,
        0x0B82..=0x0B82, // Tamil
        0x0BBE..=0x0BC2,
        0x0BC6..=0x0BC8,
        0x0BCA..=0x0BCD,
        0x0BD7..=0x0BD7,
        0x0C00..=0x0C04, // Telugu
        0x0C3C..=0x0C3C,
        0x0C3E..=0x0C44,
        0x0C46..=0x0C48,
        0x0C4A..=0x0C4D,
        0x0C55..=0x0C56,
        0x0C62..=0x0C63,
        0x0C81..=0x0C83, // Kannada
        0x0CBC..=0x0CBC,
        0x0CBE..=0x0CC4,
        0x0CC6..=0x0CC8,
        0x0CCA..=0x0CCD,
        0x0CD5..=0x0CD6,
        0x0CE2..=0x0CE3,
        0x0CF3..=0x0CF3,
        0x0D00..=0x0D03, // Malayalam
        0x0D3B..=0x0D3C,
        0x0D3E..=0x0D44,
        0x0D46..=0x0D48,
        0x0D4A..=0x0D4D,
        0x0D57..=0x0D57,
        0x0D62..=0x0D63,
        0x0D81..=0x0D83, // Sinhala
        0x0DCA..=0x0DCA,
        0x0DCF..=0x0DD4,
        0x0DD6..=0x0DD6,
        0x0DD8..=0x0DDF,
        0x0DF2..=0x0DF3,
        0x0E31..=0x0E31, // Thai
        0x0E34..=0x0E3A,
        0x0E47..=0x0E4E,
        0x0EB1..=0x0EB1, // Lao
        0x0EB4..=0x0EBC,
        0x0EC8..=0x0ECE,
        0x0F18..=0x0F19, // Tibetan
        0x0F35..=0x0F35,
        0x0F37..=0x0F37,
        0x0F39..=0x0F39,
        0x0F3E..=0x0F3F,
        0x0F71..=0x0F84,
        0x0F86..=0x0F87,
        0x0F8D..=0x0F97,
        0x0F99..=0x0FBC,
        0x0FC6..=0x0FC6,
        0x1000..=0x1000, // Myanmar (partial, just common ones)
        0x102B..=0x103E,
        0x1056..=0x1059,
        0x105E..=0x1060,
        0x1062..=0x1064,
        0x1067..=0x106D,
        0x1071..=0x1074,
        0x1082..=0x108D,
        0x108F..=0x108F,
        0x109A..=0x109D,
        0x1100..=0x1100, // placeholder
        0x135D..=0x135F, // Ethiopic
        0x1712..=0x1715, // Tagalog
        0x1732..=0x1734, // Hanunoo
        0x1752..=0x1753, // Buhid
        0x1772..=0x1773, // Tagbanwa
        0x17B4..=0x17D3, // Khmer
        0x17DD..=0x17DD,
        0x180B..=0x180D, // Mongolian
        0x180F..=0x180F,
        0x1885..=0x1886,
        0x18A9..=0x18A9,
        0x1920..=0x192B, // Limbu
        0x1930..=0x193B,
        0x1A17..=0x1A1B, // Buginese
        0x1A55..=0x1A5E, // Tai Tham
        0x1A60..=0x1A7C,
        0x1A7F..=0x1A7F,
        0x1AB0..=0x1ACE, // Combining Diacritical Marks Extended
        0x1B00..=0x1B04, // Balinese
        0x1B34..=0x1B44,
        0x1B6B..=0x1B73,
        0x1B80..=0x1B82, // Sundanese
        0x1BA1..=0x1BAD,
        0x1BE6..=0x1BF3, // Batak
        0x1C24..=0x1C37, // Lepcha
        0x1CD0..=0x1CD2, // Vedic Extensions
        0x1CD4..=0x1CE8,
        0x1CED..=0x1CED,
        0x1CF4..=0x1CF4,
        0x1CF7..=0x1CF9,
        0x1DC0..=0x1DFF, // Combining Diacritical Marks Supplement
        0x20D0..=0x20F0, // Combining Diacritical Marks for Symbols
        0x2CEF..=0x2CF1, // Coptic
        0x2D7F..=0x2D7F, // Tifinagh
        0x2DE0..=0x2DFF, // Cyrillic Extended-A
        0xA66F..=0xA672, // Combining Cyrillic
        0xA674..=0xA67D,
        0xA69E..=0xA69F,
        0xA6F0..=0xA6F1, // Bamum
        0xA802..=0xA802, // Syloti Nagri
        0xA806..=0xA806,
        0xA80B..=0xA80B,
        0xA823..=0xA827,
        0xA82C..=0xA82C,
        0xA880..=0xA881, // Saurashtra
        0xA8B4..=0xA8C5,
        0xA8E0..=0xA8F1, // Combining Devanagari
        0xA8FF..=0xA8FF,
        0xA926..=0xA92D, // Kayah Li
        0xA947..=0xA953, // Rejang
        0xA980..=0xA983, // Javanese
        0xA9B3..=0xA9C0,
        0xA9E5..=0xA9E5, // Myanmar Extended-B
        0xAA29..=0xAA36, // Cham
        0xAA43..=0xAA43,
        0xAA4C..=0xAA4D,
        0xAA7B..=0xAA7D, // Myanmar Extended-A
        0xAAB0..=0xAAB0, // Tai Viet
        0xAAB2..=0xAAB4,
        0xAAB7..=0xAAB8,
        0xAABE..=0xAABF,
        0xAAC1..=0xAAC1,
        0xAAEB..=0xAAEF, // Meetei Mayek
        0xAAF5..=0xAAF6,
        0xABE3..=0xABEA, // Meetei Mayek Extensions
        0xABEC..=0xABED,
        0xFB1E..=0xFB1E, // Hebrew
        0xFE00..=0xFE0F, // Variation Selectors
        0xFE20..=0xFE2F, // Combining Half Marks
    ];

    let cp = c as u32;
    for range in COMBINING_RANGES {
        if range.contains(&cp) {
            return true;
        }
    }

    // Also check high Unicode ranges (supplementary planes).
    // These include various script-specific combining marks.
    if (0x101FD..=0x101FD).contains(&cp)
        || (0x102E0..=0x102E0).contains(&cp)
        || (0x10376..=0x1037A).contains(&cp)
        || (0x10A01..=0x10A03).contains(&cp)
        || (0x10A05..=0x10A06).contains(&cp)
        || (0x10A0C..=0x10A0F).contains(&cp)
        || (0x10A38..=0x10A3A).contains(&cp)
        || (0x10A3F..=0x10A3F).contains(&cp)
        || (0x10AE5..=0x10AE6).contains(&cp)
        || (0x10D24..=0x10D27).contains(&cp)
        || (0x10EAB..=0x10EAC).contains(&cp)
        || (0x10EFD..=0x10EFF).contains(&cp)
        || (0x10F46..=0x10F50).contains(&cp)
        || (0x10F82..=0x10F85).contains(&cp)
        || (0x11001..=0x11001).contains(&cp)
        || (0x11038..=0x11046).contains(&cp)
        || (0x11070..=0x11070).contains(&cp)
        || (0x11073..=0x11074).contains(&cp)
        || (0x1107F..=0x11082).contains(&cp)
        || (0x110B0..=0x110BA).contains(&cp)
        || (0x110C2..=0x110C2).contains(&cp)
        || (0x11100..=0x11102).contains(&cp)
        || (0x11127..=0x11134).contains(&cp)
        || (0x11145..=0x11146).contains(&cp)
        || (0x11173..=0x11173).contains(&cp)
        || (0x11180..=0x11182).contains(&cp)
        || (0x111B3..=0x111C0).contains(&cp)
        || (0x111C9..=0x111CC).contains(&cp)
        || (0x111CE..=0x111CF).contains(&cp)
        || (0x1122C..=0x11237).contains(&cp)
        || (0x1123E..=0x1123E).contains(&cp)
        || (0x11241..=0x11241).contains(&cp)
        || (0x112DF..=0x112EA).contains(&cp)
        || (0x11300..=0x11303).contains(&cp)
        || (0x1133B..=0x1133C).contains(&cp)
        || (0x1133E..=0x11344).contains(&cp)
        || (0x11347..=0x11348).contains(&cp)
        || (0x1134B..=0x1134D).contains(&cp)
        || (0x11357..=0x11357).contains(&cp)
        || (0x11362..=0x11363).contains(&cp)
        || (0x11366..=0x1136C).contains(&cp)
        || (0x11370..=0x11374).contains(&cp)
        || (0xE0100..=0xE01EF).contains(&cp)
    {
        return true;
    }

    false
}

/// Check if a character is punctuation.
///
/// Uses a broad definition that covers ASCII punctuation and common Unicode
/// punctuation categories.
fn is_punctuation(ch: char) -> bool {
    // ASCII punctuation ranges.
    let cp = ch as u32;
    if (0x21..=0x2F).contains(&cp)       // ! " # $ % & ' ( ) * + , - . /
        || (0x3A..=0x40).contains(&cp)    // : ; < = > ? @
        || (0x5B..=0x60).contains(&cp)    // [ \ ] ^ _ `
        || (0x7B..=0x7E).contains(&cp)    // { | } ~
    {
        return true;
    }

    // Unicode general category P (Punctuation).
    // We check the most common ranges rather than the full Unicode database.
    if ch.is_ascii() {
        return false;
    }

    // General punctuation categories for non-ASCII.
    matches!(
        unicode_general_category_p(cp),
        true
    )
}

/// Rough check for Unicode General Category "P" (Punctuation) for non-ASCII chars.
///
/// Covers the most common punctuation ranges. Not exhaustive but sufficient for
/// BERT tokenization of typical text.
fn unicode_general_category_p(cp: u32) -> bool {
    // Common Unicode punctuation blocks.
    (0x00A1..=0x00BF).contains(&cp)       // Latin-1 Supplement punctuation
        || (0x2000..=0x206F).contains(&cp) // General Punctuation
        || (0x2E00..=0x2E7F).contains(&cp) // Supplemental Punctuation
        || (0x3000..=0x303F).contains(&cp) // CJK Symbols and Punctuation
        || (0xFE30..=0xFE4F).contains(&cp) // CJK Compatibility Forms
        || (0xFE50..=0xFE6F).contains(&cp) // Small Form Variants
        || (0xFF01..=0xFF0F).contains(&cp) // Fullwidth punctuation
        || (0xFF1A..=0xFF20).contains(&cp) // Fullwidth punctuation
        || (0xFF3B..=0xFF40).contains(&cp) // Fullwidth punctuation
        || (0xFF5B..=0xFF65).contains(&cp) // Fullwidth punctuation
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: build a test vocabulary
    // -----------------------------------------------------------------------

    /// Build a minimal BERT-like vocabulary for testing.
    ///
    /// Token layout:
    ///   0: [PAD]
    ///   1-99: (filler)
    ///  100: [UNK]
    ///  101: [CLS]
    ///  102: [SEP]
    ///  103: hello
    ///  104: world
    ///  105: ##ing
    ///  106: test
    ///  107: ,
    ///  108: the
    ///  109: ##s
    ///  110: run
    ///  111: ##ning
    ///  112: a
    ///  113: cafe  (will match "cafe" after accent stripping from "cafe")
    fn make_test_vocab() -> Vec<String> {
        let mut tokens: Vec<String> = (0..100).map(|_| "[PAD]".to_string()).collect();
        tokens[0] = "[PAD]".to_string();
        tokens.push("[UNK]".to_string());   // 100
        tokens.push("[CLS]".to_string());   // 101
        tokens.push("[SEP]".to_string());   // 102
        tokens.push("hello".to_string());   // 103
        tokens.push("world".to_string());   // 104
        tokens.push("##ing".to_string());   // 105
        tokens.push("test".to_string());    // 106
        tokens.push(",".to_string());       // 107
        tokens.push("the".to_string());     // 108
        tokens.push("##s".to_string());     // 109
        tokens.push("run".to_string());     // 110
        tokens.push("##ning".to_string());  // 111
        tokens.push("a".to_string());       // 112
        tokens.push("cafe".to_string());    // 113
        tokens
    }

    fn make_tokenizer() -> WordPieceTokenizer {
        WordPieceTokenizer::new(make_test_vocab(), 101, 102, 100, 0)
    }

    // -----------------------------------------------------------------------
    // Basic encoding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_hello_world() {
        let tok = make_tokenizer();
        let ids = tok.encode("hello world", true);
        // [CLS]=101, hello=103, world=104, [SEP]=102
        assert_eq!(ids, vec![101, 103, 104, 102]);
    }

    #[test]
    fn test_encode_without_special_tokens() {
        let tok = make_tokenizer();
        let ids = tok.encode("hello world", false);
        // No CLS/SEP
        assert_eq!(ids, vec![103, 104]);
    }

    #[test]
    fn test_encode_subword() {
        let tok = make_tokenizer();
        let ids = tok.encode("testing", true);
        // "testing" -> "test" + "##ing"
        // [CLS]=101, test=106, ##ing=105, [SEP]=102
        assert_eq!(ids, vec![101, 106, 105, 102]);
    }

    #[test]
    fn test_encode_unknown_word() {
        let tok = make_tokenizer();
        let ids = tok.encode("xyz", true);
        // "xyz" is not in vocab; each char is unknown.
        // [CLS], UNK, UNK, UNK, [SEP]  (one UNK per char since no single-char matches)
        assert_eq!(ids[0], 101); // CLS
        assert_eq!(*ids.last().unwrap(), 102); // SEP
        // Middle tokens are all UNK.
        for &id in &ids[1..ids.len() - 1] {
            assert_eq!(id, 100);
        }
    }

    #[test]
    fn test_encode_empty() {
        let tok = make_tokenizer();
        let ids = tok.encode("", true);
        assert_eq!(ids, vec![101, 102]); // [CLS], [SEP]
    }

    #[test]
    fn test_encode_empty_no_special() {
        let tok = make_tokenizer();
        let ids = tok.encode("", false);
        assert!(ids.is_empty());
    }

    // -----------------------------------------------------------------------
    // Case insensitivity
    // -----------------------------------------------------------------------

    #[test]
    fn test_case_insensitive() {
        let tok = make_tokenizer();
        let upper = tok.encode("HELLO World", true);
        let lower = tok.encode("hello world", true);
        assert_eq!(upper, lower);
    }

    // -----------------------------------------------------------------------
    // Accent stripping
    // -----------------------------------------------------------------------

    #[test]
    fn test_accent_stripping() {
        let tok = make_tokenizer();
        // "cafe" with accent (U+0301 combining acute) should normalize to "cafe".
        let ids = tok.encode("caf\u{00E9}", true); // e-acute
        // After NFD + strip combining marks: "cafe"
        // [CLS], cafe=113, [SEP]
        assert_eq!(ids, vec![101, 113, 102]);
    }

    // -----------------------------------------------------------------------
    // Punctuation splitting
    // -----------------------------------------------------------------------

    #[test]
    fn test_punctuation_split() {
        let tok = make_tokenizer();
        let ids = tok.encode("hello,world", true);
        // "hello" , "," , "world"
        // [CLS]=101, hello=103, ","=107, world=104, [SEP]=102
        assert_eq!(ids, vec![101, 103, 107, 104, 102]);
    }

    // -----------------------------------------------------------------------
    // Multiple continuation tokens
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_subwords() {
        let tok = make_tokenizer();
        let ids = tok.encode("running", true);
        // "running" -> "run" + "##ning"
        // [CLS]=101, run=110, ##ning=111, [SEP]=102
        assert_eq!(ids, vec![101, 110, 111, 102]);
    }

    // -----------------------------------------------------------------------
    // Whitespace handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_spaces() {
        let tok = make_tokenizer();
        let single = tok.encode("hello world", true);
        let multi = tok.encode("hello   world", true);
        assert_eq!(single, multi);
    }

    #[test]
    fn test_leading_trailing_spaces() {
        let tok = make_tokenizer();
        let ids = tok.encode("  hello  ", true);
        // Should be same as "hello".
        assert_eq!(ids, vec![101, 103, 102]);
    }

    // -----------------------------------------------------------------------
    // Decoding
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_basic() {
        let tok = make_tokenizer();
        let text = tok.decode(&[101, 103, 104, 102]);
        // Should produce "hello world" (CLS and SEP stripped).
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_decode_subwords() {
        let tok = make_tokenizer();
        let text = tok.decode(&[101, 106, 105, 102]);
        // "test" + "##ing" → "testing"
        assert_eq!(text, "testing");
    }

    #[test]
    fn test_decode_roundtrip() {
        let tok = make_tokenizer();
        let ids = tok.encode("hello world", true);
        let text = tok.decode(&ids);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_decode_roundtrip_subword() {
        let tok = make_tokenizer();
        let ids = tok.encode("testing", true);
        let text = tok.decode(&ids);
        assert_eq!(text, "testing");
    }

    // -----------------------------------------------------------------------
    // Trait interface
    // -----------------------------------------------------------------------

    #[test]
    fn test_vocab_size() {
        let tok = make_tokenizer();
        assert_eq!(tok.vocab_size(), 114);
    }

    #[test]
    fn test_special_token_ids() {
        let tok = make_tokenizer();
        assert_eq!(tok.bos_token_id(), Some(101)); // CLS
        assert_eq!(tok.eos_token_id(), Some(102)); // SEP
        assert_eq!(tok.pad_token_id(), Some(0));   // PAD
    }

    #[test]
    fn test_trait_object() {
        let tok = make_tokenizer();
        let tok: &dyn Tokenizer = &tok;
        let ids = tok.encode("hello", true);
        assert_eq!(ids, vec![101, 103, 102]);
    }

    // -----------------------------------------------------------------------
    // from_vocab_text constructor
    // -----------------------------------------------------------------------

    #[test]
    fn test_from_vocab_text() {
        let mut lines = vec!["[PAD]"; 101];
        lines[0] = "[PAD]";
        lines[100] = "[UNK]";
        lines.push("[CLS]"); // 101
        lines.push("[SEP]"); // 102
        lines.push("hello"); // 103
        lines.push("world"); // 104
        let vocab_text = lines.join("\n");

        let tok = WordPieceTokenizer::from_vocab_text(&vocab_text);
        let ids = tok.encode("hello world", true);
        assert_eq!(ids[0], 101); // CLS
        assert_eq!(ids[1], 103); // hello
        assert_eq!(ids[2], 104); // world
        assert_eq!(*ids.last().unwrap(), 102); // SEP
    }

    // -----------------------------------------------------------------------
    // Thread safety
    // -----------------------------------------------------------------------

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WordPieceTokenizer>();
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_only_punctuation() {
        let tok = make_tokenizer();
        let ids = tok.encode(",,,", true);
        // Three commas, each is a separate token.
        assert_eq!(ids, vec![101, 107, 107, 107, 102]);
    }

    #[test]
    fn test_cls_sep_always_present() {
        let tok = make_tokenizer();
        for text in &["", "hello", "hello world", "a b c d e f"] {
            let ids = tok.encode(text, true);
            assert_eq!(ids[0], 101, "Missing CLS for '{}'", text);
            assert_eq!(*ids.last().unwrap(), 102, "Missing SEP for '{}'", text);
        }
    }

    #[test]
    fn test_single_letter() {
        let tok = make_tokenizer();
        let ids = tok.encode("a", true);
        // "a" is in vocab as id 112.
        assert_eq!(ids, vec![101, 112, 102]);
    }

    // -----------------------------------------------------------------------
    // Normalization helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_combining_mark() {
        // U+0301 is COMBINING ACUTE ACCENT.
        assert!(is_combining_mark('\u{0301}'));
        // U+0041 is 'A', not a combining mark.
        assert!(!is_combining_mark('A'));
        // U+0300 is COMBINING GRAVE ACCENT.
        assert!(is_combining_mark('\u{0300}'));
    }

    #[test]
    fn test_is_punctuation() {
        assert!(is_punctuation('!'));
        assert!(is_punctuation(','));
        assert!(is_punctuation('.'));
        assert!(is_punctuation('?'));
        assert!(is_punctuation(';'));
        assert!(!is_punctuation('a'));
        assert!(!is_punctuation('5'));
        assert!(!is_punctuation(' '));
    }

    #[test]
    fn test_normalize() {
        let tok = make_tokenizer();
        assert_eq!(tok.normalize("HELLO"), "hello");
        assert_eq!(tok.normalize("caf\u{00E9}"), "cafe");
        assert_eq!(tok.normalize("na\u{00EF}ve"), "naive");
    }

    // -----------------------------------------------------------------------
    // Sequence length limiting
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_sequence_length() {
        // Build a vocab with single-letter tokens.
        let mut tokens: Vec<String> = (0..101).map(|_| "[PAD]".to_string()).collect();
        tokens[0] = "[PAD]".to_string();
        tokens[100] = "[UNK]".to_string();
        tokens.push("[CLS]".to_string()); // 101
        tokens.push("[SEP]".to_string()); // 102
        for c in b'a'..=b'z' {
            tokens.push(String::from(c as char));
        }
        let tok = WordPieceTokenizer::new(tokens, 101, 102, 100, 0);

        // Create input with > 512 words.
        let input: String = (0..600).map(|_| "a").collect::<Vec<_>>().join(" ");
        let ids = tok.encode(&input, true);

        assert!(ids.len() <= MAX_SEQ_LEN);
        assert_eq!(ids[0], 101); // CLS
        assert_eq!(*ids.last().unwrap(), 102); // SEP
    }

    // ====================================================================
    // NEW TESTS: WordPiece edge cases and extended coverage
    // ====================================================================

    #[test]
    fn test_encode_multiple_unknowns() {
        // Each character of "xyz" has no vocab entry, producing 3 UNKs.
        let tok = make_tokenizer();
        let ids = tok.encode("xyz", false);
        assert_eq!(ids, vec![100, 100, 100]);
    }

    #[test]
    fn test_decode_with_unk() {
        let tok = make_tokenizer();
        // Decode a sequence containing UNK (100)
        let text = tok.decode(&[101, 100, 102]);
        assert_eq!(text, "[UNK]");
    }

    #[test]
    fn test_decode_continuation_at_start() {
        // If a continuation token appears first (no preceding word), it should
        // still produce output rather than crash.
        let tok = make_tokenizer();
        let text = tok.decode(&[105]); // "##ing" with no preceding word
        assert_eq!(text, "ing");
    }

    #[test]
    fn test_encode_mixed_known_unknown() {
        let tok = make_tokenizer();
        // "hello xyz world" -> hello=103, x=UNK, y=UNK, z=UNK, world=104
        let ids = tok.encode("hello xyz world", true);
        assert_eq!(ids[0], 101); // CLS
        assert_eq!(ids[1], 103); // hello
        // xyz -> 3 UNKs
        assert_eq!(ids[2], 100);
        assert_eq!(ids[3], 100);
        assert_eq!(ids[4], 100);
        assert_eq!(ids[5], 104); // world
        assert_eq!(ids[6], 102); // SEP
    }

    #[test]
    fn test_normalize_all_caps() {
        let tok = make_tokenizer();
        assert_eq!(tok.normalize("TESTING"), "testing");
    }

    #[test]
    fn test_normalize_mixed_case() {
        let tok = make_tokenizer();
        assert_eq!(tok.normalize("TeSt"), "test");
    }

    #[test]
    fn test_is_punctuation_unicode() {
        // CJK fullwidth punctuation
        assert!(is_punctuation('\u{FF01}')); // fullwidth exclamation
        assert!(is_punctuation('\u{3001}')); // CJK comma
        // Regular letter should not be punctuation
        assert!(!is_punctuation('\u{4E00}')); // CJK character "one"
    }

    #[test]
    fn test_is_combining_mark_supplementary_plane() {
        // U+E0100 is a variation selector (supplementary plane combining mark)
        assert!(is_combining_mark('\u{E0100}'));
        // U+0041 ('A') is definitely not
        assert!(!is_combining_mark('A'));
    }

    #[test]
    fn test_wordpiece_multiple_continuations() {
        // "tests" -> "test" + "##s"
        let tok = make_tokenizer();
        let ids = tok.encode("tests", true);
        assert_eq!(ids, vec![101, 106, 109, 102]);
    }

    #[test]
    fn test_basic_tokenize_all_punctuation() {
        let tok = make_tokenizer();
        let tokens = tok.basic_tokenize("!@#$");
        // Each punctuation character should be its own token.
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], "!");
        assert_eq!(tokens[1], "@");
        assert_eq!(tokens[2], "#");
        assert_eq!(tokens[3], "$");
    }

    #[test]
    fn test_basic_tokenize_mixed() {
        let tok = make_tokenizer();
        let tokens = tok.basic_tokenize("hello, world!");
        assert_eq!(tokens, vec!["hello", ",", "world", "!"]);
    }

    #[test]
    fn test_from_vocab_text_missing_special_tokens() {
        // A vocab without standard BERT special tokens should use fallback IDs.
        let vocab_text = "a\nb\nc\nd\ne";
        let tok = WordPieceTokenizer::from_vocab_text(vocab_text);
        // Fallback IDs since [CLS]/[SEP]/[UNK]/[PAD] not found
        assert_eq!(tok.vocab_size(), 5);
    }

    #[test]
    fn test_encode_tabs_and_newlines() {
        // Whitespace variants should be handled like spaces.
        let tok = make_tokenizer();
        let ids_space = tok.encode("hello world", true);
        let ids_tab = tok.encode("hello\tworld", true);
        let ids_newline = tok.encode("hello\nworld", true);
        // All should produce the same tokens.
        assert_eq!(ids_space, ids_tab);
        assert_eq!(ids_space, ids_newline);
    }

    #[test]
    fn test_encode_only_whitespace() {
        let tok = make_tokenizer();
        let ids = tok.encode("   \t\n  ", true);
        // Nothing but whitespace -> [CLS] [SEP]
        assert_eq!(ids, vec![101, 102]);
    }

    #[test]
    fn test_decode_skips_pad() {
        let tok = make_tokenizer();
        // Decode a padded sequence: [CLS], hello, [PAD], [PAD], [SEP]
        let text = tok.decode(&[101, 103, 0, 0, 102]);
        assert_eq!(text, "hello");
    }

    #[test]
    fn test_decode_empty() {
        let tok = make_tokenizer();
        let text = tok.decode(&[]);
        assert_eq!(text, "");
    }

    #[test]
    fn test_encode_long_word_with_subwords() {
        // "runnings" -> "run" + "##ning" + "##s"
        let tok = make_tokenizer();
        let ids = tok.encode("runnings", true);
        assert_eq!(ids, vec![101, 110, 111, 109, 102]);
    }

    #[test]
    fn test_unicode_general_category_p() {
        // Verify some key ranges.
        assert!(unicode_general_category_p(0x00BF)); // inverted question mark
        assert!(unicode_general_category_p(0x2014)); // em dash
        assert!(!unicode_general_category_p(0x0041)); // 'A'
    }
}
