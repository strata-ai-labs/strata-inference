//! BPE (Byte-Pair Encoding) tokenizer for SentencePiece and GPT-style models.
//!
//! Supports two merge resolution strategies:
//! - **Score-based (SentencePiece):** Merges are ranked by token score/log-probability
//!   from the GGUF vocabulary. Used by Gemma, LLaMA, etc.
//! - **Rank-based (GPT/BPE):** Merges are ranked by explicit merge rules provided
//!   as `"token_a token_b"` strings. Used by GPT-2, GPT-NeoX, etc.
//!
//! The tokenizer handles Unicode text via a hand-written pre-tokenizer that splits
//! on whitespace boundaries and common patterns. SentencePiece models use the
//! `\u{2581}` character (LOWER ONE EIGHTH BLOCK) as a word-boundary marker.

use std::collections::HashMap;

use tracing::debug;

use super::Tokenizer;

/// The SentencePiece word-boundary marker character.
const SPIECE_UNDERLINE: char = '\u{2581}';

/// BPE tokenizer supporting both SentencePiece (score-based) and standard
/// (rank-based) merge strategies.
///
/// # Construction
///
/// Use [`BpeTokenizer::new`] to create a tokenizer from vocabulary data extracted
/// from a GGUF file. The `merges` parameter controls the merge strategy:
/// - If `merges` is non-empty, rank-based merging is used (GPT-style BPE).
/// - If `merges` is empty, score-based merging is used (SentencePiece BPE),
///   where `scores` determines merge priority (higher score = higher priority).
pub struct BpeTokenizer {
    /// Token string -> token ID.
    token_to_id: HashMap<String, u32>,
    /// Token ID -> token string.
    id_to_token: Vec<String>,
    /// Token scores (log-probabilities) from SentencePiece models.
    scores: Vec<f32>,
    /// Merge rank lookup: (left_token, right_token) -> rank.
    /// Only populated when explicit merge rules are provided.
    merge_ranks: HashMap<(String, String), u32>,
    /// Whether this uses score-based (SentencePiece) or rank-based (GPT) merges.
    use_scores: bool,
    /// BOS token ID.
    bos_id: Option<u32>,
    /// EOS token ID.
    eos_id: Option<u32>,
    /// PAD token ID.
    pad_id: Option<u32>,
    /// Whether to automatically prepend BOS when encoding.
    add_bos: bool,
    /// Whether to automatically append EOS when encoding.
    add_eos: bool,
}

impl BpeTokenizer {
    /// Create a new BPE tokenizer from vocabulary data.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Vocabulary tokens, indexed by token ID.
    /// * `scores` - Token scores/log-probabilities (one per token). Used for
    ///   SentencePiece-style score-based merging when `merges` is empty.
    /// * `_token_types` - Token type flags (reserved for future use).
    /// * `merges` - Explicit merge rules in `"left right"` format. If non-empty,
    ///   rank-based merging is used. If empty, score-based merging is used.
    /// * `bos_id` - Beginning-of-sequence token ID.
    /// * `eos_id` - End-of-sequence token ID.
    /// * `pad_id` - Padding token ID.
    /// * `add_bos` - Whether to prepend BOS token during encoding.
    /// * `add_eos` - Whether to append EOS token during encoding.
    pub fn new(
        tokens: Vec<String>,
        scores: Vec<f32>,
        _token_types: Vec<u32>,
        merges: Vec<String>,
        bos_id: Option<u32>,
        eos_id: Option<u32>,
        pad_id: Option<u32>,
        add_bos: bool,
        add_eos: bool,
    ) -> Self {
        let use_scores = merges.is_empty();

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as u32);
        }

        let mut merge_ranks = HashMap::new();
        if !use_scores {
            for (rank, line) in merges.iter().enumerate() {
                if let Some((left, right)) = line.split_once(' ') {
                    merge_ranks.insert((left.to_string(), right.to_string()), rank as u32);
                }
            }
        }

        debug!(
            vocab_size = tokens.len(),
            merge_count = merge_ranks.len(),
            use_scores,
            "BPE tokenizer initialized"
        );

        Self {
            token_to_id,
            id_to_token: tokens,
            scores,
            merge_ranks,
            use_scores,
            bos_id,
            eos_id,
            pad_id,
            add_bos,
            add_eos,
        }
    }

    /// Encode a single word (pre-tokenized piece) into token IDs.
    ///
    /// This performs the core BPE algorithm:
    /// 1. Split the word into individual UTF-8 characters as initial symbols.
    /// 2. Build a linked-list of symbols and seed pairs into a priority queue.
    /// 3. Repeatedly merge the best pair until no more merges are available.
    /// 4. Map resulting symbols to token IDs (with byte-fallback for unknowns).
    fn encode_word(&self, word: &str) -> Vec<u32> {
        if word.is_empty() {
            return Vec::new();
        }

        // Check if the entire word is a known token (common for single characters
        // and special tokens).
        if let Some(&id) = self.token_to_id.get(word) {
            return vec![id];
        }

        // Split word into UTF-8 characters as initial symbols.
        // Each symbol is stored as a (text, prev, next) triple in a linked list.
        let chars: Vec<String> = word
            .chars()
            .map(|c| {
                let mut buf = [0u8; 4];
                c.encode_utf8(&mut buf);
                c.to_string()
            })
            .collect();

        if chars.is_empty() {
            return Vec::new();
        }

        // Symbol storage: (text, prev_index, next_index).
        // prev/next of -1 means "no neighbor".
        let mut symbols: Vec<Symbol> = Vec::with_capacity(chars.len());
        for (i, ch) in chars.iter().enumerate() {
            symbols.push(Symbol {
                text: ch.clone(),
                prev: if i == 0 { -1 } else { i as i32 - 1 },
                next: if i == chars.len() - 1 {
                    -1
                } else {
                    i as i32 + 1
                },
                merged: false,
            });
        }

        // Seed all adjacent pairs into the work queue.
        let mut work_queue: Vec<Bigram> = Vec::new();
        for i in 1..symbols.len() {
            if let Some(bigram) = self.make_bigram(&symbols, i - 1, i) {
                work_queue.push(bigram);
            }
        }

        // Sort so we can pop the best (lowest rank / highest score) from the end.
        self.sort_queue(&mut work_queue);

        // Main BPE merge loop.
        while let Some(bigram) = work_queue.pop() {
            let left_idx = bigram.left as usize;
            let right_idx = bigram.right as usize;

            // Skip if either symbol was already merged away.
            if symbols[left_idx].merged || symbols[right_idx].merged {
                continue;
            }

            // Validate the bigram is still current (symbols haven't changed).
            let current_text =
                format!("{}{}", symbols[left_idx].text, symbols[right_idx].text);
            if current_text != bigram.text {
                continue;
            }

            // Merge: absorb right into left.
            symbols[left_idx].text = current_text;
            symbols[right_idx].merged = true;

            // Update linked list.
            symbols[left_idx].next = symbols[right_idx].next;
            if symbols[right_idx].next >= 0 {
                let next_of_right = symbols[right_idx].next as usize;
                symbols[next_of_right].prev = left_idx as i32;
            }

            // Add new bigrams for the merged symbol's neighbors.
            if symbols[left_idx].prev >= 0 {
                if let Some(bg) =
                    self.make_bigram(&symbols, symbols[left_idx].prev as usize, left_idx)
                {
                    // Insert in sorted position.
                    let pos = work_queue
                        .binary_search_by(|probe| self.compare_bigrams(probe, &bg))
                        .unwrap_or_else(|pos| pos);
                    work_queue.insert(pos, bg);
                }
            }
            if symbols[left_idx].next >= 0 {
                if let Some(bg) =
                    self.make_bigram(&symbols, left_idx, symbols[left_idx].next as usize)
                {
                    let pos = work_queue
                        .binary_search_by(|probe| self.compare_bigrams(probe, &bg))
                        .unwrap_or_else(|pos| pos);
                    work_queue.insert(pos, bg);
                }
            }
        }

        // Collect final symbols and map to token IDs.
        let mut output = Vec::new();
        let mut idx = 0i32;
        // Find the first non-merged symbol.
        while idx >= 0 && idx < symbols.len() as i32 {
            if !symbols[idx as usize].merged {
                break;
            }
            idx += 1;
        }
        // Walk the linked list.
        while idx >= 0 && (idx as usize) < symbols.len() {
            let sym = &symbols[idx as usize];
            if !sym.merged {
                if let Some(&id) = self.token_to_id.get(&sym.text) {
                    output.push(id);
                } else {
                    // Byte fallback: map each byte to its byte token <0xNN>.
                    self.byte_fallback(&sym.text, &mut output);
                }
            }
            idx = sym.next;
        }

        output
    }

    /// Create a bigram for the pair at (left, right) if a valid merge exists.
    fn make_bigram(&self, symbols: &[Symbol], left: usize, right: usize) -> Option<Bigram> {
        let left_text = &symbols[left].text;
        let right_text = &symbols[right].text;
        let merged_text = format!("{}{}", left_text, right_text);

        if self.use_scores {
            // SentencePiece: look up the merged token's score.
            if let Some(&id) = self.token_to_id.get(&merged_text) {
                let score = self.scores.get(id as usize).copied().unwrap_or(f32::NEG_INFINITY);
                Some(Bigram {
                    left: left as i32,
                    right: right as i32,
                    text: merged_text,
                    rank_or_neg_score: OrderedFloat(-(score as f64)), // negate: higher score = lower rank
                })
            } else {
                None
            }
        } else {
            // Rank-based: look up the merge rank.
            if let Some(&rank) = self
                .merge_ranks
                .get(&(left_text.clone(), right_text.clone()))
            {
                Some(Bigram {
                    left: left as i32,
                    right: right as i32,
                    text: merged_text,
                    rank_or_neg_score: OrderedFloat(rank as f64),
                })
            } else {
                None
            }
        }
    }

    /// Compare two bigrams for sorting. Lower rank/higher score comes last
    /// (so we can pop from the end of a sorted Vec).
    fn compare_bigrams(
        &self,
        a: &Bigram,
        b: &Bigram,
    ) -> std::cmp::Ordering {
        // We want the "best" merge at the end of the vec (popped last).
        // "Best" means lowest rank_or_neg_score value.
        // So we sort in reverse: higher values first, lower values last.
        b.rank_or_neg_score
            .partial_cmp(&a.rank_or_neg_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.left.cmp(&a.left))
    }

    /// Sort the work queue so the best merge is at the end.
    fn sort_queue(&self, queue: &mut Vec<Bigram>) {
        queue.sort_by(|a, b| self.compare_bigrams(a, b));
    }

    /// Encode unknown bytes using byte-fallback tokens `<0x00>` .. `<0xFF>`.
    fn byte_fallback(&self, text: &str, output: &mut Vec<u32>) {
        for byte in text.bytes() {
            let byte_token = format!("<0x{:02X}>", byte);
            if let Some(&id) = self.token_to_id.get(&byte_token) {
                output.push(id);
            }
            // If byte token not in vocab, silently skip (shouldn't happen in
            // well-formed models).
        }
    }

    /// Pre-tokenize text into pieces suitable for BPE encoding.
    ///
    /// For SentencePiece models (use_scores=true):
    /// - Replace spaces with the SentencePiece underline character `\u{2581}`.
    /// - Do NOT split the text further (SentencePiece operates on the full string).
    ///
    /// For GPT-style models (use_scores=false):
    /// - Split on whitespace boundaries, keeping whitespace attached to the
    ///   following token (GPT-2 style).
    /// - Handle English contractions as separate tokens.
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        if self.use_scores {
            // SentencePiece: replace spaces with the underline marker.
            // The underline is prepended to the text and replaces all spaces.
            let processed = format!(
                "{}{}",
                SPIECE_UNDERLINE,
                text.replace(' ', &SPIECE_UNDERLINE.to_string())
            );
            vec![processed]
        } else {
            // GPT-style: split into words with whitespace handling.
            gpt_pre_tokenize(text)
        }
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
        let mut output = Vec::new();

        // Prepend BOS if configured and requested.
        if add_special_tokens && self.add_bos {
            if let Some(bos) = self.bos_id {
                output.push(bos);
            }
        }

        // Pre-tokenize and encode each piece.
        let pieces = self.pre_tokenize(text);
        for piece in &pieces {
            let ids = self.encode_word(piece);
            output.extend_from_slice(&ids);
        }

        // Append EOS if configured and requested.
        if add_special_tokens && self.add_eos {
            if let Some(eos) = self.eos_id {
                output.push(eos);
            }
        }

        output
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut text = String::new();
        for &id in ids {
            if let Some(token) = self.id_to_token.get(id as usize) {
                // Skip special tokens during decoding.
                if Some(id) == self.bos_id
                    || Some(id) == self.eos_id
                    || Some(id) == self.pad_id
                {
                    continue;
                }
                // Handle byte tokens <0xNN>.
                if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
                    if let Ok(byte_val) = u8::from_str_radix(&token[3..5], 16) {
                        text.push(byte_val as char);
                        continue;
                    }
                }
                text.push_str(token);
            }
        }
        // Replace SentencePiece underline with space.
        let result = text.replace(SPIECE_UNDERLINE, " ");
        // Strip leading space that comes from the initial underline.
        if result.starts_with(' ') {
            result[1..].to_string()
        } else {
            result
        }
    }

    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos_id
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_id
    }

    fn pad_token_id(&self) -> Option<u32> {
        self.pad_id
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// A symbol in the BPE linked list.
#[derive(Clone, Debug)]
struct Symbol {
    text: String,
    prev: i32,
    next: i32,
    merged: bool,
}

/// A candidate bigram merge.
#[derive(Clone, Debug)]
struct Bigram {
    left: i32,
    right: i32,
    text: String,
    /// For score-based: negated score (lower = better).
    /// For rank-based: merge rank (lower = better).
    rank_or_neg_score: OrderedFloat,
}

/// Wrapper for f64 that provides total ordering (NaN sorts last).
#[derive(Clone, Copy, Debug)]
struct OrderedFloat(f64);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

// ---------------------------------------------------------------------------
// Pre-tokenization (no regex dependency)
// ---------------------------------------------------------------------------

/// GPT-2-style pre-tokenization.
///
/// Splits text into tokens by:
/// 1. Handling English contractions ('s, 't, 're, 've, 'm, 'll, 'd).
/// 2. Splitting on whitespace boundaries, keeping leading space with the word.
/// 3. Separating letters from digits from punctuation.
fn gpt_pre_tokenize(text: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];

        if ch.is_whitespace() {
            // Accumulate whitespace and attach it to the next word token.
            let mut ws = String::new();
            while i < len && chars[i].is_whitespace() {
                ws.push(chars[i]);
                i += 1;
            }
            if i < len {
                // There's a word after the whitespace: start a piece with the
                // accumulated space prefix.
                let mut piece = ws;
                accumulate_word(&chars, &mut i, &mut piece, &mut pieces);
                if !piece.is_empty() {
                    pieces.push(piece);
                }
            } else {
                // Trailing whitespace.
                if !ws.is_empty() {
                    pieces.push(ws);
                }
            }
        } else {
            // No leading whitespace.
            let mut piece = String::new();
            accumulate_word(&chars, &mut i, &mut piece, &mut pieces);
            if !piece.is_empty() {
                pieces.push(piece);
            }
        }
    }

    pieces
}

/// Accumulate characters of the same category into `piece`, handling
/// contractions as split points. `i` is advanced past consumed characters.
fn accumulate_word(
    chars: &[char],
    i: &mut usize,
    piece: &mut String,
    pieces: &mut Vec<String>,
) {
    let len = chars.len();
    if *i >= len {
        return;
    }

    let cat = char_category(chars[*i]);

    loop {
        if *i >= len || chars[*i].is_whitespace() {
            break;
        }

        // Check for contractions: apostrophe followed by a contraction suffix.
        // This must be checked before the category check because `'` is
        // category Other while we may be inside a Letter run.
        if chars[*i] == '\'' && cat == CharCategory::Letter {
            if let Some(contraction) = try_contraction(chars, *i) {
                if !piece.is_empty() {
                    pieces.push(std::mem::take(piece));
                }
                pieces.push(contraction.0);
                *i += contraction.1;
                // After a contraction, continue accumulating if the next char
                // is still a letter (e.g., "don't" -> "don" + "'t").
                continue;
            }
            // Apostrophe but not a recognized contraction: break the word.
            break;
        }

        if char_category(chars[*i]) != cat {
            break;
        }

        piece.push(chars[*i]);
        *i += 1;
    }
}

/// Character categories for pre-tokenization grouping.
#[derive(PartialEq, Eq, Clone, Copy)]
enum CharCategory {
    Letter,
    Digit,
    Other,
}

fn char_category(ch: char) -> CharCategory {
    if ch.is_alphabetic() {
        CharCategory::Letter
    } else if ch.is_ascii_digit() {
        CharCategory::Digit
    } else {
        CharCategory::Other
    }
}

/// Try to match an English contraction starting at position `i` (which should
/// be an apostrophe). Returns `Some((contraction_string, chars_consumed))` on match.
fn try_contraction(chars: &[char], i: usize) -> Option<(String, usize)> {
    if i >= chars.len() || chars[i] != '\'' {
        return None;
    }

    let remaining = chars.len() - i;
    if remaining < 2 {
        return None;
    }

    let next = chars[i + 1].to_ascii_lowercase();

    // Single-char contractions: 's, 't, 'm, 'd
    if matches!(next, 's' | 't' | 'm' | 'd') {
        let contraction: String = chars[i..i + 2].iter().collect();
        return Some((contraction, 2));
    }

    if remaining >= 3 {
        let next2 = chars[i + 2].to_ascii_lowercase();
        // Two-char contractions: 're, 've, 'll
        match (next, next2) {
            ('r', 'e') | ('v', 'e') | ('l', 'l') => {
                let contraction: String = chars[i..i + 3].iter().collect();
                return Some((contraction, 3));
            }
            _ => {}
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: Build a minimal SentencePiece-style BPE tokenizer.
    // -----------------------------------------------------------------------

    /// Creates a small SentencePiece BPE tokenizer for testing.
    ///
    /// The vocab includes all intermediate merge steps so that BPE can
    /// progressively merge character-level symbols up to full words.
    /// Each pairwise merge must produce a token that exists in the vocab.
    ///
    /// Merge chains:
    ///   h + e -> he -> he + l -> hel -> hel + l -> hell -> hell + o -> hello
    ///   w + o -> wo -> wo + r -> wor -> wor + l -> worl -> worl + d -> world
    ///   _ + h -> _h -> _h + e -> _he -> ... -> _hello
    ///   _ + w -> _w -> _w + o -> _wo -> ... -> _world
    fn make_spm_tokenizer(add_bos: bool, add_eos: bool) -> BpeTokenizer {
        let tokens = vec![
            "<pad>".to_string(),              // 0
            "<bos>".to_string(),              // 1
            "<eos>".to_string(),              // 2
            "<unk>".to_string(),              // 3
            SPIECE_UNDERLINE.to_string(),     // 4
            "h".to_string(),                  // 5
            "e".to_string(),                  // 6
            "l".to_string(),                  // 7
            "o".to_string(),                  // 8
            "w".to_string(),                  // 9
            "r".to_string(),                  // 10
            "d".to_string(),                  // 11
            "he".to_string(),                 // 12
            "ll".to_string(),                 // 13
            "lo".to_string(),                 // 14
            "wo".to_string(),                 // 15
            "or".to_string(),                 // 16
            "rl".to_string(),                 // 17
            "ld".to_string(),                 // 18
            "hel".to_string(),                // 19
            "hell".to_string(),               // 20
            "hello".to_string(),              // 21
            "wor".to_string(),                // 22
            "worl".to_string(),               // 23
            "world".to_string(),              // 24
            format!("{}h", SPIECE_UNDERLINE), // 25
            format!("{}w", SPIECE_UNDERLINE), // 26
            format!("{}he", SPIECE_UNDERLINE),// 27
            format!("{}wo", SPIECE_UNDERLINE),// 28
            format!("{}hel", SPIECE_UNDERLINE),  // 29
            format!("{}wor", SPIECE_UNDERLINE),  // 30
            format!("{}hell", SPIECE_UNDERLINE), // 31
            format!("{}worl", SPIECE_UNDERLINE), // 32
            format!("{}hello", SPIECE_UNDERLINE),// 33
            format!("{}world", SPIECE_UNDERLINE),// 34
        ];
        let scores = vec![
            0.0,    // 0:  <pad>
            0.0,    // 1:  <bos>
            0.0,    // 2:  <eos>
            0.0,    // 3:  <unk>
            -5.0,   // 4:  _ (single char, low priority)
            -5.0,   // 5:  h
            -5.0,   // 6:  e
            -5.0,   // 7:  l
            -5.0,   // 8:  o
            -5.0,   // 9:  w
            -5.0,   // 10: r
            -5.0,   // 11: d
            -4.0,   // 12: he
            -4.0,   // 13: ll
            -4.0,   // 14: lo
            -4.0,   // 15: wo
            -4.0,   // 16: or
            -4.0,   // 17: rl
            -4.0,   // 18: ld
            -3.0,   // 19: hel
            -2.0,   // 20: hell
            -1.0,   // 21: hello
            -3.0,   // 22: wor
            -2.0,   // 23: worl
            -1.0,   // 24: world
            -3.5,   // 25: _h
            -3.5,   // 26: _w
            -3.0,   // 27: _he
            -3.0,   // 28: _wo
            -2.5,   // 29: _hel
            -2.5,   // 30: _wor
            -1.5,   // 31: _hell
            -1.5,   // 32: _worl
            -0.5,   // 33: _hello
            -0.5,   // 34: _world
        ];
        let token_types = vec![0u32; tokens.len()];

        BpeTokenizer::new(
            tokens,
            scores,
            token_types,
            vec![], // empty merges → score-based (SentencePiece)
            Some(1),
            Some(2),
            Some(0),
            add_bos,
            add_eos,
        )
    }

    /// Creates a small rank-based BPE tokenizer for testing.
    ///
    /// Vocab:
    ///   0: <pad>
    ///   1: <bos>
    ///   2: <eos>
    ///   3: h, 4: e, 5: l, 6: o, 7: w, 8: r, 9: d, 10: " " (space)
    ///  11: he, 12: ll, 13: lo, 14: hel, 15: hell, 16: hello
    ///
    /// Merges (in order of priority):
    ///   h e  → he   (rank 0)
    ///   l l  → ll   (rank 1)
    ///   l o  → lo   (rank 2)
    ///   he l → hel  (rank 3)
    ///   hel l → hell (rank 4) -- note: "hel" is a single token for this merge
    fn make_rank_tokenizer() -> BpeTokenizer {
        let tokens = vec![
            "<pad>".to_string(),  // 0
            "<bos>".to_string(),  // 1
            "<eos>".to_string(),  // 2
            "h".to_string(),      // 3
            "e".to_string(),      // 4
            "l".to_string(),      // 5
            "o".to_string(),      // 6
            "w".to_string(),      // 7
            "r".to_string(),      // 8
            "d".to_string(),      // 9
            " ".to_string(),      // 10
            "he".to_string(),     // 11
            "ll".to_string(),     // 12
            "lo".to_string(),     // 13
            "hel".to_string(),    // 14
            "hell".to_string(),   // 15
            "hello".to_string(),  // 16
        ];
        let scores = vec![0.0; tokens.len()];
        let token_types = vec![0u32; tokens.len()];

        let merges = vec![
            "h e".to_string(),     // rank 0: h+e → he
            "l l".to_string(),     // rank 1: l+l → ll
            "l o".to_string(),     // rank 2: l+o → lo
            "he l".to_string(),    // rank 3: he+l → hel
            "hel l".to_string(),   // rank 4: hel+l → hell (note: uses merged token)
            "hell o".to_string(),  // rank 5: hell+o → hello
        ];

        BpeTokenizer::new(
            tokens,
            scores,
            token_types,
            merges,
            Some(1),
            Some(2),
            Some(0),
            false,
            false,
        )
    }

    // -----------------------------------------------------------------------
    // SentencePiece BPE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_spm_encode_hello_world() {
        let tok = make_spm_tokenizer(false, false);
        let ids = tok.encode("hello world", false);
        // Pre-tokenize: "\u{2581}hello\u{2581}world"
        // BPE with scores should merge to highest-score tokens.
        // _hello (score=-0.5, id=33) and _world (score=-0.5, id=34) are the best merges.
        assert_eq!(ids, vec![33, 34]); // _hello=33, _world=34
    }

    #[test]
    fn test_spm_encode_with_bos_eos() {
        let tok = make_spm_tokenizer(true, true);
        let ids = tok.encode("hello world", true);
        assert_eq!(ids[0], 1);  // BOS
        assert_eq!(*ids.last().unwrap(), 2);  // EOS
        // Middle tokens should be _hello and _world.
        assert_eq!(ids[1], 33); // _hello
        assert_eq!(ids[2], 34); // _world
    }

    #[test]
    fn test_spm_encode_no_special_tokens_flag() {
        let tok = make_spm_tokenizer(true, true);
        let ids = tok.encode("hello", false);
        // Even though add_bos/add_eos are true, the flag says don't add them.
        assert!(!ids.contains(&1)); // no BOS
        assert!(!ids.contains(&2)); // no EOS
    }

    #[test]
    fn test_spm_encode_single_char() {
        let tok = make_spm_tokenizer(false, false);
        let ids = tok.encode("h", false);
        // Pre-tokenize: "\u{2581}h"
        // _h is in vocab (id=25), so it should be a single token.
        assert_eq!(ids, vec![25]);
    }

    #[test]
    fn test_spm_decode_roundtrip() {
        let tok = make_spm_tokenizer(false, false);
        let ids = tok.encode("hello world", false);
        let text = tok.decode(&ids);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_spm_decode_with_special_tokens() {
        let tok = make_spm_tokenizer(true, true);
        let ids = tok.encode("hello", true);
        let text = tok.decode(&ids);
        // Decode should skip BOS/EOS.
        assert_eq!(text, "hello");
    }

    #[test]
    fn test_spm_empty_input() {
        let tok = make_spm_tokenizer(false, false);
        let ids = tok.encode("", false);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_spm_empty_with_special() {
        let tok = make_spm_tokenizer(true, true);
        let ids = tok.encode("", true);
        // Should have BOS + EOS only.
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn test_spm_vocab_size() {
        let tok = make_spm_tokenizer(false, false);
        assert_eq!(tok.vocab_size(), 35);
    }

    #[test]
    fn test_spm_special_token_ids() {
        let tok = make_spm_tokenizer(false, false);
        assert_eq!(tok.bos_token_id(), Some(1));
        assert_eq!(tok.eos_token_id(), Some(2));
        assert_eq!(tok.pad_token_id(), Some(0));
    }

    // -----------------------------------------------------------------------
    // Rank-based BPE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rank_encode_hello() {
        let tok = make_rank_tokenizer();
        // Input "hello" -> pre-tokenize -> ["hello"]
        // The encode_word function first checks if the whole word is in the vocab.
        // "hello" is token 16, so it matches immediately without BPE merging.
        let ids = tok.encode("hello", false);
        assert_eq!(ids, vec![16]); // hello=16 (whole word match)
    }

    #[test]
    fn test_rank_encode_hell() {
        let tok = make_rank_tokenizer();
        // Input "hell" -> pre-tokenize -> ["hell"]
        // "hell" is token 15, so it matches as a whole word.
        let ids = tok.encode("hell", false);
        assert_eq!(ids, vec![15]); // hell=15
    }

    #[test]
    fn test_rank_encode_he() {
        let tok = make_rank_tokenizer();
        // Input "he" -> pre-tokenize -> ["he"]
        // "he" is token 11, so it matches as a whole word.
        let ids = tok.encode("he", false);
        assert_eq!(ids, vec![11]); // he=11
    }

    #[test]
    fn test_rank_encode_helloworld() {
        let tok = make_rank_tokenizer();
        // Input "helloworld" -> not in vocab -> BPE splits into chars: h e l l o w o r l d
        // Merges applied:
        //   rank 0: h+e -> he    -> [he, l, l, o, w, o, r, l, d]
        //   rank 1: l+l -> ll    -> [he, ll, o, w, o, r, l, d]
        //   After these, remaining pairs may or may not have merges.
        //   (he, ll) -> "hell" if merge exists: rank 4 "hel l"... but "he"+"ll" != "hel"+"l"
        //   The merge ranks are for specific left+right pairs: "hel l" not "he ll".
        //   So no more merges. Result: [he=11, ll=12, o=6, w=7, o=6, r=8, l=5, d=9]
        let ids = tok.encode("helloworld", false);
        assert_eq!(ids, vec![11, 12, 6, 7, 6, 8, 5, 9]);
    }

    #[test]
    fn test_rank_encode_single_char() {
        let tok = make_rank_tokenizer();
        let ids = tok.encode("h", false);
        assert_eq!(ids, vec![3]); // h=3
    }

    #[test]
    fn test_rank_decode() {
        let tok = make_rank_tokenizer();
        let text = tok.decode(&[11, 12, 6]); // he, ll, o
        assert_eq!(text, "hello");
    }

    #[test]
    fn test_rank_empty() {
        let tok = make_rank_tokenizer();
        let ids = tok.encode("", false);
        assert!(ids.is_empty());
    }

    // -----------------------------------------------------------------------
    // Pre-tokenization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_spm_pre_tokenize() {
        let tok = make_spm_tokenizer(false, false);
        let pieces = tok.pre_tokenize("hello world");
        assert_eq!(pieces.len(), 1);
        assert_eq!(
            pieces[0],
            format!("{}hello{}world", SPIECE_UNDERLINE, SPIECE_UNDERLINE)
        );
    }

    #[test]
    fn test_gpt_pre_tokenize_basic() {
        let pieces = gpt_pre_tokenize("hello world");
        assert_eq!(pieces, vec!["hello", " world"]);
    }

    #[test]
    fn test_gpt_pre_tokenize_multiple_spaces() {
        let pieces = gpt_pre_tokenize("hello  world");
        assert_eq!(pieces, vec!["hello", "  world"]);
    }

    #[test]
    fn test_gpt_pre_tokenize_contractions() {
        let pieces = gpt_pre_tokenize("I'm don't we'll");
        // "I" + "'m" + " don" + "'t" + " we" + "'ll"
        assert_eq!(
            pieces,
            vec!["I", "'m", " don", "'t", " we", "'ll"]
        );
    }

    #[test]
    fn test_gpt_pre_tokenize_digits() {
        let pieces = gpt_pre_tokenize("abc123");
        // Letters and digits are different categories.
        assert_eq!(pieces, vec!["abc", "123"]);
    }

    #[test]
    fn test_gpt_pre_tokenize_punctuation() {
        let pieces = gpt_pre_tokenize("hello,world");
        // "hello" (letter), "," (other), "world" (letter)
        assert_eq!(pieces, vec!["hello", ",", "world"]);
    }

    #[test]
    fn test_gpt_pre_tokenize_empty() {
        let pieces = gpt_pre_tokenize("");
        assert!(pieces.is_empty());
    }

    #[test]
    fn test_gpt_pre_tokenize_only_spaces() {
        let pieces = gpt_pre_tokenize("   ");
        assert_eq!(pieces, vec!["   "]);
    }

    #[test]
    fn test_gpt_pre_tokenize_leading_space() {
        let pieces = gpt_pre_tokenize(" hello");
        assert_eq!(pieces, vec![" hello"]);
    }

    // -----------------------------------------------------------------------
    // Byte fallback tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_byte_fallback() {
        // Create a tokenizer with byte tokens.
        let mut tokens = vec![
            "<pad>".to_string(), // 0
            "<bos>".to_string(), // 1
            "<eos>".to_string(), // 2
        ];
        // Add byte tokens for all 256 byte values.
        for b in 0..=255u8 {
            tokens.push(format!("<0x{:02X}>", b));
        }
        let sp_underline = SPIECE_UNDERLINE.to_string();
        tokens.push(sp_underline.clone()); // add the underline token too

        let scores = vec![0.0; tokens.len()];
        let token_types = vec![0u32; tokens.len()];

        let tok = BpeTokenizer::new(
            tokens,
            scores,
            token_types,
            vec![],
            Some(1),
            Some(2),
            Some(0),
            false,
            false,
        );

        // Encoding a character not in vocab should fall back to byte tokens.
        // The char 'A' = 0x41, so it should map to <0x41> which is token 3+0x41=68.
        let ids = tok.encode("A", false);
        // Pre-tokenize adds underline prefix: "_A"
        // Neither "_A" nor its substrings (except individual chars) are in vocab.
        // So BPE falls through to byte fallback for the underline character and 'A'.
        assert!(!ids.is_empty());
    }

    // -----------------------------------------------------------------------
    // Thread safety
    // -----------------------------------------------------------------------

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BpeTokenizer>();
    }

    // -----------------------------------------------------------------------
    // Trait implementation
    // -----------------------------------------------------------------------

    #[test]
    fn test_tokenizer_trait() {
        let tok = make_spm_tokenizer(false, false);
        let tok: &dyn Tokenizer = &tok;
        assert_eq!(tok.vocab_size(), 35);
        assert_eq!(tok.bos_token_id(), Some(1));
        assert_eq!(tok.eos_token_id(), Some(2));
        assert_eq!(tok.pad_token_id(), Some(0));
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_spm_unicode_chars() {
        // Test with Unicode characters that are not in the vocab.
        // They should be handled via byte fallback or single-char tokens.
        let tok = make_spm_tokenizer(false, false);
        // Even if encoding produces no matches, it shouldn't panic.
        let ids = tok.encode("!", false);
        // '!' is not in our test vocab, so byte fallback will be attempted.
        // Since we don't have byte tokens in the test vocab, this may produce
        // an empty result, but should not panic.
        let _ = ids;
    }

    #[test]
    fn test_encode_word_single_known_token() {
        let tok = make_spm_tokenizer(false, false);
        // "hello" is directly in the vocab as id 21.
        let ids = tok.encode_word("hello");
        assert_eq!(ids, vec![21]);
    }

    #[test]
    fn test_encode_word_unknown_falls_back() {
        let tok = make_spm_tokenizer(false, false);
        // "xyz" is not in vocab and has no byte tokens, so we get nothing.
        let ids = tok.encode_word("xyz");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_spm_repeated_words() {
        let tok = make_spm_tokenizer(false, false);
        let ids = tok.encode("hello hello", false);
        // Should tokenize as: _hello _hello
        assert_eq!(ids, vec![33, 33]); // _hello=33
    }

    #[test]
    fn test_spm_only_space() {
        let tok = make_spm_tokenizer(false, false);
        let ids = tok.encode(" ", false);
        // Pre-tokenize: "_" + "_" = "__"
        // The merged text "__" is not in vocab. Individual "_" chars (id=4) should match.
        // Actually pre_tokenize produces "_ _" which becomes "__" via replace.
        // Let me trace: text=" ", pre_tokenize returns ["__"] (underline + space->underline).
        // BPE on "__": chars ['_', '_'], pair (_,_) → merged "__" not in vocab.
        // So no merge. Individual '_' tokens: id=4 each.
        assert_eq!(ids, vec![4, 4]);
    }

    // ====================================================================
    // NEW TESTS: BPE edge cases and extended coverage
    // ====================================================================

    // -- Unicode edge cases --

    #[test]
    fn test_spm_multibyte_unicode() {
        // Test encoding of a CJK character (3 bytes in UTF-8).
        // Without CJK tokens in the vocab, should go to byte fallback (empty
        // since our test vocab lacks <0xNN> tokens).
        let tok = make_spm_tokenizer(false, false);
        let ids = tok.encode("\u{4e16}", false); // Chinese character "shi" (world)
        // Should not panic, regardless of result
        let _ = ids;
    }

    #[test]
    fn test_spm_emoji_input() {
        // Emoji is 4 bytes in UTF-8. Should not panic.
        let tok = make_spm_tokenizer(false, false);
        let ids = tok.encode("\u{1F600}", false); // grinning face emoji
        let _ = ids;
    }

    #[test]
    fn test_spm_zero_width_chars() {
        // Zero-width joiner and zero-width non-joiner
        let tok = make_spm_tokenizer(false, false);
        let ids = tok.encode("\u{200D}\u{200C}", false);
        let _ = ids;
    }

    // -- Multiple spaces --

    #[test]
    fn test_spm_multiple_spaces() {
        let tok = make_spm_tokenizer(false, false);
        let ids = tok.encode("hello  world", false);
        // Pre-tokenize: "_hello__world" -- two underlines between words.
        // The BPE should handle this and produce tokens.
        assert!(!ids.is_empty());
    }

    // -- Very long input --

    #[test]
    fn test_spm_long_repeated_input() {
        // Encode a long repeated pattern. Should not panic or hang.
        let tok = make_spm_tokenizer(false, false);
        let input = "hello ".repeat(100);
        let ids = tok.encode(&input, false);
        assert!(!ids.is_empty());
    }

    // -- Pre-tokenizer edge cases --

    #[test]
    fn test_spm_pre_tokenize_empty() {
        let tok = make_spm_tokenizer(false, false);
        let pieces = tok.pre_tokenize("");
        assert!(pieces.is_empty());
    }

    #[test]
    fn test_spm_pre_tokenize_only_spaces() {
        let tok = make_spm_tokenizer(false, false);
        let pieces = tok.pre_tokenize("   ");
        // Should be a single piece with underline + underlines
        assert_eq!(pieces.len(), 1);
        let expected = format!(
            "{}{}{}{}",
            SPIECE_UNDERLINE, SPIECE_UNDERLINE, SPIECE_UNDERLINE, SPIECE_UNDERLINE
        );
        assert_eq!(pieces[0], expected);
    }

    // -- GPT pre-tokenizer extended tests --

    #[test]
    fn test_gpt_pre_tokenize_mixed_punctuation_digits() {
        let pieces = gpt_pre_tokenize("a1!b2");
        // "a" (letter), "1" (digit), "!" (other), "b" (letter), "2" (digit)
        assert_eq!(pieces, vec!["a", "1", "!", "b", "2"]);
    }

    #[test]
    fn test_gpt_pre_tokenize_tabs_and_newlines() {
        let pieces = gpt_pre_tokenize("hello\tworld\nfoo");
        // Tab and newline are whitespace; they get attached to following word.
        assert_eq!(pieces, vec!["hello", "\tworld", "\nfoo"]);
    }

    #[test]
    fn test_gpt_pre_tokenize_contraction_its() {
        let pieces = gpt_pre_tokenize("it's");
        assert_eq!(pieces, vec!["it", "'s"]);
    }

    #[test]
    fn test_gpt_pre_tokenize_contraction_theyve() {
        let pieces = gpt_pre_tokenize("they've");
        assert_eq!(pieces, vec!["they", "'ve"]);
    }

    #[test]
    fn test_gpt_pre_tokenize_contraction_shed() {
        let pieces = gpt_pre_tokenize("she'd");
        assert_eq!(pieces, vec!["she", "'d"]);
    }

    #[test]
    fn test_gpt_pre_tokenize_contraction_theyre() {
        let pieces = gpt_pre_tokenize("they're");
        assert_eq!(pieces, vec!["they", "'re"]);
    }

    #[test]
    fn test_gpt_pre_tokenize_single_char() {
        let pieces = gpt_pre_tokenize("a");
        assert_eq!(pieces, vec!["a"]);
    }

    // -- Rank-based BPE edge cases --

    #[test]
    fn test_rank_encode_single_unknown_char() {
        let tok = make_rank_tokenizer();
        // "z" is not in vocab; should fall through to byte fallback (empty since
        // the test vocab has no <0xNN> tokens).
        let ids = tok.encode("z", false);
        // Should be empty since no byte tokens exist in this vocab
        assert!(ids.is_empty());
    }

    #[test]
    fn test_rank_encode_with_special_tokens() {
        // Create a rank tokenizer with BOS/EOS enabled.
        let tokens = vec![
            "<pad>".to_string(),  // 0
            "<bos>".to_string(),  // 1
            "<eos>".to_string(),  // 2
            "h".to_string(),      // 3
            "e".to_string(),      // 4
        ];
        let scores = vec![0.0; tokens.len()];
        let token_types = vec![0u32; tokens.len()];
        let tok = BpeTokenizer::new(
            tokens, scores, token_types, vec![],
            Some(1), Some(2), Some(0),
            true, true,
        );
        let ids = tok.encode("h", true);
        assert_eq!(ids[0], 1); // BOS
        assert_eq!(*ids.last().unwrap(), 2); // EOS
    }

    // -- Decode edge cases --

    #[test]
    fn test_spm_decode_empty() {
        let tok = make_spm_tokenizer(false, false);
        let text = tok.decode(&[]);
        assert_eq!(text, "");
    }

    #[test]
    fn test_spm_decode_only_special_tokens() {
        let tok = make_spm_tokenizer(true, true);
        // Decoding just BOS+EOS should produce empty string.
        let text = tok.decode(&[1, 2]);
        assert_eq!(text, "");
    }

    #[test]
    fn test_spm_decode_out_of_range_id() {
        let tok = make_spm_tokenizer(false, false);
        // ID beyond vocab should be silently skipped (no token string found).
        let text = tok.decode(&[9999]);
        assert_eq!(text, "");
    }

    // -- OrderedFloat correctness --

    #[test]
    fn test_ordered_float_nan_ordering() {
        // NaN should not cause panics in ordering.
        let a = OrderedFloat(f64::NAN);
        let b = OrderedFloat(1.0);
        // Just verify it does not panic
        let _ = a.cmp(&b);
        let _ = b.cmp(&a);
        let _ = a.partial_cmp(&b);
    }

    #[test]
    fn test_ordered_float_equality() {
        let a = OrderedFloat(1.0);
        let b = OrderedFloat(1.0);
        assert_eq!(a, b);
    }

    // -- BPE with no BOS/EOS config --

    #[test]
    fn test_bpe_no_bos_eos_configured() {
        let tokens = vec!["a".to_string(), "b".to_string()];
        let tok = BpeTokenizer::new(
            tokens, vec![0.0; 2], vec![0; 2], vec![],
            None, None, None,
            false, false,
        );
        assert_eq!(tok.bos_token_id(), None);
        assert_eq!(tok.eos_token_id(), None);
        assert_eq!(tok.pad_token_id(), None);
        let ids = tok.encode("a", true);
        // No BOS/EOS to add since they are None
        assert_eq!(ids, vec![0]);
    }
}
