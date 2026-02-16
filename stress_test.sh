#!/usr/bin/env bash
#
# stress_test.sh — Compare strata-tokenize vs llama-tokenize on 100 sentences.
#
# Usage:
#   MODEL=path/to/model.gguf bash stress_test.sh
#
# Environment variables:
#   MODEL        — Path to GGUF model file (required)
#   STRATA_BIN   — Path to strata-tokenize binary (default: ./target/release/strata-tokenize)
#   LLAMA_BIN    — Path to llama-tokenize binary (default: ~/Documents/GitHub/llama.cpp/build/bin/llama-tokenize)
#   VERBOSE      — Set to 1 to show diffs for failures (default: 0)

set -euo pipefail

MODEL="${MODEL:?MODEL env var is required}"
STRATA_BIN="${STRATA_BIN:-./target/release/strata-tokenize}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/Documents/GitHub/llama.cpp/build/bin/llama-tokenize}"
VERBOSE="${VERBOSE:-0}"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model file not found: $MODEL"
    exit 1
fi

if [ ! -x "$STRATA_BIN" ]; then
    echo "ERROR: strata-tokenize not found at $STRATA_BIN"
    exit 1
fi

if [ ! -x "$LLAMA_BIN" ]; then
    echo "ERROR: llama-tokenize not found at $LLAMA_BIN"
    exit 1
fi

MODEL_NAME=$(basename "$MODEL")
echo "=== Stress Test: $MODEL_NAME ==="
echo "  strata-tokenize: $STRATA_BIN"
echo "  llama-tokenize:  $LLAMA_BIN"
echo ""

# 100 test sentences covering various edge cases
sentences=(
    "Hello, world!"
    "The quick brown fox jumps over the lazy dog."
    "I'm don't we'll they've she'd"
    "Testing 1234567890 numbers"
    "Special chars: @#\$%^&*()"
    "Multiple   spaces   between   words"
    "UPPERCASE AND lowercase MiXeD"
    "Tab\there and\tthere"
    "URL: https://example.com/path?query=value"
    "Email: user@example.com"
    "Math: 2 + 2 = 4, 3 * 5 = 15"
    "Quotes: \"hello\" and 'world'"
    "Parentheses: (a) [b] {c}"
    "Punctuation: ; : , . ! ?"
    "Dashes: - -- ---"
    "Ellipsis: ..."
    "Slash: a/b/c"
    "Backslash: a\b\c"
    "Unicode: cafe"
    "Numbers in words: abc123def456"
    "Leading spaces:    hello"
    "Trailing spaces: hello   "
    "Empty-ish:  "
    "Single character: a"
    "Single digit: 5"
    "Single punctuation: !"
    "Repeated: aaaaaaa"
    "Alternating: ababababab"
    "Long word: supercalifragilisticexpialidocious"
    "Hyphenated: well-known, state-of-the-art"
    "Apostrophe edge: it's, don't, I'm, we'll, they're, she'd, you've"
    "Possessive: John's, the dog's, the children's"
    "Abbreviation: U.S.A., e.g., i.e., Dr., Mr."
    "Acronym: NASA, FBI, CIA, HTML, CSS"
    "CamelCase: getElementById, innerHTML"
    "snake_case: my_variable_name"
    "kebab-case: my-css-class"
    "Code: for i in range(10): print(i)"
    "Code: fn main() { println!(\"hello\"); }"
    "JSON: {\"key\": \"value\", \"num\": 42}"
    "XML: <tag attr=\"val\">content</tag>"
    "Path: /usr/local/bin/python3"
    "Emoji description: smiley face, thumbs up"
    "Currency: \$100.00, 50.00 EUR, 1000 JPY"
    "Percentage: 99.9% accuracy"
    "Temperature: 72F, 22C"
    "Date: 2024-01-15, January 15th, 2024"
    "Time: 3:45 PM, 15:45:00"
    "Phone: +1 (555) 123-4567"
    "IP: 192.168.1.1"
    "Version: v1.2.3-beta.4"
    "Hashtag: #AI #MachineLearning"
    "Mention: @username @another_user"
    "Markdown: **bold** *italic* __underline__"
    "HTML entities: &amp; &lt; &gt; &quot;"
    "Escape: \\n \\t \\r \\\\"
    "Mixed: Hello, World! 123 @#\$ abc"
    "Sentence with newline equivalent: first line. second line."
    "Very short: hi"
    "Two chars: ab"
    "Three chars: abc"
    "Four chars: abcd"
    "Five chars: abcde"
    "Comma separated: a, b, c, d, e"
    "Semicolon separated: a; b; c; d; e"
    "Pipe separated: a | b | c | d | e"
    "Period at end."
    "Question mark?"
    "Exclamation mark!"
    "No punctuation at end"
    "Multiple sentences. Here is another. And a third."
    "The rain in Spain stays mainly in the plain."
    "To be or not to be, that is the question."
    "All that glitters is not gold."
    "A journey of a thousand miles begins with a single step."
    "In the beginning was the Word."
    "The only thing we have to fear is fear itself."
    "I think, therefore I am."
    "That which does not kill us makes us stronger."
    "The unexamined life is not worth living."
    "Knowledge is power."
    "Time flies like an arrow; fruit flies like a banana."
    "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo."
    "James while John had had had had had had had had had had had a better effect."
    "This is a test of the emergency broadcast system."
    "Pack my box with five dozen liquor jugs."
    "How vexingly quick daft zebras jump!"
    "The five boxing wizards jump quickly."
    "Jackdaws love my big sphinx of quartz."
    "Crazy Frederick bought many very exquisite opal jewels."
    "We promptly judged antique ivory buckles for the next prize."
    "A mad boxer shot a quick, gloved jab to the jaw of his dizzy opponent."
    "The job requires extra pluck and zeal from every young wage earner."
    "Few quips galvanized the mock jury box."
    "Sixty zippers were quickly picked from the woven jute bag."
    "My girl wove six dozen plaid jackets before she quit."
    "Grumpy wizards make toxic brew for the evil queen and jack."
    "The quick onyx goblin jumps over the lazy dwarf."
    "An apple a day keeps the doctor away."
    "She sells sea shells by the sea shore."
    "Peter Piper picked a peck of pickled peppers."
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"
)

pass_count=0
fail_count=0
total=${#sentences[@]}

for i in "${!sentences[@]}"; do
    sentence="${sentences[$i]}"
    idx=$((i + 1))

    # Run both tokenizers with --ids --no-bos --log-disable --no-escape
    strata_out=$("$STRATA_BIN" -m "$MODEL" -p "$sentence" --ids --no-bos --log-disable 2>/dev/null || echo "STRATA_ERROR")
    llama_out=$("$LLAMA_BIN" -m "$MODEL" -p "$sentence" --ids --no-bos --no-escape --log-disable 2>/dev/null || echo "LLAMA_ERROR")

    if [ "$strata_out" = "$llama_out" ]; then
        pass_count=$((pass_count + 1))
        printf "[%3d/%d] PASS: %s\n" "$idx" "$total" "${sentence:0:60}"
    else
        fail_count=$((fail_count + 1))
        printf "[%3d/%d] FAIL: %s\n" "$idx" "$total" "${sentence:0:60}"
        if [ "$VERBOSE" = "1" ]; then
            echo "  strata: $strata_out"
            echo "  llama:  $llama_out"
        fi
    fi
done

echo ""
echo "=== Results: $MODEL_NAME ==="
echo "  PASS: $pass_count / $total"
echo "  FAIL: $fail_count / $total"
echo ""

if [ "$fail_count" -eq 0 ]; then
    echo "ALL TESTS PASSED!"
    exit 0
else
    echo "SOME TESTS FAILED"
    exit 1
fi
