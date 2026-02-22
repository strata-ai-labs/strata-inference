#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Tokenizer stress test — 100 sentences
#
# Extracts numbered sentences from stress_sentences.md, runs both tokenizers,
# and compares token IDs.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

STRATA_TOKENIZE="${STRATA_TOKENIZE:-strata-tokenize}"
LLAMA_TOKENIZE="${LLAMA_TOKENIZE:-llama-tokenize}"
MODEL="${MODEL:-}"

[[ -z "$MODEL" ]] && { echo "error: MODEL not set"; exit 1; }
[[ ! -f "$MODEL" ]] && { echo "error: model not found: $MODEL"; exit 1; }

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
RESET='\033[0m'

PASS=0
FAIL=0
ERRORS=""

tmpdir="$(mktemp -d)"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

# Extract sentences: lines matching "N. <text>" from the md file
SENTENCES=()
while IFS= read -r line; do
    SENTENCES+=("$line")
done < <(sed -n 's/^[0-9][0-9]*\. //p' "$SCRIPT_DIR/stress_sentences.md")

echo -e "${BOLD}Tokenizer Stress Test — ${#SENTENCES[@]} sentences${RESET}"
echo "Model: $MODEL"
echo ""

for i in "${!SENTENCES[@]}"; do
    num=$((i + 1))
    prompt="${SENTENCES[$i]}"
    short="${prompt:0:60}"

    # strata
    if ! "$STRATA_TOKENIZE" -m "$MODEL" -p "$prompt" --ids --log-disable \
            > "$tmpdir/strata" 2>/dev/null; then
        echo -e "  ${RED}ERR ${RESET} #${num}: strata crashed: \"$short\""
        FAIL=$((FAIL + 1))
        ERRORS="${ERRORS}#${num} strata crash: \"$short\"\n"
        continue
    fi

    # llama.cpp
    if ! "$LLAMA_TOKENIZE" -m "$MODEL" -p "$prompt" --ids \
            > "$tmpdir/llama" 2>/dev/null; then
        echo -e "  ${RED}ERR ${RESET} #${num}: llama crashed: \"$short\""
        FAIL=$((FAIL + 1))
        ERRORS="${ERRORS}#${num} llama crash: \"$short\"\n"
        continue
    fi

    if diff -q "$tmpdir/strata" "$tmpdir/llama" &>/dev/null; then
        count=$(tr ',' '\n' < "$tmpdir/strata" | wc -l | tr -d ' ')
        echo -e "  ${GREEN}PASS${RESET} #${num} (${count} tok) $short"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}FAIL${RESET} #${num} $short"
        echo "       strata: $(head -c 200 "$tmpdir/strata")"
        echo "       llama:  $(head -c 200 "$tmpdir/llama")"
        FAIL=$((FAIL + 1))
        ERRORS="${ERRORS}#${num}: \"$short\"\n"
    fi
done

echo ""
echo -e "${BOLD}=== Summary ===${RESET}"
echo -e "  ${GREEN}PASS: $PASS${RESET}  ${RED}FAIL: $FAIL${RESET}  Total: $((PASS + FAIL))"

if [[ $FAIL -gt 0 ]]; then
    echo ""
    echo -e "${RED}Failed cases:${RESET}"
    echo -e "$ERRORS"
    exit 1
fi
exit 0
