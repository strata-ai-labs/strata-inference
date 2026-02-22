#!/usr/bin/env bash
set -euo pipefail

# Context scaling benchmark: strata vs llama.cpp
# Tests decode performance degradation as prompt length increases.
# Uses truncated portions of a long prompt at various word counts.

STRATA="/Users/aniruddhajoshi/Documents/GitHub/strata-inference/target/release/strata-generate"
LLAMA="/Users/aniruddhajoshi/Documents/GitHub/llama.cpp/build/bin/llama-completion"
MODELS_DIR="/Users/aniruddhajoshi/Documents/GitHub/strata-inference-benchmarks/models"
PROMPT_FILE="/Users/aniruddhajoshi/Documents/GitHub/strata-inference-benchmarks/long_prompt.txt"
OUTDIR="/Users/aniruddhajoshi/Documents/GitHub/strata-inference-benchmarks/output/context_scaling"
mkdir -p "$OUTDIR"

N_TOKENS=32

# Only test models that matched in the short-prompt benchmark
declare -a MODEL_NAMES=("GPT-2" "TinyLlama" "Qwen3-1.7B" "Phi-3.1-mini")
declare -a MODEL_FILES=(
    "$MODELS_DIR/gpt2.Q8_0.gguf"
    "$MODELS_DIR/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    "$MODELS_DIR/Qwen3-1.7B-Q8_0.gguf"
    "$MODELS_DIR/Phi-3.1-mini-4k-instruct-Q4_K_M.gguf"
)

# Word counts to truncate to (approximate token counts vary by model)
declare -a WORD_COUNTS=(16 64 128 256 512)

FULL_PROMPT=$(cat "$PROMPT_FILE")

echo "================================================================"
echo "Context Scaling Benchmark: strata-inference vs llama.cpp"
echo "Decode $N_TOKENS tokens after prompts of increasing length"
echo "================================================================"
echo ""

for mi in "${!MODEL_NAMES[@]}"; do
    name="${MODEL_NAMES[$mi]}"
    model="${MODEL_FILES[$mi]}"

    if [[ ! -f "$model" ]]; then
        echo "--- $name: SKIP (not found) ---"
        echo ""
        continue
    fi

    echo "=== $name ==="
    printf "  %-10s | %-8s %-14s %-14s | %-8s %-14s %-14s | %s\n" \
        "Words" \
        "S-pf ms" "S-pf tok/s" "S-dec tok/s" \
        "L-pf ms" "L-pf tok/s" "L-dec tok/s" \
        "Match"
    printf "  %-10s-+-%-8s-%-14s-%-14s-+-%-8s-%-14s-%-14s-+-%s\n" \
        "----------" \
        "--------" "--------------" "--------------" \
        "--------" "--------------" "--------------" \
        "-----"

    for wc in "${WORD_COUNTS[@]}"; do
        # Truncate prompt to wc words
        prompt=$(echo "$FULL_PROMPT" | tr '\n' ' ' | awk -v n="$wc" '{for(i=1;i<=n&&i<=NF;i++) printf "%s ", $i; print ""}' | sed 's/ $//')
        safe_name=$(echo "${name}_${wc}w" | tr ' ./' '___')

        # --- strata ---
        strata_json="$OUTDIR/${safe_name}_strata.json"
        strata_err="$OUTDIR/${safe_name}_strata_err.txt"
        s_pf_ms="FAIL"; s_pf_tps="FAIL"; s_dec_tps="FAIL"; s_output=""
        if "$STRATA" -m "$model" -p "$prompt" --temp 0 --seed 42 -n "$N_TOKENS" \
                --no-display-prompt --backend metal --output-format json --log-disable \
                > "$strata_json" 2>"$strata_err"; then
            eval "$(python3 -c "
import json, sys
d = json.load(open('$strata_json'))
t = d['timings']
print(f's_pf_ms=\"{t[\"prefill_ms\"]:.1f}\"')
print(f's_pf_tps=\"{t[\"prefill_tok_per_sec\"]:.1f}\"')
print(f's_dec_tps=\"{t[\"decode_tok_per_sec\"]:.1f}\"')
" 2>/dev/null)" || true
            s_output=$(python3 -c "import json; d = json.load(open('$strata_json')); print(d['output'])" 2>/dev/null || echo "")
        fi

        # --- llama.cpp ---
        llama_out="$OUTDIR/${safe_name}_llama.txt"
        llama_err="$OUTDIR/${safe_name}_llama_err.txt"
        l_pf_ms="FAIL"; l_pf_tps="FAIL"; l_dec_tps="FAIL"; l_output=""
        if "$LLAMA" -m "$model" -p "$prompt" --temp 0 --seed 42 -n "$N_TOKENS" \
                --no-display-prompt -no-cnv \
                > "$llama_out" 2>"$llama_err"; then
            l_pf_tps=$(grep 'prompt eval time' "$llama_err" 2>/dev/null | \
                       grep -o '[0-9.]* tokens per second' | grep -o '[0-9.]*') || l_pf_tps="?"
            l_dec_tps=$(grep '^ *eval time\|^common_perf_print: *eval time' "$llama_err" 2>/dev/null | \
                        grep -v 'prompt' | \
                        grep -o '[0-9.]* tokens per second' | grep -o '[0-9.]*') || l_dec_tps="?"
            l_pf_ms=$(grep 'prompt eval time' "$llama_err" 2>/dev/null | \
                      grep -o '= *[0-9.]*' | head -1 | grep -o '[0-9.]*') || l_pf_ms="?"
            sed -i '' '/^> EOF by user$/d' "$llama_out" 2>/dev/null || true
            l_output=$(cat "$llama_out")
        fi

        # Compare
        s_text=$(echo "$s_output" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        l_text=$(echo "$l_output" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        if [[ "$s_text" == "$l_text" ]]; then
            match="YES"
        else
            match="NO"
        fi

        printf "  %-10s | %-8s %-14s %-14s | %-8s %-14s %-14s | %s\n" \
            "${wc}" \
            "$s_pf_ms" "$s_pf_tps" "$s_dec_tps" \
            "$l_pf_ms" "$l_pf_tps" "$l_dec_tps" \
            "$match"
    done
    echo ""
done

echo "================================================================"
echo "Raw output files saved to: $OUTDIR"
echo "================================================================"
