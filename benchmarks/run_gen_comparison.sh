#!/usr/bin/env bash
set -euo pipefail

# Benchmark: strata-generate vs llama-completion
# All generation models, 1 prompt, 32 tokens, temp=0, seed=42
# Reports prefill and decode tok/s separately.

STRATA="/Users/aniruddhajoshi/Documents/GitHub/strata-inference/target/release/strata-generate"
LLAMA="/Users/aniruddhajoshi/Documents/GitHub/llama.cpp/build/bin/llama-completion"
MODELS_DIR="/Users/aniruddhajoshi/Documents/GitHub/strata-inference-benchmarks/models"

PROMPT="The quick brown fox jumps over the lazy dog. Once upon a time,"
N_TOKENS=32
OUTDIR="/Users/aniruddhajoshi/Documents/GitHub/strata-inference-benchmarks/output"
mkdir -p "$OUTDIR"

declare -a MODEL_NAMES=("GPT-2" "TinyLlama" "Qwen3-1.7B" "Phi-3.1-mini" "Phi-3.5-mini" "Gemma-3-1B" "Llama-3.1-8B-Q4_K_M" "Llama-3.1-8B-Q5_K_M" "Llama-3.1-8B-Q6_K" "Qwen3-8B-Q4_K_M" "Qwen3-8B-Q5_K_M" "Qwen3-8B-Q6_K")
declare -a MODEL_FILES=(
    "$MODELS_DIR/gpt2.Q8_0.gguf"
    "$MODELS_DIR/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    "$MODELS_DIR/Qwen3-1.7B-Q8_0.gguf"
    "$MODELS_DIR/Phi-3.1-mini-4k-instruct-Q4_K_M.gguf"
    "$MODELS_DIR/Phi-3.5-mini-instruct-Q4_K_M.gguf"
    "$MODELS_DIR/gemma-3-1b-it-Q4_K_M.gguf"
    "$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
    "$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
    "$MODELS_DIR/Qwen3-8B-Q4_K_M.gguf"
    "$MODELS_DIR/Qwen3-8B-Q5_K_M.gguf"
    "$MODELS_DIR/Qwen3-8B-Q6_K.gguf"
)

echo "================================================================"
echo "Text Generation Benchmark: strata-inference vs llama.cpp"
echo "Prompt: \"$PROMPT\""
echo "Tokens: $N_TOKENS | Temp: 0 | Seed: 42"
echo "================================================================"
echo ""

for i in "${!MODEL_NAMES[@]}"; do
    name="${MODEL_NAMES[$i]}"
    model="${MODEL_FILES[$i]}"
    safe_name=$(echo "$name" | tr ' ./' '___')

    if [[ ! -f "$model" ]]; then
        echo "--- $name ---"
        echo "  SKIP: model not found: $(basename "$model")"
        echo ""
        continue
    fi

    echo "--- $name ---"
    echo "  Model: $(basename "$model")"
    echo "  Size: $(du -h "$model" | cut -f1 | tr -d ' ')"

    # --- strata (JSON output for prefill/decode breakdown) ---
    strata_json="$OUTDIR/${safe_name}_strata.json"
    strata_err="$OUTDIR/${safe_name}_strata_err.txt"
    if "$STRATA" -m "$model" -p "$PROMPT" --temp 0 --seed 42 -n "$N_TOKENS" \
            --no-display-prompt --backend metal --output-format json --log-disable \
            > "$strata_json" 2>"$strata_err"; then
        # Extract timings from JSON
        strata_info=$(python3 -c "
import json, sys
d = json.load(open('$strata_json'))
t = d['timings']
print(f'prefill={t[\"prefill_ms\"]:.1f}ms ({t[\"prefill_tok_per_sec\"]:.1f} tok/s)')
print(f'decode={t[\"decode_ms\"]:.1f}ms ({t[\"decode_tok_per_sec\"]:.1f} tok/s)')
print(f'load={t[\"load_ms\"]:.0f}ms')
print(f'total={t[\"total_ms\"]:.0f}ms')
" 2>/dev/null) || strata_info="(parse error)"
        strata_output=$(python3 -c "
import json; d = json.load(open('$strata_json')); print(d['output'])" 2>/dev/null || echo "(parse error)")
        echo "  [strata]  $strata_info"
        echo "  [strata]  output: $strata_output"
    else
        echo "  [strata]  FAILED"
        head -3 "$strata_err" 2>/dev/null | sed 's/^/  [strata]  err: /'
        strata_output="FAIL"
    fi

    # --- llama.cpp ---
    # -no-cnv: disable chat template auto-detection (raw completion mode)
    # This ensures fair comparison — strata treats prompts as raw text.
    llama_out="$OUTDIR/${safe_name}_llama.txt"
    llama_err="$OUTDIR/${safe_name}_llama_err.txt"
    if "$LLAMA" -m "$model" -p "$PROMPT" --temp 0 --seed 42 -n "$N_TOKENS" \
            --no-display-prompt -no-cnv \
            > "$llama_out" 2>"$llama_err"; then
        # Extract prefill and decode timing from stderr
        llama_prefill=$(grep 'prompt eval time' "$llama_err" 2>/dev/null | \
                        grep -o '[0-9.]* tokens per second' | grep -o '[0-9.]*') || llama_prefill="?"
        llama_decode=$(grep '^ *eval time\|^common_perf_print: *eval time' "$llama_err" 2>/dev/null | \
                       grep -v 'prompt' | \
                       grep -o '[0-9.]* tokens per second' | grep -o '[0-9.]*') || llama_decode="?"
        llama_prefill_ms=$(grep 'prompt eval time' "$llama_err" 2>/dev/null | \
                           grep -o '= *[0-9.]*' | head -1 | grep -o '[0-9.]*') || llama_prefill_ms="?"
        llama_decode_ms=$(grep '^ *eval time\|^common_perf_print: *eval time' "$llama_err" 2>/dev/null | \
                          grep -v 'prompt' | \
                          grep -o '= *[0-9.]*' | head -1 | grep -o '[0-9.]*') || llama_decode_ms="?"
        # Clean output
        sed -i '' '/^> EOF by user$/d' "$llama_out" 2>/dev/null || true
        llama_output=$(cat "$llama_out")
        echo "  [llama]   prefill=${llama_prefill_ms}ms (${llama_prefill} tok/s)"
        echo "  [llama]   decode=${llama_decode_ms}ms (${llama_decode} tok/s)"
        echo "  [llama]   output: $llama_output"
    else
        echo "  [llama]   FAILED"
        head -3 "$llama_err" 2>/dev/null | sed 's/^/  [llama]   err: /'
        llama_output="FAIL"
    fi

    # --- Compare ---
    s_text=$(echo "$strata_output" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    l_text=$(echo "$llama_output" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    if [[ "$s_text" == "$l_text" ]]; then
        echo "  [match]   YES - outputs identical"
    else
        echo "  [match]   NO - outputs differ"
    fi
    echo ""
done

echo "================================================================"
echo "Raw output files saved to: $OUTDIR"
echo "================================================================"
