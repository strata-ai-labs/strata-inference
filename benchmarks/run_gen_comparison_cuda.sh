#!/usr/bin/env bash
set -euo pipefail

# Benchmark: strata-generate (CUDA) vs llama-completion (CUDA)
# All generation models, 1 prompt, 32 tokens, temp=0, seed=42
# Reports prefill and decode tok/s separately.
# Supports --token-ids mode for tokenizer-independent comparison.
# Adapted for Linux + NVIDIA CUDA from macOS/Metal version.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STRATA_ROOT="$(dirname "$SCRIPT_DIR")"

STRATA="${STRATA:-$STRATA_ROOT/target/release/strata-generate}"
STRATA_TOKENIZE="${STRATA_TOKENIZE:-$STRATA_ROOT/target/release/strata-tokenize}"
LLAMA="${LLAMA:-/tmp/llama.cpp/build/bin/llama-completion}"
MODELS_DIR="${MODELS_DIR:-$HOME/.strata/models}"

PROMPT="The quick brown fox jumps over the lazy dog. Once upon a time,"
N_TOKENS=32
OUTDIR="$SCRIPT_DIR/output"
mkdir -p "$OUTDIR"

# Use pre-tokenized IDs for fair comparison (bypasses tokenizer differences)
USE_TOKEN_IDS=${USE_TOKEN_IDS:-0}

# Models available through strata registry (filenames as downloaded)
declare -a MODEL_NAMES=("GPT-2" "TinyLlama" "Qwen3-1.7B" "Phi-3.5-mini" "Gemma-3-1B" "Llama-3.1-8B-Q4_K_M" "Qwen3-8B-Q4_K_M")
declare -a MODEL_FILES=(
    "$MODELS_DIR/gpt-2-q8_0.gguf"
    "$MODELS_DIR/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    "$MODELS_DIR/Qwen3-1.7B-Q8_0.gguf"
    "$MODELS_DIR/Phi-3.5-mini-instruct-Q4_K_M.gguf"
    "$MODELS_DIR/gemma-3-1b-it-Q4_K_M.gguf"
    "$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "$MODELS_DIR/Qwen3-8B-Q4_K_M.gguf"
)

# CUDA warmup: first model run pays one-time dequant cost.
# We run a warmup pass for each model to cache F32 weights on GPU,
# then the timed run measures steady-state performance.
WARMUP=${WARMUP:-1}

echo "================================================================"
echo "Text Generation Benchmark: strata-inference (CUDA) vs llama.cpp (CUDA)"
echo "Prompt: \"$PROMPT\""
echo "Tokens: $N_TOKENS | Temp: 0 | Seed: 42"
echo "Warmup: $WARMUP | Token-IDs mode: $USE_TOKEN_IDS"
echo "================================================================"
echo ""

# Collect results for summary table
declare -a SUMMARY_NAMES=()
declare -a SUMMARY_STRATA_DECODE=()
declare -a SUMMARY_LLAMA_DECODE=()
declare -a SUMMARY_STRATA_PREFILL=()
declare -a SUMMARY_LLAMA_PREFILL=()
declare -a SUMMARY_MATCH=()

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

    # Check file is not empty (failed download)
    if [[ ! -s "$model" ]]; then
        echo "--- $name ---"
        echo "  SKIP: model file is empty (download failed?): $(basename "$model")"
        echo ""
        continue
    fi

    echo "--- $name ---"
    echo "  Model: $(basename "$model")"
    echo "  Size: $(du -h "$model" | cut -f1 | tr -d ' ')"

    # --- Tokenize prompt if using token-IDs mode ---
    token_ids_arg=""
    if [[ "$USE_TOKEN_IDS" -eq 1 ]] && [[ -x "$STRATA_TOKENIZE" ]]; then
        token_ids=$(
            "$STRATA_TOKENIZE" -m "$model" -p "$PROMPT" --ids --log-disable 2>/dev/null \
            | tr -d '[]' | tr -d ' '
        ) || token_ids=""
        if [[ -n "$token_ids" ]]; then
            token_ids_arg="--token-ids $token_ids"
            echo "  [tokenize] IDs: ${token_ids:0:80}..."
        fi
    fi

    # --- strata warmup (populates F32 dequant cache) ---
    if [[ "$WARMUP" -eq 1 ]]; then
        echo "  [strata]  warming up (dequant cache)..."
        "$STRATA" -m "$model" -p "warmup" --temp 0 -n 1 \
                --no-display-prompt --backend cuda --output-format json --log-disable \
                > /dev/null 2>/dev/null || true
    fi

    # --- strata (JSON output for prefill/decode breakdown) ---
    strata_json="$OUTDIR/${safe_name}_strata.json"
    strata_err="$OUTDIR/${safe_name}_strata_err.txt"
    strata_cmd_args=(-m "$model" --temp 0 --seed 42 -n "$N_TOKENS"
            --no-display-prompt --backend cuda --output-format json --log-disable)
    if [[ -n "$token_ids_arg" ]]; then
        strata_cmd_args+=($token_ids_arg)
    else
        strata_cmd_args+=(-p "$PROMPT")
    fi
    if "$STRATA" "${strata_cmd_args[@]}" > "$strata_json" 2>"$strata_err"; then
        # Extract timings from JSON
        strata_decode_tps=""
        strata_prefill_tps=""
        strata_info=$(python3 -c "
import json, sys
d = json.load(open('$strata_json'))
t = d['timings']
print(f'prefill={t[\"prefill_ms\"]:.1f}ms ({t[\"prefill_tok_per_sec\"]:.1f} tok/s)')
print(f'decode={t[\"decode_ms\"]:.1f}ms ({t[\"decode_tok_per_sec\"]:.1f} tok/s)')
print(f'load={t[\"load_ms\"]:.0f}ms')
print(f'total={t[\"total_ms\"]:.0f}ms')
" 2>/dev/null) || strata_info="(parse error)"
        strata_decode_tps=$(python3 -c "
import json; d = json.load(open('$strata_json')); print(f'{d[\"timings\"][\"decode_tok_per_sec\"]:.0f}')
" 2>/dev/null) || strata_decode_tps="?"
        strata_prefill_tps=$(python3 -c "
import json; d = json.load(open('$strata_json')); print(f'{d[\"timings\"][\"prefill_tok_per_sec\"]:.0f}')
" 2>/dev/null) || strata_prefill_tps="?"
        strata_output=$(python3 -c "
import json; d = json.load(open('$strata_json')); print(d['output'])" 2>/dev/null || echo "(parse error)")
        echo "  [strata]  $strata_info"
        echo "  [strata]  output: $strata_output"
    else
        echo "  [strata]  FAILED"
        head -3 "$strata_err" 2>/dev/null | sed 's/^/  [strata]  err: /'
        strata_output="FAIL"
        strata_decode_tps="0"
        strata_prefill_tps="0"
    fi

    # --- llama.cpp ---
    # -no-cnv: disable chat template auto-detection (raw completion mode)
    # This ensures fair comparison -- strata treats prompts as raw text.
    llama_out="$OUTDIR/${safe_name}_llama.txt"
    llama_err="$OUTDIR/${safe_name}_llama_err.txt"
    llama_decode_tps="0"
    llama_prefill_tps="0"
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
        # Clean output (Linux sed syntax)
        sed -i '/^> EOF by user$/d' "$llama_out" 2>/dev/null || true
        llama_output=$(cat "$llama_out")
        echo "  [llama]   prefill=${llama_prefill_ms}ms (${llama_prefill} tok/s)"
        echo "  [llama]   decode=${llama_decode_ms}ms (${llama_decode} tok/s)"
        echo "  [llama]   output: $llama_output"
        llama_decode_tps=$(echo "$llama_decode" | head -1)
        llama_prefill_tps=$(echo "$llama_prefill" | head -1)
    else
        echo "  [llama]   FAILED"
        head -3 "$llama_err" 2>/dev/null | sed 's/^/  [llama]   err: /'
        llama_output="FAIL"
    fi

    # --- Compare ---
    s_text=$(echo "$strata_output" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    l_text=$(echo "$llama_output" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    match_status="NO"
    if [[ "$s_text" == "$l_text" ]]; then
        echo "  [match]   YES - outputs identical"
        match_status="YES"
    else
        echo "  [match]   NO - outputs differ"
    fi
    echo ""

    # Collect for summary
    SUMMARY_NAMES+=("$name")
    SUMMARY_STRATA_DECODE+=("${strata_decode_tps:-0}")
    SUMMARY_LLAMA_DECODE+=("${llama_decode_tps:-0}")
    SUMMARY_STRATA_PREFILL+=("${strata_prefill_tps:-0}")
    SUMMARY_LLAMA_PREFILL+=("${llama_prefill_tps:-0}")
    SUMMARY_MATCH+=("$match_status")
done

echo "================================================================"
echo "SUMMARY TABLE"
echo "================================================================"
printf "%-22s %10s %10s %7s %8s\n" "Model" "strata" "llama.cpp" "Ratio" "Match"
printf "%-22s %10s %10s %7s %8s\n" "-----" "tok/s" "tok/s" "" ""
for i in "${!SUMMARY_NAMES[@]}"; do
    s="${SUMMARY_STRATA_DECODE[$i]}"
    l="${SUMMARY_LLAMA_DECODE[$i]}"
    m="${SUMMARY_MATCH[$i]}"
    ratio="?"
    if [[ "$l" != "?" ]] && [[ "$l" != "0" ]] && [[ "$s" != "?" ]] && [[ "$s" != "0" ]]; then
        ratio=$(python3 -c "print(f'{${s}/${l}*100:.0f}%')" 2>/dev/null) || ratio="?"
    fi
    printf "%-22s %10s %10s %7s %8s\n" "${SUMMARY_NAMES[$i]}" "$s" "$l" "$ratio" "$m"
done
echo ""
echo "================================================================"
echo "Raw output files saved to: $OUTDIR"
echo "================================================================"
