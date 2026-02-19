#!/bin/bash
# bench_compare.sh — Compare strata vs llama.cpp prefill & decode throughput
set -euo pipefail

STRATA="./target/release/strata-generate"
LLAMA="$HOME/Documents/GitHub/llama.cpp/build/bin/llama-completion"
MODELS_DIR="$HOME/Documents/GitHub/strata-inference-benchmarks/models"
CTX=4096
TMPJSON="/tmp/strata_bench_$$.json"

PROMPT="In a quiet village nestled between ancient mountains, a young scholar discovered a mysterious manuscript hidden within the walls of an abandoned library. The pages, yellowed with age and covered in intricate symbols, seemed to pulse with a strange energy. As she carefully translated the first passages, she realized the text described a forgotten civilization that had mastered the art of harnessing starlight. Their technology, far beyond anything known to modern science, had allowed them to build cities that floated among the clouds."

NAMES=("GPT-2_Q8_0" "TinyLlama_Q4KM" "Qwen3-1.7B_Q8_0" "Phi3.1-mini_Q4KM" "Phi3.5-mini_Q4KM" "Gemma3-1B_Q4KM")
PATHS=(
  "$MODELS_DIR/gpt2.Q8_0.gguf"
  "$HOME/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
  "$MODELS_DIR/Qwen3-1.7B-Q8_0.gguf"
  "$MODELS_DIR/Phi-3.1-mini-4k-instruct-Q4_K_M.gguf"
  "$MODELS_DIR/Phi-3.5-mini-instruct-Q4_K_M.gguf"
  "$HOME/models/gemma-3-1b-it-Q4_K_M.gguf"
)

run_strata() {
  local model_path="$1" gen_tokens="$2" backend="$3"
  if $STRATA -m "$model_path" -p "$PROMPT" -n "$gen_tokens" --temp 0 \
    --backend "$backend" --output-format json --log-disable > "$TMPJSON" 2>/dev/null; then
    python3 -c "
import json
with open('$TMPJSON') as f:
    d = json.load(f)
t = d['timings']
print(f\"{t['prefill_tok_per_sec']:.1f}|{t['decode_tok_per_sec']:.1f}\")
"
  else
    echo "FAIL"
  fi
}

run_llama() {
  local model_path="$1" gen_tokens="$2" ngl="$3"
  local output
  output=$($LLAMA -m "$model_path" -p "$PROMPT" -n "$gen_tokens" -c $CTX \
    --temp 0 -no-cnv -ngl "$ngl" 2>&1) || { echo "FAIL"; return; }
  local pp tg
  pp=$(echo "$output" | grep "prompt eval time" | sed -E 's/.*,[[:space:]]*([0-9.]+) tokens per second.*/\1/')
  tg=$(echo "$output" | grep "eval time" | grep -v "prompt eval" | head -1 | sed -E 's/.*,[[:space:]]*([0-9.]+) tokens per second.*/\1/')
  if [ -z "$pp" ] || [ -z "$tg" ]; then
    echo "FAIL"
    return
  fi
  echo "${pp}|${tg}"
}

print_row() {
  local key="$1" sr="$2" lr="$3"
  local spp stg lpp ltg

  if [ "$sr" = "FAIL" ]; then spp="err"; stg="err"
  else spp=$(echo "$sr" | cut -d'|' -f1); stg=$(echo "$sr" | cut -d'|' -f2); fi

  if [ "$lr" = "FAIL" ]; then lpp="err"; ltg="err"
  else lpp=$(echo "$lr" | cut -d'|' -f1); ltg=$(echo "$lr" | cut -d'|' -f2); fi

  printf "%-22s | pp %8s  tg %8s | pp %8s  tg %8s\n" \
    "$key" "$spp" "$stg" "$lpp" "$ltg"
}

header() {
  printf "%-22s | %-25s | %-25s\n" "Model" "strata (tok/s)" "llama.cpp (tok/s)"
  printf "%-22s-+-%-25s-+-%-25s\n" "----------------------" "-------------------------" "-------------------------"
}

BACKEND="${1:-all}"

echo "================================================================"
echo " strata vs llama.cpp Performance Benchmark"
echo " Prompt: ~90 tokens | Greedy decoding (temp=0)"
echo " Machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
echo "================================================================"
echo ""

if [ "$BACKEND" = "metal" ] || [ "$BACKEND" = "all" ]; then
  echo "--- Metal Backend (100 generated tokens) ---"
  header
  for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"; path="${PATHS[$i]}"
    [ ! -f "$path" ] && { print_row "$name" "FAIL" "FAIL"; continue; }
    echo "  Running $name (metal)..." >&2
    sr=$(run_strata "$path" 100 "metal")
    lr=$(run_llama "$path" 100 99)
    print_row "$name" "$sr" "$lr"
  done
  echo ""
fi

if [ "$BACKEND" = "cpu" ] || [ "$BACKEND" = "all" ]; then
  echo "--- CPU Backend (20 generated tokens) ---"
  header
  for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"; path="${PATHS[$i]}"
    [ ! -f "$path" ] && { print_row "$name" "FAIL" "FAIL"; continue; }
    echo "  Running $name (cpu)..." >&2
    sr=$(run_strata "$path" 20 "cpu")
    lr=$(run_llama "$path" 20 0)
    print_row "$name" "$sr" "$lr"
  done
  echo ""
fi

echo "pp = prefill tok/s, tg = decode tok/s. Higher is better."
rm -f "$TMPJSON"
