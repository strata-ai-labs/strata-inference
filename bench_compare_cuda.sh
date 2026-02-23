#!/bin/bash
# bench_compare_cuda.sh — Compare strata CUDA vs llama.cpp on Linux/NVIDIA
set -euo pipefail

STRATA="./target/release/strata-generate"
LLAMA="/tmp/llama.cpp/build/bin/llama-cli"
MODELS_DIR="$HOME/.strata/models"
CTX=4096
TMPJSON="/tmp/strata_bench_$$.json"

PROMPT="In a quiet village nestled between ancient mountains, a young scholar discovered a mysterious manuscript hidden within the walls of an abandoned library. The pages, yellowed with age and covered in intricate symbols, seemed to pulse with a strange energy. As she carefully translated the first passages, she realized the text described a forgotten civilization that had mastered the art of harnessing starlight. Their technology, far beyond anything known to modern science, had allowed them to build cities that floated among the clouds."

NAMES=("TinyLlama_Q4KM")
PATHS=(
  "$MODELS_DIR/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
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

# Detect GPU info
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")

echo "================================================================"
echo " strata CUDA vs llama.cpp Performance Benchmark"
echo " Prompt: ~90 tokens | Greedy decoding (temp=0)"
echo " GPU: $GPU_NAME"
echo "================================================================"
echo ""

echo "--- CUDA Backend (100 generated tokens) ---"
header
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"; path="${PATHS[$i]}"
  [ ! -f "$path" ] && { print_row "$name" "FAIL" "FAIL"; continue; }
  echo "  Running $name (cuda)..." >&2
  sr=$(run_strata "$path" 100 "cuda")
  lr=$(run_llama "$path" 100 99)
  print_row "$name" "$sr" "$lr"
done
echo ""

echo "--- CPU Backend (20 generated tokens, for reference) ---"
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

echo "pp = prefill tok/s, tg = decode tok/s. Higher is better."
rm -f "$TMPJSON"
