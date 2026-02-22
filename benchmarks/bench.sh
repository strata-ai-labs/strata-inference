#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# strata-inference vs llama.cpp benchmark
#
# Treats both engines as black boxes. Runs their CLIs, captures output,
# compares results. That's it.
# ============================================================================

# --- Configuration -----------------------------------------------------------

# Paths to binaries (override via environment)
STRATA_TOKENIZE="${STRATA_TOKENIZE:-strata-tokenize}"
STRATA_EMBED="${STRATA_EMBED:-strata-embed}"
STRATA_GENERATE="${STRATA_GENERATE:-strata-generate}"
LLAMA_TOKENIZE="${LLAMA_TOKENIZE:-llama-tokenize}"
LLAMA_EMBED="${LLAMA_EMBED:-llama-embedding}"
LLAMA_GENERATE="${LLAMA_GENERATE:-llama-cli}"

# Model path (legacy single-model mode; optional if auto-detect finds models)
MODEL="${MODEL:-}"

# What to benchmark
TASK="${TASK:-all}"  # tokenize | embed | generate | all

# Generation settings
GEN_TOKENS="${GEN_TOKENS:-32}"

# Backend for strata (auto|cpu|metal|cuda)
BACKEND="${BACKEND:-auto}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Auto-detect models by category -----------------------------------------

# Helper: collect unique model paths from glob patterns
_collect_models() {
    local -a detected=()
    for f in "$@"; do
        [[ -f "$f" ]] && detected+=("$f")
    done
    # Deduplicate (bash 3 compatible)
    local -a result=()
    for f in "${detected[@]+"${detected[@]}"}"; do
        local dup=0
        for existing in "${result[@]+"${result[@]}"}"; do
            [[ "$f" == "$existing" ]] && { dup=1; break; }
        done
        [[ $dup -eq 0 ]] && result+=("$f")
    done
    echo "${result[@]+"${result[@]}"}"
}

# Embedding models
if [[ -z "${EMBED_MODELS:-}" ]]; then
    read -ra EMBED_MODELS_ARRAY <<< "$(_collect_models \
        "$SCRIPT_DIR/models/"*MiniLM*.gguf \
        "$SCRIPT_DIR/models/"*embeddinggemma*.gguf \
        "$SCRIPT_DIR/models/"*nomic-embed*.gguf \
        "$SCRIPT_DIR/models/"*bge-m3*.gguf \
        "$SCRIPT_DIR/models/"*Qwen3-Embedding*.gguf)"
else
    read -ra EMBED_MODELS_ARRAY <<< "$EMBED_MODELS"
fi

# Generation models (causal LLMs)
if [[ -z "${GEN_MODELS:-}" ]]; then
    read -ra GEN_MODELS_ARRAY <<< "$(_collect_models \
        "$SCRIPT_DIR/models/"*llama*.gguf \
        "$SCRIPT_DIR/models/"*tinyllama*.gguf \
        "$SCRIPT_DIR/models/"*gemma*-it*.gguf \
        "$SCRIPT_DIR/models/"*gemma*-instruct*.gguf \
        "$SCRIPT_DIR/models/"*mistral*.gguf \
        "$SCRIPT_DIR/models/"*qwen*[!Ee]*.gguf \
        "$SCRIPT_DIR/models/"*phi*.gguf \
        "$SCRIPT_DIR/models/"*deepseek*.gguf \
        "$SCRIPT_DIR/models/"*gpt2*.gguf)"
else
    read -ra GEN_MODELS_ARRAY <<< "$GEN_MODELS"
fi

# Prompts
TOKENIZE_PROMPTS=(
    "Hello, world!"
    "The quick brown fox jumps over the lazy dog."
    "Rust is a systems programming language focused on safety and performance."
    "42 + 7 = 49"
)

EMBED_PROMPTS=(
    "Machine learning is a subset of artificial intelligence."
    "The weather today is sunny and warm."
    "Rust provides memory safety without garbage collection."
    "Deep neural networks can approximate complex functions."
    "The capital of France is Paris."
)

GENERATE_PROMPTS=(
    "Once upon a time in a land far away,"
    "Explain the concept of recursion in simple terms:"
    "What are the three laws of thermodynamics?"
)

# --- Helpers -----------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

PASS=0
FAIL=0
SKIP=0

tmpdir=""
cleanup() { [[ -n "$tmpdir" ]] && rm -rf "$tmpdir"; }
trap cleanup EXIT
tmpdir="$(mktemp -d)"

log()  { echo -e "${CYAN}[bench]${RESET} $*"; }
pass() { echo -e "  ${GREEN}PASS${RESET} $*"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}FAIL${RESET} $*"; FAIL=$((FAIL + 1)); }
skip() { echo -e "  ${YELLOW}SKIP${RESET} $*"; SKIP=$((SKIP + 1)); }
header() { echo -e "\n${BOLD}=== $* ===${RESET}"; }

die() { echo -e "${RED}error:${RESET} $*" >&2; exit 1; }

has_cmd() { command -v "$1" &>/dev/null; }

# Time a command, write wall time (ms) to the file passed as $1,
# remaining args are the command to run.
timed() {
    local time_file="$1"; shift
    local start end
    start=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
    "$@"
    local rc=$?
    end=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
    echo "scale=1; ($end - $start) * 1000" | bc > "$time_file"
    return $rc
}

# Cosine similarity between two files of space-separated floats.
# Prints the similarity value.
cosine_sim() {
    local file_a="$1" file_b="$2"
    python3 -c "
import sys
a = list(map(float, open('$file_a').read().split()))
b = list(map(float, open('$file_b').read().split()))
if len(a) != len(b):
    print(f'DIM_MISMATCH({len(a)}vs{len(b)})')
    sys.exit(0)
dot = sum(x*y for x,y in zip(a,b))
na = sum(x*x for x in a)**0.5
nb = sum(x*x for x in b)**0.5
print(f'{dot/(na*nb):.6f}' if na*nb > 0 else '0.000000')
"
}

# --- Checks ------------------------------------------------------------------

check_binaries() {
    local bins=()
    case "$TASK" in
        tokenize) bins=("$STRATA_TOKENIZE" "$LLAMA_TOKENIZE") ;;
        embed)    bins=("$STRATA_EMBED" "$LLAMA_EMBED") ;;
        generate) bins=("$STRATA_GENERATE" "$LLAMA_GENERATE") ;;
        all)      bins=("$STRATA_TOKENIZE" "$STRATA_EMBED" "$STRATA_GENERATE"
                        "$LLAMA_TOKENIZE" "$LLAMA_EMBED" "$LLAMA_GENERATE") ;;
    esac
    local missing=0
    for bin in "${bins[@]}"; do
        if ! has_cmd "$bin"; then
            echo "  not found: $bin"
            missing=1
        fi
    done
    if [[ $missing -eq 1 ]]; then
        echo ""
        echo "Set paths via environment variables, e.g.:"
        echo "  STRATA_EMBED=./target/release/strata-embed \\"
        echo "  LLAMA_EMBED=/path/to/llama-embedding \\"
        echo "  MODEL=models/all-MiniLM-L6-v2.Q8_0.gguf ./bench.sh"
        die "missing binaries"
    fi
}

# --- Tokenize ----------------------------------------------------------------

bench_tokenize_model() {
    local model_path="$1"
    local model_name
    model_name=$(basename "$model_path" .gguf)

    log "Tokenize model: $model_name"

    for prompt in "${TOKENIZE_PROMPTS[@]}"; do
        local short="${prompt:0:50}"

        # strata: --ids outputs [1, 2, 3]
        local strata_out="$tmpdir/tok_strata"
        local strata_time="$tmpdir/tok_strata_time"
        if ! timed "$strata_time" "$STRATA_TOKENIZE" -m "$model_path" -p "$prompt" --ids --log-disable \
                > "$strata_out" 2>/dev/null; then
            fail "[$model_name] strata-tokenize crashed: \"$short\""
            continue
        fi

        # llama.cpp: --ids outputs the same format
        local llama_out="$tmpdir/tok_llama"
        local llama_time="$tmpdir/tok_llama_time"
        if ! timed "$llama_time" "$LLAMA_TOKENIZE" -m "$model_path" -p "$prompt" --ids \
                > "$llama_out" 2>/dev/null; then
            fail "[$model_name] llama-tokenize crashed: \"$short\""
            continue
        fi

        local s_ms; s_ms=$(cat "$strata_time")
        local l_ms; l_ms=$(cat "$llama_time")

        if diff -q "$strata_out" "$llama_out" &>/dev/null; then
            local count
            count=$(tr ',' '\n' < "$strata_out" | wc -l | tr -d ' ')
            pass "[$model_name] \"$short\" — ${count} tokens  (strata: ${s_ms}ms, llama: ${l_ms}ms)"
        else
            fail "[$model_name] \"$short\" — IDs differ  (strata: ${s_ms}ms, llama: ${l_ms}ms)"
            echo "    strata: $(head -c 200 "$strata_out")"
            echo "    llama:  $(head -c 200 "$llama_out")"
        fi
    done
}

bench_tokenize() {
    header "Tokenization"

    if ! has_cmd "$STRATA_TOKENIZE" || ! has_cmd "$LLAMA_TOKENIZE"; then
        skip "tokenize binaries not found"
        return
    fi

    # Use all unique models across embed + generate lists, plus MODEL if set
    local -a models_to_run=()
    for f in "${EMBED_MODELS_ARRAY[@]+"${EMBED_MODELS_ARRAY[@]}"}" \
             "${GEN_MODELS_ARRAY[@]+"${GEN_MODELS_ARRAY[@]}"}"; do
        [[ -f "$f" ]] && models_to_run+=("$f")
    done
    [[ -n "$MODEL" && -f "$MODEL" ]] && models_to_run+=("$MODEL")

    # Deduplicate
    local -a unique=()
    for f in "${models_to_run[@]+"${models_to_run[@]}"}"; do
        local dup=0
        for existing in "${unique[@]+"${unique[@]}"}"; do
            [[ "$f" == "$existing" ]] && { dup=1; break; }
        done
        [[ $dup -eq 0 ]] && unique+=("$f")
    done

    if [[ ${#unique[@]} -eq 0 ]]; then
        skip "no models found for tokenization"
        return
    fi

    for model_path in "${unique[@]}"; do
        bench_tokenize_model "$model_path"
    done
}

# --- Embed -------------------------------------------------------------------

bench_embed_model() {
    local model_path="$1"
    local model_name
    model_name=$(basename "$model_path" .gguf)

    # Show model size
    local model_size
    model_size=$(du -h "$model_path" | cut -f1 | tr -d ' ')
    log "Embedding model: $model_name (${model_size})"

    # Build strata backend args
    local strata_backend_args=()
    if [[ "$BACKEND" != "auto" ]]; then
        strata_backend_args=(--backend "$BACKEND")
    fi

    # Probe: try the first prompt to see if strata supports this model
    local probe_out="$tmpdir/emb_probe"
    local probe_err="$tmpdir/emb_probe_err"
    if ! "$STRATA_EMBED" -m "$model_path" -p "${EMBED_PROMPTS[0]}" \
            ${strata_backend_args[@]+"${strata_backend_args[@]}"} \
            --embd-output-format raw --log-disable \
            > "$probe_out" 2>"$probe_err"; then
        local err_hint
        err_hint=$(head -c 200 "$probe_err" 2>/dev/null || true)
        skip "[$model_name] strata-embed unsupported: ${err_hint:-crashed}"
        # Still run llama.cpp alone to verify the model works
        log "  (running llama.cpp only for reference)"
        local llama_raw="$tmpdir/emb_llama_raw"
        if "$LLAMA_EMBED" -m "$model_path" -p "${EMBED_PROMPTS[0]}" \
                --embd-output-format json --embd-normalize 2 \
                > "$llama_raw" 2>/dev/null; then
            local l_dim
            l_dim=$(python3 -c "
import json
d = json.load(open('$llama_raw'))
print(len(d['data'][0]['embedding']))
" 2>/dev/null || echo "?")
            log "  llama.cpp works — dim=$l_dim"
        fi
        return
    fi

    for prompt in "${EMBED_PROMPTS[@]}"; do
        local short="${prompt:0:50}"

        # strata: raw format = space-separated floats
        local strata_out="$tmpdir/emb_strata"
        local strata_time="$tmpdir/emb_strata_time"
        if ! timed "$strata_time" "$STRATA_EMBED" -m "$model_path" -p "$prompt" \
                ${strata_backend_args[@]+"${strata_backend_args[@]}"} \
                --embd-output-format raw --log-disable \
                > "$strata_out" 2>/dev/null; then
            fail "[$model_name] strata-embed crashed: \"$short\""
            continue
        fi

        # llama.cpp: json output, extract the embedding array
        local llama_raw="$tmpdir/emb_llama_raw"
        local llama_out="$tmpdir/emb_llama"
        local llama_time="$tmpdir/emb_llama_time"
        if ! timed "$llama_time" "$LLAMA_EMBED" -m "$model_path" -p "$prompt" \
                --embd-output-format json --embd-normalize 2 \
                > "$llama_raw" 2>/dev/null; then
            fail "[$model_name] llama-embedding crashed: \"$short\""
            continue
        fi

        # Extract embedding floats from llama.cpp OpenAI-style JSON
        python3 -c "
import json, sys
d = json.load(open('$llama_raw'))
emb = d['data'][0]['embedding']
print(' '.join(f'{x}' for x in emb))
" > "$llama_out" 2>/dev/null || { fail "[$model_name] parse llama JSON: \"$short\""; continue; }

        local s_ms; s_ms=$(cat "$strata_time")
        local l_ms; l_ms=$(cat "$llama_time")

        local sim
        sim=$(cosine_sim "$strata_out" "$llama_out")

        local s_dim l_dim
        s_dim=$(wc -w < "$strata_out" | tr -d ' ')
        l_dim=$(wc -w < "$llama_out" | tr -d ' ')

        if [[ "$sim" == DIM_MISMATCH* ]]; then
            fail "[$model_name] \"$short\" — $sim  (strata dim=$s_dim, llama dim=$l_dim)"
        elif python3 -c "exit(0 if float('$sim') > 0.99 else 1)"; then
            pass "[$model_name] \"$short\" — cosine=$sim dim=$s_dim  (strata: ${s_ms}ms, llama: ${l_ms}ms)"
        else
            fail "[$model_name] \"$short\" — cosine=$sim < 0.99 dim=$s_dim  (strata: ${s_ms}ms, llama: ${l_ms}ms)"
        fi
    done
}

bench_embed() {
    header "Embeddings"

    if ! has_cmd "$STRATA_EMBED" || ! has_cmd "$LLAMA_EMBED"; then
        skip "embed binaries not found"
        return
    fi

    # Determine which models to benchmark
    local models_to_run=()
    if [[ ${#EMBED_MODELS_ARRAY[@]} -gt 0 ]]; then
        models_to_run=("${EMBED_MODELS_ARRAY[@]}")
    elif [[ -n "$MODEL" ]]; then
        models_to_run=("$MODEL")
    else
        skip "no embedding models found (set MODEL or place .gguf files in models/)"
        return
    fi

    for model_path in "${models_to_run[@]}"; do
        if [[ ! -f "$model_path" ]]; then
            skip "model not found: $model_path"
            continue
        fi
        bench_embed_model "$model_path"
    done
}

# --- Generate ----------------------------------------------------------------

bench_generate_model() {
    local model_path="$1"
    local model_name
    model_name=$(basename "$model_path" .gguf)

    local model_size
    model_size=$(du -h "$model_path" | cut -f1 | tr -d ' ')
    log "Generation model: $model_name (${model_size})"

    # Build strata backend args
    local strata_backend_args=()
    if [[ "$BACKEND" != "auto" ]]; then
        strata_backend_args=(--backend "$BACKEND")
    fi

    # Probe: try a short generation to see if strata supports this model
    local probe_out="$tmpdir/gen_probe"
    local probe_err="$tmpdir/gen_probe_err"
    if ! "$STRATA_GENERATE" -m "$model_path" -p "Hello" \
            ${strata_backend_args[@]+"${strata_backend_args[@]}"} \
            --temp 0 --seed 42 -n 4 --no-display-prompt --log-disable \
            > "$probe_out" 2>"$probe_err"; then
        local err_hint
        err_hint=$(head -c 200 "$probe_err" 2>/dev/null || true)
        skip "[$model_name] strata-generate unsupported: ${err_hint:-crashed}"
        return
    fi

    for prompt in "${GENERATE_PROMPTS[@]}"; do
        local short="${prompt:0:50}"

        # strata
        local strata_out="$tmpdir/gen_strata"
        local strata_time="$tmpdir/gen_strata_time"
        if ! timed "$strata_time" "$STRATA_GENERATE" -m "$model_path" -p "$prompt" \
                ${strata_backend_args[@]+"${strata_backend_args[@]}"} \
                --temp 0 --seed 42 -n "$GEN_TOKENS" --no-display-prompt \
                --log-disable \
                > "$strata_out" 2>/dev/null; then
            fail "[$model_name] strata-generate crashed: \"$short\""
            continue
        fi

        # llama.cpp
        local llama_out="$tmpdir/gen_llama"
        local llama_stderr="$tmpdir/gen_llama_err"
        local llama_time="$tmpdir/gen_llama_time"
        if ! timed "$llama_time" "$LLAMA_GENERATE" -m "$model_path" -p "$prompt" \
                -no-cnv --no-display-prompt --temp 0 --seed 42 -n "$GEN_TOKENS" \
                > "$llama_out" 2>"$llama_stderr"; then
            fail "[$model_name] llama-cli crashed: \"$short\""
            continue
        fi

        local s_ms; s_ms=$(cat "$strata_time")
        local l_ms; l_ms=$(cat "$llama_time")

        # Compare trimmed output
        local s_text l_text
        s_text=$(sed 's/^[[:space:]]*//;s/[[:space:]]*$//' "$strata_out")
        l_text=$(sed 's/^[[:space:]]*//;s/[[:space:]]*$//' "$llama_out")

        local match="NO"
        [[ "$s_text" == "$l_text" ]] && match="YES"

        # Extract llama.cpp tok/s from stderr
        local l_toks
        l_toks=$(grep -o 'eval time.*' "$llama_stderr" 2>/dev/null | \
                 grep -o '[0-9.]*[[:space:]]*tokens per second' | \
                 grep -o '[0-9.]*' | head -1) || l_toks="?"

        if [[ "$match" == "YES" ]]; then
            pass "[$model_name] \"$short\" — output matches  (strata: ${s_ms}ms, llama: ${l_ms}ms, llama tok/s: $l_toks)"
        else
            fail "[$model_name] \"$short\" — output differs  (strata: ${s_ms}ms, llama: ${l_ms}ms)"
            echo "    strata: ${s_text:0:120}"
            echo "    llama:  ${l_text:0:120}"
        fi
    done
}

bench_generate() {
    header "Generation"

    if ! has_cmd "$STRATA_GENERATE" || ! has_cmd "$LLAMA_GENERATE"; then
        skip "generate binaries not found"
        return
    fi

    # Determine which models to benchmark
    local models_to_run=()
    if [[ ${#GEN_MODELS_ARRAY[@]} -gt 0 ]]; then
        models_to_run=("${GEN_MODELS_ARRAY[@]}")
    elif [[ -n "$MODEL" ]]; then
        models_to_run=("$MODEL")
    else
        skip "no generation models found (set MODEL or place .gguf files in models/)"
        return
    fi

    for model_path in "${models_to_run[@]}"; do
        if [[ ! -f "$model_path" ]]; then
            skip "model not found: $model_path"
            continue
        fi
        bench_generate_model "$model_path"
    done
}

# --- Main --------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: [options] ./bench.sh

Models are auto-detected from the models/ directory. Override via environment.

Environment variables:
  MODEL               Path to single GGUF model (used as fallback if auto-detect empty)
  TASK                What to benchmark: tokenize|embed|generate|all (default: all)
  BACKEND             Strata backend: auto|cpu|metal|cuda (default: auto)
  GEN_TOKENS          Max tokens for generation (default: 32)
  EMBED_MODELS        Space-separated paths to embedding models (auto-detected if unset)
  GEN_MODELS          Space-separated paths to generation models (auto-detected if unset)
  STRATA_TOKENIZE     Path to strata-tokenize binary
  STRATA_EMBED        Path to strata-embed binary
  STRATA_GENERATE     Path to strata-generate binary
  LLAMA_TOKENIZE      Path to llama-tokenize binary
  LLAMA_EMBED         Path to llama-embedding binary
  LLAMA_GENERATE      Path to llama-cli binary

Examples:
  TASK=all ./bench.sh                              # auto-detect all models
  TASK=embed ./bench.sh                            # embedding models only
  TASK=generate ./bench.sh                         # generation models only
  TASK=embed BACKEND=metal ./bench.sh              # force Metal backend
  GEN_MODELS="models/gpt2.Q8_0.gguf" TASK=generate ./bench.sh
  MODEL=models/tinyllama.gguf TASK=generate GEN_TOKENS=64 ./bench.sh
EOF
    exit 1
}

if [[ -n "$MODEL" && ! -f "$MODEL" ]]; then
    die "model not found: $MODEL"
fi

# Show what was detected
_n_embed=${#EMBED_MODELS_ARRAY[@]}
_n_gen=${#GEN_MODELS_ARRAY[@]}
log "Task:    $TASK"
log "Backend: $BACKEND"
log "Embed models:  $_n_embed detected"
log "Gen models:    $_n_gen detected"
[[ -n "$MODEL" ]] && log "MODEL override: $MODEL"
unset _n_embed _n_gen

check_binaries

case "$TASK" in
    tokenize) bench_tokenize ;;
    embed)    bench_embed ;;
    generate) bench_generate ;;
    all)
        bench_tokenize
        bench_embed
        bench_generate
        ;;
    *) die "unknown task: $TASK" ;;
esac

# --- Summary -----------------------------------------------------------------
header "Summary"
echo -e "  ${GREEN}PASS: $PASS${RESET}  ${RED}FAIL: $FAIL${RESET}  ${YELLOW}SKIP: $SKIP${RESET}"

[[ $FAIL -gt 0 ]] && exit 1
exit 0
