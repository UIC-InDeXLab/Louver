#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUTDIR="results/longbench_v2"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_EXTENSIONS_DIR="${TMPDIR:-/tmp}/torch_ext_$$"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_cache_$$"

mkdir -p logs "$OUTDIR"

MAX_SAMPLES="${MAX_SAMPLES:-20}"
TASKS="${TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique}"
TOTAL=5
STEP=0

run() {
    local tag="$1"; shift
    STEP=$((STEP + 1))
    echo "=== [$STEP/$TOTAL] $tag ===" | tee -a logs/summary.log
    python eval/longbench.py "$@" --model "$MODEL" --output_dir "$OUTDIR" \
        --max_input_length 32768 --max_samples "$MAX_SAMPLES" --tasks "$TASKS" \
        2>&1 | tee logs/lb_${tag}.log
    echo "=== [$STEP/$TOTAL] $tag DONE ===" | tee -a logs/summary.log
}

# run louver_ta_oracle   --method louver_ta   --threshold_mode oracle
# run louver_full_oracle --method louver_full --threshold_mode oracle
# run dense_sdpa         --method dense_sdpa
run louver_ta_budget   --method louver_ta   --threshold_mode budget --budget_fraction 0.1
run louver_full_budget --method louver_full --threshold_mode budget --budget_fraction 0.1

echo "ALL DONE" | tee -a logs/summary.log
