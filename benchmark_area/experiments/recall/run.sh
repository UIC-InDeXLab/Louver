#!/usr/bin/env bash
# Recall benchmark — Experiment 3.
# Comment in/out captures as needed. Results → reports/recall_*.csv
set -euo pipefail

SCRIPT="$(dirname "$0")/recall_bench.py"
CAPTURES="$(dirname "$0")/../latency/captures"
N_SAMPLES=100
MAX_N=8000   # truncate key sequence; use 'full' (remove --max-n) for full 40k
SEED=42

python "$SCRIPT" \
  --n-samples "$N_SAMPLES" \
  --max-n "$MAX_N" \
  --seed "$SEED" \
  --input-qkv \
  # "$CAPTURES/Qwen_Qwen2.5_14B_Instruct_layer24_N40000.pt"

    # "$CAPTURES/Qwen_Qwen2.5_7B_Instruct_layer14_N40000.pt"
    # "$CAPTURES/meta_llama_Llama_3.2_3B_Instruct_layer14_N40000.pt"
