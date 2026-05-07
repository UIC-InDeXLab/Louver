#!/usr/bin/env bash
# Offloading experiment — Experiment 3.1
# Run one method at a time; results → offload/results/
# Budget: 15% of tokens for all methods.
set -euo pipefail

SCRIPT="$(dirname "$0")/run_longbench_offload.py"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
TASKS="hotpotqa,2wikimqa,musique,qasper,narrativeqa"
MAX_SAMPLES=10          # 10 × 5 tasks = 50 examples → ~45 min per method at 4k context
MAX_INPUT_LENGTH=4096   # truncate context; index build dominates at 30k

# Uncomment one method per run:
METHOD="louver_offload"
# METHOD="hnsw_offload"
# METHOD="ivf_offload"
# METHOD="lsh_offload"

python "$SCRIPT" \
  --model            "$MODEL" \
  --method           "$METHOD" \
  --tasks            "$TASKS" \
  --max_samples      "$MAX_SAMPLES" \
  --max_input_length "$MAX_INPUT_LENGTH"
