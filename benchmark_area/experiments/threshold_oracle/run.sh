#!/usr/bin/env bash
# Threshold Oracle Ablation — Experiment 8
# Measures fraction retrieved + recall@10% for each oracle using latency captures.
set -euo pipefail

SCRIPT="$(dirname "$0")/bench.py"
CAPTURES="$(dirname "$0")/../latency/captures"

python "$SCRIPT" \
  --captures-dir  "$CAPTURES" \
  --n-steps       200   \
  --sample-size   256   \
  --recall-frac   0.10  \
  --output-dir    "$(dirname "$0")/results"
