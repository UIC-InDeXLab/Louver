#!/usr/bin/env bash
# Threshold Oracle Ablation — Experiment 7
# Produces:
#   results/*_threshold_oracle.csv  — frac + precision summary per oracle
#   results/*_timeseries.csv        — per-step tau + cov50_score (for oscillation plots)
#   results/threshold_oracle_all.json
#   results/figs/*_oracle_oscillation.png
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
CAPTURES="$DIR/../latency/captures"
RESULTS="$DIR/results"

python "$DIR/bench.py" \
  --captures-dir  "$CAPTURES" \
  --n-steps       200   \
  --n-steps-ts    2000  \
  --sample-size   256   \
  --top-frac      0.10  \
  --output-dir    "$RESULTS"

python "$DIR/plot_oracle_oscillation.py" \
  --results-dir "$RESULTS" \
  --out-dir     "$RESULTS/figs" \
  --smooth      7
