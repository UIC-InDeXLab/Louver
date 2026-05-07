#!/usr/bin/env bash
# Pruning Power Ablation — Experiments 4 & 4.1
#
# Two separate passes:
#   1. louver ablation: S ∈ {2,4,8,16}, r ∈ {2,4,8,16}  (main paper figure)
#   2. standard methods: (clustering × enclosing), r ∈ {4,8,16}
#
# Requires pre-captured .pt files in quick_pruning/.
# Use Delta for full runs; local for smoke tests.
#
# Usage:
#   bash run.sh                    # run all with 8k Llama capture
#   bash run.sh --mode louver      # louver ablation only
#   bash run.sh --dry-run          # print commands only
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
QP="$DIR/../../quick_pruning"
RESULTS="$DIR/results"

# ── Captures ──────────────────────────────────────────────────────────────────
CAP_LLAMA="$QP/capture_qkv_8000_meta-llama_Llama-3.2-3B-Instruct.pt"
CAP_QWEN7B="$QP/capture_qkv_8000_Qwen_Qwen2.5-7B-Instruct.pt"

MODE="${1:-all}"          # all | louver | standard
DRY_RUN="${DRY_RUN:-0}"

run() {
    echo ">> $*"
    if [[ "$DRY_RUN" == "1" ]]; then return; fi
    python "$@"
}

# ── Louver ablation (S × r sweep) ─────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "louver" ]]; then
    echo "=== Louver ablation (S × r) — Llama-3.2-3B, N=8k ==="
    run "$DIR/bench.py" \
        --input-qkv  "$CAP_LLAMA" \
        --mode       louver \
        --S-values   2,4,8,16 \
        --r-values   2,4,8,16 \
        --n-queries  30 \
        --topk       20 \
        --output-dir "$RESULTS"

    # echo "=== Louver ablation — Qwen2.5-7B, N=8k ==="
    # run "$DIR/bench.py" \
    #     --input-qkv  "$CAP_QWEN7B" \
    #     --mode       louver \
    #     --S-values   2,4,8,16 \
    #     --r-values   2,4,8,16 \
    #     --output-dir "$RESULTS"
fi

# ── Standard methods (clustering × enclosing) ─────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "standard" ]]; then
    echo "=== Standard methods — Llama-3.2-3B, N=8k ==="
    run "$DIR/bench.py" \
        --input-qkv  "$CAP_LLAMA" \
        --mode       standard \
        --clusterings kcenter,kmeans,pq_subspace,batch_nn \
        --enclosings  ball_centroid,aabb,span_ball \
        --r-values   4,8,16 \
        --n-queries  30 \
        --topk       20 \
        --output-dir "$RESULTS"
fi

echo "Done. Results in $RESULTS/"
