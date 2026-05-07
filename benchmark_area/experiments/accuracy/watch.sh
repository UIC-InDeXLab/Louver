#!/usr/bin/env bash
# Usage: bash watch.sh python eval/longbench.py --model ...
# Runs with unbuffered output, logs to /tmp/louver_<timestamp>.log, and tails it.
LOGFILE="/tmp/louver_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOGFILE"
echo "Watching live output (Ctrl+C to detach, job keeps running)..."
PYTHONUNBUFFERED=1 TORCH_EXTENSIONS_DIR=/tmp/torch_ext TRITON_CACHE_DIR=/tmp/triton_cache \
    "$@" 2>&1 | tee "$LOGFILE"
