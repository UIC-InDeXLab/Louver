#!/usr/bin/env bash
# ============================================================
# Latency benchmark — QKV capture script
# ============================================================
#
# FIRST-TIME SETUP ON A NEW MACHINE:
#   1. Make sure the venv is activated and deps installed.
#   2. Run a smoke test with existing captures (no generation needed):
#
#        python gpu_bench.py \
#            --input-qkv ../../quick_pruning/capture_qkv_12000_Qwen_Qwen2.5-7B-Instruct.pt \
#            --n-steps 200
#
#        python cpu_bench.py \
#            --input-qkv ../../quick_pruning/capture_qkv_12000_Qwen_Qwen2.5-7B-Instruct.pt \
#            --n-steps 200
#
#   3. If smoke test passes, run this script to capture all 4 models:
#        bash capture_all.sh
#
#   4. Then run the full benchmark with the saved captures:
#        python gpu_bench.py --input-qkv captures/<file>.pt --n-steps 10000
#
#   Alternatively, skip capture entirely and benchmark on-the-fly
#   (model is freed from GPU before timing starts):
#        python gpu_bench.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
#            --n-tokens 32000 --n-steps 10000
#
# DISK SPACE: ~400-600 MB per model, ~2 GB total (float16, single layer, 40k tokens).
#
# ORDER: small models first (faster), large models last.
# ============================================================
set -e
cd "$(dirname "$0")"

python - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    print("ERROR: no CUDA GPU visible", file=sys.stderr); sys.exit(1)
print(f"GPU: {torch.cuda.get_device_name(0)}  "
      f"free={torch.cuda.mem_get_info()[0]//1024**3}GB")
EOF

export PYTHONUNBUFFERED=1
OUTDIR="captures"
mkdir -p "$OUTDIR" logs

TOTAL=4
STEP=0

run_capture() {
    local tag="$1"; shift
    STEP=$((STEP + 1))
    echo "=== [$STEP/$TOTAL] capture/$tag ===" | tee -a logs/capture.log
    python capture_aime.py "$@" --output-dir "$OUTDIR" --problem-idx 0 \
        2>&1 | tee "logs/capture_${tag}.log"
    echo "=== [$STEP/$TOTAL] capture/$tag DONE ===" | tee -a logs/capture.log
}

# [HERE-DONE]
# run_capture llama_3b     --model meta-llama/Llama-3.2-3B-Instruct          --max-tokens 40000  # weights~5GB  + KV@40k~4.6GB  = ~11GB
# [HERE-DONE]
# run_capture qwen_7b      --model Qwen/Qwen2.5-7B-Instruct                  --max-tokens 40000  # weights~14GB + KV@40k~2.3GB  = ~17GB
# run_capture deepseek_14b --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B  --max-tokens 40000  # weights~28GB + KV@40k~7.9GB  = ~37GB (needs A100 40GB)
# run_capture qwen_14b     --model Qwen/Qwen2.5-14B-Instruct                 --max-tokens 40000  # weights~28GB + KV@40k~7.9GB  = ~37GB (needs A100 40GB)

echo ""
echo "ALL CAPTURES DONE" | tee -a logs/capture.log
echo "Files in $OUTDIR/:"
ls -lh "$OUTDIR/"
