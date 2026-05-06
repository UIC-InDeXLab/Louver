#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
DEEPSEEK_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
LB_OUTDIR="results/longbench_v2"
RULER_OUTDIR="results/ruler"
AIME_OUTDIR="results/aime"
MATH_OUTDIR="results/math500"

python - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    print("ERROR: no CUDA GPU visible", file=sys.stderr); sys.exit(1)
print(f"GPU: {torch.cuda.get_device_name(0)}  free={torch.cuda.mem_get_info()[0]//1024**3}GB")
EOF

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
GPU_ARCH=$(python - <<'PYEOF'
import torch
maj, min_ = torch.cuda.get_device_capability()
print(f"{maj}.{min_}")
PYEOF
)
export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
export TORCH_EXTENSIONS_DIR="${TMPDIR:-/tmp}/torch_ext_${GPU_ARCH//./_}_$$"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_cache_$$"

mkdir -p logs "$LB_OUTDIR" "$RULER_OUTDIR" "$AIME_OUTDIR" "$MATH_OUTDIR"

MAX_SAMPLES="${MAX_SAMPLES:-20}"
TASKS="${TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique}"
RULER_TASKS="${RULER_TASKS:-niah_single,niah_multi,vt}"
RULER_N="${RULER_N:-20}"
SEQ_LEN="${SEQ_LEN:-32768}"
TOTAL=11
STEP=0

run() {
    local tag="$1"; shift
    STEP=$((STEP + 1))
    echo "=== [$STEP/$TOTAL] $tag ===" | tee -a logs/summary.log
    python eval/longbench.py "$@" --model "$MODEL" --output_dir "$LB_OUTDIR" \
        --max_input_length 32768 --max_samples "$MAX_SAMPLES" --tasks "$TASKS" \
        2>&1 | tee logs/lb_${tag}.log
    echo "=== [$STEP/$TOTAL] $tag DONE ===" | tee -a logs/summary.log
}

run_baseline() {
    local tag="$1"; shift
    STEP=$((STEP + 1))
    echo "=== [$STEP/$TOTAL] $tag ===" | tee -a logs/summary.log
    python eval/longbench_baselines.py "$@" --model "$MODEL" --output_dir "$LB_OUTDIR" \
        --max_input_length 32768 --max_samples "$MAX_SAMPLES" --tasks "$TASKS" \
        2>&1 | tee logs/lb_${tag}.log
    echo "=== [$STEP/$TOTAL] $tag DONE ===" | tee -a logs/summary.log
}

run_ruler() {
    local tag="$1"; shift
    STEP=$((STEP + 1))
    echo "=== [$STEP/$TOTAL] ruler/$tag ===" | tee -a logs/summary.log
    python eval/ruler.py "$@" --model "$MODEL" --output_dir "$RULER_OUTDIR" \
        --tasks "$RULER_TASKS" --n_samples "$RULER_N" --seq_len "$SEQ_LEN" \
        2>&1 | tee logs/ruler_${tag}.log
    echo "=== [$STEP/$TOTAL] ruler/$tag DONE ===" | tee -a logs/summary.log
}

run_aime() {
    local tag="$1"; shift
    STEP=$((STEP + 1))
    echo "=== [$STEP/$TOTAL] aime/$tag ===" | tee -a logs/summary.log
    python eval/aime.py "$@" --model "$DEEPSEEK_MODEL" --output_dir "$AIME_OUTDIR" \
        2>&1 | tee logs/aime_${tag}.log
    echo "=== [$STEP/$TOTAL] aime/$tag DONE ===" | tee -a logs/summary.log
}

run_math() {
    local tag="$1"; shift
    STEP=$((STEP + 1))
    echo "=== [$STEP/$TOTAL] math500/$tag ===" | tee -a logs/summary.log
    python eval/math500.py "$@" --model "$DEEPSEEK_MODEL" --output_dir "$MATH_OUTDIR" \
        2>&1 | tee logs/math_${tag}.log
    echo "=== [$STEP/$TOTAL] math500/$tag DONE ===" | tee -a logs/summary.log
}

# ── LongBench ────────────────────────────────────────────────────────────────
# run louver_ta_oracle        --method louver_ta    --threshold_mode oracle
# run dense_sdpa              --method dense_sdpa
# run louver_ta_budget        --method louver_ta    --threshold_mode budget --budget_fraction 0.1
# run_baseline h2o_f0.1           --method h2o          --budget_fraction 0.1
# run_baseline quest_f0.1         --method quest        --budget_fraction 0.1
# run_baseline streaming_llm_f0.1 --method streaming_llm --budget_fraction 0.1
# run_baseline twilight_p0.85     --method twilight --top_p 0.85

# ── RULER ────────────────────────────────────────────────────────────────────
# run_ruler dense              --method dense_sdpa
# run_ruler louver_ta_budget   --method louver_ta   --threshold_mode budget --budget_fraction 0.1
# run_ruler h2o_f0.1           --method h2o         --budget_fraction 0.1
# run_ruler quest_f0.1         --method quest        --budget_fraction 0.1
# run_ruler streaming_llm_f0.1 --method streaming_llm --budget_fraction 0.1
# run_ruler twilight_p0.85     --method twilight     --top_p 0.85

# ── Reasoning (DeepSeek-R1-Distill-Llama-8B) ─────────────────────────────────
# run_aime dense_sdpa                --method dense_sdpa
# run_aime louver_ta_top_p0.85       --method louver_ta  --oracle sample_top_p --louver_top_p 0.85 --max_problems 5
# run_aime twilight_p0.85            --method twilight   --top_p 0.85 --max_problems 5
# run_aime h2o_b512                  --method h2o        --budget_tokens 512

# run_math dense_sdpa                --method dense_sdpa               --max_samples 50
# run_math louver_ta_top_p0.85       --method louver_ta  --oracle sample_top_p --louver_top_p 0.85 --max_samples 20
# run_math twilight_p0.85            --method twilight   --top_p 0.85 --max_samples 20
run_math h2o_b512                  --method h2o        --budget_tokens 512 --max_samples 20

echo "ALL DONE" | tee -a logs/summary.log
