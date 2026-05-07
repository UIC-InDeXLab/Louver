#!/bin/bash
#SBATCH --job-name=louver_longbench
#SBATCH --account=YOUR_ACCOUNT      # replace with your Delta allocation
#SBATCH --partition=gpuA100x4       # 4×A100-40GB node; use gpuA100x8 for 80GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/lb_%j_%x.out
#SBATCH --error=logs/lb_%j_%x.err

set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/longbench_v2

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11  # adjust to what Delta has
# If you have a venv:
source "$HOME/venv_louver/bin/activate"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_$SLURM_JOB_ID
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOB_ID

MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUTDIR="results/longbench_v2"

run() {
    local tag="$1"; shift
    echo "=== $tag ===" | tee -a logs/lb_summary_$SLURM_JOB_ID.log
    python eval/longbench.py "$@" --model "$MODEL" --output_dir "$OUTDIR" \
        2>&1 | tee logs/lb_${tag}_$SLURM_JOB_ID.log
    echo "=== $tag DONE ===" | tee -a logs/lb_summary_$SLURM_JOB_ID.log
}

run dense_sdpa         --method dense_sdpa
run louver_ta_oracle   --method louver_ta   --threshold_mode oracle
run louver_ta_budget   --method louver_ta   --threshold_mode budget --budget_fraction 0.1
run louver_full_oracle --method louver_full --threshold_mode oracle
run louver_full_budget --method louver_full --threshold_mode budget --budget_fraction 0.1

echo "ALL DONE"
