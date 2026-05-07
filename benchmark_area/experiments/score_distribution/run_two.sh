#!/bin/bash
set -e
cd "$(dirname "$0")"
python capture.py --model Qwen/Qwen2.5-7B-Instruct --max_new_tokens 2000 \
    --out snapshots/snap_qwen2k_v2.pt
python capture.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --max_new_tokens 2000 \
    --out snapshots/snap_dsr14b_2k.pt
