"""
Measure Twilight effective retrieval fraction on captured keys.
Compares:
  - Our baseline impl: fixed (1-top_p)*N fraction
  - Real Twilight: cumulative softmax top-p (variable fraction)
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
from hira.benchmark_area.quick_pruning.pruning_bench_utils import CaptureState

CAPTURE = Path(__file__).resolve().parents[2] / "quick_pruning" / \
    "capture_qkv_12000_meta-llama_Llama-3.2-3B-Instruct.pt"

TOP_P = 0.85
LAYERS = [5, 10, 15, 20]
N_STEPS = 500


def real_top_p_frac(scores: torch.Tensor, top_p: float) -> float:
    """scores: (H_q, N). Returns avg fraction retrieved by cumulative softmax top-p."""
    probs = F.softmax(scores, dim=-1)          # (H_q, N)
    sorted_p, _ = probs.sort(dim=-1, descending=True)
    cumsum = sorted_p.cumsum(dim=-1)           # (H_q, N)
    # keep tokens until cumsum >= top_p
    keep = (cumsum - sorted_p) < top_p        # shift by one: include the token that crosses
    fracs = keep.float().mean(dim=-1)          # (H_q,)
    return fracs.mean().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cap = CaptureState.load(CAPTURE)
    layers = [l for l in LAYERS if l in cap.layer_ids()]
    available = cap.generated_token_count()
    stride = max(1, available // N_STEPS)
    steps = list(range(0, available, stride))[:N_STEPS]

    baseline_fracs, real_fracs = [], []

    for layer in layers:
        prefill_k = cap.prefill_keys[layer].to(device=device, dtype=torch.float32)
        gen_q = cap.generated_queries[layer]
        gen_k = cap.generated_keys[layer]
        H_kv = prefill_k.shape[0]
        gen_k_gpu = torch.stack(gen_k, dim=1).to(device=device, dtype=torch.float32)

        for step in steps:
            if step >= len(gen_q):
                break
            q = gen_q[step].to(device=device, dtype=torch.float32)  # (H_q, D)
            H_q = q.shape[0]
            q_head_to_kv = torch.arange(H_q, device=device) // (H_q // H_kv)

            if step > 0:
                all_keys = torch.cat([prefill_k, gen_k_gpu[:, :step, :]], dim=1)
            else:
                all_keys = prefill_k
            N = all_keys.shape[1]

            k_for_q = all_keys[q_head_to_kv]          # (H_q, N, D)
            scores = torch.einsum("hd,hnd->hn", q, k_for_q)  # (H_q, N)

            # Our baseline: fixed fraction
            baseline_fracs.append(1.0 - TOP_P)

            # Real Twilight: cumulative softmax top-p
            real_fracs.append(real_top_p_frac(scores, TOP_P))

    baseline_avg = sum(baseline_fracs) / len(baseline_fracs)
    real_avg = sum(real_fracs) / len(real_fracs)
    real_min = min(real_fracs)
    real_max = max(real_fracs)

    print(f"\ntop_p={TOP_P}  steps={len(steps)}  layers={layers}")
    print(f"\nOur baseline impl:  fixed {baseline_avg*100:.1f}% every step")
    print(f"Real Twilight top-p: avg={real_avg*100:.1f}%  min={real_min*100:.1f}%  max={real_max*100:.1f}%")
    print(f"\n→ For Louver to match real Twilight: budget_fraction={real_avg:.3f}, sample_size=512")
    print(f"→ For Louver to match our baseline:  budget_fraction={baseline_avg:.3f}, sample_size=512")


if __name__ == "__main__":
    main()
