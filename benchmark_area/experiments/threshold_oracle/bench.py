"""
Threshold Oracle Ablation — Experiment 8.

Uses latency captures (saved QKV tensors) to measure for each threshold oracle:
  - fraction of keys retrieved per decode step (mean ± std)
  - recall vs exact top-RECALL_FRAC tokens (mean ± std)

No full model inference needed — operates on saved QKV tensors only.

Usage:
    python bench.py [--captures GLOB] [--n-steps 200] [--sample-size 256]
                    [--recall-frac 0.10] [--output-dir results/]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from threshold import LouverThreshold

CAPTURES_DIR = Path(__file__).resolve().parents[1] / "latency" / "captures"
RESULTS_DIR  = Path(__file__).resolve().parent  / "results"

# ── Oracle configurations ─────────────────────────────────────────────────────

ORACLES: list[tuple[str, dict]] = [
    ("sample_max",      dict(mode="oracle", oracle="sample_max")),
    ("sample_topk_2",   dict(mode="oracle", oracle="sample_topk", topk_k=2)),
    ("sample_topk_5",   dict(mode="oracle", oracle="sample_topk", topk_k=5)),
    ("sample_topk_10",  dict(mode="oracle", oracle="sample_topk", topk_k=10)),
    ("sample_mean_max", dict(mode="oracle", oracle="sample_mean_max")),
    ("sample_gap",      dict(mode="oracle", oracle="sample_gap")),
    ("budget_f05",      dict(mode="budget", budget_fraction=0.05)),
    ("budget_f10",      dict(mode="budget", budget_fraction=0.10)),
    ("budget_f15",      dict(mode="budget", budget_fraction=0.15)),
]

ORACLE_NAMES = [name for name, _ in ORACLES]


# ── Threshold helper ──────────────────────────────────────────────────────────

def _compute_tau(sample_f16: torch.Tensor, q_f16: torch.Tensor,
                 sample_size: int, oracle_kwargs: dict) -> torch.Tensor:
    """
    sample_f16: (H_kv, M, D) fp16
    q_f16:      (H_q, D)     fp16
    Returns:    (H_q,)       float32 threshold.
    """
    thresh = LouverThreshold(sample_size=sample_size, **oracle_kwargs)
    thresh.sample  = sample_f16
    thresh._filled = sample_f16.shape[1]
    thresh._N      = 0
    return thresh.get_threshold_ta(q_f16)


# ── Per-capture analysis ──────────────────────────────────────────────────────

def analyze_capture(
    capture_path: Path,
    n_steps: int,
    sample_size: int,
    recall_frac: float,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Returns: oracle_name → {frac_mean, frac_std, recall_mean, recall_std, n_obs}
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    cap = torch.load(capture_path, map_location="cpu", weights_only=False)

    layer_idx     = list(cap["prefill_keys"].keys())[0]
    pre_keys_f16  = cap["prefill_keys"][layer_idx]          # (H_kv, N_pre, D) fp16
    gen_keys_list = cap["generated_keys"][layer_idx]        # list[(H_kv, D)]
    gen_q_list    = cap["generated_queries"][layer_idx]     # list[(H_q, D)]

    H_kv, N_pre, D = pre_keys_f16.shape
    H_q = gen_q_list[0].shape[0]
    N_gen = len(gen_keys_list)
    g = H_q // H_kv

    # Build full key tensor (H_kv, N_pre + N_gen, D) float32 for exact scoring
    gen_keys_stacked = torch.stack(gen_keys_list).permute(1, 0, 2).float()  # (H_kv, N_gen, D)
    all_keys_f32 = torch.cat([pre_keys_f16.float(), gen_keys_stacked], dim=1)
    all_keys_f16 = all_keys_f32.half()

    # Evenly-spaced sample steps — skip the first 5% of generated tokens
    min_step = max(50, N_gen // 20)
    step_indices = np.linspace(min_step, N_gen - 1, n_steps, dtype=int)

    # Accumulate per oracle
    data: dict[str, dict[str, list]] = {
        name: {"fracs": [], "recalls": []} for name in ORACLE_NAMES
    }

    for step in tqdm(step_indices, desc=capture_path.stem, leave=True, dynamic_ncols=True):
        N_total = N_pre + int(step)
        keys_f32 = all_keys_f32[:, :N_total, :]  # (H_kv, N_total, D)
        keys_f16 = all_keys_f16[:, :N_total, :]

        q_f32 = gen_q_list[step].float()   # (H_q, D)
        q_f16 = q_f32.half()

        # Reservoir sample of keys (per head, same indices for all heads)
        M = min(sample_size, N_total)
        idx = torch.randperm(N_total)[:M]
        sample_f16 = keys_f16[:, idx, :]   # (H_kv, M, D)

        # Exact scores: q[h_q] · keys[h_kv]   shape (H_q, N_total)
        # Process head-by-head to avoid huge intermediate tensor
        exact = torch.empty(H_q, N_total)
        for h_q in range(H_q):
            h_kv = h_q // g
            exact[h_q] = keys_f32[h_kv] @ q_f32[h_q]  # (N_total,)

        # Ground-truth top-recall_frac indices per H_q head
        k_recall = max(1, int(recall_frac * N_total))
        topk_idx = exact.topk(k_recall, dim=-1).indices  # (H_q, k_recall)
        topk_sets = [set(topk_idx[h].tolist()) for h in range(H_q)]

        for name, oracle_kwargs in ORACLES:
            tau = _compute_tau(sample_f16, q_f16, sample_size, oracle_kwargs)  # (H_q,)

            for h_q in range(H_q):
                mask = exact[h_q] >= tau[h_q].float()
                frac = mask.float().mean().item()
                retrieved_set = set(mask.nonzero(as_tuple=True)[0].tolist())
                recall = len(topk_sets[h_q] & retrieved_set) / k_recall

                data[name]["fracs"].append(frac)
                data[name]["recalls"].append(recall)

    summary = {}
    for name, d in data.items():
        fracs   = np.array(d["fracs"])
        recalls = np.array(d["recalls"])
        summary[name] = {
            "frac_mean":   float(fracs.mean()),
            "frac_std":    float(fracs.std()),
            "recall_mean": float(recalls.mean()),
            "recall_std":  float(recalls.std()),
            "n_obs":       len(fracs),
        }
    return summary


# ── CSV / print helpers ───────────────────────────────────────────────────────

def print_table(model_tag: str, summary: dict[str, dict], recall_frac: float) -> None:
    recall_pct = int(recall_frac * 100)
    header = (f"{'Oracle':<18}  {'Frac %':>9}  {'±':>6}  "
              f"{'Recall@{:d}%'.format(recall_pct):>12}  {'±':>6}")
    print(f"\n── {model_tag} ──")
    print(header)
    print("-" * len(header))
    for name in ORACLE_NAMES:
        s = summary[name]
        print(f"{name:<18}  {s['frac_mean']*100:>8.2f}%  {s['frac_std']*100:>5.2f}  "
              f"{s['recall_mean']*100:>11.2f}%  {s['recall_std']*100:>5.2f}")


def write_csv(out_path: Path, model_tag: str, summary: dict[str, dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "oracle",
                         "frac_mean", "frac_std",
                         "recall_mean", "recall_std", "n_obs"])
        for name in ORACLE_NAMES:
            s = summary[name]
            writer.writerow([model_tag, name,
                             f"{s['frac_mean']:.6f}", f"{s['frac_std']:.6f}",
                             f"{s['recall_mean']:.6f}", f"{s['recall_std']:.6f}",
                             s["n_obs"]])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures-dir", default=str(CAPTURES_DIR),
                        help="Directory containing .pt capture files")
    parser.add_argument("--n-steps",    type=int,   default=200,
                        help="Decode steps to sample per capture")
    parser.add_argument("--sample-size", type=int,  default=256,
                        help="Reservoir sample size for threshold estimation")
    parser.add_argument("--recall-frac", type=float, default=0.10,
                        help="Fraction of top keys used as recall target (default 10%%)")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    captures_dir = Path(args.captures_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture_files = sorted(captures_dir.glob("*.pt"))
    if not capture_files:
        print(f"No .pt files found in {captures_dir}")
        return

    all_summaries = {}
    for cap_path in capture_files:
        model_tag = cap_path.stem
        print(f"\n=== {model_tag} ===")

        summary = analyze_capture(
            cap_path,
            n_steps=args.n_steps,
            sample_size=args.sample_size,
            recall_frac=args.recall_frac,
            seed=args.seed,
        )
        all_summaries[model_tag] = summary

        print_table(model_tag, summary, args.recall_frac)

        csv_path = output_dir / f"{model_tag}_threshold_oracle.csv"
        write_csv(csv_path, model_tag, summary)
        print(f"  → {csv_path}")

    # Combined JSON
    json_path = output_dir / "threshold_oracle_all.json"
    with open(json_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAll results → {json_path}")


if __name__ == "__main__":
    main()
