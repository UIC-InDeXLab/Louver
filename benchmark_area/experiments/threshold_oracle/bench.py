"""
Threshold Oracle Ablation — Experiment 7.

Uses latency captures (saved QKV tensors) to measure for each threshold oracle:
  - fraction of keys retrieved per decode step (mean ± std)
  - precision@retrieved: fraction of retrieved keys that are in exact top-TOP_FRAC (mean ± std)
  - timeseries: per-step score-distribution oscillation metrics + oracle tau traces

No full model inference needed — operates on saved QKV tensors only.

Usage:
    python bench.py [--captures-dir DIR] [--n-steps 200] [--n-steps-ts 2000]
                    [--sample-size 256] [--top-frac 0.10] [--output-dir results/]
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

_TOPK_VALS = (8, 32, 128)


def _score_dist_metrics(exact: torch.Tensor) -> dict:
    """
    exact: (H_q, N_total) float32 raw dot-product scores.
    Returns dict matching tail_metrics.py column names (mean across heads):
      topk_mass_{8,32,128}_mean  — fraction of softmax weight in top-k tokens
      cov50_weight_mean          — fraction of keys to cover 50% softmax mass
      cov50_score_mean           — fraction of keys to cover 50% shifted-score mass
    """
    H_q, N = exact.shape
    probs = torch.softmax(exact, dim=-1)  # (H_q, N)

    # topk_mass
    topk_masses = {}
    for k in _TOPK_VALS:
        topk_masses[k] = float(
            probs.topk(min(k, N), dim=-1).values.sum(-1).mean().item()
        )

    # cov50_weight: softmax-based (matches tail_metrics.py cov50_weight)
    sorted_p = probs.sort(dim=-1, descending=True).values
    k50_w = (sorted_p.cumsum(-1) < 0.5).sum(-1).float() + 1  # (H_q,)
    cov50_weight = float((k50_w / N).mean().item())

    # cov50_score: shifted-score mass (matches tail_metrics.py cov50_score)
    s_min = exact.min(dim=-1, keepdim=True).values
    shifted = (exact - s_min).clamp(min=0)
    mass_s = shifted / shifted.sum(-1, keepdim=True).clamp_min(1e-12)
    sorted_s = mass_s.sort(dim=-1, descending=True).values
    k50_s = (sorted_s.cumsum(-1) < 0.5).sum(-1).float() + 1  # (H_q,)
    cov50_score = float((k50_s / N).mean().item())

    return {
        **{f"topk_mass_{k}_mean": topk_masses[k] for k in _TOPK_VALS},
        "cov50_weight_mean": cov50_weight,
        "cov50_score_mean":  cov50_score,
    }


def _load_capture(capture_path: Path):
    cap = torch.load(capture_path, map_location="cpu", weights_only=False)
    layer_idx     = list(cap["prefill_keys"].keys())[0]
    pre_keys_f16  = cap["prefill_keys"][layer_idx]      # (H_kv, N_pre, D)
    gen_keys_list = cap["generated_keys"][layer_idx]    # list[(H_kv, D)]
    gen_q_list    = cap["generated_queries"][layer_idx] # list[(H_q, D)]
    H_kv, N_pre, D = pre_keys_f16.shape
    H_q  = gen_q_list[0].shape[0]
    N_gen = len(gen_keys_list)
    g    = H_q // H_kv
    gen_keys_stacked = torch.stack(gen_keys_list).permute(1, 0, 2).float()
    all_keys_f32 = torch.cat([pre_keys_f16.float(), gen_keys_stacked], dim=1)
    all_keys_f16 = all_keys_f32.half()
    return all_keys_f32, all_keys_f16, gen_q_list, H_kv, H_q, N_pre, N_gen, g


def analyze_capture(
    capture_path: Path,
    n_steps: int,
    n_steps_ts: int,
    sample_size: int,
    recall_frac: float,
    seed: int = 42,
) -> tuple[dict[str, dict], list[dict]]:
    """
    Returns: (summary, timeseries)
      summary    : oracle_name → {frac_mean, frac_std, precision_mean, precision_std, n_obs}
      timeseries : dense per-step rows with score-dist oscillation metrics + oracle tau means
    """
    torch.manual_seed(seed)

    all_keys_f32, all_keys_f16, gen_q_list, H_kv, H_q, N_pre, N_gen, g = \
        _load_capture(capture_path)

    min_step = max(50, N_gen // 20)

    # ── Pass 1: summary stats (sparse, 200 steps) ─────────────────────────────
    step_indices = np.linspace(min_step, N_gen - 1, n_steps, dtype=int)
    data: dict[str, dict[str, list]] = {
        name: {"fracs": [], "precisions": []} for name in ORACLE_NAMES
    }

    for step in tqdm(step_indices, desc=f"{capture_path.stem} [summary]",
                     leave=True, dynamic_ncols=True):
        N_total  = N_pre + int(step)
        keys_f32 = all_keys_f32[:, :N_total, :]
        keys_f16 = all_keys_f16[:, :N_total, :]
        q_f32    = gen_q_list[step].float()
        q_f16    = q_f32.half()

        M   = min(sample_size, N_total)
        idx = torch.randperm(N_total)[:M]
        sample_f16 = keys_f16[:, idx, :]

        exact = torch.empty(H_q, N_total)
        for h_q in range(H_q):
            exact[h_q] = keys_f32[h_q // g] @ q_f32[h_q]

        k_top    = max(1, int(recall_frac * N_total))
        topk_idx = exact.topk(k_top, dim=-1).indices
        topk_sets = [set(topk_idx[h].tolist()) for h in range(H_q)]

        taus = {name: _compute_tau(sample_f16, q_f16, sample_size, kw)
                for name, kw in ORACLES}

        for name in ORACLE_NAMES:
            tau = taus[name]
            for h_q in range(H_q):
                mask = exact[h_q] >= tau[h_q].float()
                frac = mask.float().mean().item()
                retrieved_set = set(mask.nonzero(as_tuple=True)[0].tolist())
                n_ret = len(retrieved_set)
                precision = (len(topk_sets[h_q] & retrieved_set) / n_ret
                             if n_ret > 0 else 0.0)
                data[name]["fracs"].append(frac)
                data[name]["precisions"].append(precision)

    summary = {}
    for name, d in data.items():
        fracs      = np.array(d["fracs"])
        precisions = np.array(d["precisions"])
        summary[name] = {
            "frac_mean":      float(fracs.mean()),
            "frac_std":       float(fracs.std()),
            "precision_mean": float(precisions.mean()),
            "precision_std":  float(precisions.std()),
            "n_obs":          len(fracs),
        }

    # ── Pass 2: timeseries (dense, n_steps_ts steps) ──────────────────────────
    ts_indices = np.linspace(min_step, N_gen - 1, n_steps_ts, dtype=int)
    timeseries: list[dict] = []

    for step in tqdm(ts_indices, desc=f"{capture_path.stem} [timeseries]",
                     leave=True, dynamic_ncols=True):
        N_total  = N_pre + int(step)
        keys_f32 = all_keys_f32[:, :N_total, :]
        keys_f16 = all_keys_f16[:, :N_total, :]
        q_f32    = gen_q_list[step].float()
        q_f16    = q_f32.half()

        exact = torch.empty(H_q, N_total)
        for h_q in range(H_q):
            exact[h_q] = keys_f32[h_q // g] @ q_f32[h_q]

        M   = min(sample_size, N_total)
        idx = torch.randperm(N_total)[:M]
        sample_f16 = keys_f16[:, idx, :]

        taus = {name: _compute_tau(sample_f16, q_f16, sample_size, kw)
                for name, kw in ORACLES}

        ts_row: dict = {"step": int(step), "N_total": N_total}
        ts_row.update(_score_dist_metrics(exact))
        for name in ORACLE_NAMES:
            ts_row[f"tau_{name}_mean"] = float(taus[name].mean().item())
        timeseries.append(ts_row)

    return summary, timeseries


# ── CSV / print helpers ───────────────────────────────────────────────────────

def print_table(model_tag: str, summary: dict[str, dict], top_frac: float) -> None:
    top_pct = int(top_frac * 100)
    header = (f"{'Oracle':<18}  {'Frac %':>8}  {'±':>5}  "
              f"{'Prec@top{:d}%'.format(top_pct):>13}  {'±':>5}")
    print(f"\n── {model_tag} ──")
    print(header)
    print("-" * len(header))
    for name in ORACLE_NAMES:
        s = summary[name]
        print(f"{name:<18}  {s['frac_mean']*100:>7.2f}%  {s['frac_std']*100:>4.2f}  "
              f"{s['precision_mean']*100:>12.2f}%  {s['precision_std']*100:>4.2f}")


def write_timeseries_csv(out_path: Path, timeseries: list[dict]) -> None:
    if not timeseries:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(timeseries[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in timeseries:
            writer.writerow({
                k: f"{v:.8f}" if isinstance(v, float) else v
                for k, v in row.items()
            })


def write_csv(out_path: Path, model_tag: str, summary: dict[str, dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "oracle",
                         "frac_mean", "frac_std",
                         "precision_mean", "precision_std", "n_obs"])
        for name in ORACLE_NAMES:
            s = summary[name]
            writer.writerow([model_tag, name,
                             f"{s['frac_mean']:.6f}", f"{s['frac_std']:.6f}",
                             f"{s['precision_mean']:.6f}", f"{s['precision_std']:.6f}",
                             s["n_obs"]])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures-dir", default=str(CAPTURES_DIR),
                        help="Directory containing .pt capture files")
    parser.add_argument("--n-steps",    type=int,   default=200,
                        help="Decode steps to sample per capture (summary stats)")
    parser.add_argument("--n-steps-ts", type=int,  default=2000,
                        help="Decode steps for dense timeseries (oscillation plot)")
    parser.add_argument("--sample-size", type=int,  default=256,
                        help="Reservoir sample size for threshold estimation")
    parser.add_argument("--top-frac", type=float, default=0.10,
                        help="Fraction of top keys used as precision target (default 10%%)")
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

        summary, timeseries = analyze_capture(
            cap_path,
            n_steps=args.n_steps,
            n_steps_ts=args.n_steps_ts,
            sample_size=args.sample_size,
            recall_frac=args.top_frac,
            seed=args.seed,
        )
        all_summaries[model_tag] = summary

        print_table(model_tag, summary, args.top_frac)

        csv_path = output_dir / f"{model_tag}_threshold_oracle.csv"
        write_csv(csv_path, model_tag, summary)
        print(f"  → {csv_path}")

        ts_path = output_dir / f"{model_tag}_timeseries.csv"
        write_timeseries_csv(ts_path, timeseries)
        print(f"  → {ts_path}")

    # Combined JSON
    json_path = output_dir / "threshold_oracle_all.json"
    with open(json_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAll results → {json_path}")


if __name__ == "__main__":
    main()
