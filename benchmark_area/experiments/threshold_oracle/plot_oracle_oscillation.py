"""
Plot oracle threshold oscillation vs. cov50_score (score-distribution signal).

For each model's timeseries CSV, normalizes every column via min-max and
overlays oracle tau traces on top of cov50_score_mean so oscillation
alignment is visible.

Usage:
    python plot_oracle_oscillation.py --results-dir results/ [--out-dir results/figs]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
})

# Which oracles to overlay (a representative subset — not all 9)
PLOT_ORACLES = [
    ("sample_max",      "#d62728", 1.5, "-"),
    ("sample_topk_5",   "#1f77b4", 1.5, "-"),
    ("sample_mean_max", "#2ca02c", 1.5, "--"),
    ("sample_gap",      "#9467bd", 1.5, "--"),
    ("budget_f10",      "#ff7f0e", 1.5, ":"),
]


def _load_timeseries(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    if not rows:
        return {}
    keys = list(rows[0].keys())
    return {k: np.array([r[k] for r in rows]) for k in keys}


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _smooth(arr: np.ndarray, w: int = 5) -> np.ndarray:
    """Simple uniform moving-average smoothing."""
    if w <= 1:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def plot_model(ts: dict[str, np.ndarray], model_tag: str, out_path: Path,
               smooth_w: int = 7) -> None:
    if "cov50_weight_mean" not in ts:
        print(f"  skip {model_tag}: no cov50_weight_mean column")
        return

    steps = ts["step"].astype(int)
    cov50 = _smooth(ts["cov50_weight_mean"], smooth_w)

    fig, ax_l = plt.subplots(figsize=(13, 4.2))
    ax_r = ax_l.twinx()

    # ── Left axis: cov50_weight_mean (raw, no normalization) ──────────────────
    ax_l.fill_between(steps, cov50, alpha=0.10, color="#333333")
    ax_l.plot(steps, cov50, color="#333333", linewidth=2.4,
              label="cov50_weight (osc. signal)", zorder=5)
    ax_l.set_ylabel("cov50_weight_mean  (fraction of tokens)", color="#333333")
    ax_l.tick_params(axis="y", labelcolor="#333333")

    # ── Right axis: oracle τ (raw score units, separate scale) ────────────────
    lines_r = []
    for name, color, lw, ls in PLOT_ORACLES:
        col = f"tau_{name}_mean"
        if col not in ts:
            continue
        y = _smooth(ts[col], smooth_w)
        line, = ax_r.plot(steps, y, color=color, linewidth=lw, linestyle=ls,
                          label=name, alpha=0.85)
        lines_r.append(line)

    ax_r.set_ylabel("oracle threshold τ  (raw score)", color="#555555")
    ax_r.tick_params(axis="y", labelcolor="#555555")

    # Combined legend
    h_l, lab_l = ax_l.get_legend_handles_labels()
    h_r, lab_r = ax_r.get_legend_handles_labels()
    ax_l.legend(h_l + h_r, lab_l + lab_r, loc="upper left", ncol=3, fontsize=10)

    ax_l.set_xlabel("decode step (generated token index)")
    ax_l.set_title(f"{model_tag}  —  score-distribution oscillation vs oracle τ")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path}")


def _rolling_median(arr: np.ndarray, w: int) -> np.ndarray:
    out = np.empty_like(arr)
    half = w // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out[i] = np.median(arr[lo:hi])
    return out


def plot_cov50_over_time(ts: dict[str, np.ndarray], model_tag: str,
                         out_path: Path) -> None:
    if "cov50_weight_mean" not in ts:
        print(f"  skip {model_tag}: no cov50_weight_mean")
        return

    xs = ts["N_total"].astype(int)
    ys = ts["cov50_weight_mean"] * ts["N_total"]   # k = (k/T) * T

    mask = xs <= 8000
    xs, ys = xs[mask], ys[mask]

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(xs, ys, color="#475569", linewidth=0.9, alpha=0.90)
    ax.set_ylabel("top-k 50% coverage (k, #tokens)")
    ax.set_xlabel("token ID in context")
    ax.set_title(model_tag.replace("_", " "))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results",
                    help="Directory containing *_timeseries.csv files")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory for PNGs (default: results-dir/figs)")
    ap.add_argument("--smooth", type=int, default=7,
                    help="Moving-average window for smoothing (1=off)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_files = sorted(results_dir.glob("*timeseries.csv"))
    if not ts_files:
        print(f"No *_timeseries.csv found in {results_dir}")
        return

    for ts_path in ts_files:
        model_tag = ts_path.stem.replace("_online_timeseries", "").replace("_timeseries", "")
        ts = _load_timeseries(ts_path)
        if not ts:
            print(f"  skip empty: {ts_path.name}")
            continue

        plot_cov50_over_time(ts, model_tag,
                             out_dir / f"{model_tag}_cov50_over_time.png")
        plot_model(ts, model_tag,
                   out_dir / f"{model_tag}_oracle_oscillation.png",
                   smooth_w=args.smooth)

    print("done.")


if __name__ == "__main__":
    main()
