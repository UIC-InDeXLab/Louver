#!/usr/bin/env python3
"""
Plot fullbuild_results.csv — heatmaps of search_ms, mean_frac, build_ms
for CPU and CUDA indexers across (num_levels × branching_factor).

Usage:
    python plot_fullbuild.py [--input fullbuild_results.csv] [--save fullbuild_plots.png]
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


# ── helpers ────────────────────────────────────────────────────────────
def _total_nodes(s):
    try:
        return sum(ast.literal_eval(s))
    except Exception:
        return float("nan")


def _annotate_heatmap(ax, vals, plot_vals, fmt_fn):
    """Write values in each cell, white on dark / black on light."""
    vmin, vmax = np.nanmin(plot_vals), np.nanmax(plot_vals)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="grey")
                continue
            norm_v = (plot_vals[i, j] - vmin) / (vmax - vmin + 1e-12)
            ax.text(
                j,
                i,
                fmt_fn(v),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if norm_v > 0.55 else "black",
            )


# ── main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="fullbuild_results.csv")
    parser.add_argument(
        "--save",
        type=str,
        default="fullbuild_plots.png",
        help="Save figure to file instead of showing interactively",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    for c in [
        "num_levels",
        "branching_factor",
        "mean_frac",
        "min_frac",
        "max_frac",
        "search_ms",
        "build_ms",
        "actual_depth",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["indexer"] = df["indexer"].str.lower()
    df["total_nodes"] = df["level_sizes"].apply(_total_nodes)

    # Drop rows with errors
    df = df[df["error"].isna() | (df["error"] == "")]
    df = df.dropna(subset=["search_ms", "mean_frac"])

    INDEXERS = sorted(df["indexer"].unique().tolist())
    BF_ORDER = sorted(df["branching_factor"].dropna().unique().astype(int).tolist())

    METRICS = [
        (
            "mean_frac",
            "Fraction scanned\n(lower = better pruning)",
            lambda v: f"{v:.4f}",
        ),
        ("search_ms", "Search latency (ms)\n(lower = faster)", lambda v: f"{v:.3f}"),
        ("build_ms", "Build time (ms)\n(context only)", lambda v: f"{v:.1f}"),
    ]

    n_metrics = len(METRICS)
    n_indexers = len(INDEXERS)

    fig, axes = plt.subplots(
        nrows=n_metrics,
        ncols=n_indexers,
        figsize=(6 * n_indexers + 2, 4 * n_metrics),
        gridspec_kw={"hspace": 0.55, "wspace": 0.45},
        squeeze=False,
    )

    for row, (metric, title_suffix, fmt_fn) in enumerate(METRICS):
        for col, indexer in enumerate(INDEXERS):
            ax = axes[row, col]
            sub = df[df["indexer"] == indexer]
            if sub.empty:
                ax.set_visible(False)
                continue

            # Pivot: rows = num_levels, columns = branching_factor
            piv = (
                sub.groupby(["num_levels", "branching_factor"])[metric]
                .mean()
                .unstack("branching_factor")
                .reindex(columns=BF_ORDER)
            )
            piv = piv.sort_index()

            vals = piv.values.astype(float)
            plot_vals = vals.copy()

            im = ax.imshow(
                plot_vals,
                aspect="auto",
                cmap="RdYlGn_r",
                interpolation="nearest",
                vmin=np.nanmin(plot_vals),
                vmax=np.nanmax(plot_vals),
            )
            _annotate_heatmap(ax, vals, plot_vals, fmt_fn)

            ax.set_xticks(range(len(BF_ORDER)))
            ax.set_xticklabels([f"B={b}" for b in BF_ORDER], fontsize=9)
            ax.set_yticks(range(len(piv)))
            ax.set_yticklabels([f"L={int(l)}" for l in piv.index], fontsize=9)
            ax.set_xlabel("branching_factor", fontsize=9)
            if col == 0:
                ax.set_ylabel("num_levels", fontsize=9)
            ax.set_title(
                f"{indexer.upper()} — {title_suffix}", fontsize=10, fontweight="bold"
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Full-Build Indexer Benchmark  (n = 10 000)\n"
        "rows = num_levels, cols = branching_factor; greener = better",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
