"""Plot-only entry. Single output: cov50 ratio over decoding step (weight + score).

cov50 = ratio (k / T) of past tokens whose top-cumulative mass reaches 50%.
Larger ratio = wider distribution (need many keys), smaller = narrower.

Usage:
  python plot_only.py
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 320,
    "font.family": "DejaVu Sans",
    "font.size": 22, "axes.labelsize": 20, "axes.titlesize": 24,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 13,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "lines.linewidth": 2.6,
})
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if k == "token_str":
                    continue
                try:
                    fv = float(v) if "." in v or "e" in v.lower() or v.lower() in ("nan", "inf", "-inf") else int(v)
                    if isinstance(fv, float) and (math.isnan(fv) or math.isinf(fv)):
                        r[k] = None
                    else:
                        r[k] = fv
                except ValueError:
                    pass
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=str(ROOT / "reports" / "tail_metrics_dsr.csv"))
    ap.add_argument("--out_dir", default=str(ROOT / "figs"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_csv(args.metrics)
    sorted_rows = sorted(rows, key=lambda x: x["token_index"])
    if not sorted_rows:
        print("no rows"); return
    xs = [r["token_index"] for r in sorted_rows]
    prompt_len = int(sorted_rows[0]["T"]) - int(sorted_rows[0]["token_index"])

    palette = {"weight": "#475569", "score": "#1f2937"}
    for tag in tqdm(("weight", "score"), desc="plot"):
        col = f"cov50_{tag}_mean"
        if col not in sorted_rows[0]:
            print(f"skip: {col} not in CSV"); continue
        ys = [(float("nan") if r.get(col) is None else r.get(col)) for r in sorted_rows]
        finite = sorted([y for y in ys if y == y])
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        ax.set_xlim(0, max(xs) if xs else 1)
        # split prefill (faded) and decode (full alpha) segments
        line_color = palette[tag]
        pre_x = [x for x in xs if x < prompt_len]
        pre_y = ys[:len(pre_x)]
        post_x = [x for x in xs if x >= prompt_len]
        post_y = ys[len(pre_x):]
        ax.plot(pre_x, pre_y, color=line_color, linewidth=1.0, alpha=0.30,
                label="prefill")
        ax.plot(post_x, post_y, color=line_color, linewidth=1.0, alpha=0.95,
                label="decode")
        ax.set_ylabel("top-k 50% coverage (k/T)")
        ax.set_xlabel("token ID in context", fontsize=24)
        if tag == "weight":
            ax.set_ylim(0, max(1e-3, finite[-1]) * 1.05)
        else:
            # score values cluster in narrow band; use 1st-99th pct + small pad
            lo = finite[max(0, int(0.01 * len(finite)))]
            hi = finite[min(len(finite) - 1, int(0.99 * len(finite)))]
            pad = (hi - lo) * 0.10 if hi > lo else 0.01
            ax.set_ylim(max(0, lo - pad), hi + pad)
        # annotate 3 wide + 3 narrow windows, grouped with category boxes
        if tag == "weight":
            wide_color = "#dc2626"     # vivid red — wide
            narrow_color = "#0d9488"   # teal — narrow
            # tight ylim around data range to amplify oscillation
            finite_pos = [y for y in ys if y == y]
            lo = min(finite_pos); hi = max(finite_pos)
            pad = (hi - lo) * 0.12
            ax.set_ylim(max(0, lo - pad), hi + pad)

            # auto pick top-4 wide + bottom-4 narrow (well-separated, exclude
            # prefill + special tokens)
            specials = {int(r["token_index"]) for r in sorted_rows
                        if int(r.get("is_special", 0)) == 1}
            cand = [(i, y) for i, y in enumerate(ys)
                    if y == y and i >= prompt_len and i not in specials]
            cand_w = sorted(cand, key=lambda p: -p[1])
            cand_n = sorted(cand, key=lambda p:  p[1])
            def pick(seq, n=4, sep=120):
                out = []
                for ti, _ in seq:
                    if any(abs(ti - p) < sep for p in out): continue
                    out.append(ti)
                    if len(out) >= n: break
                return out
            wide_ts = pick(cand_w)
            narrow_ts = pick(cand_n)
            for ti in wide_ts:
                ax.scatter([ti], [ys[ti]], s=120, facecolors="none",
                           edgecolors=wide_color, linewidths=2.2, zorder=7)
            for ti in narrow_ts:
                ax.scatter([ti], [ys[ti]], s=120, facecolors="none",
                           edgecolors=narrow_color, linewidths=2.2, zorder=7)
        if tag == "weight":
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], color=line_color, alpha=0.30, lw=2.0,
                       label="prefill"),
                Line2D([0], [0], color=line_color, alpha=0.95, lw=2.0,
                       label="decode"),
                Line2D([0], [0], marker="o", color="none",
                       markerfacecolor="none", markeredgecolor=wide_color,
                       markeredgewidth=2.2, markersize=11,
                       label="Wide tail examples"),
                Line2D([0], [0], marker="o", color="none",
                       markerfacecolor="none", markeredgecolor=narrow_color,
                       markeredgewidth=2.2, markersize=11,
                       label="Narrow tail examples"),
            ]
            ax.legend(handles=handles, loc="upper left")
        else:
            ax.legend(loc="upper right")
        fig.tight_layout()
        out = out_dir / f"cov50_{tag}_mean_over_time.png"
        fig.savefig(out); plt.close(fig)
        print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
