"""Paper-quality plots, two figures per model.

Figure A (exp1): P(answer changed) vs M, two lines (relevant vs irrelevant).
Figure B (exp3): error vs N, lines for {dense, fixed_K=8/16/24/32, keep_all_relevant}.

Outputs into reports/paper_figs/.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib as mpl
import matplotlib.pyplot as plt

REP = Path(__file__).parent / "reports"
OUT = REP / "paper_figs"
OUT.mkdir(exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 320,
    "font.family": "DejaVu Sans",
    "font.size": 22,
    "axes.labelsize": 20,
    "axes.titlesize": 24,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 13,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "lines.linewidth": 3.0,
    "lines.markersize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# Color palette — high contrast, colorblind-friendly
C_REL = "#d6336c"      # red-pink: relevant
C_IRR = "#1c7ed6"      # blue: irrelevant
C_DENSE = "#212529"    # near-black
C_KEEPALL = "#2f9e44"  # green
K_PALETTE = ["#dc2626", "#f97316", "#fbbf24", "#fde047"]   # K=3 red → orange → amber → yellow


def load(name):
    import math
    rows = []
    p = REP / name
    if not p.exists():
        return rows
    with open(p) as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
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


def _clean(rows, metric):
    return [r for r in rows if r.get(metric) is not None]


def avg(rows, keys, m):
    b = defaultdict(list)
    for r in rows:
        b[tuple(r[k] for k in keys)].append(r[m])
    return {k: mean(v) for k, v in b.items()}


def models_for(prefix):
    return sorted({p.name.split("__", 1)[1].replace(".csv", "")
                   for p in REP.glob(f"{prefix}__*.csv")})


def _pretty_model(tag):
    return (tag.replace("Meta-Llama-3_1", "Llama-3.1")
              .replace("Llama-3_2", "Llama-3.2")
              .replace("Qwen2_5", "Qwen2.5")
              .replace("-Instruct", ""))


def plot_exp1_bars(tags, drop_M=1):
    """Grouped bar chart across models: avg P(answer-changed) for drop of M relevant
    vs M irrelevant tokens (default M=1)."""
    import numpy as np
    rows_per_tag = {t: load(f"exp1_relevant_vs_irrelevant__{t}.csv") for t in tags}
    tags = [t for t in tags if rows_per_tag[t]]
    if not tags:
        return
    rel_vals, irr_vals, rel_se, irr_se = [], [], [], []
    for t in tags:
        rs = [r for r in rows_per_tag[t] if r["M"] == drop_M]
        rel = [r["answer_changed_vs_dense"] for r in rs if r["drop_kind"] == "relevant"]
        irr = [r["answer_changed_vs_dense"] for r in rs if r["drop_kind"] == "irrelevant"]
        rel_vals.append(mean(rel)); irr_vals.append(mean(irr))
        # binomial-style standard error
        n_r, n_i = len(rel), len(irr)
        rel_se.append((mean(rel) * (1 - mean(rel)) / max(n_r, 1)) ** 0.5)
        irr_se.append((mean(irr) * (1 - mean(irr)) / max(n_i, 1)) ** 0.5)

    x = np.arange(len(tags))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    err_kw = {"alpha": 0.35, "ecolor": "#444", "elinewidth": 1.4}
    b1 = ax.bar(x - w/2, rel_vals, w, yerr=rel_se, capsize=4,
                color=C_REL, label="drop 1 relevant token (number)",
                edgecolor="white", linewidth=0.8, error_kw=err_kw)
    b2 = ax.bar(x + w/2, irr_vals, w, yerr=irr_se, capsize=4,
                color=C_IRR, label="drop 1 irrelevant token",
                edgecolor="white", linewidth=0.8, error_kw=err_kw)
    for bars, vals in [(b1, rel_vals), (b2, irr_vals)]:
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width()/2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_model(t) for t in tags], rotation=0, ha="center", fontsize=18)
    ax.set_ylabel("P(answer changed vs dense)")
    ax.set_ylim(0, max(max(rel_vals), max(irr_vals)) * 1.25 + 0.05)
    # ax.set_title(f"Effect of dropping ONE token from KV cache")
    ax.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT / "figA_exp1_bars.png")
    fig.savefig(OUT / "figA_exp1_bars.pdf")
    plt.close(fig)
    print(f"  -> figA_exp1_bars.png")


def plot_exp3(tag, metric="answer_changed_vs_dense", ylabel="P(answer changed vs dense)"):
    rows = _clean(load(f"exp3_variable_list__{tag}.csv"), metric)
    if not rows:
        return
    a = avg(rows, ["method", "N"], metric)
    methods = sorted({m for (m, _) in a})
    Ns = sorted({n for (_, n) in a})
    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    # dense baseline
    if any(m == "dense" for m in methods):
        ys = [a[("dense", n)] for n in Ns]
        ax.plot(Ns, ys, "-", color=C_DENSE, label="dense (no drop)", linewidth=2.6)

    # fixed_K family — only show K ∈ {3, 4, 6}
    SHOW_K = {3, 4, 6}
    fixed = sorted([m for m in methods if m.startswith("fixed_K")
                    and int(m.split("K")[-1]) in SHOW_K],
                   key=lambda s: int(s.split("K")[-1]))
    for i, m in enumerate(fixed):
        K = int(m.split("K")[-1])
        c = K_PALETTE[i % len(K_PALETTE)]
        ys = [a[(m, n)] for n in Ns]
        ax.plot(Ns, ys, "--o", color=c, label=f"fixed K={K}")

    # keep_all_relevant
    if any(m == "keep_all_relevant" for m in methods):
        ys = [a[("keep_all_relevant", n)] for n in Ns]
        ax.plot(Ns, ys, "-^", color=C_KEEPALL, label="keep all N",
                linewidth=2.6)

    ax.set_xlabel("number of relevant tokens (list size)", fontsize=24)
    ax.set_ylabel(ylabel)
    ax.set_xticks(Ns)
    if metric == "answer_changed_vs_dense":
        ax.set_ylim(-0.03, 1.03)
    # ax.set_title(f"{tag.replace('_', '-')}")
    ax.legend(loc="upper left", ncol=1)

    # Spike annotation — only on Llama-3.1-8B
    if tag == "Meta-Llama-3_1-8B-Instruct":
        try:
            # for each K, the spike is the first N>K
            spike_pts = []
            for i, K_val in enumerate(sorted(SHOW_K)):
                m_name = f"fixed_K{K_val}"
                if m_name not in {mm for (mm, _) in a}:
                    continue
                ns_above = [n for n in Ns if n > K_val and (m_name, n) in a]
                if not ns_above:
                    continue
                sx = ns_above[0]
                sy = a[(m_name, sx)]
                color = K_PALETTE[i % len(K_PALETTE)]
                # bold-circle the spike point
                ax.scatter([sx], [sy], s=320, facecolors=color,
                           edgecolors="white", linewidths=2.4, zorder=6)
                ax.scatter([sx], [sy], s=540, facecolors="none",
                           edgecolors=color, linewidths=2.6, alpha=0.55, zorder=5)
                spike_pts.append((sx, sy, color, K_val))

            if spike_pts:
                # pick the dramatic one for the arrow: largest jump
                target = max(spike_pts, key=lambda p: p[1])
                sx, sy, _, _ = target
                y_max = max(a[(m, n)] for (m, n) in a)
                # middle-right placement
                x_max = max(Ns)
                text_x = x_max * 0.78
                if metric == "answer_changed_vs_dense":
                    text_y = 0.50
                else:
                    text_y = 0.50 * y_max
                ANN_COLOR = "#6a1b9a"   # deep purple — distinct from K palette and red/green
                ax.annotate(
                    "Error spike",
                    xy=(sx, sy),
                    xytext=(text_x, text_y),
                    fontsize=15, fontweight="bold", color=ANN_COLOR,
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="-|>", color=ANN_COLOR,
                                    lw=2.2, shrinkA=2, shrinkB=8,
                                    mutation_scale=16,
                                    connectionstyle="arc3,rad=-0.25"),
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec=ANN_COLOR, lw=1.6),
                    zorder=7,
                )
        except KeyError:
            pass
    fig.tight_layout()
    suffix = "chg" if metric == "answer_changed_vs_dense" else "kl"
    fig.savefig(OUT / f"figB_exp3_{suffix}__{tag}.png")
    fig.savefig(OUT / f"figB_exp3_{suffix}__{tag}.pdf")
    plt.close(fig)
    print(f"  -> figB_exp3_{suffix}__{tag}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="both", choices=["chg", "kl", "both"])
    args = ap.parse_args()

    KEEP_FOR_FIGA = {"Meta-Llama-3_1-8B-Instruct", "Qwen2_5-7B-Instruct"}
    exp1_tags = sorted([t for t in models_for("exp1_relevant_vs_irrelevant")
                        if t in KEEP_FOR_FIGA],
                       key=lambda t: ("Llama" not in t, t))
    exp3_tags = sorted(models_for("exp3_variable_list"))

    print("[exp1 bar plot across models]")
    plot_exp1_bars(exp1_tags, drop_M=1)

    for tag in exp3_tags:
        print(f"[exp3 {tag}]")
        if args.metric in ("chg", "both"):
            plot_exp3(tag, "answer_changed_vs_dense", "P(answer changed vs dense)")
        if args.metric in ("kl", "both"):
            plot_exp3(tag, "kl", "KL(p_drop ‖ p_dense)")
    print(f"\nfigs in {OUT}")
