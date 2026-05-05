"""Publication-style plots for the false-negativity story.

Outputs (per model + combined):
  reports/figs/exp1_<model>.png   — KL & answer-changed vs M, relevant vs irrelevant
  reports/figs/exp2_<model>.png   — KL & answer-changed vs sliding-window K
  reports/figs/exp3_<model>.png   — KL & answer-changed vs N, dense / fixed-K / oracle
  reports/figs/exp4_<model>.png   — coverage vs N (per T) and T_needed vs N
  reports/figs/summary_<model>.png — 2x2 summary panel
"""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import matplotlib as mpl

REP = Path(__file__).parent / "reports"
FIG = REP / "figs"
FIG.mkdir(exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 2,
    "lines.markersize": 6,
})

C_REL = "#d6336c"     # relevant — pink/red
C_IRR = "#3b82f6"     # irrelevant — blue
C_DENSE = "#444"
C_FIXED = "#d6336c"
C_ORACLE = "#10b981"
T_COLORS = ["#fde68a", "#facc15", "#f59e0b", "#dc2626", "#7c2d12"]


def load(name):
    rows = []
    p = REP / name
    if not p.exists():
        return rows
    with open(p) as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                try:
                    r[k] = float(v) if "." in v or "e" in v.lower() else int(v)
                except ValueError:
                    pass
            rows.append(r)
    return rows


def avg(rows, keys, m):
    b = defaultdict(list)
    for r in rows:
        b[tuple(r[k] for k in keys)].append(r[m])
    return {k: mean(v) for k, v in b.items()}


def models_for(prefix):
    return sorted({p.name.split("__", 1)[1].replace(".csv", "")
                   for p in REP.glob(f"{prefix}__*.csv")})


def fig_exp1(tag):
    rows = load(f"exp1_relevant_vs_irrelevant__{tag}.csv")
    if not rows:
        return
    a_kl = avg(rows, ["drop_kind", "M"], "kl")
    a_ch = avg(rows, ["drop_kind", "M"], "answer_changed_vs_dense")
    Ms = sorted({m for (_, m) in a_kl})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, src, ylabel in [(axes[0], a_kl, "KL(p_drop ‖ p_dense)"),
                            (axes[1], a_ch, "P(answer changed vs dense)")]:
        for kind, color, marker in [("relevant", C_REL, "o"), ("irrelevant", C_IRR, "s")]:
            ys = [src[(kind, m)] for m in Ms]
            ax.plot(Ms, ys, marker=marker, color=color, label=kind)
        ax.set_xlabel("# tokens dropped (M)")
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle(f"Exp 1 · false negatives on relevant vs irrelevant tokens · {tag}")
    fig.tight_layout()
    fig.savefig(FIG / f"exp1_{tag}.png")
    plt.close(fig)


def fig_exp2(tag):
    rows = load(f"exp2_fixed_k__{tag}.csv")
    if not rows:
        return
    a_kl = avg(rows, ["K"], "kl")
    a_ch = avg(rows, ["K"], "answer_changed_vs_dense")
    a_kn = avg(rows, ["K"], "kept_numbers")
    Ks = sorted({k for (k,) in a_kl})
    N = int(rows[0]["N"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.plot(Ks, [a_kl[(k,)] for k in Ks], "o-", color=C_REL, label="KL")
    ax.set_xlabel("sliding-window K")
    ax.set_ylabel("KL(p_drop ‖ p_dense)")
    ax.set_xscale("log")
    ax2 = ax.twinx()
    ax2.plot(Ks, [a_kn[(k,)] / N for k in Ks], "s--", color=C_ORACLE,
             label=f"# numbers kept / {N}")
    ax2.set_ylabel("fraction of numbers kept")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(False)
    lines, labs = ax.get_legend_handles_labels()
    l2, l2b = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labs + l2b, loc="upper right")

    axes[1].plot(Ks, [a_ch[(k,)] for k in Ks], "o-", color=C_REL)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("sliding-window K")
    axes[1].set_ylabel("P(answer changed vs dense)")
    axes[1].set_ylim(-0.05, 1.05)
    fig.suptitle(f"Exp 2 · fixed-K sliding window (N={N}) · {tag}")
    fig.tight_layout()
    fig.savefig(FIG / f"exp2_{tag}.png")
    plt.close(fig)


def fig_exp3(tag):
    rows = load(f"exp3_variable_list__{tag}.csv")
    if not rows:
        return
    a_kl = avg(rows, ["method", "N"], "kl")
    a_ch = avg(rows, ["method", "N"], "answer_changed_vs_dense")
    Ns = sorted({n for (_, n) in a_kl})
    methods = sorted({m for (m, _) in a_kl})
    style = {"dense": (C_DENSE, "-", "o"),
             "oracle_dynamic": (C_ORACLE, "-", "^")}
    fixed_keys = [m for m in methods if m.startswith("fixed_")]
    if fixed_keys:
        style[fixed_keys[0]] = (C_FIXED, "--", "s")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, src, ylabel in [(axes[0], a_kl, "KL(p_drop ‖ p_dense)"),
                            (axes[1], a_ch, "P(answer changed vs dense)")]:
        for m in methods:
            c, ls, mk = style.get(m, ("#888", "-", "o"))
            ys = [src[(m, n)] for n in Ns]
            ax.plot(Ns, ys, marker=mk, linestyle=ls, color=c, label=m)
        ax.set_xlabel("list size N")
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle(f"Exp 3 · variable list size — fixed-K vs dynamic · {tag}")
    fig.tight_layout()
    fig.savefig(FIG / f"exp3_{tag}.png")
    plt.close(fig)


def fig_exp4(tag):
    rows = load(f"exp4_fixed_sum_threshold__{tag}.csv")
    if not rows:
        return
    a_cov = avg(rows, ["T", "N"], "coverage")
    a_K = avg(rows, ["T", "N"], "K_at_T")
    a_T = avg(rows, ["N"], "T_needed_for_full")
    Ns = sorted({n for (_, n) in a_cov})
    Ts = sorted({t for (t, _) in a_cov})

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    # panel A: coverage vs N, line per T
    for i, T in enumerate(Ts):
        c = T_COLORS[i % len(T_COLORS)]
        ys = [a_cov[(T, n)] for n in Ns]
        axes[0].plot(Ns, ys, "o-", color=c, label=f"T={T:.2f}")
    axes[0].set_xlabel("list size N")
    axes[0].set_ylabel("coverage of relevant tokens")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(title="fixed cumsum T")

    # panel B: K_at_T vs N
    for i, T in enumerate(Ts):
        c = T_COLORS[i % len(T_COLORS)]
        ys = [a_K[(T, n)] for n in Ns]
        axes[1].plot(Ns, ys, "o-", color=c, label=f"T={T:.2f}")
    axes[1].set_xlabel("list size N")
    axes[1].set_ylabel("K(T) tokens kept by threshold")
    axes[1].legend(title="fixed cumsum T")

    # panel C: T_needed
    ys = [a_T[(n,)] for n in Ns]
    axes[2].plot(Ns, ys, "o-", color=C_REL)
    axes[2].axhline(1.0, ls=":", color="#666")
    axes[2].set_xlabel("list size N")
    axes[2].set_ylabel("min T to cover ALL relevant tokens")
    axes[2].set_ylim(0.85, 1.01)

    fig.suptitle(f"Exp 4 · fixed cumsum threshold cannot adapt to N · {tag}")
    fig.tight_layout()
    fig.savefig(FIG / f"exp4_{tag}.png")
    plt.close(fig)


def fig_summary(tag):
    """Single 2x2 figure summarising headline finding from each experiment."""
    e1 = load(f"exp1_relevant_vs_irrelevant__{tag}.csv")
    e2 = load(f"exp2_fixed_k__{tag}.csv")
    e3 = load(f"exp3_variable_list__{tag}.csv")
    e4 = load(f"exp4_fixed_sum_threshold__{tag}.csv")
    if not all([e1, e2, e3, e4]):
        return

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))

    # 1: answer-changed vs M, rel vs irr
    a = avg(e1, ["drop_kind", "M"], "answer_changed_vs_dense")
    Ms = sorted({m for (_, m) in a})
    ax[0, 0].plot(Ms, [a[("relevant", m)] for m in Ms], "o-", color=C_REL, label="relevant (numbers)")
    ax[0, 0].plot(Ms, [a[("irrelevant", m)] for m in Ms], "s-", color=C_IRR, label="irrelevant (filler)")
    ax[0, 0].set_xlabel("# tokens dropped (M)")
    ax[0, 0].set_ylabel("P(answer changed)")
    ax[0, 0].set_title("(a) Exp 1 — false-negative on relevant > irrelevant")
    ax[0, 0].legend()

    # 2: KL vs K
    a = avg(e2, ["K"], "kl")
    a_kn = avg(e2, ["K"], "kept_numbers")
    Ks = sorted({k for (k,) in a})
    N2 = int(e2[0]["N"])
    ax[0, 1].plot(Ks, [a[(k,)] for k in Ks], "o-", color=C_REL, label="KL")
    ax2t = ax[0, 1].twinx()
    ax2t.plot(Ks, [a_kn[(k,)] / N2 for k in Ks], "s--", color=C_ORACLE,
              label=f"frac numbers kept (N={N2})")
    ax2t.set_ylim(-0.05, 1.05); ax2t.grid(False)
    ax[0, 1].set_xscale("log"); ax[0, 1].set_xlabel("sliding-window K")
    ax[0, 1].set_ylabel("KL(p_drop ‖ p_dense)")
    ax[0, 1].set_title("(b) Exp 2 — fixed-K sliding window")
    h1, l1 = ax[0, 1].get_legend_handles_labels()
    h2, l2 = ax2t.get_legend_handles_labels()
    ax[0, 1].legend(h1 + h2, l1 + l2, loc="upper right")

    # 3: KL vs N for fixed-K vs oracle
    a = avg(e3, ["method", "N"], "kl")
    Ns3 = sorted({n for (_, n) in a})
    methods = sorted({m for (m, _) in a})
    style = {"dense": (C_DENSE, "-", "o"), "oracle_dynamic": (C_ORACLE, "-", "^")}
    for m in methods:
        if m.startswith("fixed_"): style[m] = (C_FIXED, "--", "s")
        c, ls, mk = style.get(m, ("#888", "-", "o"))
        ax[1, 0].plot(Ns3, [a[(m, n)] for n in Ns3], marker=mk, ls=ls, color=c, label=m)
    ax[1, 0].set_xlabel("list size N")
    ax[1, 0].set_ylabel("KL(p_drop ‖ p_dense)")
    ax[1, 0].set_title("(c) Exp 3 — fixed-K vs dynamic")
    ax[1, 0].legend()

    # 4: coverage vs N at fixed T
    a = avg(e4, ["T", "N"], "coverage")
    Ns4 = sorted({n for (_, n) in a})
    Ts = sorted({t for (t, _) in a})
    for i, T in enumerate(Ts):
        c = T_COLORS[i % len(T_COLORS)]
        ax[1, 1].plot(Ns4, [a[(T, n)] for n in Ns4], "o-", color=c, label=f"T={T:.2f}")
    ax[1, 1].set_xlabel("list size N")
    ax[1, 1].set_ylabel("coverage of relevant tokens")
    ax[1, 1].set_title("(d) Exp 4 — fixed cumsum threshold T")
    ax[1, 1].set_ylim(-0.05, 1.05)
    ax[1, 1].legend(title="fixed T", ncol=2)

    fig.suptitle(f"False-negativity in sparse attention · {tag}", y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / f"summary_{tag}.png")
    plt.close(fig)


if __name__ == "__main__":
    tags = (set(models_for("exp1_relevant_vs_irrelevant"))
            | set(models_for("exp2_fixed_k"))
            | set(models_for("exp3_variable_list"))
            | set(models_for("exp4_fixed_sum_threshold")))
    for tag in sorted(tags):
        print(f"-> {tag}")
        fig_exp1(tag); fig_exp2(tag); fig_exp3(tag); fig_exp4(tag)
        fig_summary(tag)
    print(f"\nfigures in {FIG}")
