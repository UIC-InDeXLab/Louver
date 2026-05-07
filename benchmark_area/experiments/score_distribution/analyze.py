"""From tail_metrics.csv + snapshot: pick wide vs narrow steps, plot examples,
sort decoding windows by tail-size, dump top/bottom contexts."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent
FIXED_K_CHAL = ROOT.parent.parent / "fixed_k_chal"
sys.path.insert(0, str(FIXED_K_CHAL))

from helpers import ObserveAttentionHelper  # noqa: E402

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 220,
    "font.size": 14, "axes.labelsize": 14, "axes.titlesize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linestyle": "--",
})


def load_csv(p):
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if k in ("token_str",):
                    continue
                try:
                    r[k] = float(v) if "." in v or v.lower() in ("nan", "inf", "-inf") else int(v)
                except ValueError:
                    pass
            rows.append(r)
    return rows


_LAYER_CACHE = None


def get_step_raw_scores(helper, t, layers, device="cuda" if torch.cuda.is_available() else "cpu"):
    global _LAYER_CACHE
    from tail_metrics import build_layer_tensors
    if _LAYER_CACHE is None:
        _LAYER_CACHE = build_layer_tensors(helper, layers, device)
    rows = []
    global_pos = helper.prompt_length + t
    for Q, K, ratio in _LAYER_CACHE:
        if t >= Q.shape[1]:
            continue
        q = Q[:, t, :]
        K_pref = K[:, :global_pos, :]
        K_exp = K_pref.repeat_interleave(ratio, dim=0)
        D = q.shape[-1]
        rows.append(torch.einsum("hd,htd->ht", q, K_exp) / (D ** 0.5))
    return torch.cat(rows, dim=0)  # (H, T) raw pre-softmax


def get_step_scores(helper, t, layers, device="cuda" if torch.cuda.is_available() else "cpu"):
    return torch.softmax(get_step_raw_scores(helper, t, layers, device), dim=-1)


def plot_pair_rank(p_narrow, p_wide, t_n, t_w, tail_frac, out_path):
    """Two stacked panels (narrow top, wide bottom) sharing x-axis = rank.
    Right tail (rank > head_size of narrow case) shaded differently."""
    import numpy as np
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4), sharex=True)
    panels = [(axes[0], p_narrow, "NARROW", t_n, "#1c7ed6"),
              (axes[1], p_wide, "WIDE", t_w, "#d6336c")]
    # Use wide T as common x-axis upper bound
    T_max = max(p_narrow.shape[1], p_wide.shape[1])
    for ax, p, label, t, col in panels:
        H, T = p.shape
        p_max = p.max(0).values.cpu().numpy()
        p_mean = p.mean(0).cpu().numpy()
        order = (-p_max).argsort()
        x = np.arange(1, T + 1)
        head = max(1, int(T * tail_frac))
        ax.plot(x, p_max[order], color=col, linewidth=1.6, alpha=0.95,
                label=f"max-head ({label})")
        ax.plot(x, p_mean[order], color="#212529", linewidth=2.0, alpha=0.7,
                label="mean-head")
        ax.axvspan(head, T, color="#fde047", alpha=0.25, label=f"tail (>top {tail_frac:.0%})")
        ax.set_yscale("log")
        ax.set_ylabel("attn prob (log)")
        ax.set_title(f"{label}  ·  t={t}  ·  T={T}")
        ax.legend(loc="upper right")
        ax.set_xlim(1, T_max)
    axes[1].set_xlabel("rank of past token (sorted)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path.name}")


def plot_pair_pdf(p_narrow, p_wide, t_n, t_w, out_path):
    """Two stacked PDF panels. Shade HIGH-score region (prob > uniform)."""
    import numpy as np
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4), sharex=True)
    pn = np.log10(np.clip(p_narrow.cpu().numpy().reshape(-1), 1e-12, 1.0))
    pw = np.log10(np.clip(p_wide.cpu().numpy().reshape(-1), 1e-12, 1.0))
    x_lo = float(min(pn.min(), pw.min()))
    x_hi = float(max(pn.max(), pw.max()))
    bins = np.linspace(x_lo, x_hi, 70)
    panels = [(axes[0], p_narrow, pn, "NARROW", t_n, "#1c7ed6"),
              (axes[1], p_wide, pw, "WIDE", t_w, "#d6336c")]
    for ax, p, log_p, label, t, col in panels:
        T = p.shape[1]
        unif = np.log10(1.0 / T)
        ax.hist(log_p, bins=bins, density=True, color=col,
                edgecolor="white", alpha=0.85)
        # high-score region: probs > uniform
        ax.axvspan(unif, x_hi, color="#fde047", alpha=0.32,
                   label="high (> uniform)")
        ax.axvline(unif, color="#444", linestyle=":", linewidth=1.6,
                   label=f"uniform 1/T={1/T:.2g}")
        ax.set_ylabel("density")
        ax.set_title(f"{label}  ·  t={t}  ·  T={T}")
        ax.legend(loc="upper left")
    axes[1].set_xlabel("log10(attention prob)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path.name}")


def plot_pair_rank_scores(s_narrow, s_wide, t_n, t_w, out_path):
    """Pre-softmax raw scores, sorted by rank, narrow top / wide bottom."""
    import numpy as np
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4), sharex=True)
    panels = [(axes[0], s_narrow, "NARROW", t_n, "#1c7ed6"),
              (axes[1], s_wide, "WIDE", t_w, "#d6336c")]
    T_max = max(s_narrow.shape[1], s_wide.shape[1])
    for ax, s, label, t, col in panels:
        H, T = s.shape
        s_max = s.max(0).values.cpu().numpy()
        s_mean = s.mean(0).cpu().numpy()
        order = (-s_max).argsort()
        x = np.arange(1, T + 1)
        ax.plot(x, s_max[order], color=col, linewidth=1.6, alpha=0.95,
                label=f"max-head ({label})")
        ax.plot(x, s_mean[order], color="#212529", linewidth=2.0, alpha=0.7,
                label="mean-head")
        ax.set_ylabel("attn score (raw)")
        ax.set_title(f"{label}  ·  t={t}  ·  T={T}")
        ax.legend(loc="upper right")
        ax.set_xlim(1, T_max)
    axes[1].set_xlabel("rank of past token (sorted)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path.name}")


def plot_pair_pdf_scores(s_narrow, s_wide, t_n, t_w, out_path):
    """Pre-softmax raw scores PDF, narrow top / wide bottom.
    Use percentile clipping so outliers don't collapse into one bar."""
    import numpy as np
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4), sharex=True)
    sn = s_narrow.cpu().numpy().reshape(-1)
    sw = s_wide.cpu().numpy().reshape(-1)
    pooled = np.concatenate([sn, sw])
    # robust x range: 0.5%–99.5% percentile of pooled values
    x_lo = float(np.percentile(pooled, 0.5))
    x_hi = float(np.percentile(pooled, 99.5))
    if x_hi - x_lo < 1e-6:
        x_hi = x_lo + 1.0
    bins = np.linspace(x_lo, x_hi, 80)
    panels = [(axes[0], s_narrow, sn, "NARROW", t_n, "#1c7ed6"),
              (axes[1], s_wide, sw, "WIDE", t_w, "#d6336c")]
    for ax, s, vals, label, t, col in panels:
        T = s.shape[1]
        clipped = np.clip(vals, x_lo, x_hi)
        top1 = float(np.percentile(vals, 99))
        ax.hist(clipped, bins=bins, density=True, color=col,
                edgecolor="white", alpha=0.85)
        ax.axvspan(top1, x_hi, color="#fde047", alpha=0.32,
                   label=f"high (>p99={top1:.2f})")
        ax.axvline(top1, color="#444", linestyle=":", linewidth=1.6)
        ax.set_ylabel("density")
        ax.set_yscale("log")
        ax.set_title(f"{label}  ·  t={t}  ·  T={T}  ·  range=[{x_lo:.1f},{x_hi:.1f}]")
        ax.legend(loc="upper left")
    axes[1].set_xlabel("attn score (raw, pre-softmax)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path.name}")


def plot_distribution(probs_HT, title, out_path):
    """probs: (H, T). Two-panel:
       (top) PDF of attention scores (KDE-style histogram on log10 prob).
       (bot) sorted prob-by-rank (legacy view).
    Aggregates: pool across heads."""
    import numpy as np
    p = probs_HT.cpu().numpy()  # (H, T)
    H, T = p.shape
    p_max = p.max(0)             # worst-head per position
    p_mean = p.mean(0)
    p_pool = p.reshape(-1)       # all (head,pos) probs

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.0))

    # PDF on log10(prob)
    ax = axes[0]
    log_p = np.log10(np.clip(p_pool, 1e-12, 1.0))
    ax.hist(log_p, bins=60, density=True, color="#1c7ed6",
            edgecolor="white", alpha=0.85)
    ax.axvline(np.log10(1.0 / T), color="#444", linestyle=":",
               linewidth=1.6, label=f"uniform 1/T={1/T:.3g}")
    ax.set_xlabel("log10(attention prob)")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(loc="upper left")

    # rank curve
    ax = axes[1]
    order = (-p_max).argsort()
    x = range(1, T + 1)
    ax.plot(x, p_max[order], color="#d6336c", linewidth=1.8,
            alpha=0.85, label="max over heads")
    ax.plot(x, p_mean[order], color="#1c7ed6", linewidth=2.2,
            label="mean over heads")
    ax.set_yscale("log")
    ax.set_xlabel("rank of past token (sorted)")
    ax.set_ylabel("attention prob (log)")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path.name}")


def context_around(helper, t, window=10):
    """Return string showing tokens around generated index t."""
    parts = []
    for idx in range(max(0, t - window), min(len(helper.generated_tokens), t + window + 1)):
        s = helper.get_token_string(idx)
        if idx == t:
            parts.append(f"[[{s}]]")
        else:
            parts.append(s)
    return "".join(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", default=str(ROOT / "snapshots" / "snap_long.pt"))
    ap.add_argument("--metrics", default=str(ROOT / "reports" / "tail_metrics.csv"))
    ap.add_argument("--out_dir", default=str(ROOT / "figs"))
    ap.add_argument("--report", default=str(ROOT / "reports" / "windows.md"))
    ap.add_argument("--metric", default="eff_size_max",
                    help="column used to rank tail width")
    ap.add_argument("--metrics_list", nargs="+", default=None,
                    help="if given, generate dist_pair_*.png for each metric")
    ap.add_argument("--top_n", type=int, default=8,
                    help="how many wide / narrow steps to extract")
    ap.add_argument("--window", type=int, default=20,
                    help="±window of decoded tokens to show around each pick")
    ap.add_argument("--topk_coverage", type=int, nargs="+", default=[8, 32, 128],
                    help="k values for top-k coverage line plot")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    helper = ObserveAttentionHelper.from_file(args.snap)
    num_layers = len(helper.queries)
    start = (num_layers * 3) // 4
    layers = list(range(start, num_layers))

    raw = load_csv(args.metrics)

    def rank_for(metric):
        # higher value = wider (user convention: larger top-k mass = more important tokens)
        rs = [r for r in raw if isinstance(r.get(metric), (int, float))]
        rs.sort(key=lambda r: r[metric])  # ascending → rs[0] narrow, rs[-1] wide
        return rs[: args.top_n], rs[-args.top_n :][::-1]

    metrics_to_plot = args.metrics_list or [args.metric]

    # Use args.metric for windows.md report (and rank/over-time line plot below)
    narrow, wide = rank_for(args.metric)
    rows = sorted(
        [r for r in raw if isinstance(r.get(args.metric), (int, float))],
        key=lambda x: x["token_index"],
    )

    # per-metric pair plots — restrict to top-k mass metrics only
    metrics_to_plot = [m for m in metrics_to_plot if m.startswith("topk_mass_")]
    if not metrics_to_plot:
        metrics_to_plot = ["topk_mass_32_min"]
    for m in metrics_to_plot:
        nar, wid = rank_for(m)
        if not nar or not wid:
            continue
        n0, w0 = nar[0], wid[0]
        p_n = get_step_scores(helper, n0["token_index"], layers)
        p_w = get_step_scores(helper, w0["token_index"], layers)
        s_n = get_step_raw_scores(helper, n0["token_index"], layers)
        s_w = get_step_raw_scores(helper, w0["token_index"], layers)
        # post-softmax
        plot_pair_rank(p_n, p_w, n0["token_index"], w0["token_index"],
                       tail_frac=0.05, out_path=out_dir / f"dist_pair_rank__{m}.png")
        plot_pair_pdf(p_n, p_w, n0["token_index"], w0["token_index"],
                      out_path=out_dir / f"dist_pair_pdf__{m}.png")
        # pre-softmax raw scores
        plot_pair_rank_scores(s_n, s_w, n0["token_index"], w0["token_index"],
                              out_path=out_dir / f"dist_pair_rank_raw__{m}.png")
        plot_pair_pdf_scores(s_n, s_w, n0["token_index"], w0["token_index"],
                             out_path=out_dir / f"dist_pair_pdf_raw__{m}.png")

    # Line plot of metric over time (with 8B-like cosmetics)
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    xs = [r["token_index"] for r in sorted(rows, key=lambda x: x["token_index"])]
    ys = [r[args.metric] for r in sorted(rows, key=lambda x: x["token_index"])]
    ax.plot(xs, ys, color="#212529", linewidth=1.4)
    # mark wide/narrow picks
    for r in wide:
        ax.scatter([r["token_index"]], [r[args.metric]], s=80,
                   color="#d6336c", zorder=5, label="wide" if r is wide[0] else None)
    for r in narrow:
        ax.scatter([r["token_index"]], [r[args.metric]], s=80,
                   color="#1c7ed6", zorder=5, label="narrow" if r is narrow[0] else None)
    ax.set_xlabel("decoding step (generated token index)")
    ax.set_ylabel(args.metric)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "tail_over_time.png")
    plt.close(fig)
    print(f"  -> tail_over_time.png")

    # Top-k mean coverage with spike annotations (widest = lowest top-k mean mass)
    import numpy as np
    sorted_rows = sorted(rows, key=lambda x: x["token_index"])
    xs = np.array([r["token_index"] for r in sorted_rows])
    ks = args.topk_coverage
    palette_k = ["#dc2626", "#1c7ed6", "#2f9e44"]
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ymat = []
    for i, k in enumerate(ks):
        col = f"topk_mass_{k}_mean"
        ys = np.array([(r.get(col) if r.get(col) is not None else float("nan"))
                       for r in sorted_rows])
        ymat.append(ys)
        ax.plot(xs, ys, color=palette_k[i % len(palette_k)],
                linewidth=1.2, alpha=0.85, label=f"top-{k} mean")
    ymat = np.stack(ymat, axis=0)  # (K, G)

    # pick 2 widest spikes (lowest mean top-32 mass), well-separated
    rank_key = f"topk_mass_32_mean"
    rk = np.array([(r.get(rank_key) if r.get(rank_key) is not None else float("nan"))
                   for r in sorted_rows])
    order = np.argsort(-rk)  # descending → widest (highest top-k mass) first
    picks = []
    min_sep = max(50, len(xs) // 20)
    for idx in order:
        if any(abs(int(xs[idx]) - p) < min_sep for p in picks):
            continue
        picks.append(int(xs[idx]))
        if len(picks) >= 2:
            break

    # text-window placements: alternate top-left / top-right
    placements = [(0.03, 0.95, "left"), (0.55, 0.95, "left")]
    for j, x_pick in enumerate(picks):
        idx = int(np.where(xs == x_pick)[0][0])
        # bounding box: vertical span over all top-k values at this x
        y_at_x = ymat[:, idx]
        y_lo, y_hi = float(np.nanmin(y_at_x)), float(np.nanmax(y_at_x))
        pad = 0.02
        # rectangle highlight via vertical span (a bit narrower than data)
        ax.axvspan(x_pick - 8, x_pick + 8, ymin=max(0, y_lo - pad),
                   ymax=min(1, y_hi + pad), color="none",
                   edgecolor="#6a1b9a", linewidth=0)
        # actual rectangle in data coordinates
        from matplotlib.patches import Rectangle
        rect = Rectangle((x_pick - 12, y_lo - pad), 24, (y_hi - y_lo) + 2 * pad,
                         linewidth=2.0, edgecolor="#6a1b9a", facecolor="none",
                         zorder=6)
        ax.add_patch(rect)
        # mark each top-k value as a dot
        for ki, k in enumerate(ks):
            ax.scatter([x_pick], [y_at_x[ki]], s=60,
                       color=palette_k[ki % len(palette_k)],
                       edgecolors="white", linewidths=1.6, zorder=7)

        # decoded window
        try:
            ctx = context_around(helper, x_pick, 12)
        except Exception:
            ctx = ""
        # truncate
        ctx_show = ctx.replace("\n", " ").strip()
        if len(ctx_show) > 130:
            ctx_show = ctx_show[:130] + "..."
        # arrow to spike
        ax_x_frac, ax_y_frac, ha = placements[j]
        ax.annotate(
            f"wide @ t={x_pick}\n«{ctx_show}»",
            xy=(x_pick, y_lo),
            xytext=(ax_x_frac, ax_y_frac),
            textcoords="axes fraction",
            ha=ha, va="top",
            fontsize=10, color="#311b6f",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff7c2",
                      ec="#6a1b9a", lw=1.4),
            arrowprops=dict(arrowstyle="-|>", color="#6a1b9a",
                            lw=1.8, shrinkA=2, shrinkB=4,
                            connectionstyle="arc3,rad=-0.2"),
            zorder=8,
        )

    ax.set_xlabel("decoding step (generated token index)")
    ax.set_ylabel("top-k attention mass (mean across heads)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", ncol=len(ks))
    fig.tight_layout()
    fig.savefig(out_dir / "topk_mean_annotated.png")
    plt.close(fig)
    print(f"  -> topk_mean_annotated.png")

    # Top-k coverage line plot vs decoding step (mean across heads only)
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    sorted_rows = sorted(rows, key=lambda x: x["token_index"])
    xs = [r["token_index"] for r in sorted_rows]
    palette = ["#dc2626", "#f97316", "#1c7ed6", "#2f9e44"]
    for i, k in enumerate(args.topk_coverage):
        col = f"topk_mass_{k}_mean"
        ys = [r.get(col) for r in sorted_rows]
        ys = [(float("nan") if y is None else y) for y in ys]
        ax.plot(xs, ys, color=palette[i % len(palette)],
                linewidth=1.1, alpha=0.85, label=f"top-{k}")
    ax.set_ylabel("top-k mass (mean across heads)")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("decoding step (generated token index)")
    ax.legend(loc="lower right", ncol=len(args.topk_coverage))
    fig.tight_layout()
    fig.savefig(out_dir / "topk_coverage_over_time.png")
    plt.close(fig)
    print(f"  -> topk_coverage_over_time.png")

    # Markdown report with windows
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Score-distribution windows  (metric: `{args.metric}`)\n\n"]
    lines.append(f"Layers aggregated: {layers}\n\n")
    lines.append("LLM-output windows enclosed in code fences. `[[token]]` marks the decoding step.\n")

    def fmt(section_title, items):
        lines.append(f"\n## {section_title}\n")
        for r in items:
            ti = int(r["token_index"])
            ctx = context_around(helper, ti, args.window)
            lines.append(
                f"\n### t={ti}  ·  `{args.metric}={r[args.metric]:.2f}`  ·  T={int(r['T'])}  "
                f"·  topk_mass_8_min={r.get('topk_mass_8_min', float('nan')):.3f}\n"
            )
            lines.append("\n```text\n")
            lines.append(ctx)
            lines.append("\n```\n")

    fmt("Wide-tail decoding steps (broad attention — likely planning/pivots)", wide)
    fmt("Narrow-tail decoding steps (focused attention — likely local computation)", narrow)

    Path(args.report).write_text("".join(lines))
    print(f"wrote {args.report}")
