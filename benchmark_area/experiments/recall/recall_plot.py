import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.45,
    "grid.linewidth": 0.7,
})

FILES = {
    "Llama-3.2-3B":  "reports/recall_meta_llama_Llama.csv",
    "Qwen2.5-7B":    "reports/recall_Qwen_Qwen2.5_7B_.csv",
    "Qwen2.5-14B":   "reports/recall_Qwen_Qwen2.5_14B.csv",
}

METHOD_STYLE = {
    "louver":       dict(color="#1967a8", lw=3.2, ls="-",  zorder=10, marker="o", ms=8),
    "hnsw":         dict(color="#e67e22", lw=1.8, ls="--", zorder=5,  marker="s", ms=6),
    "ivf":          dict(color="#d35400", lw=1.8, ls="-.", zorder=5,  marker="^", ms=6),
    "pq":           dict(color="#f39c12", lw=1.8, ls=":",  zorder=5,  marker="D", ms=6),
    "lsh":          dict(color="#e74c3c", lw=1.8, ls="--", zorder=5,  marker="v", ms=6),
    "quest":        dict(color="#8e44ad", lw=1.8, ls="--", zorder=5,  marker="s", ms=6),
    "streamingllm": dict(color="#6c3483", lw=1.8, ls="-.", zorder=5,  marker="^", ms=6),
    "twilight":     dict(color="#2c3e50", lw=1.8, ls=":",  zorder=5,  marker="D", ms=6),
}

LABELS = {
    "louver":       "Louver (ours)",
    "hnsw":         "RetrievalAttention (HNSW)",
    "ivf":          "InfLLM (IVF)",
    "pq":           "PQCache (PQ)",
    "lsh":          "MagicPIG (LSH)",
    "quest":        "Quest",
    "streamingllm": "StreamingLLM",
    "twilight":     "Twilight",
}

PHASE_ORDER = ["louver", "hnsw", "ivf", "pq", "lsh", "quest", "streamingllm", "twilight"]

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
fig.subplots_adjust(wspace=0.08, top=0.77)

for ax, (model, fpath) in zip(axes, FILES.items()):
    df = pd.read_csv(fpath)

    # light highlight band near perfect recall
    ax.axhspan(0.98, 1.02, color="#1967a8", alpha=0.06, zorder=0)
    ax.axhline(y=1.0, color="#1967a8", lw=0.9, ls="--", alpha=0.4, zorder=1)

    for method in PHASE_ORDER:
        sub = df[df["method"] == method].sort_values("k")
        if sub.empty:
            continue
        style = METHOD_STYLE[method]
        ax.plot(sub["k"], sub["recall_mean"],
                color=style["color"], lw=style["lw"],
                ls=style["ls"], marker=style["marker"],
                markersize=style["ms"], zorder=style["zorder"])

    ax.set_title(model, fontsize=16, fontweight="semibold", pad=8)
    ax.set_xlabel("Budget $k$", fontsize=16)
    ax.set_xticks([10, 20, 50, 100])
    ax.set_xticklabels(["10", "20", "50", "100"])
    ax.set_xlim(8, 108)
    ax.set_ylim(-0.02, 1.10)
    ax.tick_params(length=3, width=0.7)

axes[0].set_ylabel("Recall@$k$", fontsize=16)

# legend handles — Louver first, styled to stand out
handles = []
for m in PHASE_ORDER:
    style = METHOD_STYLE[m]
    h = mlines.Line2D([], [],
                      color=style["color"], lw=style["lw"],
                      ls=style["ls"], marker=style["marker"],
                      markersize=style["ms"], label=LABELS[m])
    handles.append(h)

leg = fig.legend(handles=handles, loc="upper center", ncol=4,
                 fontsize=14, frameon=True, framealpha=0.97,
                 edgecolor="#cccccc", bbox_to_anchor=(0.5, 1.0),
                 handlelength=2.4, columnspacing=1.2, borderpad=0.7)

# bold + blue the Louver legend entry
for text in leg.get_texts():
    if "Louver" in text.get_text():
        text.set_fontweight("bold")
        text.set_color("#1967a8")

fig.patch.set_facecolor("white")
out = "reports/recall_plot.pdf"
plt.savefig(out, bbox_inches="tight", dpi=200, facecolor="white")
print(f"Saved: {out}")
