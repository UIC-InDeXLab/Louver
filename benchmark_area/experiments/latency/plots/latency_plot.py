import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

FILES = {
    ("GPU", "Llama-3.2-3B"): "/home/mohsen/kvcache/hira/benchmark_area/experiments/latency/reports/gpu_bench_meta_llama_Llama_3.2_3B_Instruct_layer14_N40000.csv",
    ("GPU", "Qwen2.5-7B"):   "/home/mohsen/kvcache/hira/benchmark_area/experiments/latency/reports/gpu_bench_Qwen_Qwen2.5_7B_Instruct_layer14_N40000.csv",
    ("CPU", "Llama-3.2-3B"): "/home/mohsen/kvcache/hira/benchmark_area/experiments/latency/reports/cpu_bench_meta_llama_Llama_3.2_3B_Instruct_layer14_N40000.csv",
    ("CPU", "Qwen2.5-7B"):   "/home/mohsen/kvcache/hira/benchmark_area/experiments/latency/reports/cpu_bench_Qwen_Qwen2.5_7B_Instruct_layer14_N40000.csv",
}

WINDOW = 600

COLORS = {
    "Louver":       "#1967a8",
    "Torch Eager":  "#c0392b",
    "Dense (Flash)":"#27ae60",
    "Twilight":     "#6c3483",
    "Torch SDPA":   "#d35400",
}
STYLES = {
    "Louver":       dict(lw=2.8, ls="-",  zorder=6),
    "Torch Eager":  dict(lw=1.5, ls="--", zorder=3),
    "Dense (Flash)":dict(lw=1.5, ls="-.", zorder=4),
    "Twilight":     dict(lw=1.5, ls=":",  zorder=5),
    "Torch SDPA":   dict(lw=1.5, ls="-.", zorder=4),
}

HW_METHODS = {
    "GPU": {
        "Louver":       "louver_ms",
        "Torch Eager":  "dense_eager_ms",
        "Dense (Flash)":"dense_flash_ms",
        "Twilight":     "twilight_ms",
    },
    "CPU": {
        "Louver":       "louver_ms",
        "Torch Eager":  "dense_eager_ms",
        "Torch SDPA":   "torch_sdpa_ms",
    },
}

PANEL_LABELS = [["(a)", "(b)"], ["(c)", "(d)"]]

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.6,
    "grid.alpha":        0.5,
})

fig, axes = plt.subplots(2, 2, figsize=(13, 6.2), sharey="row", sharex=True)
fig.subplots_adjust(hspace=0.42, wspace=0.08)

for row, hw in enumerate(["GPU", "CPU"]):
    for col, model in enumerate(["Llama-3.2-3B", "Qwen2.5-7B"]):
        ax = axes[row][col]
        df = pd.read_csv(FILES[(hw, model)])
        df = df.set_index("n_keys").rolling(WINDOW, min_periods=1).mean().reset_index()

        methods = HW_METHODS[hw]
        avgs = {label: df[col_name].mean()
                for label, col_name in methods.items()
                if col_name in df.columns}
        louver_avg = avgs.get("Louver", 1.0)

        handles, legend_labels = [], []

        for label, col_name in methods.items():
            if col_name not in df.columns:
                continue
            x = df["n_keys"] / 1000
            y = df[col_name]

            avg = avgs[label]
            if label == "Louver":
                leg_text = f"Louver — {avg:.2f} ms"
            elif label == "Twilight":
                leg_text = f"Twilight — {avg:.2f} ms  (off-chart)"
            else:
                leg_text = f"{label} — {avg:.2f} ms"

            if label == "Twilight":
                from matplotlib.lines import Line2D
                dummy = Line2D([0], [0], color=COLORS[label], **STYLES[label])
                handles.append(dummy)
            else:
                line, = ax.plot(x, y, color=COLORS[label], **STYLES[label])
                handles.append(line)

            legend_labels.append(leg_text)

        leg = ax.legend(handles, legend_labels,
                        loc="upper left", fontsize=11,
                        frameon=True, framealpha=0.95,
                        edgecolor="#dddddd", handlelength=2.2,
                        labelspacing=0.35, borderpad=0.6)
        for i, (lbl, txt) in enumerate(zip(legend_labels, leg.get_texts())):
            if i == 0:  # Louver
                txt.set_fontweight("bold")
                txt.set_color(COLORS["Louver"])
            elif "Twilight" in lbl:
                txt.set_fontweight("bold")
                txt.set_color("#8B0000")

        # ── Speedup arrow: Louver vs Torch Eager (GPU) / Torch SDPA (CPU) ────
        ref_label = "Torch SDPA" if hw == "CPU" else "Torch Eager"
        eager_col = methods.get(ref_label)
        louver_col = methods.get("Louver")
        if eager_col and louver_col and eager_col in df.columns:
            x_arr = 39.0  # k tokens
            idx = (df["n_keys"] / 1000 - x_arr).abs().argmin()
            y_louver = df[louver_col].iloc[idx]
            y_eager  = df[eager_col].iloc[idx]
            speedup  = y_eager / y_louver

            ax.annotate("", xy=(x_arr, y_louver), xytext=(x_arr, y_eager),
                        arrowprops=dict(arrowstyle="<->", color="#222222",
                                        lw=1.6, mutation_scale=12))
            ax.text(x_arr + 0.8, (y_louver + y_eager) / 2,
                    f"{speedup:.1f}×",
                    fontsize=11, fontweight="bold", va="center", ha="left",
                    color="#222222",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="none", alpha=0.85))


        ax.set_title(f"{hw} · {model}", fontsize=13, pad=7, fontweight="semibold")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}k"))

        if col == 0:
            ax.set_ylabel("Latency (ms)", fontsize=11)
        if row == 1:
            ax.set_xlabel("Tokens generated", fontsize=11)

        # lighter tick marks
        ax.tick_params(axis="both", length=3, width=0.6)

fig.patch.set_facecolor("white")

out = "/home/mohsen/kvcache/hira/benchmark_area/experiments/latency/plots/latency_grid.pdf"
plt.savefig(out, bbox_inches="tight", dpi=200, facecolor="white")
print(f"Saved: {out}")
