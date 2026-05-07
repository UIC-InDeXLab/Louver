import matplotlib.pyplot as plt
import pandas as pd


SEARCH_TIME_UNIT = "ms"
UPDATE_TIME_UNIT = "ms"
CSV_PATH = "cuda_results_values.csv"
OUT_PATH = "result_GQA_values.png"
LAST_POSITIONS_TO_SHOW = 2


def _is_three_level(level_value) -> bool:
    return "THREE_LEVELS" in str(level_value)


def _is_bruteforce(method_value) -> bool:
    return str(method_value) == "brute_force"


def _linestyle_for_series(method_value, level_value) -> str:
    if _is_bruteforce(method_value) or _is_three_level(level_value):
        return "--"
    return "-"


def _prepare_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["method_plot"] = out["method"].astype(str)

    if "num_levels" in out.columns:
        out["num_levels_plot"] = out["num_levels"].astype(str)
    if "branching_factor" in out.columns:
        out["branching_factor_plot"] = out["branching_factor"].astype(str)
    if "update_every" in out.columns:
        out["update_every_plot"] = out["update_every"].astype(str)

    # Merge all brute-force runs into one series.
    bf = out["method_plot"] == "brute_force"
    for col in ["num_levels_plot", "branching_factor_plot", "update_every_plot"]:
        if col in out.columns:
            out.loc[bf, col] = "ALL"

    return out


def _build_group_specs(df: pd.DataFrame):
    specs = [("method", "method_plot")]
    for pretty, col in [
        ("num_levels", "num_levels_plot"),
        ("branching_factor", "branching_factor_plot"),
        ("update_every", "update_every_plot"),
    ]:
        if col in df.columns:
            specs.append((pretty, col))
    return specs


def _format_label(key_map: dict, specs):
    parts = []
    for pretty, col in specs:
        val = key_map[col]
        if pretty != "method" and val == "ALL":
            continue
        parts.append(f"{pretty}={val}")
    return " | ".join(parts) if parts else "series"


result = pd.read_csv(CSV_PATH)
if result.empty:
    raise ValueError(f"No rows found in {CSV_PATH}")
if "position" not in result.columns or "time" not in result.columns:
    raise ValueError("CSV must contain at least: position, time")
if "method" not in result.columns:
    result["method"] = "unknown"

plot_df = _prepare_plot_df(result)
group_specs = _build_group_specs(plot_df)
group_cols = [col for _, col in group_specs]

last_positions = sorted(result["position"].dropna().unique())[-LAST_POSITIONS_TO_SHOW:]
print(f"Showing positions: {[int(x) for x in last_positions]}")

if "update_time" in plot_df.columns:
    result_for_update_print = plot_df[plot_df["position"].isin(last_positions)]
    update_print_df = (
        result_for_update_print.groupby(["position"] + group_cols, as_index=False)[
            "update_time"
        ]
        .mean()
        .sort_values(["position"] + group_cols)
    )
    print("Average update times:")
    for row in update_print_df.to_dict("records"):
        key_map = {col: row[col] for col in group_cols}
        label = _format_label(key_map, group_specs)
        print(
            f"position={int(row['position'])}, {label}, avg_update_time={row['update_time']:.6f}{UPDATE_TIME_UNIT}"
        )
else:
    print("No update_time column found in CSV.")

fig, (ax_search, ax_update) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

search_plot_df = (
    plot_df.groupby(group_cols + ["position"], as_index=False)["time"]
    .mean()
    .sort_values(group_cols + ["position"])
)

search_legend_entries = []
for key_vals, grp in search_plot_df.groupby(group_cols, dropna=False):
    if not isinstance(key_vals, tuple):
        key_vals = (key_vals,)
    key_map = dict(zip(group_cols, key_vals))
    avg_time = grp["time"].mean()
    linestyle = _linestyle_for_series(
        key_map.get("method_plot", ""),
        key_map.get("num_levels_plot", ""),
    )
    label = _format_label(key_map, group_specs)
    (line,) = ax_search.plot(
        grp["position"],
        grp["time"],
        linestyle=linestyle,
        label=f"{label} (avg={avg_time:.3f} {SEARCH_TIME_UNIT})",
    )
    search_legend_entries.append((avg_time, line))

ax_search.set_ylabel(f"Time ({SEARCH_TIME_UNIT})")
ax_search.grid(True, alpha=0.25)
if search_legend_entries:
    search_legend_entries.sort(key=lambda x: x[0])
    handles = [line for _, line in search_legend_entries]
    labels = [line.get_label() for line in handles]
    ax_search.legend(handles, labels, fontsize=8)

if "update_time" in plot_df.columns:
    update_plot_df = (
        plot_df.groupby(group_cols + ["position"], as_index=False)["update_time"]
        .mean()
        .sort_values(group_cols + ["position"])
    )

    update_legend_entries = []
    for key_vals, grp in update_plot_df.groupby(group_cols, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        key_map = dict(zip(group_cols, key_vals))
        avg_update = grp["update_time"].mean()
        linestyle = _linestyle_for_series(
            key_map.get("method_plot", ""),
            key_map.get("num_levels_plot", ""),
        )
        label = _format_label(key_map, group_specs)
        (line,) = ax_update.plot(
            grp["position"],
            grp["update_time"],
            linestyle=linestyle,
            label=f"{label} (avg={avg_update:.3f} {UPDATE_TIME_UNIT})",
        )
        update_legend_entries.append((avg_update, line))

    if update_legend_entries:
        update_legend_entries.sort(key=lambda x: x[0])
        handles = [line for _, line in update_legend_entries]
        labels = [line.get_label() for line in handles]
        ax_update.legend(handles, labels, fontsize=8)

ax_update.set_xlabel("Query Position")
ax_update.set_ylabel(f"Update Time ({UPDATE_TIME_UNIT})")
ax_update.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300)
plt.show()
