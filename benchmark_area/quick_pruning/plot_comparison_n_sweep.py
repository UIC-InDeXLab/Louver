#!/usr/bin/env python3
"""
Plot results from comparison_n_sweep.py.

This script is intentionally simple:
- edit the hardcoded method selections below
- point it at the generated all_results.csv
- it writes a few PNGs plus a Markdown summary table

Notes:
- Pair plots use exact (clustering, enclosing) pairs.
- Clustering build-time lines are deduplicated over enclosing methods.
- Enclosing build/search lines are shown under one reference clustering, since
  enc_ms/search_ms depend on the clustering assignments they are built on.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = (
    SCRIPT_DIR / "result" / "comparison_n_tokens_sweep" / "all_results.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "comparison_n_tokens_sweep" / "plots"


# ---------------------------------------------------------------------
# Available methods from clusterings/__init__.py and enclosings/__init__.py.
# Keep this list in sync with the active registries.
# ---------------------------------------------------------------------

AVAILABLE_CLUSTERINGS = [
    "kmeans",
    "random_proj",
    "pq_subspace",
    "kcenter",
    "pca_pq",
    "whitened_pq",
    "batch_nn",
]

AVAILABLE_ENCLOSINGS = [
    "ball_centroid",
    "min_enclosing_ball",
    "aabb",
    "ellipsoid",
    "outlier_aabb",
    "outlier_ball_centroid",
]

AVAILABLE_PAIRS = [
    ("kmeans", "ball_centroid"),
    ("kmeans", "min_enclosing_ball"),
    ("kmeans", "aabb"),
    ("kmeans", "ellipsoid"),
    ("kmeans", "outlier_aabb"),
    ("kmeans", "outlier_ball_centroid"),
    ("random_proj", "ball_centroid"),
    ("random_proj", "min_enclosing_ball"),
    ("random_proj", "aabb"),
    ("random_proj", "ellipsoid"),
    ("random_proj", "outlier_aabb"),
    ("random_proj", "outlier_ball_centroid"),
    ("pq_subspace", "ball_centroid"),
    ("pq_subspace", "min_enclosing_ball"),
    ("pq_subspace", "aabb"),
    ("pq_subspace", "ellipsoid"),
    ("pq_subspace", "outlier_aabb"),
    ("pq_subspace", "outlier_ball_centroid"),
    ("kcenter", "ball_centroid"),
    ("kcenter", "min_enclosing_ball"),
    ("kcenter", "aabb"),
    ("kcenter", "ellipsoid"),
    ("kcenter", "outlier_aabb"),
    ("kcenter", "outlier_ball_centroid"),
    ("pca_pq", "ball_centroid"),
    ("pca_pq", "min_enclosing_ball"),
    ("pca_pq", "aabb"),
    ("pca_pq", "ellipsoid"),
    ("pca_pq", "outlier_aabb"),
    ("pca_pq", "outlier_ball_centroid"),
    ("whitened_pq", "ball_centroid"),
    ("whitened_pq", "min_enclosing_ball"),
    ("whitened_pq", "aabb"),
    ("whitened_pq", "ellipsoid"),
    ("whitened_pq", "outlier_aabb"),
    ("whitened_pq", "outlier_ball_centroid"),
    ("batch_nn", "ball_centroid"),
    ("batch_nn", "min_enclosing_ball"),
    ("batch_nn", "aabb"),
    ("batch_nn", "ellipsoid"),
    ("batch_nn", "outlier_aabb"),
    ("batch_nn", "outlier_ball_centroid"),
]


# ---------------------------------------------------------------------
# Hardcoded selections: edit these lists for the methods you want.
# ---------------------------------------------------------------------

SELECTED_PAIRS = [
    ("batch_nn", "aabb"),
    ("batch_nn", "ball_centroid"),
    # ("batch_nn", "ellipsoid"),
    ("kcenter", "aabb"),
    ("kcenter", "ball_centroid"),
    # ("kcenter", "ellipsoid"),
    ("pq_subspace", "aabb"),
    ("pq_subspace", "ball_centroid"),
    # ("pq_subspace", "ellipsoid"),
]

SELECTED_CLUSTERINGS = [
    "batch_nn",
    "kcenter",
    "pq_subspace",
]

SELECTED_ENCLOSINGS = [
    "aabb",
    "ball_centroid",
    # "ellipsoid",
]

# Enclosing build/search depends on the clustering assignments. Keep one
# clustering fixed so the enclosing-only timing comparison is meaningful.
REFERENCE_CLUSTERING_FOR_ENCLOSINGS = "batch_nn"


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed: dict[str, object] = dict(row)
            for key in (
                "n_tokens",
                "prompt_len",
                "total_keys",
                "layer",
                "bf",
                "topk",
                "n_queries",
            ):
                if key in parsed and parsed[key] not in ("", None):
                    parsed[key] = int(str(parsed[key]))
            for key in (
                "scanned_frac",
                "pruned_frac",
                "search_ms",
                "build_ms",
                "clust_ms",
                "enc_ms",
                "gate_cost_dp",
                "ratio",
                "speedup",
            ):
                if key in parsed and parsed[key] not in ("", None):
                    parsed[key] = float(str(parsed[key]))
            total_keys = int(parsed["total_keys"])
            bf = int(parsed["bf"])
            num_parents = (total_keys + bf - 1) // bf
            parsed["num_parents"] = num_parents
            parsed["avg_enc_ms"] = float(parsed["enc_ms"]) / max(1, num_parents)
            parsed["avg_search_ms"] = float(parsed["search_ms"]) / max(1, num_parents)
            rows.append(parsed)
    return rows


def pair_label(clustering: str, enclosing: str) -> str:
    return f"{clustering} + {enclosing}"


def filter_selected_pairs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    wanted = set(SELECTED_PAIRS)
    return [
        row for row in rows if (str(row["clustering"]), str(row["enclosing"])) in wanted
    ]


def _markdown_table_lines(
    rows: list[dict[str, object]],
    title: str,
) -> list[str]:
    headers = [
        "n_tokens",
        "total_keys",
        "num_parents",
        "clustering",
        "enclosing",
        "scanned_frac",
        "pruned_frac",
        "gate_cost_dp",
        "speedup",
        "ratio",
        "clust_ms",
        "enc_ms",
        "search_ms",
        "avg_enc_ms",
        "avg_search_ms",
    ]

    rows_by_n: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_n[int(row["n_tokens"])].append(row)

    lines = [f"## {title}", ""]
    for n_tokens in sorted(rows_by_n):
        group_rows = sorted(
            rows_by_n[n_tokens],
            key=lambda row: (
                float(row["scanned_frac"]),
                str(row["clustering"]),
                str(row["enclosing"]),
            ),
        )
        lines.extend(
            [
                f"### n_tokens = {n_tokens}",
                "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
            ]
        )
        for row in group_rows:
            values = []
            for header in headers:
                value = row[header]
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    return lines


def write_markdown_table(
    all_rows: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
    input_csv: Path,
    output_path: Path,
) -> None:
    lines = [
        "# comparison_n_tokens_sweep results",
        "",
        f"Input CSV: `{input_csv}`",
        "",
    ]
    lines.extend(_markdown_table_lines(selected_rows, "Selected Pair Rows"))
    lines.extend(["", ""])
    lines.extend(_markdown_table_lines(all_rows, "All Rows"))
    output_path.write_text("\n".join(lines) + "\n")


def _plot_pair_metric(
    rows: list[dict[str, object]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

    by_pair: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_pair[(str(row["clustering"]), str(row["enclosing"]))].append(row)

    fig, ax = plt.subplots(figsize=(11, 6))
    for clustering, enclosing in SELECTED_PAIRS:
        pair_rows = sorted(
            by_pair.get((clustering, enclosing), []),
            key=lambda row: int(row["n_tokens"]),
        )
        if not pair_rows:
            continue
        xs = [int(row["n_tokens"]) for row in pair_rows]
        ys = [float(row[metric_key]) for row in pair_rows]
        ax.plot(
            xs, ys, marker="o", linewidth=2, label=pair_label(clustering, enclosing)
        )

    if metric_key == "speedup":
        ax.axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label="1.0x baseline",
        )
    elif metric_key == "scanned_frac":
        ys_all = [float(row[metric_key]) for row in rows]
        y_min = min(ys_all)
        y_max = max(ys_all)
        y_pad = max(0.005, 0.08 * (y_max - y_min if y_max > y_min else 0.02))
        lower = max(0.0, y_min - y_pad)
        upper = min(1.0, y_max + y_pad)
        ax.set_ylim(lower, upper)

        span = upper - lower
        if span <= 0.05:
            major_step = 0.005
        elif span <= 0.10:
            major_step = 0.01
        elif span <= 0.20:
            major_step = 0.02
        else:
            major_step = 0.05

        ax.yaxis.set_major_locator(MultipleLocator(major_step))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, which="minor", alpha=0.15)

    ax.set_xlabel("n_tokens")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_clustering_build_time(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values: dict[str, dict[int, list[float]]] = {
        clustering: defaultdict(list) for clustering in SELECTED_CLUSTERINGS
    }
    for row in rows:
        clustering = str(row["clustering"])
        if clustering not in values:
            continue
        values[clustering][int(row["n_tokens"])].append(float(row["clust_ms"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    for clustering in SELECTED_CLUSTERINGS:
        series = values[clustering]
        if not series:
            continue
        xs = sorted(series)
        ys = [sum(series[n]) / len(series[n]) for n in xs]
        ax.plot(xs, ys, marker="o", linewidth=2, label=clustering)

    ax.set_xlabel("n_tokens")
    ax.set_ylabel("clust_ms")
    ax.set_title("Clustering Build Time vs n_tokens")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_enclosing_timing(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    enc_build: dict[str, dict[int, list[float]]] = {
        enclosing: defaultdict(list) for enclosing in SELECTED_ENCLOSINGS
    }
    enc_search: dict[str, dict[int, list[float]]] = {
        enclosing: defaultdict(list) for enclosing in SELECTED_ENCLOSINGS
    }

    for row in rows:
        if str(row["clustering"]) != REFERENCE_CLUSTERING_FOR_ENCLOSINGS:
            continue
        enclosing = str(row["enclosing"])
        if enclosing not in enc_build:
            continue
        n_tokens = int(row["n_tokens"])
        enc_build[enclosing][n_tokens].append(float(row["avg_enc_ms"]))
        enc_search[enclosing][n_tokens].append(float(row["avg_search_ms"]))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for enclosing in SELECTED_ENCLOSINGS:
        build_series = enc_build[enclosing]
        search_series = enc_search[enclosing]
        if build_series:
            xs = sorted(build_series)
            ys = [sum(build_series[n]) / len(build_series[n]) for n in xs]
            axes[0].plot(xs, ys, marker="o", linewidth=2, label=enclosing)
        if search_series:
            xs = sorted(search_series)
            ys = [sum(search_series[n]) / len(search_series[n]) for n in xs]
            axes[1].plot(xs, ys, marker="o", linewidth=2, label=enclosing)

    axes[0].set_ylabel("avg enc_ms per parent")
    axes[0].set_title(
        f"Average Enclosing Build Time vs n_tokens ({REFERENCE_CLUSTERING_FOR_ENCLOSINGS})"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].set_xlabel("n_tokens")
    axes[1].set_ylabel("avg search_ms per parent")
    axes[1].set_title(
        f"Average Gate/Search Time vs n_tokens ({REFERENCE_CLUSTERING_FOR_ENCLOSINGS})"
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    input_csv = DEFAULT_INPUT_CSV
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_csv)
    selected_pair_rows = filter_selected_pairs(rows)

    if not selected_pair_rows:
        raise RuntimeError(
            "No selected pair rows found. Edit SELECTED_PAIRS or check the input CSV."
        )

    write_markdown_table(
        all_rows=rows,
        selected_rows=selected_pair_rows,
        input_csv=input_csv,
        output_path=output_dir / "selected_results.md",
    )
    _plot_pair_metric(
        selected_pair_rows,
        metric_key="scanned_frac",
        ylabel="scanned_frac",
        title="Scanned Fraction vs n_tokens",
        output_path=output_dir / "selected_scanned_frac_vs_n.png",
    )
    _plot_pair_metric(
        selected_pair_rows,
        metric_key="speedup",
        ylabel="asymptotic speedup",
        title="Asymptotic Speedup vs n_tokens",
        output_path=output_dir / "selected_speedup_vs_n.png",
    )
    _plot_clustering_build_time(rows, output_dir / "selected_clustering_build_vs_n.png")
    _plot_enclosing_timing(rows, output_dir / "selected_enclosing_timing_vs_n.png")

    print(f"Wrote plots and Markdown table to {output_dir}")


if __name__ == "__main__":
    main()
