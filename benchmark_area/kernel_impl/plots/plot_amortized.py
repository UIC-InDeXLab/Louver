"""Plot amortized attention time vs decoding step from a bench CSV.

Compares our fused attention (with amortized update cost) against the two
dense attention baselines: `baseline_attention` (einsum + softmax) and
`torch.nn.functional.scaled_dot_product_attention` (SDPA).

Output goes to kernel_impl/reports/<csv_stem>_amortized.png.

Usage:
    python -m hira.benchmark_area.kernel_impl.plots.plot_amortized \\
        kernel_impl/reports/bench_inc.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def _default_csv() -> Path:
    """Most recently modified bench_*.csv under reports/."""
    candidates = sorted(
        REPORTS_DIR.glob("bench_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No bench_*.csv found in {REPORTS_DIR}. Run bench.py first "
            f"or pass a CSV path explicitly."
        )
    return candidates[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path, nargs="?", default=None,
                   help="Path to bench CSV. Defaults to the most recently "
                        "modified reports/bench_*.csv.")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--skip-warmup", type=int, default=1,
                   help="Drop the first N steps from the plot. The first step "
                        "right after prefill carries CUDA compile/alloc overhead "
                        "and distorts the y-axis. Default: 1.")
    args = p.parse_args()

    csv_path = args.csv if args.csv is not None else _default_csv()
    rows = load_csv(csv_path)
    if not rows:
        raise RuntimeError(f"Empty CSV: {csv_path}")

    rows = rows[args.skip_warmup:]
    if not rows:
        raise RuntimeError(
            f"No rows left after skipping {args.skip_warmup} warmup step(s)."
        )

    steps = [int(r["step"]) for r in rows]
    amort = [float(r["amortized_ours_ms"]) for r in rows]
    attend = [float(r["attend_ours_ms"]) for r in rows]
    dense = [float(r["dense_attn_ms"]) for r in rows]
    sdpa = [float(r["sdpa_ms"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, amort, label="ours (amortized attend+update)", linewidth=1.5)
    ax.plot(steps, attend, label="ours (attend only)", linestyle="--", alpha=0.7)
    ax.plot(steps, dense, label="dense attention (einsum + softmax)",
            linewidth=1.5, alpha=0.7)
    ax.plot(steps, sdpa, label="SDPA (torch F.scaled_dot_product_attention)",
            linewidth=1.5, alpha=0.7)
    ax.set_xlabel("decoding step")
    ax.set_ylabel("time (ms)")
    ax.set_title(f"Amortized attention time — {csv_path.name}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out = args.out or (REPORTS_DIR / f"{csv_path.stem}_amortized.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
