"""
Import and reformat existing quick_pruning results into the pruning experiment format.

Sources:
  quick_pruning/result/comparison_n_tokens_sweep/all_results.csv
      — standard (clustering × enclosing) pairs at n_tokens=6000, r=4
  quick_pruning/compare.txt
      — subspace_kcenter (louver) sweep at n_tokens=4000, r=4, S∈{2,4,8,16}

Writes results/existing_qp_standard.csv and results/existing_qp_louver.csv.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

_HERE   = Path(__file__).resolve().parent
_QP     = _HERE.parents[1] / "quick_pruning"
RESULTS = _HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

GATE_COST_DP = {
    "ball_centroid": 1.0,
    "min_enclosing_ball": 1.0,
    "span_ball": 1.0,
    "aabb": 2.0,
    "ellipsoid": 2.5,
    "outlier_aabb": 3.0,
    "outlier_ball_centroid": 2.0,
}

FIELDS = [
    "model", "n_tokens", "layer", "method_family", "method",
    "clustering", "enclosing", "S", "r", "strategy",
    "scanned_frac", "pruned_frac", "recall",
    "gate_cost_dp", "ratio", "speedup",
    "build_ms", "search_ms",
]


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"  → {path}  ({len(rows)} rows)")


# ── 1. Standard methods from all_results.csv ──────────────────────────────────

def import_standard() -> None:
    src = _QP / "result" / "comparison_n_tokens_sweep" / "all_results.csv"
    if not src.exists():
        print(f"  SKIP {src} (not found)")
        return

    rows: list[dict] = []
    with open(src) as f:
        for r in csv.DictReader(f):
            n_tokens  = int(r["n_tokens"])
            total_keys = int(r["total_keys"])
            layer     = int(r["layer"])
            rf        = int(r["bf"])
            cname     = r["clustering"]
            ename     = r["enclosing"]
            scanned   = float(r["scanned_frac"])
            pruned    = float(r["pruned_frac"])
            search_ms = float(r["search_ms"])
            build_ms  = float(r["build_ms"])
            g         = float(r["gate_cost_dp"]) if r["gate_cost_dp"] != "inf" \
                        else GATE_COST_DP.get(ename, 2.0)
            ratio     = float(r["ratio"])
            speedup   = float(r["speedup"])
            label     = f"{cname}+{ename}"
            model_tag = f"capture_qkv_{n_tokens}_meta-llama_Llama-3.2-3B-Instruct"

            rows.append({
                "model":         model_tag,
                "n_tokens":      total_keys,
                "layer":         layer,
                "method_family": "standard",
                "method":        label,
                "clustering":    cname,
                "enclosing":     ename,
                "S":             "N/A",
                "r":             rf,
                "strategy":      "N/A",
                "scanned_frac":  scanned,
                "pruned_frac":   pruned,
                "recall":        1.0,  # guaranteed by design
                "gate_cost_dp":  g,
                "ratio":         ratio,
                "speedup":       speedup,
                "build_ms":      build_ms,
                "search_ms":     search_ms,
            })

    write_csv(rows, RESULTS / "existing_qp_standard.csv")
    print(f"    Imported {len(rows)} rows (n_tokens={set(r['n_tokens'] for r in rows)})")


# ── 2. Louver results from compare.txt ────────────────────────────────────────
# Format: "sub_kcenter_S{S}_{strategy}: ... scanned={x}  pruned={y}  search={z}ms  build={b}ms  g={g}  ratio={r}  (...)"

LOUVER_RE = re.compile(
    r"sub_kcenter_S(\d+)_(\w+): .*?"
    r"scanned=([\d.]+)\s+pruned=([\d.]+)\s+"
    r"search=([\d.]+)ms\s+build=([\d.]+)ms\s+"
    r"g=([\d.]+)\s+ratio=([\d.]+)",
    re.DOTALL,
)


def import_louver() -> None:
    src = _QP / "compare.txt"
    if not src.exists():
        print(f"  SKIP {src} (not found)")
        return

    text = src.read_text()
    # Detect n_tokens from header line: "N=4461" or "n_tokens 4000"
    n_match = re.search(r"n_tokens=(\d+)|N=(\d+)", text)
    n_tokens = int(n_match.group(1) or n_match.group(2)) if n_match else 4000
    model_tag = f"capture_qkv_4000_meta-llama_Llama-3.2-3B-Instruct"

    rows: list[dict] = []
    for m in LOUVER_RE.finditer(text):
        S        = int(m.group(1))
        strategy = m.group(2)
        scanned  = float(m.group(3))
        pruned   = float(m.group(4))
        search_ms= float(m.group(5))
        build_ms = float(m.group(6))
        g        = float(m.group(7))
        ratio    = float(m.group(8))
        r        = 4  # compare.txt was run with bf=4

        rows.append({
            "model":         model_tag,
            "n_tokens":      4461,   # 4000 gen + ~461 prompt
            "layer":         15,
            "method_family": "louver",
            "method":        f"louver_S{S}_r{r}_{strategy}",
            "clustering":    "subspace_kcenter",
            "enclosing":     "ball_per_subspace",
            "S":             S,
            "r":             r,
            "strategy":      strategy,
            "scanned_frac":  scanned,
            "pruned_frac":   pruned,
            "recall":        1.0,
            "gate_cost_dp":  g,
            "ratio":         ratio,
            "speedup":       1.0 / ratio,
            "build_ms":      build_ms,
            "search_ms":     search_ms,
        })

    write_csv(rows, RESULTS / "existing_qp_louver.csv")
    print(f"    Imported {len(rows)} louver rows")


if __name__ == "__main__":
    print("Importing standard methods …")
    import_standard()
    print("Importing louver results …")
    import_louver()
    print("Done.")
