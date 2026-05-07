"""
Pruning Power Ablation — Experiments 4 & 4.1.

Measures for each index design:
  scanned_frac  — fraction of keys that pass the gate (empirically measured)
  pruned_frac   — 1 - scanned_frac
  gate_cost_dp  — gate evaluation cost in dot-product equivalents (analytical)
  ratio         — g/r + scanned_frac  (asymptotic cost vs full scan)
  speedup       — 1/ratio
  recall        — fraction of true top-k keys retrieved (always 1.000 by design)
  build_ms      — index build time (one-off)
  search_ms     — per-decode-step gate evaluation time

Two method families:
  1. standard: (clustering × enclosing) pairs — kcenter, kmeans, pq_subspace × ball, aabb, ...
  2. louver:   subspace_kcenter with ablation over S (n_subspaces) and r (branching factor)

Asymptotic cost formula
  ratio = g/r + scanned_frac
  • g   = gate cost per cluster in dot-product equivalents
  • r   = children per cluster (branching factor)
  • scanned_frac = fraction of all keys that pass the gate

Usage:
    python bench.py --input-qkv PATH/capture_qkv_8000_meta-llama_Llama-3.2-3B-Instruct.pt
    python bench.py --input-qkv PATH.pt --mode louver   --r-values 2,4,8,16 --S-values 2,4,8,16
    python bench.py --input-qkv PATH.pt --mode standard --r-values 4,8,16
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_QP_DIR  = _HERE.parents[1] / "quick_pruning"  # benchmark_area/quick_pruning
_REPO    = _HERE.parents[3]                      # repo root (contains hira/)

for p in (_QP_DIR, _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pruning_bench_utils import CaptureState, _q_to_kv_map
from clusterings import CLUSTERING_METHODS
from enclosings import ENCLOSING_METHODS
from clusterings.subspace_kcenter_ball import (
    build_subspace_kcenter,
    project_keys_for_index,
    project_query_for_index,
    subspace_ball_gate,
)

# ── Gate cost table (dot-product equivalents per cluster) ─────────────────────
# A "dot product" here means one D-dim inner product (2D FLOPs).
# g values copied from quick_pruning/comparison.py.
GATE_COST_DP: dict[str, float] = {
    "ball_centroid":        1.0,
    "min_enclosing_ball":   1.0,
    "span_ball":            1.0,
    "aabb":                 2.0,
    "ellipsoid":            2.5,
    "outlier_aabb":         3.0,
    "outlier_ball_centroid":2.0,
    "fp16_aabb":            2.0,
    "partial_aabb_d8":      1.06,
    "partial_aabb_d16":     1.12,
    "partial_aabb_d32":     1.25,
    "partial_aabb_d64":     1.50,
}

# ── Selected method sets for the paper table ──────────────────────────────────
# Standard: representative (clustering, enclosing) pairs
STANDARD_CLUSTERINGS = ["kcenter", "kmeans", "pq_subspace", "batch_nn"]
STANDARD_ENCLOSINGS  = ["ball_centroid", "aabb", "span_ball"]

# Louver default: contiguous subspace split (best empirically in compare.txt)
LOUVER_STRATEGY = "contiguous"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _topk_threshold(q: torch.Tensor, keys: torch.Tensor, k: int) -> torch.Tensor:
    """Exact top-k threshold per query head. q: (H_q, D), keys: (H_q, N, D)."""
    scores = torch.einsum("hd,hnd->hn", q, keys)
    k = min(k, scores.shape[-1])
    return scores.topk(k, dim=-1).values[:, -1]


def _topk_indices(q: torch.Tensor, keys: torch.Tensor, k: int) -> torch.Tensor:
    """Exact top-k key indices per head. Returns (H_q, k)."""
    scores = torch.einsum("hd,hnd->hn", q, keys)
    k = min(k, scores.shape[-1])
    return scores.topk(k, dim=-1).indices


def _subspace_thresholds(q: torch.Tensor, keys_proj: torch.Tensor,
                          topk: int, dim_slices: list[tuple[int, int]]) -> torch.Tensor:
    """Per-subspace min scores of the exact top-k set. Returns (S, H_q)."""
    scores = torch.einsum("hd,hnd->hn", q, keys_proj)
    k = min(topk, scores.shape[-1])
    topk_idx = scores.topk(k, dim=-1).indices  # (H_q, k)

    taus = []
    for start, end in dim_slices:
        q_sub   = q[:, start:end]
        k_sub   = keys_proj[:, :, start:end]
        sub_sc  = torch.einsum("hd,hnd->hn", q_sub, k_sub)
        top_sub = sub_sc.gather(1, topk_idx)          # (H_q, k)
        taus.append(top_sub.min(dim=1).values)         # (H_q,)
    return torch.stack(taus, dim=0)  # (S, H_q)


def _select_q_indices(total: int, n: int) -> list[int]:
    stride = max(1, total // n)
    idx = list(range(total - 1, max(0, total - n * stride) - 1, -stride))
    return idx[:n]


# ── Standard method measurement ───────────────────────────────────────────────

def measure_standard(
    gate_fn,
    assign_q: torch.Tensor,    # (H_q, N) int64
    keys_q: torch.Tensor,      # (H_q, N, D)
    queries: torch.Tensor,     # (H_q, T, D) CPU
    q_indices: list[int],
    topk: int,
) -> tuple[float, float, float]:
    """Returns (scanned_frac, recall, search_ms)."""
    H_q, N, D = keys_q.shape
    device = keys_q.device
    fracs, recalls, times = [], [], []

    for qi in q_indices:
        q = queries[:, qi, :].to(device=device, dtype=torch.float32)
        q_norm = q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q_n = q / q_norm

        th = _topk_threshold(q_n, keys_q, topk)
        topk_idx = _topk_indices(q_n, keys_q, topk)  # (H_q, k)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        parent_pass = gate_fn(q_n, th)              # (H_q, K)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        point_pass = parent_pass.gather(1, assign_q)  # (H_q, N)

        frac = point_pass.float().mean(dim=1).mean().item()
        fracs.append(frac)

        # Recall: fraction of top-k indices that are in point_pass
        hit = point_pass.gather(1, topk_idx).float().mean().item()
        recalls.append(hit)

    return (
        float(sum(fracs)   / len(fracs)),
        float(sum(recalls) / len(recalls)),
        float(sum(times)   / len(times)) * 1000,
    )


# ── Louver measurement ────────────────────────────────────────────────────────

def measure_louver(
    idx,
    keys_f32: torch.Tensor,    # (H_kv, N, D)
    queries: torch.Tensor,     # (H_q, T, D) CPU
    q_indices: list[int],
    q_head_to_kv: torch.Tensor | None,
    topk: int,
) -> tuple[float, float, float]:
    """Returns (scanned_frac, recall, search_ms)."""
    H_kv, N, D = keys_f32.shape
    device = keys_f32.device
    H_q = queries.shape[0]
    fracs, recalls, times = [], [], []

    keys_eval = project_keys_for_index(keys_f32, idx, q_head_to_kv)  # (H_q, N, D_idx)
    keys_full = keys_f32[q_head_to_kv] if q_head_to_kv is not None else keys_f32

    for qi in q_indices:
        q = queries[:, qi, :].to(device=device, dtype=torch.float32)
        q_norm = q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q_n = q / q_norm

        q_eval = project_query_for_index(q_n, idx, q_head_to_kv)
        tau_per_sub = _subspace_thresholds(q_eval, keys_eval, topk, idx.dim_slices)

        # Ground-truth top-k for recall
        topk_idx = _topk_indices(q_n, keys_full, topk)  # (H_q, k)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        survive = subspace_ball_gate(idx, q_n, tau_per_sub, q_head_to_kv)  # (H_q, N)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        frac = survive.float().mean(dim=1).mean().item()
        fracs.append(frac)

        hit = survive.gather(1, topk_idx).float().mean().item()
        recalls.append(hit)

    return (
        float(sum(fracs)   / len(fracs)),
        float(sum(recalls) / len(recalls)),
        float(sum(times)   / len(times)) * 1000,
    )


# ── CSV writer ────────────────────────────────────────────────────────────────

FIELDS = [
    "model", "n_tokens", "layer", "method_family", "method",
    "clustering", "enclosing", "S", "r", "strategy",
    "scanned_frac", "pruned_frac", "recall",
    "gate_cost_dp", "ratio", "speedup",
    "build_ms", "search_ms",
]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({
                k: (f"{v:.6f}" if isinstance(v, float) else v)
                for k, v in row.items()
            })
    print(f"  → {path}  ({len(rows)} rows)")


def print_table(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r["ratio"])
    hdr = (f"{'METHOD':<30s} {'r':>3s} {'S':>3s}  "
           f"{'SCANNED':>8s} {'PRUNED':>8s} {'RECALL':>7s}  "
           f"{'g':>5s} {'RATIO':>7s} {'SPEEDUP':>8s}  "
           f"{'SEARCH_ms':>10s} {'BUILD_ms':>9s}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        S_str = str(r["S"]) if r["S"] != "N/A" else "  -"
        print(
            f"{r['method']:<30s} {r['r']:>3}  {S_str:>3}  "
            f"{r['scanned_frac']:>8.4f} {r['pruned_frac']:>8.4f} {r['recall']:>7.4f}  "
            f"{r['gate_cost_dp']:>5.2f} {r['ratio']:>7.4f} {1/r['ratio']:>7.2f}x  "
            f"{r['search_ms']:>10.3f} {r['build_ms']:>9.1f}"
        )
    print("=" * len(hdr))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-qkv", type=Path, required=True,
                    help="Captured .pt file (quick_pruning/capture.py format)")
    ap.add_argument("--layer",      type=int,   default=None,
                    help="Layer index to use (default: middle captured layer)")
    ap.add_argument("--n-queries",  type=int,   default=30)
    ap.add_argument("--topk",       type=int,   default=20,
                    help="k for top-k recall / threshold")
    ap.add_argument("--mode",       default="all",
                    choices=["all", "louver", "standard"],
                    help="Which method families to benchmark")
    # Ablation axes
    ap.add_argument("--r-values",   default="2,4,8,16",
                    help="Branching factors r to sweep")
    ap.add_argument("--S-values",   default="2,4,8,16",
                    help="Subspace counts S to sweep (louver only)")
    ap.add_argument("--strategy",   default=LOUVER_STRATEGY,
                    help="Subspace split strategy for louver")
    ap.add_argument("--refine-iter",type=int,   default=5)
    ap.add_argument("--clusterings",default=",".join(STANDARD_CLUSTERINGS),
                    help="Comma-separated clustering names for standard mode")
    ap.add_argument("--enclosings", default=",".join(STANDARD_ENCLOSINGS),
                    help="Comma-separated enclosing names for standard mode")
    ap.add_argument("--output-dir", type=Path,
                    default=_HERE / "results",
                    help="Directory for output CSV")
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    r_values = [int(x) for x in args.r_values.split(",")]
    S_values = [int(x) for x in args.S_values.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load capture ──────────────────────────────────────────────────────────
    print(f"Loading capture from {args.input_qkv} …")
    capture = CaptureState.load(args.input_qkv)
    layer_ids = capture.layer_ids()
    layer = args.layer if (args.layer is not None and args.layer in layer_ids) \
            else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, _ = capture.to_layer_tensors(layer)

    keys  = keys_cpu.to(device=device, dtype=torch.float32)
    H_kv, N, D = keys.shape
    H_q   = queries_cpu.shape[0]
    q_head_to_kv = _q_to_kv_map(H_q, H_kv, device) if H_q != H_kv else None

    q_indices = _select_q_indices(queries_cpu.shape[1], args.n_queries)
    model_tag = args.input_qkv.stem  # e.g. capture_qkv_8000_meta-llama_Llama-3.2-3B-Instruct

    print(f"Layer {layer}: H_kv={H_kv}, H_q={H_q}, N={N}, D={D}")
    print(f"Queries: {len(q_indices)}, topk={args.topk}")

    # For standard methods: expand keys to query heads
    keys_q = keys[q_head_to_kv] if q_head_to_kv is not None else keys  # (H_q, N, D)

    rows: list[dict] = []

    # ── Standard (clustering × enclosing) ─────────────────────────────────────
    if args.mode in ("all", "standard"):
        clust_names = [x.strip() for x in args.clusterings.split(",") if x.strip()]
        enc_names   = [x.strip() for x in args.enclosings.split(",")   if x.strip()]

        for r in r_values:
            K = max(1, math.ceil(N / r))
            print(f"\n── Standard  r={r}  K={K} ──")

            for cname in clust_names:
                if cname not in CLUSTERING_METHODS:
                    print(f"  SKIP unknown clustering: {cname}")
                    continue
                clust_fn = CLUSTERING_METHODS[cname]

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                assign, centers = clust_fn(keys, r)
                torch.cuda.synchronize()
                clust_ms = (time.perf_counter() - t0) * 1000

                assign_q  = assign[q_head_to_kv]  if q_head_to_kv is not None else assign
                centers_q = centers[q_head_to_kv] if q_head_to_kv is not None else centers

                for ename in enc_names:
                    if ename not in ENCLOSING_METHODS:
                        print(f"  SKIP unknown enclosing: {ename}")
                        continue
                    enc_fn = ENCLOSING_METHODS[ename]
                    g = GATE_COST_DP.get(ename, 2.0)

                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    gate_fn, _ = enc_fn(keys_q, assign_q, centers_q, K, r)
                    torch.cuda.synchronize()
                    enc_ms = (time.perf_counter() - t1) * 1000

                    scanned, recall, search_ms = measure_standard(
                        gate_fn, assign_q, keys_q, queries_cpu, q_indices, args.topk,
                    )
                    pruned = 1.0 - scanned
                    ratio  = g / r + scanned
                    label  = f"{cname}+{ename}"

                    print(f"  {label:<32s}  scanned={scanned:.4f}  pruned={pruned:.4f}  "
                          f"recall={recall:.4f}  ratio={ratio:.4f}  ({1/ratio:.2f}x)  "
                          f"search={search_ms:.3f}ms")

                    rows.append({
                        "model":         model_tag,
                        "n_tokens":      N,
                        "layer":         layer,
                        "method_family": "standard",
                        "method":        label,
                        "clustering":    cname,
                        "enclosing":     ename,
                        "S":             "N/A",
                        "r":             r,
                        "strategy":      "N/A",
                        "scanned_frac":  scanned,
                        "pruned_frac":   pruned,
                        "recall":        recall,
                        "gate_cost_dp":  g,
                        "ratio":         ratio,
                        "speedup":       1.0 / ratio,
                        "build_ms":      clust_ms + enc_ms,
                        "search_ms":     search_ms,
                    })

    # ── Louver (subspace_kcenter) ablation ────────────────────────────────────
    if args.mode in ("all", "louver"):
        print(f"\n── Louver ({args.strategy})  S×r ablation ──")

        for r in r_values:
            for S in S_values:
                K = max(1, math.ceil(N / r))
                label = f"louver_S{S}_r{r}"
                print(f"\n  {label}: S={S}, r={r}, K={K} …", end="", flush=True)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                idx = build_subspace_kcenter(
                    keys, r,
                    n_subspaces=S,
                    refine_iter=args.refine_iter,
                    strategy=args.strategy,
                )
                torch.cuda.synchronize()
                build_ms = (time.perf_counter() - t0) * 1000

                g = 1.0  # S × (D/S) = D total dims = 1 full dot equivalent

                scanned, recall, search_ms = measure_louver(
                    idx, keys, queries_cpu, q_indices, q_head_to_kv, args.topk,
                )
                pruned = 1.0 - scanned
                ratio  = g / r + scanned

                print(f"  scanned={scanned:.4f}  pruned={pruned:.4f}  "
                      f"recall={recall:.4f}  ratio={ratio:.4f}  ({1/ratio:.2f}x)  "
                      f"search={search_ms:.3f}ms  build={build_ms:.0f}ms")

                rows.append({
                    "model":         model_tag,
                    "n_tokens":      N,
                    "layer":         layer,
                    "method_family": "louver",
                    "method":        label,
                    "clustering":    "subspace_kcenter",
                    "enclosing":     "ball_per_subspace",
                    "S":             S,
                    "r":             r,
                    "strategy":      args.strategy,
                    "scanned_frac":  scanned,
                    "pruned_frac":   pruned,
                    "recall":        recall,
                    "gate_cost_dp":  g,
                    "ratio":         ratio,
                    "speedup":       1.0 / ratio,
                    "build_ms":      build_ms,
                    "search_ms":     search_ms,
                })

    # ── Output ────────────────────────────────────────────────────────────────
    print_table(rows)

    stem = model_tag.replace("capture_qkv_", "").replace("_", "-")
    out_path = args.output_dir / f"{stem}_pruning_ablation.csv"
    write_csv(rows, out_path)


if __name__ == "__main__":
    main()
