#!/usr/bin/env python3
"""
Benchmark hierarchical subspace k-center pruning.

Builds a multi-level hierarchy per subspace (contiguous dims only), runs
queries through it with top-down pruning, and reports:

  - Per-level pass rates (fraction of checked clusters that survive)
  - Cascading scanned fraction at the leaf
  - Asymptotic gate cost and speedup

The gate cost model:
  At the top level (L-1), all K_{L-1} clusters are checked.
  At level l, only children of survivors from level l+1 are checked.
  So checked(l) = survived(l+1) * bf.
  Total gate cost = sum_l checked(l), in units of cluster dot products.
  Each cluster check costs d_s dims in one subspace, so across S subspaces
  the total FLOPs per check = S * 2*d_s = 2*D (same as one full-D dot).

  Effective speedup = N / (gate_cost + scanned_points).

Usage:
    python benchmark_hierarchical.py --bf 4 --num-levels 2 --n-subspaces 4
    python benchmark_hierarchical.py --bf 8 --num-levels 3 --n-subspaces 8
    python benchmark_hierarchical.py --input-qkv capture.pt --num-levels 2,3,4
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
QUICK_PRUNING = REPO_ROOT / "benchmark_area" / "quick_pruning"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(QUICK_PRUNING) not in sys.path:
    sys.path.insert(0, str(QUICK_PRUNING))

from pruning_bench_utils import CaptureState, _capture_qkv, _q_to_kv_map
from hierarchical_subspace_kcenter import (
    build_hierarchical_index,
    hierarchical_gate,
)

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LAYER_IDX = 15
DEVICE = "cuda"
DTYPE = torch.float32

PROMPT = (
    "Solve the following problem step by step, showing all intermediate "
    "reasoning, calculations, and verification.\n\n"
    "A research lab is designing a distributed computing cluster. They have "
    "a budget for 120 machines. Each machine can be configured as a CPU node "
    "(32 cores, 128 GB RAM, $5000), a GPU node (8 cores, 64 GB RAM, 4×A100 "
    "GPUs, $35000), or a storage node (8 cores, 256 GB RAM, 100 TB disk, "
    "$12000). The workload consists of three phases that repeat in a cycle:\n\n"
    "Phase 1 (Training): Requires at least 200 A100 GPUs running in parallel. "
    "Each training job needs 4 GPUs and 48 GB RAM. Communication overhead "
    "between nodes adds 12% latency per additional node beyond the first. "
    "Calculate the optimal GPU node count to minimize total training time for "
    "a 500-epoch run where each epoch takes 45 minutes on a single 4-GPU node.\n\n"
    "Phase 2 (Data Processing): Must process 50 PB of raw data. Each CPU core "
    "can process 2 TB/hour. Storage nodes can serve data at 20 GB/s each but "
    "need 3 replicas for fault tolerance. Calculate the minimum storage and "
    "CPU nodes needed to finish processing within 72 hours.\n\n"
    "Phase 3 (Inference): Must serve 10,000 requests/second with p99 latency "
    "under 100ms. Each GPU can handle 150 requests/second. Each CPU core can "
    "handle 8 requests/second as fallback. The system must maintain 99.99% "
    "uptime, requiring N+2 redundancy.\n\n"
    "Determine the optimal allocation of the 120 machines across all three "
    "node types. Then analyze: What happens if the budget increases by 20%? "
    "What if training data doubles? What if inference load triples? For each "
    "scenario, re-derive the full allocation from scratch, show the math, "
    "compare trade-offs, and explain your reasoning at every step. Finally, "
    "prove mathematically that your allocation is Pareto-optimal across the "
    "three phases, or explain why no single allocation can be."
)


# ── Threshold computation ──────────────────────────────────────────────


def subspace_topk_thresholds(q, keys, topk, dim_slices):
    """Per-subspace thresholds from the true full-space top-k set.

    Finds the global top-k keys by full dot product, then for each subspace
    computes the minimum partial dot product among those top-k keys.
    This threshold is the tightest that guarantees no true top-k is pruned.

    Args:
        q: (H, D) unit-norm query.
        keys: (H, N, D) key vectors.
        topk: number of top keys.
        dim_slices: list of (start, end) for subspaces.

    Returns:
        (S, H) thresholds.
    """
    scores = torch.einsum("hd,hnd->hn", q, keys)  # (H, N)
    k = min(topk, scores.shape[-1])
    topk_idx = scores.topk(k, dim=-1).indices  # (H, k)

    thresholds = []
    for start, end in dim_slices:
        q_sub = q[:, start:end]
        keys_sub = keys[:, :, start:end]
        sub_scores = torch.einsum("hd,hnd->hn", q_sub, keys_sub)  # (H, N)
        sub_topk = sub_scores.gather(1, topk_idx)  # (H, k)
        thresholds.append(sub_topk.min(dim=1).values)  # (H,)

    return torch.stack(thresholds, dim=0)  # (S, H)


def exact_topk_mask(q, keys, topk):
    """(H, N) bool mask of the true top-k keys."""
    scores = torch.einsum("hd,hnd->hn", q, keys)
    k = min(topk, scores.shape[-1])
    topk_idx = scores.topk(k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, topk_idx, True)
    return mask


# ── Measurement ────────────────────────────────────────────────────────


def measure_hierarchical(
    idx, queries, keys, q_indices, q_head_to_kv, topk,
):
    """Run queries through the hierarchical gate and collect stats.

    Returns:
        mean_frac: mean scanned fraction across queries.
        mean_search_ms: mean wall-clock gate time.
        avg_level_stats: per-level average stats across queries and subspaces.
        missed: total number of true top-k points missed (should be 0).
    """
    H_kv, N, D = keys.shape
    device = keys.device
    fracs = []
    search_times = []
    total_missed = 0

    # Accumulate per-level stats. Levels are ordered top-down in the stats output.
    num_levels = idx.num_levels
    S = idx.n_subspaces
    level_checked_accum = [[[] for _ in range(num_levels)] for _ in range(S)]
    level_survived_accum = [[[] for _ in range(num_levels)] for _ in range(S)]
    level_pass_rate_accum = [[[] for _ in range(num_levels)] for _ in range(S)]

    for qi in q_indices:
        q = queries[:, qi, :].to(device=device, dtype=torch.float32)
        q_norm = q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q_normal = q / q_norm

        # Expand keys to H_q for threshold & correctness check
        if q_head_to_kv is not None:
            keys_expanded = keys[q_head_to_kv]
        else:
            keys_expanded = keys

        th_per_sub = subspace_topk_thresholds(
            q_normal, keys_expanded, topk, idx.dim_slices,
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        survive, all_stats = hierarchical_gate(
            idx, q_normal, th_per_sub, q_head_to_kv,
        )
        torch.cuda.synchronize()
        search_times.append(time.perf_counter() - t0)

        frac = survive.float().sum(dim=1) / max(1, N)
        fracs.append(frac.mean().item())

        # Check correctness
        topk_mask = exact_topk_mask(q_normal, keys_expanded, topk)
        missed = (topk_mask & ~survive).sum().item()
        total_missed += int(missed)

        # Collect level stats
        for s in range(S):
            for li, st in enumerate(all_stats[s]):
                level_checked_accum[s][li].append(st["checked"])
                level_survived_accum[s][li].append(st["survived"])
                level_pass_rate_accum[s][li].append(st["pass_rate"])

    mean_frac = sum(fracs) / len(fracs) if fracs else 1.0
    mean_search_ms = (sum(search_times) / len(search_times)) * 1000 if search_times else 0.0

    # Average level stats across queries, then across subspaces
    avg_level_stats = []
    for li in range(num_levels):
        checked_all_sub = []
        survived_all_sub = []
        pass_rate_all_sub = []
        for s in range(S):
            vals = level_checked_accum[s][li]
            checked_all_sub.append(sum(vals) / len(vals) if vals else 0)
            vals_surv = level_survived_accum[s][li]
            survived_all_sub.append(sum(vals_surv) / len(vals_surv) if vals_surv else 0)
            vals_pr = level_pass_rate_accum[s][li]
            pass_rate_all_sub.append(sum(vals_pr) / len(vals_pr) if vals_pr else 1.0)

        # Level index in the hierarchy (top-down in stats)
        # all_stats[s][0] is the top level, all_stats[s][-1] is level 0 (leaf)
        # We report in top-down order.
        if level_checked_accum[0][li]:
            actual_level = int(all_stats[0][li]["level"])
        else:
            actual_level = num_levels - 1 - li

        avg_level_stats.append({
            "level": actual_level,
            "mean_checked": sum(checked_all_sub) / S,
            "mean_survived": sum(survived_all_sub) / S,
            "mean_pass_rate": sum(pass_rate_all_sub) / S,
        })

    return mean_frac, mean_search_ms, avg_level_stats, total_missed


def compute_asymptotic_speedup(
    N: int,
    bf: int,
    num_levels: int,
    S: int,
    D: int,
    level_stats: list[dict],
    measured_scanned_frac: float,
):
    """Compute effective speedup from measured per-level pruning.

    For each level, the gate work is proportional to the fraction that survived
    down to that level, which is exactly captured by `mean_checked`. We sum that
    gate work across levels, then add the actual final scanned points after the
    AND across subspaces. All statistics are already averaged over heads,
    queries, and subspaces in `measure_hierarchical`.

    Returns:
        dict with gate_cost, scan_cost, ratio, speedup, and per-level breakdown.
    """
    # Cluster counts at each level
    K = [0] * num_levels
    K[0] = max(1, math.ceil(N / bf))
    for l in range(1, num_levels):
        K[l] = max(1, math.ceil(K[l - 1] / bf))

    breakdown = []
    total_gate_clusters = 0.0

    for st in level_stats:
        lvl = st["level"]
        checked = float(st["mean_checked"])
        survived = float(st["mean_survived"])
        pass_rate = st["mean_pass_rate"]
        total_gate_clusters += checked

        breakdown.append({
            "level": lvl,
            "K": K[lvl],
            "checked": checked,
            "survived": survived,
            "pass_rate": pass_rate,
            "checked_frac": checked / max(1.0, float(K[lvl])),
            "filtered_frac": 1.0 - (checked / max(1.0, float(K[lvl]))),
        })

    scanned_frac = measured_scanned_frac
    scanned_points = scanned_frac * N

    # Gate cost in dot-product equivalents:
    # Each cluster check across S subspaces = S * (2 * d_s) FLOPs = 2D FLOPs = 1 dot
    gate_cost_dots = total_gate_clusters  # in dot-product equivalents

    # Full scan cost = N dot products
    # Total effective cost = gate_cost + scanned_points (dot products on scanned keys)
    ratio = (gate_cost_dots + scanned_points) / N
    speedup = 1.0 / ratio if ratio > 0 else float("inf")

    return {
        "gate_cost_dots": gate_cost_dots,
        "scanned_points": scanned_points,
        "scanned_frac": scanned_frac,
        "ratio": ratio,
        "speedup": speedup,
        "breakdown": breakdown,
    }


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf", type=int, default=4)
    parser.add_argument("--num-levels", type=str, default="2,3",
                        help="Comma-separated level counts to sweep")
    parser.add_argument("--n-subspaces", type=str, default="4",
                        help="Comma-separated subspace counts to sweep")
    parser.add_argument("--n-tokens", type=int, default=2000)
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--refine-iter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-qkv", type=Path, default=None)
    parser.add_argument("--fp16-keys", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    level_counts = [int(x.strip()) for x in args.num_levels.split(",")]
    subspace_counts = [int(x.strip()) for x in args.n_subspaces.split(",")]

    # ── Capture / load QKV ──
    if args.input_qkv is not None:
        print(f"Loading captured QKV from {args.input_qkv} ...")
        capture = CaptureState.load(args.input_qkv)
    else:
        print(f"Capturing {args.n_tokens} tokens from {args.model} ...")
        t0 = time.perf_counter()
        capture = _capture_qkv(
            model_name=args.model,
            prompt_text=PROMPT,
            n=args.n_tokens,
            device=DEVICE,
            torch_dtype=DTYPE,
            show_progress=True,
        )
        print(f"Capture done in {time.perf_counter() - t0:.1f}s")

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    layer_ids = capture.layer_ids()
    layer = LAYER_IDX if LAYER_IDX in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, _ = capture.to_layer_tensors(layer)

    keys_dtype = torch.float16 if args.fp16_keys else torch.float32
    keys = keys_cpu.to(device=DEVICE, dtype=keys_dtype)
    queries = queries_cpu
    H_kv, N, D = keys.shape
    H_q = queries.shape[0]

    q_head_to_kv = _q_to_kv_map(H_q, H_kv, DEVICE) if H_q != H_kv else None
    keys_f32 = keys.float() if keys.dtype != torch.float32 else keys

    total_q = queries.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride))
    q_indices = q_indices[:args.n_queries]

    print(f"Layer {layer}: H_kv={H_kv}, H_q={H_q}, N={N}, D={D}")
    print(f"bf={args.bf}, {len(q_indices)} queries, topk={args.topk}")
    print("=" * 110)

    results = []

    for S in subspace_counts:
        for L in level_counts:
            label = f"hier_S{S}_L{L}"

            # Check if hierarchy makes sense: need K_{top} >= 1
            K_check = N
            for _ in range(L):
                K_check = max(1, math.ceil(K_check / args.bf))
            if K_check <= 1 and L > 1:
                print(f"\n{label}: SKIP — top level has only {K_check} cluster "
                      f"(N={N}, bf={args.bf}, levels={L})")
                continue

            print(f"\n{label}: building {L}-level hierarchy with {S} subspaces ...")

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            idx = build_hierarchical_index(
                keys_f32, args.bf, n_subspaces=S, num_levels=L,
                refine_iter=args.refine_iter,
            )
            torch.cuda.synchronize()
            build_ms = (time.perf_counter() - t0) * 1000

            # Print hierarchy structure
            for s in range(S):
                dims = idx.dim_slices[s]
                structure = " → ".join(
                    f"L{lvl.K}" for lvl in reversed(idx.levels[s])
                )
                if s == 0:
                    print(f"  subspace dims={dims}: {structure} → {N} points")

            frac, search_ms, level_stats, missed = measure_hierarchical(
                idx, queries, keys_f32, q_indices, q_head_to_kv, args.topk,
            )

            speedup_info = compute_asymptotic_speedup(
                N, args.bf, L, S, D, level_stats, frac,
            )

            results.append({
                "label": label,
                "S": S,
                "L": L,
                "scanned_frac": frac,
                "build_ms": build_ms,
                "search_ms": search_ms,
                "missed": missed,
                **speedup_info,
            })

            pruned = 1.0 - frac
            print(f"  build={build_ms:.1f}ms  search={search_ms:.3f}ms  "
                  f"missed={missed}")
            print(f"  scanned={frac:.4f}  pruned={pruned:.4f}")

            # Per-level breakdown
            print(f"\n  {'LEVEL':>5s}  {'K':>6s}  {'CHECKED':>8s}  "
                  f"{'SURVIVED':>9s}  {'PASS_RATE':>9s}")
            print(f"  {'-'*45}")
            for b in speedup_info["breakdown"]:
                print(f"  {b['level']:>5d}  {b['K']:>6d}  {b['checked']:>8.1f}  "
                      f"{b['survived']:>9.1f}  {b['pass_rate']:>9.4f}")

            print(f"\n  Asymptotic model:")
            print(f"    gate_cost   = {speedup_info['gate_cost_dots']:.1f} dot-equiv")
            print(f"    scan_cost   = {speedup_info['scanned_points']:.1f} points")
            print(f"    total / N   = {speedup_info['ratio']:.4f}")
            print(f"    speedup     = {speedup_info['speedup']:.2f}x")

    # ── Summary ──
    print("\n" + "=" * 110)
    print(
        f"{'METHOD':<20s} {'S':>3s} {'L':>3s} {'SCANNED':>8s} {'PRUNED':>8s} "
        f"{'GATE':>8s} {'SCAN':>8s} {'RATIO':>7s} {'SPEEDUP':>8s} "
        f"{'SEARCH_ms':>10s} {'BUILD_ms':>9s} {'MISSED':>7s}"
    )
    print("-" * 110)

    results.sort(key=lambda r: r["ratio"])
    for r in results:
        pruned = 1.0 - r["scanned_frac"]
        print(
            f"{r['label']:<20s} {r['S']:>3d} {r['L']:>3d} "
            f"{r['scanned_frac']:>8.4f} {pruned:>8.4f} "
            f"{r['gate_cost_dots']:>8.1f} {r['scanned_points']:>8.1f} "
            f"{r['ratio']:>7.4f} {r['speedup']:>7.2f}x "
            f"{r['search_ms']:>10.3f} {r['build_ms']:>9.1f} {r['missed']:>7d}"
        )

    print("=" * 110)
    if results:
        best = results[0]
        print(
            f"\nBest asymptotic speedup: {best['label']} → "
            f"ratio={best['ratio']:.4f} ({best['speedup']:.2f}x), "
            f"scanned={best['scanned_frac']:.4f}"
        )


if __name__ == "__main__":
    main()
