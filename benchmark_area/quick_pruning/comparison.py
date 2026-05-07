#!/usr/bin/env python3
"""
Compare clustering + enclosing methods for halfspace pruning.

For each (clustering_method, enclosing_method) pair, measures what fraction
of children must be scanned when using the parent-level gate to prune.

Usage:
    python method_comparison_bench.py [--bf 16] [--n-tokens 2000] [--n-queries 50]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pruning_bench_utils import CaptureState, _capture_qkv, _q_to_kv_map
from clusterings import CLUSTERING_METHODS
from enclosings import ENCLOSING_METHODS

# ── Model / capture settings ──
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
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


# =====================================================================
#  BENCHMARK CORE
# =====================================================================


def topk_threshold(q_normal, keys, k=20):
    """Ground-truth top-k threshold over all keys."""
    H_kv, N, D = keys.shape
    qg = q_normal.view(H_kv, -1, D)
    w = qg @ keys.transpose(-2, -1)
    w = w.reshape(q_normal.shape[0], -1)
    k = min(k, w.shape[-1])
    th, _ = w.topk(k, dim=-1)
    return th[:, -1]


def measure_scanned_fraction(gate_fn, queries, keys, q_indices, q_head_to_kv, K, bf, topk, assign=None):
    """Run queries through the gate and measure scanned fraction + search time.

    queries may be a CPU tensor; each query is moved to GPU on demand to avoid
    holding all T queries on GPU simultaneously.
    """
    H_kv, N, D = keys.shape
    device = keys.device
    fracs = []
    search_times = []

    for qi in q_indices:
        # Move only this query to GPU (queries may live on CPU)
        q = queries[:, qi, :].to(device=device, dtype=torch.float32)
        q_norm = q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q_normal = q / q_norm

        # Expand to query heads via q_head_to_kv
        q_kv = q_normal[q_head_to_kv] if q_head_to_kv is not None else q_normal

        th = topk_threshold(q_kv, keys, k=topk)

        # Gate: (H_q, K) bool — timed
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        parent_pass = gate_fn(q_kv, th)
        torch.cuda.synchronize()
        search_times.append(time.perf_counter() - t0)

        scanned = parent_pass.gather(1, assign).sum(dim=1).float()
        frac = scanned / max(1, N)
        fracs.append(frac.mean().item())

    mean_frac = sum(fracs) / len(fracs) if fracs else 1.0
    mean_search_ms = (sum(search_times) / len(search_times)) * 1000 if search_times else 0.0
    return mean_frac, mean_search_ms


def format_speedup(ratio: float) -> str:
    """Format analytical speedup relative to full scan."""
    return f"{1 / ratio:.2f}x"


def _select_methods(methods: dict[str, object], wanted: str, label: str):
    if wanted == "all":
        return methods

    names = [name.strip() for name in wanted.split(",") if name.strip()]
    selected = {}
    for name in names:
        if name not in methods:
            available = ", ".join(sorted(methods))
            raise ValueError(f"Unknown {label} method '{name}'. Available: {available}")
        selected[name] = methods[name]
    return selected


GATE_COST_DP = {
    "ball_centroid": 1.0,       # einsum (2D) + add + cmp
    "l1_ball": 1.0,             # centroid dot; ||q||_inf is query-only overhead
    "min_enclosing_ball": 1.0,  # same gate as ball
    "aabb": 2.0,                # 2 muls + max + sum (3D)
    "cone": 1.5,                # einsum + trig (~2D+20)
    "hybrid": 4.5,              # ball + AABB + cone
    "ellipsoid": 2.5,           # einsum + scaled norm (5D)
    "split_aabb": 4.0,          # 2x AABB
    "split_hybrid": 7.5,        # split_aabb + ball + ellipsoid
    "split_full_hybrid": 11.0,  # split_aabb + ball + AABB + cone + ellipsoid
    "hybrid_plus": 10.0,        # ball + AABB + cone + ellipsoid + centerline
    "quad_aabb": 8.0,           # 4x AABB
    "bisect_aabb": 5.0,         # 2x AABB + ball
    "slab_bundle": 2.0,         # projection + ball
    "pca_obb": 2.0,             # rotated AABB
    "topk_aabb_residual": 3.0,  # partial AABB + residual
    "centerline": 3.0,          # einsum + proj + residual
    "span_ball": 1.0,           # einsum + add (same as ball)
    "pair_ball": 1.0,           # exact midpoint ball for bf=2 pairs
    "outlier_ball_centroid": 2.0,  # core ball (1.0) + outlier dot (1.0)
    "outlier_span_ball": 2.0,  # core span-ball (1.0) + outlier dot (1.0)
    "axis_interval": 1.0,       # one axis projection + orth residual
    "dual_axis_interval": 2.0,  # two axis projections + orth residual
    "pca_interval": 1.0,        # one local-PCA axis projection + orth residual
    "outlier_aabb": 3.0,        # AABB on core (2.0) + dot for outlier (1.0)
    "pca_aabb_resid": 1.2,      # centroid dot (1.0) + AABB in 16d (~0.2)
    "centered_pca_d4": 0.05,
    "centered_pca_d8": 0.10,
    "centered_pca_d16": 0.19,
    "centered_pca_d32": 0.38,
    "centered_pca_d64": 0.75,
    "partial_aabb_d4": 1.03,    # mid dot (1.0) + 4 exact dims (~0.03)
    "partial_aabb_d8": 1.06,    # mid dot (1.0) + 8 exact dims
    "partial_aabb_d16": 1.12,   # mid dot (1.0) + 16 exact dims
    "partial_aabb_d32": 1.25,   # mid dot (1.0) + 32 exact dims
    "partial_aabb_d64": 1.50,   # approaches full AABB
    "cheap_outlier_aabb": 2.0,  # tight AABB + outlier norm (free)
    "cheap_outlier_ball_aabb": 3.0,  # tight AABB + centroid ball
    "pca_proj_d4": 0.05,        # 3*4/(2*128) per cluster + amortized shared
    "pca_proj_d8": 0.10,        # 3*8/(2*128) per cluster
    "pca_proj_d16": 0.19,       # 3*16/(2*128) per cluster
    "pca_proj_d32": 0.38,       # 3*32/(2*128) per cluster
    "centroid_pca_d4": 1.05,    # centroid (1.0) + PCA residual (~0.05)
    "centroid_pca_d8": 1.10,    # centroid (1.0) + PCA residual (~0.10)
    "centroid_pca_d16": 1.19,   # centroid (1.0) + PCA residual (~0.19)
}


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf", type=int, default=4, help="Branching factor")
    parser.add_argument("--n-tokens", type=int, default=2000, help="Tokens to capture")
    parser.add_argument("--n-queries", type=int, default=30, help="Number of queries to evaluate")
    parser.add_argument("--topk", type=int, default=20, help="Top-k for threshold")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--clusterings", type=str, default="all", help='Comma-separated clustering names or "all"')
    parser.add_argument("--enclosings", type=str, default="all", help='Comma-separated enclosing names or "all"')
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible comparisons")
    parser.add_argument(
        "--input-qkv",
        type=Path,
        default=None,
        help="Optional path to a saved QKV capture from debug_capture.py.",
    )
    parser.add_argument("--fp16-keys", action="store_true",
                        help="Store keys on GPU as float16 instead of float32 (~2x memory reduction)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    clustering_methods = _select_methods(CLUSTERING_METHODS, args.clusterings, "clustering")
    enclosing_methods = _select_methods(ENCLOSING_METHODS, args.enclosings, "enclosing")

    if args.input_qkv is not None:
        print(f"Loading captured QKV from {args.input_qkv} ...")
        t0 = time.perf_counter()
        capture = CaptureState.load(args.input_qkv)
        print(f"Load done in {time.perf_counter() - t0:.1f}s")
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

    # Free GPU memory used by the model — it is no longer needed.
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory after model free: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

    layer_ids = capture.layer_ids()
    layer = LAYER_IDX if LAYER_IDX in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, _ = capture.to_layer_tensors(layer)

    # Keys go to GPU; queries stay on CPU and are moved one-at-a-time during measurement.
    keys_dtype = torch.float16 if args.fp16_keys else torch.float32
    keys = keys_cpu.to(device=DEVICE, dtype=keys_dtype)
    queries = queries_cpu  # CPU tensor — moved per-query in measure_scanned_fraction
    H_kv, N, D = keys.shape
    H_q = queries.shape[0]

    q_head_to_kv = _q_to_kv_map(H_q, H_kv, DEVICE) if H_q != H_kv else None
    K = max(1, math.ceil(N / args.bf))

    # Query indices: sample from end of sequence
    total_q = queries.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride))
    q_indices = q_indices[: args.n_queries]

    keys_mb = keys.numel() * keys.element_size() / 1e6
    queries_mb = queries.numel() * queries.element_size() / 1e6
    print(f"Layer {layer}: H_kv={H_kv}, H_q={H_q}, N={N}, D={D}")
    print(f"K={K} parents (bf={args.bf}), {len(q_indices)} queries, topk={args.topk}")
    print(f"Memory: keys={keys_mb:.0f} MB on GPU ({keys.dtype}), "
          f"queries={queries_mb:.0f} MB on CPU (moved per-query)")
    print(f"GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print("=" * 90)

    results = []

    # Clustering and enclosing methods expect float32.
    # If we stored keys as float16 to save memory, cast here.
    keys_f32 = keys.float() if keys.dtype != torch.float32 else keys

    # Warn if clustering will require a large N×N distance matrix.
    # nn_greedy / fast_balanced_nn use cdist -> O(N²) memory.
    cdist_gb = N * N * 4 / 1e9
    if cdist_gb > 1.0 and any(
        name in clustering_methods for name in ("nn_greedy", "fast_balanced_nn", "block_nn")
    ):
        print(f"WARNING: N={N} — cdist-based clustering will allocate ~{cdist_gb:.1f} GB. "
              f"Use --clusterings kcenter,kmeans to avoid OOM.")


    for clust_name, clust_fn in clustering_methods.items():
        print(f"\nClustering: {clust_name} ...")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        assign, centers = clust_fn(keys_f32, args.bf)
        torch.cuda.synchronize()
        clust_time = time.perf_counter() - t0

        # Expand centers/assign to query heads if needed
        if q_head_to_kv is not None:
            assign_q = assign[q_head_to_kv]
            centers_q = centers[q_head_to_kv]
            keys_q = keys_f32[q_head_to_kv]
        else:
            assign_q = assign
            centers_q = centers
            keys_q = keys_f32

        for enc_name, enc_fn in enclosing_methods.items():
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            gate_fn, enc_info = enc_fn(keys_q, assign_q, centers_q, K, args.bf)
            torch.cuda.synchronize()
            enc_time = time.perf_counter() - t1

            frac, search_ms = measure_scanned_fraction(
                gate_fn, queries, keys_q, q_indices, None, K, args.bf, args.topk,
                assign=assign_q,
            )

            results.append({
                "clustering": clust_name,
                "enclosing": enc_name,
                "scanned_frac": frac,
                "clust_ms": clust_time * 1000,
                "enc_ms": enc_time * 1000,
                "search_ms": search_ms,
                **{f"enc_{k}": v for k, v in enc_info.items()},
            })

            pruning = 1.0 - frac
            print(
                f"  {enc_name:<20s}  scanned={frac:.4f}  pruned={pruning:.4f}  "
                f"search={search_ms:.3f}ms  "
                f"clust={clust_time*1000:.1f}ms  enc={enc_time*1000:.1f}ms  "
                + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in enc_info.items())
            )

    # ── Gate cost per enclosing method (in dot-product equivalents) ──
    # A dot product of D-dim vectors costs 2D FLOPs.
    # gate_g = gate_FLOPs_per_cluster / (2*D)
    # ── Summary table ──
    print("\n" + "=" * 120)
    print(
        f"{'CLUSTERING':<22s} {'ENCLOSING':<22s} {'SCANNED':>8s} {'PRUNED':>8s} "
        f"{'SEARCH_ms':>10s} {'BUILD_ms':>9s} {'g':>5s} {'RATIO':>7s} {'SPEEDUP':>8s}"
    )
    print("-" * 120)

    results.sort(key=lambda r: r["scanned_frac"])
    for r in results:
        build_ms = r["clust_ms"] + r["enc_ms"]
        pruned = 1.0 - r["scanned_frac"]
        g = GATE_COST_DP.get(r["enclosing"], 2.0)
        ratio = g / args.bf + (1.0 - pruned)  # g/bf + (1-p)  where p=pruned
        print(
            f"{r['clustering']:<22s} {r['enclosing']:<22s} "
            f"{r['scanned_frac']:>8.4f} {pruned:>8.4f} "
            f"{r['search_ms']:>10.3f} {build_ms:>9.1f} "
            f"{g:>5.1f} {ratio:>7.3f} {format_speedup(ratio):>8s}"
        )

    print("=" * 120)
    best = results[0]
    print(
        f"\nBest pruning: {best['clustering']} + {best['enclosing']} "
        f"-> scanned={best['scanned_frac']:.4f} (pruned {1-best['scanned_frac']:.4f})"
    )
    # Find best analytical speedup
    best_speedup = min(results, key=lambda r:
        GATE_COST_DP.get(r["enclosing"], 2.0) / args.bf + r["scanned_frac"])
    g_best = GATE_COST_DP.get(best_speedup["enclosing"], 2.0)
    ratio_best = g_best / args.bf + best_speedup["scanned_frac"]
    print(
        f"Best speedup: {best_speedup['clustering']} + {best_speedup['enclosing']} "
        f"-> ratio={ratio_best:.3f} ({format_speedup(ratio_best)})"
    )


if __name__ == "__main__":
    main()
