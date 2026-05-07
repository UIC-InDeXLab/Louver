#!/usr/bin/env python3
"""
Benchmark subspace k-center ball pruning.

Splits key dimensions into disjoint subspaces (like product quantization),
builds independent k-center + ball_centroid indexes per subspace, and prunes
by AND-filtering across all subspaces.

Compares against baselines:
  - kcenter + ball_centroid (full-dim)
  - kmeans  + ball_centroid (full-dim)

Usage:
    python comparison_subspace_kcenter.py [--bf 4] [--n-tokens 2000] [--n-queries 30]
    python comparison_subspace_kcenter.py --n-subspaces 2,4,8 --bf 4
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
from clusterings.subspace_kcenter_ball import (
    SUBSPACE_STRATEGIES,
    build_subspace_kcenter,
    project_keys_for_index,
    project_query_for_index,
    subspace_ball_gate,
    subspace_cluster_gate,
)
from clusterings import CLUSTERING_METHODS
from enclosings import ENCLOSING_METHODS

# ── Model / capture settings ──
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


def topk_threshold(q_normal, keys, k=20):
    """Ground-truth top-k threshold over all keys."""
    H_kv, N, D = keys.shape
    qg = q_normal.view(H_kv, -1, D)
    w = qg @ keys.transpose(-2, -1)
    w = w.reshape(q_normal.shape[0], -1)
    k = min(k, w.shape[-1])
    th, _ = w.topk(k, dim=-1)
    return th[:, -1]


def subspace_topk_thresholds(q_normal, keys_proj, topk, dim_slices):
    """Per-subspace thresholds from the true full-space top-k set in index space."""
    scores = torch.einsum("hd,hnd->hn", q_normal, keys_proj)
    k = min(topk, scores.shape[-1])
    topk_idx = scores.topk(k, dim=-1).indices

    thresholds = []
    for start, end in dim_slices:
        q_sub = q_normal[:, start:end]
        keys_sub = keys_proj[:, :, start:end]
        sub_scores = torch.einsum("hd,hnd->hn", q_sub, keys_sub)
        sub_topk_scores = sub_scores.gather(1, topk_idx)
        thresholds.append(sub_topk_scores.min(dim=1).values)
    return torch.stack(thresholds, dim=0)


# ── Measurement ────────────────────────────────────────────────────────


def measure_subspace_kcenter(idx, queries, keys, q_indices, q_head_to_kv, topk):
    """Measure scanned fraction for the subspace k-center per-subspace AND gate."""
    H_kv, N, D = keys.shape
    device = keys.device
    fracs = []
    search_times = []
    per_subspace_fracs = [[] for _ in range(idx.n_subspaces)]

    for qi in q_indices:
        q = queries[:, qi, :].to(device=device, dtype=torch.float32)
        q_norm = q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q_normal = q / q_norm

        keys_eval = project_keys_for_index(keys, idx, q_head_to_kv)
        q_eval = project_query_for_index(q_normal, idx, q_head_to_kv)
        th_per_subspace = subspace_topk_thresholds(
            q_eval, keys_eval, topk, idx.dim_slices
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        survive = subspace_ball_gate(idx, q_normal, th_per_subspace, q_head_to_kv)
        torch.cuda.synchronize()
        search_times.append(time.perf_counter() - t0)

        frac = survive.float().sum(dim=1) / max(1, N)
        fracs.append(frac.mean().item())

        sub_masks = subspace_cluster_gate(idx, q_normal, th_per_subspace, q_head_to_kv)
        for s in range(idx.n_subspaces):
            assign_s = idx.assigns[s][q_head_to_kv] if q_head_to_kv is not None else idx.assigns[s]
            point_pass = sub_masks[s].gather(1, assign_s)
            sf = point_pass.float().sum(dim=1) / max(1, N)
            per_subspace_fracs[s].append(sf.mean().item())

    mean_frac = sum(fracs) / len(fracs) if fracs else 1.0
    mean_search_ms = (sum(search_times) / len(search_times)) * 1000 if search_times else 0.0

    per_sub_means = []
    for s in range(idx.n_subspaces):
        vals = per_subspace_fracs[s]
        per_sub_means.append(sum(vals) / len(vals) if vals else 1.0)

    return mean_frac, mean_search_ms, per_sub_means


def measure_baseline(gate_fn, assign, queries, keys, q_indices, topk):
    """Measure scanned fraction for a standard (clustering, enclosing) baseline.

    keys, assign, and centers must already be expanded to H_q if GQA.
    """
    H, N, D = keys.shape
    device = keys.device
    fracs = []
    search_times = []

    for qi in q_indices:
        q = queries[:, qi, :].to(device=device, dtype=torch.float32)
        q_norm = q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q_normal = q / q_norm

        th = topk_threshold(q_normal, keys, k=topk)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        parent_pass = gate_fn(q_normal, th)
        torch.cuda.synchronize()
        search_times.append(time.perf_counter() - t0)

        scanned = parent_pass.gather(1, assign).sum(dim=1).float()
        frac = scanned / max(1, N)
        fracs.append(frac.mean().item())

    mean_frac = sum(fracs) / len(fracs) if fracs else 1.0
    mean_search_ms = (sum(search_times) / len(search_times)) * 1000 if search_times else 0.0
    return mean_frac, mean_search_ms


def format_speedup(ratio: float) -> str:
    return f"{1 / ratio:.2f}x"


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf", type=int, default=4)
    parser.add_argument("--n-tokens", type=int, default=2000)
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--n-subspaces", type=str, default="2,4,8",
                        help="Comma-separated subspace counts to sweep")
    parser.add_argument("--subspace-strategy", type=str, default="contiguous",
                        help='Subspace splitting strategy: "contiguous", "interleaved", '
                             '"random", "pca", or "all" to sweep all four')
    parser.add_argument("--refine-iter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-qkv", type=Path, default=None)
    parser.add_argument("--fp16-keys", action="store_true")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip baseline (kcenter, kmeans) runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    subspace_counts = [int(s.strip()) for s in args.n_subspaces.split(",")]

    if args.subspace_strategy == "all":
        strategies = list(SUBSPACE_STRATEGIES)
    else:
        strategies = [s.strip() for s in args.subspace_strategy.split(",")]

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
    K = max(1, math.ceil(N / args.bf))

    total_q = queries.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride))
    q_indices = q_indices[:args.n_queries]

    print(f"Layer {layer}: H_kv={H_kv}, H_q={H_q}, N={N}, D={D}")
    print(f"K={K} clusters (bf={args.bf}), {len(q_indices)} queries, topk={args.topk}")
    print("=" * 100)

    keys_f32 = keys.float() if keys.dtype != torch.float32 else keys
    results = []

    # ── Subspace k-center runs ──
    for strat in strategies:
        for S in subspace_counts:
            if D % S != 0:
                print(f"\nWARNING: D={D} not divisible by n_subspaces={S}, dims will be uneven")

            label = f"sub_kcenter_S{S}_{strat}"
            print(f"\n{label}: building {S} subspace indexes (K={K} each) ...")

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            idx = build_subspace_kcenter(
                keys_f32, args.bf, n_subspaces=S,
                refine_iter=args.refine_iter, strategy=strat,
            )
            torch.cuda.synchronize()
            build_ms = (time.perf_counter() - t0) * 1000

            frac, search_ms, per_sub = measure_subspace_kcenter(
                idx, queries, keys_f32, q_indices, q_head_to_kv, args.topk,
            )

            pruned = 1.0 - frac
            # Gate cost: S ball tests, each on d=D/S dims => S * (D/S) / D = 1.0 dot equiv
            g = 1.0  # S subspaces of dim D/S is the same total work as 1 full-D dot
            ratio = g / args.bf + frac

            results.append({
                "method": label,
                "scanned_frac": frac,
                "pruned": pruned,
                "build_ms": build_ms,
                "search_ms": search_ms,
                "g": g,
                "ratio": ratio,
            })

            sub_str = "  ".join(f"S{s}_scan={per_sub[s]:.4f}" for s in range(S))
            print(
                f"  scanned={frac:.4f}  pruned={pruned:.4f}  "
                f"search={search_ms:.3f}ms  build={build_ms:.1f}ms  "
                f"g={g:.1f}  ratio={ratio:.3f}  ({format_speedup(ratio)})"
            )
            print(f"  per-subspace scanned fraction: {sub_str}")

    # ── Baselines ──
    if not args.no_baselines:
        if q_head_to_kv is not None:
            keys_q = keys_f32[q_head_to_kv]
        else:
            keys_q = keys_f32

        for clust_name in ("kcenter", "kmeans"):
            if clust_name not in CLUSTERING_METHODS:
                continue
            clust_fn = CLUSTERING_METHODS[clust_name]
            enc_fn = ENCLOSING_METHODS["ball_centroid"]

            print(f"\nBaseline: {clust_name} + ball_centroid ...")

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            assign, centers = clust_fn(keys_f32, args.bf)
            torch.cuda.synchronize()
            clust_ms = (time.perf_counter() - t0) * 1000

            if q_head_to_kv is not None:
                assign_q = assign[q_head_to_kv]
                centers_q = centers[q_head_to_kv]
            else:
                assign_q = assign
                centers_q = centers

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gate_fn, enc_info = enc_fn(keys_q, assign_q, centers_q, K, args.bf)
            torch.cuda.synchronize()
            enc_ms = (time.perf_counter() - t0) * 1000

            frac, search_ms = measure_baseline(
                gate_fn, assign_q, queries, keys_q, q_indices, args.topk,
            )

            pruned = 1.0 - frac
            g = 1.0
            ratio = g / args.bf + frac

            results.append({
                "method": f"{clust_name}+ball_centroid",
                "scanned_frac": frac,
                "pruned": pruned,
                "build_ms": clust_ms + enc_ms,
                "search_ms": search_ms,
                "g": g,
                "ratio": ratio,
            })

            print(
                f"  scanned={frac:.4f}  pruned={pruned:.4f}  "
                f"search={search_ms:.3f}ms  build={clust_ms + enc_ms:.1f}ms  "
                f"g={g:.1f}  ratio={ratio:.3f}  ({format_speedup(ratio)})"
            )

    # ── Summary ──
    print("\n" + "=" * 100)
    print(
        f"{'METHOD':<35s} {'SCANNED':>8s} {'PRUNED':>8s} "
        f"{'SEARCH_ms':>10s} {'BUILD_ms':>9s} {'g':>5s} {'RATIO':>7s} {'SPEEDUP':>8s}"
    )
    print("-" * 100)

    results.sort(key=lambda r: r["scanned_frac"])
    for r in results:
        print(
            f"{r['method']:<35s} "
            f"{r['scanned_frac']:>8.4f} {r['pruned']:>8.4f} "
            f"{r['search_ms']:>10.3f} {r['build_ms']:>9.1f} "
            f"{r['g']:>5.1f} {r['ratio']:>7.3f} {format_speedup(r['ratio']):>8s}"
        )

    print("=" * 100)
    best = results[0]
    print(
        f"\nBest: {best['method']} -> scanned={best['scanned_frac']:.4f} "
        f"(pruned {best['pruned']:.4f}, {format_speedup(best['ratio'])})"
    )


if __name__ == "__main__":
    main()
