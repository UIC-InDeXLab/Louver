#!/usr/bin/env python3
"""Benchmark subspace k-center pruning with OR filtering only.

The index is the same subspace k-center + ball-centroid index used by
``comparison_subspace_kcenter.py``.  For each query, thresholds are derived
from the exact full-space top-k keys: gather the true top-k set over all keys,
then use the minimum score inside that set as the threshold for each subspace.

The online gate is OR across subspaces: a key is scanned if any subspace's
cluster upper bound passes its corresponding threshold.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (THIS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pruning_bench_utils import CaptureState, _capture_qkv, _q_to_kv_map
from clusterings.subspace_kcenter_ball import (
    build_subspace_kcenter,
    project_keys_for_index,
    project_query_for_index,
    subspace_cluster_gate,
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
    "(32 cores, 128 GB RAM, $5000), a GPU node (8 cores, 64 GB RAM, 4xA100 "
    "GPUs, $35000), or a storage node (8 cores, 256 GB RAM, 100 TB disk, "
    "$12000). The workload consists of three phases that repeat in a cycle. "
    "Determine the optimal allocation and analyze changed-budget, doubled-data, "
    "and tripled-inference scenarios."
)


def subspace_topk_thresholds(q_normal, keys_proj, topk, dim_slices):
    """Per-subspace thresholds from exact full-space top-k keys."""
    scores = torch.einsum("hd,hnd->hn", q_normal, keys_proj)
    k = min(topk, scores.shape[-1])
    topk_idx = scores.topk(k, dim=-1).indices

    thresholds = []
    for start, end in dim_slices:
        sub_scores = torch.einsum(
            "hd,hnd->hn",
            q_normal[:, start:end],
            keys_proj[:, :, start:end],
        )
        topk_sub_scores = sub_scores.gather(1, topk_idx)
        thresholds.append(topk_sub_scores.min(dim=1).values)
    return torch.stack(thresholds, dim=0)


def subspace_ball_gate_or(idx, q_normal, th_per_subspace, q_head_to_kv=None):
    """OR-filter point mask from subspace cluster gates."""
    h_q = q_normal.shape[0]
    n = idx.assigns[0].shape[1]
    survive = torch.zeros(h_q, n, dtype=torch.bool, device=q_normal.device)

    sub_masks = subspace_cluster_gate(idx, q_normal, th_per_subspace, q_head_to_kv)
    for s, cluster_pass in enumerate(sub_masks):
        assign_s = (
            idx.assigns[s][q_head_to_kv] if q_head_to_kv is not None else idx.assigns[s]
        )
        survive |= cluster_pass.gather(1, assign_s)
    return survive, sub_masks


def measure_subspace_kcenter_or(idx, queries, keys, q_indices, q_head_to_kv, topk):
    """Measure scanned/pruned fractions for OR-filter subspace k-center."""
    _, n, _ = keys.shape
    device = keys.device
    fracs = []
    search_times = []
    per_subspace_fracs = [[] for _ in range(idx.n_subspaces)]

    keys_eval = project_keys_for_index(keys, idx, q_head_to_kv)

    for qi in q_indices:
        q = queries[:, qi, :].to(device=device, dtype=torch.float32)
        q_normal = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        q_eval = project_query_for_index(q_normal, idx, q_head_to_kv)
        th_per_subspace = subspace_topk_thresholds(
            q_eval, keys_eval, topk, idx.dim_slices
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        survive, sub_masks = subspace_ball_gate_or(
            idx, q_normal, th_per_subspace, q_head_to_kv
        )
        torch.cuda.synchronize()
        search_times.append(time.perf_counter() - t0)

        frac = survive.float().sum(dim=1) / max(1, n)
        fracs.append(frac.mean().item())

        for s in range(idx.n_subspaces):
            assign_s = (
                idx.assigns[s][q_head_to_kv]
                if q_head_to_kv is not None
                else idx.assigns[s]
            )
            point_pass = sub_masks[s].gather(1, assign_s)
            sf = point_pass.float().sum(dim=1) / max(1, n)
            per_subspace_fracs[s].append(sf.mean().item())

    mean_frac = sum(fracs) / len(fracs) if fracs else 1.0
    mean_search_ms = (
        (sum(search_times) / len(search_times)) * 1000 if search_times else 0.0
    )
    per_sub_means = [
        sum(vals) / len(vals) if vals else 1.0 for vals in per_subspace_fracs
    ]
    return mean_frac, mean_search_ms, per_sub_means


def format_speedup(ratio: float) -> str:
    return f"{1 / ratio:.2f}x"


def _csv_ints(spec: str, name: str) -> list[int]:
    values = [int(s.strip()) for s in spec.split(",") if s.strip()]
    if not values:
        raise ValueError(f"--{name} must contain at least one integer.")
    if any(v <= 0 for v in values):
        raise ValueError(f"--{name} values must be positive; got {values}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf", type=int, default=4)
    parser.add_argument("--n-tokens", type=int, default=2000)
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument(
        "--n-subspaces",
        type=str,
        default="2,4,8",
        help="Comma-separated subspace counts to sweep.",
    )
    parser.add_argument("--refine-iter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input",
        "--input-qkv",
        dest="input_qkv",
        type=Path,
        default=None,
        help="Path to an existing captured QKV .pt file.",
    )
    parser.add_argument("--fp16-keys", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    subspace_counts = _csv_ints(args.n_subspaces, "n-subspaces")
    strategy = "contiguous"

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
    keys_f32 = keys.float() if keys.dtype != torch.float32 else keys
    queries = queries_cpu
    h_kv, n, d = keys.shape
    h_q = queries.shape[0]
    q_head_to_kv = _q_to_kv_map(h_q, h_kv, DEVICE) if h_q != h_kv else None
    k_clusters = max(1, math.ceil(n / args.bf))

    total_q = queries.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(
        range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride)
    )
    q_indices = q_indices[: args.n_queries]

    print(f"Layer {layer}: H_kv={h_kv}, H_q={h_q}, N={n}, D={d}")
    print(
        f"K={k_clusters} clusters (bf={args.bf}), {len(q_indices)} queries, "
        f"topk={args.topk}, filter=OR"
    )
    print("=" * 100)

    results = []
    for s_count in subspace_counts:
        if d % s_count != 0:
            print(
                f"\nWARNING: D={d} not divisible by n_subspaces={s_count}, "
                "dims will be uneven"
            )

        label = f"sub_kcenter_OR_S{s_count}_contiguous"
        print(
            f"\n{label}: building {s_count} subspace indexes (K={k_clusters} each) ..."
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        idx = build_subspace_kcenter(
            keys_f32,
            args.bf,
            n_subspaces=s_count,
            refine_iter=args.refine_iter,
            strategy=strategy,
        )
        torch.cuda.synchronize()
        build_ms = (time.perf_counter() - t0) * 1000

        frac, search_ms, per_sub = measure_subspace_kcenter_or(
            idx, queries, keys_f32, q_indices, q_head_to_kv, args.topk
        )
        pruned = 1.0 - frac
        gate_cost_dp = 1.0
        ratio = gate_cost_dp / args.bf + frac

        results.append(
            {
                "method": label,
                "scanned_frac": frac,
                "pruned": pruned,
                "build_ms": build_ms,
                "search_ms": search_ms,
                "g": gate_cost_dp,
                "ratio": ratio,
            }
        )

        sub_str = "  ".join(f"S{s}_scan={per_sub[s]:.4f}" for s in range(s_count))
        print(
            f"  scanned={frac:.4f}  pruned={pruned:.4f}  "
            f"search={search_ms:.3f}ms  build={build_ms:.1f}ms  "
            f"g={gate_cost_dp:.1f}  ratio={ratio:.3f}  ({format_speedup(ratio)})"
        )
        print(f"  per-subspace scanned fraction: {sub_str}")

    print("\n" + "=" * 100)
    print(
        f"{'METHOD':<35s} {'SCANNED':>8s} {'PRUNED':>8s} "
        f"{'SEARCH_ms':>10s} {'BUILD_ms':>9s} {'g':>5s} {'RATIO':>7s} {'SPEEDUP':>8s}"
    )
    print("-" * 100)

    results.sort(key=lambda r: r["scanned_frac"])
    for row in results:
        print(
            f"{row['method']:<35s} "
            f"{row['scanned_frac']:>8.4f} {row['pruned']:>8.4f} "
            f"{row['search_ms']:>10.3f} {row['build_ms']:>9.1f} "
            f"{row['g']:>5.1f} {row['ratio']:>7.3f} {format_speedup(row['ratio']):>8s}"
        )

    print("=" * 100)
    best = results[0]
    print(
        f"\nBest: {best['method']} -> scanned={best['scanned_frac']:.4f} "
        f"(pruned {best['pruned']:.4f}, {format_speedup(best['ratio'])})"
    )


if __name__ == "__main__":
    main()
