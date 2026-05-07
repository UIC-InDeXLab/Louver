#!/usr/bin/env python3
"""Analyze row depth for sorted subspace-kcenter ball lower bounds.

For each query, this script:

1. Builds the existing ``subspace_kcenter`` index.
2. Computes exact full-vector dot-product thresholds from true top-k keys.
3. Sorts each subspace's balls by either the upper bound
   ``q_s dot center + radius * ||q_s||``, lower bound
   ``q_s dot center - radius * ||q_s||``, or middle score
   ``q_s dot center``.
4. Treats the sorted subspace lists as an ``S``-column table and stops at the
   first row whose row-sum bound is below the threshold.
5. Measures how many true top-k keys are covered by the union of clusters in
   rows ``1..L`` across all subspaces.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
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
)


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LAYER_IDX = 15
PROMPT = (
    "Solve the following problem step by step, showing all intermediate "
    "reasoning, calculations, and verification.\n\n"
    "A research lab is designing a distributed computing cluster. They have "
    "a budget for 120 machines. Each machine can be configured as a CPU node "
    "(32 cores, 128 GB RAM, $5000), a GPU node (8 cores, 64 GB RAM, 4xA100 "
    "GPUs, $35000), or a storage node (8 cores, 256 GB RAM, 100 TB disk, "
    "$12000). Determine the optimal allocation and analyze changed-budget, "
    "doubled-data, and tripled-inference scenarios."
)


@dataclass
class StatBucket:
    rows: list[int]
    candidate_counts: list[int]
    candidate_fracs: list[float]
    hit_counts: list[int]
    recalls: list[float]
    row_bound_sums: list[float]
    thresholds: list[float]


def _csv_ints(spec: str, name: str) -> list[int]:
    values = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not values:
        raise ValueError(f"--{name} must contain at least one integer.")
    if any(v <= 0 for v in values):
        raise ValueError(f"--{name} values must be positive; got {values}")
    return values


def _quantile(values: list[float] | list[int], q: float) -> float:
    if not values:
        return 0.0
    x = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(x, q).item())


def _mean(values: list[float] | list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def exact_scores_and_topk(
    q: torch.Tensor,
    keys_eval: torch.Tensor,
    max_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact scores, top values, and top indices for each query head."""
    scores = torch.einsum("hd,hnd->hn", q, keys_eval)
    k = min(max_topk, scores.shape[-1])
    top_vals, top_idx = scores.topk(k, dim=-1)
    return scores, top_vals, top_idx


def bound_table(
    idx,
    q: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    bound: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sorted bound table and matching cluster ids.

    Returns:
        sorted_bound: (H_q, S, K), descending bounds.
        order:     (H_q, S, K), cluster ids in sorted order.
    """
    q_proj = project_query_for_index(q, idx, q_head_to_kv)
    bounds = []
    orders = []

    for s, (start, end) in enumerate(idx.dim_slices):
        q_sub = q_proj[:, start:end]
        q_sub_norm = q_sub.norm(dim=-1)
        centers = idx.centers[s] if q_head_to_kv is None else idx.centers[s][q_head_to_kv]
        radii = idx.radii[s] if q_head_to_kv is None else idx.radii[s][q_head_to_kv]

        center_dots = torch.einsum("hkd,hd->hk", centers, q_sub)
        radius_term = radii * q_sub_norm.unsqueeze(-1)
        if bound == "upper":
            bound_values = center_dots + radius_term
        elif bound == "middle":
            bound_values = center_dots
        elif bound == "lower":
            bound_values = center_dots - radius_term
        else:
            raise ValueError(f"Unknown bound: {bound}")

        bound_sorted, order = torch.sort(bound_values, dim=-1, descending=True)
        bounds.append(bound_sorted)
        orders.append(order)

    return torch.stack(bounds, dim=1), torch.stack(orders, dim=1)


def choose_depth(row_sums: torch.Tensor, threshold: float) -> int:
    """Return 1-based inclusive row depth for one head."""
    hit = (row_sums < threshold).nonzero(as_tuple=True)[0]
    if hit.numel() == 0:
        return int(row_sums.numel())
    return int(hit[0].item()) + 1


def selected_point_mask(
    idx,
    order_h: torch.Tensor,
    head: int,
    q_head_to_kv: torch.Tensor | None,
    depth: int,
    n_keys: int,
) -> torch.Tensor:
    """Union of points in clusters from rows 1..depth across subspaces."""
    device = order_h.device
    selected = torch.zeros(n_keys, dtype=torch.bool, device=device)
    h_kv = int(q_head_to_kv[head].item()) if q_head_to_kv is not None else head

    for s in range(idx.n_subspaces):
        chosen_clusters = order_h[s, :depth]
        assign = idx.assigns[s][h_kv]
        selected |= torch.isin(assign, chosen_clusters)
    return selected


def analyze_query(
    idx,
    q: torch.Tensor,
    keys_eval: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    topks: list[int],
    bound: str,
    buckets: dict[int, StatBucket],
) -> None:
    _, n_keys, _ = keys_eval.shape
    max_topk = min(max(topks), n_keys)
    scores, top_vals, top_idx = exact_scores_and_topk(q, keys_eval, max_topk)
    sorted_bound, order = bound_table(idx, q, q_head_to_kv, bound)
    row_bound_sums = sorted_bound.sum(dim=1)  # (H_q, K)

    for h in range(q.shape[0]):
        for topk in topks:
            k_eff = min(topk, n_keys)
            threshold = float(top_vals[h, k_eff - 1].item())
            depth = choose_depth(row_bound_sums[h], threshold)
            selected = selected_point_mask(
                idx=idx,
                order_h=order[h],
                head=h,
                q_head_to_kv=q_head_to_kv,
                depth=depth,
                n_keys=n_keys,
            )

            kth_top_idx = top_idx[h, :k_eff]
            hit_count = int(selected[kth_top_idx].sum().item())
            candidate_count = int(selected.sum().item())
            candidate_frac = candidate_count / max(1, n_keys)

            bucket = buckets[topk]
            bucket.rows.append(depth)
            bucket.candidate_counts.append(candidate_count)
            bucket.candidate_fracs.append(candidate_frac)
            bucket.hit_counts.append(hit_count)
            bucket.recalls.append(hit_count / max(1, k_eff))
            bucket.row_bound_sums.append(float(row_bound_sums[h, depth - 1].item()))
            bucket.thresholds.append(threshold)


def print_summary(
    buckets: dict[int, StatBucket],
    n_cases: int,
    k_clusters: int,
    bound: str,
) -> None:
    print()
    print(
        f"Bound: {bound}; stop rule: first row where row_bound_sum < T "
        f"({n_cases} query-head cases, K={k_clusters})"
    )
    print(
        f"{'topk':>6s} {'rows_mean':>10s} {'keys_mean':>10s} "
        f"{'cand_frac':>10s} {'hits_mean':>10s} {'recall':>8s}"
    )
    print("-" * 64)
    for topk, b in buckets.items():
        print(
            f"{topk:6d} "
            f"{_mean(b.rows):10.2f} "
            f"{_mean(b.candidate_counts):10.1f} "
            f"{_mean(b.candidate_fracs):10.4f} "
            f"{_mean(b.hit_counts):10.2f} {_mean(b.recalls):8.4f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", "--input-qkv", dest="input_qkv", type=Path, default=None)
    p.add_argument("--model", type=str, default=MODEL_NAME)
    p.add_argument("--n-tokens", type=int, default=2000)
    p.add_argument("--n-queries", type=int, default=30)
    p.add_argument("--layer", type=int, default=LAYER_IDX)
    p.add_argument("--bf", type=int, default=4)
    p.add_argument("--S", "--n-subspaces", dest="n_subspaces", type=int, default=8)
    p.add_argument("--refine-iter", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--fp16-keys", action="store_true")
    p.add_argument("--topks", type=str, default="5,10,20,50",
                   help="Comma-separated k values. T is the exact kth full-dot score.")
    p.add_argument(
        "--bound",
        choices=("upper", "middle", "lower"),
        default="upper",
        help=(
            "Sort subspace balls by this score. Upper is the safe TA-style bound; "
            "middle uses center dots only."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    topks = _csv_ints(args.topks, "topks")

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
            device=args.device,
            torch_dtype=torch.float32,
            show_progress=True,
        )
        print(f"Capture done in {time.perf_counter() - t0:.1f}s")

    layer_ids = capture.layer_ids()
    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, _ = capture.to_layer_tensors(layer)

    keys_dtype = torch.float16 if args.fp16_keys else torch.float32
    keys = keys_cpu.to(device=args.device, dtype=keys_dtype)
    keys_f32 = keys.float() if keys.dtype != torch.float32 else keys
    queries = queries_cpu

    h_kv, n_keys, d = keys.shape
    h_q = queries.shape[0]
    q_head_to_kv = _q_to_kv_map(h_q, h_kv, args.device) if h_q != h_kv else None
    k_clusters = max(1, math.ceil(n_keys / args.bf))

    total_q = queries.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(
        range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride)
    )[: args.n_queries]

    print(
        f"Layer {layer}: H_kv={h_kv}, H_q={h_q}, N={n_keys}, D={d}, "
        f"S={args.n_subspaces}, bf={args.bf}, K={k_clusters}"
    )
    print(
        f"queries={len(q_indices)}, topks={topks}, "
        f"strategy=contiguous, query=raw, bound={args.bound}"
    )

    if d % args.n_subspaces != 0:
        print(f"WARNING: D={d} is not divisible by S={args.n_subspaces}; slices are uneven.")

    if args.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    idx = build_subspace_kcenter(
        keys_f32,
        args.bf,
        n_subspaces=args.n_subspaces,
        refine_iter=args.refine_iter,
        strategy="contiguous",
    )
    if args.device == "cuda":
        torch.cuda.synchronize()
    print(f"Index build: {(time.perf_counter() - t0) * 1000:.1f} ms")

    keys_eval = project_keys_for_index(keys_f32, idx, q_head_to_kv)
    buckets = {k: StatBucket([], [], [], [], [], [], []) for k in topks}

    for qi in q_indices:
        q = queries[:, qi, :].to(device=args.device, dtype=torch.float32)
        analyze_query(
            idx=idx,
            q=q,
            keys_eval=keys_eval,
            q_head_to_kv=q_head_to_kv,
            topks=topks,
            bound=args.bound,
            buckets=buckets,
        )

    n_cases = len(q_indices) * h_q
    print_summary(buckets, n_cases, k_clusters, args.bound)


if __name__ == "__main__":
    main()
