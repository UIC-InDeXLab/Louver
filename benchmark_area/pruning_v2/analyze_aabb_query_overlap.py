#!/usr/bin/env python3
"""
Analyze overlap of selected final AABBs across queries.

Fixed setup:
  - clustering: pq_subspace(ns=2, it=10)
  - enclosing:  aabb

For each sampled query token, we collect the final parent/AABB pass mask and
report pairwise similarity statistics (intersection, Jaccard, overlap coeff).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from pruning_bench_utils import _capture_qkv, _q_to_kv_map
from method_comparison_bench import (
    DEVICE,
    DTYPE,
    LAYER_IDX,
    MODEL_NAME,
    PROMPT,
    cluster_pq_subspace,
    enclose_aabb,
    topk_threshold,
)


def _quantile(v: torch.Tensor, q: float) -> float:
    if v.numel() == 0:
        return float("nan")
    return float(torch.quantile(v, torch.tensor(q, device=v.device)).item())


def _summarize_pairwise(
    masks: torch.Tensor, query_ids: list[int], title: str, top_pairs: int = 5
) -> None:
    """
    masks: (Q, F) bool
    query_ids: query token indices
    """
    q, f = masks.shape
    x = masks.to(dtype=torch.int32)
    cnt = x.sum(dim=1).to(dtype=torch.float32)  # (Q,)

    print("\n" + "-" * 110)
    print(title)
    print("-" * 110)
    print(
        f"Queries={q}, Universe size={f}, selected mean={float(cnt.mean()):.2f}, "
        f"min={int(cnt.min())}, max={int(cnt.max())}"
    )

    if q < 2:
        print("Need at least 2 queries for pairwise overlap.")
        return

    inter = x @ x.transpose(0, 1)  # (Q,Q)
    inter_f = inter.to(dtype=torch.float32)
    union = cnt[:, None] + cnt[None, :] - inter_f
    denom_overlap = torch.minimum(cnt[:, None], cnt[None, :]).clamp_min(1.0)

    jacc = torch.where(union > 0, inter_f / union, torch.zeros_like(union))
    overlap_coeff = inter_f / denom_overlap

    iu = torch.triu_indices(q, q, offset=1, device=masks.device)
    inter_u = inter_f[iu[0], iu[1]]
    jacc_u = jacc[iu[0], iu[1]]
    overlap_u = overlap_coeff[iu[0], iu[1]]

    print(
        f"Pairwise intersection: mean={float(inter_u.mean()):.2f}, "
        f"p50={_quantile(inter_u, 0.50):.2f}, p90={_quantile(inter_u, 0.90):.2f}"
    )
    print(
        f"Pairwise Jaccard:      mean={float(jacc_u.mean()):.4f}, "
        f"p50={_quantile(jacc_u, 0.50):.4f}, p90={_quantile(jacc_u, 0.90):.4f}, "
        f"max={float(jacc_u.max()):.4f}"
    )
    print(
        f"Pairwise overlap coef: mean={float(overlap_u.mean()):.4f}, "
        f"p50={_quantile(overlap_u, 0.50):.4f}, p90={_quantile(overlap_u, 0.90):.4f}, "
        f"max={float(overlap_u.max()):.4f}"
    )

    common_all = masks.all(dim=0).sum().item()
    print(f"AABBs selected by all queries: {int(common_all)} ({common_all / f:.6f} of universe)")

    freq = x.to(dtype=torch.float32).mean(dim=0)  # fraction of queries selecting id
    topk = min(top_pairs, freq.numel())
    top_vals, top_idx = torch.topk(freq, k=topk)
    print(f"Top-{topk} most reused IDs (fraction of queries selecting):")
    for rank in range(topk):
        idx = int(top_idx[rank].item())
        val = float(top_vals[rank].item())
        print(f"  {rank+1:>2d}. id={idx:<8d} freq={val:.4f}")

    # Most/least similar query pairs by Jaccard
    pair_scores = jacc_u
    order_desc = torch.argsort(pair_scores, descending=True)
    order_asc = torch.argsort(pair_scores, descending=False)

    print(f"Most similar query pairs by Jaccard (top {min(top_pairs, pair_scores.numel())}):")
    for rank in range(min(top_pairs, pair_scores.numel())):
        p = int(order_desc[rank].item())
        i = int(iu[0, p].item())
        j = int(iu[1, p].item())
        print(
            f"  {rank+1:>2d}. q_token=({query_ids[i]},{query_ids[j]}) "
            f"jacc={float(jacc[i, j]):.4f} inter={int(inter[i, j])}"
        )

    print(f"Least similar query pairs by Jaccard (top {min(top_pairs, pair_scores.numel())}):")
    for rank in range(min(top_pairs, pair_scores.numel())):
        p = int(order_asc[rank].item())
        i = int(iu[0, p].item())
        j = int(iu[1, p].item())
        print(
            f"  {rank+1:>2d}. q_token=({query_ids[i]},{query_ids[j]}) "
            f"jacc={float(jacc[i, j]):.4f} inter={int(inter[i, j])}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bf", type=int, default=4, help="Branching factor")
    p.add_argument("--n-tokens", type=int, default=2000, help="Generated tokens to capture")
    p.add_argument("--n-queries", type=int, default=30, help="Number of query tokens to analyze")
    p.add_argument("--topk", type=int, default=20, help="Top-k threshold")
    p.add_argument("--model", type=str, default=MODEL_NAME)
    p.add_argument("--layer", type=int, default=LAYER_IDX)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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

    layer_ids = capture.layer_ids()
    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, _ = capture.to_layer_tensors(layer)

    keys = keys_cpu.to(device=DEVICE, dtype=torch.float32)
    queries = queries_cpu.to(device=DEVICE, dtype=torch.float32)
    h_kv, n, d = keys.shape
    h_q = queries.shape[0]

    q_head_to_kv = _q_to_kv_map(h_q, h_kv, DEVICE) if h_q != h_kv else None
    k = max(1, math.ceil(n / args.bf))

    total_q = queries.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(
        range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride)
    )[: args.n_queries]

    print(
        f"Layer {layer}: H_q={h_q}, H_kv={h_kv}, N={n}, D={d}, K={k}, "
        f"queries={len(q_indices)}, topk={args.topk}"
    )
    print("Clustering: pq_subspace(ns=2,it=10), Enclosing: aabb")

    t_cluster = time.perf_counter()
    assign, centers = cluster_pq_subspace(keys, args.bf, n_subspaces=2, max_iter=10)
    clust_ms = (time.perf_counter() - t_cluster) * 1000.0

    if q_head_to_kv is not None:
        assign_q = assign[q_head_to_kv]
        centers_q = centers[q_head_to_kv]
        keys_q = keys[q_head_to_kv]
    else:
        assign_q = assign
        centers_q = centers
        keys_q = keys

    t_enclose = time.perf_counter()
    gate_fn, enc_info = enclose_aabb(keys_q, assign_q, centers_q, k, args.bf)
    enc_ms = (time.perf_counter() - t_enclose) * 1000.0
    print(
        f"Build: cluster={clust_ms:.1f}ms, enclosing={enc_ms:.1f}ms, "
        f"vol_mean={enc_info['vol_mean']:.4f}"
    )

    masks_qhead: list[torch.Tensor] = []
    masks_kv: list[torch.Tensor] = []
    scanned_frac: list[float] = []

    # Build KV-head grouping once for collapsed analysis.
    groups: list[torch.Tensor] = []
    if q_head_to_kv is not None:
        for kvh in range(h_kv):
            groups.append((q_head_to_kv == kvh).nonzero(as_tuple=False).view(-1))

    for qi in q_indices:
        q = queries[:, qi, :]
        q_norm = q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q_normal = q / q_norm
        q_kv = q_normal[q_head_to_kv] if q_head_to_kv is not None else q_normal
        th = topk_threshold(q_kv, keys_q, k=args.topk)
        parent_pass = gate_fn(q_kv, th)  # (H_q, K)

        masks_qhead.append(parent_pass.reshape(-1).to(device="cpu"))
        scanned = parent_pass.sum(dim=1).float() * args.bf
        scanned_frac.append(float((scanned / max(1, n)).mean().item()))

        if q_head_to_kv is None:
            masks_kv.append(parent_pass.reshape(-1).to(device="cpu"))
        else:
            kv_mask = torch.zeros(h_kv, k, dtype=torch.bool, device=parent_pass.device)
            for kvh, idx in enumerate(groups):
                if idx.numel() > 0:
                    kv_mask[kvh] = parent_pass.index_select(0, idx).any(dim=0)
            masks_kv.append(kv_mask.reshape(-1).to(device="cpu"))

    print(
        f"Mean scanned fraction over analyzed queries: "
        f"{sum(scanned_frac) / max(1, len(scanned_frac)):.4f}"
    )

    qhead_matrix = torch.stack(masks_qhead, dim=0)
    kv_matrix = torch.stack(masks_kv, dim=0)

    _summarize_pairwise(
        qhead_matrix,
        query_ids=q_indices,
        title="Pairwise Similarity on (query_head, parent_id) AABB IDs",
    )
    _summarize_pairwise(
        kv_matrix,
        query_ids=q_indices,
        title="Pairwise Similarity on KV-collapsed (kv_head, parent_id) AABB IDs",
    )


if __name__ == "__main__":
    main()

