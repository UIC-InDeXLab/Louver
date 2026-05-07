#!/usr/bin/env python3
"""
Track AABB relation cases over a query sequence for pq_subspace clustering.

Cases per (query_head, parent_aabb):
  - inside:     entire AABB is above threshold (lower_bound > th)
  - outside:    entire AABB is below threshold (upper_bound <= th)
  - intersect:  AABB straddles threshold
  - intersect_true: intersecting AABB with >=1 real key above threshold
  - intersect_false: intersecting AABB with no real key above threshold

Reports per-query and cumulative ("so far") frequencies.
"""

from __future__ import annotations

import argparse
import csv
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
    topk_threshold,
)


def _quantile(v: torch.Tensor, q: float) -> float:
    if v.numel() == 0:
        return float("nan")
    return float(torch.quantile(v, torch.tensor(q, device=v.device)).item())


def _report_false_recurrence(
    false_matrix: torch.Tensor,
    query_ids: list[int],
    n_heads: int,
    k_parents: int,
    title: str,
    top_n: int = 10,
) -> None:
    """
    false_matrix: (Q, n_heads*k_parents) bool
    """
    q, f = false_matrix.shape
    x = false_matrix.to(dtype=torch.int32)
    freq = x.to(dtype=torch.float32).mean(dim=0)  # fraction over queries
    cnt = x.sum(dim=0)  # absolute query count
    ever = freq > 0
    ever_count = int(ever.sum().item())

    print("\n" + "-" * 118)
    print(title)
    print("-" * 118)
    print(f"Queries={q}, Universe size={f}, IDs ever false={ever_count} ({ever_count / f:.4f})")
    if ever_count == 0:
        print("No intersect_false IDs observed.")
        return

    freq_ever = freq[ever]
    print(
        f"False-frequency over ever-false IDs: "
        f"mean={float(freq_ever.mean()):.4f}, p50={_quantile(freq_ever, 0.50):.4f}, "
        f"p90={_quantile(freq_ever, 0.90):.4f}, max={float(freq_ever.max()):.4f}"
    )

    for thr in (0.50, 0.80, 0.90, 0.95):
        c = int((freq >= thr).sum().item())
        print(
            f"IDs false in >= {thr:.0%} of queries: {c} "
            f"({c / f:.4f} of universe, {c / ever_count:.4f} of ever-false)"
        )

    # Pairwise similarity of false sets across queries.
    inter = x @ x.transpose(0, 1)
    inter_f = inter.to(dtype=torch.float32)
    sizes = x.sum(dim=1).to(dtype=torch.float32)
    union = sizes[:, None] + sizes[None, :] - inter_f
    jacc = torch.where(union > 0, inter_f / union, torch.zeros_like(union))
    iu = torch.triu_indices(q, q, offset=1, device=false_matrix.device)
    if iu.shape[1] > 0:
        jacc_u = jacc[iu[0], iu[1]]
        print(
            f"Pairwise Jaccard of intersect_false sets: "
            f"mean={float(jacc_u.mean()):.4f}, p50={_quantile(jacc_u, 0.50):.4f}, "
            f"p90={_quantile(jacc_u, 0.90):.4f}, max={float(jacc_u.max()):.4f}"
        )

    top_n = min(top_n, f)
    top_vals, top_idx = torch.topk(freq, k=top_n)
    print(f"Top-{top_n} most recurring intersect_false IDs:")
    for rank in range(top_n):
        flat = int(top_idx[rank].item())
        h = flat // k_parents
        p = flat % k_parents
        ff = float(top_vals[rank].item())
        cc = int(cnt[top_idx[rank]].item())
        print(
            f"  {rank+1:>2d}. id={flat:<7d} head={h:<3d} parent={p:<5d} "
            f"false_freq={ff:.4f} ({cc}/{q})"
        )


def _build_aabb_bounds(
    keys: torch.Tensor, assign: torch.Tensor, k_parents: int
) -> tuple[torch.Tensor, torch.Tensor]:
    h, _, d = keys.shape
    device = keys.device

    lo = torch.full((h, k_parents, d), float("inf"), device=device)
    hi = torch.full((h, k_parents, d), float("-inf"), device=device)
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, d)

    lo.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)

    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    return lo, hi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bf", type=int, default=4, help="Branching factor")
    p.add_argument("--n-tokens", type=int, default=600, help="Generated tokens to capture")
    p.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Max number of queries to process from sequence order (0=all).",
    )
    p.add_argument("--stride", type=int, default=1, help="Query stride over the sequence.")
    p.add_argument("--topk", type=int, default=20, help="Top-k threshold.")
    p.add_argument("--model", type=str, default=MODEL_NAME)
    p.add_argument("--layer", type=int, default=LAYER_IDX)
    p.add_argument("--ns", type=int, default=2, help="pq_subspace n_subspaces.")
    p.add_argument("--it", type=int, default=10, help="pq_subspace max_iter.")
    p.add_argument(
        "--report-every",
        type=int,
        default=50,
        help="Print every N-th query row (also first and last).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("aabb_case_progression.csv"),
        help="Per-query output CSV.",
    )
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
    h_kv, n, _ = keys.shape
    h_q = queries.shape[0]

    q_head_to_kv = _q_to_kv_map(h_q, h_kv, DEVICE) if h_q != h_kv else None
    k_parents = max(1, math.ceil(n / args.bf))

    # Sequence order: early -> late.
    all_q_indices = list(range(0, queries.shape[1], max(1, args.stride)))
    if args.max_queries > 0:
        all_q_indices = all_q_indices[: args.max_queries]

    print(
        f"Layer {layer}: H_q={h_q}, H_kv={h_kv}, N={n}, K={k_parents}, "
        f"queries={len(all_q_indices)}, topk={args.topk}"
    )
    print(f"Clustering: pq_subspace(ns={args.ns},it={args.it}), Enclosing: aabb")

    t_cluster = time.perf_counter()
    assign, centers = cluster_pq_subspace(
        keys, args.bf, n_subspaces=args.ns, max_iter=args.it
    )
    clust_ms = (time.perf_counter() - t_cluster) * 1000.0

    if q_head_to_kv is not None:
        assign_q = assign[q_head_to_kv]
        keys_q = keys[q_head_to_kv]
    else:
        assign_q = assign
        keys_q = keys

    t_box = time.perf_counter()
    lo, hi = _build_aabb_bounds(keys_q, assign_q, k_parents)
    box_ms = (time.perf_counter() - t_box) * 1000.0
    print(f"Build: cluster={clust_ms:.1f}ms, aabb={box_ms:.1f}ms")

    rows: list[dict[str, float | int]] = []
    false_masks_qhead: list[torch.Tensor] = []
    false_masks_kv: list[torch.Tensor] = []
    cum_inside = 0
    cum_outside = 0
    cum_inter = 0
    cum_inter_true = 0
    cum_inter_false = 0
    pairs_per_query = h_q * k_parents

    header_printed = False
    every = max(1, args.report_every)
    kv_groups: list[torch.Tensor] = []
    if q_head_to_kv is not None:
        for kvh in range(h_kv):
            kv_groups.append((q_head_to_kv == kvh).nonzero(as_tuple=False).view(-1))

    for i, qi in enumerate(all_q_indices):
        q = queries[:, qi, :]
        q_norm = q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q_normal = q / q_norm

        q_eval = q_normal[q_head_to_kv] if q_head_to_kv is not None else q_normal
        th = topk_threshold(q_eval, keys_q, k=args.topk)  # (H_q,)

        q_exp = q_eval.unsqueeze(1)  # (H_q,1,D)
        upper = torch.maximum(q_exp * lo, q_exp * hi).sum(dim=-1)  # (H_q,K)
        lower = torch.minimum(q_exp * lo, q_exp * hi).sum(dim=-1)  # (H_q,K)

        inside = lower > th.unsqueeze(-1)
        outside = upper <= th.unsqueeze(-1)
        inter = (~inside) & (~outside)

        # Exact check on real children: does each parent contain any true passing key?
        # scores: (H_q, N), max_score_per_parent: (H_q, K)
        scores = torch.einsum("hnd,hd->hn", keys_q, q_eval)
        max_score_per_parent = torch.full(
            (h_q, k_parents), float("-inf"), device=scores.device
        )
        max_score_per_parent.scatter_reduce_(
            1, assign_q, scores, reduce="amax", include_self=True
        )
        has_real_passing = max_score_per_parent > th.unsqueeze(-1)

        inter_true = inter & has_real_passing
        inter_false = inter & (~has_real_passing)
        false_masks_qhead.append(inter_false.reshape(-1).to(device="cpu"))

        inside_c = int(inside.sum().item())
        outside_c = int(outside.sum().item())
        inter_c = int(inter.sum().item())
        inter_true_c = int(inter_true.sum().item())
        inter_false_c = int(inter_false.sum().item())
        if inside_c + outside_c + inter_c != pairs_per_query:
            raise RuntimeError("Case partition failed sanity check.")
        if inter_true_c + inter_false_c != inter_c:
            raise RuntimeError("Intersect split sanity check failed.")

        if q_head_to_kv is None:
            false_masks_kv.append(inter_false.reshape(-1).to(device="cpu"))
        else:
            kv_false = torch.zeros(
                h_kv, k_parents, dtype=torch.bool, device=inter_false.device
            )
            for kvh, idx in enumerate(kv_groups):
                if idx.numel() > 0:
                    kv_false[kvh] = inter_false.index_select(0, idx).any(dim=0)
            false_masks_kv.append(kv_false.reshape(-1).to(device="cpu"))

        cum_inside += inside_c
        cum_outside += outside_c
        cum_inter += inter_c
        cum_inter_true += inter_true_c
        cum_inter_false += inter_false_c
        total_so_far = (i + 1) * pairs_per_query

        row = {
            "query_order": i,
            "query_token_idx": qi,
            "inside_count": inside_c,
            "outside_count": outside_c,
            "intersect_count": inter_c,
            "intersect_true_count": inter_true_c,
            "intersect_false_count": inter_false_c,
            "inside_frac": inside_c / pairs_per_query,
            "outside_frac": outside_c / pairs_per_query,
            "intersect_frac": inter_c / pairs_per_query,
            "intersect_true_frac": inter_true_c / pairs_per_query,
            "intersect_false_frac": inter_false_c / pairs_per_query,
            "intersect_false_given_intersect": (
                inter_false_c / inter_c if inter_c > 0 else 0.0
            ),
            "cum_inside_frac": cum_inside / total_so_far,
            "cum_outside_frac": cum_outside / total_so_far,
            "cum_intersect_frac": cum_inter / total_so_far,
            "cum_intersect_true_frac": cum_inter_true / total_so_far,
            "cum_intersect_false_frac": cum_inter_false / total_so_far,
            "cum_intersect_false_given_intersect": (
                cum_inter_false / cum_inter if cum_inter > 0 else 0.0
            ),
        }
        rows.append(row)

        should_print = (i == 0) or (i == len(all_q_indices) - 1) or ((i + 1) % every == 0)
        if should_print:
            if not header_printed:
                print("-" * 114)
                print(
                    f"{'Q#':>4s} {'TOKEN':>7s} {'INSIDE':>9s} {'OUTSIDE':>9s} {'INT':>8s} "
                    f"{'INT_T':>8s} {'INT_F':>8s} {'FP|INT':>8s} "
                    f"{'C_IN':>7s} {'C_OUT':>7s} {'C_INT':>7s} {'C_FP|I':>7s}"
                )
                print("-" * 114)
                header_printed = True

            print(
                f"{i:>4d} {qi:>7d} "
                f"{row['inside_frac']:>9.4f} {row['outside_frac']:>9.4f} {row['intersect_frac']:>8.4f} "
                f"{row['intersect_true_frac']:>8.4f} {row['intersect_false_frac']:>8.4f} "
                f"{row['intersect_false_given_intersect']:>8.4f} "
                f"{row['cum_inside_frac']:>7.4f} {row['cum_outside_frac']:>7.4f} {row['cum_intersect_frac']:>7.4f} "
                f"{row['cum_intersect_false_given_intersect']:>7.4f}"
            )

    print("-" * 114)
    print(
        f"Final cumulative: inside={rows[-1]['cum_inside_frac']:.4f}, "
        f"outside={rows[-1]['cum_outside_frac']:.4f}, "
        f"intersect={rows[-1]['cum_intersect_frac']:.4f}, "
        f"intersect_true={rows[-1]['cum_intersect_true_frac']:.4f}, "
        f"intersect_false={rows[-1]['cum_intersect_false_frac']:.4f}, "
        f"fp_given_intersect={rows[-1]['cum_intersect_false_given_intersect']:.4f}"
    )

    qhead_false_matrix = torch.stack(false_masks_qhead, dim=0)
    kv_false_matrix = torch.stack(false_masks_kv, dim=0)
    _report_false_recurrence(
        qhead_false_matrix,
        query_ids=all_q_indices,
        n_heads=h_q,
        k_parents=k_parents,
        title="Intersect-False Recurrence on (query_head, parent_id)",
    )
    _report_false_recurrence(
        kv_false_matrix,
        query_ids=all_q_indices,
        n_heads=h_kv,
        k_parents=k_parents,
        title="Intersect-False Recurrence on KV-collapsed (kv_head, parent_id)",
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
