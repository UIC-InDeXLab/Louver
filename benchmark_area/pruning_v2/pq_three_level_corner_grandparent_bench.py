#!/usr/bin/env python3
"""
Hierarchical benchmark with configurable levels (2/3/4):
  level 1 (always): children -> parents
  level 2 (if >=3): parents -> grandparents
  level 3 (if ==4): grandparents -> great-grandparents

Upper-level construction mode:
  - corner: use only child AABB top-right corners (current behavior).
  - enclosing: build parent AABBs as unions of assigned child AABBs.

Reports per config:
- children_scanned_frac
- parents_scanned_frac (after upper-level filters)
- grandparents_scanned_frac (after upper-level filters)
- search_ms (mean per-query gate time)
- build_ms
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

from pruning_bench_utils import _capture_qkv, _q_to_kv_map
from method_comparison_bench import (
    DEVICE,
    DTYPE,
    LAYER_IDX,
    MODEL_NAME,
    PROMPT,
    cluster_pq_subspace,
)


def _topk_threshold(q_normal: torch.Tensor, keys: torch.Tensor, k: int) -> torch.Tensor:
    h_kv, _, d = keys.shape
    qg = q_normal.view(h_kv, -1, d)
    scores = qg @ keys.transpose(-2, -1)
    scores = scores.reshape(q_normal.shape[0], -1)
    k = min(k, scores.shape[-1])
    th, _ = scores.topk(k, dim=-1)
    return th[:, -1]


def _build_aabb(
    points: torch.Tensor, assign: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-cluster AABB from points."""
    h, _, d = points.shape
    device = points.device
    idx = assign.unsqueeze(-1).expand(-1, -1, d)

    lo = torch.full((h, k, d), float("inf"), device=device, dtype=points.dtype)
    hi = torch.full((h, k, d), float("-inf"), device=device, dtype=points.dtype)

    lo.scatter_reduce_(1, idx, points, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx, points, reduce="amax", include_self=False)

    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0
    return lo, hi


def _build_enclosing_aabb_from_aabbs(
    child_lo: torch.Tensor,
    child_hi: torch.Tensor,
    assign: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build parent AABBs by enclosing assigned child AABBs."""
    h, _, d = child_lo.shape
    device = child_lo.device
    idx = assign.unsqueeze(-1).expand(-1, -1, d)

    lo = torch.full((h, k, d), float("inf"), device=device, dtype=child_lo.dtype)
    hi = torch.full((h, k, d), float("-inf"), device=device, dtype=child_hi.dtype)

    lo.scatter_reduce_(1, idx, child_lo, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx, child_hi, reduce="amax", include_self=False)

    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0
    return lo, hi


def _aabb_gate(lo: torch.Tensor, hi: torch.Tensor, q: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
    q_exp = q.unsqueeze(1)
    max_dot = torch.maximum(q_exp * lo, q_exp * hi).sum(dim=-1)
    return max_dot > th.unsqueeze(-1)


def _eval_hierarchy(
    *,
    num_levels: int,
    keys_q: torch.Tensor,
    queries: torch.Tensor,
    q_indices: list[int],
    bf: int,
    topk: int,
    parent_lo_q: torch.Tensor,
    parent_hi_q: torch.Tensor,
    parent_to_gp_q: torch.Tensor | None,
    gp_lo_q: torch.Tensor | None,
    gp_hi_q: torch.Tensor | None,
    gp_to_gg_q: torch.Tensor | None,
    gg_lo_q: torch.Tensor | None,
    gg_hi_q: torch.Tensor | None,
) -> tuple[float, float, float, float]:
    """
    Returns:
      mean_children_scanned_frac,
      mean_parents_scanned_frac,
      mean_grandparents_scanned_frac,
      mean_search_ms
    """
    _, n, _ = keys_q.shape
    kp = parent_lo_q.shape[1]
    kg = gp_lo_q.shape[1] if gp_lo_q is not None else 0

    child_fracs = []
    parent_fracs = []
    gp_fracs = []
    search_times = []

    for qi in q_indices:
        q = queries[:, qi, :]
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        th = _topk_threshold(q, keys_q, k=topk)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if num_levels == 2:
            parent_from_upper = torch.ones(
                (q.shape[0], kp), device=q.device, dtype=torch.bool
            )
            gp_from_upper = None
            gp_pass = None
        elif num_levels == 3:
            if parent_to_gp_q is None or gp_lo_q is None or gp_hi_q is None:
                raise RuntimeError("Missing grandparent tensors for 3-level evaluation.")
            gp_pass = _aabb_gate(gp_lo_q, gp_hi_q, q, th)  # (H, Kg)
            gp_from_upper = gp_pass
            parent_from_upper = torch.gather(gp_pass, 1, parent_to_gp_q)  # (H, Kp)
        elif num_levels == 4:
            if (
                parent_to_gp_q is None
                or gp_lo_q is None
                or gp_hi_q is None
                or gp_to_gg_q is None
                or gg_lo_q is None
                or gg_hi_q is None
            ):
                raise RuntimeError("Missing tensors for 4-level evaluation.")
            gg_pass = _aabb_gate(gg_lo_q, gg_hi_q, q, th)  # (H, Kgg)
            gp_from_upper = torch.gather(gg_pass, 1, gp_to_gg_q)  # (H, Kg)
            gp_pass = gp_from_upper & _aabb_gate(gp_lo_q, gp_hi_q, q, th)  # (H, Kg)
            parent_from_upper = torch.gather(gp_pass, 1, parent_to_gp_q)  # (H, Kp)
        else:
            raise ValueError(f"Unsupported num_levels={num_levels}")

        parent_aabb_pass = _aabb_gate(parent_lo_q, parent_hi_q, q, th)  # (H, Kp)
        parent_pass = parent_from_upper & parent_aabb_pass
        torch.cuda.synchronize()
        search_times.append(time.perf_counter() - t0)

        scanned_children = parent_pass.sum(dim=1).float() * bf
        child_frac = (scanned_children / max(1, n)).mean().item()
        child_fracs.append(child_frac)

        parent_frac = (
            parent_from_upper.sum(dim=1).float() / max(1, kp)
        ).mean().item()
        parent_fracs.append(parent_frac)

        if num_levels >= 3 and gp_pass is not None:
            gp_frac = (gp_pass.sum(dim=1).float() / max(1, kg)).mean().item()
            gp_fracs.append(gp_frac)
        else:
            gp_fracs.append(float("nan"))

    return (
        sum(child_fracs) / len(child_fracs),
        sum(parent_fracs) / len(parent_fracs),
        sum(gp_fracs) / len(gp_fracs),
        (sum(search_times) / len(search_times)) * 1000.0,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bf", type=int, default=4)
    p.add_argument("--num-levels", type=int, choices=[2, 3, 4], default=3)
    p.add_argument(
        "--upper-aabb-mode",
        type=str,
        choices=["corner", "enclosing"],
        default="corner",
        help=(
            "corner: upper-level AABB from assigned hi-corner points. "
            "enclosing: upper-level AABB is the union of assigned child AABBs."
        ),
    )
    p.add_argument("--n-tokens", type=int, default=2000)
    p.add_argument("--n-queries", type=int, default=24)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--model", type=str, default=MODEL_NAME)
    p.add_argument("--layer", type=int, default=LAYER_IDX)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

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
    print(f"Capture done in {time.perf_counter() - t0:.1f}s\n")

    layer_ids = capture.layer_ids()
    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, _ = capture.to_layer_tensors(layer)

    keys = keys_cpu.to(device=DEVICE, dtype=torch.float32)
    queries = queries_cpu.to(device=DEVICE, dtype=torch.float32)
    h_kv, n, d = keys.shape
    h_q = queries.shape[0]

    q_head_to_kv = _q_to_kv_map(h_q, h_kv, DEVICE) if h_q != h_kv else None
    k_parent = max(1, math.ceil(n / args.bf))
    k_grand = max(1, math.ceil(k_parent / args.bf)) if args.num_levels >= 3 else 0
    k_great = max(1, math.ceil(k_grand / args.bf)) if args.num_levels >= 4 else 0

    total_q = queries.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(
        range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride)
    )
    q_indices = q_indices[: args.n_queries]

    print(
        f"Layer {layer}: H_kv={h_kv}, H_q={h_q}, N={n}, D={d}, "
        f"K_parent={k_parent}, "
        + (f"K_grand={k_grand}, " if args.num_levels >= 3 else "")
        + (f"K_great={k_great}, " if args.num_levels >= 4 else "")
        + f"levels={args.num_levels}, upper_mode={args.upper_aabb_mode}, queries={len(q_indices)}"
    )
    print("=" * 118)

    # CUDA warmup so the first measured config is not penalized by one-time setup.
    warm_n = min(128, n)
    warm_k = max(1, math.ceil(warm_n / args.bf))
    warm_keys = keys[:, :warm_n, :].contiguous()
    warm_assign, _ = cluster_pq_subspace(warm_keys, args.bf, n_subspaces=2, max_iter=1)
    warm_lo, warm_hi = _build_aabb(warm_keys, warm_assign, warm_k)
    warm_q = queries[:, -1, :]
    warm_q = warm_q / warm_q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if q_head_to_kv is not None:
        warm_q = warm_q[q_head_to_kv]
        warm_lo = warm_lo[q_head_to_kv]
        warm_hi = warm_hi[q_head_to_kv]
    warm_th = _topk_threshold(warm_q, warm_keys, k=min(args.topk, warm_n))
    _ = _aabb_gate(warm_lo, warm_hi, warm_q, warm_th)
    torch.cuda.synchronize()

    ns = 2
    its = [5, 10, 15]
    rows = []

    for it in its:
        seed = 9000 + it
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        t_build = time.perf_counter()
        # Level 1: children -> parents via pq_subspace(ns=2,it).
        assign_cp, parent_centers = cluster_pq_subspace(
            keys, args.bf, n_subspaces=ns, max_iter=it
        )
        parent_lo, parent_hi = _build_aabb(keys, assign_cp, k_parent)

        assign_pg = None
        gp_lo = None
        gp_hi = None
        assign_gg = None
        gg_lo = None
        gg_hi = None

        if args.num_levels >= 3:
            # Level 2: parents -> grandparents; clustering still uses parent_hi.
            assign_pg, _ = cluster_pq_subspace(
                parent_hi, args.bf, n_subspaces=ns, max_iter=it
            )
            if args.upper_aabb_mode == "corner":
                gp_lo, gp_hi = _build_aabb(parent_hi, assign_pg, k_grand)
            else:
                gp_lo, gp_hi = _build_enclosing_aabb_from_aabbs(
                    parent_lo, parent_hi, assign_pg, k_grand
                )

        if args.num_levels >= 4:
            # Level 3: grandparents -> great-grandparents; clustering uses gp_hi.
            assign_gg, _ = cluster_pq_subspace(
                gp_hi, args.bf, n_subspaces=ns, max_iter=it
            )
            if args.upper_aabb_mode == "corner":
                gg_lo, gg_hi = _build_aabb(gp_hi, assign_gg, k_great)
            else:
                gg_lo, gg_hi = _build_enclosing_aabb_from_aabbs(
                    gp_lo, gp_hi, assign_gg, k_great
                )

        build_ms = (time.perf_counter() - t_build) * 1000.0

        if q_head_to_kv is not None:
            keys_q = keys[q_head_to_kv]
            parent_lo_q = parent_lo[q_head_to_kv]
            parent_hi_q = parent_hi[q_head_to_kv]
            gp_lo_q = gp_lo[q_head_to_kv] if gp_lo is not None else None
            gp_hi_q = gp_hi[q_head_to_kv] if gp_hi is not None else None
            parent_to_gp_q = assign_pg[q_head_to_kv] if assign_pg is not None else None
            gg_lo_q = gg_lo[q_head_to_kv] if gg_lo is not None else None
            gg_hi_q = gg_hi[q_head_to_kv] if gg_hi is not None else None
            gp_to_gg_q = assign_gg[q_head_to_kv] if assign_gg is not None else None
            queries_q = queries
        else:
            keys_q = keys
            parent_lo_q = parent_lo
            parent_hi_q = parent_hi
            gp_lo_q = gp_lo
            gp_hi_q = gp_hi
            parent_to_gp_q = assign_pg
            gg_lo_q = gg_lo
            gg_hi_q = gg_hi
            gp_to_gg_q = assign_gg
            queries_q = queries

        child_frac, parent_frac, gp_frac, search_ms = _eval_hierarchy(
            num_levels=args.num_levels,
            keys_q=keys_q,
            queries=queries_q,
            q_indices=q_indices,
            bf=args.bf,
            topk=args.topk,
            parent_lo_q=parent_lo_q,
            parent_hi_q=parent_hi_q,
            parent_to_gp_q=parent_to_gp_q,
            gp_lo_q=gp_lo_q,
            gp_hi_q=gp_hi_q,
            gp_to_gg_q=gp_to_gg_q,
            gg_lo_q=gg_lo_q,
            gg_hi_q=gg_hi_q,
        )

        rows.append(
            {
                "config": f"pq_subspace(ns=2,it={it})",
                "it": it,
                "children_scanned_frac": child_frac,
                "parents_scanned_frac": parent_frac,
                "grandparents_scanned_frac": gp_frac,
                "search_ms": search_ms,
                "build_ms": build_ms,
            }
        )
        gp_str = f"{gp_frac:.4f}" if args.num_levels >= 3 else "n/a"
        print(
            f"pq_subspace(ns=2,it={it})  "
            f"children_scanned={child_frac:.4f}  "
            f"parents_scanned={parent_frac:.4f}  "
            f"grandparents_scanned={gp_str}  "
            f"search_ms={search_ms:.3f}  "
            f"build_ms={build_ms:.1f}"
        )

    print("\n" + "=" * 118)
    print(
        f"{'CONFIG':<24s} {'CHILD_SCANNED':>14s} {'PARENT_SCANNED':>15s} {'GRAND_SCANNED':>15s} "
        f"{'SEARCH_ms':>10s} {'BUILD_ms':>10s}"
    )
    print("-" * 118)
    for r in sorted(rows, key=lambda x: x["children_scanned_frac"]):
        gp_str = (
            f"{r['grandparents_scanned_frac']:>15.4f}"
            if args.num_levels >= 3
            else f"{'n/a':>15s}"
        )
        print(
            f"{r['config']:<24s} "
            f"{r['children_scanned_frac']:>14.4f} "
            f"{r['parents_scanned_frac']:>15.4f} "
            f"{gp_str} "
            f"{r['search_ms']:>10.3f} "
            f"{r['build_ms']:>10.1f}"
        )
    print("=" * 118)


if __name__ == "__main__":
    main()
