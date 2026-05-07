"""Span-minimizing k-means: minimize total AABB span instead of L2."""

from __future__ import annotations

import math

import torch


def cluster_span_kmeans(keys: torch.Tensor, bf: int, max_iter: int = 15):
    """
    K-means variant that assigns keys to minimize total per-dimension span.

    The AABB gate's tightness is determined by (hi_d - lo_d) per cluster per
    dimension. This method assigns each key to the cluster where it causes
    the least increase in total span: sum_d (hi_d - lo_d).

    Uses standard k-means for initialization, then refines with span-aware
    assignment.

    Returns:
        assign: (H, N) long.
        centers: (H, K, D).
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    # ── Phase 1: k-means init (5 iters) ──
    perm = torch.argsort(torch.rand(H, N, device=device), dim=1)
    centers = keys.gather(1, perm[:, :K].unsqueeze(-1).expand(-1, -1, D)).clone()

    for _ in range(5):
        dists = torch.cdist(keys, centers)
        assign = dists.argmin(dim=2)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(H, K, device=device)
        new_centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
        counts.scatter_add_(1, assign, torch.ones(H, N, device=device))
        counts = counts.clamp_min(1)
        centers = new_centers / counts.unsqueeze(-1)

    # ── Phase 2: span-aware refinement ──
    # Instead of L2 distance, use sum of per-dim "distance to box edge":
    # For each key k and cluster c: if k_d is outside [lo_d, hi_d], it extends
    # the span. The cost is sum_d max(0, k_d - hi_d) + max(0, lo_d - k_d).
    # If k_d is inside, cost is 0. This rewards assigning keys that fit inside
    # existing boxes.

    for _ in range(max_iter):
        idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
        lo = torch.full((H, K, D), float("inf"), device=device)
        hi = torch.full((H, K, D), float("-inf"), device=device)
        lo.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
        hi.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)
        empty = lo[:, :, 0].isinf()
        if empty.any():
            lo[empty] = 0.0
            hi[empty] = 0.0

        # Span-aware distance: how much would each key extend each cluster's box
        # keys: (H, N, D) -> (H, N, 1, D)
        # lo, hi: (H, K, D) -> (H, 1, K, D)
        k_exp = keys.unsqueeze(2)  # (H, N, 1, D)
        lo_exp = lo.unsqueeze(1)  # (H, 1, K, D)
        hi_exp = hi.unsqueeze(1)

        # Extension beyond the box edges
        over = (k_exp - hi_exp).clamp_min(0)
        under = (lo_exp - k_exp).clamp_min(0)
        span_cost = (over + under).sum(dim=-1)  # (H, N, K)

        assign = span_cost.argmin(dim=2)

        # Update centers as means
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(H, K, device=device)
        idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
        new_centers.scatter_add_(1, idx_exp, keys)
        counts.scatter_add_(1, assign, torch.ones(H, N, device=device))
        counts = counts.clamp_min(1)
        centers = new_centers / counts.unsqueeze(-1)

    assign = assign  # already final
    return assign, centers
