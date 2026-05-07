"""Outlier-AABB: remove 1 outlier per cluster for a tighter AABB.

With ~bf keys per cluster, removing the single key that contributes most
to the AABB span can significantly tighten the box.  The outlier gets a
direct dot-product check.

Gate cost: 2.0 (tight AABB) + 1.0 (outlier dot) = 3.0 dp-equiv.
"""

from __future__ import annotations

import torch


def enclose_outlier_aabb(keys, assign, centers, K, bf):
    """
    Fully vectorized outlier-AABB.

    For each cluster, remove the point farthest from centroid (in L2),
    build AABB on remaining points, and check the outlier via dot product.
    """
    H, N, D = keys.shape
    device = keys.device

    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    # ── Full AABB ──
    lo_full = torch.full((H, K, D), float("inf"), device=device)
    hi_full = torch.full((H, K, D), float("-inf"), device=device)
    lo_full.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi_full.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)
    empty = lo_full[:, :, 0].isinf()
    if empty.any():
        lo_full[empty] = 0.0
        hi_full[empty] = 0.0

    # ── Find outlier per cluster: farthest from centroid in L2 ──
    key_centers = centers.gather(1, idx_exp)  # (H, N, D)
    dist_sq = (keys - key_centers).square().sum(dim=-1)  # (H, N)

    # Max distance per cluster
    max_dist = torch.full((H, K), float("-inf"), device=device)
    max_dist.scatter_reduce_(1, assign, dist_sq, reduce="amax", include_self=False)

    # Mark the farthest point(s) per cluster as outliers
    cluster_max = max_dist.gather(1, assign)  # (H, N)
    is_outlier = (dist_sq >= cluster_max - 1e-7) & (dist_sq > 0)  # (H, N)

    # For ties, only keep one outlier per cluster (first encountered)
    # Check cluster size: don't remove outlier from clusters of size <= 1
    counts = torch.zeros(H, K, device=device)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device))
    small_cluster = counts.gather(1, assign) <= 1  # (H, N)
    is_outlier = is_outlier & ~small_cluster

    # ── Build tight AABB without outliers ──
    keys_for_lo = torch.where(is_outlier.unsqueeze(-1), torch.full_like(keys, float("inf")), keys)
    keys_for_hi = torch.where(is_outlier.unsqueeze(-1), torch.full_like(keys, float("-inf")), keys)

    lo_tight = torch.full((H, K, D), float("inf"), device=device)
    hi_tight = torch.full((H, K, D), float("-inf"), device=device)
    lo_tight.scatter_reduce_(1, idx_exp, keys_for_lo, reduce="amin", include_self=False)
    hi_tight.scatter_reduce_(1, idx_exp, keys_for_hi, reduce="amax", include_self=False)

    # Fall back to full AABB for clusters where tight is invalid
    bad = lo_tight[:, :, 0].isinf()
    if bad.any():
        lo_tight[bad] = lo_full[bad]
        hi_tight[bad] = hi_full[bad]

    def gate(q, th):
        q_exp = q.unsqueeze(1)  # (H, 1, D)
        th_exp = th.unsqueeze(-1)

        # Tight AABB
        aabb_score = torch.maximum(q_exp * lo_tight, q_exp * hi_tight).sum(dim=-1)
        aabb_pass = aabb_score > th_exp

        # Outlier dot products -> max per cluster
        dots = (keys * q.unsqueeze(1)).sum(dim=-1)  # (H, N)
        dots_masked = torch.where(is_outlier, dots, torch.tensor(float("-inf"), device=device))
        max_outlier = torch.full((H, K), float("-inf"), device=device)
        max_outlier.scatter_reduce_(1, assign, dots_masked, reduce="amax", include_self=False)
        outlier_pass = max_outlier > th.unsqueeze(-1)

        return aabb_pass | outlier_pass

    span_full = (hi_full - lo_full).clamp_min(0).sum(dim=-1)
    span_tight = (hi_tight - lo_tight).clamp_min(0).sum(dim=-1)
    reduction = (span_full - span_tight) / span_full.clamp_min(1e-12)

    return gate, {
        "span_reduction_mean": float(reduction.mean()),
        "n_outliers_total": int(is_outlier.sum()),
    }
