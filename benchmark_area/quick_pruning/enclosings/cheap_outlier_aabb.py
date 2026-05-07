"""Cheap outlier AABB: tight AABB on core keys + cheap norm-ball bound on outlier.

Instead of computing an exact dot product for the outlier (g=1.0 per outlier),
use a ball bound centered at the cluster centroid with radius = outlier distance.
This is essentially what ball_centroid gives, but applied only to decide whether
the outlier could pass.

Gate = tight_AABB_pass OR (centroid_score + outlier_dist > threshold)

Since centroid_score is already needed for the ball check, we get the outlier
handling for ~0 extra cost on top of ball_centroid.

Alternatively: just check if outlier norm > threshold (even cheaper, g≈0).

This gives g ≈ 2.0 (tight AABB) + 0 (outlier norm check) = 2.0.
Compare to outlier_aabb: g = 2.0 + 1.0 = 3.0.
"""

from __future__ import annotations

import torch


def enclose_cheap_outlier_aabb(keys, assign, centers, K, bf):
    """
    Tight AABB on core (bf-1) keys + norm-based check on the outlier.

    The outlier's contribution is bounded by its norm: q·outlier ≤ ||outlier||.
    This is much cheaper than exact dot product but loose.
    """
    H, N, D = keys.shape
    device = keys.device
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    # Full AABB for fallback
    lo_full = torch.full((H, K, D), float("inf"), device=device)
    hi_full = torch.full((H, K, D), float("-inf"), device=device)
    lo_full.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi_full.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)
    empty = lo_full[:, :, 0].isinf()
    if empty.any():
        lo_full[empty] = 0.0
        hi_full[empty] = 0.0

    # Find outlier: farthest from centroid
    key_centers = centers.gather(1, idx_exp)
    dist_sq = (keys - key_centers).square().sum(dim=-1)

    max_dist = torch.full((H, K), float("-inf"), device=device)
    max_dist.scatter_reduce_(1, assign, dist_sq, reduce="amax", include_self=False)

    cluster_max = max_dist.gather(1, assign)
    is_outlier = (dist_sq >= cluster_max - 1e-7) & (dist_sq > 0)

    counts = torch.zeros(H, K, device=device)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device))
    small_cluster = counts.gather(1, assign) <= 1
    is_outlier = is_outlier & ~small_cluster

    # Tight AABB without outliers
    keys_for_lo = torch.where(is_outlier.unsqueeze(-1), torch.full_like(keys, float("inf")), keys)
    keys_for_hi = torch.where(is_outlier.unsqueeze(-1), torch.full_like(keys, float("-inf")), keys)

    lo_tight = torch.full((H, K, D), float("inf"), device=device)
    hi_tight = torch.full((H, K, D), float("-inf"), device=device)
    lo_tight.scatter_reduce_(1, idx_exp, keys_for_lo, reduce="amin", include_self=False)
    hi_tight.scatter_reduce_(1, idx_exp, keys_for_hi, reduce="amax", include_self=False)
    bad = lo_tight[:, :, 0].isinf()
    if bad.any():
        lo_tight[bad] = lo_full[bad]
        hi_tight[bad] = hi_full[bad]

    # Outlier norms per cluster: max ||outlier|| in each cluster
    key_norms = keys.norm(dim=-1)
    outlier_norms = torch.where(is_outlier, key_norms, torch.zeros_like(key_norms))
    max_outlier_norm = torch.zeros(H, K, device=device)
    max_outlier_norm.scatter_reduce_(1, assign, outlier_norms, reduce="amax", include_self=True)

    def gate(q, th):
        q_exp = q.unsqueeze(1)
        th_exp = th.unsqueeze(-1)

        # Tight AABB check
        aabb_score = torch.maximum(q_exp * lo_tight, q_exp * hi_tight).sum(dim=-1)
        aabb_pass = aabb_score > th_exp

        # Outlier norm check: q·outlier ≤ ||q|| * ||outlier|| ≤ ||outlier|| for unit q
        # More precisely: for unit q, q·outlier ≤ ||outlier||
        outlier_pass = max_outlier_norm > th.unsqueeze(-1)

        return aabb_pass | outlier_pass

    span_full = (hi_full - lo_full).clamp_min(0).sum(dim=-1)
    span_tight = (hi_tight - lo_tight).clamp_min(0).sum(dim=-1)
    reduction = (span_full - span_tight) / span_full.clamp_min(1e-12)

    return gate, {
        "span_reduction_mean": float(reduction.mean()),
        "n_outliers": int(is_outlier.sum()),
        "max_outlier_norm_mean": float(max_outlier_norm.mean()),
    }


def enclose_cheap_outlier_ball_aabb(keys, assign, centers, K, bf):
    """
    Tight AABB on core keys + ball check on outlier using centroid.

    Outlier bound: q·centroid + ||outlier - centroid|| (ball bound).
    Gate cost: ~2.0 (tight AABB) + ~1.0 (centroid dot for outlier ball) = ~3.0
    But if we already compute q·centroid for another purpose, the marginal
    cost of the outlier ball is ~0 (just a comparison with centroid score + dist).

    Actually this is the same total cost as outlier_aabb. So this variant
    uses the ball check: faster to compute than scatter-based outlier dots.
    """
    H, N, D = keys.shape
    device = keys.device
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    # Find outlier
    key_centers = centers.gather(1, idx_exp)
    dist_sq = (keys - key_centers).square().sum(dim=-1)
    max_dist = torch.full((H, K), float("-inf"), device=device)
    max_dist.scatter_reduce_(1, assign, dist_sq, reduce="amax", include_self=False)
    cluster_max = max_dist.gather(1, assign)
    is_outlier = (dist_sq >= cluster_max - 1e-7) & (dist_sq > 0)
    counts = torch.zeros(H, K, device=device)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device))
    small_cluster = counts.gather(1, assign) <= 1
    is_outlier = is_outlier & ~small_cluster

    # Tight AABB
    keys_lo = torch.where(is_outlier.unsqueeze(-1), torch.full_like(keys, float("inf")), keys)
    keys_hi = torch.where(is_outlier.unsqueeze(-1), torch.full_like(keys, float("-inf")), keys)
    lo = torch.full((H, K, D), float("inf"), device=device)
    hi = torch.full((H, K, D), float("-inf"), device=device)
    lo.scatter_reduce_(1, idx_exp, keys_lo, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_exp, keys_hi, reduce="amax", include_self=False)
    lo_full = torch.full((H, K, D), float("inf"), device=device)
    hi_full = torch.full((H, K, D), float("-inf"), device=device)
    lo_full.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi_full.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)
    bad = lo[:, :, 0].isinf()
    if bad.any():
        lo[bad] = lo_full[bad] if not lo_full[bad].isinf().any() else 0.0
        hi[bad] = hi_full[bad] if not hi_full[bad].isinf().any() else 0.0

    # Ball radius = max outlier distance from centroid per cluster
    outlier_dist = torch.where(is_outlier, dist_sq.sqrt(), torch.zeros_like(dist_sq))
    ball_radius = torch.zeros(H, K, device=device)
    ball_radius.scatter_reduce_(1, assign, outlier_dist, reduce="amax", include_self=True)

    def gate(q, th):
        q_exp = q.unsqueeze(1)
        th_exp = th.unsqueeze(-1)

        # Tight AABB
        aabb_score = torch.maximum(q_exp * lo, q_exp * hi).sum(dim=-1)
        aabb_pass = aabb_score > th_exp

        # Ball check for outlier
        centroid_score = torch.einsum("hkd,hd->hk", centers, q)
        ball_pass = (centroid_score + ball_radius) > th.unsqueeze(-1)

        return aabb_pass | ball_pass

    return gate, {
        "ball_radius_mean": float(ball_radius.mean()),
    }
