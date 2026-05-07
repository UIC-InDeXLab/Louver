"""Outlier span-ball: remove one farthest point, then fit a span ball."""

from __future__ import annotations

import torch


def enclose_outlier_span_ball(keys, assign, centers, K, bf):
    """
    Remove one farthest point per cluster, build the span-ball on the
    remaining points, and check the removed point with a direct dot product.
    """
    H, N, D = keys.shape
    device = keys.device
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    counts = torch.zeros(H, K, device=device)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device))

    parent_full = centers.gather(1, idx_exp)
    dist_sq = (keys - parent_full).square().sum(dim=-1)
    max_dist = torch.full((H, K), float("-inf"), device=device)
    max_dist.scatter_reduce_(1, assign, dist_sq, reduce="amax", include_self=False)
    cluster_max = max_dist.gather(1, assign)
    is_farthest = dist_sq >= (cluster_max - 1e-7)

    positions = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(H, -1)
    masked_pos = torch.where(is_farthest, positions, torch.full_like(positions, N))
    outlier_pos = torch.full((H, K), N, device=device, dtype=torch.long)
    outlier_pos.scatter_reduce_(1, assign, masked_pos, reduce="amin", include_self=True)

    small_cluster = counts.gather(1, assign) <= 1
    is_outlier = (positions == outlier_pos.gather(1, assign)) & ~small_cluster

    lo_full = torch.full((H, K, D), float("inf"), device=device)
    hi_full = torch.full((H, K, D), float("-inf"), device=device)
    lo_full.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi_full.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)
    empty = lo_full[:, :, 0].isinf()
    if empty.any():
        lo_full[empty] = 0.0
        hi_full[empty] = 0.0

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

    mid = (lo_tight + hi_tight) / 2
    half_span = (hi_tight - lo_tight) / 2
    tight_radius = half_span.norm(dim=-1)

    full_half_span = (hi_full - lo_full) / 2
    full_radius = full_half_span.norm(dim=-1)

    def gate(q, th):
        span_scores = torch.einsum("hkd,hd->hk", mid, q)
        span_pass = (span_scores + tight_radius) > th.unsqueeze(-1)

        dots = (keys * q.unsqueeze(1)).sum(dim=-1)
        dots_masked = torch.where(is_outlier, dots, torch.full_like(dots, float("-inf")))
        max_outlier = torch.full((H, K), float("-inf"), device=device)
        max_outlier.scatter_reduce_(1, assign, dots_masked, reduce="amax", include_self=False)
        outlier_pass = max_outlier > th.unsqueeze(-1)

        return span_pass | outlier_pass

    radius_reduction = (full_radius - tight_radius) / full_radius.clamp_min(1e-12)
    return gate, {
        "radius_reduction_mean": float(radius_reduction.mean()),
        "n_outliers_total": int(is_outlier.sum()),
    }
