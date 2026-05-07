"""Outlier-ball: remove one farthest point, then fit a centroid ball."""

from __future__ import annotations

import torch


def enclose_outlier_ball_centroid(keys, assign, centers, K, bf):
    """
    Remove one farthest point per cluster, build a centroid ball on the
    remaining points, and check the removed point with a direct dot product.
    """
    H, N, D = keys.shape
    device = keys.device
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    counts = torch.zeros(H, K, device=device)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device))

    parent_full = centers.gather(1, idx_exp)
    full_dists = (keys - parent_full).norm(dim=-1)
    full_radii = torch.full((H, K), 0.0, device=device)
    full_radii.scatter_reduce_(1, assign, full_dists, reduce="amax", include_self=True)

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

    inlier_mask = ~is_outlier
    inlier_counts = counts - (counts > 1).to(counts.dtype)
    key_sums = torch.zeros(H, K, D, device=device)
    key_sums.scatter_add_(1, idx_exp, keys * inlier_mask.unsqueeze(-1))

    inlier_centers = torch.zeros(H, K, D, device=device)
    nonempty = inlier_counts > 0
    inlier_centers[nonempty] = key_sums[nonempty] / inlier_counts[nonempty].unsqueeze(-1)

    parent_inlier = inlier_centers.gather(1, idx_exp)
    inlier_dists = (keys - parent_inlier).norm(dim=-1)
    inlier_dists = torch.where(is_outlier, torch.full_like(inlier_dists, float("-inf")), inlier_dists)

    inlier_radii = torch.full((H, K), float("-inf"), device=device)
    inlier_radii.scatter_reduce_(1, assign, inlier_dists, reduce="amax", include_self=False)
    bad = ~torch.isfinite(inlier_radii)
    if bad.any():
        inlier_radii[bad] = full_radii[bad]
        inlier_centers[bad] = centers[bad]

    def gate(q, th):
        ball_scores = torch.einsum("hkd,hd->hk", inlier_centers, q)
        ball_pass = (ball_scores + inlier_radii) > th.unsqueeze(-1)

        dots = (keys * q.unsqueeze(1)).sum(dim=-1)
        dots_masked = torch.where(is_outlier, dots, torch.full_like(dots, float("-inf")))
        max_outlier = torch.full((H, K), float("-inf"), device=device)
        max_outlier.scatter_reduce_(1, assign, dots_masked, reduce="amax", include_self=False)
        outlier_pass = max_outlier > th.unsqueeze(-1)

        return ball_pass | outlier_pass

    radius_reduction = (full_radii - inlier_radii) / full_radii.clamp_min(1e-12)
    return gate, {
        "radius_reduction_mean": float(radius_reduction.mean()),
        "n_outliers_total": int(is_outlier.sum()),
    }
