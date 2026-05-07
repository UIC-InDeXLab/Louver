"""Minimum enclosing ball via iterative farthest-point centering."""

from __future__ import annotations

import torch


def enclose_min_ball(keys, assign, centers, K, bf):
    """
    Approximate minimum enclosing ball per cluster.
    Iteratively shifts each cluster's center toward its farthest point.

    Returns:
        gate: callable(q, th) -> (H, K) bool.
        info: dict with diagnostic stats.
    """
    H, N, D = keys.shape
    device = keys.device
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    meb_centers = centers.clone()

    for _ in range(10):
        parent_c = meb_centers.gather(1, idx_exp)  # (H, N, D)
        dists = (keys - parent_c).norm(dim=-1)  # (H, N)

        max_dists = torch.full((H, K), 0.0, device=device)
        max_dists.scatter_reduce_(1, assign, dists, reduce="amax", include_self=True)

        cluster_max_for_child = max_dists.gather(1, assign)  # (H, N)
        is_farthest = (dists >= cluster_max_for_child - 1e-6) & (dists > 0)

        direction = torch.zeros(H, K, D, device=device)
        weight = torch.zeros(H, K, device=device)
        diff = keys - parent_c  # (H, N, D)
        masked_diff = diff * is_farthest.unsqueeze(-1).float()
        direction.scatter_add_(1, idx_exp, masked_diff)
        weight.scatter_add_(1, assign, is_farthest.float())
        weight = weight.clamp_min(1)
        direction = direction / weight.unsqueeze(-1)

        meb_centers = meb_centers + 0.5 * direction

    # Final radii
    parent_c = meb_centers.gather(1, idx_exp)
    dists = (keys - parent_c).norm(dim=-1)
    meb_radii = torch.full((H, K), 0.0, device=device)
    meb_radii.scatter_reduce_(1, assign, dists, reduce="amax", include_self=True)

    def gate(q, th):
        scores = torch.einsum("hkd,hd->hk", meb_centers, q)
        return (scores + meb_radii) > th.unsqueeze(-1)

    return gate, {"radii_mean": float(meb_radii.mean()), "radii_max": float(meb_radii.max())}
