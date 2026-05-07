"""Ball enclosure centered at the k-means centroid."""

from __future__ import annotations

import torch


def enclose_ball_centroid(keys, assign, centers, K, bf):
    """
    Ball centered at the cluster centroid with radius = max distance
    from the centroid to any child key.

    Returns:
        gate: callable(q, th) -> (H, K) bool mask of clusters that pass.
        info: dict with diagnostic stats.
    """
    H, N, D = keys.shape
    device = keys.device

    parent_for_child = centers.gather(1, assign.unsqueeze(-1).expand(-1, -1, D))
    dists = (keys - parent_for_child).norm(dim=-1)  # (H, N)

    radii = torch.full((H, K), 0.0, device=device)
    radii.scatter_reduce_(1, assign, dists, reduce="amax", include_self=True)

    def gate(q, th):
        scores = torch.einsum("hkd,hd->hk", centers, q)  # (H, K)
        return (scores + radii) > th.unsqueeze(-1)

    return gate, {"radii_mean": float(radii.mean()), "radii_max": float(radii.max())}
