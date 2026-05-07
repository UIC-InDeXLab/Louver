"""Standard Lloyd's k-means clustering on GPU."""

from __future__ import annotations

import math

import torch


def cluster_kmeans(keys: torch.Tensor, bf: int, max_iter: int = 20):
    """
    Standard Lloyd's k-means.

    Args:
        keys: (H, N, D) key tensor.
        bf: branching factor — target children per cluster.
        max_iter: number of Lloyd iterations.

    Returns:
        assign: (H, N) long tensor mapping each key to a cluster index.
        centers: (H, K, D) cluster centroids.
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    # Random init
    perm = torch.argsort(torch.rand(H, N, device=device), dim=1)
    centers = keys.gather(1, perm[:, :K].unsqueeze(-1).expand(-1, -1, D)).clone()

    for _ in range(max_iter):
        dists = torch.cdist(keys, centers)  # (H, N, K)
        assign = dists.argmin(dim=2)  # (H, N)

        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(H, K, device=device)
        new_centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
        counts.scatter_add_(1, assign, torch.ones(H, N, device=device))

        empty = counts == 0
        if empty.any():
            for h in range(H):
                ek = empty[h].nonzero(as_tuple=False).view(-1)
                if ek.numel() > 0:
                    far_idx = torch.randperm(N, device=device)[: ek.numel()]
                    new_centers[h, ek] = keys[h, far_idx]
                    counts[h, ek] = 1

        mask = counts > 0
        new_centers[mask] /= counts[mask].unsqueeze(-1)
        centers = new_centers

    assign = torch.cdist(keys, centers).argmin(dim=2)
    return assign, centers
