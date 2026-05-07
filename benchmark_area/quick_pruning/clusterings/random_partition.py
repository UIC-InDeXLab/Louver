"""Random partition baseline — no clustering structure."""

from __future__ import annotations

import math

import torch


def cluster_random_partition(keys: torch.Tensor, bf: int):
    """
    Assign keys to clusters uniformly at random (baseline).

    Returns:
        assign: (H, N) cluster assignments.
        centers: (H, K, D) cluster centroids.
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    assign = torch.randint(0, K, (H, N), device=device)

    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    counts = torch.zeros(H, K, device=device)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device))
    mask = counts > 0
    centers[mask] /= counts[mask].unsqueeze(-1)

    return assign, centers
