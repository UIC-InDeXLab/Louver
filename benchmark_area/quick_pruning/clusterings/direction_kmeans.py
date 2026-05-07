"""Direction-first k-means with original-space centroids."""

from __future__ import annotations

import math

import torch

from .kmeans_pp import cluster_kmeans_pp


def cluster_direction_kmeans(keys: torch.Tensor, bf: int, max_iter: int = 12):
    """
    Cluster on L2-normalised directions, then rebuild centroids in the original
    key space for downstream bounds.
    """
    key_norms = keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dir_keys = keys / key_norms
    assign, _ = cluster_kmeans_pp(dir_keys, bf, max_iter=max_iter)
    centers = _centers_from_assign(keys, assign, max(1, math.ceil(keys.shape[1] / bf)))
    return assign, centers


def _centers_from_assign(keys: torch.Tensor, assign: torch.Tensor, K: int) -> torch.Tensor:
    H, N, D = keys.shape
    device = keys.device

    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    counts = torch.zeros(H, K, device=device, dtype=keys.dtype)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device, dtype=keys.dtype))

    empty = counts == 0
    if empty.any():
        for h in range(H):
            empty_ids = empty[h].nonzero(as_tuple=False).flatten()
            if empty_ids.numel() == 0:
                continue
            refill = torch.randperm(N, device=device)[: empty_ids.numel()]
            centers[h, empty_ids] = keys[h, refill]
            counts[h, empty_ids] = 1

    return centers / counts.clamp_min(1).unsqueeze(-1)
