"""LSH-style random-projection clustering."""

from __future__ import annotations

import math

import torch


def cluster_random_projection(keys: torch.Tensor, bf: int, n_projections: int = 8):
    """
    Project keys onto random unit vectors, binarise by median, hash into
    buckets, then refine with a nearest-center reassignment.

    Returns:
        assign: (H, N) cluster assignments.
        centers: (H, K, D) cluster centroids.
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    n_proj = min(n_projections, max(1, int(math.log2(K)) + 1))
    proj = torch.randn(D, n_proj, device=device)
    proj = proj / proj.norm(dim=0, keepdim=True)

    projected = keys @ proj  # (H, N, n_proj)

    medians = projected.median(dim=1, keepdim=True).values
    bits = (projected > medians).long()  # (H, N, n_proj)

    powers = (2 ** torch.arange(n_proj, device=device)).long()
    bucket_ids = (bits * powers.view(1, 1, -1)).sum(dim=-1)  # (H, N)

    assign = bucket_ids % K

    # Compute centers from assignment
    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    counts = torch.zeros(H, K, device=device)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device))
    empty = counts == 0
    if empty.any():
        for h in range(H):
            ek = empty[h].nonzero(as_tuple=False).view(-1)
            if ek.numel() > 0:
                centers[h, ek] = keys[h, torch.randperm(N, device=device)[: ek.numel()]]
                counts[h, ek] = 1
    mask = counts > 0
    centers[mask] /= counts[mask].unsqueeze(-1)

    # Reassign to nearest center
    assign = torch.cdist(keys, centers).argmin(dim=2)
    return assign, centers
