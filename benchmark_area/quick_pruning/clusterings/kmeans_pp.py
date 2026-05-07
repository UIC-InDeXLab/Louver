"""K-means with k-means++ initialization."""

from __future__ import annotations

import math

import torch


def cluster_kmeans_pp(keys: torch.Tensor, bf: int, max_iter: int = 20):
    """
    K-means with k-means++ seeding: pick initial centers with probability
    proportional to squared distance from the nearest existing center.
    This avoids poor local minima from random init.

    Args:
        keys: (H, N, D) key tensor.
        bf: branching factor.
        max_iter: Lloyd iterations after init.

    Returns:
        assign: (H, N) long.
        centers: (H, K, D).
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    # ── k-means++ initialization ──
    centers = torch.empty(H, K, D, device=device, dtype=keys.dtype)

    # First center: random
    idx0 = torch.randint(N, (H,), device=device)
    centers[:, 0, :] = keys[torch.arange(H, device=device), idx0]

    # Squared distances to nearest chosen center
    min_dist_sq = torch.full((H, N), float("inf"), device=device)

    for j in range(1, K):
        # Update min distances with the last added center
        d = ((keys - centers[:, j - 1 : j, :]) ** 2).sum(dim=-1)  # (H, N)
        min_dist_sq = torch.minimum(min_dist_sq, d)

        # Sample next center proportional to min_dist_sq
        probs = min_dist_sq / min_dist_sq.sum(dim=1, keepdim=True).clamp_min(1e-30)
        chosen = torch.multinomial(probs, 1).squeeze(-1)  # (H,)
        centers[:, j, :] = keys[torch.arange(H, device=device), chosen]

    # ── Lloyd's iterations ──
    for _ in range(max_iter):
        dists = torch.cdist(keys, centers)  # (H, N, K)
        assign = dists.argmin(dim=2)

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
