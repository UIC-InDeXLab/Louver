"""Variance-whitened PQ-subspace clustering."""

from __future__ import annotations

import math

import torch


def cluster_whitened_pq(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int = 4,
    max_iter: int = 10,
):
    """
    Run PQ-subspace clustering in globally whitened coordinates, then compute
    final centroids in the original space.
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    mean = keys.mean(dim=1, keepdim=True)
    scale = keys.std(dim=1, keepdim=True).clamp_min(1e-4)
    norm_keys = (keys - mean) / scale

    sub_dim = D // n_subspaces
    remainder = D % n_subspaces
    sub_k = max(2, int(round(K ** (1.0 / n_subspaces))))

    sub_assigns = []
    offset = 0
    for s in range(n_subspaces):
        sd = sub_dim + (1 if s < remainder else 0)
        sub_keys = norm_keys[:, :, offset : offset + sd].contiguous()
        offset += sd

        perm = torch.argsort(torch.rand(H, N, device=device), dim=1)
        centers = sub_keys.gather(1, perm[:, :sub_k].unsqueeze(-1).expand(-1, -1, sd)).clone()

        for _ in range(max_iter):
            dists = torch.cdist(sub_keys, centers)
            sa = dists.argmin(dim=2)
            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(H, sub_k, device=device, dtype=keys.dtype)
            new_centers.scatter_add_(1, sa.unsqueeze(-1).expand(-1, -1, sd), sub_keys)
            counts.scatter_add_(1, sa, torch.ones(H, N, device=device, dtype=keys.dtype))
            mask = counts > 0
            new_centers[mask] /= counts[mask].unsqueeze(-1)
            new_centers[~mask] = centers[~mask]
            centers = new_centers

        sub_assigns.append(torch.cdist(sub_keys, centers).argmin(dim=2))

    composite = sub_assigns[0]
    multiplier = sub_k
    for sa in sub_assigns[1:]:
        composite = composite * multiplier + sa
        multiplier *= sub_k
    assign = composite % K

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

    centers = centers / counts.clamp_min(1).unsqueeze(-1)
    norm_centers = (centers - mean) / scale
    assign = torch.cdist(norm_keys, norm_centers).argmin(dim=2)
    return assign, centers
