"""Low-rank PCA-space k-means clustering."""

from __future__ import annotations

import math

import torch


def cluster_pca_kmeans(keys: torch.Tensor, bf: int, rank: int = 8, max_iter: int = 15):
    """
    Run Lloyd's algorithm in a small global PCA subspace, then compute the
    final cluster centroids in the original space.

    This is intended to pair well with rotated-box style enclosures while
    reducing clustering cost relative to full-space k-means.
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device
    R = min(max(1, rank), D)

    key_mean = keys.mean(dim=1, keepdim=True)
    centered = keys - key_mean
    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    basis = vt[:, :R, :].transpose(-2, -1).contiguous()
    proj = torch.bmm(centered, basis)

    perm = torch.argsort(torch.rand(H, N, device=device), dim=1)
    proj_centers = proj.gather(1, perm[:, :K].unsqueeze(-1).expand(-1, -1, R)).clone()
    centers = keys.gather(1, perm[:, :K].unsqueeze(-1).expand(-1, -1, D)).clone()

    for _ in range(max_iter):
        dists = torch.cdist(proj, proj_centers)
        assign = dists.argmin(dim=2)

        new_centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
        counts = torch.zeros(H, K, device=device, dtype=keys.dtype)
        new_centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
        counts.scatter_add_(1, assign, torch.ones(H, N, device=device, dtype=keys.dtype))

        empty = counts == 0
        if empty.any():
            for h in range(H):
                empty_ids = empty[h].nonzero(as_tuple=False).flatten()
                if empty_ids.numel() == 0:
                    continue
                refill = torch.randperm(N, device=device)[: empty_ids.numel()]
                new_centers[h, empty_ids] = keys[h, refill]
                counts[h, empty_ids] = 1

        centers = new_centers / counts.clamp_min(1).unsqueeze(-1)
        proj_centers = torch.bmm(centers - key_mean, basis)

    assign = torch.cdist(proj, proj_centers).argmin(dim=2)
    return assign, centers
