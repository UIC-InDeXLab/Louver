"""Balanced PCA-tree clustering for rotated, low-span partitions."""

from __future__ import annotations

import math

import torch


def cluster_pca_tree(keys: torch.Tensor, bf: int, rank: int = 4):
    """
    Project each head onto a small global PCA basis, then recursively split
    along the widest projected axis at the median.

    This keeps the balanced-build behavior of KD-tree clustering while making
    the split axes data-adaptive instead of tied to the original coordinates.
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

    assign = torch.zeros(H, N, device=device, dtype=torch.long)
    for h in range(H):
        _split_projected_tree(proj[h], assign[h], K)

    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    counts = torch.zeros(H, K, device=device, dtype=keys.dtype)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device, dtype=keys.dtype))
    centers = centers / counts.clamp_min(1).unsqueeze(-1)
    return assign, centers


def _split_projected_tree(proj_h: torch.Tensor, assign_h: torch.Tensor, total_k: int) -> None:
    n, _ = proj_h.shape
    device = proj_h.device
    stack = [(torch.ones(n, device=device, dtype=torch.bool), 0, total_k)]

    while stack:
        mask, start_id, num_clusters = stack.pop()
        if num_clusters <= 1:
            assign_h[mask] = start_id
            continue

        subset = proj_h[mask]
        n_subset = int(subset.shape[0])
        if n_subset == 0:
            continue

        lo = subset.min(dim=0).values
        hi = subset.max(dim=0).values
        split_dim = int((hi - lo).argmax())

        vals = subset[:, split_dim]
        order = vals.argsort()
        left_count = n_subset // 2
        left_local = torch.zeros(n_subset, device=device, dtype=torch.bool)
        left_local[order[:left_count]] = True

        indices = mask.nonzero(as_tuple=False).squeeze(-1)
        left_indices = indices[left_local]
        right_indices = indices[~left_local]
        if left_indices.numel() == 0 or right_indices.numel() == 0:
            assign_h[indices] = start_id
            continue

        left_k = max(1, round(num_clusters * left_indices.numel() / n_subset))
        left_k = min(left_k, num_clusters - 1)
        right_k = num_clusters - left_k

        left_mask = torch.zeros(n, device=device, dtype=torch.bool)
        left_mask[left_indices] = True
        right_mask = torch.zeros(n, device=device, dtype=torch.bool)
        right_mask[right_indices] = True

        stack.append((right_mask, start_id + left_k, right_k))
        stack.append((left_mask, start_id, left_k))
