"""Balanced recursive PCA-tree clustering."""

from __future__ import annotations

import math

import torch

from ._balanced_utils import balanced_refine, dominant_axis, target_cluster_sizes


def _assign_tree(
    points: torch.Tensor,
    all_indices: torch.Tensor,
    leaf_sizes: torch.Tensor,
    assign_h: torch.Tensor,
    cluster_offset: int,
) -> None:
    if leaf_sizes.numel() == 1:
        assign_h[all_indices] = cluster_offset
        return

    mid = leaf_sizes.numel() // 2
    left_sizes = leaf_sizes[:mid]
    right_sizes = leaf_sizes[mid:]
    left_count = int(left_sizes.sum().item())

    subset = points[all_indices]
    axis = dominant_axis(subset)
    scores = subset @ axis
    order = torch.argsort(scores)

    left_local = order[:left_count]
    right_local = order[left_count:]
    left_idx = all_indices[left_local]
    right_idx = all_indices[right_local]

    _assign_tree(points, left_idx, left_sizes, assign_h, cluster_offset)
    _assign_tree(points, right_idx, right_sizes, assign_h, cluster_offset + mid)


def cluster_balanced_pca_tree(keys: torch.Tensor, bf: int, refine_iter: int = 1):
    """
    Recursively split each head along its local dominant direction.

    The split sizes are driven by the final leaf capacities, so every leaf ends
    up with an exact near-``bf`` point count.
    """
    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    device = keys.device
    target_sizes = target_cluster_sizes(n, bf, device)

    assign = torch.empty(h, n, dtype=torch.long, device=device)
    centers = torch.empty(h, k, d, dtype=keys.dtype, device=device)

    for head in range(h):
        points = keys[head]
        assign_h = torch.empty(n, dtype=torch.long, device=device)
        all_idx = torch.arange(n, device=device)
        _assign_tree(points, all_idx, target_sizes, assign_h, 0)
        assign_h, centers_h = balanced_refine(points, assign_h, target_sizes, refine_iter)
        assign[head] = assign_h
        centers[head] = centers_h

    return assign, centers
