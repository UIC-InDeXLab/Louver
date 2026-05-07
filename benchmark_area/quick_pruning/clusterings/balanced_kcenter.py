"""Balanced k-center seeding with exact-capacity refinement."""

from __future__ import annotations

import math

import torch

from ._balanced_utils import (
    balanced_assign_from_cost,
    balanced_refine,
    pairwise_sq_dists,
    target_cluster_sizes,
)


def _kcenter_init(points: torch.Tensor, k: int) -> torch.Tensor:
    """One-head farthest-point seeding."""
    n, d = points.shape
    device = points.device
    centers = torch.empty(k, d, device=device, dtype=points.dtype)

    idx0 = torch.randint(n, (1,), device=device)
    centers[0] = points[idx0]
    min_dist = pairwise_sq_dists(points, centers[:1]).squeeze(1)

    for j in range(1, k):
        farthest = min_dist.argmax()
        centers[j] = points[farthest]
        dist_j = pairwise_sq_dists(points, centers[j : j + 1]).squeeze(1)
        min_dist = torch.minimum(min_dist, dist_j)

    return centers


def cluster_balanced_kcenter(keys: torch.Tensor, bf: int, refine_iter: int = 1):
    """
    Balanced variant of farthest-point clustering.

    Farthest-point seeding targets small worst-case radius; exact-capacity
    reassignment keeps each cluster near ``bf`` points instead of letting a few
    centers absorb many keys.
    """
    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    device = keys.device
    target_sizes = target_cluster_sizes(n, bf, device)

    assign = torch.empty(h, n, dtype=torch.long, device=device)
    centers = torch.empty(h, k, d, dtype=keys.dtype, device=device)

    for head in range(h):
        points = keys[head]
        centers_h = _kcenter_init(points, k)
        assign_h = balanced_assign_from_cost(pairwise_sq_dists(points, centers_h), target_sizes)
        assign_h, centers_h = balanced_refine(points, assign_h, target_sizes, refine_iter)
        assign[head] = assign_h
        centers[head] = centers_h

    return assign, centers
