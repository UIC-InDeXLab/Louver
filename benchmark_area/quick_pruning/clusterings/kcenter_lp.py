"""Lp-aware k-center clustering."""

from __future__ import annotations

import math

import torch

from ._lp_utils import cdist_lp, recompute_centers_lp, validate_p


def cluster_kcenter_lp(keys: torch.Tensor, bf: int, p: float = 2.0, refine_iter: int = 5):
    """
    K-center clustering with Lp distances.

    Seeding uses greedy farthest-point insertion under the requested Lp metric.
    Refinement keeps the same metric for reassignment and uses a simple
    Lp-aware recentering rule:
    - p=1: coordinate-wise median
    - p=inf: axis-aligned box midpoint
    - otherwise: arithmetic mean
    """
    p = validate_p(p)
    h, n, d = keys.shape
    device = keys.device
    k = max(1, math.ceil(n / bf))

    center_indices = torch.zeros(h, k, dtype=torch.long, device=device)
    center_indices[:, 0] = torch.randint(0, n, (h,), device=device)

    first = keys.gather(1, center_indices[:, :1].unsqueeze(-1).expand(-1, 1, d))
    min_dist = cdist_lp(keys, first, p).squeeze(-1)

    for i in range(1, k):
        farthest = min_dist.argmax(dim=1)
        center_indices[:, i] = farthest
        new_center = keys.gather(1, farthest.view(h, 1, 1).expand(-1, 1, d))
        new_dist = cdist_lp(keys, new_center, p).squeeze(-1)
        min_dist = torch.minimum(min_dist, new_dist)

    centers = keys.gather(1, center_indices.unsqueeze(-1).expand(-1, -1, d))

    for _ in range(refine_iter):
        dists = cdist_lp(keys, centers, p)
        assign = dists.argmin(dim=2)
        new_centers = recompute_centers_lp(keys, assign, k, p)
        next_assign = cdist_lp(keys, new_centers, p).argmin(dim=2)
        centers = new_centers
        if torch.equal(next_assign, assign):
            assign = next_assign
            break
        assign = next_assign

    assign = cdist_lp(keys, centers, p).argmin(dim=2)
    return assign, centers


def make_cluster_kcenter_lp(p: float, refine_iter: int = 5):
    """Factory that binds a concrete p value into the clustering callable."""
    p = validate_p(p)

    def _cluster_kcenter_lp(keys: torch.Tensor, bf: int):
        return cluster_kcenter_lp(keys, bf, p=p, refine_iter=refine_iter)

    return _cluster_kcenter_lp
