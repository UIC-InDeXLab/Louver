"""Balanced farthest-point clustering on ray features."""

from __future__ import annotations

import math

import torch

from ._balanced_utils import balanced_assign_from_cost, target_cluster_sizes, recompute_centers
from .balanced_ray_kmeans import _pairwise_sq_dists, _ray_features, _recompute_feature_centers


def _ray_kcenter_init(features: torch.Tensor, k: int) -> torch.Tensor:
    n, d = features.shape
    device = features.device
    centers = torch.empty(k, d, device=device, dtype=features.dtype)

    idx0 = torch.randint(n, (1,), device=device)
    centers[0] = features[idx0]
    min_dist = _pairwise_sq_dists(features, centers[:1]).squeeze(1)

    for j in range(1, k):
        farthest = min_dist.argmax()
        centers[j] = features[farthest]
        dist_j = _pairwise_sq_dists(features, centers[j : j + 1]).squeeze(1)
        min_dist = torch.minimum(min_dist, dist_j)

    return centers


def cluster_balanced_ray_kcenter(
    keys: torch.Tensor,
    bf: int,
    norm_weight: float = 0.75,
    refine_iter: int = 2,
):
    """
    Balanced k-center in ray space.

    Compared with ray k-means, this prioritizes small worst-case angular/norm
    spread, which is often a better proxy for the final enclosing-ball radius.
    """
    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    device = keys.device
    target_sizes = target_cluster_sizes(n, bf, device)

    assign = torch.empty(h, n, dtype=torch.long, device=device)
    centers = torch.empty(h, k, d, dtype=keys.dtype, device=device)

    for head in range(h):
        points = keys[head]
        feat = _ray_features(points, norm_weight=norm_weight)
        feat_centers = _ray_kcenter_init(feat, k)
        assign_h = balanced_assign_from_cost(_pairwise_sq_dists(feat, feat_centers), target_sizes)

        for _ in range(refine_iter):
            feat_centers = _recompute_feature_centers(feat, assign_h, k)
            assign_next = balanced_assign_from_cost(_pairwise_sq_dists(feat, feat_centers), target_sizes)
            if torch.equal(assign_next, assign_h):
                break
            assign_h = assign_next

        assign[head] = assign_h
        centers[head] = recompute_centers(points, assign_h, k)

    return assign, centers
