"""Balanced clustering on ray features for ball-style enclosures."""

from __future__ import annotations

import math

import torch

from ._balanced_utils import balanced_assign_from_cost, target_cluster_sizes, recompute_centers


def _ray_features(points: torch.Tensor, norm_weight: float) -> torch.Tensor:
    norms = points.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dirs = points / norms
    log_norm = norms.log()
    log_norm = log_norm - log_norm.mean(dim=0, keepdim=True)
    log_norm = log_norm / log_norm.std(dim=0, keepdim=True).clamp_min(1e-6)
    return torch.cat([dirs, log_norm * norm_weight], dim=-1)


def _kmeans_pp_init(points: torch.Tensor, k: int) -> torch.Tensor:
    n, d = points.shape
    device = points.device
    centers = torch.empty(k, d, device=device, dtype=points.dtype)

    idx0 = torch.randint(n, (1,), device=device)
    centers[0] = points[idx0]
    min_dist = (points - centers[:1]).square().sum(dim=-1)

    for j in range(1, k):
        probs = min_dist / min_dist.sum().clamp_min(1e-12)
        idx = torch.multinomial(probs, 1)
        centers[j] = points[idx]
        dist_j = (points - centers[j : j + 1]).square().sum(dim=-1)
        min_dist = torch.minimum(min_dist, dist_j)

    return centers


def _recompute_feature_centers(features: torch.Tensor, assign: torch.Tensor, k: int) -> torch.Tensor:
    n, d = features.shape
    centers = torch.zeros(k, d, device=features.device, dtype=features.dtype)
    centers.scatter_add_(0, assign.unsqueeze(-1).expand(-1, d), features)
    counts = torch.bincount(assign, minlength=k).to(features.dtype).clamp_min(1)
    centers = centers / counts.unsqueeze(-1)

    # Renormalize directional part so the next assignment keeps angular meaning.
    dir_part = centers[:, :-1]
    dir_norm = dir_part.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    centers[:, :-1] = dir_part / dir_norm
    return centers


def _pairwise_sq_dists(points: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (points[:, None, :] - centers[None, :, :]).square().sum(dim=-1)


def cluster_balanced_ray_kmeans(
    keys: torch.Tensor,
    bf: int,
    norm_weight: float = 0.75,
    refine_iter: int = 3,
):
    """
    Balanced Lloyd updates in a direction+log-norm feature space.

    Ball gates benefit when a cluster has a large center norm relative to its
    radius. Grouping points by a shared ray (direction plus similar norm) is a
    direct proxy for that objective.
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
        feat_centers = _kmeans_pp_init(feat, k)
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
