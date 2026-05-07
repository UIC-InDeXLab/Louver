"""Clustering on ray features for ball-friendly partitions."""

from __future__ import annotations

import math

import torch


def _ray_features(points: torch.Tensor, norm_weight: float) -> torch.Tensor:
    norms = points.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dirs = points / norms
    log_norm = norms.log()
    log_norm = log_norm - log_norm.mean(dim=0, keepdim=True)
    log_norm = log_norm / log_norm.std(dim=0, keepdim=True).clamp_min(1e-6)
    return torch.cat([dirs, norm_weight * log_norm], dim=-1)


def _kmeans_pp_init(points: torch.Tensor, k: int) -> torch.Tensor:
    n, d = points.shape
    centers = torch.empty(k, d, device=points.device, dtype=points.dtype)

    idx0 = torch.randint(n, (1,), device=points.device)
    centers[0] = points[idx0]
    min_dist_sq = (points - centers[:1]).square().sum(dim=-1)

    for j in range(1, k):
        probs = min_dist_sq / min_dist_sq.sum().clamp_min(1e-30)
        idx = torch.multinomial(probs, 1)
        centers[j] = points[idx]
        dist_sq = (points - centers[j : j + 1]).square().sum(dim=-1)
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq)

    return centers


def _kcenter_init(points: torch.Tensor, k: int) -> torch.Tensor:
    n, d = points.shape
    centers = torch.empty(k, d, device=points.device, dtype=points.dtype)

    idx0 = torch.randint(n, (1,), device=points.device)
    centers[0] = points[idx0]
    min_dist_sq = (points - centers[:1]).square().sum(dim=-1)

    for j in range(1, k):
        farthest = min_dist_sq.argmax()
        centers[j] = points[farthest]
        dist_sq = (points - centers[j : j + 1]).square().sum(dim=-1)
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq)

    return centers


def _recompute_feature_centers(features: torch.Tensor, assign: torch.Tensor, k: int) -> torch.Tensor:
    n, d = features.shape
    centers = torch.zeros(k, d, device=features.device, dtype=features.dtype)
    counts = torch.zeros(k, device=features.device, dtype=features.dtype)
    centers.scatter_add_(0, assign.unsqueeze(-1).expand(-1, d), features)
    counts.scatter_add_(0, assign, torch.ones(n, device=features.device, dtype=features.dtype))

    empty = counts == 0
    if empty.any():
        refill = torch.randperm(n, device=features.device)[: int(empty.sum().item())]
        centers[empty] = features[refill]
        counts[empty] = 1

    centers = centers / counts.clamp_min(1).unsqueeze(-1)
    dir_part = centers[:, :-1]
    dir_norm = dir_part.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    centers[:, :-1] = dir_part / dir_norm
    return centers


def _recompute_data_centers(points: torch.Tensor, assign: torch.Tensor, k: int) -> torch.Tensor:
    n, d = points.shape
    centers = torch.zeros(k, d, device=points.device, dtype=points.dtype)
    counts = torch.zeros(k, device=points.device, dtype=points.dtype)
    centers.scatter_add_(0, assign.unsqueeze(-1).expand(-1, d), points)
    counts.scatter_add_(0, assign, torch.ones(n, device=points.device, dtype=points.dtype))

    empty = counts == 0
    if empty.any():
        refill = torch.randperm(n, device=points.device)[: int(empty.sum().item())]
        centers[empty] = points[refill]
        counts[empty] = 1

    return centers / counts.clamp_min(1).unsqueeze(-1)


def _approx_meb_center(points: torch.Tensor, n_iter: int) -> torch.Tensor:
    if points.shape[0] == 1:
        return points[0]

    center = points.mean(dim=0)
    for t in range(1, n_iter + 1):
        dists = (points - center).norm(dim=-1)
        farthest = points[dists.argmax()]
        center = center + (farthest - center) / (t + 1)
    return center


def _recompute_meb_centers(points: torch.Tensor, assign: torch.Tensor, k: int, n_iter: int) -> torch.Tensor:
    n, d = points.shape
    centers = torch.empty(k, d, device=points.device, dtype=points.dtype)

    for cid in range(k):
        idx = (assign == cid).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            centers[cid] = points[torch.randint(n, (1,), device=points.device)]
            continue
        centers[cid] = _approx_meb_center(points[idx], n_iter=n_iter)

    return centers


def _run_ray(points: torch.Tensor, k: int, max_iter: int, norm_weight: float, init_mode: str):
    features = _ray_features(points, norm_weight=norm_weight)
    if init_mode == "kcenter":
        feature_centers = _kcenter_init(features, k)
    else:
        feature_centers = _kmeans_pp_init(features, k)

    assign = torch.cdist(features.unsqueeze(0), feature_centers.unsqueeze(0)).squeeze(0).argmin(dim=1)
    for _ in range(max_iter):
        next_feature_centers = _recompute_feature_centers(features, assign, k)
        next_assign = torch.cdist(features.unsqueeze(0), next_feature_centers.unsqueeze(0)).squeeze(0).argmin(dim=1)
        feature_centers = next_feature_centers
        if torch.equal(next_assign, assign):
            break
        assign = next_assign

    centers = _recompute_data_centers(points, assign, k)
    return assign, centers


def cluster_ray_kmeans(keys: torch.Tensor, bf: int, max_iter: int = 12, norm_weight: float = 0.5):
    """
    Cluster on shared direction plus log-norm similarity.

    Ball bounds improve when points in a cluster lie on a common ray from the
    origin instead of merely sharing a nearby Euclidean mean.
    """
    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    assign = torch.empty(h, n, dtype=torch.long, device=keys.device)
    centers = torch.empty(h, k, d, dtype=keys.dtype, device=keys.device)

    for head in range(h):
        assign_h, centers_h = _run_ray(keys[head], k, max_iter=max_iter, norm_weight=norm_weight, init_mode="kmeans")
        assign[head] = assign_h
        centers[head] = centers_h

    return assign, centers


def cluster_ray_kcenter(keys: torch.Tensor, bf: int, max_iter: int = 10, norm_weight: float = 0.5):
    """Ray-feature clustering with farthest-point seeding."""
    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    assign = torch.empty(h, n, dtype=torch.long, device=keys.device)
    centers = torch.empty(h, k, d, dtype=keys.dtype, device=keys.device)

    for head in range(h):
        assign_h, centers_h = _run_ray(keys[head], k, max_iter=max_iter, norm_weight=norm_weight, init_mode="kcenter")
        assign[head] = assign_h
        centers[head] = centers_h

    return assign, centers


def cluster_ray_kcenter_meb(
    keys: torch.Tensor,
    bf: int,
    max_iter: int = 10,
    norm_weight: float = 0.5,
    meb_iter: int = 10,
):
    """
    Ray-space k-center assignments with approximate MEB centers in data space.

    This keeps the ray-oriented partition but hands a tighter center to
    ``ball_centroid`` than the plain cluster mean.
    """
    assign, _ = cluster_ray_kcenter(keys, bf, max_iter=max_iter, norm_weight=norm_weight)
    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    centers = torch.empty(h, k, d, dtype=keys.dtype, device=keys.device)

    for head in range(h):
        centers[head] = _recompute_meb_centers(keys[head], assign[head], k, n_iter=meb_iter)

    return assign, centers
