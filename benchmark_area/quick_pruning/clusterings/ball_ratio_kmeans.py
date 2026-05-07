"""Clustering methods that optimize ball radius relative to center norm."""

from __future__ import annotations

import math

import torch


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


def _kcenter_init(points: torch.Tensor, k: int) -> torch.Tensor:
    n, d = points.shape
    device = points.device
    centers = torch.empty(k, d, device=device, dtype=points.dtype)

    idx0 = torch.randint(n, (1,), device=device)
    centers[0] = points[idx0]
    min_dist = (points - centers[:1]).square().sum(dim=-1)

    for j in range(1, k):
        farthest = min_dist.argmax()
        centers[j] = points[farthest]
        dist_j = (points - centers[j : j + 1]).square().sum(dim=-1)
        min_dist = torch.minimum(min_dist, dist_j)

    return centers


def _centers_from_assign(keys: torch.Tensor, assign: torch.Tensor, k: int) -> torch.Tensor:
    n, d = keys.shape
    centers = torch.zeros(k, d, device=keys.device, dtype=keys.dtype)
    counts = torch.zeros(k, device=keys.device, dtype=keys.dtype)
    centers.scatter_add_(0, assign.unsqueeze(-1).expand(-1, d), keys)
    counts.scatter_add_(0, assign, torch.ones(n, device=keys.device, dtype=keys.dtype))

    empty = counts == 0
    if empty.any():
        refill = torch.randperm(n, device=keys.device)[: int(empty.sum().item())]
        centers[empty] = keys[refill]
        counts[empty] = 1

    return centers / counts.clamp_min(1).unsqueeze(-1)


def _ball_ratio_assign(points: torch.Tensor, centers: torch.Tensor, norm_power: float) -> torch.Tensor:
    dist_sq = torch.cdist(points.unsqueeze(0), centers.unsqueeze(0)).squeeze(0).square()
    center_norm_sq = centers.square().sum(dim=-1).clamp_min(1e-6)
    score = dist_sq / center_norm_sq.pow(norm_power).unsqueeze(0)
    return score.argmin(dim=1)


def _run_ball_ratio(points: torch.Tensor, k: int, max_iter: int, norm_power: float, init_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if init_mode == "kcenter":
        centers = _kcenter_init(points, k)
    else:
        centers = _kmeans_pp_init(points, k)

    assign = _ball_ratio_assign(points, centers, norm_power=norm_power)
    for _ in range(max_iter):
        centers_next = _centers_from_assign(points, assign, k)
        assign_next = _ball_ratio_assign(points, centers_next, norm_power=norm_power)
        if torch.equal(assign_next, assign):
            centers = centers_next
            break
        centers = centers_next
        assign = assign_next

    centers = _centers_from_assign(points, assign, k)
    return assign, centers


def cluster_ball_ratio_kmeans(
    keys: torch.Tensor,
    bf: int,
    max_iter: int = 12,
    norm_power: float = 1.0,
):
    """
    Lloyd-style clustering with assignment cost ``||x-c||^2 / ||c||^(2p)``.

    Ball gates prefer clusters with small radius around a strong center.
    This objective is a direct proxy for that ratio.
    """
    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    device = keys.device

    assign = torch.empty(h, n, dtype=torch.long, device=device)
    centers = torch.empty(h, k, d, dtype=keys.dtype, device=device)
    for head in range(h):
        assign_h, centers_h = _run_ball_ratio(
            keys[head], k=k, max_iter=max_iter, norm_power=norm_power, init_mode="kmeans"
        )
        assign[head] = assign_h
        centers[head] = centers_h
    return assign, centers


def cluster_ball_ratio_kcenter(
    keys: torch.Tensor,
    bf: int,
    max_iter: int = 12,
    norm_power: float = 1.0,
):
    """Ball-ratio clustering with farthest-point seeding."""
    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    device = keys.device

    assign = torch.empty(h, n, dtype=torch.long, device=device)
    centers = torch.empty(h, k, d, dtype=keys.dtype, device=device)
    for head in range(h):
        assign_h, centers_h = _run_ball_ratio(
            keys[head], k=k, max_iter=max_iter, norm_power=norm_power, init_mode="kcenter"
        )
        assign[head] = assign_h
        centers[head] = centers_h
    return assign, centers
