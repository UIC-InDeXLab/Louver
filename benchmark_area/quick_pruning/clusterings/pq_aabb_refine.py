"""PQ-initialized refinements for AABB-style enclosures."""

from __future__ import annotations

import math

import torch

from ._balanced_utils import balanced_assign_from_cost, recompute_centers, target_cluster_sizes
from .pq_subspace import cluster_pq_subspace


def _box_stats(points: torch.Tensor, assign: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    n, d = points.shape
    idx_exp = assign.unsqueeze(-1).expand(-1, d)

    lo = torch.full((k, d), float("inf"), device=points.device, dtype=points.dtype)
    hi = torch.full((k, d), float("-inf"), device=points.device, dtype=points.dtype)
    lo.scatter_reduce_(0, idx_exp, points, reduce="amin", include_self=False)
    hi.scatter_reduce_(0, idx_exp, points, reduce="amax", include_self=False)

    empty = lo[:, 0].isinf()
    if empty.any():
        refill = torch.randperm(n, device=points.device)[: int(empty.sum().item())]
        lo[empty] = points[refill]
        hi[empty] = points[refill]

    return lo, hi


def _box_extension_cost(
    points: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    dim_weights: torch.Tensor,
    ext_scale: float,
) -> torch.Tensor:
    mid = (lo + hi) / 2
    over = (points[:, None, :] - hi[None, :, :]).clamp_min(0)
    under = (lo[None, :, :] - points[:, None, :]).clamp_min(0)
    extension = over + under
    interior = (points[:, None, :] - mid[None, :, :]).abs()
    return ext_scale * (extension * dim_weights).sum(dim=-1) + 0.05 * (interior * dim_weights).sum(dim=-1)


def _unbalanced_refine(
    points: torch.Tensor,
    assign: torch.Tensor,
    k: int,
    n_iter: int,
    dim_weights: torch.Tensor,
    ext_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = recompute_centers(points, assign, k)
    for _ in range(n_iter):
        lo, hi = _box_stats(points, assign, k)
        cost = _box_extension_cost(points, lo, hi, dim_weights=dim_weights, ext_scale=ext_scale)
        next_assign = cost.argmin(dim=1)
        next_centers = recompute_centers(points, next_assign, k)
        if torch.equal(next_assign, assign):
            assign = next_assign
            centers = next_centers
            break
        assign = next_assign
        centers = next_centers
    return assign, centers


def _balanced_refine(
    points: torch.Tensor,
    assign: torch.Tensor,
    target_sizes: torch.Tensor,
    n_iter: int,
    dim_weights: torch.Tensor,
    ext_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = int(target_sizes.numel())
    centers = recompute_centers(points, assign, k)
    for _ in range(n_iter):
        lo, hi = _box_stats(points, assign, k)
        cost = _box_extension_cost(points, lo, hi, dim_weights=dim_weights, ext_scale=ext_scale)
        next_assign = balanced_assign_from_cost(cost, target_sizes)
        next_centers = recompute_centers(points, next_assign, k)
        if torch.equal(next_assign, assign):
            assign = next_assign
            centers = next_centers
            break
        assign = next_assign
        centers = next_centers
    return assign, centers


def _dim_weights(points: torch.Tensor, weight_power: float) -> torch.Tensor:
    scale = points.std(dim=0).clamp_min(1e-4).pow(weight_power)
    return scale / scale.mean().clamp_min(1e-12)


def cluster_pq_span_refine(
    keys: torch.Tensor,
    bf: int,
    refine_iter: int = 6,
    weight_power: float = 1.0,
    ext_scale: float = 8.0,
):
    """
    Start from PQ-subspace groups, then reassign by weighted box-extension cost.

    This keeps PQ's strong subspace initialization but optimizes the quantity
    that actually matters to AABB gates: how much a point enlarges a box.
    """
    assign, _ = cluster_pq_subspace(keys, bf)
    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    centers = torch.empty(h, k, d, dtype=keys.dtype, device=keys.device)

    for head in range(h):
        weights = _dim_weights(keys[head], weight_power=weight_power)
        assign_h, centers_h = _unbalanced_refine(
            keys[head],
            assign[head],
            k,
            n_iter=refine_iter,
            dim_weights=weights,
            ext_scale=ext_scale,
        )
        assign[head] = assign_h
        centers[head] = centers_h

    return assign, centers


def cluster_pq_balanced_span(
    keys: torch.Tensor,
    bf: int,
    refine_iter: int = 6,
    weight_power: float = 1.0,
    ext_scale: float = 8.0,
):
    """PQ-subspace initialization plus exact-capacity box-extension refinement."""
    assign, _ = cluster_pq_subspace(keys, bf)
    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    target_sizes = target_cluster_sizes(n, bf, keys.device)
    centers = torch.empty(h, k, d, dtype=keys.dtype, device=keys.device)

    for head in range(h):
        weights = _dim_weights(keys[head], weight_power=weight_power)
        assign_h, centers_h = _balanced_refine(
            keys[head],
            assign[head],
            target_sizes,
            n_iter=refine_iter,
            dim_weights=weights,
            ext_scale=ext_scale,
        )
        assign[head] = assign_h
        centers[head] = centers_h

    return assign, centers
