"""Low-dimensional chunking methods tuned for AABB-style enclosures."""

from __future__ import annotations

import math

import torch

from ._balanced_utils import (
    balanced_assign_from_cost,
    recompute_centers,
    target_cluster_sizes,
)


def _build_chunk_assign(order: torch.Tensor, chunk_sizes: torch.Tensor) -> torch.Tensor:
    n = int(order.numel())
    assign_sorted = torch.empty(n, dtype=torch.long, device=order.device)
    start = 0
    for cid, size in enumerate(chunk_sizes.tolist()):
        end = start + size
        assign_sorted[start:end] = cid
        start = end

    assign = torch.empty_like(assign_sorted)
    assign[order] = assign_sorted
    return assign


def _morton_codes(coords: torch.Tensor, bits: int = 10) -> torch.Tensor:
    q = coords.shape[1]
    lo = coords.min(dim=0, keepdim=True).values
    hi = coords.max(dim=0, keepdim=True).values
    span = (hi - lo).clamp_min(1e-12)
    quant = ((coords - lo) / span * ((1 << bits) - 1)).long().clamp_(0, (1 << bits) - 1)

    codes = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
    for bit in range(bits):
        for dim in range(q):
            codes |= ((quant[:, dim] >> bit) & 1) << (bit * q + dim)
    return codes


def _box_cost(
    points: torch.Tensor,
    assign: torch.Tensor,
    k: int,
    dim_weights: torch.Tensor,
    ext_scale: float,
) -> torch.Tensor:
    n, d = points.shape
    device = points.device

    idx_exp = assign.unsqueeze(-1).expand(-1, d)
    lo = torch.full((k, d), float("inf"), device=device, dtype=points.dtype)
    hi = torch.full((k, d), float("-inf"), device=device, dtype=points.dtype)
    lo.scatter_reduce_(0, idx_exp, points, reduce="amin", include_self=False)
    hi.scatter_reduce_(0, idx_exp, points, reduce="amax", include_self=False)

    empty = lo[:, 0].isinf()
    if empty.any():
        refill = torch.randperm(n, device=device)[: int(empty.sum().item())]
        lo[empty] = points[refill]
        hi[empty] = points[refill]

    mid = (lo + hi) / 2
    over = (points[:, None, :] - hi[None, :, :]).clamp_min(0)
    under = (lo[None, :, :] - points[:, None, :]).clamp_min(0)
    extension = over + under
    interior = (points[:, None, :] - mid[None, :, :]).abs()

    # Heavy penalty for expanding a box, small tie-break toward its middle.
    return ext_scale * (extension * dim_weights).sum(dim=-1) + 0.05 * (interior * dim_weights).sum(dim=-1)


def _balanced_span_refine(
    points: torch.Tensor,
    assign: torch.Tensor,
    target_sizes: torch.Tensor,
    n_iter: int,
    weight_power: float,
    ext_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = int(target_sizes.numel())
    dim_scale = points.std(dim=0).clamp_min(1e-4)
    dim_weights = dim_scale.pow(weight_power)
    dim_weights = dim_weights / dim_weights.mean().clamp_min(1e-12)

    centers = recompute_centers(points, assign, k)
    for _ in range(n_iter):
        cost = _box_cost(points, assign, k, dim_weights=dim_weights, ext_scale=ext_scale)
        assign_next = balanced_assign_from_cost(cost, target_sizes)
        centers_next = recompute_centers(points, assign_next, k)
        if torch.equal(assign_next, assign):
            assign = assign_next
            centers = centers_next
            break
        assign = assign_next
        centers = centers_next

    return assign, centers


def _pca_project(keys: torch.Tensor, rank: int) -> torch.Tensor:
    centered = keys - keys.mean(dim=1, keepdim=True)
    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    basis = vt[:, :rank, :].transpose(-2, -1).contiguous()
    return torch.bmm(centered, basis)


def cluster_pca_axis_chunk(
    keys: torch.Tensor,
    bf: int,
    rank: int = 2,
    refine_iter: int = 4,
    weight_power: float = 1.0,
    ext_scale: float = 8.0,
):
    """
    Sort by the leading PCA axis, chunk contiguously, then refine for AABB span.

    Higher-bf AABB pruning benefits from clusters that are contiguous slabs in a
    low-dimensional projection instead of diffuse Voronoi cells in L2.
    """
    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    target_sizes = target_cluster_sizes(n, bf, keys.device)
    rank = min(max(1, rank), d, n)
    projected = _pca_project(keys, rank)

    assign = torch.empty(h, n, dtype=torch.long, device=keys.device)
    centers = torch.empty(h, k, d, dtype=keys.dtype, device=keys.device)
    for head in range(h):
        order = torch.argsort(projected[head, :, 0])
        assign_h = _build_chunk_assign(order, target_sizes)
        assign_h, centers_h = _balanced_span_refine(
            keys[head],
            assign_h,
            target_sizes,
            n_iter=refine_iter,
            weight_power=weight_power,
            ext_scale=ext_scale,
        )
        assign[head] = assign_h
        centers[head] = centers_h

    return assign, centers


def cluster_pca_morton_span(
    keys: torch.Tensor,
    bf: int,
    rank: int = 3,
    refine_iter: int = 4,
    weight_power: float = 1.0,
    ext_scale: float = 8.0,
):
    """
    Morton-order chunking in a PCA subspace, then balanced span refinement.

    This keeps nearby points together in multiple projected directions, which is
    often a better proxy for non-overlapping AABBs than centroidal clustering.
    """
    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    target_sizes = target_cluster_sizes(n, bf, keys.device)
    rank = min(max(2, rank), d, n)
    projected = _pca_project(keys, rank)

    assign = torch.empty(h, n, dtype=torch.long, device=keys.device)
    centers = torch.empty(h, k, d, dtype=keys.dtype, device=keys.device)
    for head in range(h):
        codes = _morton_codes(projected[head, :, :rank])
        order = torch.argsort(codes)
        assign_h = _build_chunk_assign(order, target_sizes)
        assign_h, centers_h = _balanced_span_refine(
            keys[head],
            assign_h,
            target_sizes,
            n_iter=refine_iter,
            weight_power=weight_power,
            ext_scale=ext_scale,
        )
        assign[head] = assign_h
        centers[head] = centers_h

    return assign, centers
