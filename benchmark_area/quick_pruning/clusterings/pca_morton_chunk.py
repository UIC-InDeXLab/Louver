"""Balanced chunking after PCA projection and Morton ordering."""

from __future__ import annotations

import math

import torch

from ._balanced_utils import balanced_refine, target_cluster_sizes


def _build_chunk_assign(sorted_idx: torch.Tensor, chunk_sizes: torch.Tensor) -> torch.Tensor:
    n = int(sorted_idx.numel())
    device = sorted_idx.device
    assign_sorted = torch.empty(n, dtype=torch.long, device=device)
    start = 0
    for cluster_id, size in enumerate(chunk_sizes.tolist()):
        end = start + size
        assign_sorted[start:end] = cluster_id
        start = end
    assign = torch.empty_like(assign_sorted)
    assign[sorted_idx] = assign_sorted
    return assign


def _morton_codes(coords: torch.Tensor, bits: int = 10) -> torch.Tensor:
    """Interleave bits of 3 quantized coordinates."""
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


def cluster_pca_morton_chunk(keys: torch.Tensor, bf: int, rank: int = 3, refine_iter: int = 0):
    """
    Fast space-filling-curve partitioning.

    Keys are projected to a small PCA subspace, ordered by a Morton code in
    that subspace, then chunked into exact-capacity groups.
    """
    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    device = keys.device
    target_sizes = target_cluster_sizes(n, bf, device)

    rank = min(max(2, rank), d, n)
    centered = keys - keys.mean(dim=1, keepdim=True)
    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    basis = vt[:, :rank, :].transpose(-2, -1).contiguous()
    projected = torch.bmm(centered, basis)

    assign = torch.empty(h, n, dtype=torch.long, device=device)
    centers = torch.empty(h, k, d, dtype=keys.dtype, device=device)

    for head in range(h):
        codes = _morton_codes(projected[head])
        sorted_idx = torch.argsort(codes)
        assign_h = _build_chunk_assign(sorted_idx, target_sizes)
        assign_h, centers_h = balanced_refine(keys[head], assign_h, target_sizes, refine_iter)
        assign[head] = assign_h
        centers[head] = centers_h

    return assign, centers
