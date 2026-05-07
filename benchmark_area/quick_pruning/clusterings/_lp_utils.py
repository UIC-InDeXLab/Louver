"""Shared helpers for Lp-aware clustering methods."""

from __future__ import annotations

import math

import torch


def validate_p(p: float) -> float:
    if math.isinf(p):
        return float("inf")
    p = float(p)
    if p < 1.0:
        raise ValueError(f"L_p clustering requires p >= 1, got {p}")
    return p


def cdist_lp(x: torch.Tensor, y: torch.Tensor, p: float) -> torch.Tensor:
    p = validate_p(p)
    if math.isinf(p):
        return torch.cdist(x, y, p=float("inf"))
    return torch.cdist(x, y, p=p)


def cluster_center_lp(points: torch.Tensor, p: float) -> torch.Tensor:
    p = validate_p(p)
    if points.shape[0] == 0:
        raise ValueError("empty cluster")
    if points.shape[0] == 1:
        return points[0]
    if p == 1.0:
        return points.median(dim=0).values
    if math.isinf(p):
        return 0.5 * (points.amin(dim=0) + points.amax(dim=0))
    return points.mean(dim=0)


def recompute_centers_lp(keys: torch.Tensor, assign: torch.Tensor, k: int, p: float) -> torch.Tensor:
    h, n, d = keys.shape
    centers = torch.empty(h, k, d, dtype=keys.dtype, device=keys.device)
    for head in range(h):
        points = keys[head]
        for cid in range(k):
            idx = (assign[head] == cid).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                centers[head, cid] = points[torch.randint(n, (1,), device=keys.device)]
                continue
            centers[head, cid] = cluster_center_lp(points[idx], p)
    return centers
