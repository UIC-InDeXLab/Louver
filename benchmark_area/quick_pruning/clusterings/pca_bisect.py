"""Balanced recursive PCA bisection clustering."""

from __future__ import annotations

import math

import torch


def _cluster_score(points: torch.Tensor) -> float:
    if points.shape[0] <= 1:
        return 0.0
    centered = points - points.mean(dim=0, keepdim=True)
    return float(centered.square().sum())


def _dominant_direction(points: torch.Tensor) -> torch.Tensor:
    centered = points - points.mean(dim=0, keepdim=True)
    if centered.shape[0] <= 1 or float(centered.square().sum()) <= 1e-12:
        axis = torch.zeros(points.shape[-1], device=points.device, dtype=points.dtype)
        axis[0] = 1.0
        return axis

    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
    except RuntimeError:
        var = centered.square().mean(dim=0)
        direction = torch.zeros_like(var)
        direction[var.argmax()] = 1.0

    norm = direction.norm().clamp_min(1e-12)
    return direction / norm


def _split_indices(points: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort((points - points.mean(dim=0, keepdim=True)) @ _dominant_direction(points))
    mid = max(1, order.numel() // 2)
    mid = min(mid, order.numel() - 1)
    return indices[order[:mid]], indices[order[mid:]]


def cluster_pca_bisect(keys: torch.Tensor, bf: int):
    """
    Recursively split the loosest cluster along its dominant PCA direction.

    The median split keeps clusters balanced, which matters when the branch
    factor is small and pruning quality is driven by the worst child in each
    parent.
    """
    H, N, D = keys.shape
    device = keys.device
    K = max(1, math.ceil(N / bf))

    assign = torch.zeros(H, N, dtype=torch.long, device=device)
    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)

    for h in range(H):
        all_idx = torch.arange(N, device=device)
        clusters = [all_idx]
        scores = [_cluster_score(keys[h])]

        while len(clusters) < K:
            split_pos = None
            split_score = 0.0
            for i, idx in enumerate(clusters):
                if idx.numel() <= 1:
                    continue
                if scores[i] > split_score:
                    split_pos = i
                    split_score = scores[i]

            if split_pos is None:
                break

            idx = clusters.pop(split_pos)
            scores.pop(split_pos)
            left_idx, right_idx = _split_indices(keys[h, idx], idx)
            clusters.extend([left_idx, right_idx])
            scores.extend([
                _cluster_score(keys[h, left_idx]),
                _cluster_score(keys[h, right_idx]),
            ])

        if len(clusters) != K:
            raise RuntimeError(f"PCA bisection produced {len(clusters)} clusters, expected {K}.")

        for k, idx in enumerate(clusters):
            assign[h, idx] = k
            centers[h, k] = keys[h, idx].mean(dim=0)

    return assign, centers
