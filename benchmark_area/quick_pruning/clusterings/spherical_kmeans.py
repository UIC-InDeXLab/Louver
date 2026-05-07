"""Spherical k-means: k-means on L2-normalized keys."""

from __future__ import annotations

import torch

from .kmeans import cluster_kmeans


def cluster_spherical_kmeans(keys: torch.Tensor, bf: int, max_iter: int = 20):
    """
    K-means on L2-normalized keys so that clustering optimises cosine distance.

    Returns:
        assign: (H, N) cluster assignments.
        centers: (H, K, D) unit-normalized centroids.
    """
    norms = keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    keys_normed = keys / norms
    assign, centers = cluster_kmeans(keys_normed, bf, max_iter)
    centers = centers / centers.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return assign, centers
