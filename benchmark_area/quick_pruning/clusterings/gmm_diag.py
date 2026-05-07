"""Diagonal-covariance GMM clustering via hard-EM."""

from __future__ import annotations

import math

import torch

from .kmeans import cluster_kmeans


def cluster_gmm_diag(keys: torch.Tensor, bf: int, max_iter: int = 20):
    """
    Gaussian mixture model with diagonal covariances.
    Uses hard-EM (argmax responsibilities), initialised from k-means.

    Returns:
        assign: (H, N) cluster assignments.
        centers: (H, K, D) cluster means.
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    # Initialize from k-means
    _, means = cluster_kmeans(keys, bf, max_iter=5)

    variances = torch.ones(H, K, D, device=device)
    weights = torch.ones(H, K, device=device) / K

    for _ in range(max_iter):
        # E-step
        diff = keys.unsqueeze(2) - means.unsqueeze(1)  # (H, N, K, D)
        var_exp = variances.unsqueeze(1).clamp_min(1e-8)  # (H, 1, K, D)
        log_prob = -0.5 * ((diff * diff / var_exp) + var_exp.log()).sum(dim=-1)  # (H, N, K)
        log_resp = log_prob + weights.log().unsqueeze(1)  # (H, N, K)

        assign = log_resp.argmax(dim=2)  # (H, N)

        # M-step
        idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
        new_means = torch.zeros(H, K, D, device=device)
        new_var_sum = torch.zeros(H, K, D, device=device)
        counts = torch.zeros(H, K, device=device)

        new_means.scatter_add_(1, idx_exp, keys)
        counts.scatter_add_(1, assign, torch.ones(H, N, device=device))

        empty = counts == 0
        if empty.any():
            for h in range(H):
                ek = empty[h].nonzero(as_tuple=False).view(-1)
                if ek.numel() > 0:
                    far_idx = torch.randperm(N, device=device)[: ek.numel()]
                    new_means[h, ek] = keys[h, far_idx]
                    counts[h, ek] = 1

        mask = counts > 0
        new_means[mask] /= counts[mask].unsqueeze(-1)
        means = new_means

        centered = keys - means.gather(1, idx_exp)
        sq = centered * centered
        new_var_sum.scatter_add_(1, idx_exp, sq)
        variances = torch.ones(H, K, D, device=device)
        variances[mask] = (new_var_sum[mask] / counts[mask].unsqueeze(-1)).clamp_min(1e-8)

        weights = (counts / N).clamp_min(1e-8)

    # Final assignment
    diff = keys.unsqueeze(2) - means.unsqueeze(1)
    var_exp = variances.unsqueeze(1).clamp_min(1e-8)
    log_prob = -0.5 * ((diff * diff / var_exp) + var_exp.log()).sum(dim=-1)
    assign = log_prob.argmax(dim=2)

    return assign, means
