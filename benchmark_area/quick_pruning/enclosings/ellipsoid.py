"""Diagonal Mahalanobis ellipsoid enclosure."""

from __future__ import annotations

import torch


def enclose_ellipsoid(keys, assign, centers, K, bf):
    """
    Per-cluster ellipsoid with diagonal covariance.

    For each cluster, computes the per-dimension max deviation (sigma_d)
    and the Mahalanobis radius r to contain all points.

    Upper bound: q*mu + r * ||diag(sigma) * q||_2

    This captures anisotropic cluster shapes: dimensions with small spread
    contribute less to the slack, giving tighter bounds than isotropic balls.
    Includes a ball fallback to guarantee it's never worse than ball_centroid.

    Returns:
        gate: callable(q, th) -> (H, K) bool.
        info: dict with diagnostic stats.
    """
    H, N, D = keys.shape
    device = keys.device
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    # Per-cluster per-dimension max absolute deviation from centroid
    parent_for_child = centers.gather(1, idx_exp)  # (H, N, D)
    diff = keys - parent_for_child
    abs_diff = diff.abs()

    sigma = torch.full((H, K, D), 0.0, device=device)
    sigma.scatter_reduce_(1, idx_exp, abs_diff, reduce="amax", include_self=True)
    sigma = sigma.clamp_min(1e-12)

    # Mahalanobis distance for each key: sqrt(sum_d ((k_d - mu_d) / sigma_d)^2)
    sigma_for_child = sigma.gather(1, idx_exp)  # (H, N, D)
    mahal = ((diff / sigma_for_child) ** 2).sum(dim=-1).sqrt()  # (H, N)

    # Per-cluster max Mahalanobis radius
    mahal_radii = torch.full((H, K), 0.0, device=device)
    mahal_radii.scatter_reduce_(1, assign, mahal, reduce="amax", include_self=True)

    # Ball radii for fallback (always valid, sometimes tighter)
    eucl_dists = diff.norm(dim=-1)  # (H, N)
    ball_radii = torch.full((H, K), 0.0, device=device)
    ball_radii.scatter_reduce_(1, assign, eucl_dists, reduce="amax", include_self=True)

    def gate(q, th):
        scores = torch.einsum("hkd,hd->hk", centers, q)  # (H, K)

        # Ellipsoid slack: r_mahal * ||diag(sigma) * q||_2
        q_scaled = q.unsqueeze(1) * sigma  # (H, K, D)
        ellipsoid_slack = mahal_radii * q_scaled.norm(dim=-1)  # (H, K)

        # Take min of ellipsoid and ball slack (ball_slack = ball_radii for unit q)
        slack = torch.min(ellipsoid_slack, ball_radii)

        return (scores + slack) > th.unsqueeze(-1)

    return gate, {
        "mahal_r_mean": float(mahal_radii.mean()),
        "mahal_r_max": float(mahal_radii.max()),
        "ball_r_mean": float(ball_radii.mean()),
    }
