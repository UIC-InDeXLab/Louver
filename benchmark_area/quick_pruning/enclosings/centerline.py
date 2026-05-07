"""Rank-1 enclosure aligned to each cluster centroid direction."""

from __future__ import annotations

import torch


def enclose_centerline(keys, assign, centers, K, bf):
    """
    Use the centroid direction as a cheap per-cluster axis and bound the
    remaining orthogonal energy with a residual ball.

    This is a rank-1 analogue of subspace_box with no per-cluster SVD.
    """
    H, N, D = keys.shape
    device = keys.device
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    axis = centers / centers.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis_child = axis.gather(1, idx_exp)

    parent = centers.gather(1, idx_exp)
    diff = keys - parent
    coeff = (diff * axis_child).sum(dim=-1)

    alpha = torch.zeros(H, K, device=device, dtype=keys.dtype)
    alpha.scatter_reduce_(1, assign, coeff.abs(), reduce="amax", include_self=True)

    residual = torch.zeros(H, K, device=device, dtype=keys.dtype)
    residual.scatter_reduce_(
        1,
        assign,
        (diff - coeff.unsqueeze(-1) * axis_child).norm(dim=-1),
        reduce="amax",
        include_self=True,
    )

    ball_radii = torch.zeros(H, K, device=device, dtype=keys.dtype)
    ball_radii.scatter_reduce_(
        1,
        assign,
        diff.norm(dim=-1),
        reduce="amax",
        include_self=True,
    )

    def gate(q, th):
        q_norm_sq = q.square().sum(dim=-1, keepdim=True)
        center_scores = torch.einsum("hkd,hd->hk", centers, q)
        proj = torch.einsum("hkd,hd->hk", axis, q)
        line_upper = center_scores + alpha * proj.abs()
        line_upper = line_upper + residual * (q_norm_sq - proj.square()).clamp_min(0).sqrt()

        ball_upper = center_scores + ball_radii * q_norm_sq.sqrt()
        return torch.minimum(line_upper, ball_upper) > th.unsqueeze(-1)

    return gate, {
        "alpha_mean": float(alpha.mean()),
        "residual_mean": float(residual.mean()),
        "ball_r_mean": float(ball_radii.mean()),
    }
