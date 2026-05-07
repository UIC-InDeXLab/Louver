"""Rank-2 centerline enclosure with a cheap second residual axis."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def enclose_dual_centerline(keys, assign, centers, K, bf):
    """
    Extend centerline with a second cluster-local axis recovered from signed
    residual accumulation around a pivot dimension.

    The bound remains fully vectorized: two signed axial terms plus one
    orthogonal residual ball, intersected with the centroid ball.
    """
    H, _, D = keys.shape
    device = keys.device
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    parent = centers.gather(1, idx_exp)
    diff = keys - parent

    axis_u = centers / centers.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis_u_child = axis_u.gather(1, idx_exp)
    coeff_u = (diff * axis_u_child).sum(dim=-1)
    resid_u = diff - coeff_u.unsqueeze(-1) * axis_u_child

    max_abs = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    max_abs.scatter_reduce_(1, idx_exp, resid_u.abs(), reduce="amax", include_self=True)
    pivot_dim = max_abs.argmax(dim=-1)
    pivot_child = pivot_dim.gather(1, assign)
    pivot_vals = resid_u.gather(2, pivot_child.unsqueeze(-1)).squeeze(-1)
    pivot_sign = torch.where(
        pivot_vals < 0,
        -torch.ones_like(pivot_vals),
        torch.ones_like(pivot_vals),
    )

    axis_v_raw = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    axis_v_raw.scatter_add_(1, idx_exp, resid_u * pivot_sign.unsqueeze(-1))
    axis_v_raw = axis_v_raw - (axis_v_raw * axis_u).sum(dim=-1, keepdim=True) * axis_u

    fallback = F.one_hot(pivot_dim, num_classes=D).to(device=device, dtype=keys.dtype)
    fallback = fallback - (fallback * axis_u).sum(dim=-1, keepdim=True) * axis_u

    raw_norm = axis_v_raw.norm(dim=-1, keepdim=True)
    fallback_norm = fallback.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis_v = torch.where(
        raw_norm > 1e-6,
        axis_v_raw / raw_norm.clamp_min(1e-12),
        fallback / fallback_norm,
    )

    axis_v_child = axis_v.gather(1, idx_exp)
    coeff_v = (diff * axis_v_child).sum(dim=-1)
    resid_uv = diff - coeff_u.unsqueeze(-1) * axis_u_child - coeff_v.unsqueeze(-1) * axis_v_child

    alpha = torch.zeros(H, K, device=device, dtype=keys.dtype)
    beta = torch.zeros(H, K, device=device, dtype=keys.dtype)
    residual = torch.zeros(H, K, device=device, dtype=keys.dtype)
    ball_radii = torch.zeros(H, K, device=device, dtype=keys.dtype)
    alpha.scatter_reduce_(1, assign, coeff_u.abs(), reduce="amax", include_self=True)
    beta.scatter_reduce_(1, assign, coeff_v.abs(), reduce="amax", include_self=True)
    residual.scatter_reduce_(1, assign, resid_uv.norm(dim=-1), reduce="amax", include_self=True)
    ball_radii.scatter_reduce_(1, assign, diff.norm(dim=-1), reduce="amax", include_self=True)

    def gate(q, th):
        q_norm_sq = q.square().sum(dim=-1, keepdim=True)
        center_scores = torch.einsum("hkd,hd->hk", centers, q)
        proj_u = torch.einsum("hkd,hd->hk", axis_u, q)
        proj_v = torch.einsum("hkd,hd->hk", axis_v, q)
        perp = (q_norm_sq - proj_u.square() - proj_v.square()).clamp_min(0).sqrt()

        plane_upper = center_scores + alpha * proj_u.abs() + beta * proj_v.abs()
        plane_upper = plane_upper + residual * perp

        ball_upper = center_scores + ball_radii * q_norm_sq.sqrt()
        return torch.minimum(plane_upper, ball_upper) > th.unsqueeze(-1)

    return gate, {
        "alpha_mean": float(alpha.mean()),
        "beta_mean": float(beta.mean()),
        "residual_mean": float(residual.mean()),
    }
