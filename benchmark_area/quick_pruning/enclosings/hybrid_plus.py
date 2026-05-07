"""Extended hybrid: 5-way intersection of ball, AABB, cone, ellipsoid, centerline."""

from __future__ import annotations

import math

import torch


def enclose_hybrid_plus(keys, assign, centers, K, bf):
    """
    Intersects five independent upper bounds:
      ball ∩ AABB ∩ cone ∩ ellipsoid ∩ centerline

    Each bound captures different geometric structure; their intersection
    is strictly tighter than any subset. Gate cost is ~10x ball but pruning
    should be substantially better than the 3-way hybrid.

    Returns:
        gate: callable(q, th) -> (H, K) bool.
        info: dict with diagnostic stats.
    """
    H, N, D = keys.shape
    device = keys.device
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
    parent_for_child = centers.gather(1, idx_exp)
    diff = keys - parent_for_child

    # ── 1. Ball ──
    eucl_dists = diff.norm(dim=-1)  # (H, N)
    ball_radii = torch.full((H, K), 0.0, device=device)
    ball_radii.scatter_reduce_(1, assign, eucl_dists, reduce="amax", include_self=True)

    # ── 2. AABB ──
    lo = torch.full((H, K, D), float("inf"), device=device)
    hi = torch.full((H, K, D), float("-inf"), device=device)
    lo.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)
    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    # ── 3. Cone ──
    key_norms = keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    keys_normed = keys / key_norms
    cone_dir = torch.zeros(H, K, D, device=device)
    cone_dir.scatter_add_(1, idx_exp, keys_normed)
    cone_dir = cone_dir / cone_dir.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    dir_for_child = cone_dir.gather(1, idx_exp)
    cos_angles = (keys_normed * dir_for_child).sum(dim=-1).clamp(-1, 1)
    min_cos = torch.full((H, K), 1.0, device=device)
    min_cos.scatter_reduce_(1, assign, cos_angles, reduce="amin", include_self=True)
    half_angles = torch.acos(min_cos.clamp(-1, 1))

    max_norm = torch.full((H, K), 0.0, device=device)
    max_norm.scatter_reduce_(1, assign, key_norms.squeeze(-1), reduce="amax", include_self=True)

    # ── 4. Ellipsoid (diagonal) ──
    abs_diff = diff.abs()
    sigma = torch.full((H, K, D), 0.0, device=device)
    sigma.scatter_reduce_(1, idx_exp, abs_diff, reduce="amax", include_self=True)
    sigma = sigma.clamp_min(1e-12)

    sigma_for_child = sigma.gather(1, idx_exp)
    mahal = ((diff / sigma_for_child) ** 2).sum(dim=-1).sqrt()
    mahal_radii = torch.full((H, K), 0.0, device=device)
    mahal_radii.scatter_reduce_(1, assign, mahal, reduce="amax", include_self=True)

    # ── 5. Centerline ──
    axis = centers / centers.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis_child = axis.gather(1, idx_exp)
    coeff = (diff * axis_child).sum(dim=-1)

    alpha = torch.zeros(H, K, device=device, dtype=keys.dtype)
    alpha.scatter_reduce_(1, assign, coeff.abs(), reduce="amax", include_self=True)

    cl_residual = torch.zeros(H, K, device=device, dtype=keys.dtype)
    cl_residual.scatter_reduce_(
        1, assign,
        (diff - coeff.unsqueeze(-1) * axis_child).norm(dim=-1),
        reduce="amax", include_self=True,
    )

    def gate(q, th):
        th_exp = th.unsqueeze(-1)
        center_scores = torch.einsum("hkd,hd->hk", centers, q)

        # 1. Ball
        ball_pass = (center_scores + ball_radii) > th_exp

        # 2. AABB
        q_exp = q.unsqueeze(1)
        aabb_scores = torch.maximum(q_exp * lo, q_exp * hi).sum(dim=-1)
        aabb_pass = aabb_scores > th_exp

        # 3. Cone
        q_dot_dir = torch.einsum("hkd,hd->hk", cone_dir, q).clamp(-1, 1)
        angle_q_dir = torch.acos(q_dot_dir)
        effective_angle = (angle_q_dir - half_angles).clamp_min(0)
        cone_upper = max_norm * torch.cos(effective_angle)
        cone_pass = cone_upper > th_exp

        # 4. Ellipsoid
        q_scaled = q.unsqueeze(1) * sigma
        ell_slack = mahal_radii * q_scaled.norm(dim=-1)
        ell_upper = center_scores + torch.min(ell_slack, ball_radii)
        ell_pass = ell_upper > th_exp

        # 5. Centerline
        proj = torch.einsum("hkd,hd->hk", axis, q)
        q_norm_sq = q.square().sum(dim=-1, keepdim=True)
        cl_upper = center_scores + alpha * proj.abs()
        cl_upper = cl_upper + cl_residual * (q_norm_sq - proj.square()).clamp_min(0).sqrt()
        cl_pass = cl_upper > th_exp

        return ball_pass & aabb_pass & cone_pass & ell_pass & cl_pass

    return gate, {
        "ball_r_mean": float(ball_radii.mean()),
        "cone_ha_deg": float(half_angles.mean() * 180 / math.pi),
        "mahal_r_mean": float(mahal_radii.mean()),
    }
