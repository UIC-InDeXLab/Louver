"""Hybrid enclosure: intersection of ball, AABB, and cone bounds."""

from __future__ import annotations

import math

import torch


def enclose_hybrid(keys, assign, centers, K, bf):
    """
    Combines ball-centroid, AABB, and cone bounds.
    A cluster passes the gate ONLY if ALL three bounds exceed the threshold.
    Strictly tighter than any individual method at ~4.5x ball-gate cost.

    Returns:
        gate: callable(q, th) -> (H, K) bool.
        info: dict with diagnostic stats.
    """
    H, N, D = keys.shape
    device = keys.device
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    # ── Ball component ──
    parent_for_child = centers.gather(1, idx_exp)
    dists = (keys - parent_for_child).norm(dim=-1)  # (H, N)
    ball_radii = torch.full((H, K), 0.0, device=device)
    ball_radii.scatter_reduce_(1, assign, dists, reduce="amax", include_self=True)

    # ── AABB component ──
    lo = torch.full((H, K, D), float("inf"), device=device)
    hi = torch.full((H, K, D), float("-inf"), device=device)
    lo.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)
    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    # ── Cone component ──
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
    max_norm.scatter_reduce_(
        1, assign, key_norms.squeeze(-1), reduce="amax", include_self=True
    )

    def gate(q, th):
        th_exp = th.unsqueeze(-1)  # (H, 1)

        # Ball check: q*center + radius > th
        ball_scores = torch.einsum("hkd,hd->hk", centers, q)
        ball_pass = (ball_scores + ball_radii) > th_exp

        # AABB check: sum_d max(q_d*lo_d, q_d*hi_d) > th
        q_exp = q.unsqueeze(1)  # (H, 1, D)
        aabb_scores = torch.maximum(q_exp * lo, q_exp * hi).sum(dim=-1)
        aabb_pass = aabb_scores > th_exp

        # Cone check: max_norm * cos(effective_angle) > th
        q_dot_dir = torch.einsum("hkd,hd->hk", cone_dir, q).clamp(-1, 1)
        angle_q_dir = torch.acos(q_dot_dir)
        effective_angle = (angle_q_dir - half_angles).clamp_min(0)
        cone_upper = max_norm * torch.cos(effective_angle)
        cone_pass = cone_upper > th_exp

        return ball_pass & aabb_pass & cone_pass

    return gate, {
        "ball_r_mean": float(ball_radii.mean()),
        "cone_ha_deg": float(half_angles.mean() * 180 / math.pi),
    }
