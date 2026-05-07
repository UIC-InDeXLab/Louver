"""Cone enclosure for (approximately) normalized keys."""

from __future__ import annotations

import math

import torch


def enclose_cone(keys, assign, centers, K, bf):
    """
    Angular cone per cluster.  Direction = mean direction of normalised
    children; half-angle = max angle between direction and any child.

    Returns:
        gate: callable(q, th) -> (H, K) bool.
        info: dict with diagnostic stats.
    """
    H, N, D = keys.shape
    device = keys.device

    key_norms = keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    keys_normed = keys / key_norms

    cone_dir = torch.zeros(H, K, D, device=device)
    cone_dir.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys_normed)
    cone_dir = cone_dir / cone_dir.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    dir_for_child = cone_dir.gather(1, assign.unsqueeze(-1).expand(-1, -1, D))
    cos_angles = (keys_normed * dir_for_child).sum(dim=-1).clamp(-1, 1)  # (H, N)

    min_cos = torch.full((H, K), 1.0, device=device)
    min_cos.scatter_reduce_(1, assign, cos_angles, reduce="amin", include_self=True)
    half_angles = torch.acos(min_cos.clamp(-1, 1))  # (H, K)

    max_norm = torch.full((H, K), 0.0, device=device)
    max_norm.scatter_reduce_(1, assign, key_norms.squeeze(-1), reduce="amax", include_self=True)

    def gate(q, th):
        q_dot_dir = torch.einsum("hkd,hd->hk", cone_dir, q).clamp(-1, 1)
        angle_q_dir = torch.acos(q_dot_dir)  # (H, K)
        effective_angle = (angle_q_dir - half_angles).clamp_min(0)
        upper_bound = max_norm * torch.cos(effective_angle)
        return upper_bound > th.unsqueeze(-1)

    return gate, {
        "half_angle_mean_deg": float(half_angles.mean() * 180 / math.pi),
        "half_angle_max_deg": float(half_angles.max() * 180 / math.pi),
    }
