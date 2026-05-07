"""PCA-aligned oriented bounding box enclosure."""

from __future__ import annotations

import torch


def enclose_pca_obb(keys, assign, centers, K, bf):
    """
    Oriented bounding box via global PCA rotation.

    Computes PCA of the full key set (per head), rotates keys into PCA space,
    then builds axis-aligned bounding boxes in that space.  Because attention
    key dimensions are highly correlated, the PCA-aligned box is much tighter
    than an axis-aligned one.

    Cost: one SVD per head (small: N x D with D ~ 128) + same as AABB.

    Returns:
        gate: callable(q, th) -> (H, K) bool.
        info: dict with diagnostic stats.
    """
    H, N, D = keys.shape
    device = keys.device

    # ── Global PCA per head ──
    # Center the keys for PCA (but keep original for dot products)
    key_mean = keys.mean(dim=1, keepdim=True)  # (H, 1, D)
    keys_centered = keys - key_mean

    # SVD to get rotation matrix: V columns are principal directions
    # Use the smaller of N, D for efficiency
    # keys_centered: (H, N, D) -> we want eigenvectors of (D, D) covariance
    # torch.linalg.svd on (H, N, D) gives V of shape (H, D, D)
    _, _, Vt = torch.linalg.svd(keys_centered, full_matrices=False)  # Vt: (H, D, D) or (H, min(N,D), D)
    V = Vt.transpose(-2, -1)[:, :, :D]  # (H, D, D) — rotation matrix

    # ── Rotate keys and centers into PCA space ──
    keys_rot = torch.bmm(keys, V)  # (H, N, D)
    centers_rot = torch.bmm(centers, V)  # (H, K, D)

    # ── Build AABB in PCA space ──
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    lo = torch.full((H, K, D), float("inf"), device=device)
    hi = torch.full((H, K, D), float("-inf"), device=device)
    lo.scatter_reduce_(1, idx_exp, keys_rot, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_exp, keys_rot, reduce="amax", include_self=False)

    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    def gate(q, th):
        # Rotate query into PCA space
        q_rot = torch.bmm(q.unsqueeze(1), V).squeeze(1)  # (H, D)
        q_exp = q_rot.unsqueeze(1)  # (H, 1, D)
        max_dot = torch.maximum(q_exp * lo, q_exp * hi).sum(dim=-1)  # (H, K)
        return max_dot > th.unsqueeze(-1)

    vol = (hi - lo).clamp_min(0).prod(dim=-1)
    return gate, {"vol_mean": float(vol.mean()), "vol_max": float(vol.max())}
