"""Centroid dot + PCA residual box: tight bound at g ≈ 1.0 + epsilon.

Decomposes q·x = q·c_k + q·(x - c_k).  The centroid part is exact per
cluster.  The residual part is bounded via a d-dim PCA box plus a ball
for the orthogonal component.  Much tighter than plain ball_centroid
because the PCA box captures directional spread.
"""

from __future__ import annotations

import torch


def _make_centroid_pca(rank: int):
    def enclose(keys, assign, centers, K, bf):
        return enclose_centroid_pca_box(keys, assign, centers, K, bf, rank=rank)

    enclose.__name__ = f"enclose_centroid_pca_d{rank}"
    return enclose


def enclose_centroid_pca_box(keys, assign, centers, K, bf, rank: int = 8):
    """
    Centroid dot product + PCA box on residuals.

    Gate cost: 1.0 (centroid dot) + ~d/D (PCA interval) ≈ 1.0 dp-equiv.
    But pruning is much tighter than plain ball because the residual bound
    is direction-aware.
    """
    H, N, D = keys.shape
    device = keys.device
    d = min(max(1, rank), D)

    # Compute residuals from cluster centers
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
    parent = centers.gather(1, idx_exp)  # (H, N, D)
    residuals = keys - parent  # (H, N, D)

    # Global PCA on residuals
    res_mean = residuals.mean(dim=1, keepdim=True)
    centered_res = residuals - res_mean
    _, _, Vt = torch.linalg.svd(centered_res, full_matrices=False)
    U = Vt[:, :d, :]  # (H, d, D)

    # Project residuals onto PCA
    res_proj = torch.bmm(residuals, U.transpose(-2, -1))  # (H, N, d)

    # Per-cluster intervals
    idx_proj = assign.unsqueeze(-1).expand(-1, -1, d)
    lo = torch.full((H, K, d), float("inf"), device=device, dtype=keys.dtype)
    hi = torch.full((H, K, d), float("-inf"), device=device, dtype=keys.dtype)
    lo.scatter_reduce_(1, idx_proj, res_proj, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_proj, res_proj, reduce="amax", include_self=False)
    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    # Orthogonal residual norm per key
    res_recon = torch.bmm(res_proj, U)
    orth_norm = (residuals - res_recon).norm(dim=-1)  # (H, N)
    orth_max = torch.zeros(H, K, device=device, dtype=keys.dtype)
    orth_max.scatter_reduce_(1, assign, orth_norm, reduce="amax", include_self=True)

    # Also compute ball radius for min(pca_bound, ball_bound)
    ball_r = torch.zeros(H, K, device=device, dtype=keys.dtype)
    ball_r.scatter_reduce_(
        1, assign, residuals.norm(dim=-1), reduce="amax", include_self=True
    )

    def gate(q, th):
        # Centroid dot product
        c_score = torch.einsum("hkd,hd->hk", centers, q)  # (H, K)

        # Project query onto PCA basis
        p = torch.bmm(q.unsqueeze(1), U.transpose(-2, -1)).squeeze(1)  # (H, d)
        q_perp = (q.square().sum(-1) - p.square().sum(-1)).clamp_min(0).sqrt()  # (H,)

        # PCA box upper bound on residual part
        p_exp = p.unsqueeze(1)  # (H, 1, d)
        pca_upper = torch.maximum(p_exp * lo, p_exp * hi).sum(dim=-1)  # (H, K)
        res_upper = pca_upper + orth_max * q_perp.unsqueeze(-1)

        # Ball upper bound on residual part (for tightening)
        q_norm = q.norm(dim=-1, keepdim=True)  # (H, 1)
        ball_upper = ball_r * q_norm

        # Take tighter of PCA box and ball bounds
        upper = c_score + torch.minimum(res_upper, ball_upper)
        return upper > th.unsqueeze(-1)

    return gate, {
        "rank": d,
        "orth_max_mean": float(orth_max.mean()),
        "ball_r_mean": float(ball_r.mean()),
    }


enclose_centroid_pca_d4 = _make_centroid_pca(4)
enclose_centroid_pca_d8 = _make_centroid_pca(8)
enclose_centroid_pca_d16 = _make_centroid_pca(16)
