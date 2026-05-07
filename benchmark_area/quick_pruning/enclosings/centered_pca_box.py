"""Centered PCA projection box: subtract global mean first.

Key insight: keys x have large norms (~11). Global mean μ has norm ~10.
After centering: x' = x - μ has norm ~5.  PCA on centered data captures
much more of the remaining variance, making the residual small (~2-3).

Gate decomposition:
  q·x = q·μ + q·x'

The q·μ term is shared (1 dp amortized across K clusters).
The q·x' term is bounded via d-dim PCA interval box + residual ball.

Gate cost per cluster ≈ (3d+2)/(2D) dp-equiv ≈ 0.1 for d=8.
Shared cost: d+1 dot products (amortized).

Total effective g ≈ 0.1 + (d+1)/K ≈ 0.1 for large K.
"""

from __future__ import annotations

import torch


def _make_centered_pca_box(rank: int):
    def enclose(keys, assign, centers, K, bf):
        return enclose_centered_pca_box(keys, assign, centers, K, bf, rank=rank)
    enclose.__name__ = f"enclose_centered_pca_d{rank}"
    return enclose


def enclose_centered_pca_box(keys, assign, centers, K, bf, rank: int = 8):
    H, N, D = keys.shape
    device = keys.device
    d = min(max(1, rank), D)

    # Global mean per head
    mu = keys.mean(dim=1, keepdim=True)  # (H, 1, D)
    centered = keys - mu  # (H, N, D)

    # PCA on centered data
    _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
    U = Vt[:, :d, :]  # (H, d, D)

    # Project centered keys onto PCA
    proj = torch.bmm(centered, U.transpose(-2, -1))  # (H, N, d)

    # Per-cluster intervals on PCA projections
    idx_proj = assign.unsqueeze(-1).expand(-1, -1, d)
    lo = torch.full((H, K, d), float("inf"), device=device, dtype=keys.dtype)
    hi = torch.full((H, K, d), float("-inf"), device=device, dtype=keys.dtype)
    lo.scatter_reduce_(1, idx_proj, proj, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_proj, proj, reduce="amax", include_self=False)
    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    # Residual norm: ||x' - U U^T x'|| per key
    recon = torch.bmm(proj, U)  # (H, N, D)
    resid_norm = (centered - recon).norm(dim=-1)  # (H, N)

    # Per-cluster max residual
    res_max = torch.zeros(H, K, device=device, dtype=keys.dtype)
    res_max.scatter_reduce_(1, assign, resid_norm, reduce="amax", include_self=True)

    mu_squeezed = mu.squeeze(1)  # (H, D)

    def gate(q, th):
        # Shared: q·μ (1 dp amortized)
        q_mu = (q * mu_squeezed).sum(dim=-1)  # (H,)

        # Shared: project q onto PCA (d dot products amortized)
        p = torch.bmm(q.unsqueeze(1), U.transpose(-2, -1)).squeeze(1)  # (H, d)

        # ||q_perp|| in PCA-orthogonal space
        q_perp_norm = (q.square().sum(-1) - p.square().sum(-1)).clamp_min(0).sqrt()  # (H,)

        # Per cluster: PCA interval upper bound
        p_exp = p.unsqueeze(1)  # (H, 1, d)
        pca_upper = torch.maximum(p_exp * lo, p_exp * hi).sum(dim=-1)  # (H, K)

        # Total upper bound
        upper = q_mu.unsqueeze(-1) + pca_upper + res_max * q_perp_norm.unsqueeze(-1)

        return upper > th.unsqueeze(-1)

    # Diagnostics
    centered_norm = centered.norm(dim=-1)
    var_explained = proj.square().sum(-1).mean() / centered.square().sum(-1).mean()

    return gate, {
        "rank": d,
        "res_max_mean": float(res_max.mean()),
        "res_max_max": float(res_max.max()),
        "centered_norm_mean": float(centered_norm.mean()),
        "mu_norm_mean": float(mu_squeezed.norm(dim=-1).mean()),
        "var_explained": float(var_explained),
        "g_per_cluster": round(3 * d / (2 * D), 4),
    }


enclose_centered_pca_d4 = _make_centered_pca_box(4)
enclose_centered_pca_d8 = _make_centered_pca_box(8)
enclose_centered_pca_d16 = _make_centered_pca_box(16)
enclose_centered_pca_d32 = _make_centered_pca_box(32)
enclose_centered_pca_d64 = _make_centered_pca_box(64)
