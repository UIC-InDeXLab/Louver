"""Ultra-cheap gate via global PCA projections + residual ball.

Key insight: avoid per-cluster dot products entirely.  Instead, project
the query onto d global PCA directions (shared cost, amortized over K
clusters), then use per-cluster intervals on those projections plus a
residual ball bound.

Gate cost per cluster: ~3d/(2D) dp-equiv.  For d=8, D=128: g ≈ 0.10.
This is 10x cheaper than ball_centroid (g=1.0).
"""

from __future__ import annotations

import torch


def _make_enclose_pca_proj_box(rank: int):
    """Factory that returns an enclosing function with a fixed PCA rank."""

    def enclose(keys, assign, centers, K, bf):
        return enclose_pca_projection_box(keys, assign, centers, K, bf, rank=rank)

    enclose.__name__ = f"enclose_pca_proj_box_d{rank}"
    enclose.__qualname__ = enclose.__name__
    return enclose


def enclose_pca_projection_box(keys, assign, centers, K, bf, rank: int = 8):
    """
    Global PCA box with ultra-cheap per-cluster gating.

    For each cluster, stores intervals [lo, hi] on d global PCA directions
    plus the max residual norm (orthogonal to PCA subspace).

    Gate for query q:
      1. Shared: p = q @ U^T  (d dot products, amortized over K clusters)
      2. Per cluster: UB = sum_j max(p_j * lo_kj, p_j * hi_kj) + res_k * ||q_perp||
      3. Pass if UB > threshold

    Returns:
        gate: callable(q, th) -> (H, K) bool.
        info: dict with diagnostic stats.
    """
    H, N, D = keys.shape
    device = keys.device
    d = min(max(1, rank), D)

    # Global PCA: top-d directions per head
    key_mean = keys.mean(dim=1, keepdim=True)  # (H, 1, D)
    centered = keys - key_mean
    # SVD on (H, N, D) — we only need top-d right singular vectors
    _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
    U = Vt[:, :d, :]  # (H, d, D) — top-d PCA directions

    # Project all keys onto PCA directions: (H, N, d)
    key_proj = torch.bmm(keys, U.transpose(-2, -1))  # (H, N, d)

    # Per-cluster intervals on each PCA direction
    idx_proj = assign.unsqueeze(-1).expand(-1, -1, d)  # (H, N, d)
    lo = torch.full((H, K, d), float("inf"), device=device, dtype=keys.dtype)
    hi = torch.full((H, K, d), float("-inf"), device=device, dtype=keys.dtype)
    lo.scatter_reduce_(1, idx_proj, key_proj, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_proj, key_proj, reduce="amax", include_self=False)

    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    # Residual norm: ||x - U U^T x|| for each key
    key_proj_back = torch.bmm(key_proj, U)  # (H, N, D) — reconstruction
    residual_norm = (keys - key_proj_back).norm(dim=-1)  # (H, N)

    # Per-cluster max residual norm
    res_max = torch.zeros(H, K, device=device, dtype=keys.dtype)
    res_max.scatter_reduce_(1, assign, residual_norm, reduce="amax", include_self=True)

    def gate(q, th):
        # Shared computation: project query onto PCA directions
        # q: (H, D), U: (H, d, D)
        p = torch.bmm(q.unsqueeze(1), U.transpose(-2, -1)).squeeze(1)  # (H, d)
        q_perp_norm = (q.square().sum(dim=-1) - p.square().sum(dim=-1)).clamp_min(0).sqrt()  # (H,)

        # Per-cluster upper bound
        p_exp = p.unsqueeze(1)  # (H, 1, d)
        proj_upper = torch.maximum(p_exp * lo, p_exp * hi).sum(dim=-1)  # (H, K)
        upper = proj_upper + res_max * q_perp_norm.unsqueeze(-1)  # (H, K)

        return upper > th.unsqueeze(-1)

    # Compute explained variance ratio for diagnostics
    key_var = keys.square().sum(dim=-1).mean(dim=-1)  # (H,)
    proj_var = key_proj.square().sum(dim=-1).mean(dim=-1)  # (H,)  -- not exact but informative
    explained = (proj_var / key_var.clamp_min(1e-12)).mean().item()

    return gate, {
        "rank": d,
        "res_max_mean": float(res_max.mean()),
        "res_max_max": float(res_max.max()),
        "explained_var_approx": explained,
        "g_theory": round(3 * d / (2 * D), 4),
    }


# Pre-built variants for the benchmark registry
enclose_pca_proj_d4 = _make_enclose_pca_proj_box(4)
enclose_pca_proj_d8 = _make_enclose_pca_proj_box(8)
enclose_pca_proj_d16 = _make_enclose_pca_proj_box(16)
enclose_pca_proj_d32 = _make_enclose_pca_proj_box(32)
