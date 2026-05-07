"""Generic Lp-ball enclosure centered at the cluster centroid."""

from __future__ import annotations

import math

import torch


def _validate_p(p: float) -> float:
    if math.isinf(p):
        return float("inf")
    p = float(p)
    if p < 1.0:
        raise ValueError(f"L_p ball requires p >= 1, got {p}")
    return p


def _dual_p(p: float) -> float:
    if p == 1.0:
        return float("inf")
    if math.isinf(p):
        return 1.0
    return p / (p - 1.0)


def _vector_norm(x: torch.Tensor, p: float) -> torch.Tensor:
    if p == 1.0:
        return x.abs().sum(dim=-1)
    if math.isinf(p):
        return x.abs().amax(dim=-1)
    return torch.linalg.vector_norm(x, ord=p, dim=-1)


def enclose_lp_ball(keys, assign, centers, K, bf, p: float = 2.0):
    """
    Per-cluster Lp ball around the centroid.

    The gate uses the support function of the Lp ball:
        max_{||x-c||_p <= r} q·x = q·c + r ||q||_{p*}
    where p* is the dual exponent.

    Returns:
        gate: callable(q, th) -> (H, K) bool mask of clusters that pass.
        info: dict with diagnostic stats.
    """
    p = _validate_p(p)
    dual_p = _dual_p(p)
    H, N, D = keys.shape
    device = keys.device

    parent_for_child = centers.gather(1, assign.unsqueeze(-1).expand(-1, -1, D))
    radii_per_child = _vector_norm(keys - parent_for_child, p)  # (H, N)

    radii_p = torch.full((H, K), 0.0, device=device)
    radii_p.scatter_reduce_(1, assign, radii_per_child, reduce="amax", include_self=True)

    def gate(q, th):
        scores = torch.einsum("hkd,hd->hk", centers, q)  # (H, K)
        q_dual = _vector_norm(q, dual_p).unsqueeze(-1)   # (H, 1)
        return (scores + radii_p * q_dual) > th.unsqueeze(-1)

    return gate, {
        "p": float(p),
        "dual_p": float(dual_p),
        "radii_lp_mean": float(radii_p.mean()),
        "radii_lp_max": float(radii_p.max()),
    }


def make_enclose_lp_ball(p: float):
    """Factory that binds a concrete p value into the enclosing callable."""
    p = _validate_p(p)

    def _enclose_lp_ball(keys, assign, centers, K, bf):
        return enclose_lp_ball(keys, assign, centers, K, bf, p=p)

    return _enclose_lp_ball
