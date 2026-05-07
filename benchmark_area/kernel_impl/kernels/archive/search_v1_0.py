"""search_v1.0 — torch-vectorized subspace AND-gate search + buffer dot.

Given a query q: (H_q, D), per-subspace thresholds th: (S, H_q), and
the index state, return a (H_q, N_total) fp tensor of dot products with
-inf at non-surviving positions.

N_total = N_children + N_buffer (buffer entries are always "survivors").
"""

from __future__ import annotations

import torch

KERNEL_VERSION = "v1.0"

_NEG_INF = float("-inf")


def _expand(t, q_head_to_kv):
    return t if q_head_to_kv is None else t[q_head_to_kv]


def search(
    q: torch.Tensor,                           # (H_q, D)
    th_per_subspace: torch.Tensor,             # (S, H_q)
    state: dict,                               # index state
    buffer_keys: torch.Tensor | None,          # (H_kv, B, D) or None
    keys_children: torch.Tensor,               # (H_kv, N, D)  — full keys in index
    q_head_to_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return (H_q, N_total) dot products; non-survivors = -inf."""
    H_q, D = q.shape
    device = q.device

    dim_slices = state["dim_slices"]
    assigns = state["assigns"]
    centers = state["centers"]
    radii = state["radii"]
    N = state["N"]

    # ── Subspace AND gate: cluster pass -> point pass ─────────────────
    survive = torch.ones(H_q, N, dtype=torch.bool, device=device)
    for s, (start, end) in enumerate(dim_slices):
        q_sub = q[:, start:end]
        q_sub_norm = q_sub.norm(dim=-1)                                     # (H_q,)
        centers_s = _expand(centers[s], q_head_to_kv)                      # (H_q, K, d_s)
        radii_s = _expand(radii[s], q_head_to_kv)                          # (H_q, K)
        assign_s = _expand(assigns[s], q_head_to_kv)                       # (H_q, N)

        center_dots = torch.einsum("hkd,hd->hk", centers_s, q_sub)          # (H_q, K)
        cluster_ub = center_dots + radii_s * q_sub_norm.unsqueeze(-1)
        cluster_pass = cluster_ub >= th_per_subspace[s].unsqueeze(-1)
        point_pass = cluster_pass.gather(1, assign_s)                       # (H_q, N)
        survive &= point_pass

    # ── Dot product against surviving index children ──────────────────
    keys_q = _expand(keys_children, q_head_to_kv)                          # (H_q, N, D)
    # Full dot product, then mask non-survivors to -inf.
    # Rationale from user: single tensor with -inf at non-survivors.
    dots = torch.einsum("hd,hnd->hn", q, keys_q)                           # (H_q, N)
    dots = dots.masked_fill(~survive, _NEG_INF)

    # ── Dot product against buffer (always scanned) ───────────────────
    if buffer_keys is not None and buffer_keys.shape[1] > 0:
        buf_q = _expand(buffer_keys, q_head_to_kv)                          # (H_q, B, D)
        buf_dots = torch.einsum("hd,hbd->hb", q, buf_q)
        dots = torch.cat([dots, buf_dots], dim=1)

    return dots


KERNEL = search
