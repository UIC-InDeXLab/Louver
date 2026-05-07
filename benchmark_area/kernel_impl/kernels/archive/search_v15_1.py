"""search_v15.1 — full-fp16 build and search path."""

from __future__ import annotations

import torch

try:
    import triton

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from ._search_fp16_utils import (
    _get_layout_v15_fp16,
    _next_pow2,
    buffer_dot_to_dtype,
    pack_query_for_fp16_search,
)
from ._search_triton import triton_fused_cluster_pass
from .search_v15_0 import _PARENTS_PER_PROG, _fp16_fused_kernel

KERNEL_VERSION = "v15.1"


def search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    keys_children: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("search_v15_1 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("search_v15_1 requires build_v2-style state")

    layout = _get_layout_v15_fp16(state, q_head_to_kv, q, "_search_v15_1_cache")
    q_cast, q_packed, q_norm, th_packed = pack_query_for_fp16_search(q, th_per_subspace, layout)

    cluster_pass_flat = triton_fused_cluster_pass(
        q_packed,
        q_norm,
        th_packed,
        layout["centers"],
        layout["radii"],
        layout["groups"],
    )

    h_q = q_cast.shape[0]
    d = q_cast.shape[1]
    h_kv = layout["base_heads"]
    k = layout["K"]
    bf = layout["bf"]
    n_pad = layout["N_pad"]
    groups = layout["groups"]
    groups_pow = max(_next_pow2(groups), 8)
    anchor_s = layout["anchor_subspace"]

    out = torch.empty(h_q, n_pad, device=q_cast.device, dtype=q_cast.dtype)
    grid = (h_kv, triton.cdiv(k, _PARENTS_PER_PROG))
    _fp16_fused_kernel[grid](
        q_cast.contiguous(),
        layout["keys_blocks_t_f16"],
        layout["assigns_blocks"],
        cluster_pass_flat,
        layout["invalid_blocks_i8"],
        out,
        h_q,
        h_kv,
        k,
        n_pad,
        ANCHOR_S=anchor_s,
        D=d,
        BF=bf,
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        S=layout["num_subspaces"],
        PARENTS_PER_PROG=_PARENTS_PER_PROG,
        num_warps=4,
    )

    buf_shim = {"mode": layout["mode"], "groups": groups, "base_heads": h_kv}
    buf_dots = buffer_dot_to_dtype(q_cast, buffer_keys, q_head_to_kv, buf_shim, out.dtype)
    return out if buf_dots is None else torch.cat([out, buf_dots], dim=1)


KERNEL = search
