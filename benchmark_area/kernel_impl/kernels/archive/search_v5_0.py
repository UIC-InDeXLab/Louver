"""search_v5.0 — Triton direct point-gate + pruned dot in one kernel."""

from __future__ import annotations

import torch

from ._search_triton import triton_direct_search
from ._search_utils import buffer_dot, get_layout_cache

KERNEL_VERSION = "v5.0"


def search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    keys_children: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    layout = get_layout_cache(
        state=state,
        keys_children=keys_children,
        q_head_to_kv=q_head_to_kv,
        cache_name="_search_v5_0_cache",
    )
    dots = triton_direct_search(q, th_per_subspace, layout)
    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, layout)
    return dots if buf_dots is None else torch.cat([dots, buf_dots], dim=1)


KERNEL = search
