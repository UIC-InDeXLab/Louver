"""search_v2.0 — packed subspace gate + grouped dense dot product."""

from __future__ import annotations

import torch

from ._search_utils import buffer_dot, dense_index_search, get_layout_cache

KERNEL_VERSION = "v2.0"


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
        cache_name="_search_v2_0_cache",
    )
    dots = dense_index_search(q, th_per_subspace, layout)
    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, layout)
    return dots if buf_dots is None else torch.cat([dots, buf_dots], dim=1)


KERNEL = search
