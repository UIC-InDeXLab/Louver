"""search_v9.0 — GQA-aware Triton kernel: keys tile loaded once per kv head.

The grid is (H_kv, N_tiles). Each program loads the key tile for its kv
head once, then loops over the GROUPS query heads that share that kv
head. For each query head it recomputes the subspace AND-gate (cheap)
and accumulates the dot against the *shared* key tile in registers.

This is a major saving vs. the (H_q, N_tiles) layouts of v4/v5/v7 where
every query-head program reloads the same keys GROUPS times.

Falls back to the clusterpass kernel when the mapping mode is not
grouped (identity or expanded).
"""

from __future__ import annotations

import torch

from ._search_triton import triton_gqa_clusterpass_search
from ._search_utils import buffer_dot, get_layout_cache

KERNEL_VERSION = "v9.0"


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
        cache_name="_search_v9_0_cache",
    )
    dots = triton_gqa_clusterpass_search(q, th_per_subspace, layout, block_n=256, num_warps=4)
    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, layout)
    return dots if buf_dots is None else torch.cat([dots, buf_dots], dim=1)


KERNEL = search
