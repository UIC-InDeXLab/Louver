"""search_v8.1 — anchor-cluster Triton kernel with cluster-batched programs.

Like v8_0 but each program processes CLUSTERS_PER_PROG clusters sequentially
to amortize launch overhead. Useful when K is large and each cluster has
few children.
"""

from __future__ import annotations

import torch

from ._search_triton import triton_anchor_cluster_batched_search
from ._search_utils import buffer_dot, get_layout_cache

KERNEL_VERSION = "v8.1"


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
        cache_name="_search_v8_1_cache",
    )
    dots = triton_anchor_cluster_batched_search(
        q, th_per_subspace, layout,
        anchor_s=0, max_cc=32, clusters_per_prog=8, num_warps=4,
    )
    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, layout)
    return dots if buf_dots is None else torch.cat([dots, buf_dots], dim=1)


KERNEL = search
