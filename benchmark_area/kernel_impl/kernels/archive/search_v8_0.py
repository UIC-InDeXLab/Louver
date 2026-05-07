"""search_v8.0 — anchor-cluster Triton kernel.

Per head, pick a fixed anchor subspace (s=0). Launch one program per
(head, cluster_in_anchor_subspace). Programs whose cluster fails the
anchor gate return immediately, skipping all children in that cluster.
Surviving programs iterate children in the cluster's parent-major range,
verify the remaining subspaces with per-point assigns + cluster_pass,
then compute the dot product only for survivors.

Output is initialized to -inf once; the kernel scatters values into the
surviving positions. Buffer-dot is concatenated afterwards.
"""

from __future__ import annotations

import torch

from ._search_triton import triton_anchor_cluster_search
from ._search_utils import buffer_dot, get_layout_cache

KERNEL_VERSION = "v8.0"


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
        cache_name="_search_v8_0_cache",
    )
    dots = triton_anchor_cluster_search(q, th_per_subspace, layout, anchor_s=0)
    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, layout)
    return dots if buf_dots is None else torch.cat([dots, buf_dots], dim=1)


KERNEL = search
