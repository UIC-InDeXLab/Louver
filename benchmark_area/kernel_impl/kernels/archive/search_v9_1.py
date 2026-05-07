"""search_v9.1 — GQA-aware Triton kernel using tl.dot (TF32 tensor cores).

Grid: (H_kv, N_tiles). Each program:
  1. Loads the key tile (D, BLOCK_N) ONCE.
  2. Loads all GROUPS query rows for this kv head at once: (GROUPS, D).
  3. Does a single tl.dot → (GROUPS, BLOCK_N) dot product using TF32.
  4. Loops over groups only to apply the per-head survive mask and store.
"""

from __future__ import annotations

import torch

from ._search_triton import triton_gqa_tcore_search
from ._search_utils import buffer_dot, get_layout_cache

KERNEL_VERSION = "v9.1"


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
        cache_name="_search_v9_1_cache",
    )
    dots = triton_gqa_tcore_search(q, th_per_subspace, layout, block_n=128, num_warps=2)
    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, layout)
    return dots if buf_dots is None else torch.cat([dots, buf_dots], dim=1)


KERNEL = search
