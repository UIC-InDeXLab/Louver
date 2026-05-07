"""search_v3.0 — compiled packed subspace gate + grouped dense dot product."""

from __future__ import annotations

import torch

from ._search_utils import buffer_dot, get_layout_cache, make_compiled_dense_core

KERNEL_VERSION = "v3.0"


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
        cache_name="_search_v3_0_cache",
    )

    compiled = state.get("_search_v3_0_compiled")
    compiled_key = state.get("_search_v3_0_compiled_key")
    layout_key = (
        q_head_to_kv.shape[0] if q_head_to_kv is not None else keys_children.shape[0],
        keys_children.data_ptr(),
        tuple(keys_children.shape),
        q_head_to_kv.data_ptr() if q_head_to_kv is not None else None,
    )
    if compiled is None or compiled_key != layout_key:
        compiled = make_compiled_dense_core(layout)
        state["_search_v3_0_compiled"] = compiled
        state["_search_v3_0_compiled_key"] = layout_key

    dots = compiled(q, th_per_subspace)
    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, layout)
    return dots if buf_dots is None else torch.cat([dots, buf_dots], dim=1)


KERNEL = search
