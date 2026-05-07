"""search_v2.1 — packed subspace gate + survivor-only sparse dot product."""

from __future__ import annotations

import torch

from ._search_utils import _NEG_INF, buffer_dot, gate_and_group_query, get_layout_cache

KERNEL_VERSION = "v2.1"


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
        cache_name="_search_v2_1_cache",
    )
    survive, _ = gate_and_group_query(q, th_per_subspace, layout)
    dots = q.new_full((q.shape[0], layout["num_points"]), _NEG_INF)

    idx = survive.nonzero(as_tuple=False)
    if idx.numel() > 0:
        head_idx = idx[:, 0]
        point_idx = idx[:, 1]
        if layout["mode"] == "grouped":
            kv_head_idx = head_idx // layout["groups"]
        elif q_head_to_kv is None:
            kv_head_idx = head_idx
        else:
            kv_head_idx = q_head_to_kv.index_select(0, head_idx)

        values = (q.index_select(0, head_idx) * keys_children[kv_head_idx, point_idx]).sum(dim=-1)
        dots[head_idx, point_idx] = values

    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, layout)
    return dots if buf_dots is None else torch.cat([dots, buf_dots], dim=1)


KERNEL = search
