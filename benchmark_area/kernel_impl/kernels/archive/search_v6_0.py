"""search_v6.0 — anchor-subspace sparse search over parent-major child ranges.

Uses the per-subspace parent-major child layout emitted by build_v1_0:
pick the most selective subspace for each query head, gather its candidate
children from contiguous parent ranges, then verify the remaining subspaces
only on that reduced set.
"""

from __future__ import annotations

import torch

from ._search_utils import _NEG_INF, buffer_dot, cluster_pass_only, flatten_cluster_pass, get_layout_cache

KERNEL_VERSION = "v6.0"


def _head_base_index(head_idx: int, groups: int) -> int:
    return head_idx // groups


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
        cache_name="_search_v6_0_cache",
    )
    if "child_order" not in layout or "child_counts" not in layout:
        raise ValueError("search_v6_0 requires build state with child_order/child_offsets/child_counts.")

    cluster_pass, _ = cluster_pass_only(q, th_per_subspace, layout)
    cluster_pass_flat = flatten_cluster_pass(cluster_pass, layout)

    h_q, _, dim = q.shape[0], layout["num_points"], q.shape[1]
    out = q.new_full((h_q, layout["num_points"]), _NEG_INF)

    for h in range(h_q):
        base_h = _head_base_index(h, layout["groups"])
        counts_h = layout["child_counts"][:, base_h].to(torch.int64)           # (S, K)
        sub_counts = (cluster_pass_flat[:, h].to(torch.int64) * counts_h).sum(dim=1)

        anchor_s = int(sub_counts.argmin().item())
        anchor_mask_ordered = torch.repeat_interleave(
            cluster_pass_flat[anchor_s, h],
            counts_h[anchor_s],
            output_size=layout["num_points"],
        )
        if not bool(anchor_mask_ordered.any()):
            continue

        candidate_idx = layout["child_order"][anchor_s, base_h][anchor_mask_ordered]
        keep = torch.ones(candidate_idx.shape[0], dtype=torch.bool, device=q.device)

        for s in range(layout["num_subspaces"]):
            if s == anchor_s or candidate_idx.numel() == 0:
                continue
            assign_s = layout["assigns"][s, base_h].index_select(0, candidate_idx)
            keep &= cluster_pass_flat[s, h].gather(0, assign_s)
            if not bool(keep.any()):
                break

        if not bool(keep.any()):
            continue

        surviving_idx = candidate_idx[keep]
        key_rows = layout["keys"][base_h].index_select(0, surviving_idx)
        out[h, surviving_idx] = (key_rows * q[h].view(1, dim)).sum(dim=1)

    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, layout)
    return out if buf_dots is None else torch.cat([out, buf_dots], dim=1)


KERNEL = search
