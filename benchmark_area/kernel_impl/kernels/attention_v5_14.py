"""attention_v5.14 — PPP=8, splits=128."""

from __future__ import annotations

from ._attention_v5_14_helper import attend_v5

KERNEL_VERSION = "v5.14"


def attend(
    q, th_per_subspace, state, buffer_keys, buffer_values,
    keys_children, q_head_to_kv=None, scale=None, num_splits: int = 128,
):
    return attend_v5(
        q=q, th_per_subspace=th_per_subspace, state=state,
        buffer_keys=buffer_keys, buffer_values=buffer_values,
        keys_children=keys_children, q_head_to_kv=q_head_to_kv,
        scale=scale, num_splits=num_splits,
        cache_ns="_attn_v5_14_fixed",
        num_warps=4, num_stages=3,
        parents_per_prog_override=8,
    )


KERNEL = attend
