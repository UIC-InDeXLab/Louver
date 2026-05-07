"""attention_v2.3 — v2.0 (exp2) with num_splits=42 (~1 wave for 4 KV heads)."""

from __future__ import annotations

from . import attention_v2_0 as _base

KERNEL_VERSION = "v2.3"


def attend(
    q, th_per_subspace, state, buffer_keys, buffer_values,
    keys_children, q_head_to_kv=None, scale=None, num_splits: int = 42,
):
    return _base.attend(
        q=q, th_per_subspace=th_per_subspace, state=state,
        buffer_keys=buffer_keys, buffer_values=buffer_values,
        keys_children=keys_children, q_head_to_kv=q_head_to_kv,
        scale=scale, num_splits=num_splits,
    )


KERNEL = attend
