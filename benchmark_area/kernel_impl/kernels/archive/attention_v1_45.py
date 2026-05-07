"""attention_v1.45 — v1.43 with num_splits=85 (2 full waves on 170 SMs)."""

from __future__ import annotations

from . import attention_v1_43 as _base

KERNEL_VERSION = "v1.45"


def attend(
    q,
    th_per_subspace,
    state,
    buffer_keys,
    buffer_values,
    keys_children,
    q_head_to_kv=None,
    scale=None,
    num_splits: int = 85,
):
    return _base.attend(
        q=q,
        th_per_subspace=th_per_subspace,
        state=state,
        buffer_keys=buffer_keys,
        buffer_values=buffer_values,
        keys_children=keys_children,
        q_head_to_kv=q_head_to_kv,
        scale=scale,
        num_splits=num_splits,
    )


KERNEL = attend
