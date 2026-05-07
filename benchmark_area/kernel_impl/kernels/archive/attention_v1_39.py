"""attention_v1.39 — v1.34 with fixed split count 40 for build_v2.7."""

from __future__ import annotations

from . import attention_v1_34 as _v34

KERNEL_VERSION = "v1.39"


def attend(
    q,
    th_per_subspace,
    state,
    buffer_keys,
    buffer_values,
    keys_children,
    q_head_to_kv=None,
    scale=None,
    num_splits: int = 40,
):
    return _v34.attend(
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
