"""update_v3.2 — preallocated growing arena update.

The first call for a base state caches an arena with spare K capacity. Later
calls write only the appended buffer slice into that arena.
"""

from __future__ import annotations

import torch

from ._update_v3_utils import build_sub_cpu, merge_arena

KERNEL_VERSION = "v3.2"


def update(
    state: dict,
    old_keys: torch.Tensor,
    buffer_keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    old_values: torch.Tensor | None = None,
    buffer_values: torch.Tensor | None = None,
    anchor_subspace: int = 0,
    return_merged: bool = False,
) -> tuple[dict, torch.Tensor | None, torch.Tensor | None]:
    if buffer_keys.shape[1] == 0:
        return state, old_keys if return_merged else None, old_values if return_merged else None

    sub = build_sub_cpu(
        buffer_keys,
        bf,
        n_subspaces,
        anchor_subspace,
        buffer_values,
        with_values=buffer_values is not None,
    )
    return merge_arena(
        state,
        sub,
        old_keys,
        buffer_keys,
        old_values,
        buffer_values,
        bf,
        anchor_subspace,
        return_merged,
    )


KERNEL = update
