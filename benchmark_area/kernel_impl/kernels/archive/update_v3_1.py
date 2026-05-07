"""update_v3.1 — v3.0 plus GPU-only balanced buffer assignment."""

from __future__ import annotations

import torch

from ._update_v3_utils import build_sub_cpu, build_sub_gpu_rounds, merge_cat

KERNEL_VERSION = "v3.1"


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

    try:
        sub = build_sub_gpu_rounds(
            buffer_keys,
            bf,
            n_subspaces,
            anchor_subspace,
            buffer_values,
            with_values=buffer_values is not None,
        )
    except Exception:
        sub = build_sub_cpu(
            buffer_keys,
            bf,
            n_subspaces,
            anchor_subspace,
            buffer_values,
            with_values=buffer_values is not None,
        )

    return merge_cat(
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
