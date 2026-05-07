"""CPU update kernel — calls the index extension."""

from __future__ import annotations

import torch

from ._cpu_ext_loader import index_ext

KERNEL_VERSION = "cpu_v1.0"


def update(
    state: dict,
    old_keys: torch.Tensor,
    buffer_keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 0,
    old_values: torch.Tensor | None = None,
    buffer_values: torch.Tensor | None = None,
    anchor_subspace: int | None = None,
    return_merged: bool = False,
):
    if buffer_keys.shape[1] == 0:
        return (
            state,
            old_keys if return_merged else None,
            old_values if return_merged else None,
            None,
        )
    if anchor_subspace is None:
        anchor_subspace = int(state.get("orig_anchor_subspace", n_subspaces - 1))
    return index_ext().update_index(
        state, old_keys, buffer_keys,
        int(bf), int(n_subspaces), int(refine_iter),
        old_values, buffer_values,
        int(anchor_subspace), bool(return_merged),
    )


KERNEL = update
