"""update_v2.0 — full rebuild update for build_v2.4 state.

Concatenates old keys/values with the buffer and calls build_v2_4 from scratch.
Serves as correctness baseline for the incremental variants (v2.1, v2.2).
"""

from __future__ import annotations

import torch

from .build_v2_4 import build as build_v2_4

KERNEL_VERSION = "v2.0"


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
) -> tuple[dict, torch.Tensor, torch.Tensor | None]:
    new_keys = torch.cat([old_keys, buffer_keys], dim=1).contiguous()
    new_values = None
    if old_values is not None and buffer_values is not None:
        new_values = torch.cat([old_values, buffer_values], dim=1).contiguous()
    new_state = build_v2_4(
        keys=new_keys,
        bf=bf,
        n_subspaces=n_subspaces,
        refine_iter=refine_iter,
        anchor_subspace=anchor_subspace,
        values=new_values,
    )
    return new_state, new_keys, new_values


KERNEL = update
