"""build_v2.1 — build_v2.0 plus block-packed tensors for search_v11.

Keeps build_v2.0's anchor-block physical ordering and additionally emits:
  - keys_blocks_t:    (H, K, D, BF) contiguous tensor-core-friendly blocks
  - assigns_blocks:   (S, H, K, BF) packed per-parent child assigns
  - invalid_blocks_i8:(H, K, BF) int8 invalid flags for padded children

These tensors let the search kernel batch multiple parents per program and
avoid re-packing state on the search path.
"""

from __future__ import annotations

import torch

from .build_v2_0 import ANCHOR_SUBSPACE, build as build_v2_0

KERNEL_VERSION = "v2.1"


def _assign_dtype(k: int) -> torch.dtype:
    return torch.int16 if k < 32768 else torch.int32


def build(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    anchor_subspace: int = ANCHOR_SUBSPACE,
):
    state = build_v2_0(
        keys=keys,
        bf=bf,
        n_subspaces=n_subspaces,
        refine_iter=refine_iter,
        anchor_subspace=anchor_subspace,
    )

    keys_reord = state["keys_reord"]
    invalid_mask = state["invalid_mask"]
    assigns_reord = state["assigns_reord"]

    h_kv, _, d = keys_reord.shape
    k = state["K"]
    bf = state["bf"]
    s = len(assigns_reord)

    keys_blocks_t = (
        keys_reord.view(h_kv, k, bf, d).permute(0, 1, 3, 2).contiguous()
    )
    assigns_blocks = (
        torch.stack(assigns_reord, dim=0)
        .to(_assign_dtype(k))
        .view(s, h_kv, k, bf)
        .contiguous()
    )
    invalid_blocks_i8 = invalid_mask.view(h_kv, k, bf).to(torch.int8).contiguous()

    state["keys_blocks_t"] = keys_blocks_t
    state["assigns_blocks"] = assigns_blocks
    state["invalid_blocks_i8"] = invalid_blocks_i8
    return state


KERNEL = build
