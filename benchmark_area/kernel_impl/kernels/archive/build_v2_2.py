"""build_v2.2 — build_v2.1 plus parent-major assign packing for search_v12.

Adds:
  - assigns_parent_major: (H, K, S, BF) contiguous tensor where all subspace
    cluster ids for a parent block live next to each other in memory.
"""

from __future__ import annotations

import torch

from .build_v2_1 import build as build_v2_1

KERNEL_VERSION = "v2.2"


def _assign_dtype(k: int) -> torch.dtype:
    return torch.int16 if k < 32768 else torch.int32


def build(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    anchor_subspace: int = 0,
):
    state = build_v2_1(
        keys=keys,
        bf=bf,
        n_subspaces=n_subspaces,
        refine_iter=refine_iter,
        anchor_subspace=anchor_subspace,
    )

    assigns_reord = state["assigns_reord"]
    h_kv = state["keys_reord"].shape[0]
    k = state["K"]
    bf = state["bf"]
    s = len(assigns_reord)

    assigns_parent_major = (
        torch.stack(assigns_reord, dim=0)
        .to(_assign_dtype(k))
        .view(s, h_kv, k, bf)
        .permute(1, 2, 0, 3)
        .contiguous()
    )
    state["assigns_parent_major"] = assigns_parent_major
    return state


KERNEL = build
