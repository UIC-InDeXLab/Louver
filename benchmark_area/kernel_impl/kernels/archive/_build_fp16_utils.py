"""Shared helpers for fp16 build variants."""

from __future__ import annotations

import torch

from .build_v2_0 import ANCHOR_SUBSPACE, build as build_v2_0


def _assign_dtype(k: int) -> torch.dtype:
    return torch.int16 if k < 32768 else torch.int32


def _to_fp16_keys(keys: torch.Tensor) -> torch.Tensor:
    return keys if keys.dtype == torch.float16 else keys.to(torch.float16)


def build_v2_1_fp16_state(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    anchor_subspace: int = ANCHOR_SUBSPACE,
) -> dict:
    keys_fp16 = _to_fp16_keys(keys)
    state = build_v2_0(
        keys=keys_fp16,
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

    state["precision"] = "fp16_full"
    state["keys_blocks_t_f16"] = (
        keys_reord.view(h_kv, k, bf, d).permute(0, 1, 3, 2).contiguous()
    )
    state["assigns_blocks"] = (
        torch.stack(assigns_reord, dim=0)
        .to(_assign_dtype(k))
        .view(s, h_kv, k, bf)
        .contiguous()
    )
    state["invalid_blocks_i8"] = invalid_mask.view(h_kv, k, bf).to(torch.int8).contiguous()
    return state


def build_v2_2_fp16_state(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    anchor_subspace: int = ANCHOR_SUBSPACE,
) -> dict:
    state = build_v2_1_fp16_state(
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

    state["assigns_parent_major"] = (
        torch.stack(assigns_reord, dim=0)
        .to(_assign_dtype(k))
        .view(s, h_kv, k, bf)
        .permute(1, 2, 0, 3)
        .contiguous()
    )
    return state
