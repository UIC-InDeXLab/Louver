"""update_v2.2 — incremental update for build_v2.4 state with cheaper buffer build.

Same merge strategy as update_v2.1, but the buffer-only clustering caps the
Lloyd-style refinement to a single pass. On bench_update this trims most of the
buffer build cost while keeping pruning/recall effectively unchanged.
"""

from __future__ import annotations

import torch

from .build_v2_4 import build as build_v2_4

KERNEL_VERSION = "v2.2"
_BUFFER_REFINE_CAP = 1


def _assign_dtype(k: int) -> torch.dtype:
    return torch.int16 if k < 32768 else torch.int32


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
    if buffer_keys.shape[1] == 0:
        new_keys = old_keys
        new_values = old_values
        return state, new_keys, new_values

    # Extra buffer-only refinement showed little quality gain in bench_update.
    buffer_refine_iter = min(refine_iter, _BUFFER_REFINE_CAP)
    sub = build_v2_4(
        keys=buffer_keys,
        bf=bf,
        n_subspaces=n_subspaces,
        refine_iter=buffer_refine_iter,
        anchor_subspace=anchor_subspace,
        values=buffer_values,
    )

    k_old = int(state["K"])
    k_sub = int(sub["K"])
    k_new = k_old + k_sub
    n_old = int(state["N"])
    n_sub = int(sub["N"])
    n_pad_new = k_new * bf

    keys_reord_new = torch.cat([state["keys_reord"], sub["keys_reord"]], dim=1).contiguous()
    invalid_mask_new = torch.cat([state["invalid_mask"], sub["invalid_mask"]], dim=1).contiguous()
    reorder_perm_new = torch.cat(
        [state["reorder_perm"], sub["reorder_perm"] + n_old], dim=1
    ).contiguous()

    assigns_reord_new: list[torch.Tensor] = []
    for s_old, s_sub in zip(state["assigns_reord"], sub["assigns_reord"]):
        merged = torch.cat([s_old, s_sub + k_old], dim=1).contiguous()
        assigns_reord_new.append(merged)

    centers_new: list[torch.Tensor] = [
        torch.cat([c_old, c_sub], dim=1).contiguous()
        for c_old, c_sub in zip(state["centers"], sub["centers"])
    ]
    radii_new: list[torch.Tensor] = [
        torch.cat([r_old, r_sub], dim=1).contiguous()
        for r_old, r_sub in zip(state["radii"], sub["radii"])
    ]

    keys_blocks_t_new = torch.cat(
        [state["keys_blocks_t"], sub["keys_blocks_t"]], dim=1
    ).contiguous()
    invalid_blocks_i8_new = torch.cat(
        [state["invalid_blocks_i8"], sub["invalid_blocks_i8"]], dim=1
    ).contiguous()

    assigns_dtype = _assign_dtype(k_new)
    if state["assigns_blocks"].dtype != assigns_dtype:
        old_blocks = state["assigns_blocks"].to(assigns_dtype)
    else:
        old_blocks = state["assigns_blocks"]
    sub_blocks = sub["assigns_blocks"].to(assigns_dtype) + k_old
    assigns_blocks_new = torch.cat([old_blocks, sub_blocks], dim=2).contiguous()

    new_state: dict = {
        "dim_slices": state["dim_slices"],
        "centers": centers_new,
        "radii": radii_new,
        "assigns_reord": assigns_reord_new,
        "keys_reord": keys_reord_new,
        "invalid_mask": invalid_mask_new,
        "reorder_perm": reorder_perm_new,
        "K": k_new,
        "N": n_old + n_sub,
        "bf": bf,
        "N_pad": n_pad_new,
        "anchor_subspace": state.get("anchor_subspace", anchor_subspace),
        "keys_blocks_t": keys_blocks_t_new,
        "assigns_blocks": assigns_blocks_new,
        "invalid_blocks_i8": invalid_blocks_i8_new,
    }

    new_keys = torch.cat([old_keys, buffer_keys], dim=1).contiguous()
    new_values: torch.Tensor | None = None
    if "values_blocks_f16" in state and "values_blocks_f16" in sub:
        new_state["values_blocks_f16"] = torch.cat(
            [state["values_blocks_f16"], sub["values_blocks_f16"]], dim=1
        ).contiguous()
        new_state["values_reord"] = torch.cat(
            [state["values_reord"], sub["values_reord"]], dim=1
        ).contiguous()
        new_state["D_v"] = state["D_v"]
        if old_values is not None and buffer_values is not None:
            new_values = torch.cat([old_values, buffer_values], dim=1).contiguous()

    return new_state, new_keys, new_values


KERNEL = update
