"""CPU incremental update — clusters 256 buffer keys into 64 groups, writes
new centers/assigns/keys/values into the arena tail."""
from __future__ import annotations

import torch

from .._cpu_loader import load_ext

KERNEL_VERSION = "v1.0"
BF = 4
S = 4
BUFFER_SIZE = 256
K_BUF = BUFFER_SIZE // BF

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_ext("ta_cpu_update_v1", "update/update_v1.cpp")
    return _EXT


def update(state: dict, buffer_keys: torch.Tensor, buffer_values: torch.Tensor) -> None:
    if buffer_keys.shape[1] != BUFFER_SIZE:
        raise ValueError(f"need {BUFFER_SIZE} buffer keys; got {buffer_keys.shape[1]}")
    k_used = int(state["K_used"])
    n_used = int(state["N_used"])
    k_cap = int(state["K_cap"])
    n_cap = int(state["N_pad"])
    if k_used + K_BUF > k_cap or n_used + BUFFER_SIZE > n_cap:
        raise RuntimeError("arena full")

    _ext().cluster(
        buffer_keys.contiguous(),
        state["dim_offsets"], state["dim_widths"],
        state["centers_padded_f32"], state["assigns_padded_i32"],
        state["parent_children_i32"], state["parent_counts_i32"],
        k_used, n_used,
    )

    keys = state["keys_padded_f32"]
    values = state["values_padded_f32"]
    keys[:, n_used:n_used + BUFFER_SIZE, :].copy_(buffer_keys)
    values[:, n_used:n_used + BUFFER_SIZE, :].copy_(buffer_values)
    state["invalid_mask"][:, n_used:n_used + BUFFER_SIZE] = 0
    state["K_used"] = k_used + K_BUF
    state["N_used"] = n_used + BUFFER_SIZE

    if "keys_padded_bf16" in state:
        bk = buffer_keys.to(torch.bfloat16)
        bv = buffer_values.to(torch.bfloat16)
        state["keys_padded_bf16"][:, n_used:n_used + BUFFER_SIZE, :].copy_(bk)
        state["values_padded_bf16"][:, n_used:n_used + BUFFER_SIZE, :].copy_(bv)
        # Centers slab grew by K_BUF rows; refresh those rows from fp32.
        cb = state["centers_padded_f32"][:, :, k_used:k_used + K_BUF, :].to(torch.bfloat16)
        state["centers_padded_bf16"][:, :, k_used:k_used + K_BUF, :].copy_(cb)
