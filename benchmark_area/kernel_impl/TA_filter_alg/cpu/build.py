"""CPU TA-filter index build.

Wraps the existing torch-side balanced-PCA tree builder from
``TA_filter_alg/kernels/TA_build.py`` and re-emits the state in fp32 layouts
the CPU kernels expect:

    centers_padded_f32   [S, H_kv, K, max_w]
    assigns_padded_i32   [S, H_kv, N_pad]
    keys_padded_f32      [H_kv, N_pad, D]
    values_padded_f32    [H_kv, N_pad, D_v]
    invalid_mask         [H_kv, N_pad] uint8
    dim_offsets          [S] int32
    dim_widths           [S] int32

Build is **not** the optimisation target (per instructions); we re-use the
correctness-tested torch builder and convert dtypes for cache friendliness.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from ..kernels import TA_build

S_FIXED = 4
BF = 4
BUFFER_SIZE = 256
K_BUF = BUFFER_SIZE // BF


def build_state(keys: torch.Tensor, values: torch.Tensor, *, refine_iter: int = 5) -> dict:
    """Build CPU TA-filter index state.

    keys/values: (H_kv, N, D) fp32 tensors on CPU.
    """
    if keys.device.type != "cpu":
        raise ValueError("CPU build requires CPU tensors")
    if keys.dtype != torch.float32 or values.dtype != torch.float32:
        raise ValueError("CPU pipeline uses fp32 storage")

    raw = TA_build.build(
        keys=keys.to(torch.float16),
        bf=BF,
        n_subspaces=S_FIXED,
        refine_iter=refine_iter,
        values=values.to(torch.float16),
    )

    out = {
        "centers_padded_f32": raw["centers_padded_f16"].to(torch.float32).contiguous(),
        "assigns_padded_i32": raw["assigns_padded"].to(torch.int32).contiguous(),
        "keys_padded_f32":    raw["keys_padded_f16"].to(torch.float32).contiguous(),
        "values_padded_f32":  raw["values_padded_f16"].to(torch.float32).contiguous(),
        "invalid_mask":       raw["invalid_mask"].to(torch.uint8).contiguous(),
        "dim_offsets":        raw["dim_offsets"].to(torch.int32).contiguous(),
        "dim_widths":         raw["dim_widths"].to(torch.int32).contiguous(),
        "dim_slices":         list(raw["dim_slices"]),
        "max_width":          int(raw["max_width"]),
        "K":                  int(raw["K"]),
        "N":                  int(raw["N"]),
        "N_pad":              int(raw["N_pad"]),
        "bf":                 int(raw["bf"]),
        "D":                  int(raw["D"]),
        "n_subspaces":        int(raw["n_subspaces"]),
    }
    out["K_used"] = out["K"]
    out["N_used"] = out["N"]
    out["K_cap"] = out["K"]
    _build_parent_children(out)
    refresh_bf16_views(out)
    return out


def refresh_bf16_views(state: dict) -> None:
    """Materialise bf16 mirrors of centers/keys/values for the v3 kernel."""
    state["centers_padded_bf16"] = state["centers_padded_f32"].to(torch.bfloat16).contiguous()
    state["keys_padded_bf16"]    = state["keys_padded_f32"].to(torch.bfloat16).contiguous()
    state["values_padded_bf16"]  = state["values_padded_f32"].to(torch.bfloat16).contiguous()


def _build_parent_children(state: dict) -> None:
    """Build parent_children[S, H_kv, K_cap, bf] int32 (inverse of assigns).

    Vectorised per-(s, h) using torch sort + bincount.
    """
    assigns = state["assigns_padded_i32"]   # (S, H_kv, N_pad)
    invalid = state["invalid_mask"]         # (H_kv, N_pad)
    S, H_kv, N_pad = assigns.shape
    K_cap = int(state.get("K_cap", state["K_used"]))
    bf = int(state["bf"])
    K_alloc = max(K_cap, int(state["K_used"]))
    pc = torch.full((S, H_kv, K_alloc, bf), -1, dtype=torch.int32)
    counts = torch.zeros(S, H_kv, K_alloc, dtype=torch.int32)

    pos = torch.arange(N_pad, dtype=torch.long)
    for s in range(S):
        a_s = assigns[s].to(torch.long)  # (H_kv, N_pad)
        for h in range(H_kv):
            valid_mask = invalid[h] == 0
            keys = pos[valid_mask]
            parents = a_s[h][valid_mask]
            order = torch.argsort(parents, stable=True)
            keys_sorted = keys[order]
            parents_sorted = parents[order]
            cnt_p = torch.bincount(parents_sorted, minlength=K_alloc)
            offsets = torch.cat([torch.zeros(1, dtype=torch.long), cnt_p.cumsum(0)[:-1]])
            slots = torch.arange(keys_sorted.numel(), dtype=torch.long) - offsets[parents_sorted]
            ok = slots < bf
            pc[s, h, parents_sorted[ok], slots[ok]] = keys_sorted[ok].to(torch.int32)
            counts[s, h] = cnt_p.clamp_max(bf).to(torch.int32)
    state["parent_children_i32"] = pc.contiguous()
    state["parent_counts_i32"] = counts.contiguous()


def expand_arena(state: dict[str, Any], *, k_cap: int, n_cap: int) -> None:
    """Pad the arena tensors so update() can append into the tail."""
    def grow(t: torch.Tensor, dim: int, new: int, fill) -> torch.Tensor:
        old = t.shape[dim]
        if old >= new:
            return t
        pad_shape = list(t.shape)
        pad_shape[dim] = new - old
        pad = torch.full(pad_shape, fill, device=t.device, dtype=t.dtype)
        return torch.cat([t, pad], dim=dim).contiguous()

    state["centers_padded_f32"] = grow(state["centers_padded_f32"], 2, k_cap, 0.0)
    state["assigns_padded_i32"] = grow(state["assigns_padded_i32"], 2, n_cap, 0)
    state["keys_padded_f32"]    = grow(state["keys_padded_f32"],    1, n_cap, 0.0)
    state["values_padded_f32"]  = grow(state["values_padded_f32"],  1, n_cap, 0.0)

    inv = state["invalid_mask"]
    if inv.shape[1] < n_cap:
        pad = torch.ones(inv.shape[0], n_cap - inv.shape[1], dtype=torch.uint8)
        state["invalid_mask"] = torch.cat([inv, pad], dim=1).contiguous()

    if "parent_children_i32" in state:
        state["parent_children_i32"] = grow(state["parent_children_i32"], 2, k_cap, -1)
        state["parent_counts_i32"]   = grow(state["parent_counts_i32"],   2, k_cap, 0)

    state["K_cap"] = k_cap
    state["N_pad"] = n_cap
    refresh_bf16_views(state)
