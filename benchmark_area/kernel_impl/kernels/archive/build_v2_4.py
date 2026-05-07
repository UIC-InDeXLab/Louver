"""build_v2.4 — build_v2.1 plus block-packed values for fused attention.

Produces everything build_v2.1 does and additionally:
  - values_blocks_f16: (H_kv, K, BF, D_v) fp16
    Same physical reordering as keys_reord. Laid out so that for a given
    (parent, child) the D_v values are contiguous in memory — ideal for the
    fused attention kernel's `tl.dot(p, V_tile)` step.

Notes on conventions used by the attention kernel:
  - Attention uses no causal mask inside the kernel: everything in the index
    (and the decoding buffer) is a valid past key/value. Masking is handled
    by the caller's choice of what to append to the index/buffer.
  - Values are stored fp16 to match keys; softmax accumulators stay fp32.
"""

from __future__ import annotations

import torch

from .build_v2_1 import build as build_v2_1

KERNEL_VERSION = "v2.4"


def build(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    anchor_subspace: int = 0,
    values: torch.Tensor | None = None,
):
    """Build the subspace k-center index and pack values alongside keys.

    Args:
        keys:   (H_kv, N, D) keys.
        values: (H_kv, N, D_v) values aligned with keys. If None the values
                slot is left empty; the attention kernel will raise if called.
    """
    state = build_v2_1(
        keys=keys,
        bf=bf,
        n_subspaces=n_subspaces,
        refine_iter=refine_iter,
        anchor_subspace=anchor_subspace,
    )

    if values is not None:
        _pack_values_into_state(state, values)

    return state


def _pack_values_into_state(state: dict, values: torch.Tensor) -> None:
    """Permute + pack values using the same physical reorder as keys."""
    reorder_perm: torch.Tensor = state["reorder_perm"]  # (H_kv, N_pad) int64
    invalid_mask: torch.Tensor = state["invalid_mask"]  # (H_kv, N_pad) bool
    h_kv, n_pad_state = reorder_perm.shape
    h_kv_v, n_raw, d_v = values.shape
    assert h_kv == h_kv_v, f"head mismatch: reorder={h_kv} vs values={h_kv_v}"

    pad = n_pad_state - n_raw
    if pad > 0:
        pad_zeros = torch.zeros(
            h_kv, pad, d_v, device=values.device, dtype=values.dtype
        )
        values_padded = torch.cat([values, pad_zeros], dim=1)
    elif pad == 0:
        values_padded = values
    else:
        raise ValueError(
            f"values has more rows ({n_raw}) than N_pad ({n_pad_state})"
        )

    # Gather rows in the same physical order as keys.
    values_reord = values_padded.gather(
        1, reorder_perm[..., None].expand(-1, -1, d_v)
    ).contiguous()
    # Zero out padded slots so they contribute nothing if accidentally read.
    values_reord = values_reord.masked_fill(invalid_mask[..., None], 0.0)

    k = state["K"]
    bf = state["bf"]
    # Target layout: (H_kv, K, BF, D_v) — keys use (H_kv, K, D, BF).
    values_blocks_f16 = values_reord.view(h_kv, k, bf, d_v).to(torch.float16).contiguous()

    state["values_reord"] = values_reord
    state["values_blocks_f16"] = values_blocks_f16
    state["D_v"] = d_v


KERNEL = build
