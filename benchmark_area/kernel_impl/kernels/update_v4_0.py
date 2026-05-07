"""update_v4.0 — overlap-aware arena update.

Same data flow as update_v3.3 (arena + Triton-direct value pack), but the
merge into the arena is split into two phases so it can run in parallel
with attention:

  Phase 1 — merge_arena_async (this kernel):
      Writes the buffer's clustered data into the arena's unused
      [k_old:k_new] slice WITHOUT flipping the invalid flags. Attention on
      another stream observes that range as invalid and skips it.

  Phase 2 — apply_pending_publish (called by the index later):
      Runs on the attention stream after a wait on ``update_done_event``.
      Flips the invalid flags (= publish), advances K_used / N_used,
      invalidates the cached attention layouts.

The kernel returns ``(arena_state, new_keys, new_values, pending_publish)``.
The bench / index machinery is responsible for actually scheduling phase 2.
"""

from __future__ import annotations

import torch

from ._update_v3_utils import (
    build_sub_cpu,
    build_sub_gpu_rounds,
    merge_arena_async,
    pack_values_direct,
)

KERNEL_VERSION = "v4.0"


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
) -> tuple[dict, torch.Tensor | None, torch.Tensor | None, dict | None]:
    if buffer_keys.shape[1] == 0:
        return (
            state,
            old_keys if return_merged else None,
            old_values if return_merged else None,
            None,
        )

    # Prefer the GPU-rounds builder so the host thread doesn't stall on a
    # CUDA→CPU sync mid-update (build_sub_cpu's balanced-assign path does a
    # `dists.cpu().numpy()` round-trip which would block attention launches).
    # Fall back to the CPU builder if the GPU path is unavailable.
    try:
        sub = build_sub_gpu_rounds(
            buffer_keys,
            bf,
            n_subspaces,
            anchor_subspace,
            None,
            with_values=False,
        )
    except Exception:
        sub = build_sub_cpu(
            buffer_keys,
            bf,
            n_subspaces,
            anchor_subspace,
            None,
            with_values=False,
        )
    if buffer_values is not None:
        pack_values_direct(sub, buffer_values)

    return merge_arena_async(
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
