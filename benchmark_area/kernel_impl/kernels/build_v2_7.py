"""build_v2.7 — experimental last-subspace anchor for anchor-only attention.

This variant is intentionally narrow: it exists to benchmark whether a fixed
trailing contiguous subspace works better as the anchor for v1.34-style
anchor-only attention on real captures than v2.5's generic tightness heuristic.
"""

from __future__ import annotations

import torch

from ._build_update_active import build as build_active

KERNEL_VERSION = "v2.7"


def build(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    anchor_subspace: int | None = None,
    values: torch.Tensor | None = None,
):
    if anchor_subspace is None:
        anchor_subspace = n_subspaces - 1
    state = build_active(
        keys=keys,
        bf=bf,
        n_subspaces=n_subspaces,
        refine_iter=refine_iter,
        anchor_subspace=int(anchor_subspace),
        values=values,
    )
    state["anchor_selection_strategy"] = "fixed_last_subspace"
    return state


KERNEL = build
