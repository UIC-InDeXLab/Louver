"""build_v2.1-fp16 — build_v2.1 state emitted directly in fp16."""

from __future__ import annotations

from ._build_fp16_utils import build_v2_1_fp16_state
from .build_v2_0 import ANCHOR_SUBSPACE

KERNEL_VERSION = "v2.1-fp16"


def build(
    keys,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    anchor_subspace: int = ANCHOR_SUBSPACE,
):
    return build_v2_1_fp16_state(
        keys=keys,
        bf=bf,
        n_subspaces=n_subspaces,
        refine_iter=refine_iter,
        anchor_subspace=anchor_subspace,
    )


KERNEL = build
