"""CPU build kernel — calls the index extension. Not the primary optimization
target; same correctness as before, just isolated in `_index_ext.cpp`."""

from __future__ import annotations

import torch

from ._cpu_ext_loader import index_ext

KERNEL_VERSION = "cpu_v1.0"


def build(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 2,
    anchor_subspace: int | None = None,
    values: torch.Tensor | None = None,
) -> dict:
    if anchor_subspace is None:
        anchor_subspace = n_subspaces - 1
    return index_ext().build_index(
        keys, int(bf), int(n_subspaces),
        int(refine_iter), int(anchor_subspace), values,
    )


KERNEL = build
