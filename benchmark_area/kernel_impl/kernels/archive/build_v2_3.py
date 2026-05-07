"""build_v2.3 — build_v2.1 plus a two-level anchor hierarchy.

Extends build_v2.1 with a coarse hierarchy over anchor parents:
  - super_centers_anchor: (H, K2, d_anchor)
  - super_radii_anchor:   (H, K2)
  - super_parent_ids:     (H, K2, SUPER_BF) int32
  - super_parent_invalid_i8: (H, K2, SUPER_BF) int8

Each anchor parent belongs to exactly one super-parent slot. The hierarchy is
only used at search time to prune whole groups of anchor parents before
evaluating parent-level anchor bounds.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .build_v2_0 import _balanced_assign_per_head, _kcenter_subspace
from .build_v2_1 import build as build_v2_1

KERNEL_VERSION = "v2.3"
ANCHOR_SUPER_BF = 8


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

    anchor_s = state.get("anchor_subspace", anchor_subspace)
    anchor_centers: torch.Tensor = state["centers"][anchor_s]
    anchor_radii: torch.Tensor = state["radii"][anchor_s]
    h_kv, k, d_anchor = anchor_centers.shape
    device = anchor_centers.device
    dtype = anchor_centers.dtype

    super_bf = ANCHOR_SUPER_BF
    super_k = max(1, math.ceil(k / super_bf))
    refine_super = max(1, min(refine_iter, 2))

    _, super_centers_seed = _kcenter_subspace(anchor_centers, super_k, refine_super)
    dists = torch.cdist(anchor_centers, super_centers_seed)
    dists_np = dists.cpu().numpy()
    super_assign_np = np.empty((h_kv, k), dtype=np.int64)
    for h in range(h_kv):
        super_assign_np[h] = _balanced_assign_per_head(dists_np[h], super_bf)
    super_assign = torch.from_numpy(super_assign_np).to(device=device)

    super_parent_ids = torch.zeros(
        h_kv, super_k, super_bf, device=device, dtype=torch.int32
    )
    super_parent_invalid_i8 = torch.ones(
        h_kv, super_k, super_bf, device=device, dtype=torch.int8
    )
    super_centers_anchor = torch.zeros(
        h_kv, super_k, d_anchor, device=device, dtype=dtype
    )
    super_radii_anchor = torch.zeros(h_kv, super_k, device=device, dtype=dtype)

    for h in range(h_kv):
        for sk in range(super_k):
            idx = (super_assign[h] == sk).nonzero(as_tuple=True)[0]
            count = int(idx.numel())
            if count == 0:
                super_centers_anchor[h, sk] = super_centers_seed[h, sk]
                continue

            super_parent_ids[h, sk, :count] = idx.to(torch.int32)
            super_parent_invalid_i8[h, sk, :count] = 0

            pts = anchor_centers[h, idx]
            ctr = pts.mean(dim=0)
            super_centers_anchor[h, sk] = ctr
            center_dists = (pts - ctr).norm(dim=-1)
            super_radii_anchor[h, sk] = (center_dists + anchor_radii[h, idx]).max()

    state["anchor_super_bf"] = super_bf
    state["anchor_super_k"] = super_k
    state["super_centers_anchor"] = super_centers_anchor.contiguous()
    state["super_radii_anchor"] = super_radii_anchor.contiguous()
    state["super_parent_ids"] = super_parent_ids.contiguous()
    state["super_parent_invalid_i8"] = super_parent_invalid_i8.contiguous()
    return state


KERNEL = build
