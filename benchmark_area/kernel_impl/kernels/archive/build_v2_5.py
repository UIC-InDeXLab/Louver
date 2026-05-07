"""build_v2.5 — build_v2.4 plus auto anchor selection and subspace reordering.

This build variant keeps build_v2.0/2.4's physical blocked layout, but changes
two pieces of metadata organization:
  - choose the anchor subspace automatically using a build-time tightness score
  - reorder the non-anchor subspaces so tighter ones run earlier at search time

The goal is to make the anchor gate more selective and shrink live columns
earlier in the sparse attention kernel.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .build_v2_0 import (
    _ball_centroid,
    _balanced_assign_per_head,
    _kcenter_subspace,
    _split_contiguous,
)
from .build_v2_4 import _pack_values_into_state

KERNEL_VERSION = "v2.5"


def _subspace_tightness_score(radii: torch.Tensor, width: int) -> float:
    # Width-normalize so narrower subspaces do not win trivially.
    width_scale = math.sqrt(max(width, 1))
    return float(radii.mean().item() / width_scale)


def _anchor_candidate_layout(
    keys: torch.Tensor,
    keys_padded: torch.Tensor,
    slices: list[tuple[int, int]],
    centers_per_sub: list[torch.Tensor],
    anchor_subspace: int,
    bf: int,
    n_real: int,
) -> dict:
    h_kv, _, d = keys_padded.shape
    k = centers_per_sub[anchor_subspace].shape[1]
    device = keys.device
    dtype = keys.dtype
    n_pad = k * bf

    s0, e0 = slices[anchor_subspace]
    keys_anchor = keys_padded[:, :, s0:e0].contiguous()
    centers_anchor = centers_per_sub[anchor_subspace]
    dists_anchor = torch.cdist(keys_anchor, centers_anchor)
    dists_np = dists_anchor.cpu().numpy()
    bal_assign_np = np.empty((h_kv, n_pad), dtype=np.int64)
    for h in range(h_kv):
        bal_assign_np[h] = _balanced_assign_per_head(dists_np[h], bf)
    bal_assign = torch.from_numpy(bal_assign_np).to(device=device)

    sort_order = torch.argsort(bal_assign, dim=1, stable=True)
    keys_reord = keys_padded.gather(1, sort_order[..., None].expand(-1, -1, d)).contiguous()

    src_idx = torch.arange(n_pad, device=device).expand(h_kv, -1)
    invalid_src = src_idx >= n_real
    invalid_mask = invalid_src.gather(1, sort_order)
    reorder_perm = sort_order.contiguous()

    keys_grouped = keys_reord.view(h_kv, k, bf, d)
    inv_grouped = invalid_mask.view(h_kv, k, bf)
    real_mask = (~inv_grouped).to(dtype).unsqueeze(-1)
    real_count = real_mask.sum(dim=2).clamp_min(1.0)
    sub_anchor = keys_grouped[..., s0:e0]
    center_anchor_new = (sub_anchor * real_mask).sum(dim=2) / real_count
    diff = sub_anchor - center_anchor_new.unsqueeze(2)
    dist = diff.norm(dim=-1).masked_fill(inv_grouped, 0.0)
    radius_anchor_new = dist.max(dim=2).values

    width = e0 - s0
    return {
        "anchor_subspace": anchor_subspace,
        "keys_reord": keys_reord,
        "invalid_mask": invalid_mask,
        "reorder_perm": reorder_perm,
        "center_anchor_new": center_anchor_new.contiguous(),
        "radius_anchor_new": radius_anchor_new.contiguous(),
        "score": _subspace_tightness_score(radius_anchor_new, width),
    }


def build(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    anchor_subspace: int | None = None,
    values: torch.Tensor | None = None,
):
    h_kv, n_real, d = keys.shape
    k = max(1, math.ceil(n_real / bf))
    n_pad = k * bf
    pad = n_pad - n_real
    device = keys.device
    dtype = keys.dtype

    if pad > 0:
        zeros = torch.zeros(h_kv, pad, d, device=device, dtype=dtype)
        keys_padded = torch.cat([keys, zeros], dim=1)
    else:
        keys_padded = keys

    slices = _split_contiguous(d, n_subspaces)

    assigns_orig: list[torch.Tensor] = []
    centers_per_sub: list[torch.Tensor] = []
    radii_per_sub: list[torch.Tensor] = []
    for start, end in slices:
        keys_sub = keys[:, :, start:end].contiguous()
        assign, centers = _kcenter_subspace(keys_sub, k, refine_iter)
        radii = _ball_centroid(keys_sub, assign, centers, k)
        assigns_orig.append(assign)
        centers_per_sub.append(centers.contiguous())
        radii_per_sub.append(radii.contiguous())

    if anchor_subspace is None:
        candidates = [
            _anchor_candidate_layout(
                keys=keys,
                keys_padded=keys_padded,
                slices=slices,
                centers_per_sub=centers_per_sub,
                anchor_subspace=anchor_idx,
                bf=bf,
                n_real=n_real,
            )
            for anchor_idx in range(n_subspaces)
        ]
        anchor_layout = min(candidates, key=lambda item: item["score"])
    else:
        anchor_layout = _anchor_candidate_layout(
            keys=keys,
            keys_padded=keys_padded,
            slices=slices,
            centers_per_sub=centers_per_sub,
            anchor_subspace=int(anchor_subspace),
            bf=bf,
            n_real=n_real,
        )

    chosen_anchor = int(anchor_layout["anchor_subspace"])
    keys_reord = anchor_layout["keys_reord"]
    invalid_mask = anchor_layout["invalid_mask"]
    reorder_perm = anchor_layout["reorder_perm"]

    centers_per_sub = list(centers_per_sub)
    radii_per_sub = list(radii_per_sub)
    centers_per_sub[chosen_anchor] = anchor_layout["center_anchor_new"]
    radii_per_sub[chosen_anchor] = anchor_layout["radius_anchor_new"]

    assigns_reord_list: list[torch.Tensor] = []
    for assign_orig in assigns_orig:
        assign_padded = torch.zeros(h_kv, n_pad, dtype=torch.long, device=device)
        assign_padded[:, :n_real] = assign_orig
        assign_reord = assign_padded.gather(1, reorder_perm)
        assign_reord = assign_reord.masked_fill(invalid_mask, 0)
        assigns_reord_list.append(assign_reord.to(torch.int32).contiguous())

    non_anchor = []
    for subspace_idx, (start, end) in enumerate(slices):
        width = end - start
        if subspace_idx == chosen_anchor:
            continue
        non_anchor.append(
            (
                _subspace_tightness_score(radii_per_sub[subspace_idx], width),
                subspace_idx,
            )
        )
    ordered_subspaces = [chosen_anchor] + [subspace_idx for _, subspace_idx in sorted(non_anchor)]

    state = {
        "dim_slices": [slices[idx] for idx in ordered_subspaces],
        "centers": [centers_per_sub[idx] for idx in ordered_subspaces],
        "radii": [radii_per_sub[idx] for idx in ordered_subspaces],
        "assigns_reord": [assigns_reord_list[idx] for idx in ordered_subspaces],
        "keys_reord": keys_reord,
        "invalid_mask": invalid_mask,
        "reorder_perm": reorder_perm,
        "K": k,
        "N": n_real,
        "bf": bf,
        "N_pad": n_pad,
        "anchor_subspace": 0,
        "orig_anchor_subspace": chosen_anchor,
        "subspace_order": tuple(int(idx) for idx in ordered_subspaces),
        "subspace_tightness_scores": tuple(
            _subspace_tightness_score(
                radii_per_sub[idx],
                slices[idx][1] - slices[idx][0],
            )
            for idx in ordered_subspaces
        ),
    }

    if values is not None:
        _pack_values_into_state(state, values)

    return state


KERNEL = build
