"""build_v2.6 — build_v2.5 with sampled anchor-selectivity scoring.

The v1.34+ anchor-only attention path lives or dies on the anchor subspace.
v2.6 chooses the anchor by estimating actual pass rate on sampled normalized
keys, rather than only using a radius-based tightness proxy.
"""

from __future__ import annotations

import math

import torch

from .build_v2_5 import _anchor_candidate_layout, _subspace_tightness_score
from .build_v2_0 import _ball_centroid, _kcenter_subspace, _split_contiguous
from .build_v2_4 import _pack_values_into_state

KERNEL_VERSION = "v2.6"


def _sampled_anchor_pass_rate(
    keys: torch.Tensor,
    slices: list[tuple[int, int]],
    candidate: dict,
    anchor_subspace: int,
    topk: int,
    max_queries: int,
) -> float:
    h_kv, n_real, _ = keys.shape
    start, end = slices[anchor_subspace]
    keys_sub = keys[:, :, start:end].contiguous()
    centers = candidate["center_anchor_new"]
    radii = candidate["radius_anchor_new"]

    n_q = min(max_queries, n_real)
    q_idx = torch.linspace(0, n_real - 1, steps=n_q, device=keys.device)
    q_idx = q_idx.round().to(torch.long).unique(sorted=True)
    q = keys.index_select(1, q_idx)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q_sub = q[:, :, start:end].contiguous()

    full_scores = torch.einsum("hqd,hnd->hqn", q, keys)
    topk_eff = min(topk, n_real)
    top_idx = full_scores.topk(topk_eff, dim=-1).indices

    sub_scores = torch.einsum("hqd,hnd->hqn", q_sub, keys_sub)
    sub_top = sub_scores.gather(2, top_idx)
    th = sub_top.min(dim=-1).values
    qn = q_sub.norm(dim=-1)

    ub = torch.einsum("hqd,hkd->hqk", q_sub, centers) + radii[:, None, :] * qn[:, :, None]
    passed = ub >= th[:, :, None]
    return float(passed.float().mean().item())


def build(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    anchor_subspace: int | None = None,
    values: torch.Tensor | None = None,
    anchor_probe_topk: int = 20,
    anchor_probe_queries: int = 64,
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
        candidates = []
        for anchor_idx in range(n_subspaces):
            candidate = _anchor_candidate_layout(
                keys=keys,
                keys_padded=keys_padded,
                slices=slices,
                centers_per_sub=centers_per_sub,
                anchor_subspace=anchor_idx,
                bf=bf,
                n_real=n_real,
            )
            candidate["pass_rate_score"] = _sampled_anchor_pass_rate(
                keys=keys,
                slices=slices,
                candidate=candidate,
                anchor_subspace=anchor_idx,
                topk=anchor_probe_topk,
                max_queries=anchor_probe_queries,
            )
            candidates.append(candidate)
        anchor_layout = min(
            candidates,
            key=lambda item: (item["pass_rate_score"], item["score"]),
        )
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
        anchor_layout["pass_rate_score"] = _sampled_anchor_pass_rate(
            keys=keys,
            slices=slices,
            candidate=anchor_layout,
            anchor_subspace=int(anchor_subspace),
            topk=anchor_probe_topk,
            max_queries=anchor_probe_queries,
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
        "anchor_pass_rate_score": float(anchor_layout["pass_rate_score"]),
        "anchor_probe_topk": int(anchor_probe_topk),
        "anchor_probe_queries": int(anchor_probe_queries),
    }

    if values is not None:
        _pack_values_into_state(state, values)

    return state


KERNEL = build
