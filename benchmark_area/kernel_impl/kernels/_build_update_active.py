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

import torch

import numpy as np

ANCHOR_SUBSPACE = 0

KERNEL_VERSION = "v2.5"


def _assign_dtype(k: int) -> torch.dtype:
    return torch.int16 if k < 32768 else torch.int32


def _split_contiguous(d: int, s_count: int) -> list[tuple[int, int]]:
    sub = d // s_count
    rem = d % s_count
    out, off = [], 0
    for idx in range(s_count):
        width = sub + (1 if idx < rem else 0)
        out.append((off, off + width))
        off += width
    return out


def _kcenter_subspace(keys_sub: torch.Tensor, k: int, refine_iter: int):
    h, n, d = keys_sub.shape
    device = keys_sub.device
    k = min(k, n)

    center_idx = torch.empty(h, k, dtype=torch.long, device=device)
    center_idx[:, 0] = torch.randint(0, n, (h,), device=device)
    first = keys_sub.gather(1, center_idx[:, :1, None].expand(-1, 1, d))
    min_dist = (keys_sub - first).norm(dim=-1)
    for idx in range(1, k):
        farthest = min_dist.argmax(dim=1)
        center_idx[:, idx] = farthest
        new_c = keys_sub.gather(1, farthest.view(h, 1, 1).expand(-1, 1, d))
        min_dist = torch.minimum(min_dist, (keys_sub - new_c).norm(dim=-1))
    centers = keys_sub.gather(1, center_idx[..., None].expand(-1, -1, d))

    ones_hn = torch.ones(h, n, device=device, dtype=keys_sub.dtype)
    for _ in range(refine_iter):
        dists = torch.cdist(keys_sub, centers)
        assign = dists.argmin(dim=2)
        new_centers = torch.zeros_like(centers)
        new_centers.scatter_add_(1, assign[..., None].expand(-1, -1, d), keys_sub)
        counts = torch.zeros(h, k, device=device, dtype=keys_sub.dtype)
        counts.scatter_add_(1, assign, ones_hn)
        empty = counts == 0
        counts = counts.clamp_min(1.0)
        new_centers = new_centers / counts.unsqueeze(-1)
        if empty.any():
            cur_d = torch.cdist(keys_sub, new_centers).min(dim=2).values
            for head in range(h):
                for k_idx in empty[head].nonzero(as_tuple=True)[0]:
                    far = cur_d[head].argmax()
                    new_centers[head, k_idx] = keys_sub[head, far]
                    cur_d[head, far] = 0.0
        centers = new_centers

    assign = torch.cdist(keys_sub, centers).argmin(dim=2)
    return assign, centers


def _ball_centroid(keys_sub, assign, centers, k):
    h, _, d = keys_sub.shape
    parent = centers.gather(1, assign[..., None].expand(-1, -1, d))
    dists = (keys_sub - parent).norm(dim=-1)
    radii = torch.zeros(h, k, device=keys_sub.device, dtype=keys_sub.dtype)
    radii.scatter_reduce_(1, assign, dists, reduce="amax", include_self=True)
    return radii


def _balanced_assign_per_head(d_h: np.ndarray, bf: int) -> np.ndarray:
    n, k = d_h.shape
    cap = np.zeros(k, dtype=np.int32)
    assign = np.empty(n, dtype=np.int64)
    best_d = d_h.min(axis=1)
    order = np.argsort(best_d)
    ranked = np.argsort(d_h, axis=1)
    for p_idx in order:
        for c_idx in ranked[p_idx]:
            if cap[c_idx] < bf:
                assign[p_idx] = c_idx
                cap[c_idx] += 1
                break
    return assign


def _pack_values_into_state(state: dict, values: torch.Tensor) -> None:
    reorder_perm: torch.Tensor = state["reorder_perm"]
    invalid_mask: torch.Tensor = state["invalid_mask"]
    h_kv, n_pad_state = reorder_perm.shape
    h_kv_v, n_raw, d_v = values.shape
    if h_kv != h_kv_v:
        raise ValueError(f"head mismatch: reorder={h_kv} vs values={h_kv_v}")

    pad = n_pad_state - n_raw
    if pad > 0:
        pad_zeros = torch.zeros(
            h_kv, pad, d_v, device=values.device, dtype=values.dtype
        )
        values_padded = torch.cat([values, pad_zeros], dim=1)
    elif pad == 0:
        values_padded = values
    else:
        raise ValueError(f"values has more rows ({n_raw}) than N_pad ({n_pad_state})")

    values_reord = values_padded.gather(
        1, reorder_perm[..., None].expand(-1, -1, d_v)
    ).contiguous()
    values_reord = values_reord.masked_fill(invalid_mask[..., None], 0.0)

    k = state["K"]
    bf = state["bf"]
    state["values_reord"] = values_reord
    state["values_blocks_f16"] = (
        values_reord.view(h_kv, k, bf, d_v).to(torch.float16).contiguous()
    )
    state["D_v"] = d_v


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


def _match_seed_centers(
    keys_sub: torch.Tensor,
    seed_centers: torch.Tensor | None,
    k_target: int,
) -> torch.Tensor | None:
    if seed_centers is None:
        return None

    h, _, d = keys_sub.shape
    if seed_centers.ndim != 3 or seed_centers.shape[0] != h or seed_centers.shape[-1] != d:
        return None

    centers = seed_centers.to(device=keys_sub.device, dtype=keys_sub.dtype).contiguous()
    k_seed = int(centers.shape[1])
    if k_seed == 0:
        return None
    if k_seed >= k_target:
        return centers[:, :k_target].contiguous()

    min_dist = torch.cdist(keys_sub, centers).min(dim=2).values
    extra: list[torch.Tensor] = []
    for _ in range(k_seed, k_target):
        farthest = min_dist.argmax(dim=1)
        new_c = keys_sub.gather(1, farthest.view(h, 1, 1).expand(-1, 1, d))
        extra.append(new_c)
        min_dist = torch.minimum(min_dist, (keys_sub - new_c).norm(dim=-1))
    return torch.cat([centers, *extra], dim=1).contiguous()


def _kcenter_subspace_seeded(
    keys_sub: torch.Tensor,
    k: int,
    refine_iter: int,
    seed_centers: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    h, n, d = keys_sub.shape
    k = min(k, n)
    centers = _match_seed_centers(keys_sub, seed_centers, k)
    if centers is None:
        return _kcenter_subspace(keys_sub, k, refine_iter)

    ones_hn = torch.ones(h, n, device=keys_sub.device, dtype=keys_sub.dtype)
    for _ in range(refine_iter):
        dists = torch.cdist(keys_sub, centers)
        assign = dists.argmin(dim=2)
        new_centers = torch.zeros_like(centers)
        new_centers.scatter_add_(1, assign[..., None].expand(-1, -1, d), keys_sub)
        counts = torch.zeros(h, k, device=keys_sub.device, dtype=keys_sub.dtype)
        counts.scatter_add_(1, assign, ones_hn)

        empty = counts == 0
        counts = counts.clamp_min(1.0)
        new_centers = new_centers / counts.unsqueeze(-1)
        if empty.any():
            cur_d = torch.cdist(keys_sub, new_centers).min(dim=2).values
            for head in range(h):
                for k_idx in empty[head].nonzero(as_tuple=True)[0]:
                    far = cur_d[head].argmax()
                    new_centers[head, k_idx] = keys_sub[head, far]
                    cur_d[head, far] = 0.0
        centers = new_centers

    assign = torch.cdist(keys_sub, centers).argmin(dim=2)
    return assign, centers


def _balanced_assign_gpu_rounds(dists: torch.Tensor, bf: int) -> torch.Tensor:
    h, n_pad, k = dists.shape
    device = dists.device

    ranked_idx = torch.argsort(dists, dim=2)
    point_best = dists.min(dim=2).values.reshape(-1)
    head_offsets = (torch.arange(h, device=device, dtype=torch.long) * k)[:, None]

    assigned = torch.full((h, n_pad), -1, device=device, dtype=torch.long)
    unassigned = torch.ones((h, n_pad), device=device, dtype=torch.bool)
    cap_used = torch.zeros(h * k, device=device, dtype=torch.int32)

    for rank_idx in range(k):
        active = unassigned.reshape(-1)
        if not active.any():
            break

        active_idx = active.nonzero(as_tuple=True)[0]
        proposal_gid = (
            ranked_idx[:, :, rank_idx] + head_offsets
        ).reshape(-1).index_select(0, active_idx)
        proposal_pri = point_best.index_select(0, active_idx)

        order_pri = torch.argsort(proposal_pri, stable=True)
        idx_pri = active_idx.index_select(0, order_pri)
        gid_pri = proposal_gid.index_select(0, order_pri)

        order_gid = torch.argsort(gid_pri, stable=True)
        idx_sorted = idx_pri.index_select(0, order_gid)
        gid_sorted = gid_pri.index_select(0, order_gid)

        pos = torch.arange(idx_sorted.numel(), device=device, dtype=torch.long)
        change = torch.ones_like(gid_sorted, dtype=torch.bool)
        if change.numel() > 1:
            change[1:] = gid_sorted[1:] != gid_sorted[:-1]
        group_start = torch.where(change, pos, torch.zeros_like(pos))
        group_start = torch.cummax(group_start, dim=0).values
        group_rank = pos - group_start

        avail = bf - cap_used.index_select(0, gid_sorted).to(torch.long)
        take = group_rank < avail
        if take.any():
            win_idx = idx_sorted[take]
            win_gid = gid_sorted[take]
            assigned.view(-1)[win_idx] = win_gid % k
            unassigned.view(-1)[win_idx] = False
            cap_used.scatter_add_(
                0,
                win_gid,
                torch.ones_like(win_gid, dtype=cap_used.dtype),
            )

    if unassigned.any():
        raise RuntimeError(
            f"gpu_rounds assignment left {int(unassigned.sum().item())} points unassigned"
        )
    return assigned


def build_v2_0_seeded_state(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    anchor_subspace: int = ANCHOR_SUBSPACE,
    seed_state: dict | None = None,
    max_refine_with_seed: int | None = None,
    balance_mode: str = "cpu",
) -> dict:
    h, n, d_total = keys.shape
    k = max(1, math.ceil(n / bf))
    n_pad = k * bf
    pad = n_pad - n
    device = keys.device
    dtype = keys.dtype

    if pad > 0:
        zeros = torch.zeros(h, pad, d_total, device=device, dtype=dtype)
        keys_padded = torch.cat([keys, zeros], dim=1)
    else:
        keys_padded = keys

    slices = _split_contiguous(d_total, n_subspaces)
    seed_centers_all = seed_state.get("centers") if isinstance(seed_state, dict) else None

    assigns_orig: list[torch.Tensor] = []
    centers_per_sub: list[torch.Tensor] = []
    radii_per_sub: list[torch.Tensor] = []
    for idx, (start, end) in enumerate(slices):
        keys_sub = keys[:, :, start:end].contiguous()
        seed_centers = None
        if isinstance(seed_centers_all, list) and idx < len(seed_centers_all):
            seed_centers = seed_centers_all[idx]
        local_refine = refine_iter
        if seed_centers is not None and max_refine_with_seed is not None:
            local_refine = min(refine_iter, max_refine_with_seed)
        a, c = _kcenter_subspace_seeded(keys_sub, k, local_refine, seed_centers)
        r = _ball_centroid(keys_sub, a, c, k)
        assigns_orig.append(a)
        centers_per_sub.append(c.contiguous())
        radii_per_sub.append(r.contiguous())

    s0, e0 = slices[anchor_subspace]
    keys_anchor = keys_padded[:, :, s0:e0].contiguous()
    centers_anchor = centers_per_sub[anchor_subspace]
    dists_anchor = torch.cdist(keys_anchor, centers_anchor)
    if balance_mode == "cpu":
        dists_np = dists_anchor.cpu().numpy()
        bal_assign_np = np.empty((h, n_pad), dtype=np.int64)
        for head in range(h):
            bal_assign_np[head] = _balanced_assign_per_head(dists_np[head], bf)
        bal_assign = torch.from_numpy(bal_assign_np).to(device=device)
    elif balance_mode == "gpu_rounds":
        bal_assign = _balanced_assign_gpu_rounds(dists_anchor, bf)
    else:
        raise ValueError(f"Unknown balance_mode: {balance_mode!r}")

    sort_order = torch.argsort(bal_assign, dim=1, stable=True)
    keys_reord = keys_padded.gather(1, sort_order[..., None].expand(-1, -1, d_total)).contiguous()

    src_idx = torch.arange(n_pad, device=device).expand(h, -1)
    invalid_src = src_idx >= n
    invalid_mask = invalid_src.gather(1, sort_order)
    reorder_perm = sort_order.contiguous()

    keys_grouped = keys_reord.view(h, k, bf, d_total)
    inv_grouped = invalid_mask.view(h, k, bf)
    real_mask = (~inv_grouped).to(dtype).unsqueeze(-1)
    real_count = real_mask.sum(dim=2).clamp_min(1.0)
    sub_anchor = keys_grouped[..., s0:e0]
    center_anchor_new = (sub_anchor * real_mask).sum(dim=2) / real_count
    diff = sub_anchor - center_anchor_new.unsqueeze(2)
    dist = diff.norm(dim=-1).masked_fill(inv_grouped, 0.0)
    radius_anchor_new = dist.max(dim=2).values
    centers_per_sub[anchor_subspace] = center_anchor_new.contiguous()
    radii_per_sub[anchor_subspace] = radius_anchor_new.contiguous()

    assigns_reord_list: list[torch.Tensor] = []
    for a_orig in assigns_orig:
        a_padded = torch.zeros(h, n_pad, dtype=torch.long, device=device)
        a_padded[:, :n] = a_orig
        a_reord = a_padded.gather(1, reorder_perm)
        a_reord = a_reord.masked_fill(invalid_mask, 0)
        assigns_reord_list.append(a_reord.to(torch.int32).contiguous())

    return {
        "dim_slices": slices,
        "centers": centers_per_sub,
        "radii": radii_per_sub,
        "assigns_reord": assigns_reord_list,
        "keys_reord": keys_reord,
        "invalid_mask": invalid_mask,
        "reorder_perm": reorder_perm,
        "K": k,
        "N": n,
        "bf": bf,
        "N_pad": n_pad,
        "anchor_subspace": anchor_subspace,
    }


def build_v2_4_state(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    anchor_subspace: int = ANCHOR_SUBSPACE,
    values: torch.Tensor | None = None,
    balance_mode: str = "cpu",
) -> dict:
    state = build_v2_0_seeded_state(
        keys=keys,
        bf=bf,
        n_subspaces=n_subspaces,
        refine_iter=refine_iter,
        anchor_subspace=anchor_subspace,
        balance_mode=balance_mode,
    )
    keys_reord = state["keys_reord"]
    invalid_mask = state["invalid_mask"]
    assigns_reord = state["assigns_reord"]
    h_kv, _, d = keys_reord.shape
    k = state["K"]
    s = len(assigns_reord)
    state["keys_blocks_t"] = (
        keys_reord.view(h_kv, k, bf, d).permute(0, 1, 3, 2).contiguous()
    )
    state["assigns_blocks"] = (
        torch.stack(assigns_reord, dim=0)
        .to(_assign_dtype(k))
        .view(s, h_kv, k, bf)
        .contiguous()
    )
    state["invalid_blocks_i8"] = invalid_mask.view(h_kv, k, bf).to(torch.int8).contiguous()
    if values is not None:
        _pack_values_into_state(state, values)
    return state
