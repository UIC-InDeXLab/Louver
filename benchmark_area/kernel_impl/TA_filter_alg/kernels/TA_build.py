"""TA_build — single-file build pipeline (v13.0) and shared query helpers.

Inlines (in order):
  commons/_TA_build_legacy.py  — clustering / layout builders
  commons/_TA_common.py        — centroid scoring, depth, candidate mask
  v11/TA_build_v_11_0.py      — v11 builder (cluster-streaming layouts)
  v13/TA_build_v_13_0.py      — v13 builder (parent_children layout, top-level)

Public API used by bench_ta_filtering.py:
  build(keys, bf, n_subspaces, ...) -> state dict   (was TA_build_v_13_0.build)
  compute_centroid_scores(...)
  stop_depth_per_head(...)
  build_selected_clusters(...)
  per_key_candidate_mask(...)
"""
from __future__ import annotations

import math

import torch

KERNEL_VERSION = "v13.0"

# ──────────────────────────────────────────────────────────────────────────────
# Clustering / layout helpers (inlined from commons/_TA_build_legacy.py)
# ──────────────────────────────────────────────────────────────────────────────

def split_contiguous(d: int, s_count: int) -> list[tuple[int, int]]:
    sub = d // s_count
    rem = d % s_count
    out: list[tuple[int, int]] = []
    off = 0
    for idx in range(s_count):
        width = sub + (1 if idx < rem else 0)
        out.append((off, off + width))
        off += width
    return out


def assigns_dtype(k: int) -> torch.dtype:
    return torch.int16 if k < 32768 else torch.int32


def _dominant_axis(points: torch.Tensor) -> torch.Tensor:
    _, d = points.shape
    if points.shape[0] <= 1:
        axis = torch.zeros(d, device=points.device, dtype=points.dtype)
        axis[0] = 1.0
        return axis
    var = points.float().var(dim=0, unbiased=False)
    axis_idx = int(var.argmax().item())
    axis = torch.zeros(d, device=points.device, dtype=points.dtype)
    axis[axis_idx] = 1.0
    return axis


def _target_cluster_sizes(n_points: int, bf: int, device: torch.device) -> torch.Tensor:
    k = max(1, math.ceil(n_points / bf))
    base = n_points // k
    extra = n_points % k
    sizes = torch.full((k,), base, device=device, dtype=torch.long)
    if extra:
        sizes[:extra] += 1
    return sizes


def _assign_balanced_tree_h(
    points: torch.Tensor,
    indices: torch.Tensor,
    leaf_sizes: torch.Tensor,
    assign: torch.Tensor,
    cluster_offset: int,
) -> None:
    if int(leaf_sizes.numel()) == 1:
        assign[indices] = cluster_offset
        return
    mid = int(leaf_sizes.numel()) // 2
    left_sizes = leaf_sizes[:mid]
    right_sizes = leaf_sizes[mid:]
    left_count = int(left_sizes.sum().item())
    subset = points.index_select(0, indices)
    axis = _dominant_axis(subset)
    order = torch.argsort(subset @ axis)
    left_idx = indices.index_select(0, order[:left_count])
    right_idx = indices.index_select(0, order[left_count:])
    _assign_balanced_tree_h(points, left_idx, left_sizes, assign, cluster_offset)
    _assign_balanced_tree_h(points, right_idx, right_sizes, assign, cluster_offset + mid)


def _recompute_centers(keys_h: torch.Tensor, assign_h: torch.Tensor, k: int) -> torch.Tensor:
    n, d = keys_h.shape
    centers = torch.zeros(k, d, device=keys_h.device, dtype=keys_h.dtype)
    centers.scatter_add_(0, assign_h[:, None].expand(n, d), keys_h)
    counts = torch.bincount(assign_h, minlength=k).to(keys_h.dtype).clamp_min(1.0)
    return centers / counts[:, None]


def _balanced_pca_tree_subspace(
    keys_sub: torch.Tensor, bf: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized balanced KD-tree clustering over all heads simultaneously.

    Replaces the original O(K*H) Python recursion (with GPU-CPU syncs at every
    node) with O(log K) Python iterations, each doing fully-batched GPU ops over
    all heads and clusters at once.

    Same algorithm as before (max-variance axis-aligned binary split), same
    output shapes. Results are equivalent but not bit-identical when n is not
    an exact multiple of bf (cluster boundaries shift slightly for the padded
    tail points — irrelevant for real keys).
    """
    h, n, d = keys_sub.shape
    device = keys_sub.device
    dtype = keys_sub.dtype
    k = max(1, math.ceil(n / bf))

    if k == 1:
        centers = keys_sub.float().mean(dim=1).to(dtype).unsqueeze(1)  # (h, 1, d)
        return torch.zeros(h, n, device=device, dtype=torch.long), centers

    # Round k up to the next power of two so every level splits cleanly.
    # Padding points (filled with +inf) sort to the end at every split and end
    # up in the "extra" tail clusters that contain no real keys.
    k_pow2 = 1 << (k - 1).bit_length()   # smallest power-of-2 >= k
    n_pad  = k_pow2 * bf                  # total slots (>= n)

    keys_f32 = keys_sub.float()           # (h, n, d) — fp32 for numerics

    # Pad real keys; tail slots become +inf so they sort last on every axis.
    if n_pad > n:
        pad = torch.full((h, n_pad - n, d), float('inf'), device=device, dtype=torch.float32)
        kp  = torch.cat([keys_f32, pad], dim=1)   # (h, n_pad, d)
    else:
        kp = keys_f32

    # perm[hi, i] = original key index currently at slot i for head hi.
    # Starts as identity; we reorder it via in-place scatter at each level.
    perm = torch.arange(n_pad, device=device, dtype=torch.long) \
               .unsqueeze(0).expand(h, -1).contiguous()          # (h, n_pad)

    n_c = 1                               # number of live clusters
    while n_c < k_pow2:
        cs = n_pad // n_c                 # current cluster size (exact: n_pad = k_pow2 * bf)

        # ── gather current sorted order ───────────────────────────────────
        # pts[hi, c, i, :] = kp[hi, perm[hi, c*cs + i], :]
        pts = kp.gather(
            1,
            perm.unsqueeze(-1).expand(-1, -1, d)
        ).reshape(h, n_c, cs, d)          # (h, n_c, cs, d)

        # ── real-point mask ───────────────────────────────────────────────
        real = perm.reshape(h, n_c, cs) < n   # (h, n_c, cs)  True = real key

        # ── variance over real points per cluster ─────────────────────────
        cnt   = real.float().sum(dim=2, keepdim=True).clamp_min_(1)  # (h, n_c, 1)
        real4 = real.unsqueeze(-1)         # (h, n_c, cs, 1) broadcasts with (h,n_c,cs,d)
        pts_m = pts.masked_fill(~real4, 0.0)
        # cnt is (h,n_c,1); sum is (h,n_c,1,d) → need cnt as (h,n_c,1,1) to divide correctly
        mu    = pts_m.sum(dim=2, keepdim=True) / cnt.unsqueeze(-1)   # (h, n_c, 1, d)
        diff  = pts_m - mu              # (h, n_c, cs, d) via broadcast
        diff  = diff.masked_fill(~real4, 0.0)
        var   = (diff * diff).sum(dim=2) / cnt                       # (h, n_c, d)

        # ── split axis (no .item()) ───────────────────────────────────────
        ax = var.argmax(dim=-1)                                       # (h, n_c)

        # ── coordinates along split axis ──────────────────────────────────
        coord = pts.gather(
            3,
            ax.unsqueeze(-1).unsqueeze(-1).expand(h, n_c, cs, 1)
        ).squeeze(-1)                                                 # (h, n_c, cs)
        coord[~real] = float('inf')        # padding always sorts last

        # ── sort within each cluster ──────────────────────────────────────
        order = coord.argsort(dim=2, stable=True)                    # (h, n_c, cs)

        # ── update perm ───────────────────────────────────────────────────
        perm = perm.reshape(h, n_c, cs).gather(2, order).reshape(h, n_pad)

        n_c *= 2

    # ── assign: slot j → cluster j // bf ─────────────────────────────────
    slot_ids = torch.arange(n_pad, device=device, dtype=torch.long) // bf   # (n_pad,)
    assign_all = torch.empty(h, n_pad, device=device, dtype=torch.long)
    assign_all.scatter_(1, perm, slot_ids.unsqueeze(0).expand(h, -1))
    assign = assign_all[:, :n]            # (h, n) — only real keys

    # ── cluster centers ───────────────────────────────────────────────────
    centers = torch.zeros(h, k, d, device=device, dtype=torch.float32)
    centers.scatter_add_(
        1,
        assign.unsqueeze(-1).expand(-1, -1, d),
        keys_f32
    )
    counts = torch.zeros(h, k, device=device, dtype=torch.float32).scatter_add_(
        1, assign, torch.ones(h, n, device=device)
    ).clamp_min_(1).unsqueeze(-1)
    centers = (centers / counts).to(dtype)   # (h, k, d)

    return assign, centers


def _children_from_assign(
    assign: torch.Tensor,
    *,
    k: int,
    bf: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    h, n = assign.shape
    device = assign.device
    children = torch.full((h, k, bf), -1, device=device, dtype=torch.int32)
    counts_out = torch.zeros(h, k, device=device, dtype=torch.int16)
    for head in range(h):
        assign_h = assign[head].to(torch.long)
        counts = torch.bincount(assign_h, minlength=k)
        if bool((counts > bf).any().item()):
            max_count = int(counts.max().item())
            raise RuntimeError(f"capacity violation: max cluster size {max_count} > bf={bf}")
        counts_out[head] = counts.to(torch.int16)
        order = torch.argsort(assign_h, stable=True)
        offsets = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.long), counts.cumsum(dim=0)]
        )
        for cluster in range(k):
            count = int(counts[cluster].item())
            if count:
                start = int(offsets[cluster].item())
                children[head, cluster, :count] = order[start : start + count].to(torch.int32)
    return children.contiguous(), counts_out.contiguous()


def build_v1_1_state(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    values: torch.Tensor | None = None,
) -> dict:
    del refine_iter
    if keys.ndim != 3:
        raise ValueError(f"keys must be (H_kv, N, D); got shape {tuple(keys.shape)}")
    if bf <= 0:
        raise ValueError(f"bf must be positive; got {bf}")

    h_kv, n_real, d = keys.shape
    k = max(1, math.ceil(n_real / bf))
    n_pad = k * bf
    pad = n_pad - n_real
    device = keys.device
    dtype = keys.dtype

    if pad:
        keys_padded = torch.cat(
            [keys, torch.zeros(h_kv, pad, d, device=device, dtype=dtype)], dim=1
        )
    else:
        keys_padded = keys

    invalid_mask = torch.zeros(h_kv, n_pad, dtype=torch.bool, device=device)
    if pad:
        invalid_mask[:, n_real:] = True

    slices = split_contiguous(d, n_subspaces)
    widths = [end - start for start, end in slices]
    offsets = [start for start, _ in slices]
    max_w = max(widths)

    # Slim build: only the artefacts that ta_filter v8 + sparse_attn v2.4 +
    # the incremental update kernel actually read.  Dropped:
    #   children_padded_i32, cluster_counts_i16, centers_per_sub,
    #   keys_padded_t_f16  (unused by the live pipeline; archived TA
    #   attention kernels referenced them).
    assigns_padded_list: list[torch.Tensor] = []
    centers_padded = torch.zeros(
        n_subspaces, h_kv, k, max_w, device=device, dtype=torch.float16
    )
    for s_idx, (start, end) in enumerate(slices):
        keys_sub = keys[:, :, start:end].contiguous()
        assign, centers = _balanced_pca_tree_subspace(keys_sub, bf)
        centers_padded[s_idx, :, :, : centers.shape[-1]] = centers.to(torch.float16)

        ap = torch.zeros(h_kv, n_pad, dtype=torch.long, device=device)
        ap[:, :n_real] = assign
        assigns_padded_list.append(ap.to(assigns_dtype(k)).contiguous())

    centers_padded = centers_padded.contiguous()
    keys_padded_f16 = keys_padded.to(torch.float16).contiguous()
    assigns_stack = torch.stack(assigns_padded_list, dim=0).contiguous()

    state = {
        "version": "v1.1",
        "dim_slices": slices,
        "dim_offsets": torch.tensor(offsets, dtype=torch.int32, device=device),
        "dim_widths": torch.tensor(widths, dtype=torch.int32, device=device),
        "max_width": int(max_w),
        "centers_padded_f16": centers_padded,
        "assigns_padded": assigns_stack,
        "keys_padded_f16": keys_padded_f16,
        "invalid_mask": invalid_mask.contiguous(),
        "K": k,
        "N": n_real,
        "N_pad": n_pad,
        "bf": bf,
        "D": d,
        "n_subspaces": n_subspaces,
    }

    if values is not None:
        if values.shape[0] != h_kv or values.shape[1] != n_real:
            raise ValueError(
                f"values shape mismatch: expected ({h_kv}, {n_real}, *); got {tuple(values.shape)}"
            )
        d_v = int(values.shape[-1])
        if pad:
            values_padded = torch.cat(
                [values, torch.zeros(h_kv, pad, d_v, device=device, dtype=values.dtype)],
                dim=1,
            )
        else:
            values_padded = values
        values_padded = values_padded.masked_fill(invalid_mask[..., None], 0.0)
        state["values_padded_f16"] = values_padded.to(torch.float16).contiguous()
        state["D_v"] = d_v

    return state


def add_v10_layouts(state: dict) -> None:
    """Deprecated — kept as a no-op for archived TA attention kernels."""
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Shared query helpers (inlined from commons/_TA_common.py)
# ──────────────────────────────────────────────────────────────────────────────

_Q_PACK_CACHE: dict = {}


def _q_subspace_pack_indices(
    dim_slices: list[tuple[int, int]],
    max_w: int,
    d: int,
    device: torch.device,
) -> torch.Tensor:
    key = (id(dim_slices), tuple(dim_slices), max_w, d, device)
    cached = _Q_PACK_CACHE.get(key)
    if cached is not None:
        return cached
    s_count = len(dim_slices)
    idx = torch.full((s_count, max_w), d, dtype=torch.long, device=device)
    for s_idx, (start, end) in enumerate(dim_slices):
        w = end - start
        idx[s_idx, :w] = torch.arange(start, end, device=device)
    _Q_PACK_CACHE[key] = idx
    return idx


def compute_centroid_scores(
    q: torch.Tensor,
    centers_padded_f16: torch.Tensor,
    dim_slices: list[tuple[int, int]],
    q_head_to_kv: torch.Tensor | None,
) -> torch.Tensor:
    s_count, h_kv, k, max_w = centers_padded_f16.shape
    h_q, d = q.shape
    device = q.device

    pad_idx = _q_subspace_pack_indices(dim_slices, max_w, d, device)
    q_padded = torch.cat(
        [q.float(), torch.zeros(h_q, 1, device=device, dtype=torch.float32)], dim=1
    )
    q_packed = q_padded.index_select(1, pad_idx.view(-1)).view(h_q, s_count, max_w)

    if q_head_to_kv is None:
        centers_eff = centers_padded_f16.float()
        out = torch.einsum("shkw,hsw->hsk", centers_eff, q_packed)
    else:
        centers_eff = centers_padded_f16.index_select(1, q_head_to_kv).float()
        out = torch.einsum("shkw,hsw->hsk", centers_eff, q_packed)
    return out.contiguous()


def stop_depth_per_head(
    sorted_scores: torch.Tensor, threshold: torch.Tensor
) -> torch.Tensor:
    h_q, _s, k = sorted_scores.shape
    row_sums = sorted_scores.sum(dim=1)
    below = row_sums < threshold.unsqueeze(-1)
    has = below.any(dim=-1)
    first = below.float().argmax(dim=-1)
    depth = torch.where(has, first + 1, torch.full_like(first, k))
    return depth


def build_selected_clusters(
    order: torch.Tensor, depth: torch.Tensor
) -> torch.Tensor:
    h_q, s_count, k = order.shape
    device = order.device
    rank_pos = torch.arange(k, device=device).view(1, 1, k)
    in_top = rank_pos < depth.view(h_q, 1, 1)
    in_top_b = in_top.expand(h_q, s_count, k).contiguous()
    selected = torch.zeros(h_q, s_count, k, dtype=torch.bool, device=device)
    selected.scatter_(2, order, in_top_b)
    return selected


def per_key_candidate_mask(
    selected: torch.Tensor,
    assigns_padded: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
) -> torch.Tensor:
    s_count, h_kv, n_pad = assigns_padded.shape
    device = selected.device
    h_q = int(selected.shape[0])

    if q_head_to_kv is None:
        assigns_eff = assigns_padded
    else:
        assigns_eff = assigns_padded.index_select(1, q_head_to_kv).contiguous()

    cand = torch.zeros(h_q, n_pad, dtype=torch.bool, device=device)
    for s_idx in range(s_count):
        parents = assigns_eff[s_idx].to(torch.int64)
        sel_s = selected[:, s_idx, :]
        passed = sel_s.gather(1, parents)
        cand |= passed
    return cand


# ──────────────────────────────────────────────────────────────────────────────
# v11 builder (inlined from v11/TA_build_v_11_0.py)
# ──────────────────────────────────────────────────────────────────────────────

def _build_v11(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    values: torch.Tensor | None = None,
) -> dict:
    if bf != 4 or n_subspaces != 4:
        raise ValueError(
            f"TA_build_v11.0 is specialized for bf=4 and n_subspaces=4; "
            f"got bf={bf}, n_subspaces={n_subspaces}"
        )
    state = build_v1_1_state(
        keys=keys,
        bf=bf,
        n_subspaces=n_subspaces,
        refine_iter=refine_iter,
        values=values,
    )
    add_v10_layouts(state)
    state["version"] = "v11.0"
    return state


# ──────────────────────────────────────────────────────────────────────────────
# v13 builder — top-level public API (inlined from v13/TA_build_v_13_0.py)
# ──────────────────────────────────────────────────────────────────────────────

def _add_v13_layouts(state: dict) -> None:
    """Deprecated — no longer needed (parent_children_i32 was used by archived
    TA attention kernels only). Kept as no-op for compatibility."""
    return None


def build(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    values: torch.Tensor | None = None,
) -> dict:
    state = _build_v11(
        keys=keys,
        bf=bf,
        n_subspaces=n_subspaces,
        refine_iter=refine_iter,
        values=values,
    )
    _add_v13_layouts(state)
    state["version"] = KERNEL_VERSION
    return state


KERNEL = build
