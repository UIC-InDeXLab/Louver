"""build_v2.0 — per-subspace k-center + bf-block reorder by anchor subspace.

Spec from optimization.md / user request:
  - kcenter clustering ON EACH subspace (preserves tight per-subspace radii).
  - reorder children so that for the ANCHOR subspace, children of parent i
    sit at children[bf*i:bf*(i+1)] (no offsets table).
  - non-anchor subspaces' assignments are also re-indexed into the new
    physical order so the search kernel can look them up by physical position.

Search-time use:
  - Anchor cluster_pass gates entire bf-blocks.
  - For each surviving block, look up per-child non-anchor assigns and AND
    against non-anchor cluster_pass.

Output dict:
  - dim_slices, centers (list S), radii (list S), assigns_reord (list S of
    (H, N_pad) int32 — cluster id for child at physical position j),
  - keys_reord (H, N_pad, D), invalid_mask (H, N_pad bool),
  - reorder_perm (H, N_pad int64) — original index of physical position j,
  - K, N, bf, N_pad, anchor_subspace.
"""

from __future__ import annotations

import math

import numpy as np
import torch

KERNEL_VERSION = "v2.0"

ANCHOR_SUBSPACE = 0


def _split_contiguous(D: int, S: int):
    sub = D // S
    rem = D % S
    out, off = [], 0
    for s in range(S):
        d = sub + (1 if s < rem else 0)
        out.append((off, off + d))
        off += d
    return out


def _kcenter_subspace(keys_sub: torch.Tensor, K: int, refine_iter: int):
    """Same as build_v1's _kcenter_subspace."""
    H, N, d = keys_sub.shape
    device = keys_sub.device
    K = min(K, N)

    center_idx = torch.empty(H, K, dtype=torch.long, device=device)
    center_idx[:, 0] = torch.randint(0, N, (H,), device=device)
    first = keys_sub.gather(1, center_idx[:, :1, None].expand(-1, 1, d))
    min_dist = (keys_sub - first).norm(dim=-1)
    for i in range(1, K):
        farthest = min_dist.argmax(dim=1)
        center_idx[:, i] = farthest
        new_c = keys_sub.gather(1, farthest.view(H, 1, 1).expand(-1, 1, d))
        min_dist = torch.minimum(min_dist, (keys_sub - new_c).norm(dim=-1))
    centers = keys_sub.gather(1, center_idx[..., None].expand(-1, -1, d))

    ones_hn = torch.ones(H, N, device=device, dtype=keys_sub.dtype)
    for _ in range(refine_iter):
        dists = torch.cdist(keys_sub, centers)
        assign = dists.argmin(dim=2)
        new_centers = torch.zeros_like(centers)
        new_centers.scatter_add_(1, assign[..., None].expand(-1, -1, d), keys_sub)
        counts = torch.zeros(H, K, device=device, dtype=keys_sub.dtype)
        counts.scatter_add_(1, assign, ones_hn)
        empty = counts == 0
        counts = counts.clamp_min(1.0)
        new_centers = new_centers / counts.unsqueeze(-1)
        if empty.any():
            cur_d = torch.cdist(keys_sub, new_centers).min(dim=2).values
            for h in range(H):
                for k_idx in empty[h].nonzero(as_tuple=True)[0]:
                    far = cur_d[h].argmax()
                    new_centers[h, k_idx] = keys_sub[h, far]
                    cur_d[h, far] = 0.0
        centers = new_centers

    assign = torch.cdist(keys_sub, centers).argmin(dim=2)
    return assign, centers


def _ball_centroid(keys_sub, assign, centers, K):
    H, _, d = keys_sub.shape
    parent = centers.gather(1, assign[..., None].expand(-1, -1, d))
    dists = (keys_sub - parent).norm(dim=-1)
    radii = torch.zeros(H, K, device=keys_sub.device, dtype=keys_sub.dtype)
    radii.scatter_reduce_(1, assign, dists, reduce="amax", include_self=True)
    return radii


def _balanced_assign_per_head(d_h: np.ndarray, bf: int) -> np.ndarray:
    """Greedy capacity-balanced assignment. d_h: (N, K) distances."""
    N, K = d_h.shape
    cap = np.zeros(K, dtype=np.int32)
    assign = np.empty(N, dtype=np.int64)
    best_d = d_h.min(axis=1)
    order = np.argsort(best_d)
    ranked = np.argsort(d_h, axis=1)
    for p in order:
        for c in ranked[p]:
            if cap[c] < bf:
                assign[p] = c
                cap[c] += 1
                break
    return assign


def build(keys: torch.Tensor, bf: int, n_subspaces: int, refine_iter: int = 5,
          anchor_subspace: int = ANCHOR_SUBSPACE):
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    N_pad = K * bf
    pad = N_pad - N
    device = keys.device
    dtype = keys.dtype

    if pad > 0:
        zeros = torch.zeros(H, pad, D, device=device, dtype=dtype)
        keys_padded = torch.cat([keys, zeros], dim=1)
    else:
        keys_padded = keys

    slices = _split_contiguous(D, n_subspaces)

    # 1. Per-subspace k-center (on the real N points). Compute centers + radii
    #    from THIS subspace's clustering (same as build_v1) — they stay tight.
    assigns_orig: list[torch.Tensor] = []  # (H, N) per subspace
    centers_per_sub: list[torch.Tensor] = []
    radii_per_sub: list[torch.Tensor] = []
    for start, end in slices:
        keys_sub = keys[:, :, start:end].contiguous()
        a, c = _kcenter_subspace(keys_sub, K, refine_iter)
        r = _ball_centroid(keys_sub, a, c, K)
        assigns_orig.append(a)
        centers_per_sub.append(c.contiguous())
        radii_per_sub.append(r.contiguous())

    # 2. Balanced reassignment for the anchor subspace (capacity = bf).
    s0, e0 = slices[anchor_subspace]
    keys_anchor = keys_padded[:, :, s0:e0].contiguous()
    centers_anchor = centers_per_sub[anchor_subspace]
    dists_anchor = torch.cdist(keys_anchor, centers_anchor)  # (H, N_pad, K)
    dists_np = dists_anchor.cpu().numpy()
    bal_assign_np = np.empty((H, N_pad), dtype=np.int64)
    for h in range(H):
        bal_assign_np[h] = _balanced_assign_per_head(dists_np[h], bf)
    bal_assign = torch.from_numpy(bal_assign_np).to(device)

    # 3. Reorder physically so cluster c's children sit at [c*bf:(c+1)*bf).
    sort_order = torch.argsort(bal_assign, dim=1, stable=True)  # (H, N_pad)
    keys_reord = keys_padded.gather(1, sort_order[..., None].expand(-1, -1, D)).contiguous()

    src_idx = torch.arange(N_pad, device=device).expand(H, -1)
    invalid_src = src_idx >= N  # (H, N_pad)
    invalid_mask = invalid_src.gather(1, sort_order)  # (H, N_pad) — True for padded slots

    reorder_perm = sort_order.contiguous()  # (H, N_pad), original index of physical j

    # 4. (centers_per_sub / radii_per_sub already filled from per-subspace
    #     kcenter — same as build_v1; tight per-subspace radii are critical.)

    # 5. For the ANCHOR subspace, override centers/radii using the bf-block
    #    grouped children (since we re-clustered into balanced groups).
    keys_grouped = keys_reord.view(H, K, bf, D)
    inv_grouped = invalid_mask.view(H, K, bf)
    real_mask = (~inv_grouped).to(dtype).unsqueeze(-1)
    real_count = real_mask.sum(dim=2).clamp_min(1.0)
    s0, e0 = slices[anchor_subspace]
    sub_anchor = keys_grouped[..., s0:e0]
    center_anchor_new = (sub_anchor * real_mask).sum(dim=2) / real_count
    diff = sub_anchor - center_anchor_new.unsqueeze(2)
    dist = diff.norm(dim=-1).masked_fill(inv_grouped, 0.0)
    radius_anchor_new = dist.max(dim=2).values
    centers_per_sub[anchor_subspace] = center_anchor_new.contiguous()
    radii_per_sub[anchor_subspace] = radius_anchor_new.contiguous()

    # 6. Re-index per-subspace assigns into physical order.
    assigns_reord_list: list[torch.Tensor] = []
    for a_orig in assigns_orig:
        # a_orig: (H, N). Pad with 0s for padded src positions; we'll mask later.
        a_padded = torch.zeros(H, N_pad, dtype=torch.long, device=device)
        a_padded[:, :N] = a_orig
        # Gather using sort_order: physical position j -> original index reorder_perm[h,j]
        a_reord = a_padded.gather(1, reorder_perm)  # (H, N_pad)
        # For padded slots, set to 0 (any cluster — invalid_mask filters them).
        a_reord = a_reord.masked_fill(invalid_mask, 0)
        assigns_reord_list.append(a_reord.to(torch.int32).contiguous())

    return {
        "dim_slices": slices,
        "centers": centers_per_sub,           # used by search to compute cluster_pass
        "radii": radii_per_sub,
        "assigns_reord": assigns_reord_list,  # (S,) of (H, N_pad) int32
        "keys_reord": keys_reord,
        "invalid_mask": invalid_mask,
        "reorder_perm": reorder_perm,
        "K": K,
        "N": N,
        "bf": bf,
        "N_pad": N_pad,
        "anchor_subspace": anchor_subspace,
    }


KERNEL = build
