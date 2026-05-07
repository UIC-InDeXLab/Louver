"""update_v1.0 — TA-filter incremental arena update.

Mirrors the design of ``benchmark_area/kernel_impl/kernels/update_v4_0.py``
but adapted for the TA-filter index produced by ``TA_build.build``:

    * S=4, bf=4, BUFFER=256 are hard constants.
    * Buffer is re-clustered (256 -> 64 parents per subspace via balanced
      PCA-tree, same recipe as TA_build) and the new clusters are appended
      to the pre-allocated arena tail [K_used : K_used + 64].
    * Touches every arena tensor that the filter / sparse-attn kernels
      depend on:
        - centers_padded_f16
        - assigns_padded
        - children_padded_i32
        - parent_children_i32
        - keys_padded_f16
        - values_padded_f16
        - invalid_mask
        - cluster_keys_t_f16
        - cluster_values_f16
        - _assigns_packed_u64_v34

The phase split (write data first, publish later) lets the caller run the
update on a side stream concurrently with attention.  ``apply_publish``
flips the invalid flags + bumps K_used / N_used; until that runs the
filter still sees the new range as invalid (sentinel-packed) and skips it.

The whole thing is implemented with PyTorch ops on the supplied stream
(clustering of 256 points x 4 subspaces is cheap; the dominant cost is
the per-H_kv balanced bisection done on CPU/GPU in TA_build).  A pure
.cu scatter kernel could replace the data writes for lower per-call
overhead, but is not on the critical path while running async.
"""
from __future__ import annotations

import math
from typing import Any

import torch

KERNEL_VERSION = "v1.0"

BF = 4
S = 4
BUFFER_SIZE = 256
K_BUF = BUFFER_SIZE // BF  # 64 new parents per subspace per flush


# ───────────────────────────────────────────────────────────────────
# Clustering (matches TA_build's balanced PCA tree)
# ───────────────────────────────────────────────────────────────────


def _dominant_axis(points: torch.Tensor) -> torch.Tensor:
    if points.shape[0] <= 1:
        axis = torch.zeros(points.shape[1], device=points.device, dtype=points.dtype)
        axis[0] = 1.0
        return axis
    var = points.float().var(dim=0, unbiased=False)
    axis = torch.zeros(points.shape[1], device=points.device, dtype=points.dtype)
    axis[int(var.argmax().item())] = 1.0
    return axis


def _assign_balanced_tree(
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
    _assign_balanced_tree(points, left_idx, left_sizes, assign, cluster_offset)
    _assign_balanced_tree(
        points, right_idx, right_sizes, assign, cluster_offset + mid
    )


def _cluster_subspace(
    keys_sub: torch.Tensor,  # (H_kv, B, w)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return assign (H_kv, B) int64 and centers (H_kv, K_BUF, w)."""
    h_kv, n, w = keys_sub.shape
    device = keys_sub.device
    target_sizes = torch.full((K_BUF,), BF, device=device, dtype=torch.long)
    assign = torch.empty(h_kv, n, device=device, dtype=torch.long)
    centers = torch.empty(h_kv, K_BUF, w, device=device, dtype=keys_sub.dtype)
    all_idx = torch.arange(n, device=device)
    for h in range(h_kv):
        a = torch.empty(n, device=device, dtype=torch.long)
        _assign_balanced_tree(keys_sub[h], all_idx, target_sizes, a, 0)
        assign[h] = a
        # center = mean of assigned points per cluster
        sums = torch.zeros(K_BUF, w, device=device, dtype=keys_sub.dtype)
        sums.scatter_add_(0, a[:, None].expand(n, w), keys_sub[h])
        counts = torch.bincount(a, minlength=K_BUF).to(keys_sub.dtype).clamp_min(1.0)
        centers[h] = sums / counts[:, None]
    return assign, centers


# ───────────────────────────────────────────────────────────────────
# Update (phase 1: write into arena tail; phase 2: publish flags)
# ───────────────────────────────────────────────────────────────────


def update_v1_0(
    state: dict[str, Any],
    buffer_keys: torch.Tensor,    # (H_kv, B, D) fp16
    buffer_values: torch.Tensor,  # (H_kv, B, D_v) fp16
) -> dict[str, Any]:
    """Phase 1 — scatter cluster data into arena tail. Returns a publish dict.

    Requires the arena to have at least K_BUF unused parent slots and
    BUFFER_SIZE unused child slots remaining (build with K_cap large enough).
    Caller is responsible for calling ``apply_publish`` after waiting on the
    update stream.
    """
    if buffer_keys.shape[1] != BUFFER_SIZE:
        raise ValueError(
            f"update_v1_0 requires exactly {BUFFER_SIZE} buffer keys; "
            f"got {buffer_keys.shape[1]}"
        )

    h_kv = int(state["centers_padded_f16"].shape[1])
    d = int(state["D"])
    d_v = int(state.get("D_v", buffer_values.shape[-1]))
    max_w = int(state["max_width"])
    k_used = int(state["K_used"])
    n_used = int(state["N_used"])
    k_cap = int(state["K_cap"])
    n_cap = int(state["N_pad"])

    if k_used + K_BUF > k_cap:
        raise RuntimeError(
            f"arena full: K_used={k_used} + K_BUF={K_BUF} > K_cap={k_cap}"
        )
    if n_used + BUFFER_SIZE > n_cap:
        raise RuntimeError(
            f"arena full: N_used={n_used} + B={BUFFER_SIZE} > N_pad={n_cap}"
        )

    dim_slices = state["dim_slices"]
    device = buffer_keys.device

    # Per-subspace clustering of buffer keys.
    new_centers_padded = torch.zeros(
        S, h_kv, K_BUF, max_w, device=device, dtype=torch.float16
    )
    # local cluster-id (in [0, K_BUF)) per (s, h_kv, b)
    new_assign_local = torch.empty(
        S, h_kv, BUFFER_SIZE, device=device, dtype=torch.int32
    )

    # Per-subspace cluster.
    for s_idx, (start, end) in enumerate(dim_slices):
        keys_sub = buffer_keys[:, :, start:end].contiguous()  # (H_kv, B, w)
        w = end - start
        assign_s, centers_s = _cluster_subspace(keys_sub.float())
        new_centers_padded[s_idx, :, :, :w] = centers_s.to(torch.float16)
        new_assign_local[s_idx] = assign_s.to(torch.int32)

    # Pre-compute the packed-assigns row for the buffer tail (to be
    # committed at publish; until then assigns_packed tail stays at the
    # invalid sentinel, so the filter ignores it).
    a64 = (new_assign_local.to(torch.int64) + k_used) & 0xFFFF
    packed_buf = (
        a64[0]
        | (a64[1] << 16)
        | (a64[2] << 32)
        | (a64[3] << 48)
    )  # (H_kv, B)

    # ── Phase 1 writes: data only, no publish ──
    centers_arena = state["centers_padded_f16"]
    keys_arena = state["keys_padded_f16"]
    values_arena = state["values_padded_f16"]
    assigns_arena = state["assigns_padded"]

    centers_arena[:, :, k_used:k_used + K_BUF, :].copy_(new_centers_padded)
    keys_arena[:, n_used:n_used + BUFFER_SIZE, :].copy_(buffer_keys)
    values_arena[:, n_used:n_used + BUFFER_SIZE, :].copy_(buffer_values)

    assigns_dtype = assigns_arena.dtype
    assigns_arena[:, :, n_used:n_used + BUFFER_SIZE].copy_(
        (new_assign_local.long() + k_used).to(assigns_dtype)
    )

    pending = {
        "state": state,
        "k_used_after": k_used + K_BUF,
        "n_used_after": n_used + BUFFER_SIZE,
        "n_used_before": n_used,
        "n_added": BUFFER_SIZE,
        "packed_buf": packed_buf.contiguous(),
    }
    return pending


def apply_publish(pending: dict[str, Any]) -> None:
    """Phase 2 — flip invalid flags / publish packed assigns / bump counters."""
    state = pending["state"]
    n0 = pending["n_used_before"]
    n_add = pending["n_added"]
    invalid_mask = state["invalid_mask"]
    packed = state["_assigns_packed_u64_v34"]

    invalid_mask[:, n0:n0 + n_add] = False
    packed[:, n0:n0 + n_add] = pending["packed_buf"]

    state["K_used"] = pending["k_used_after"]
    state["N_used"] = pending["n_used_after"]


KERNEL = update_v1_0
