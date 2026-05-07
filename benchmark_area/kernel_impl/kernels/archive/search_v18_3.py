"""search_v18.3 — v18.1 plus anchor-parent compaction after cluster_pass.

This variant keeps v18.1's persistent workspace / optional CUDA Graphs, but:
  1. Prefills output to -inf once per query.
  2. Compacts parents whose anchor block survives for at least one group.
  3. Launches the dot kernel only over that compact parent list.

The math matches v15/v18.1; only the traversal differs.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from ._search_triton import triton_fused_cluster_pass_out
from ._search_utils import buffer_dot
from .search_v15_0 import (
    _PARENTS_PER_PROG,
    _get_layout_v15,
)

KERNEL_VERSION = "v18.3"


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _pack_q_into(q: torch.Tensor, layout: dict, out: torch.Tensor) -> None:
    h_q = q.shape[0]
    s = layout["num_subspaces"]
    max_d = layout["max_d"]
    d = q.shape[1]

    if d == s * max_d:
        out.copy_(q.view(h_q, s, max_d).transpose(0, 1))
        return

    out.zero_()
    for si, (s0, e0) in enumerate(layout["dim_slices"]):
        out[si, :, : e0 - s0].copy_(q[:, s0:e0])


def _compact_anchor_parents_out(
    cluster_pass: torch.Tensor,
    invalid_blocks: torch.Tensor,
    anchor_s: int,
    groups: int,
    compact_ids_out: torch.Tensor,
    counts_out: torch.Tensor,
) -> None:
    h_kv, k = invalid_blocks.shape[:2]
    anchor_any = cluster_pass[anchor_s].view(h_kv, groups, k).any(dim=1)
    all_invalid = (invalid_blocks != 0).all(dim=-1)
    survive = anchor_any & ~all_invalid

    counts_out.copy_(survive.sum(dim=1).to(torch.int32))
    compact_ids_out.copy_(
        survive.to(torch.int32).argsort(dim=1, descending=True, stable=True).to(torch.int32)
    )


if HAS_TRITON:

    @triton.jit
    def _compact_sparse_fp16_kernel(
        Q_ptr,              # (H_q, D) f32
        KeysBlocksT_ptr,    # (H_kv, K, D, BF) f16
        AssignsBlocks_ptr,  # (S, H_kv, K, BF)
        ClusterPass_ptr,    # (S, H_q, K) int8
        InvalidBlocks_ptr,  # (H_kv, K, BF) int8
        CompactIds_ptr,     # (H_kv, MAX_SURVIVE) int32
        Counts_ptr,         # (H_kv,) int32
        Out_ptr,            # (H_q, N_pad) f32, prefilled to -inf
        H_Q,
        H_KV,
        K,
        N_PAD,
        ANCHOR_S: tl.constexpr,
        D: tl.constexpr,
        BF: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        S: tl.constexpr,
        PARENTS_PER_PROG: tl.constexpr,
        MAX_SURVIVE: tl.constexpr,
    ):
        kvh = tl.program_id(0)
        parent_block = tl.program_id(1)

        block_start = parent_block * PARENTS_PER_PROG
        count = tl.load(Counts_ptr + kvh)
        if block_start >= count:
            return

        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        hq_vec = kvh * GROUPS + g_range

        cols = tl.arange(0, PARENTS_PER_PROG * BF)
        parent_rel = cols // BF
        child_rel = cols % BF

        compact_pos = block_start + parent_rel
        col_valid = compact_pos < count
        compact_safe = tl.where(col_valid, compact_pos, 0)
        parent_idx = tl.load(
            CompactIds_ptr + kvh * MAX_SURVIVE + compact_safe,
            mask=col_valid,
            other=0,
        ).to(tl.int32)
        parent_idx_safe = tl.where(col_valid, parent_idx, 0)
        child_idx = parent_idx_safe * BF + child_rel

        out_offs = hq_vec[:, None] * N_PAD + child_idx[None, :]
        out_mask = g_valid[:, None] & col_valid[None, :]

        anchor_pass = tl.load(
            ClusterPass_ptr + (ANCHOR_S * H_Q + hq_vec[:, None]) * K + parent_idx_safe[None, :],
            mask=out_mask,
            other=0,
        )
        survive = (anchor_pass != 0) & out_mask

        inv = tl.load(
            InvalidBlocks_ptr + ((kvh * K + parent_idx_safe) * BF + child_rel),
            mask=col_valid,
            other=1,
        )
        survive = survive & (inv[None, :] == 0)

        for s in tl.static_range(0, S):
            if s != ANCHOR_S:
                assign = tl.load(
                    AssignsBlocks_ptr
                    + ((s * H_KV + kvh) * K + parent_idx_safe) * BF
                    + child_rel,
                    mask=col_valid,
                    other=0,
                ).to(tl.int32)
                passed = tl.load(
                    ClusterPass_ptr + (s * H_Q + hq_vec[:, None]) * K + assign[None, :],
                    mask=survive,
                    other=0,
                )
                survive = survive & (passed != 0)

        live_cols = tl.max(survive.to(tl.int32), axis=0) != 0
        if tl.max(live_cols.to(tl.int32), axis=0) == 0:
            return

        d_range = tl.arange(0, D)
        q_full_f32 = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        q_full = q_full_f32.to(tl.float16)

        keys_tile = tl.load(
            KeysBlocksT_ptr
            + ((kvh * K + parent_idx_safe[None, :]) * D + d_range[:, None]) * BF
            + child_rel[None, :],
            mask=live_cols[None, :],
            other=0.0,
        )
        acc = tl.dot(q_full, keys_tile)

        tl.store(Out_ptr + out_offs, acc, mask=survive)


def _make_workspace(layout: dict, q: torch.Tensor, th: torch.Tensor) -> dict:
    s = layout["num_subspaces"]
    h_q = q.shape[0]
    max_d = layout["max_d"]
    k = layout["K"]
    n_pad = layout["N_pad"]
    device = q.device
    h_kv = layout["base_heads"]

    return {
        "static_q": torch.empty_like(q),
        "static_th": torch.empty(s, h_q, device=device, dtype=th.dtype),
        "q_packed": torch.empty(s, h_q, max_d, device=device, dtype=q.dtype),
        "q_norm": torch.empty(s, h_q, device=device, dtype=q.dtype),
        "cluster_pass": torch.empty(s, h_q, k, device=device, dtype=torch.int8),
        "compact_ids": torch.empty(h_kv, k, device=device, dtype=torch.int32),
        "counts": torch.empty(h_kv, device=device, dtype=torch.int32),
        "out": torch.empty(h_q, n_pad, device=device, dtype=torch.float32),
        "graph": None,
        "capture_failed": False,
    }


def _prepare_inputs(work: dict, layout: dict) -> None:
    _pack_q_into(work["static_q"], layout, work["q_packed"])
    torch.linalg.vector_norm(work["q_packed"], dim=-1, out=work["q_norm"])


def _launch_kernels(work: dict, layout: dict, h_q: int, d: int) -> None:
    work["out"].fill_(float("-inf"))

    triton_fused_cluster_pass_out(
        q_packed=work["q_packed"],
        q_norm=work["q_norm"],
        th=work["static_th"],
        centers=layout["centers"],
        radii=layout["radii"],
        groups=layout["groups"],
        out=work["cluster_pass"],
    )

    _compact_anchor_parents_out(
        cluster_pass=work["cluster_pass"],
        invalid_blocks=layout["invalid_blocks_i8"],
        anchor_s=layout["anchor_subspace"],
        groups=layout["groups"],
        compact_ids_out=work["compact_ids"],
        counts_out=work["counts"],
    )

    h_kv = layout["base_heads"]
    k = layout["K"]
    bf = layout["bf"]
    n_pad = layout["N_pad"]
    groups = layout["groups"]
    groups_pow = max(_next_pow2(groups), 8)
    anchor_s = layout["anchor_subspace"]
    grid = (h_kv, triton.cdiv(k, _PARENTS_PER_PROG))

    _compact_sparse_fp16_kernel[grid](
        work["static_q"],
        layout["keys_blocks_t_f16"],
        layout["assigns_blocks"],
        work["cluster_pass"],
        layout["invalid_blocks_i8"],
        work["compact_ids"],
        work["counts"],
        work["out"],
        h_q,
        h_kv,
        k,
        n_pad,
        ANCHOR_S=anchor_s,
        D=d,
        BF=bf,
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        S=layout["num_subspaces"],
        PARENTS_PER_PROG=_PARENTS_PER_PROG,
        MAX_SURVIVE=k,
        num_warps=4,
    )


def _graph_cache_key(layout: dict, q: torch.Tensor, th: torch.Tensor) -> tuple:
    return (
        q.device.index,
        q.dtype,
        tuple(q.shape),
        th.dtype,
        tuple(th.shape),
        layout["keys_blocks_t_f16"].data_ptr(),
        layout["centers"].data_ptr(),
        layout["radii"].data_ptr(),
        layout["assigns_blocks"].data_ptr(),
        layout["invalid_blocks_i8"].data_ptr(),
    )


def _get_workspace(
    state: dict,
    layout: dict,
    q: torch.Tensor,
    th: torch.Tensor,
) -> dict:
    cache = state.setdefault("_search_v18_3_cache", {})
    key = _graph_cache_key(layout, q, th)
    work = cache.get("work")
    if cache.get("key") == key and work is not None:
        return work

    work = _make_workspace(layout, q, th)
    cache["key"] = key
    cache["work"] = work
    return work


def _maybe_capture_graph(state: dict, layout: dict, work: dict, h_q: int, d: int) -> None:
    if work["graph"] is not None or work["capture_failed"]:
        return
    if not bool(state.get("_search_v18_3_use_cuda_graphs", True)):
        work["capture_failed"] = True
        return

    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                _launch_kernels(work, layout, h_q, d)
        current.wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_kernels(work, layout, h_q, d)
        work["graph"] = graph
    except Exception:
        work["capture_failed"] = True


def search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    keys_children: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("search_v18_3 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("search_v18_3 requires build_v2-style state")

    layout = _get_layout_v15(state, q_head_to_kv, q)
    h_q = q.shape[0]
    d = q.shape[1]
    th_view = th_per_subspace.reshape(layout["num_subspaces"], h_q)
    work = _get_workspace(state, layout, q, th_view)

    work["static_q"].copy_(q)
    work["static_th"].copy_(th_view)
    _prepare_inputs(work, layout)

    _maybe_capture_graph(state, layout, work, h_q, d)
    if work["graph"] is not None:
        work["graph"].replay()
    else:
        _launch_kernels(work, layout, h_q, d)

    out = work["out"]
    if bool(state.get("_search_v18_3_clone_output", False)):
        out = out.clone()

    buf_shim = {
        "mode": layout["mode"],
        "groups": layout["groups"],
        "base_heads": layout["base_heads"],
    }
    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, buf_shim)
    return out if buf_dots is None else torch.cat([out, buf_dots], dim=1)


KERNEL = search
