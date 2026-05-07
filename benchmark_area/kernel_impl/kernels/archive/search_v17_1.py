"""search_v17.1 — v17.0 + stream compaction (idea 5).

Combines:
  - FP16 keys + fused q-pack (v15)
  - Bitmask cluster_pass (idea 4): (S, H_kv, K) int32 with GROUPS bits packed
  - Stream compaction (idea 5): per-kv-head compact list of parents that pass
    anchor for at least one group. At low scan fraction (real data ~22%)
    the grid shrinks ~4.5x.

Reuses kernel + bitmask helpers from v17.0 with USE_COMPACT=True.
"""

from __future__ import annotations

import torch

try:
    import triton

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from ._search_utils import _mapping_mode, buffer_dot
from .search_v17_0 import (
    _PARENTS_PER_PROG,
    _bitmask_batched_kernel,
    _compact_anchor_parents,
    _fused_cluster_pass_bitmask,
    _next_pow2,
)

KERNEL_VERSION = "v17.1"


def _assign_dtype(k: int) -> torch.dtype:
    return torch.int16 if k < 32768 else torch.int32


def _get_or_make_fp16_block_pack(state: dict):
    cache = state.setdefault("_search_v17_1_block_pack", {})
    keys_reord = state["keys_reord"]
    cache_key = (keys_reord.data_ptr(), tuple(keys_reord.shape), state["K"], state["bf"], len(state["assigns_reord"]))
    if cache.get("key") == cache_key:
        p = cache["pack"]
        return p["keys_blocks_t_f16"], p["assigns_blocks"], p["invalid_blocks_i8"]

    h_kv, _, d = keys_reord.shape
    k = state["K"]
    bf = state["bf"]
    s = len(state["assigns_reord"])
    pack = {
        "keys_blocks_t_f16": keys_reord.view(h_kv, k, bf, d).permute(0, 1, 3, 2).to(torch.float16).contiguous(),
        "assigns_blocks": torch.stack(state["assigns_reord"], dim=0).to(_assign_dtype(k)).view(s, h_kv, k, bf).contiguous(),
        "invalid_blocks_i8": state["invalid_mask"].view(h_kv, k, bf).to(torch.int8).contiguous(),
    }
    cache["key"] = cache_key
    cache["pack"] = pack
    return pack["keys_blocks_t_f16"], pack["assigns_blocks"], pack["invalid_blocks_i8"]


def _get_layout_v17_1(state, q_head_to_kv, q):
    keys_reord = state["keys_reord"]
    h_kv = keys_reord.shape[0]
    h_q = int(q.shape[0]) if q_head_to_kv is None else int(q_head_to_kv.shape[0])
    mode, groups, _ = _mapping_mode(q_head_to_kv, h_q, h_kv)

    keys_f16, assigns_blocks, invalid_blocks = _get_or_make_fp16_block_pack(state)

    dim_slices = state["dim_slices"]
    widths = [end - start for start, end in dim_slices]
    max_d = max(widths)
    s = len(dim_slices)
    k = state["K"]
    bf = state["bf"]
    n_pad = state["N_pad"]
    centers_src = state["centers"]
    radii_src = state["radii"]

    if mode == "expanded":
        assert q_head_to_kv is not None
        centers_src = [t.index_select(0, q_head_to_kv).contiguous() for t in centers_src]
        radii_src = [t.index_select(0, q_head_to_kv).contiguous() for t in radii_src]
        keys_f16 = keys_f16.index_select(0, q_head_to_kv).contiguous()
        assigns_blocks = assigns_blocks.index_select(1, q_head_to_kv).contiguous()
        invalid_blocks = invalid_blocks.index_select(0, q_head_to_kv).contiguous()
        h_kv_eff = h_q
    else:
        h_kv_eff = h_kv

    centers = torch.zeros(s, h_kv_eff, k, max_d, device=q.device, dtype=torch.float32)
    for idx, c in enumerate(centers_src):
        centers[idx, :, :, : c.shape[-1]] = c

    return {
        "mode": mode, "groups": groups, "base_heads": h_kv_eff,
        "num_subspaces": s, "max_d": max_d, "dim_slices": tuple(dim_slices),
        "centers": centers.contiguous(),
        "radii": torch.stack(radii_src, dim=0).contiguous(),
        "keys_blocks_t_f16": keys_f16,
        "assigns_blocks": assigns_blocks,
        "invalid_blocks_i8": invalid_blocks,
        "K": k, "bf": bf, "N_pad": n_pad,
        "anchor_subspace": state.get("anchor_subspace", 0),
    }


def search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    keys_children: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("search_v17_1 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("search_v17_1 requires build_v2-style state")

    layout = _get_layout_v17_1(state, q_head_to_kv, q)
    h_q = q.shape[0]
    s = layout["num_subspaces"]
    max_d = layout["max_d"]
    d = q.shape[1]

    if d == s * max_d:
        q_packed = q.view(h_q, s, max_d).transpose(0, 1).contiguous()
    else:
        q_packed = q.new_zeros(s, h_q, max_d)
        for si, (s0, e0) in enumerate(layout["dim_slices"]):
            q_packed[si, :, : e0 - s0] = q[:, s0:e0]
        q_packed = q_packed.contiguous()
    q_norm = q_packed.norm(dim=-1).contiguous()
    th_packed = th_per_subspace.reshape(s, h_q).contiguous()

    h_kv = layout["base_heads"]
    k = layout["K"]
    bf = layout["bf"]
    n_pad = layout["N_pad"]
    groups = layout["groups"]
    groups_pow = max(_next_pow2(groups), 8)
    anchor_s = layout["anchor_subspace"]

    bitmask_pass = _fused_cluster_pass_bitmask(
        q_packed, q_norm, th_packed,
        layout["centers"], layout["radii"], groups,
    )

    compact_ids, counts, max_survive = _compact_anchor_parents(
        bitmask_pass, layout["invalid_blocks_i8"], anchor_s,
    )

    # Pre-fill -inf since compact kernel only writes visited parents
    out = torch.full((h_q, n_pad), float("-inf"), device=q.device, dtype=torch.float32)

    grid = (h_kv, triton.cdiv(max_survive, _PARENTS_PER_PROG))
    _bitmask_batched_kernel[grid](
        q.contiguous(),
        layout["keys_blocks_t_f16"],
        layout["assigns_blocks"],
        bitmask_pass,
        layout["invalid_blocks_i8"],
        compact_ids,
        counts,
        out,
        h_q, h_kv, k, n_pad,
        ANCHOR_S=anchor_s,
        D=d, BF=bf,
        GROUPS=groups, GROUPS_POW=groups_pow,
        S=s, PARENTS_PER_PROG=_PARENTS_PER_PROG,
        USE_COMPACT=True,
        MAX_SURVIVE=max_survive,
        num_warps=4,
    )

    buf_shim = {"mode": layout["mode"], "groups": groups, "base_heads": h_kv}
    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, buf_shim)
    return out if buf_dots is None else torch.cat([out, buf_dots], dim=1)


KERNEL = search
