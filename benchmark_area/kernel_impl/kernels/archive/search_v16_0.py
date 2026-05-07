"""search_v16.0 — per-group-head iteration with stream compaction.

Idea 3a: instead of grid (H_kv, K_tiles) with GROUPS inside tl.dot,
iterate per query-head over only surviving parents.

Pipeline:
  1. Compute cluster_pass (S, H_q, K) as usual.
  2. AND-reduce across subspaces per (hq, parent) → survive_any (H_q, K).
  3. Stream-compact surviving parent indices per head → compact_parents (H_q, max_survive).
  4. Triton kernel grid (H_q, ceil(max_survive / BLOCK)). Each program
     loads 1 query head and processes a block of surviving parents.
     tl.dot: (1_POW, D) x (D, BF*BLOCK) — no wasted group rows.

FP16 keys for 2x bandwidth (same as v15). Fused q-pack (idea 2).

Trade-off vs v11: loses key-sharing across groups (each group re-reads keys)
but eliminates ~87% wasted tensor-core rows when only 1-2 of 7 groups pass.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from ._search_triton import triton_fused_cluster_pass
from ._search_utils import _mapping_mode, buffer_dot

KERNEL_VERSION = "v16.0"
_PARENTS_PER_PROG = 4


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _assign_dtype(k: int) -> torch.dtype:
    return torch.int16 if k < 32768 else torch.int32


def _get_or_make_fp16_block_pack(state: dict):
    cache = state.setdefault("_search_v16_0_block_pack", {})
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


def _compact_surviving_parents(
    cluster_pass_flat: torch.Tensor,  # (S, H_q, K) int8
    assigns_blocks: torch.Tensor,     # (S, H_kv, K, BF) int16/32
    invalid_blocks: torch.Tensor,     # (H_kv, K, BF) int8
    groups: int,
    h_kv: int,
    anchor_s: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute per-(hq, parent) survive and compact.

    Returns:
      compact_ids: (H_q, max_survive) int32 — parent indices that survive
      counts: (H_q,) int32 — number of surviving parents per head
      max_survive: int
    """
    S, H_q, K = cluster_pass_flat.shape

    # For each (hq, parent): does the anchor pass?
    anchor_pass = cluster_pass_flat[anchor_s] != 0  # (H_q, K) bool

    # For non-anchor: a parent survives for hq if anchor passes AND
    # at least one child passes all non-anchor subspaces.
    # But computing per-child AND across S subspaces here is expensive.
    # Simpler approximation: anchor-only compaction. The kernel still
    # does per-child non-anchor gating, but we skip parents that fail anchor.
    survive = anchor_pass  # (H_q, K) bool

    counts = survive.sum(dim=1).to(torch.int32)  # (H_q,)
    max_survive = int(counts.max().item())
    if max_survive == 0:
        max_survive = 1  # avoid zero-size tensor

    # Compact via argsort trick: sort survive descending, take [:max_survive]
    # This puts True values first.
    sort_key = survive.to(torch.int32)
    sorted_idx = sort_key.argsort(dim=1, descending=True, stable=True)
    compact_ids = sorted_idx[:, :max_survive].to(torch.int32).contiguous()

    return compact_ids, counts, max_survive


if HAS_TRITON:

    @triton.jit
    def _per_head_compact_kernel(
        Q_ptr,              # (H_q, D) f32
        KeysBlocksT_ptr,    # (H_kv, K, D, BF) f16
        AssignsBlocks_ptr,  # (S, H_kv, K, BF)
        ClusterPass_ptr,    # (S, H_q, K) int8
        InvalidBlocks_ptr,  # (H_kv, K, BF) int8
        CompactIds_ptr,     # (H_q, max_survive) int32
        Counts_ptr,         # (H_q,) int32
        Out_ptr,            # (H_q, N_pad) f32
        H_Q,
        H_KV,
        K,
        N_PAD,
        GROUPS,
        ANCHOR_S: tl.constexpr,
        D: tl.constexpr,
        BF: tl.constexpr,
        MAX_SURVIVE: tl.constexpr,
        S: tl.constexpr,
        PARENTS_PER_PROG: tl.constexpr,
    ):
        neg_inf = float("-inf")
        hq = tl.program_id(0)
        pb = tl.program_id(1)
        kvh = hq // GROUPS

        count = tl.load(Counts_ptr + hq)
        base = pb * PARENTS_PER_PROG

        # Load query (single head, no wasted rows)
        d_range = tl.arange(0, D)
        q_f32 = tl.load(Q_ptr + hq * D + d_range)
        q_f16 = q_f32.to(tl.float16)

        for pp in range(PARENTS_PER_PROG):
            ci = base + pp
            if ci < count:
                parent = tl.load(CompactIds_ptr + hq * MAX_SURVIVE + ci).to(tl.int32)

                bf_range = tl.arange(0, BF)
                child_idx = parent * BF + bf_range

                # Invalid check
                inv = tl.load(InvalidBlocks_ptr + (kvh * K + parent) * BF + bf_range)
                survive = inv == 0

                # Non-anchor subspace gate per child
                for s in tl.static_range(0, S):
                    if s != ANCHOR_S:
                        assign = tl.load(
                            AssignsBlocks_ptr + ((s * H_KV + kvh) * K + parent) * BF + bf_range,
                            mask=survive, other=0,
                        ).to(tl.int32)
                        passed = tl.load(
                            ClusterPass_ptr + (s * H_Q + hq) * K + assign,
                            mask=survive, other=0,
                        )
                        survive = survive & (passed != 0)

                has_survivors = tl.max(survive.to(tl.int32), axis=0) != 0
                if has_survivors:
                    # Load keys tile (D, BF) f16
                    keys_tile = tl.load(
                        KeysBlocksT_ptr + ((kvh * K + parent) * D + d_range[:, None]) * BF + bf_range[None, :],
                        mask=survive[None, :], other=0.0,
                    )

                    # Dot: (D,) x (D, BF) → (BF,) via manual reduction
                    acc = tl.sum(q_f16[:, None] * keys_tile, axis=0).to(tl.float32)

                    out = tl.where(survive, acc, neg_inf)
                    tl.store(Out_ptr + hq * N_PAD + child_idx, out)
                else:
                    tl.store(Out_ptr + hq * N_PAD + child_idx, neg_inf, mask=tl.arange(0, BF) < BF)


def _get_layout_v16(state, q_head_to_kv, q):
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
        raise RuntimeError("search_v16 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("search_v16 requires build_v2-style state")

    layout = _get_layout_v16(state, q_head_to_kv, q)
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

    cluster_pass_flat = triton_fused_cluster_pass(
        q_packed, q_norm, th_packed,
        layout["centers"], layout["radii"], layout["groups"],
    )

    h_kv = layout["base_heads"]
    k = layout["K"]
    bf = layout["bf"]
    n_pad = layout["N_pad"]
    groups = layout["groups"]
    anchor_s = layout["anchor_subspace"]

    compact_ids, counts, max_survive = _compact_surviving_parents(
        cluster_pass_flat, layout["assigns_blocks"], layout["invalid_blocks_i8"],
        groups, h_kv, anchor_s,
    )
    max_survive_pow = _next_pow2(max_survive)

    out = torch.full((h_q, n_pad), float("-inf"), device=q.device, dtype=torch.float32)

    if max_survive > 0:
        grid = (h_q, triton.cdiv(max_survive, _PARENTS_PER_PROG))
        _per_head_compact_kernel[grid](
            q.contiguous(),
            layout["keys_blocks_t_f16"],
            layout["assigns_blocks"],
            cluster_pass_flat,
            layout["invalid_blocks_i8"],
            compact_ids,
            counts,
            out,
            h_q, h_kv, k, n_pad,
            groups,
            ANCHOR_S=anchor_s,
            D=d, BF=bf,
            MAX_SURVIVE=max_survive,
            S=s,
            PARENTS_PER_PROG=_PARENTS_PER_PROG,
            num_warps=2,
        )

    buf_shim = {"mode": layout["mode"], "groups": groups, "base_heads": h_kv}
    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, buf_shim)
    return out if buf_dots is None else torch.cat([out, buf_dots], dim=1)


KERNEL = search
