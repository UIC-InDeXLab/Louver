"""search_v10.0 — anchor-bf-block fused search.

Pairs with build_v2_0:
  - keys_reord (H_kv, N_pad, D): physically reordered such that anchor
    subspace cluster c's children sit at slots [c*bf, (c+1)*bf).
  - assigns_reord[s] (H_kv, N_pad): cluster id (in subspace s) of the
    child at each physical slot.
  - per-subspace centers/radii (used to compute cluster_pass in fp32 bmm).

Search:
  1. Outside the kernel: cluster_pass = (S, H_q, K) via cuBLAS bmm of
     packed q against centers_t (matches existing _search_utils path).
  2. Triton kernel grid (H_kv, K). Per program (kvh, parent_anchor):
     a. Load anchor cluster_pass for the GROUPS q-heads. If none pass,
        store -inf for the bf children and return.
     b. Else load (D, BF) keys tile for this parent's bf children.
     c. tl.dot tensor-core matmul (GROUPS_POW, D) x (D, BF).
     d. For each non-anchor subspace, look up assigns_reord[s, kvh, child_idx]
        and gate by cluster_pass[s, hq, that_cluster]. AND across subspaces.
     e. Mask invalid groups, padded children, gate-failed (g, j) pairs and
        store the (GROUPS, BF) block.

Fuses: anchor-block skipping + tensor-core dot + per-child multi-subspace AND
in a single launch. Subspaces are SIMD-vectorized inside the program (no
cross-program reduction).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from ._search_utils import buffer_dot, _mapping_mode, pack_query_subspaces
from ._search_triton import triton_fused_cluster_pass

KERNEL_VERSION = "v10.0"


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


if HAS_TRITON:

    @triton.jit
    def _fused_anchor_kernel(
        Q_ptr,                # (H_q, D)
        Keys_ptr,             # (H_kv, N_pad, D)
        AssignsReord_ptr,     # (S, H_kv, N_pad)  int32
        ClusterPass_ptr,      # (S, H_q, K)       int8
        Invalid_ptr,          # (H_kv, N_pad)     int8
        Out_ptr,              # (H_q, N_pad)      f32
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
    ):
        neg_inf = float("-inf")
        kvh = tl.program_id(0)
        parent = tl.program_id(1)

        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        d_full = tl.arange(0, D)
        bf_range = tl.arange(0, BF)

        hq_vec = kvh * GROUPS + g_range  # (GROUPS_POW,)

        # --- Output offsets / mask
        out_offs = hq_vec[:, None] * N_PAD + parent * BF + bf_range[None, :]
        out_mask = g_valid[:, None]

        # --- Anchor parent-level gate: cluster_pass[ANCHOR_S, hq_vec, parent]
        anchor_pass_offs = (ANCHOR_S * H_Q + hq_vec) * K + parent
        anchor_pass = tl.load(
            ClusterPass_ptr + anchor_pass_offs, mask=g_valid, other=0
        )
        anchor_pass_b = (anchor_pass != 0) & g_valid  # (GROUPS_POW,)

        if tl.max(anchor_pass_b.to(tl.int32), axis=0) == 0:
            tl.store(Out_ptr + out_offs, neg_inf, mask=out_mask)
            return

        # --- Load (D, BF) keys tile for this parent's bf children.
        keys_offs = (kvh * N_PAD + parent * BF + bf_range[None, :]) * D + d_full[:, None]
        keys_tile = tl.load(Keys_ptr + keys_offs)  # (D, BF)

        # --- Load q_full: (GROUPS_POW, D)
        q_full_offs = hq_vec[:, None] * D + d_full[None, :]
        q_full = tl.load(Q_ptr + q_full_offs, mask=g_valid[:, None], other=0.0)

        # --- Tensor-core matmul.
        acc = tl.dot(q_full, keys_tile, allow_tf32=True)  # (GROUPS_POW, BF)

        # --- Invalid-child mask (shared across groups).
        inv_offs = kvh * N_PAD + parent * BF + bf_range
        inv = tl.load(Invalid_ptr + inv_offs)
        valid_child = inv == 0  # (BF,)

        # --- Per-child gate from non-anchor subspaces. Start from anchor pass
        #     (broadcast across BF) and AND with each non-anchor subspace.
        survive = anchor_pass_b[:, None] & valid_child[None, :]  # (GROUPS_POW, BF)

        # Loop over non-anchor subspaces (compile-time unroll). For each:
        #   assign_s[j] = AssignsReord[s, kvh, parent*BF + j]
        #   pass_s[g, j] = ClusterPass[s, hq_vec[g], assign_s[j]]
        for s in tl.static_range(0, S):
            if s != ANCHOR_S:
                # Assigns for this subspace's children at our physical slots.
                a_offs = (s * H_KV + kvh) * N_PAD + parent * BF + bf_range
                assign_s = tl.load(a_offs + AssignsReord_ptr)  # (BF,) int32

                # Cluster_pass[s, hq, assign_s]: shape (GROUPS_POW, BF)
                cp_offs = (s * H_Q + hq_vec[:, None]) * K + assign_s[None, :]
                passed = tl.load(
                    ClusterPass_ptr + cp_offs,
                    mask=g_valid[:, None],
                    other=0,
                )
                survive = survive & (passed != 0)

        out = tl.where(survive, acc, neg_inf)
        tl.store(Out_ptr + out_offs, out, mask=out_mask)


def _get_layout_v10(state: dict, q_head_to_kv: torch.Tensor | None, q: torch.Tensor):
    keys_reord: torch.Tensor = state["keys_reord"]
    H_kv = keys_reord.shape[0]
    H_q = int(q.shape[0]) if q_head_to_kv is None else int(q_head_to_kv.shape[0])
    mode, groups, mapping_sig = _mapping_mode(q_head_to_kv, H_q, H_kv)

    cache = state.setdefault("_search_v10_0_cache", {})
    cache_key = (mode, groups, mapping_sig, keys_reord.data_ptr(), tuple(keys_reord.shape))
    if cache.get("key") == cache_key:
        return cache["layout"]

    dim_slices = state["dim_slices"]
    widths = [end - start for start, end in dim_slices]
    max_d = max(widths)
    S = len(dim_slices)
    K = state["K"]
    bf = state["bf"]
    N_pad = state["N_pad"]

    centers_src = state["centers"]
    radii_src = state["radii"]
    assigns_reord_src = state["assigns_reord"]
    invalid_mask = state["invalid_mask"]

    if mode == "expanded":
        assert q_head_to_kv is not None
        centers_src = [t.index_select(0, q_head_to_kv).contiguous() for t in centers_src]
        radii_src = [t.index_select(0, q_head_to_kv).contiguous() for t in radii_src]
        assigns_reord_src = [t.index_select(0, q_head_to_kv).contiguous() for t in assigns_reord_src]
        keys_base = keys_reord.index_select(0, q_head_to_kv).contiguous()
        invalid_base = invalid_mask.index_select(0, q_head_to_kv).contiguous()
        H_kv_eff = H_q
    else:
        keys_base = keys_reord
        invalid_base = invalid_mask
        H_kv_eff = H_kv

    centers = torch.zeros(S, H_kv_eff, K, max_d, device=keys_base.device, dtype=centers_src[0].dtype)
    for s, c in enumerate(centers_src):
        centers[s, :, :, : c.shape[-1]] = c
    centers = centers.contiguous()
    centers_t = centers.reshape(S * H_kv_eff, K, max_d).transpose(1, 2).contiguous()

    radii = torch.stack(radii_src, dim=0).contiguous()       # (S, H_kv_eff, K)
    assigns_reord = torch.stack(assigns_reord_src, dim=0).contiguous()  # (S, H_kv_eff, N_pad)
    invalid_i8 = invalid_base.to(torch.int8).contiguous()

    layout = {
        "mode": mode,
        "groups": groups,
        "base_heads": H_kv_eff,
        "num_subspaces": S,
        "num_points": N_pad,  # cluster_pass_only uses this for shape inference
        "max_d": max_d,
        "dim_slices": tuple(dim_slices),
        "centers": centers,
        "centers_t": centers_t,
        "radii": radii,
        "keys_reord": keys_base,
        "assigns_reord": assigns_reord,
        "invalid_mask_i8": invalid_i8,
        "K": K,
        "bf": bf,
        "N_pad": N_pad,
        "anchor_subspace": state.get("anchor_subspace", 0),
    }
    cache["key"] = cache_key
    cache["layout"] = layout
    return layout


def search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    keys_children: torch.Tensor,  # ignored — keys come from build state
    q_head_to_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("search_v10 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("search_v10 requires build_v2 state")

    layout = _get_layout_v10(state, q_head_to_kv, q)
    H_q = q.shape[0]
    S = layout["num_subspaces"]
    max_d = layout["max_d"]
    D = q.shape[1]

    # Fast q_packed: when all subspace widths are equal (D % S == 0), this is
    # a single reshape — no Python loop and no copy beyond contiguous-ize.
    if D == S * max_d:
        q_packed = q.view(H_q, S, max_d).transpose(0, 1).contiguous()
    else:
        from ._search_utils import pack_query_subspaces
        _, q_packed, _ = pack_query_subspaces(q, layout)
        q_packed = q_packed.reshape(S, H_q, max_d).contiguous()
    q_norm = q_packed.norm(dim=-1).contiguous()
    th_packed = th_per_subspace.reshape(S, H_q).contiguous()
    cluster_pass_flat = triton_fused_cluster_pass(
        q_packed, q_norm, th_packed, layout["centers"], layout["radii"], layout["groups"]
    )

    H_kv = layout["base_heads"]
    K = layout["K"]
    bf = layout["bf"]
    N_pad = layout["N_pad"]
    D = q.shape[1]
    S = layout["num_subspaces"]
    groups = layout["groups"]
    groups_pow = max(_next_pow2(groups), 8)
    anchor_s = layout["anchor_subspace"]

    out = torch.empty(H_q, N_pad, device=q.device, dtype=torch.float32)

    grid = (H_kv, K)
    _fused_anchor_kernel[grid](
        q.contiguous(),
        layout["keys_reord"],
        layout["assigns_reord"],
        cluster_pass_flat,
        layout["invalid_mask_i8"],
        out,
        H_q,
        H_kv,
        K,
        N_pad,
        ANCHOR_S=anchor_s,
        D=D,
        BF=bf,
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        S=S,
        num_warps=4,
    )

    buf_layout_shim = {
        "mode": layout["mode"],
        "groups": groups,
        "base_heads": H_kv,
    }
    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, buf_layout_shim)
    return out if buf_dots is None else torch.cat([out, buf_dots], dim=1)


KERNEL = search
