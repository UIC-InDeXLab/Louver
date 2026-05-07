"""attention_v1.9 — fixed BF=4/S=8 with packed cluster-pass gates."""

from __future__ import annotations

import math

import torch

try:
    import triton

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from ._attention_fixed_utils import (
    buffer_partial,
    get_layout_attn_rawq,
    next_pow2,
    require_fixed_bf_s,
)
from ._attention_triton import NEG_SENT, run_attn_reduce
from ._attention_triton_v1_9 import (
    run_fused_attn_index_packed,
    triton_fused_cluster_pass_packed,
)

KERNEL_VERSION = "v1.9"
_PARENTS_PER_PROG = 8
_DEFAULT_NUM_SPLITS = 32
_GROUPS_POW_FLOOR = 4


def attend(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    buffer_values: torch.Tensor | None,
    keys_children: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
    num_splits: int = _DEFAULT_NUM_SPLITS,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("attention_v1 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("attention_v1 requires build_v2-style state")

    layout = get_layout_attn_rawq(
        state,
        q_head_to_kv,
        q,
        cache_name="_attn_v1_9_layout",
    )
    require_fixed_bf_s(layout, bf=4, s=8, groups_max=8)

    h_q = q.shape[0]
    d = q.shape[1]
    d_v = layout["D_v"]
    s = layout["num_subspaces"]
    h_kv_eff = layout["base_heads"]
    k = layout["K"]
    groups = layout["groups"]
    groups_pow = max(next_pow2(groups), _GROUPS_POW_FLOOR)
    anchor_s = layout["anchor_subspace"]

    if scale is None:
        scale = 1.0 / math.sqrt(d)

    q_c = q if q.is_contiguous() else q.contiguous()
    th_packed = th_per_subspace.reshape(s, h_q).contiguous()

    ws = state.setdefault("_attn_v1_9_ws", {})
    ws_key = (h_q, num_splits, d_v, k, q.device.index)
    if ws.get("key") != ws_key:
        ws["m_idx"] = torch.empty(h_q, num_splits, device=q.device, dtype=torch.float32)
        ws["l_idx"] = torch.empty(h_q, num_splits, device=q.device, dtype=torch.float32)
        ws["o_idx"] = torch.empty(h_q, num_splits, d_v, device=q.device, dtype=torch.float32)
        ws["out"] = torch.empty(h_q, d_v, device=q.device, dtype=torch.float32)
        ws["packed_pass"] = torch.empty(s, h_kv_eff, k, device=q.device, dtype=torch.uint8)
        ws["buf_m"] = torch.full((h_q,), NEG_SENT, device=q.device, dtype=torch.float32)
        ws["buf_l"] = torch.zeros((h_q,), device=q.device, dtype=torch.float32)
        ws["buf_o"] = torch.zeros((h_q, d_v), device=q.device, dtype=torch.float32)
        ws["key"] = ws_key

    m_idx = ws["m_idx"]
    l_idx = ws["l_idx"]
    o_idx = ws["o_idx"]
    out = ws["out"]
    packed_pass = ws["packed_pass"]
    sentinels = (ws["buf_m"], ws["buf_l"], ws["buf_o"])

    triton_fused_cluster_pass_packed(
        q=q_c,
        th=th_packed,
        dim_offsets=layout["dim_offsets"],
        dim_widths=layout["dim_widths"],
        centers=layout["centers"],
        radii=layout["radii"],
        groups=groups,
        out=packed_pass,
    )

    run_fused_attn_index_packed(
        q=q_c,
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        assigns_blocks=layout["assigns_blocks"],
        packed_pass=packed_pass,
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        h_q=h_q,
        h_kv_eff=h_kv_eff,
        k=k,
        groups=groups,
        groups_pow=groups_pow,
        s_subspaces=s,
        parents_per_prog=_PARENTS_PER_PROG,
        num_splits=num_splits,
        anchor_s=anchor_s,
        scale=float(scale),
        out_m=m_idx,
        out_l=l_idx,
        out_o=o_idx,
    )

    m_buf, l_buf, o_buf = buffer_partial(
        q,
        buffer_keys,
        buffer_values,
        q_head_to_kv,
        layout,
        scale,
        d_v,
        sentinels=sentinels,
    )

    run_attn_reduce(m_idx, l_idx, o_idx, m_buf, l_buf, o_buf, out)
    return out


KERNEL = attend
