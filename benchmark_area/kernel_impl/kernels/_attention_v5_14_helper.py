"""attention_v5_base — parameterized variant of v2.6.

Same overall structure as v2.6 (fused index+buffer kernel + reduce kernel)
but exposes (num_warps, num_stages, num_splits, parents_per_prog, index_kernel)
as keyword arguments.  Each v5.x kernel calls this with its own config and a
unique ``cache_ns`` so state caches and CUDA graphs do not collide.
"""

from __future__ import annotations

import math

import torch

try:
    import triton  # noqa: F401
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

from ._attention_fixed_utils import get_layout_attn_rawq, next_pow2
from ._attention_copy_triton import fused_copy_q_th
from ._attention_triton_v2_0_reduce import run_attn_reduce_v2_0
from ._attention_triton_v2_6_index import run_fused_attn_index_buf_v2_6
from ._attention_triton_v2_0_index import run_fused_attn_index_v2_0

_LOG2E = 1.4426950408889634
_BUCKETS = (64, 128, 256, 512)
_GROUPS_MAX = 8
_ALLOWED_S = (8, 16)
_TARGET_COLS_PER_CHUNK = 64
_TARGET_COLS_PER_CHUNK_HIGH_GROUPS = 128


def _require_fp16(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is not None and tensor.dtype != torch.float16:
        raise RuntimeError(
            f"attention_v5_14 expects {name} in fp16; got {tensor.dtype}. "
            "Convert inputs outside timing before calling this kernel."
        )


def _parents_per_prog_for_bf(bf: int, groups: int) -> int:
    target_cols = _TARGET_COLS_PER_CHUNK
    if bf == 4 and groups >= 6:
        target_cols = _TARGET_COLS_PER_CHUNK_HIGH_GROUPS
    return max(1, target_cols // max(bf, 1))


def _bucket_for(l_buf: int) -> int:
    for bucket in _BUCKETS:
        if l_buf <= bucket:
            return bucket
    raise ValueError(
        f"attention_v5_14: buffer length {l_buf} exceeds max bucket {_BUCKETS[-1]}"
    )


def _empty_buffer(buffer_keys, buffer_values) -> bool:
    return (
        buffer_keys is None
        or buffer_values is None
        or int(buffer_keys.shape[1]) == 0
    )


def _require_supported(layout: dict) -> None:
    s = int(layout["num_subspaces"])
    if s not in _ALLOWED_S:
        raise RuntimeError(f"attention_v5_14 requires S in {_ALLOWED_S}; got S={s}")
    if int(layout["groups"]) > _GROUPS_MAX:
        raise RuntimeError(
            f"attention_v5_14 requires groups <= {_GROUPS_MAX}; got {layout['groups']}"
        )


def _prepare_buffer_effective(
    buffer_keys: torch.Tensor,
    buffer_values: torch.Tensor,
    layout: dict,
    q_head_to_kv: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if layout["mode"] == "expanded":
        assert q_head_to_kv is not None
        return (
            buffer_keys.index_select(0, q_head_to_kv),
            buffer_values.index_select(0, q_head_to_kv),
        )
    return buffer_keys, buffer_values


def _fixed_cache_key(
    state: dict,
    q: torch.Tensor,
    th: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    num_splits: int,
) -> tuple:
    q_map_ptr = 0 if q_head_to_kv is None else q_head_to_kv.data_ptr()
    q_map_shape = () if q_head_to_kv is None else tuple(q_head_to_kv.shape)
    return (
        q.device.index,
        tuple(q.shape),
        q.dtype,
        th.dtype,
        tuple(th.shape),
        num_splits,
        q_map_ptr,
        q_map_shape,
        state["keys_reord"].data_ptr(),
        state["values_blocks_f16"].data_ptr(),
    )


def _get_layout_fp16(
    state: dict,
    q_head_to_kv: torch.Tensor | None,
    q: torch.Tensor,
) -> dict:
    layout = get_layout_attn_rawq(
        state,
        q_head_to_kv,
        q,
        cache_name="_attn_v5_14_layout",
    )
    cache = state.setdefault("_attn_v5_14_layout_fp16", {})
    cache_key = (
        layout["mode"],
        layout["groups"],
        layout["base_heads"],
        layout["num_subspaces"],
        layout["K"],
        layout.get("K_used", layout["K"]),
        layout.get("K_stride", layout["K"]),
        layout["bf"],
        layout["centers"].data_ptr(),
        layout["radii"].data_ptr(),
        layout["keys_blocks_t_f16"].data_ptr(),
        layout["values_blocks_f16"].data_ptr(),
    )
    if cache.get("key") == cache_key:
        return cache["layout"]

    layout_fp16 = dict(layout)
    layout_fp16["centers"] = layout["centers"].to(torch.float16).contiguous()
    layout_fp16["radii"] = layout["radii"].to(torch.float16).contiguous()
    cache["key"] = cache_key
    cache["layout"] = layout_fp16
    return layout_fp16


def _make_shared_workspace(layout: dict, q: torch.Tensor, th: torch.Tensor, num_splits: int) -> dict:
    h_q = q.shape[0]
    d_v = layout["D_v"]
    s = layout["num_subspaces"]
    device = q.device
    static_th_packed = torch.empty((2 * s, h_q), device=device, dtype=torch.float16)
    return {
        "static_q": torch.empty(q.shape, device=device, dtype=torch.float16),
        "static_th_packed": static_th_packed,
        "static_th": static_th_packed[:s],
        "static_q_norms": static_th_packed[s:],
        "m_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "l_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "o_idx": torch.empty(h_q, num_splits, d_v, device=device, dtype=torch.float32),
        "out": torch.empty(h_q, d_v, device=device, dtype=torch.float32),
        "buf_m": torch.full((h_q,), -1.0e30, device=device, dtype=torch.float32),
        "buf_l": torch.zeros((h_q,), device=device, dtype=torch.float32),
        "buf_o": torch.zeros((h_q, d_v), device=device, dtype=torch.float32),
    }


def _make_bucket_staging(layout: dict, q: torch.Tensor, bucket: int) -> dict:
    device = q.device
    d = q.shape[1]
    d_v = layout["D_v"]
    h_kv_eff = int(layout["base_heads"])
    return {
        "buf_keys_t": torch.zeros(h_kv_eff, d, bucket, device=device, dtype=torch.float16),
        "buf_values": torch.zeros(h_kv_eff, bucket, d_v, device=device, dtype=torch.float16),
        "buf_invalid": torch.ones(h_kv_eff, bucket, device=device, dtype=torch.int8),
        "valid_len": 0,
        "graph": None,
        "capture_failed": False,
    }


def _copy_buffer_into_stage_incremental(
    stage: dict,
    buffer_keys_eff: torch.Tensor,
    buffer_values_eff: torch.Tensor,
) -> None:
    _require_fp16("buffer_keys", buffer_keys_eff)
    _require_fp16("buffer_values", buffer_values_eff)

    l_buf = int(buffer_keys_eff.shape[1])
    prev_len = int(stage["valid_len"])
    if l_buf < prev_len:
        stage["buf_invalid"].fill_(1)
        prev_len = 0

    if l_buf == prev_len:
        return

    keys_src = buffer_keys_eff[:, prev_len:l_buf, :].transpose(-1, -2)
    stage["buf_keys_t"][:, :, prev_len:l_buf].copy_(keys_src)
    stage["buf_values"][:, prev_len:l_buf, :].copy_(buffer_values_eff[:, prev_len:l_buf, :])
    stage["buf_invalid"][:, prev_len:l_buf].zero_()
    stage["valid_len"] = l_buf


def _anchor_layout(layout: dict) -> dict:
    anchor_s = int(layout["anchor_subspace"])
    key = (
        int(layout["centers"].data_ptr()),
        int(layout["radii"].data_ptr()),
        anchor_s,
    )
    cached = layout.get("_attn_v5_14_anchor")
    if cached is not None and cached["key"] == key:
        return cached
    width = int(layout["dim_widths"][anchor_s].item())
    cached = {
        "key": key,
        "centers": layout["centers"][anchor_s][..., :width].contiguous().to(torch.float16),
        "radii": layout["radii"][anchor_s].contiguous().to(torch.float16),
        "dim_offset": int(layout["dim_offsets"][anchor_s].item()),
        "width": width,
    }
    layout["_attn_v5_14_anchor"] = cached
    return cached


def _launch_no_buffer(
    shared, layout, h_q, k, k_stride, groups, groups_pow,
    num_splits, anchor_s, scale_log2e, parents_per_prog, anchor,
    *, num_warps, num_stages, run_index_no_buf,
):
    run_index_no_buf(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        centers_anchor=anchor["centers"], radii_anchor=anchor["radii"],
        th_anchor=shared["static_th"][anchor_s],
        q_norm_anchor=shared["static_q_norms"][anchor_s],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        dim_offset=anchor["dim_offset"],
        h_q=h_q, h_kv_eff=layout["base_heads"], k=k, k_stride=k_stride,
        groups=groups, groups_pow=groups_pow,
        parents_per_prog=parents_per_prog,
        num_splits=num_splits, scale_log2e=scale_log2e,
        out_m=shared["m_idx"], out_l=shared["l_idx"], out_o=shared["o_idx"],
        num_warps=num_warps, num_stages=num_stages,
    )
    run_attn_reduce_v2_0(
        shared["m_idx"], shared["l_idx"], shared["o_idx"],
        shared["buf_m"], shared["buf_l"], shared["buf_o"], shared["out"],
    )


def _launch_with_buffer(
    shared, stage, layout, h_q, k, k_stride, groups, groups_pow,
    num_splits, anchor_s, scale_log2e, parents_per_prog,
    bucket, anchor,
    *, num_warps, num_stages, run_index_with_buf,
):
    run_index_with_buf(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        centers_anchor=anchor["centers"], radii_anchor=anchor["radii"],
        th_anchor=shared["static_th"][anchor_s],
        q_norm_anchor=shared["static_q_norms"][anchor_s],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        buf_keys_t_f16=stage["buf_keys_t"],
        buf_values_f16=stage["buf_values"],
        buf_invalid_i8=stage["buf_invalid"],
        dim_offset=anchor["dim_offset"],
        h_q=h_q, h_kv_eff=layout["base_heads"], k=k, k_stride=k_stride,
        groups=groups, groups_pow=groups_pow,
        parents_per_prog=parents_per_prog,
        num_splits=num_splits, scale_log2e=scale_log2e,
        l_buf_max=bucket,
        out_m=shared["m_idx"], out_l=shared["l_idx"], out_o=shared["o_idx"],
        num_warps=num_warps, num_stages=num_stages,
    )
    run_attn_reduce_v2_0(
        shared["m_idx"], shared["l_idx"], shared["o_idx"],
        shared["buf_m"], shared["buf_l"], shared["buf_o"], shared["out"],
    )


def _capture_graph(state, stage, launch_fn, launch_args, kwargs, cache_ns):
    if stage["graph"] is not None or stage["capture_failed"]:
        return
    if not bool(state.get(f"{cache_ns}_use_cuda_graphs", True)):
        stage["capture_failed"] = True
        return
    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                launch_fn(*launch_args, **kwargs)
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch_fn(*launch_args, **kwargs)
        stage["graph"] = graph
    except Exception:
        stage["capture_failed"] = True


def attend_v5(
    q, th_per_subspace, state, buffer_keys, buffer_values,
    keys_children, q_head_to_kv=None, scale=None,
    *,
    cache_ns: str,
    num_warps: int = 4,
    num_stages: int = 3,
    num_splits: int = 85,
    parents_per_prog_override: int | None = None,
    run_index_no_buf=run_fused_attn_index_v2_0,
    run_index_with_buf=run_fused_attn_index_buf_v2_6,
):
    del keys_children
    if not HAS_TRITON:
        raise RuntimeError(f"{cache_ns} requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError(f"{cache_ns} requires build_v2-style state")

    _require_fp16("q", q)
    _require_fp16("th_per_subspace", th_per_subspace)
    _require_fp16("buffer_keys", buffer_keys)
    _require_fp16("buffer_values", buffer_values)

    h_q = q.shape[0]
    d = q.shape[1]
    q_c = q if q.is_contiguous() else q.contiguous()

    rows = int(th_per_subspace.shape[0])
    s_hint = rows // 2
    packed = th_per_subspace if th_per_subspace.is_contiguous() else th_per_subspace.contiguous()
    th_view = packed[:s_hint]

    cache_key = _fixed_cache_key(state, q_c, th_view, q_head_to_kv, num_splits)
    cache = state.setdefault(cache_ns, {})
    fixed = cache.get("fixed")
    if cache.get("key") != cache_key or fixed is None:
        layout = _get_layout_fp16(state, q_head_to_kv, q_c)
        _require_supported(layout)
        groups = int(layout["groups"])
        if groups > _GROUPS_MAX:
            raise RuntimeError(f"groups={groups} > {_GROUPS_MAX}")
        groups_pow = max(next_pow2(groups), 4)
        anchor_s = int(layout["anchor_subspace"])
        if parents_per_prog_override is not None:
            parents_per_prog = int(parents_per_prog_override)
        else:
            parents_per_prog = _parents_per_prog_for_bf(int(layout["bf"]), groups)
        shared = _make_shared_workspace(layout, q_c, th_view, num_splits)
        fixed = {
            "layout": layout, "shared": shared,
            "groups": groups, "groups_pow": groups_pow,
            "anchor_s": anchor_s, "parents_per_prog": parents_per_prog,
            "buckets": {}, "no_buffer_stage": None,
        }
        cache["key"] = cache_key
        cache["fixed"] = fixed

    layout = fixed["layout"]
    shared = fixed["shared"]
    groups = fixed["groups"]
    groups_pow = fixed["groups_pow"]
    anchor_s = fixed["anchor_s"]
    parents_per_prog = fixed["parents_per_prog"]
    k = int(layout.get("K_used", layout["K"]))
    k_stride = int(layout.get("K_stride", layout["K"]))
    anchor = _anchor_layout(layout)

    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale_log2e = float(scale) * _LOG2E

    fused_copy_q_th(q_c, packed, shared["static_q"], shared["static_th_packed"])

    launch_kwargs_no_buf = dict(
        num_warps=num_warps, num_stages=num_stages,
        run_index_no_buf=run_index_no_buf,
    )
    launch_kwargs_buf = dict(
        num_warps=num_warps, num_stages=num_stages,
        run_index_with_buf=run_index_with_buf,
    )

    if _empty_buffer(buffer_keys, buffer_values):
        stage = fixed.get("no_buffer_stage")
        if stage is None:
            stage = {"graph": None, "capture_failed": False}
            fixed["no_buffer_stage"] = stage
        _capture_graph(state, stage, _launch_no_buffer,
                       (shared, layout, h_q, k, k_stride, groups, groups_pow,
                        num_splits, anchor_s, scale_log2e, parents_per_prog, anchor),
                       launch_kwargs_no_buf, cache_ns)
        if stage["graph"] is not None:
            stage["graph"].replay()
        else:
            _launch_no_buffer(shared, layout, h_q, k, k_stride, groups, groups_pow,
                              num_splits, anchor_s, scale_log2e, parents_per_prog, anchor,
                              **launch_kwargs_no_buf)
        return shared["out"]

    l_buf = int(buffer_keys.shape[1])
    bucket = _bucket_for(l_buf)
    stage = fixed["buckets"].get(bucket)
    if stage is None:
        stage = _make_bucket_staging(layout, q_c, bucket)
        fixed["buckets"][bucket] = stage

    buf_keys_eff, buf_values_eff = _prepare_buffer_effective(
        buffer_keys, buffer_values, layout, q_head_to_kv)
    _copy_buffer_into_stage_incremental(stage, buf_keys_eff, buf_values_eff)

    _capture_graph(state, stage, _launch_with_buffer,
                   (shared, stage, layout, h_q, k, k_stride, groups, groups_pow,
                    num_splits, anchor_s, scale_log2e, parents_per_prog,
                    bucket, anchor),
                   launch_kwargs_buf, cache_ns)
    if stage["graph"] is not None:
        stage["graph"].replay()
    else:
        _launch_with_buffer(shared, stage, layout, h_q, k, k_stride, groups, groups_pow,
                            num_splits, anchor_s, scale_log2e, parents_per_prog,
                            bucket, anchor, **launch_kwargs_buf)
    return shared["out"]
