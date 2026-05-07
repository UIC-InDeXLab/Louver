"""attention_v1.16 — v1.15 + non-empty buffer support.

Extends v1.15's fixed-shape / CUDA-graph pipeline with a Triton buffer-attention
kernel. User passes `buffer_keys: (H_kv, L_buf, D)` and
`buffer_values: (H_kv, L_buf, D_v)`; the Python layer buckets `L_buf` into
one of {64, 128, 256, 512} via an invalid-mask padding scheme, then replays a
per-bucket CUDA graph.

Specialization strategy (matches the recommendation: one source, many SASS):
  * `L_BUF_MAX`        — constexpr → Triton compiles one kernel per bucket.
  * `BUF_COLS_PER_PROG`— constexpr → unrolled inner loop, tile size per bucket.
  * `num_warps`, `num_stages` — per-bucket config from `_DEFAULT_BUFFER_CFG`
    (overridable via `state['_attn_v1_16_buffer_cfg']` for tuning sweeps).

S specialization (8 or 16):
  * Accepted via the layout's `num_subspaces`. The underlying index kernel is
    already generic over S (`tl.static_range(0, S)`) — we just validate and
    forward. BF is kept runtime-variable, with `PARENTS_PER_PROG` scaled to
    target ~64 cols/chunk (same logic as v1_15_s16).

Graph cache is keyed by bucket (plus `None` when the buffer is empty), so a
caller that sees varying `L_buf` pays at most one capture per bucket.
"""

from __future__ import annotations

import math

import torch

try:
    import triton

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from ._attention_buffer_triton import run_buffer_attn
from ._attention_fixed_utils import get_layout_attn_rawq, next_pow2
from ._attention_triton import NEG_SENT, run_attn_reduce
from ._attention_triton_v1_5 import triton_fused_cluster_pass_rawq
from ._attention_triton_v1_14 import run_fused_attn_index_ns

KERNEL_VERSION = "v1.16"

_ALLOWED_S = (8, 16)
_BUCKETS = (64, 128, 256, 512)

_DEFAULT_NUM_SPLITS = 32
_GROUPS_POW_FLOOR = 4
_GROUPS_MAX = 8

_NUM_STAGES = 3
_NUM_WARPS = 4
_TARGET_COLS_PER_CHUNK = 64  # PARENTS_PER_PROG * BF target, matches v1_15_s16

# Per-bucket buffer-kernel config. Tuned on RTX 5090 against REAL captures
# (Llama-3.2-3B layer 15 S=8, Qwen2.5-7B layer 15 S=16) with tight top-k
# thresholds that produce ~7-16% index-kernel pruning — the regime the user
# actually runs in. Synthetic + loose-threshold tuning picks different
# winners because it underweights the buffer's share of total latency.
#
# Cross-setup compromise: each entry stays within ~10% of the per-setup
# winner on both models (see kernels/kernel_bench/tune_attn_v1_16.py, plus
# the tie-break runs in the commit description). Override per-run via
# state["_attn_v1_16_buffer_cfg"] = {bucket: {...}}.
_DEFAULT_BUFFER_CFG: dict[int, dict[str, int]] = {
    64:  {"cols": 64,  "num_warps": 4, "num_stages": 4},
    128: {"cols": 128, "num_warps": 4, "num_stages": 3},
    256: {"cols": 64,  "num_warps": 8, "num_stages": 3},
    512: {"cols": 64,  "num_warps": 8, "num_stages": 3},
}


def _parents_per_prog_for_bf(bf: int) -> int:
    return max(1, _TARGET_COLS_PER_CHUNK // max(bf, 1))


def _bucket_for(l_buf: int) -> int:
    for b in _BUCKETS:
        if l_buf <= b:
            return b
    raise ValueError(
        f"attention_v1_16: buffer length {l_buf} exceeds max bucket {_BUCKETS[-1]}"
    )


def _require_supported(layout: dict) -> None:
    s = int(layout["num_subspaces"])
    if s not in _ALLOWED_S:
        raise RuntimeError(
            f"attention_v1_16 requires S in {_ALLOWED_S}; got S={s}"
        )
    if int(layout["groups"]) > _GROUPS_MAX:
        raise RuntimeError(
            f"attention_v1_16 requires groups <= {_GROUPS_MAX}; got {layout['groups']}"
        )


def _buffer_cfg(state: dict, bucket: int) -> dict[str, int]:
    override = state.get("_attn_v1_16_buffer_cfg") or {}
    cfg = dict(_DEFAULT_BUFFER_CFG[bucket])
    cfg.update(override.get(bucket, {}))
    return cfg


def _buffer_shape_for_layout(layout: dict) -> int:
    """Returns the effective kv-dim size of the buffer (h_kv_eff)."""
    return int(layout["base_heads"])


def _prepare_buffer_effective(
    buffer_keys: torch.Tensor,
    buffer_values: torch.Tensor,
    layout: dict,
    q_head_to_kv: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map user buffer (H_kv, L, *) to the (H_kv_eff, L, *) layout used here.

    Note: unlike v1_5's bmm-based buffer path, we DO NOT repeat-interleave in
    grouped mode. The kernel iterates GROUPS q-heads per kv-program, so keys
    stay per-kv.
    """
    mode = layout["mode"]
    if mode == "expanded":
        assert q_head_to_kv is not None
        return (
            buffer_keys.index_select(0, q_head_to_kv),
            buffer_values.index_select(0, q_head_to_kv),
        )
    return buffer_keys, buffer_values


def _make_shared_workspace(layout: dict, q: torch.Tensor, th: torch.Tensor, num_splits: int) -> dict:
    h_q = q.shape[0]
    d_v = layout["D_v"]
    s = layout["num_subspaces"]
    k = layout["K"]
    device = q.device
    return {
        # static_q is always fp32 (the cluster-pass + index kernels need fp32).
        # Users may pass fp16 Q; copy_() handles the dtype conversion for free.
        "static_q": torch.empty(q.shape, device=device, dtype=torch.float32),
        "static_th": torch.empty_like(th),
        "cluster_pass": torch.empty(s, h_q, k, device=device, dtype=torch.int8),
        "m_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "l_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "o_idx": torch.empty(h_q, num_splits, d_v, device=device, dtype=torch.float32),
        "out": torch.empty(h_q, d_v, device=device, dtype=torch.float32),
        # Buffer partial slots (shared across buckets — shape is per-hq).
        "buf_m": torch.full((h_q,), NEG_SENT, device=device, dtype=torch.float32),
        "buf_l": torch.zeros((h_q,), device=device, dtype=torch.float32),
        "buf_o": torch.zeros((h_q, d_v), device=device, dtype=torch.float32),
    }


def _make_bucket_staging(layout: dict, q: torch.Tensor, bucket: int) -> dict:
    """Per-bucket fixed-shape staging tensors for the buffer kernel."""
    device = q.device
    d = q.shape[1]
    d_v = layout["D_v"]
    h_kv_eff = _buffer_shape_for_layout(layout)
    return {
        "buf_keys_t": torch.zeros(h_kv_eff, d, bucket, device=device, dtype=torch.float16),
        "buf_values": torch.zeros(h_kv_eff, bucket, d_v, device=device, dtype=torch.float16),
        "buf_invalid": torch.ones(h_kv_eff, bucket, device=device, dtype=torch.int8),
        "graph": None,
        "capture_failed": False,
    }


def _launch_no_buffer(
    shared: dict,
    layout: dict,
    h_q: int,
    k: int,
    groups: int,
    groups_pow: int,
    num_splits: int,
    anchor_s: int,
    scale: float,
    parents_per_prog: int,
    s_subspaces: int,
) -> None:
    triton_fused_cluster_pass_rawq(
        q=shared["static_q"],
        th=shared["static_th"],
        dim_offsets=layout["dim_offsets"],
        dim_widths=layout["dim_widths"],
        centers=layout["centers"],
        radii=layout["radii"],
        groups=groups,
        out=shared["cluster_pass"],
    )

    run_fused_attn_index_ns(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        assigns_blocks=layout["assigns_blocks"],
        cluster_pass=shared["cluster_pass"],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        h_q=h_q,
        h_kv_eff=layout["base_heads"],
        k=k,
        groups=groups,
        groups_pow=groups_pow,
        s_subspaces=s_subspaces,
        parents_per_prog=parents_per_prog,
        num_splits=num_splits,
        anchor_s=anchor_s,
        scale=scale,
        out_m=shared["m_idx"],
        out_l=shared["l_idx"],
        out_o=shared["o_idx"],
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    run_attn_reduce(
        shared["m_idx"], shared["l_idx"], shared["o_idx"],
        shared["buf_m"], shared["buf_l"], shared["buf_o"],
        shared["out"],
    )


def _launch_with_buffer(
    shared: dict,
    stage: dict,
    layout: dict,
    h_q: int,
    k: int,
    groups: int,
    groups_pow: int,
    num_splits: int,
    anchor_s: int,
    scale: float,
    parents_per_prog: int,
    s_subspaces: int,
    bucket: int,
    buf_cols: int,
    buf_warps: int,
    buf_stages: int,
) -> None:
    triton_fused_cluster_pass_rawq(
        q=shared["static_q"],
        th=shared["static_th"],
        dim_offsets=layout["dim_offsets"],
        dim_widths=layout["dim_widths"],
        centers=layout["centers"],
        radii=layout["radii"],
        groups=groups,
        out=shared["cluster_pass"],
    )

    run_fused_attn_index_ns(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        assigns_blocks=layout["assigns_blocks"],
        cluster_pass=shared["cluster_pass"],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        h_q=h_q,
        h_kv_eff=layout["base_heads"],
        k=k,
        groups=groups,
        groups_pow=groups_pow,
        s_subspaces=s_subspaces,
        parents_per_prog=parents_per_prog,
        num_splits=num_splits,
        anchor_s=anchor_s,
        scale=scale,
        out_m=shared["m_idx"],
        out_l=shared["l_idx"],
        out_o=shared["o_idx"],
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    run_buffer_attn(
        q=shared["static_q"],
        buf_keys_t_f16=stage["buf_keys_t"],
        buf_values_f16=stage["buf_values"],
        buf_invalid_i8=stage["buf_invalid"],
        h_kv_eff=layout["base_heads"],
        groups=groups,
        groups_pow=groups_pow,
        l_buf_max=bucket,
        buf_cols_per_prog=buf_cols,
        scale=scale,
        out_m=shared["buf_m"],
        out_l=shared["buf_l"],
        out_o=shared["buf_o"],
        num_warps=buf_warps,
        num_stages=buf_stages,
    )

    run_attn_reduce(
        shared["m_idx"], shared["l_idx"], shared["o_idx"],
        shared["buf_m"], shared["buf_l"], shared["buf_o"],
        shared["out"],
    )


def _fixed_cache_key(
    state: dict,
    q: torch.Tensor,
    th: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    num_splits: int,
) -> tuple:
    q_map_ptr = 0 if q_head_to_kv is None else q_head_to_kv.data_ptr()
    q_map_shape = () if q_head_to_kv is None else tuple(q_head_to_kv.shape)
    # Intentionally exclude q.dtype: static_q is pinned to fp32 regardless of
    # input dtype (copy_ handles any fp16→fp32 cast), so the cached graph is
    # valid for both fp16 and fp32 callers.
    return (
        q.device.index,
        tuple(q.shape),
        th.dtype,
        tuple(th.shape),
        num_splits,
        q_map_ptr,
        q_map_shape,
        state["keys_reord"].data_ptr(),
        state["values_blocks_f16"].data_ptr(),
    )


def _capture_no_buffer_graph(
    state: dict,
    shared: dict,
    stage: dict,
    layout: dict,
    h_q: int,
    k: int,
    groups: int,
    groups_pow: int,
    num_splits: int,
    anchor_s: int,
    scale: float,
    parents_per_prog: int,
    s_subspaces: int,
) -> None:
    if stage["graph"] is not None or stage["capture_failed"]:
        return
    if not bool(state.get("_attn_v1_16_use_cuda_graphs", True)):
        stage["capture_failed"] = True
        return

    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                _launch_no_buffer(
                    shared, layout, h_q, k, groups, groups_pow, num_splits,
                    anchor_s, scale, parents_per_prog, s_subspaces,
                )
        current.wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_no_buffer(
                shared, layout, h_q, k, groups, groups_pow, num_splits,
                anchor_s, scale, parents_per_prog, s_subspaces,
            )
        stage["graph"] = graph
    except Exception:
        stage["capture_failed"] = True


def _capture_with_buffer_graph(
    state: dict,
    shared: dict,
    stage: dict,
    layout: dict,
    h_q: int,
    k: int,
    groups: int,
    groups_pow: int,
    num_splits: int,
    anchor_s: int,
    scale: float,
    parents_per_prog: int,
    s_subspaces: int,
    bucket: int,
    buf_cols: int,
    buf_warps: int,
    buf_stages: int,
) -> None:
    if stage["graph"] is not None or stage["capture_failed"]:
        return
    if not bool(state.get("_attn_v1_16_use_cuda_graphs", True)):
        stage["capture_failed"] = True
        return

    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                _launch_with_buffer(
                    shared, stage, layout, h_q, k, groups, groups_pow, num_splits,
                    anchor_s, scale, parents_per_prog, s_subspaces,
                    bucket, buf_cols, buf_warps, buf_stages,
                )
        current.wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_with_buffer(
                shared, stage, layout, h_q, k, groups, groups_pow, num_splits,
                anchor_s, scale, parents_per_prog, s_subspaces,
                bucket, buf_cols, buf_warps, buf_stages,
            )
        stage["graph"] = graph
    except Exception:
        stage["capture_failed"] = True


def _get_fixed_runtime(
    state: dict,
    q: torch.Tensor,
    th: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    num_splits: int,
) -> dict:
    cache_key = _fixed_cache_key(state, q, th, q_head_to_kv, num_splits)
    cache = state.setdefault("_attn_v1_16_fixed", {})
    fixed = cache.get("fixed")
    if cache.get("key") == cache_key and fixed is not None:
        return fixed

    layout = get_layout_attn_rawq(
        state,
        q_head_to_kv,
        q,
        cache_name="_attn_v1_16_layout",
    )
    _require_supported(layout)

    groups = int(layout["groups"])
    groups_pow = max(next_pow2(groups), _GROUPS_POW_FLOOR)
    anchor_s = int(layout["anchor_subspace"])
    s_subspaces = int(layout["num_subspaces"])
    parents_per_prog = _parents_per_prog_for_bf(int(layout["bf"]))
    shared = _make_shared_workspace(layout, q, th, num_splits)

    fixed = {
        "layout": layout,
        "shared": shared,
        "groups": groups,
        "groups_pow": groups_pow,
        "anchor_s": anchor_s,
        "s_subspaces": s_subspaces,
        "parents_per_prog": parents_per_prog,
        "buckets": {},        # bucket -> stage dict (includes graph)
        "no_buffer_stage": None,
    }
    cache["key"] = cache_key
    cache["fixed"] = fixed
    return fixed


def _copy_buffer_into_stage(
    stage: dict,
    buffer_keys_eff: torch.Tensor,
    buffer_values_eff: torch.Tensor,
    bucket: int,
) -> None:
    l_buf = buffer_keys_eff.shape[1]
    # Keys: (H_kv_eff, L_buf, D) → stage layout (H_kv_eff, D, L_buf).
    # `copy_` handles strided sources, so we skip the .contiguous() alloc.
    # Padded slots are never read by the buffer kernel (every tl.load is
    # gated by `buf_invalid`), so we don't zero the padded regions of
    # buf_keys_t / buf_values — those 2 kernel launches were redundant.
    keys_src = buffer_keys_eff.transpose(-1, -2)
    if keys_src.dtype != torch.float16:
        keys_src = keys_src.to(torch.float16)
    stage["buf_keys_t"][:, :, :l_buf].copy_(keys_src)

    values_src = buffer_values_eff
    if values_src.dtype != torch.float16:
        values_src = values_src.to(torch.float16)
    stage["buf_values"][:, :l_buf, :].copy_(values_src)

    stage["buf_invalid"].fill_(1)
    stage["buf_invalid"][:, :l_buf].zero_()


def _empty_buffer(buffer_keys, buffer_values) -> bool:
    return (
        buffer_keys is None
        or buffer_values is None
        or buffer_keys.shape[1] == 0
    )


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
        raise RuntimeError("attention_v1_16 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("attention_v1_16 requires build_v2-style state")

    h_q = q.shape[0]
    d = q.shape[1]
    q_c = q if q.is_contiguous() else q.contiguous()

    fixed_probe = state.get("_attn_v1_16_fixed", {}).get("fixed")
    s_hint = int(fixed_probe["s_subspaces"]) if fixed_probe is not None else None
    if s_hint is None:
        # First call: infer S from threshold tensor or state.
        if th_per_subspace.dim() == 2 and th_per_subspace.shape[0] in _ALLOWED_S:
            s_hint = int(th_per_subspace.shape[0])
        else:
            s_hint = int(th_per_subspace.numel() // h_q)
        if s_hint not in _ALLOWED_S:
            raise RuntimeError(
                f"attention_v1_16 requires S in {_ALLOWED_S}; inferred S={s_hint}"
            )

    if th_per_subspace.shape == (s_hint, h_q) and th_per_subspace.is_contiguous():
        th_view = th_per_subspace
    else:
        th_view = th_per_subspace.reshape(s_hint, h_q).contiguous()

    fixed = _get_fixed_runtime(state, q_c, th_view, q_head_to_kv, num_splits)
    layout = fixed["layout"]
    shared = fixed["shared"]
    groups = fixed["groups"]
    groups_pow = fixed["groups_pow"]
    anchor_s = fixed["anchor_s"]
    s_subspaces = fixed["s_subspaces"]
    parents_per_prog = fixed["parents_per_prog"]
    k = int(layout["K"])

    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale = float(scale)

    shared["static_q"].copy_(q_c)
    shared["static_th"].copy_(th_view)

    empty = _empty_buffer(buffer_keys, buffer_values)
    if empty:
        stage = fixed.get("no_buffer_stage")
        if stage is None:
            stage = {"graph": None, "capture_failed": False}
            fixed["no_buffer_stage"] = stage
        _capture_no_buffer_graph(
            state, shared, stage, layout, h_q, k, groups, groups_pow,
            num_splits, anchor_s, scale, parents_per_prog, s_subspaces,
        )
        if stage["graph"] is not None:
            stage["graph"].replay()
        else:
            _launch_no_buffer(
                shared, layout, h_q, k, groups, groups_pow, num_splits,
                anchor_s, scale, parents_per_prog, s_subspaces,
            )
        return shared["out"]

    l_buf = int(buffer_keys.shape[1])
    bucket = _bucket_for(l_buf)

    stage = fixed["buckets"].get(bucket)
    if stage is None:
        stage = _make_bucket_staging(layout, q_c, bucket)
        fixed["buckets"][bucket] = stage

    buf_keys_eff, buf_values_eff = _prepare_buffer_effective(
        buffer_keys, buffer_values, layout, q_head_to_kv,
    )
    _copy_buffer_into_stage(stage, buf_keys_eff, buf_values_eff, bucket)

    cfg = _buffer_cfg(state, bucket)
    _capture_with_buffer_graph(
        state, shared, stage, layout, h_q, k, groups, groups_pow,
        num_splits, anchor_s, scale, parents_per_prog, s_subspaces,
        bucket, cfg["cols"], cfg["num_warps"], cfg["num_stages"],
    )
    if stage["graph"] is not None:
        stage["graph"].replay()
    else:
        _launch_with_buffer(
            shared, stage, layout, h_q, k, groups, groups_pow, num_splits,
            anchor_s, scale, parents_per_prog, s_subspaces,
            bucket, cfg["cols"], cfg["num_warps"], cfg["num_stages"],
        )
    return shared["out"]


KERNEL = attend
