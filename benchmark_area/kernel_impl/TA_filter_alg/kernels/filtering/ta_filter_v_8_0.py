"""ta_filter_v8.0 — auto-dispatcher between v7.10 (TILE_N=2048) and v7.11 (TILE_N=4096).

v7.10 wins on small/medium grids. v7.11 wins when the cooperative grid gets
big enough that TILE_N=2048's larger N_TILES costs more than TILE_N=4096's
heavier per-block work.

Heuristic: cooperative grid for v7.10 = Hq × max(4, N_TILES_2048). When this
exceeds GRID_BLOCKS_BOUNDARY, switch to v7.11.

Empirical (RTX 5090, bf=4, S=4):
  Llama 4k  Hq=24 N_TILES=3 → 24*max(4,3)=96  → v7.10 (16.4 vs 16.9)
  Llama 8k  Hq=24 N_TILES=5 → 120             → v7.10 (18.7 vs 20.5)
  Qwen  8k  Hq=28 N_TILES=5 → 140             → v7.10 (20.4 vs 20.5)
  Llama 12k Hq=24 N_TILES=7 → 168             → v7.10 (22.6 vs 24.3)
  Qwen  12k Hq=28 N_TILES=7 → 196             → v7.11 (24.6 vs 28.8)
  Qwen  20k Hq=28 N_TILES=11→ 308             → v7.11 (30.7 vs 42.0)

Boundary B = 180 (covers all observed cases).

Dependencies inlined from:
  _ta_filter_cuda_v7_10.py  — ext loader
  ta_filter_v_3_4.py        — _build_packed_assigns
  ta_filter_v_7_10.py       — ta_filter_v7_10
  ta_filter_v_7_11.py       — ta_filter_v7_11
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

KERNEL_VERSION = "v8.0"
GRID_BLOCKS_BOUNDARY = 180
TILE_N_V710 = 2048

# ──────────────────────────────────────────────────────────────────────────────
# CUDA extension loader (inlined from _ta_filter_cuda_v7_10.py)
# ──────────────────────────────────────────────────────────────────────────────

_EXT = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    base = Path(__file__).resolve().parent
    # Detect CUDA_HOME: prefer env var, then search common paths
    if "CUDA_HOME" not in os.environ:
        for _p in ["/usr/local/cuda", "/usr/local/cuda-12.8", "/usr/local/cuda-12.4", "/usr/local/cuda-12"]:
            if os.path.isdir(_p):
                os.environ["CUDA_HOME"] = _p
                break
    os.environ["PATH"] = f"{os.environ.get('CUDA_HOME', '/usr/local/cuda')}/bin:" + os.environ.get("PATH", "")
    # Detect arch from current GPU; fall back to env var or 8.0 (A100 safe default)
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        try:
            maj, min_ = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{maj}.{min_}"
        except Exception:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    _EXT = load(
        name="ta_filter_cuda_stage1_v7_10",
        sources=[
            str(base / "ta_filter_cuda_v7_10.cpp"),
            str(base / "ta_filter_cuda_kernel_v7_10.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _EXT


# ──────────────────────────────────────────────────────────────────────────────
# Packed-assigns builder (inlined from ta_filter_v_3_4._build_packed_assigns)
# ──────────────────────────────────────────────────────────────────────────────

_PACKED_KEY = "_assigns_packed_u64_v34"


def _build_packed_assigns(state: dict) -> torch.Tensor:
    cached = state.get(_PACKED_KEY)
    if cached is not None:
        return cached
    a = state["assigns_padded"]   # [4, Hkv, Npad] int16/int32
    inv = state["invalid_mask"]   # [Hkv, Npad] bool
    if a.dtype not in (torch.int16, torch.int32):
        raise TypeError("v3.4 requires assigns int16 or int32")
    a64 = a.to(torch.int64) & 0xFFFF
    packed = (a64[0]
              | (a64[1] << 16)
              | (a64[2] << 32)
              | (a64[3] << 48))
    sentinel_all = torch.tensor(-1, dtype=torch.int64, device=a.device)
    packed = torch.where(inv, sentinel_all, packed).contiguous()
    state[_PACKED_KEY] = packed
    return packed


# ──────────────────────────────────────────────────────────────────────────────
# v7.10 — TILE_N=2048 (inlined from ta_filter_v_7_10.py)
# ──────────────────────────────────────────────────────────────────────────────

_L_FORCE = 256
_CACHE_V710: dict[tuple, dict[str, torch.Tensor]] = {}


def _workspace_v710(*, device, h_q, s_sub, n_pad):
    key = (device.index, h_q, s_sub, _L_FORCE, n_pad)
    ws = _CACHE_V710.get(key)
    if ws is not None:
        return ws
    ws = {
        "top_scores":  torch.empty(h_q, s_sub, _L_FORCE, device=device, dtype=torch.float32),
        "top_indices": torch.empty(h_q, s_sub, _L_FORCE, device=device, dtype=torch.int32),
        "depth":       torch.empty(h_q, device=device, dtype=torch.int32),
        "live_idx":    torch.empty(h_q, n_pad, device=device, dtype=torch.int32),
        "live_count":  torch.zeros(h_q, device=device, dtype=torch.int32),
    }
    _CACHE_V710.clear()
    _CACHE_V710[key] = ws
    return ws


def ta_filter_v7_10(q, threshold, state, q_head_to_kv=None, *, return_aux=False):
    if q.dtype != torch.float16:
        raise TypeError("ta_filter_v7_10 expects q as fp16")
    if threshold.dtype != torch.float32:
        threshold = threshold.float()
    q = q.contiguous()
    threshold = threshold.contiguous()

    centers = state["centers_padded_f16"].contiguous()
    dim_offsets = state["dim_offsets"].contiguous()
    dim_widths = state["dim_widths"].contiguous()

    h_q, _d = q.shape
    s_sub, _h_kv, k_clusters, _mw = centers.shape
    n_pad = int(state["N_pad"])

    if int(s_sub) != 4 or int(state["bf"]) != 4:
        raise ValueError("ta_filter_v7_10 requires S=4, bf=4")

    assigns_packed = _build_packed_assigns(state)

    if q_head_to_kv is None:
        q_head_to_kv = torch.empty(0, device=q.device, dtype=torch.long)
    else:
        q_head_to_kv = q_head_to_kv.contiguous()

    ws = _workspace_v710(device=q.device, h_q=h_q, s_sub=int(s_sub), n_pad=n_pad)
    ext = _load_ext()

    K_stride = int(centers.shape[2])
    K_used = int(state.get("K_used", k_clusters))
    K_eff = min(K_used, K_stride) if K_used > 0 else K_stride
    N_eff = int(state.get("N_used", n_pad))
    ext.fused_pipeline(
        q, centers, dim_offsets, dim_widths, q_head_to_kv, threshold,
        assigns_packed,
        ws["top_scores"], ws["top_indices"], ws["depth"], ws["live_idx"], ws["live_count"],
        int(K_eff), int(K_stride), int(N_eff), 2048,
    )

    if return_aux:
        return ws["live_idx"], ws["live_count"], ws["depth"]
    return ws["live_idx"], ws["live_count"]


# v7.11 wrapper uses _TILE_N_V711=4096 and same K_eff/N_eff pattern.
# (Kept above — replace_all caught both call sites.)
_v7_11_marker = True


# ──────────────────────────────────────────────────────────────────────────────
# v7.11 — TILE_N=4096 (inlined from ta_filter_v_7_11.py)
# ──────────────────────────────────────────────────────────────────────────────

_TILE_N_V711 = 4096
_CACHE_V711: dict[tuple, dict[str, torch.Tensor]] = {}


def _workspace_v711(*, device, h_q, s_sub, n_pad):
    key = (device.index, h_q, s_sub, _L_FORCE, n_pad)
    ws = _CACHE_V711.get(key)
    if ws is not None:
        return ws
    ws = {
        "top_scores":  torch.empty(h_q, s_sub, _L_FORCE, device=device, dtype=torch.float32),
        "top_indices": torch.empty(h_q, s_sub, _L_FORCE, device=device, dtype=torch.int32),
        "depth":       torch.empty(h_q, device=device, dtype=torch.int32),
        "live_idx":    torch.empty(h_q, n_pad, device=device, dtype=torch.int32),
        "live_count":  torch.zeros(h_q, device=device, dtype=torch.int32),
    }
    _CACHE_V711.clear()
    _CACHE_V711[key] = ws
    return ws


def ta_filter_v7_11(q, threshold, state, q_head_to_kv=None, *, return_aux=False):
    if q.dtype != torch.float16:
        raise TypeError("ta_filter_v7_11 expects q as fp16")
    if threshold.dtype != torch.float32:
        threshold = threshold.float()
    q = q.contiguous()
    threshold = threshold.contiguous()

    centers = state["centers_padded_f16"].contiguous()
    dim_offsets = state["dim_offsets"].contiguous()
    dim_widths = state["dim_widths"].contiguous()

    h_q, _d = q.shape
    s_sub, _h_kv, k_clusters, _mw = centers.shape
    n_pad = int(state["N_pad"])

    if int(s_sub) != 4 or int(state["bf"]) != 4:
        raise ValueError("ta_filter_v7_11 requires S=4, bf=4")

    assigns_packed = _build_packed_assigns(state)

    if q_head_to_kv is None:
        q_head_to_kv = torch.empty(0, device=q.device, dtype=torch.long)
    else:
        q_head_to_kv = q_head_to_kv.contiguous()

    ws = _workspace_v711(device=q.device, h_q=h_q, s_sub=int(s_sub), n_pad=n_pad)
    ext = _load_ext()

    K_stride = int(centers.shape[2])
    K_used = int(state.get("K_used", k_clusters))
    K_eff = min(K_used, K_stride) if K_used > 0 else K_stride
    ext.fused_pipeline(
        q, centers, dim_offsets, dim_widths, q_head_to_kv, threshold,
        assigns_packed,
        ws["top_scores"], ws["top_indices"], ws["depth"], ws["live_idx"], ws["live_count"],
        int(K_eff), int(K_stride), _TILE_N_V711,
    )

    if return_aux:
        return ws["live_idx"], ws["live_count"], ws["depth"]
    return ws["live_idx"], ws["live_count"]


# ──────────────────────────────────────────────────────────────────────────────
# v8.0 dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def _pick_variant(n_pad: int, h_q: int):
    n_tiles = (n_pad + TILE_N_V710 - 1) // TILE_N_V710
    grid_blocks = h_q * max(4, n_tiles)
    return ta_filter_v7_11 if grid_blocks > GRID_BLOCKS_BOUNDARY else ta_filter_v7_10


def ta_filter_v8_0(q, threshold, state, q_head_to_kv=None, *, return_aux=False):
    h_q = int(q.shape[0])
    n_pad = int(state["N_pad"])
    fn = _pick_variant(n_pad, h_q)
    return fn(q, threshold, state, q_head_to_kv, return_aux=return_aux)
