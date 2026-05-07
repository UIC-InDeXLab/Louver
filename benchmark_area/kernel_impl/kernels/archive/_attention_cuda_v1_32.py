"""JIT loader for the v1.32 CUDA sparse attention index kernel."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_EXT = None
_EXT_ERROR: Exception | None = None
_SRC = Path(__file__).with_name("_cuda_attn_v1_32_simt.cu")


def _keys_blocks_cuda(layout: dict) -> torch.Tensor:
    keys_cuda = layout.get("_keys_blocks_f16_cuda")
    src = layout["keys_blocks_t_f16"]
    key = (src.data_ptr(), tuple(src.shape))
    if keys_cuda is not None and layout.get("_keys_blocks_f16_cuda_key") == key:
        return keys_cuda
    keys_cuda = src.permute(0, 1, 3, 2).contiguous()
    layout["_keys_blocks_f16_cuda"] = keys_cuda
    layout["_keys_blocks_f16_cuda_key"] = key
    return keys_cuda


def _ensure_ninja_on_path() -> None:
    bin_dir = str(Path(sys.executable).resolve().parent)
    path = os.environ.get("PATH", "")
    if bin_dir not in path.split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + path


def load_ext():
    global _EXT, _EXT_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_ERROR is not None:
        raise _EXT_ERROR

    from torch.utils.cpp_extension import load

    _ensure_ninja_on_path()
    try:
        _EXT = load(
            name="hira_cuda_attn_v1_32_simt",
            sources=[str(_SRC)],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        )
        return _EXT
    except Exception as exc:  # pragma: no cover - build/runtime dependent
        _EXT_ERROR = exc
        raise


def is_supported(
    q: torch.Tensor,
    layout: dict,
) -> bool:
    return (
        q.is_cuda
        and q.dtype == torch.float16
        and q.shape[1] == 128
        and int(layout["D_v"]) == 128
        and int(layout["bf"]) == 4
        and int(layout["num_subspaces"]) == 8
        and int(layout["groups"]) <= 8
    )


def run_fused_attn_index_cuda_v1_32(
    q: torch.Tensor,
    layout: dict,
    cluster_pass: torch.Tensor,
    h_q: int,
    groups: int,
    num_splits: int,
    anchor_s: int,
    scale: float,
    out_m: torch.Tensor,
    out_l: torch.Tensor,
    out_o: torch.Tensor,
) -> None:
    mod = load_ext()
    mod.sparse_attn_index_v1_32(
        q,
        _keys_blocks_cuda(layout),
        layout["values_blocks_f16"],
        cluster_pass,
        layout["invalid_blocks_i8"],
        out_m,
        out_l,
        out_o,
        h_q,
        int(layout["base_heads"]),
        int(layout["K"]),
        num_splits,
        groups,
        anchor_s,
        float(scale),
    )
