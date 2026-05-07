"""JIT loader for the v1.46 native CUDA fused anchor-gate + attention kernel."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_EXT = None
_EXT_ERROR: Exception | None = None
_SRC = Path(__file__).with_name("_cuda_attn_v1_46.cu")


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
        cc = torch.cuda.get_device_capability()
        sm = cc[0] * 10 + cc[1]
        arch_flag = f"-gencode=arch=compute_{sm},code=sm_{sm}"
        _EXT = load(
            name="hira_cuda_attn_v1_46",
            sources=[str(_SRC)],
            verbose=False,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                arch_flag,
                "-std=c++17",
            ],
        )
        return _EXT
    except Exception as exc:
        _EXT_ERROR = exc
        raise


def is_supported(q: torch.Tensor, layout: dict) -> bool:
    return (
        q.is_cuda
        and q.dtype == torch.float16
        and q.shape[1] == 128
        and int(layout["D_v"]) == 128
        and int(layout["bf"]) == 4
        and int(layout["groups"]) <= 8
    )


def run_fused_attn_index_cuda_v1_46(
    q: torch.Tensor,
    keys_blocks: torch.Tensor,
    values_blocks: torch.Tensor,
    centers_anchor: torch.Tensor,
    radii_anchor: torch.Tensor,
    th_anchor: torch.Tensor,
    qnorm_anchor: torch.Tensor,
    invalid_blocks: torch.Tensor,
    h_q: int,
    h_kv: int,
    k: int,
    num_splits: int,
    groups: int,
    dim_offset: int,
    anchor_width: int,
    scale: float,
    out_m: torch.Tensor,
    out_l: torch.Tensor,
    out_o: torch.Tensor,
    cols_per_chunk: int = 128,
) -> None:
    mod = load_ext()
    mod.sparse_attn_v1_46_index(
        q, keys_blocks, values_blocks,
        centers_anchor, radii_anchor,
        th_anchor, qnorm_anchor,
        invalid_blocks,
        out_m, out_l, out_o,
        h_q, h_kv, k, num_splits, groups,
        dim_offset, anchor_width, float(scale),
        cols_per_chunk,
    )


def run_attn_reduce_cuda_v1_46(
    m_idx: torch.Tensor,
    l_idx: torch.Tensor,
    o_idx: torch.Tensor,
    m_buf: torch.Tensor,
    l_buf: torch.Tensor,
    o_buf: torch.Tensor,
    out: torch.Tensor,
    num_splits: int,
) -> None:
    mod = load_ext()
    mod.sparse_attn_v1_46_reduce(
        m_idx, l_idx, o_idx,
        m_buf, l_buf, o_buf, out,
        num_splits,
    )
