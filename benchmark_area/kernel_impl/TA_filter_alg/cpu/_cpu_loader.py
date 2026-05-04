"""JIT extension loader for the CPU TA-filter pipeline.

Each kernel ships its own .cpp; we cache the compiled module per name.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load


_venv_bin = Path(sys.executable).parent
if str(_venv_bin) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_venv_bin) + os.pathsep + os.environ.get("PATH", "")

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions_ta_cpu")
os.environ.setdefault("MAX_JOBS", "8")
Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)

_DIR = Path(__file__).resolve().parent

_CFLAGS = [
    "-O3",
    "-march=native",
    "-mtune=native",
    "-ffast-math",
    "-fopenmp",
    "-funroll-loops",
    "-fno-trapping-math",
    "-fno-math-errno",
    "-mavx512f",
    "-mavx512bw",
    "-mavx512vl",
    "-mavx512dq",
    "-mavx512bf16",
]
_LDFLAGS = ["-fopenmp"]


@lru_cache(maxsize=None)
def load_ext(name: str, *sources: str):
    src_paths = [str(_DIR / s) for s in sources]
    for sp in src_paths:
        if not Path(sp).exists():
            raise FileNotFoundError(f"Missing source: {sp}")
    return load(
        name=name,
        sources=src_paths,
        extra_cflags=list(_CFLAGS),
        extra_ldflags=list(_LDFLAGS),
        verbose=False,
    )
