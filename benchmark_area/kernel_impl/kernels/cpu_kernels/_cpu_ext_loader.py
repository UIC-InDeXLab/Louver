"""Per-kernel JIT loader for CPU HIRA extensions.

Each attention kernel version has its own .cpp file so we can profile and tune
them independently. The build/update extension lives in `_index_ext.cpp` and is
loaded via `index_ext()`. Use `attention_ext(name)` to load (and cache) a
specific attention version's extension.
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

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
os.environ.setdefault("MAX_JOBS", "8")
Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)

_DIR = Path(__file__).resolve().parent

_BASE_CFLAGS = [
    "-O3",
    "-march=native",
    "-mtune=native",
    "-ffast-math",
    "-fopenmp",
    "-funroll-loops",
    "-fno-trapping-math",
    "-fno-math-errno",
]
_BASE_LDFLAGS = ["-fopenmp"]


@lru_cache(maxsize=None)
def index_ext():
    """JIT-build (or load cached) index extension exposing build/update_index."""
    return load(
        name="hira_cpu_index_ext",
        sources=[str(_DIR / "_index_ext.cpp")],
        extra_cflags=list(_BASE_CFLAGS),
        extra_ldflags=list(_BASE_LDFLAGS),
        verbose=False,
    )


@lru_cache(maxsize=None)
def attention_ext(version: str):
    """Load `_attention_<version>_ext.cpp` and cache the resulting module.

    `version` is something like "v1_0". We use the version in the module name
    so that the JIT cache is per-version (avoids cross-version stale builds).
    """
    src = _DIR / f"_attention_{version}_ext.cpp"
    if not src.exists():
        raise FileNotFoundError(f"Missing CPU attention source: {src}")
    return load(
        name=f"hira_cpu_attention_{version}_ext",
        sources=[str(src)],
        extra_cflags=list(_BASE_CFLAGS),
        extra_ldflags=list(_BASE_LDFLAGS),
        verbose=False,
    )
