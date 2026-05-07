"""TA-filter (Fagin/threshold-style) attention algorithm kernels.

The TA-filter algorithm scores keys whose centroid scores in any subspace rank
in the top L* sorted positions, where L* is the first row at which the sum of
sorted centroid scores across subspaces falls below a scalar threshold T.
The final survivor set is { k : q.k >= T }, and attention is computed as
softmax over the survivor scores weighted by their values.

See ``benchmark_area/kernel_impl/TA_filter_algorithm.md`` for the full spec.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_BUILDS: dict[str, "KernelInfo"] = {}
_ATTNS: dict[str, "KernelInfo"] = {}


@dataclass
class KernelInfo:
    name: str
    version: str
    fn: Callable


def _discover(prefix: str, registry: dict) -> None:
    pkg = importlib.import_module(f"{__name__}.kernels")
    root = Path(next(iter(pkg.__path__)))
    for info in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg.__name__}."):
        mod_name = info.name
        short_name = mod_name.rsplit(".", 1)[-1]
        if not short_name.startswith(prefix):
            continue
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, "KERNEL") or not hasattr(mod, "KERNEL_VERSION"):
            continue
        try:
            module_file = Path(mod.__file__).resolve()
            rel = module_file.relative_to(root).with_suffix("")
            key = ".".join(rel.parts)
        except Exception:
            key = short_name
        registry[key] = KernelInfo(
            name=key, version=mod.KERNEL_VERSION, fn=mod.KERNEL
        )


def discover_all() -> None:
    _BUILDS.clear()
    _ATTNS.clear()
    _discover("TA_build_v", _BUILDS)
    _discover("TA_attention_v", _ATTNS)


def build_kernels() -> dict[str, KernelInfo]:
    if not _BUILDS:
        discover_all()
    return dict(_BUILDS)


def attention_kernels() -> dict[str, KernelInfo]:
    if not _ATTNS:
        discover_all()
    return dict(_ATTNS)
