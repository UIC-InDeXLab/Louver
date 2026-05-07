"""CPU C++ kernels and micro-benchmarks for the kernel_impl experiments."""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable

_BUILD_KERNELS: dict[str, "KernelInfo"] = {}
_UPDATE_KERNELS: dict[str, "KernelInfo"] = {}
_ATTENTION_KERNELS: dict[str, "KernelInfo"] = {}


@dataclass
class KernelInfo:
    name: str
    version: str
    fn: Callable


def _discover(prefix: str, registry: dict[str, KernelInfo]) -> None:
    pkg_mod = importlib.import_module(__name__)
    for info in pkgutil.iter_modules(pkg_mod.__path__):
        if not info.name.startswith(prefix):
            continue
        mod = importlib.import_module(f"{__name__}.{info.name}")
        if hasattr(mod, "KERNEL") and hasattr(mod, "KERNEL_VERSION"):
            registry[info.name] = KernelInfo(info.name, mod.KERNEL_VERSION, mod.KERNEL)


def discover_all() -> None:
    _BUILD_KERNELS.clear()
    _UPDATE_KERNELS.clear()
    _ATTENTION_KERNELS.clear()
    _discover("build_v", _BUILD_KERNELS)
    _discover("update_v", _UPDATE_KERNELS)
    _discover("attention_v", _ATTENTION_KERNELS)


def build_kernels() -> dict[str, KernelInfo]:
    if not _BUILD_KERNELS:
        discover_all()
    return dict(_BUILD_KERNELS)


def update_kernels() -> dict[str, KernelInfo]:
    if not _UPDATE_KERNELS:
        discover_all()
    return dict(_UPDATE_KERNELS)


def attention_kernels() -> dict[str, KernelInfo]:
    if not _ATTENTION_KERNELS:
        discover_all()
    return dict(_ATTENTION_KERNELS)

