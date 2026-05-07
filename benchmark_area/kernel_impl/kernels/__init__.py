"""Kernel implementations for subspace k-center index.

Discovery: any module in this package named build_v*.py, search_v*.py,
update_v*.py, or attention_v*.py that exposes a top-level KERNEL and
KERNEL_VERSION is auto-registered.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable

_BUILD_KERNELS: dict[str, Callable] = {}
_SEARCH_KERNELS: dict[str, Callable] = {}
_UPDATE_KERNELS: dict[str, Callable] = {}
_ATTENTION_KERNELS: dict[str, Callable] = {}


@dataclass
class KernelInfo:
    name: str
    version: str
    fn: Callable


def _discover(prefix: str, registry: dict):
    package = __name__
    pkg_mod = importlib.import_module(package)
    for info in pkgutil.iter_modules(pkg_mod.__path__):
        if not info.name.startswith(prefix):
            continue
        mod = importlib.import_module(f"{package}.{info.name}")
        if not hasattr(mod, "KERNEL") or not hasattr(mod, "KERNEL_VERSION"):
            continue
        registry[info.name] = KernelInfo(
            name=info.name, version=mod.KERNEL_VERSION, fn=mod.KERNEL
        )


def discover_all():
    _BUILD_KERNELS.clear()
    _SEARCH_KERNELS.clear()
    _UPDATE_KERNELS.clear()
    _ATTENTION_KERNELS.clear()
    _discover("build_v", _BUILD_KERNELS)
    _discover("search_v", _SEARCH_KERNELS)
    _discover("update_v", _UPDATE_KERNELS)
    _discover("attention_v", _ATTENTION_KERNELS)


def build_kernels() -> dict[str, KernelInfo]:
    if not _BUILD_KERNELS:
        discover_all()
    return dict(_BUILD_KERNELS)


def search_kernels() -> dict[str, KernelInfo]:
    if not _SEARCH_KERNELS:
        discover_all()
    return dict(_SEARCH_KERNELS)


def update_kernels() -> dict[str, KernelInfo]:
    if not _UPDATE_KERNELS:
        discover_all()
    return dict(_UPDATE_KERNELS)


def attention_kernels() -> dict[str, KernelInfo]:
    if not _ATTENTION_KERNELS:
        discover_all()
    return dict(_ATTENTION_KERNELS)


def get_build(name: str) -> Callable:
    return build_kernels()[name].fn


def get_search(name: str) -> Callable:
    return search_kernels()[name].fn


def get_update(name: str) -> Callable:
    return update_kernels()[name].fn


def get_attention(name: str) -> Callable:
    return attention_kernels()[name].fn
