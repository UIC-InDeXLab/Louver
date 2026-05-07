from .base import BaseSearcher

__all__ = [
    "BaseSearcher",
    "CUDASearcher",
    "CPUSearcher",
]


def __getattr__(name: str):
    if name == "CUDASearcher":
        from .cuda import CUDASearcher

        return CUDASearcher
    if name == "CPUSearcher":
        from .cpu import CPUSearcher

        return CPUSearcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
