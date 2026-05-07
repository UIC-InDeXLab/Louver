"""Shared utilities for offloading experiments: timing and GPU memory accounting."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class OffloadStats:
    """Per-step timing and GPU memory. Accumulate over decode steps, then average."""
    search_ms: list[float] = field(default_factory=list)    # GPU filter or CPU index search
    transfer_ms: list[float] = field(default_factory=list)  # CPU→GPU transfer of retrieved KV
    gpu_bytes: int = 0                                       # persistent GPU bytes (set once)

    def record(self, search_ms: float, transfer_ms: float) -> None:
        self.search_ms.append(search_ms)
        self.transfer_ms.append(transfer_ms)

    def mean_search_ms(self) -> float:
        return sum(self.search_ms) / len(self.search_ms) if self.search_ms else 0.0

    def mean_transfer_ms(self) -> float:
        return sum(self.transfer_ms) / len(self.transfer_ms) if self.transfer_ms else 0.0

    def summary(self) -> dict:
        return {
            "search_ms": round(self.mean_search_ms(), 4),
            "transfer_ms": round(self.mean_transfer_ms(), 4),
            "total_ms": round(self.mean_search_ms() + self.mean_transfer_ms(), 4),
            "gpu_mb": round(self.gpu_bytes / 1e6, 3),
            "n_steps": len(self.search_ms),
        }


def cpu_timer() -> float:
    """Current time in ms (CPU wall clock)."""
    return time.perf_counter() * 1000.0


def gpu_sync_timer(device: torch.device | str = "cuda") -> float:
    """Sync CUDA then return wall-clock ms."""
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return time.perf_counter() * 1000.0


def tensor_bytes(*tensors: torch.Tensor | None) -> int:
    """Total bytes of all given tensors (None tensors skipped)."""
    return sum(t.element_size() * t.numel() for t in tensors if t is not None)
