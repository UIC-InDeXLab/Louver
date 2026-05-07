"""
CPU-index offloading baselines: RetrievalAttention (HNSW), InfLLM (IVF), MagicPIG (LSH).

All methods:
  - Build CPU index during prefill
  - KV pairs on CPU pinned memory
  - Per decode step: CPU index search (timed) → gather KV → CPU→GPU transfer (timed) → SDPA
  - GPU memory: ~0 (no persistent GPU tensors besides small buffer, not counted)
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.modeling_utils import AttentionInterface
from transformers import PreTrainedConfig

_HERE = Path(__file__).resolve().parent

# faiss import
import faiss

from base import OffloadStats, cpu_timer, gpu_sync_timer, tensor_bytes  # noqa: E402

BUDGET_FRACTION = 0.15
LSH_N_PLANES    = 64


# ── CPU index builders ────────────────────────────────────────────────────────

def _build_hnsw(k_np: np.ndarray, M: int = 16) -> faiss.IndexHNSWFlat:
    n, d = k_np.shape
    norms = np.linalg.norm(k_np, axis=1, keepdims=True) + 1e-12
    k_norm = k_np / norms
    idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = 40   # fast build; quality still ok for offload demo
    idx.add(k_norm.astype(np.float32))
    return idx


def _build_ivf(k_np: np.ndarray) -> faiss.IndexIVFFlat:
    n, d = k_np.shape
    nlist = max(1, min(int(math.sqrt(n)), n // 4))
    q     = faiss.IndexFlatIP(d)
    idx   = faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_INNER_PRODUCT)
    idx.train(k_np.astype(np.float32))
    idx.add(k_np.astype(np.float32))
    idx.nprobe = max(1, nlist // 4)
    return idx


def _build_lsh(k_np: np.ndarray, n_planes: int = LSH_N_PLANES, seed: int = 42) -> dict:
    rng    = np.random.RandomState(seed)
    d      = k_np.shape[1]
    planes = rng.randn(n_planes, d).astype(np.float32)
    planes /= np.linalg.norm(planes, axis=1, keepdims=True) + 1e-12
    key_bits = (k_np @ planes.T) >= 0    # (N, n_planes) bool
    return {"planes": planes, "key_bits": key_bits}


# ── CPU index queries → indices ───────────────────────────────────────────────

def _query_hnsw(idx: faiss.IndexHNSWFlat, q_np: np.ndarray, k: int) -> np.ndarray:
    q_norm = q_np / (np.linalg.norm(q_np) + 1e-12)
    idx.hnsw.efSearch = max(64, k * 2)
    _, I = idx.search(q_norm.reshape(1, -1).astype(np.float32), k)
    return I[0]


def _query_ivf(idx: faiss.IndexIVFFlat, q_np: np.ndarray, k: int) -> np.ndarray:
    _, I = idx.search(q_np.reshape(1, -1).astype(np.float32), k)
    return I[0]


def _query_lsh(lsh: dict, q_np: np.ndarray, k: int) -> np.ndarray:
    q_bits  = (q_np @ lsh["planes"].T) >= 0
    hamming = (lsh["key_bits"] != q_bits).sum(axis=1)
    return np.argsort(hamming)[:k]


# ── Per-layer cache ───────────────────────────────────────────────────────────

@dataclass
class ANNOffloadOutput:
    layer_cache: "ANNOffloadCacheLayer"
    is_prefill: bool
    prefill_keys: torch.Tensor | None = None
    prefill_values: torch.Tensor | None = None


class ANNOffloadCacheLayer(CacheLayerMixin):
    """Generic CPU-ANN offload cache layer. method ∈ {'hnsw','ivf','lsh'}."""

    def __init__(self, method: str, budget_fraction: float = BUDGET_FRACTION):
        super().__init__()
        self.method = method
        self.budget_fraction = budget_fraction
        self.stats = OffloadStats()

        self._cpu_keys: torch.Tensor | None = None
        self._cpu_values: torch.Tensor | None = None
        self._n_stored: int = 0
        self._cpu_indices: list | None = None   # one index per KV head
        self._seq_len: int = 0

    def lazy_initialization(self, key_states, value_states) -> None:
        pass

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               cache_kwargs: dict | None = None):
        B, H_kv, L, D = key_states.shape
        assert B == 1

        if not self.is_initialized:
            ks = key_states.squeeze(0).to(torch.float16).contiguous()  # (H_kv, L, D)
            vs = value_states.squeeze(0).to(torch.float16).contiguous()

            # Build CPU index per KV head
            k_np_all = ks.float().cpu().numpy()    # (H_kv, L, D)
            self._cpu_indices = []
            for h_kv in tqdm(range(H_kv), desc=f"  build {self.method.upper()} index",
                             leave=False, dynamic_ncols=True):
                k_np = k_np_all[h_kv]         # (L, D)
                if self.method == "hnsw":
                    self._cpu_indices.append(_build_hnsw(k_np))
                elif self.method == "ivf":
                    self._cpu_indices.append(_build_ivf(k_np))
                else:  # lsh
                    self._cpu_indices.append(_build_lsh(k_np))

            # CPU pinned KV store (pre-allocate with growth room)
            N_cap = L + 4096
            self._cpu_keys   = torch.zeros(H_kv, N_cap, D, dtype=torch.float16,
                                           pin_memory=True)
            self._cpu_values = torch.zeros(H_kv, N_cap, D, dtype=torch.float16,
                                           pin_memory=True)
            self._cpu_keys[:, :L, :]   = ks.cpu()
            self._cpu_values[:, :L, :] = vs.cpu()
            self._n_stored = L
            self._seq_len  = L
            self.is_initialized = True

            self.stats.gpu_bytes = 0  # no persistent GPU objects

            return ANNOffloadOutput(layer_cache=self, is_prefill=True,
                                    prefill_keys=key_states, prefill_values=value_states), None

        # Decode: append to CPU store only; re-query index (indices are static after prefill)
        assert L == 1
        new_k = key_states.squeeze(0).to(torch.float16)
        new_v = value_states.squeeze(0).to(torch.float16)

        if self._n_stored >= self._cpu_keys.shape[1]:
            grow = max(1024, self._cpu_keys.shape[1] // 2)
            D_   = self._cpu_keys.shape[2]
            ext_k = torch.zeros(new_k.shape[0], grow, D_, dtype=torch.float16, pin_memory=True)
            ext_v = torch.zeros_like(ext_k)
            self._cpu_keys   = torch.cat([self._cpu_keys, ext_k], dim=1)
            self._cpu_values = torch.cat([self._cpu_values, ext_v], dim=1)

        self._cpu_keys[:, self._n_stored:self._n_stored+1, :]   = new_k.cpu()
        self._cpu_values[:, self._n_stored:self._n_stored+1, :] = new_v.cpu()
        self._n_stored += 1
        self._seq_len  += 1

        return ANNOffloadOutput(layer_cache=self, is_prefill=False), None

    def get_seq_length(self) -> int:
        return self._seq_len

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, cache_position):
        q_len = cache_position.shape[0] if hasattr(cache_position, "shape") else 1
        return self._seq_len + q_len, 0

    def reset(self):
        self._cpu_keys = None
        self._cpu_values = None
        self._cpu_indices = None
        self._n_stored = 0
        self._seq_len = 0
        self.is_initialized = False
        self.stats = OffloadStats()


class ANNOffloadCache(Cache):
    def __init__(self, model_config: PreTrainedConfig, method: str,
                 budget_fraction: float = BUDGET_FRACTION):
        config = model_config.get_text_config(decoder=True)
        n = config.num_hidden_layers
        layers = [ANNOffloadCacheLayer(method, budget_fraction) for _ in range(n)]
        super().__init__(layers=layers, offloading=False, offload_only_non_sliding=None)
        self.method = method

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        out, _ = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)
        return out, None

    def aggregate_stats(self) -> dict:
        all_search   = []
        all_transfer = []
        for layer in self.layers:
            all_search.extend(layer.stats.search_ms)
            all_transfer.extend(layer.stats.transfer_ms)
        n = max(len(all_search), 1)
        return {
            "search_ms":   round(sum(all_search) / n, 4),
            "transfer_ms": round(sum(all_transfer) / n, 4),
            "gpu_mb":      0.0,
        }


# ── Attention forward ─────────────────────────────────────────────────────────

def ann_offload_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: ANNOffloadOutput,
    value: None,
    attention_mask,
    dropout: float,
    scaling: float,
    **kwargs,
):
    if key.is_prefill:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(
            module, query, key.prefill_keys, key.prefill_values,
            attention_mask, dropout, scaling=scaling, **kwargs,
        )

    layer: ANNOffloadCacheLayer = key.layer_cache
    device = query.device
    H_q    = query.shape[1]
    H_kv   = len(layer._cpu_indices)
    D      = query.shape[-1]

    q2d = query.squeeze(0).squeeze(-2).float().cpu().numpy()  # (H_q, D)

    k_budget = max(1, int(layer.budget_fraction * layer._n_stored))

    if not hasattr(module, "_ann_q2kv") or module._ann_q2kv.shape[0] != H_q:
        g = H_q // H_kv
        module._ann_q2kv = torch.arange(H_q, dtype=torch.int64) // g

    q_head_to_kv = module._ann_q2kv

    # ── CPU index search — once per h_kv, reused for all h_q in group (timed) ──
    # Use mean query over the group as the representative query for search.
    g = H_q // H_kv
    q_per_kv = q2d.reshape(H_kv, g, D)   # (H_kv, g, D)

    t0 = cpu_timer()
    indices_per_kv = []
    for h_kv in range(H_kv):
        q_rep = q_per_kv[h_kv].mean(0)    # (D,) — representative query
        idx   = layer._cpu_indices[h_kv]
        if layer.method == "hnsw":
            idxs = _query_hnsw(idx, q_rep, k_budget)
        elif layer.method == "ivf":
            idxs = _query_ivf(idx, q_rep, k_budget)
        else:
            idxs = _query_lsh(idx, q_rep, k_budget)
        idxs = idxs[idxs < layer._n_stored]
        indices_per_kv.append(idxs)
    search_ms = cpu_timer() - t0

    # ── Gather per h_kv + transfer (timed) ───────────────────────────────────
    max_len = max((len(i) for i in indices_per_kv), default=1)

    gk_kv = []   # (H_kv, max_len, D) CPU
    gv_kv = []
    for h_kv in range(H_kv):
        idxs = indices_per_kv[h_kv]
        kt = layer._cpu_keys[h_kv][idxs]
        vt = layer._cpu_values[h_kv][idxs]
        if kt.shape[0] < max_len:
            pad = torch.zeros(max_len - kt.shape[0], D, dtype=torch.float16)
            kt = torch.cat([kt, pad], 0); vt = torch.cat([vt, pad], 0)
        gk_kv.append(kt); gv_kv.append(vt)

    gk_kv_cpu = torch.stack(gk_kv, 0)   # (H_kv, max_len, D)
    gv_kv_cpu = torch.stack(gv_kv, 0)

    # Expand to H_q by repeating each kv block g times
    gk_cpu = gk_kv_cpu.repeat_interleave(g, dim=0)   # (H_q, max_len, D)
    gv_cpu = gv_kv_cpu.repeat_interleave(g, dim=0)

    t_start = gpu_sync_timer(device)
    gk_gpu  = gk_cpu.to(device, non_blocking=False)
    gv_gpu  = gv_cpu.to(device, non_blocking=False)
    torch.cuda.synchronize(device)
    transfer_ms = gpu_sync_timer(device) - t_start

    layer.stats.record(search_ms, transfer_ms)

    # ── Dense SDPA on retrieved set ───────────────────────────────────────────
    q4 = query.to(torch.float16)                          # (1, H_q, 1, D)
    k4 = gk_gpu.unsqueeze(0).to(torch.float16)
    v4 = gv_gpu.unsqueeze(0).to(torch.float16)
    out = F.scaled_dot_product_attention(q4, k4, v4, scale=scaling)
    return out.to(query.dtype), None


def register_attention():
    AttentionInterface.register("ann_offload", ann_offload_attention_forward)


register_attention()
