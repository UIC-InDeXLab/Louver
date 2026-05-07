"""
Louver offloading cache:
  - Parent cluster centers → GPU (used for halfspace filter via CUDA kernels)
  - All KV pairs → CPU pinned memory
  - Per decode step: GPU filter → live token indices → gather from CPU → transfer → SDPA
  - Records: search_ms (GPU filter), transfer_ms (CPU→GPU), gpu_bytes (parents only)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.modeling_utils import AttentionInterface
from transformers import PreTrainedConfig

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
_REPO  = _HERE.parents[3]
for _p in (str(_BENCH), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import types as _t
_hira = _t.ModuleType("hira"); _hira.__path__ = [str(_REPO)]; _hira.__package__ = "hira"
sys.modules["hira"] = _hira

from kernel_impl.TA_filter_alg.index import (         # noqa: E402
    TAIndex, TAIndexConfig, _filter_with_workspace, _build_packed_assigns,
)
from kernel_impl.TA_filter_alg.kernels.update.update_v1_1 import (  # noqa: E402
    BF, BUFFER_SIZE, S as S_FIXED,
)
from base import OffloadStats, cpu_timer, gpu_sync_timer, tensor_bytes  # noqa: E402

# reuse threshold from louver_hf (same experiment directory)
_ACC = _HERE.parents[0].parent / "experiments" / "accuracy"
sys.path.insert(0, str(_ACC))
from louver_hf.threshold import LouverThreshold  # noqa: E402


BUDGET_FRACTION = 0.15


@dataclass
class LouverOffloadOutput:
    layer_cache: "LouverOffloadCacheLayer"
    is_prefill: bool
    prefill_keys: torch.Tensor | None = None
    prefill_values: torch.Tensor | None = None


class LouverOffloadCacheLayer(CacheLayerMixin):
    """
    TA index parents on GPU; full KV pairs on CPU pinned memory.
    """

    def __init__(self, update_interval: int = BUFFER_SIZE, budget_fraction: float = BUDGET_FRACTION):
        super().__init__()
        self.update_interval = update_interval
        self.budget_fraction  = budget_fraction

        self.index: TAIndex | None = None
        self.threshold: LouverThreshold | None = None
        self.stats = OffloadStats()

        # CPU-pinned KV store (populated at prefill)
        self._cpu_keys: torch.Tensor | None = None    # (H_kv, N, D) fp16 pinned
        self._cpu_values: torch.Tensor | None = None  # (H_kv, N, D) fp16 pinned
        self._n_stored: int = 0

        self._seq_len: int = 0
        self._steps: int = 0
        self._scale: float = 1.0

    def lazy_initialization(self, key_states, value_states) -> None:
        pass  # handled inside update() on first call

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               cache_kwargs: dict | None = None):
        B, H_kv, L, D = key_states.shape
        assert B == 1

        # ── Prefill ──────────────────────────────────────────────────────────────
        if not self.is_initialized:
            device = key_states.device
            self._scale = float((cache_kwargs or {}).get("scaling", D ** -0.5))

            ks = key_states.squeeze(0).to(torch.float16).contiguous()   # (H_kv, L, D)
            vs = value_states.squeeze(0).to(torch.float16).contiguous()

            # Build TA index on GPU (keeps centers + assigns on GPU)
            with tqdm(total=1, desc="  build TA index", leave=False, dynamic_ncols=True) as _pbar:
                self.index = TAIndex(TAIndexConfig(parallel_update=False))
                self.index.build(ks, vs)
                _pbar.update(1)

            # Offload KV to CPU pinned memory; keep parents (state minus KV) on GPU
            N_pad = int(self.index.state["N_pad"])
            self._cpu_keys   = torch.zeros(H_kv, N_pad, D, dtype=torch.float16,
                                           pin_memory=True)
            self._cpu_values = torch.zeros(H_kv, N_pad, D, dtype=torch.float16,
                                           pin_memory=True)
            # Copy from GPU index state to CPU
            self._cpu_keys[:, :L, :]   = self.index.state["keys_padded_f16"][:, :L, :].cpu()
            self._cpu_values[:, :L, :] = self.index.state["values_padded_f16"][:, :L, :].cpu()
            # Remove KV from GPU state — free GPU memory, leave parents
            del self.index.state["keys_padded_f16"]
            del self.index.state["values_padded_f16"]

            self._n_stored = L
            self.threshold = LouverThreshold(
                mode="budget", budget_fraction=self.budget_fraction, sample_size=512,
            )
            self.threshold.prefill_prep(ks)
            self._seq_len = L
            self.is_initialized = True

            # Record GPU memory: centers + assigns + small metadata only
            gpu_bytes = 0
            for k, v in self.index.state.items():
                if isinstance(v, torch.Tensor):
                    gpu_bytes += tensor_bytes(v)
            if self.index._buf_keys_arena is not None:
                gpu_bytes += tensor_bytes(self.index._buf_keys_arena,
                                         self.index._buf_values_arena)
            self.stats.gpu_bytes = gpu_bytes

            return LouverOffloadOutput(
                layer_cache=self, is_prefill=True,
                prefill_keys=key_states, prefill_values=value_states,
            ), None

        # ── Decode step ──────────────────────────────────────────────────────────
        assert L == 1
        new_k = key_states.squeeze(0).to(torch.float16)   # (H_kv, 1, D)
        new_v = value_states.squeeze(0).to(torch.float16)

        # Grow CPU KV store if needed
        n_pad = self._cpu_keys.shape[1]
        if self._n_stored >= n_pad:
            grow = max(BUFFER_SIZE * 64, n_pad // 2)
            D_ = self._cpu_keys.shape[2]
            ext_k = torch.zeros(new_k.shape[0], grow, D_, dtype=torch.float16, pin_memory=True)
            ext_v = torch.zeros_like(ext_k)
            self._cpu_keys   = torch.cat([self._cpu_keys, ext_k], dim=1)
            self._cpu_values = torch.cat([self._cpu_values, ext_v], dim=1)

        self._cpu_keys[:, self._n_stored:self._n_stored+1, :]   = new_k.cpu()
        self._cpu_values[:, self._n_stored:self._n_stored+1, :] = new_v.cpu()
        self._n_stored += 1

        self.index.append_decoding_kv(new_k, new_v)
        self._seq_len += 1
        self._steps += 1
        self.threshold.update(new_k, self._seq_len)

        if self.index.needs_update(self.update_interval):
            self.index.update()

        return LouverOffloadOutput(layer_cache=self, is_prefill=False), None

    def get_seq_length(self) -> int:
        return self._seq_len

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, cache_position):
        q_len = cache_position.shape[0] if hasattr(cache_position, "shape") else 1
        return self._seq_len + q_len, 0

    def reset(self):
        self.index = None
        self.threshold = None
        self._cpu_keys = None
        self._cpu_values = None
        self._n_stored = 0
        self._seq_len = 0
        self._steps = 0
        self.is_initialized = False
        self.stats = OffloadStats()


class LouverOffloadCache(Cache):
    def __init__(self, model_config: PreTrainedConfig,
                 budget_fraction: float = BUDGET_FRACTION,
                 update_interval: int = BUFFER_SIZE):
        config = model_config.get_text_config(decoder=True)
        n = config.num_hidden_layers
        layers = [LouverOffloadCacheLayer(update_interval, budget_fraction)
                  for _ in range(n)]
        super().__init__(layers=layers, offloading=False, offload_only_non_sliding=None)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        out, _ = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)
        return out, None

    def aggregate_stats(self) -> dict:
        """Average stats across all layers."""
        all_search  = []
        all_transfer = []
        gpu_bytes    = 0
        for layer in self.layers:
            all_search.extend(layer.stats.search_ms)
            all_transfer.extend(layer.stats.transfer_ms)
            gpu_bytes += layer.stats.gpu_bytes
        n = max(len(all_search), 1)
        return {
            "search_ms":   round(sum(all_search) / n, 4),
            "transfer_ms": round(sum(all_transfer) / n, 4),
            "gpu_mb":      round(gpu_bytes / 1e6, 3),
        }


# ── Attention forward ─────────────────────────────────────────────────────────

def louver_offload_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,          # (B, H_q, 1, D)
    key: LouverOffloadOutput,
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

    layer: LouverOffloadCacheLayer = key.layer_cache
    index  = layer.index
    thresh = layer.threshold
    device = query.device

    H_q = query.shape[1]
    H_kv = layer._cpu_keys.shape[0]
    D    = query.shape[-1]

    q2d = query.squeeze(0).squeeze(-2).to(torch.float16).contiguous()  # (H_q, D)

    if not hasattr(module, "_offload_q2kv") or module._offload_q2kv.shape[0] != H_q:
        g = H_q // H_kv
        module._offload_q2kv = torch.arange(H_q, device=device, dtype=torch.int64) // g

    q_head_to_kv = module._offload_q2kv

    # ── 1. GPU filter (timed as search_ms) ───────────────────────────────────
    th = thresh.get_threshold_ta(q2d)                           # (H_q,) fp32
    ws = index._ensure_workspace(H_q)
    ws["live_count"].zero_()

    t0 = gpu_sync_timer(device)
    _filter_with_workspace(
        q=q2d, threshold=th, state=index.state,
        q_head_to_kv=q_head_to_kv,
        live_idx=ws["live_idx_filter"], live_count=ws["live_count"],
        top_scores=ws["top_scores"], top_indices=ws["top_indices"],
        depth=ws["depth"],
    )
    torch.cuda.synchronize(device)
    search_ms = gpu_sync_timer(device) - t0

    # ── 2. Gather from CPU + transfer (timed as transfer_ms) ─────────────────
    live_idx_cpu  = ws["live_idx_filter"].cpu()   # (H_q, N_pad)
    live_count_cpu = ws["live_count"].cpu()       # (H_q,)

    gathered_keys_list   = []
    gathered_values_list = []
    for h_q in range(H_q):
        h_kv = int(q_head_to_kv[h_q].item())
        cnt  = int(live_count_cpu[h_q].item())
        if cnt == 0:
            # fallback: take budget tokens
            cnt = max(1, int(layer.budget_fraction * layer._n_stored))
            idxs = torch.arange(cnt, dtype=torch.long)
        else:
            idxs = live_idx_cpu[h_q, :cnt].long()
        # Gather from CPU pinned store
        gathered_keys_list.append(layer._cpu_keys[h_kv][idxs])      # (cnt, D)
        gathered_values_list.append(layer._cpu_values[h_kv][idxs])  # (cnt, D)

    # Stack: we need uniform length for batched SDPA, pad to max
    max_len = max(t.shape[0] for t in gathered_keys_list)

    def _pad(lst, max_l):
        out = []
        for t in lst:
            if t.shape[0] < max_l:
                pad = torch.zeros(max_l - t.shape[0], t.shape[1], dtype=t.dtype)
                t = torch.cat([t, pad], dim=0)
            out.append(t)
        return torch.stack(out, dim=0)  # (H_q, max_l, D)

    gk = _pad(gathered_keys_list,   max_len)  # CPU
    gv = _pad(gathered_values_list, max_len)  # CPU

    # Also include buffer tokens (already on GPU — small, fixed BUFFER_SIZE)
    buf_k = index._buf_keys_arena[:, :index._l_buf, :].contiguous()    # (H_kv, l_buf, D) GPU
    buf_v = index._buf_values_arena[:, :index._l_buf, :].contiguous()

    # Transfer gathered KV from CPU to GPU
    t_transfer_start = gpu_sync_timer(device)
    gk_gpu = gk.to(device, non_blocking=False)
    gv_gpu = gv.to(device, non_blocking=False)
    torch.cuda.synchronize(device)
    transfer_ms = gpu_sync_timer(device) - t_transfer_start

    layer.stats.record(search_ms, transfer_ms)

    # ── 3. Dense SDPA on retrieved set (GQA: expand KV to H_q) ──────────────
    # gk_gpu: (H_q, max_len, D); q2d: (H_q, D)
    # Append buffer per-head using q_head_to_kv mapping
    buf_k_expanded = buf_k[q_head_to_kv]   # (H_q, l_buf, D)
    buf_v_expanded = buf_v[q_head_to_kv]

    keys_full   = torch.cat([gk_gpu.to(torch.float16),
                              buf_k_expanded.to(torch.float16)], dim=1)   # (H_q, T, D)
    values_full = torch.cat([gv_gpu.to(torch.float16),
                              buf_v_expanded.to(torch.float16)], dim=1)

    q4 = q2d.unsqueeze(1).unsqueeze(0)                # (1, H_q, 1, D)
    k4 = keys_full.unsqueeze(0)                        # (1, H_q, T, D)
    v4 = values_full.unsqueeze(0)

    out = F.scaled_dot_product_attention(q4, k4, v4, scale=scaling)  # (1, H_q, 1, D)
    return out.to(query.dtype), None


def register_attention():
    AttentionInterface.register("louver_offload", louver_offload_attention_forward)


register_attention()
