"""
ClusterKV KV cache via AttentionInterface — pure-PyTorch reimplementation.

During prefill: cluster key vectors into `nlist` centroids per head using
Lloyd's K-means (no pylibraft/RAFT). During decode: score centroids via
query·centroid dot-product, select enough top clusters to fill token_budget,
gather selected token KV pairs + sink tokens.

Registers: "clusterkv" attention forward.

Ref: Liu et al., "ClusterKV: Manipulating LLM KV Cache in Semantic Space for
     Recallable Compression and Streaming"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.modeling_utils import AttentionInterface
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS


# ── K-means (Lloyd's, pure PyTorch) ──────────────────────────────────────────

def _kmeans(x: torch.Tensor, k: int, n_iter: int = 10) -> torch.Tensor:
    """
    x: (N, D) float32
    Returns centroids: (k, D)
    """
    N, D = x.shape
    k = min(k, N)
    # Init: random subset
    idx = torch.randperm(N, device=x.device)[:k]
    centroids = x[idx].clone()

    for _ in range(n_iter):
        # Assignment: (N,) cluster index
        dists = torch.cdist(x, centroids)       # (N, k)
        assign = dists.argmin(dim=1)            # (N,)
        # Update
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=x.device, dtype=torch.float32)
        new_centroids.scatter_add_(0, assign.unsqueeze(1).expand(-1, D), x)
        counts.scatter_add_(0, assign, torch.ones(N, device=x.device))
        mask = counts > 0
        new_centroids[mask] /= counts[mask].unsqueeze(1)
        # Re-seed empty clusters from random points
        empty = (~mask).nonzero(as_tuple=True)[0]
        if len(empty):
            reseeds = torch.randint(N, (len(empty),), device=x.device)
            new_centroids[empty] = x[reseeds]
        centroids = new_centroids

    return centroids


# ── Cache layer ───────────────────────────────────────────────────────────────

class ClusterKVCacheLayer(CacheLayerMixin):
    def __init__(self, nlist: int, token_budget: int, sink_size: int, n_iter: int):
        super().__init__()
        self.nlist = nlist
        self.token_budget = token_budget
        self.sink_size = sink_size
        self.n_iter = n_iter
        # Built after prefill
        self.centroids: Optional[torch.Tensor] = None   # (H_kv, nlist, D)
        self.cluster_assign: Optional[torch.Tensor] = None  # (H_kv, N_prefill)
        self.prompt_len: int = 0

    def lazy_initialization(self, key_states, value_states) -> None:
        pass

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """key_states/value_states: (H_kv, T, D)."""
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
            self.is_initialized = True
        else:
            self.keys = torch.cat([self.keys, key_states], dim=1)
            self.values = torch.cat([self.values, value_states], dim=1)
        return self.keys, self.values

    def build_clusters(self) -> None:
        """Run K-means on prefill keys (called once after prefill)."""
        H_kv, N, D = self.keys.shape
        sink = self.sink_size
        keys_f32 = self.keys[:, sink:, :].float()  # cluster non-sink keys
        N_clust = N - sink
        k = min(self.nlist, N_clust)

        all_centroids = []
        all_assign = []
        for h in range(H_kv):
            centroids_h = _kmeans(keys_f32[h], k, self.n_iter)  # (k, D)
            dists = torch.cdist(keys_f32[h], centroids_h)
            assign_h = dists.argmin(dim=1)  # (N_clust,)
            all_centroids.append(centroids_h)
            all_assign.append(assign_h)

        self.centroids = torch.stack(all_centroids)     # (H_kv, k, D)
        self.cluster_assign = torch.stack(all_assign)   # (H_kv, N_clust)
        self.prompt_len = N

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self.get_seq_length() + cache_position.shape[0], 0

    def get_seq_length(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.shape[1]

    def get_max_cache_shape(self) -> int:
        return -1


# ── Cache (multi-layer) ───────────────────────────────────────────────────────

@dataclass
class ClusterKVCacheOutput:
    keys: torch.Tensor
    values: torch.Tensor
    layer_cache: ClusterKVCacheLayer
    is_prefill: bool


class ClusterKVCache(Cache):
    def __init__(self, nlist: int, token_budget: int, sink_size: int,
                 num_layers: int, n_iter: int = 10):
        layers = [
            ClusterKVCacheLayer(nlist, token_budget, sink_size, n_iter)
            for _ in range(num_layers)
        ]
        super().__init__(layers=layers)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)
        layer: ClusterKVCacheLayer = self.layers[layer_idx]
        is_prefill = not layer.is_initialized
        layer.update(key_states, value_states)
        return ClusterKVCacheOutput(
            keys=layer.keys,
            values=layer.values,
            layer_cache=layer,
            is_prefill=is_prefill,
        ), None

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self):
        return None


# ── Attention forward ─────────────────────────────────────────────────────────

def clusterkv_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: ClusterKVCacheOutput,
    value: None,
    attention_mask,
    dropout: float,
    scaling: float,
    **kwargs,
):
    cache_out = key
    layer = cache_out.layer_cache
    keys = cache_out.keys.unsqueeze(0)
    values = cache_out.values.unsqueeze(0)

    B, H_q, T, D = query.shape
    H_kv = keys.shape[1]
    N = keys.shape[2]
    group_size = H_q // H_kv

    # Prefill: SDPA
    if T > 1:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(
            module, query, keys, values, attention_mask, dropout, scaling=scaling, **kwargs
        )

    # ── First decode step: build clusters ───────────────────────────────
    if layer.centroids is None:
        layer.build_clusters()

    sink = layer.sink_size
    prompt_len = layer.prompt_len
    token_budget = min(layer.token_budget, prompt_len - sink)
    nlist_actual = layer.centroids.shape[1]

    # query per kv-head: (H_kv, D)
    q_2d = query[0, :, 0, :].float()
    if group_size > 1:
        q_kv = q_2d.view(H_kv, group_size, D).mean(dim=1)
    else:
        q_kv = q_2d

    # Score centroids: (H_kv, nlist)
    centroids_f = layer.centroids.float()
    c_scores = torch.bmm(q_kv.unsqueeze(1), centroids_f.transpose(1, 2)).squeeze(1)

    # Sort centroids descending; accumulate cluster sizes until budget filled
    _, c_order = c_scores.sort(dim=1, descending=True)  # (H_kv, nlist)

    # Expand to H_q for gather
    if group_size > 1:
        q_kv_idx = torch.arange(H_q, device=query.device) // group_size  # (H_q,)
    else:
        q_kv_idx = torch.arange(H_q, device=query.device)

    # Select tokens per kv-head: gather indices of selected cluster members
    assign = layer.cluster_assign  # (H_kv, N_prefill) — indices for non-sink tokens
    N_prefill_clust = assign.shape[1]

    keep_masks = []  # one per kv-head: (N_prefill_clust,) bool
    for h in range(H_kv):
        order_h = c_order[h]  # (nlist,)
        mask_h = torch.zeros(N_prefill_clust, device=query.device, dtype=torch.bool)
        count = 0
        for ci in order_h:
            if count >= token_budget:
                break
            cluster_mask = assign[h] == ci
            mask_h |= cluster_mask
            count += cluster_mask.sum().item()
        keep_masks.append(mask_h)
    keep_masks = torch.stack(keep_masks)  # (H_kv, N_prefill_clust)

    # Build selected KV per head via gather; pad to same length
    # Use the mask to get indices, then gather from keys/values
    # For efficiency: use the mask directly for each kv-head
    selected_keys_list = []
    selected_vals_list = []
    max_sel = keep_masks.sum(dim=1).max().item()

    for h in range(H_kv):
        idx = keep_masks[h].nonzero(as_tuple=True)[0]  # selected positions in non-sink
        idx_full = idx + sink  # offset for sink tokens
        k_h = layer.keys[h, :sink, :]  # sink
        v_h = layer.values[h, :sink, :]
        k_sel = layer.keys[h, idx_full, :]
        v_sel = layer.values[h, idx_full, :]
        # decode tokens (after prompt_len)
        k_dec = layer.keys[h, prompt_len:, :]
        v_dec = layer.values[h, prompt_len:, :]
        k_full = torch.cat([k_h, k_sel, k_dec], dim=0)   # (sink + n_sel + n_dec, D)
        v_full = torch.cat([v_h, v_sel, v_dec], dim=0)
        selected_keys_list.append(k_full)
        selected_vals_list.append(v_full)

    # Pad to same N across heads if needed
    max_n = max(k.shape[0] for k in selected_keys_list)
    if any(k.shape[0] != max_n for k in selected_keys_list):
        selected_keys_list = [
            F.pad(k, (0, 0, 0, max_n - k.shape[0])) for k in selected_keys_list
        ]
        selected_vals_list = [
            F.pad(v, (0, 0, 0, max_n - v.shape[0])) for v in selected_vals_list
        ]

    sel_keys = torch.stack(selected_keys_list).unsqueeze(0)   # (1, H_kv, N_sel, D)
    sel_vals = torch.stack(selected_vals_list).unsqueeze(0)

    # GQA expand
    if group_size > 1:
        sel_keys = sel_keys.repeat_interleave(group_size, dim=1)
        sel_vals = sel_vals.repeat_interleave(group_size, dim=1)

    scores = torch.matmul(query, sel_keys.transpose(-2, -1)) * scaling
    attn_weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
    if dropout > 0.0 and module.training:
        attn_weights = F.dropout(attn_weights, p=dropout)
    return torch.matmul(attn_weights, sel_vals), None


# ── Registration ──────────────────────────────────────────────────────────────

AttentionInterface.register("clusterkv", clusterkv_attention_forward)
ALL_MASK_ATTENTION_FUNCTIONS.register("clusterkv", ALL_MASK_ATTENTION_FUNCTIONS["eager"])
