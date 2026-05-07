from __future__ import annotations

import torch
from dataclasses import dataclass
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.configuration_utils import PreTrainedConfig
from typing import Any, TYPE_CHECKING

from hira.cache.hira_config import HiraConfig
from hira.cache.hira_config import DeviceMode
from hira.indexer import CPUIndexer, CUDAIndexer

if TYPE_CHECKING:
    from hira.searcher.cpu import CPUSearcher
    from hira.searcher.cuda import CUDASearcher

"""
Tasks:
    - [x] Update every, keep a local list of keys.
    - [ ] Implement threshold finding.
"""


@dataclass(slots=True)
class CacheOutput:
    queued_keys: torch.Tensor
    queued_values: torch.Tensor
    queued_len: int
    prefill_keys: torch.Tensor | None
    prefill_values: torch.Tensor | None
    indexer: CPUIndexer | CUDAIndexer
    searcher: CPUSearcher | CUDASearcher


class HiraCacheLayer(CacheLayerMixin):
    """
    Single layer, but different heads (KV heads).

    Definitions:
        - D: head dim (128)
        - B: Batch size (1)
        - L: sequence length
        - H: # of heads
    e.g.
        key_states, value_states: (B, H_kv, L, D)
        query_states: (B, H_q, L, D)
        : H_q = 28
        : H_kv = 8
    """

    def __init__(
        self,
        device_mode: DeviceMode,
        update_every: int,
        indexer_kwargs: dict[str, Any],
        searcher_kwargs: dict[str, Any],
    ):
        super().__init__()
        self.device_mode = device_mode
        self.update_every = update_every
        self.indexer_kwargs = indexer_kwargs
        self.searcher_kwargs = searcher_kwargs

        self.indexer_cls = None
        self.searcher_cls = None
        if self.device_mode == DeviceMode.CPU_ONLY:
            from hira.searcher.cpu import CPUSearcher

            self.indexer_cls = CPUIndexer
            self.searcher_cls = CPUSearcher
        elif self.device_mode == DeviceMode.CUDA_ONLY:
            from hira.searcher.cuda import CUDASearcher

            self.indexer_cls = CUDAIndexer
            self.searcher_cls = CUDASearcher
        else:
            raise NotImplementedError(
                f"Device mode {self.device_mode} not supported yet"
            )

    def lazy_initialization(self, key_states, value_states):
        # key_states and value_states might be fake and empty
        # this is called once after prefilling
        self.dim = key_states.shape[-1]
        self.H_kv = key_states.shape[-3]
        self.dtype = key_states.dtype
        self.device = key_states.device

        # create indexer and searcher
        self.indexer = self.indexer_cls(**self.indexer_kwargs)
        self.searcher = self.searcher_cls(**self.searcher_kwargs)

        # queued key/values. Pre-allocated ring buffers
        self.queued_keys = torch.empty(
            (self.H_kv, self.update_every, self.dim),
            dtype=self.dtype,
            device=self.device,
        ).contiguous()
        self.queued_values = torch.empty(
            (self.H_kv, self.update_every, self.dim),
            dtype=self.dtype,
            device=self.device,
        ).contiguous()
        self.q_len = 0

        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self.indexer.build(key_states, value_states)
            self.indexed_len = key_states.shape[-2]
            # prefill step
            return (
                CacheOutput(
                    queued_keys=self.queued_keys,
                    queued_values=self.queued_values,
                    queued_len=self.q_len,
                    prefill_keys=key_states,
                    prefill_values=value_states,
                    indexer=self.indexer,
                    searcher=self.searcher,
                ),
                None,
            )

        # concat to ring buffer
        n = key_states.shape[-2]
        self.queued_keys[:, self.q_len : self.q_len + n, :] = key_states
        self.queued_values[:, self.q_len : self.q_len + n, :] = value_states
        self.q_len += n

        # periodic update
        if self.q_len >= self.update_every:
            self.update_index()

        return (
            CacheOutput(
                queued_keys=self.queued_keys,
                queued_values=self.queued_values,
                queued_len=self.q_len,
                prefill_keys=None,
                prefill_values=None,
                indexer=self.indexer,
                searcher=self.searcher,
            ),
            None,
        )  # backward compatibility

    def update_index(self):
        # append keys to index
        self.indexer.update(self.queued_keys, self.queued_values)
        self.indexed_len += self.q_len
        self.q_len = 0

    def get_mask_sizes(self, cache_position):
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self):
        """Returns the sequence length of the cached states."""
        if not self.is_initialized:
            return 0
        return self.indexed_len + self.q_len

    def get_max_cache_shape(self):
        return -1

    def reset(self):
        """Resets the cache values while preserving the objects"""
        if self.is_initialized:
            self.indexer = self.indexer_cls(**self.indexer_kwargs)
            self.queued_keys = torch.empty(
                (self.H_kv, self.update_every, self.dim),
                dtype=self.dtype,
                device=self.device,
            ).contiguous()
            self.queued_values = torch.empty(
                (self.H_kv, self.update_every, self.dim),
                dtype=self.dtype,
                device=self.device,
            ).contiguous()
            self.q_len = 0
            self.is_initialized = False
        # This attribute is set on several Layers
        if hasattr(self, "cumulative_length"):
            self.cumulative_length = 0


class HiraCache(Cache):
    """
    Implements HuggingFace's Cache interface for HIRA index.
    """

    def __init__(
        self,
        cache_config: PreTrainedConfig,
        hira_config: HiraConfig,
    ):
        self.device_mode = hira_config.device_mode
        self.update_every = hira_config.update_every

        # extract num layers [COPIED CODE from 'transformers']
        config = cache_config.get_text_config(decoder=True)
        layer_types = getattr(config, "layer_types", None)
        # If `layer_types` is not explicitly provided, infer if the model is fully sliding
        if layer_types is None:
            if getattr(config, "sliding_window", None) is not None:
                layer_types = [
                    "sliding_attention" for _ in range(config.num_hidden_layers)
                ]
            elif getattr(config, "attention_chunk_size", None) is not None:
                layer_types = [
                    "chunked_attention" for _ in range(config.num_hidden_layers)
                ]
            else:
                layer_types = [
                    "full_attention" for _ in range(config.num_hidden_layers)
                ]
        # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
        if hasattr(config, "num_kv_shared_layers"):
            layer_types = layer_types[: -config.num_kv_shared_layers]

        # build layers
        layers = []
        for _ in layer_types:
            # treating all layer types the same
            layer = HiraCacheLayer(
                device_mode=self.device_mode,
                update_every=self.update_every,
                indexer_kwargs=hira_config.get_indexer_kwargs(),
                searcher_kwargs=hira_config.get_searcher_kwargs(),
            )
            layers.append(layer)

        super().__init__(
            layers=layers,
            offloading=False,  # handle manually
            offload_only_non_sliding=None,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ):
        cache_output, _ = self.layers[layer_idx].update(
            key_states, value_states, cache_kwargs
        )
        return cache_output, None
