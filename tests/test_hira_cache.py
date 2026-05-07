# import os
# import sys
# from pathlib import Path

# import pytest
# import torch


# os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
# os.environ.setdefault("MAX_JOBS", "4")
# Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)

# HIRA_ROOT = Path(__file__).resolve().parents[1]
# if str(HIRA_ROOT) not in sys.path:
#     sys.path.insert(0, str(HIRA_ROOT))

# import hira.cache.hira_cache as cache_mod
# from hira.cache.hira_cache import CacheOutput, HiraCache, HiraCacheLayer
# from hira.cache.hira_config import DeviceMode, HiraConfig


# def _make_kv(*, h: int, l: int, d: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
#     g = torch.Generator().manual_seed(seed)
#     keys = torch.randn((1, h, l, d), generator=g, dtype=torch.float32)
#     values = (keys * 0.5 + 0.25).contiguous()
#     return keys, values


# class _DummyIndexer:
#     def __init__(self, **kwargs):
#         self.kwargs = kwargs
#         self.build_calls = []
#         self.update_calls = []
#         self.values = None

#     def build(self, keys: torch.Tensor, values: torch.Tensor):
#         self.build_calls.append((keys, values))
#         self.values = values.clone()
#         return self

#     def update(self, keys: torch.Tensor, values: torch.Tensor):
#         self.update_calls.append((keys, values))
#         if self.values is None:
#             self.values = values.clone()
#         else:
#             self.values = torch.cat([self.values, values], dim=-2)
#         return self


# class _DummySearcher:
#     def __init__(self, **kwargs):
#         self.kwargs = kwargs


# # @pytest.fixture
# # def patched_backends(monkeypatch):
# #     class DummyCPUIndexer(_DummyIndexer):
# #         pass

# #     class DummyCUDAIndexer(_DummyIndexer):
# #         pass

# #     class DummyCPUSearcher(_DummySearcher):
# #         pass

# #     class DummyCUDASearcher(_DummySearcher):
# #         pass

# #     monkeypatch.setattr(cache_mod, "CPUIndexer", DummyCPUIndexer)
# #     monkeypatch.setattr(cache_mod, "CUDAIndexer", DummyCUDAIndexer)
# #     monkeypatch.setattr(cache_mod, "CPUSearcher", DummyCPUSearcher)
# #     monkeypatch.setattr(cache_mod, "CUDASearcher", DummyCUDASearcher)
# #     return {
# #         "cpu_indexer": DummyCPUIndexer,
# #         "cuda_indexer": DummyCUDAIndexer,
# #         "cpu_searcher": DummyCPUSearcher,
# #         "cuda_searcher": DummyCUDASearcher,
# #     }


# def _make_layer(device_mode: str, update_every: int = 3) -> HiraCacheLayer:
#     return HiraCacheLayer(
#         device_mode=device_mode,
#         update_every=update_every,
#         indexer_kwargs={"num_levels": 3, "branching_factor": 2, "max_iterations": 1},
#         searcher_kwargs={"chunk_size": 128},
#         threshold_alg="sample_max",
#         threshold_alg_kwargs={"sample_size": 100},
#     )


# def test_hira_cache_layer_rejects_unsupported_device_mode():
#     with pytest.raises(NotImplementedError, match="not supported yet"):
#         _make_layer(device_mode=DeviceMode.CPU_CUDA)


# def test_hira_cache_layer_prefill_update_initializes_and_returns_cache_output():
#     layer = _make_layer(device_mode=DeviceMode.CPU_ONLY, update_every=4)
#     k0, v0 = _make_kv(h=2, l=3, d=8, seed=1)

#     out, _ = layer.update(k0, v0)

#     assert isinstance(out, CacheOutput)
#     assert out.prefill_keys is k0
#     assert out.prefill_values is v0
#     assert out.queued_keys.shape == (1, 2, 0, 8)
#     assert out.queued_values.shape == (1, 2, 0, 8)
#     # assert isinstance(out.indexer, patched_backends["cpu_indexer"])
#     # assert isinstance(out.searcher, patched_backends["cpu_searcher"])

#     assert layer.is_initialized
#     assert layer.indexed_len == 3
#     assert layer.q_len == 0
#     assert layer.get_seq_length() == 3
#     assert layer.get_max_cache_shape() == -1
#     # assert len(layer.indexer.build_calls) == 1
#     # assert layer.indexer.build_calls[0] == (k0, v0)
#     torch.testing.assert_close(layer.indexer.values, v0, atol=0.0, rtol=0.0)


# def test_hira_cache_layer_decode_update_returns_backward_compat_tuple():
#     layer = _make_layer(device_mode=DeviceMode.CPU_ONLY, update_every=10)
#     k0, v0 = _make_kv(h=2, l=2, d=8, seed=2)
#     layer.update(k0, v0)

#     k1, v1 = _make_kv(h=2, l=1, d=8, seed=3)
#     out, backward_compat = layer.update(k1, v1)

#     assert isinstance(out, CacheOutput)
#     assert backward_compat is None
#     assert out.prefill_keys is None
#     assert out.prefill_values is None
#     torch.testing.assert_close(out.queued_keys, k1, atol=0.0, rtol=0.0)
#     torch.testing.assert_close(out.queued_values, v1, atol=0.0, rtol=0.0)
#     assert layer.q_len == 1
#     assert layer.indexed_len == 2
#     assert layer.get_seq_length() == 3


# def test_hira_cache_layer_periodic_flush_updates_indexer_and_clears_queue():
#     layer = _make_layer(device_mode=DeviceMode.CPU_ONLY, update_every=3)
#     k0, v0 = _make_kv(h=2, l=2, d=8, seed=4)
#     layer.update(k0, v0)

#     k1, v1 = _make_kv(h=2, l=1, d=8, seed=5)
#     out1, _ = layer.update(k1, v1)
#     assert out1.queued_keys[:, : layer.q_len, :].shape == (2, 1, 8)
#     assert layer.q_len == 1

#     k2, v2 = _make_kv(h=2, l=2, d=8, seed=6)
#     out2, _ = layer.update(k2, v2)

#     expected_queued_values = torch.cat([v1, v2], dim=-2)

#     assert out2.queued_keys[:, : layer.q_len, :].shape == (2, 0, 8)
#     assert out2.queued_values[:, : layer.q_len, :].shape == (2, 0, 8)
#     assert layer.q_len == 0
#     assert layer.indexed_len == 5
#     assert layer.get_seq_length() == 5

#     expected_values = torch.cat([v0, expected_queued_values], dim=-2)
#     torch.testing.assert_close(
#         layer.indexer.values, expected_values, atol=0.0, rtol=0.0
#     )


# def test_hira_cache_layer_mask_size_helpers():
#     layer = _make_layer(device_mode=DeviceMode.CPU_ONLY, update_every=3)
#     assert layer.get_seq_length() == 0

#     cache_position = torch.tensor([0, 1, 2], dtype=torch.long)
#     kv_length, kv_offset = layer.get_mask_sizes(cache_position)
#     assert kv_length == 3
#     assert kv_offset == 0

#     k0, v0 = _make_kv(h=1, l=4, d=8, seed=7)
#     layer.update(k0, v0)
#     kv_length_after, kv_offset_after = layer.get_mask_sizes(torch.tensor([10, 11]))
#     assert kv_length_after == 6
#     assert kv_offset_after == 0


# def test_hira_cache_layer_reset_reinitializes_runtime_state():
#     layer = _make_layer(device_mode=DeviceMode.CPU_ONLY, update_every=4)
#     k0, v0 = _make_kv(h=2, l=2, d=8, seed=8)
#     layer.update(k0, v0)
#     k1, v1 = _make_kv(h=2, l=1, d=8, seed=9)
#     layer.update(k1, v1)

#     old_indexer = layer.indexer
#     layer.cumulative_length = 123
#     layer.reset()

#     assert not layer.is_initialized
#     assert layer.indexer is not old_indexer
#     assert layer.q_len == 0
#     assert layer.queued_keys[:, : layer.q_len, :].shape == (2, 0, 8)
#     assert layer.queued_values[:, : layer.q_len, :].shape == (2, 0, 8)
#     assert layer.get_seq_length() == 0
#     assert layer.cumulative_length == 0

#     k2, v2 = _make_kv(h=2, l=1, d=8, seed=10)
#     out, _ = layer.update(k2, v2)
#     assert isinstance(out, CacheOutput)
#     assert layer.is_initialized
#     # assert len(layer.indexer.build_calls) == 1
#     # assert layer.indexer.build_calls[0] == (k2, v2)


# class _DummyTextConfig:
#     def __init__(
#         self,
#         *,
#         num_hidden_layers: int,
#         layer_types=None,
#         sliding_window=None,
#         attention_chunk_size=None,
#         num_kv_shared_layers=None,
#     ):
#         self.num_hidden_layers = num_hidden_layers
#         if layer_types is not None:
#             self.layer_types = layer_types
#         if sliding_window is not None:
#             self.sliding_window = sliding_window
#         if attention_chunk_size is not None:
#             self.attention_chunk_size = attention_chunk_size
#         if num_kv_shared_layers is not None:
#             self.num_kv_shared_layers = num_kv_shared_layers


# class _DummyCacheConfig:
#     def __init__(self, text_cfg: _DummyTextConfig):
#         self._text_cfg = text_cfg

#     def get_text_config(self, decoder: bool = True):
#         assert decoder is True
#         return self._text_cfg


# @pytest.mark.parametrize(
#     "text_cfg,expected_layers",
#     [
#         (
#             _DummyTextConfig(
#                 num_hidden_layers=3,
#                 layer_types=["full_attention", "full_attention", "full_attention"],
#             ),
#             3,
#         ),
#         (_DummyTextConfig(num_hidden_layers=4, sliding_window=4096), 4),
#         (_DummyTextConfig(num_hidden_layers=5, attention_chunk_size=1024), 5),
#         (_DummyTextConfig(num_hidden_layers=2), 2),
#     ],
# )
# def test_hira_cache_builds_expected_number_of_layers(text_cfg, expected_layers):
#     hira_cfg = HiraConfig(device_mode=DeviceMode.CPU_ONLY, update_every=6)
#     cache = HiraCache(
#         cache_config=_DummyCacheConfig(text_cfg=text_cfg),
#         hira_config=hira_cfg,
#     )

#     assert len(cache.layers) == expected_layers
#     for layer in cache.layers:
#         assert layer.update_every == hira_cfg.update_every
#         assert layer.indexer_kwargs == hira_cfg.get_indexer_kwargs()
#         assert layer.searcher_kwargs == hira_cfg.get_searcher_kwargs()


# def test_hira_cache_excludes_shared_kv_layers():
#     text_cfg = _DummyTextConfig(num_hidden_layers=6, num_kv_shared_layers=2)
#     hira_cfg = HiraConfig(device_mode=DeviceMode.CPU_ONLY, update_every=4)
#     cache = HiraCache(
#         cache_config=_DummyCacheConfig(text_cfg=text_cfg),
#         hira_config=hira_cfg,
#     )
#     assert len(cache.layers) == 4


# def test_hira_cache_update_delegates_to_selected_layer_after_prefill():
#     text_cfg = _DummyTextConfig(num_hidden_layers=3)
#     hira_cfg = HiraConfig(device_mode=DeviceMode.CPU_ONLY, update_every=8)
#     cache = HiraCache(
#         cache_config=_DummyCacheConfig(text_cfg=text_cfg),
#         hira_config=hira_cfg,
#     )

#     k_prefill, v_prefill = _make_kv(h=2, l=2, d=8, seed=12)
#     prefill_out, _ = cache.layers[1].update(k_prefill, v_prefill)
#     assert isinstance(prefill_out, CacheOutput)

#     assert not getattr(cache.layers[0], "is_initialized", False)
#     assert getattr(cache.layers[1], "is_initialized", False)
#     assert not getattr(cache.layers[2], "is_initialized", False)

#     k_decode, v_decode = _make_kv(h=2, l=1, d=8, seed=13)
#     out, backward_compat = cache.update(k_decode, v_decode, layer_idx=1)
#     assert isinstance(out, CacheOutput)
#     assert backward_compat is None
#     assert out.indexer is cache.layers[1].indexer
#     assert out.searcher is cache.layers[1].searcher
#     assert out.prefill_keys is None
#     assert out.prefill_values is None
