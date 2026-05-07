import torch
from transformers.models.llama.modeling_llama import repeat_kv


class BaseThreshold:
    def prefill_prep(self, prefill_keys):
        """Prefilling. Prepare the thresholder if required."""
        raise NotImplementedError

    def update(self, new_keys: torch.Tensor, cache_len: int):
        """Update the thresholder with new keys if required."""
        raise NotImplementedError

    def get_threshold(self, q2d_normal: torch.Tensor) -> torch.Tensor:
        """Get threshold"""
        raise NotImplementedError


class SamplingThreshold(BaseThreshold):
    """
    Works by keeping a random sample of keys (Reservoir sampling) and computing the threshold based on that sample.
    """

    def __init__(self, sample_size: int = 100):
        self.sample_size = sample_size
        self.sample = None

    def prefill_prep(self, prefill_keys):
        _, H_kv, L, D = prefill_keys.shape
        m = min(self.sample_size, L)

        # allocate fixed-size reservoir
        self.sample = torch.zeros(
            (1, H_kv, self.sample_size, D),
            device=prefill_keys.device,
            dtype=prefill_keys.dtype,
        )

        # choose m keys from prefill (or just take first m; random is fine too)
        idx = torch.randperm(L, device=prefill_keys.device)[:m]
        self.sample[:, :, :m, :] = prefill_keys[:, :, idx, :]

        self._filled = m  # IMPORTANT

    def update(self, new_key, cache_len):
        """
        cache_len: number of keys AFTER adding this new key
        """
        assert new_key.shape[-2] == 1, "Is it decoding??"
        x = new_key.squeeze(-2)  # (1, H_kv, D)

        # fill remaining slots first (if prefill was short)
        if self._filled < self.sample_size:
            self.sample[:, :, self._filled, :] = x
            self._filled += 1
            return

        # Reservoir sampling
        j = torch.randint(0, cache_len, (1,), device=new_key.device).item()
        if j < self.sample_size:
            self.sample[:, :, j, :] = x

    def get_threshold(self, q2d_normal):
        raise NotImplementedError(
            "This is a base class for sampling-based thresholds. Implement get_threshold in subclasses."
        )


class SampleMaxThreshold(SamplingThreshold):
    def get_threshold(self, q2d_normal):
        # get threshold
        H_q, D = q2d_normal.shape
        _, H_kv, Lk, D = self.sample.shape

        qg = q2d_normal.view(H_kv, H_q // H_kv, D)  # [H_kv, g, D]
        w = qg @ self.sample.squeeze(0).transpose(-2, -1)  # [H_kv, g, Lk]
        w = w.reshape(H_q, Lk)  # [24, Lk]

        return w.max(dim=-1).values  # (24,)


class SampleMeanMaxThreshold(SamplingThreshold):
    def get_threshold(self, q2d_normal):
        # get threshold
        H_q, D = q2d_normal.shape
        _, H_kv, Lk, D = self.sample.shape

        qg = q2d_normal.view(H_kv, H_q // H_kv, D)  # [H_kv, g, D]
        w = qg @ self.sample.squeeze(0).transpose(-2, -1)  # [H_kv, g, Lk]
        w = w.reshape(H_q, Lk)  # [24, Lk]

        mean_w = w.mean(dim=-1)
        max_w = w.max(dim=-1).values

        return (mean_w + max_w) / 2  # (24,)


class FullSearchThreshold(BaseThreshold):
    def prefill_prep(self, prefill_keys):
        pass

    def update(self, new_key, cache_len):
        pass

    def get_threshold(self, q2d_normal):
        return torch.full(
            (q2d_normal.shape[0],), -float("inf"), device=q2d_normal.device
        )


class TopKThreshold(BaseThreshold):
    def __init__(self, k: int = 64):
        self.k = k

    def prefill_prep(self, prefill_keys):
        pass

    def update(self, new_key, cache_len):
        pass

    def get_threshold(self, q2d_normal, indexer):
        children = indexer.children  # (H_kv, N, D)
        qg = q2d_normal.view(children.shape[0], -1, children.shape[-1])  # [H_kv, g, D]
        w = qg @ children.transpose(-2, -1)  # [H_kv, g, N]
        w = w.reshape(q2d_normal.shape[0], -1)  # [H_q, N]
        k = self.k
        th, _ = w.topk(k, dim=-1)  # (H_q, k)
        return th[:, -1]  # (H_q,)


class FusedSearchThreshold(BaseThreshold):
    pass


ALL_THRESHOLD_ALGS = {
    "sample_max": SampleMaxThreshold,
    "sample_mean_max": SampleMeanMaxThreshold,
    "full_search": FullSearchThreshold,
    "topk": TopKThreshold,
}
