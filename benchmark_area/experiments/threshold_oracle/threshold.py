"""
Threshold computation for Louver index — extended with sample_topk.

Modes:
  - 'budget': fixed fraction f of tokens retrieved.
  - 'oracle': threshold from reservoir sample.
    - 'sample_max'      : threshold = max sample score (tightest, fewest tokens).
    - 'sample_topk'     : threshold = k-th highest sample score (topk_k controls aggressiveness).
    - 'sample_mean_max' : threshold = (max + mean) / 2.
    - 'sample_gap'      : cut at largest score gap in top fraction of sample.
"""
from __future__ import annotations

import torch


class LouverThreshold:
    def __init__(
        self,
        mode: str = "oracle",
        oracle: str = "sample_max",
        budget_fraction: float = 0.1,
        sample_size: int = 256,
        topk_k: int = 5,
        gap_search_frac: float = 0.5,
        gap_topk: int = 3,
    ):
        assert mode in ("budget", "oracle")
        assert oracle in ("sample_max", "sample_topk", "sample_mean_max", "sample_gap")
        self.mode = mode
        self.oracle = oracle
        self.budget_fraction = budget_fraction
        self.sample_size = sample_size
        self.topk_k = topk_k
        self.gap_search_frac = gap_search_frac
        self.gap_topk = gap_topk

        self.sample: torch.Tensor | None = None  # (H_kv, M, D) fp16
        self._filled = 0
        self._N = 0

    # ── Population ────────────────────────────────────────────────────────────

    def prefill_prep(self, keys_f16: torch.Tensor) -> None:
        """keys_f16: (H_kv, N, D) fp16."""
        H_kv, N, D = keys_f16.shape
        M = min(self.sample_size, N)
        idx = torch.randperm(N, device=keys_f16.device)[:M]
        self.sample = torch.empty(H_kv, self.sample_size, D,
                                  device=keys_f16.device, dtype=torch.float16)
        self.sample[:, :M, :] = keys_f16[:, idx, :]
        self._filled = M
        self._N = N

    def update(self, new_key_f16: torch.Tensor, total_N: int) -> None:
        """new_key_f16: (H_kv, 1, D) fp16."""
        self._N = total_N
        if self._filled < self.sample_size:
            self.sample[:, self._filled, :] = new_key_f16[:, 0, :]
            self._filled += 1
        else:
            j = torch.randint(0, total_N, (1,), device=self.sample.device).item()
            if j < self.sample_size:
                self.sample[:, j, :] = new_key_f16[:, 0, :]

    # ── Threshold computation ─────────────────────────────────────────────────

    def _sample_scores(self, q_f16: torch.Tensor) -> torch.Tensor:
        """q_f16: (H_q, D) → (H_q, M) float32 dot products against sample."""
        H_q, D = q_f16.shape
        H_kv = self.sample.shape[0]
        M = self._filled
        g = H_q // H_kv
        q_3d = q_f16.view(H_kv, g, D).float()
        s = self.sample[:, :M, :].float()
        return torch.einsum("hgd,hmd->hgm", q_3d, s).reshape(H_q, M)

    def _gap_threshold(self, scores: torch.Tensor) -> torch.Tensor:
        """scores: (H_q, M) → (H_q,) threshold via score-gap oracle."""
        H_q, M = scores.shape
        search_k = max(2, min(int(self.gap_search_frac * M), M))
        sorted_s, _ = scores.sort(dim=-1, descending=True)
        top_s = sorted_s[:, :search_k]
        gaps = top_s[:, :-1] - top_s[:, 1:]
        k = min(self.gap_topk, gaps.shape[1])
        _, topk_pos = gaps.topk(k, dim=-1)
        best_pos = topk_pos.min(dim=-1).values
        return top_s.gather(1, (best_pos + 1).unsqueeze(1)).squeeze(1)

    def get_threshold_ta(self, q_f16: torch.Tensor) -> torch.Tensor:
        """q_f16: (H_q, D) → (H_q,) float32 threshold."""
        if self.sample is None or self._filled == 0:
            return torch.full((q_f16.shape[0],), float("-inf"), device=q_f16.device)

        scores = self._sample_scores(q_f16)  # (H_q, M)

        if self.mode == "budget":
            k = max(1, int(self.budget_fraction * self._filled))
            return scores.topk(k, dim=-1).values[:, -1].float()

        if self.oracle == "sample_max":
            return scores.max(dim=-1).values.float()

        if self.oracle == "sample_topk":
            k = min(self.topk_k, self._filled)
            return scores.topk(k, dim=-1).values[:, -1].float()

        if self.oracle == "sample_mean_max":
            return ((scores.max(dim=-1).values + scores.mean(dim=-1)) / 2).float()

        # sample_gap
        return self._gap_threshold(scores).float()
