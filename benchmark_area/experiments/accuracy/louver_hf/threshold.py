"""
Threshold computation for kernel_impl indices.

Two modes:
  - 'budget': fixed fraction f of tokens retrieved; threshold estimated from sample.
  - 'oracle': threshold from reservoir sample (SampleMax or SampleMeanMax).

Works with raw (un-normalized) fp16 queries and keys, matching kernel_impl expectations.
"""
from __future__ import annotations

import torch


class LouverThreshold:
    """
    Per-layer threshold state. Maintains a reservoir sample of seen keys
    and computes per-head (TA filter) or per-subspace (full-subspace) thresholds.
    """

    def __init__(
        self,
        mode: str = "oracle",          # "budget" | "oracle"
        oracle: str = "sample_max",    # "sample_max" | "sample_mean_max" | "sample_top_p"
        budget_fraction: float = 0.1,  # used when mode="budget"
        sample_size: int = 256,
        top_p: float = 0.85,           # used when oracle="sample_top_p"
    ):
        assert mode in ("budget", "oracle")
        assert oracle in ("sample_max", "sample_mean_max", "sample_top_p")
        self.mode = mode
        self.oracle = oracle
        self.budget_fraction = budget_fraction
        self.sample_size = sample_size
        self.top_p = top_p

        self.sample: torch.Tensor | None = None  # (H_kv, M, D) fp16
        self._filled = 0
        self._N = 0

    # ── Population ───────────────────────────────────────────────────

    def prefill_prep(self, keys_f16: torch.Tensor) -> None:
        """keys_f16: (H_kv, N, D) fp16 — all prefill keys."""
        H_kv, N, D = keys_f16.shape
        M = min(self.sample_size, N)
        idx = torch.randperm(N, device=keys_f16.device)[:M]
        self.sample = torch.empty(
            H_kv, self.sample_size, D, device=keys_f16.device, dtype=torch.float16
        )
        self.sample[:, :M, :] = keys_f16[:, idx, :]
        self._filled = M
        self._N = N

    def update(self, new_key_f16: torch.Tensor, total_N: int) -> None:
        """new_key_f16: (H_kv, 1, D) fp16 — one new decoded key."""
        self._N = total_N
        if self._filled < self.sample_size:
            self.sample[:, self._filled, :] = new_key_f16[:, 0, :]
            self._filled += 1
        else:
            j = torch.randint(0, total_N, (1,), device=self.sample.device).item()
            if j < self.sample_size:
                self.sample[:, j, :] = new_key_f16[:, 0, :]

    # ── Threshold computation ────────────────────────────────────────

    def _sample_scores(self, q_f16: torch.Tensor) -> torch.Tensor:
        """
        q_f16: (H_q, D) fp16
        Returns: (H_q, M) float32 raw dot products against sample.
        """
        H_q, D = q_f16.shape
        H_kv = self.sample.shape[0]
        M = self._filled
        g = H_q // H_kv

        q_3d = q_f16.view(H_kv, g, D).float()
        s = self.sample[:, :M, :].float()  # (H_kv, M, D)
        return torch.einsum("hgd,hmd->hgm", q_3d, s).reshape(H_q, M)

    def get_threshold_ta(self, q_f16: torch.Tensor) -> torch.Tensor:
        """
        Returns (H_q,) float32 threshold for TAIndex.attend().
        q_f16: (H_q, D) fp16 raw query.
        """
        if self.sample is None or self._filled == 0:
            return torch.full((q_f16.shape[0],), float("-inf"), device=q_f16.device)

        scores = self._sample_scores(q_f16)  # (H_q, M)

        if self.mode == "budget":
            k = max(1, int(self.budget_fraction * self._filled))
            topk_vals = scores.topk(k, dim=-1).values  # (H_q, k)
            return topk_vals[:, -1].float()  # min of top-k

        # oracle modes
        max_val = scores.max(dim=-1).values  # (H_q,)
        if self.oracle == "sample_max":
            return max_val.float()
        elif self.oracle == "sample_mean_max":
            mean_val = scores.mean(dim=-1)
            return ((max_val + mean_val) / 2).float()
        else:  # sample_top_p: keep top (1-top_p) fraction by score rank
            k = max(1, int((1.0 - self.top_p) * self._filled))
            topk_vals = scores.topk(k, dim=-1).values           # (H_q, k)
            return topk_vals[:, -1].float()                     # min of top-k = threshold

    def get_subspace_threshold(
        self, q_f16: torch.Tensor, dim_slices: list[tuple[int, int]]
    ) -> torch.Tensor:
        """
        Returns (2*S, H_q) fp16 packed threshold for SubspaceKCenterIndex.attend().
        q_f16: (H_q, D) fp16 raw query.
        dim_slices: state["dim_slices"] from the index.
        """
        if self.sample is None or self._filled == 0:
            S = len(dim_slices)
            H_q = q_f16.shape[0]
            neg_inf = torch.full((S, H_q), float("-inf"), device=q_f16.device, dtype=torch.float16)
            q_norms = torch.stack(
                [q_f16[:, s:e].float().norm(dim=-1).half() for s, e in dim_slices], dim=0
            )
            return torch.cat([neg_inf, q_norms], dim=0).contiguous()

        H_q, D = q_f16.shape
        H_kv = self.sample.shape[0]
        M = self._filled
        g = H_q // H_kv

        scores = self._sample_scores(q_f16)  # (H_q, M)

        if self.mode == "budget":
            k = max(1, int(self.budget_fraction * self._filled))
            topk_idx = scores.topk(k, dim=-1).indices  # (H_q, k)
        else:
            # oracle: use keys above oracle threshold as the "top" set
            max_val = scores.max(dim=-1).values  # (H_q,)
            if self.oracle == "sample_mean_max":
                tau = (max_val + scores.mean(dim=-1)) / 2
            else:
                tau = max_val
            # find indices above tau; fall back to top-1 if none
            above = (scores >= tau.unsqueeze(-1))  # (H_q, M)
            k = max(1, int(above.float().sum(dim=-1).max().item()))
            topk_idx = scores.topk(k, dim=-1).indices

        q_3d = q_f16.view(H_kv, g, D)
        sample_f16 = self.sample[:, :M, :]  # (H_kv, M, D)

        ths = []
        for (s, e) in dim_slices:
            q_sub = q_3d[:, :, s:e].float()                      # (H_kv, g, d_s)
            s_sub = sample_f16[:, :, s:e].float()                 # (H_kv, M, d_s)
            sub_scores = torch.einsum("hgd,hmd->hgm", q_sub, s_sub).reshape(H_q, M)
            # clamp topk_idx to valid range
            idx = topk_idx.clamp(0, M - 1)
            sub_topk = sub_scores.gather(1, idx)                  # (H_q, k)
            ths.append(sub_topk.min(dim=-1).values.half())        # (H_q,)

        th = torch.stack(ths, dim=0)  # (S, H_q) fp16
        q_norms = torch.stack(
            [q_f16[:, s:e].float().norm(dim=-1).half() for s, e in dim_slices], dim=0
        )
        return torch.cat([th, q_norms], dim=0).contiguous()
