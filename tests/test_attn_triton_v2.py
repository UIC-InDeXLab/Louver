"""
Correctness tests for the v2 HIRA attention Triton kernels.

Verifies:
  1. Single-block kernel matches PyTorch reference
  2. SplitKV kernel matches PyTorch reference
  3. Ring-buffer queued_len < buffer_size is respected
  4. Edge cases: Q_LEN=0, N=0, all weights pruned
"""

import pytest
import torch
import math

from hira.attention.kernels.attn_triton_kernels_v2 import (
    fused_hira_attention_v2,
    _compute_num_splits,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Triton kernel tests",
)


# ──────────────────────────────────────────────────────────────────────
# Reference implementation (pure PyTorch)
# ──────────────────────────────────────────────────────────────────────
def _reference_hira_attention(
    index_weights: torch.Tensor,  # (H_q, N)
    index_values: torch.Tensor,   # (H_kv, N, D)
    query: torch.Tensor,          # (H_q, D)
    queued_keys: torch.Tensor,    # (H_kv, BUF, D)
    queued_values: torch.Tensor,  # (H_kv, BUF, D)
    queued_len: int,
    q_head_to_kv: torch.Tensor,   # (H_q,)
    scaling: float,
) -> torch.Tensor:
    """Naive PyTorch attention for correctness comparison."""
    H_q, D = query.shape
    H_kv = index_values.shape[0]
    N = index_values.shape[1]

    output = torch.zeros(H_q, D, device=query.device, dtype=torch.float32)

    for h_q in range(H_q):
        kv_h = q_head_to_kv[h_q].item()

        # Collect all weights and values
        all_weights = []
        all_values = []

        # Indexed keys: use pre-computed weights
        iw = index_weights[h_q].float()  # (N,)
        iv = index_values[kv_h].float()  # (N, D)

        # Mark pruned entries (w == 0) as -inf
        mask = iw != 0.0
        iw_masked = torch.where(mask, iw, torch.tensor(float("-inf"), device=iw.device))
        all_weights.append(iw_masked)
        all_values.append(iv)

        # Queued keys: compute QK on the fly
        if queued_len > 0:
            qk = queued_keys[kv_h, :queued_len].float()  # (Q_LEN, D)
            qv = queued_values[kv_h, :queued_len].float()  # (Q_LEN, D)
            q_vec = query[h_q].float()  # (D,)
            w = (qk @ q_vec) * scaling  # (Q_LEN,)
            all_weights.append(w)
            all_values.append(qv)

        if len(all_weights) == 0:
            continue

        weights = torch.cat(all_weights, dim=0)  # (N + Q_LEN,)
        values = torch.cat(all_values, dim=0)    # (N + Q_LEN, D)

        # Softmax
        w_max = weights.max()
        if w_max == float("-inf"):
            continue
        exp_w = torch.exp(weights - w_max)
        exp_w = torch.where(weights == float("-inf"), torch.zeros_like(exp_w), exp_w)
        denom = exp_w.sum()
        if denom > 0:
            probs = exp_w / denom
            output[h_q] = (probs.unsqueeze(-1) * values).sum(dim=0)

    return output


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _make_test_data(
    H_q=8, H_kv=2, N=512, Q_LEN=16, D=128, BUF=64, prune_rate=0.5, seed=42,
):
    """Generate random test data with controlled sparsity."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    query = torch.randn(H_q, D, device="cuda", dtype=torch.float16, generator=g)
    index_values = torch.randn(H_kv, N, D, device="cuda", dtype=torch.float16, generator=g)
    queued_keys = torch.randn(H_kv, BUF, D, device="cuda", dtype=torch.float16, generator=g)
    queued_values = torch.randn(H_kv, BUF, D, device="cuda", dtype=torch.float16, generator=g)

    # Simulate searcher output: random weights with some pruned (zeroed)
    index_weights = torch.randn(H_q, N, device="cuda", dtype=torch.float16, generator=g)
    prune_mask = torch.rand(H_q, N, device="cuda", generator=g) < prune_rate
    index_weights[prune_mask] = 0.0

    num_kv_groups = H_q // H_kv
    q_head_to_kv = (
        torch.arange(H_q, device="cuda", dtype=torch.int64) // num_kv_groups
    )

    scaling = 1.0 / math.sqrt(D)

    return dict(
        index_weights=index_weights,
        index_values=index_values,
        query=query,
        queued_keys=queued_keys,
        queued_values=queued_values,
        queued_len=Q_LEN,
        q_head_to_kv=q_head_to_kv,
        scaling=scaling,
    )


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("H_q,H_kv,N,Q_LEN,D", [
    (8, 2, 256, 16, 128),     # small, single-block
    (8, 2, 256, 16, 64),      # D=64
    (28, 4, 1024, 32, 128),   # medium
    (28, 4, 4096, 64, 128),   # large, likely SplitKV
    (32, 8, 8192, 128, 128),  # large, definitely SplitKV
])
def test_correctness_vs_reference(H_q, H_kv, N, Q_LEN, D):
    """Verify kernel output matches PyTorch reference within tolerance."""
    data = _make_test_data(H_q=H_q, H_kv=H_kv, N=N, Q_LEN=Q_LEN, D=D, BUF=max(Q_LEN, 64))

    result = fused_hira_attention_v2(**data)
    reference = _reference_hira_attention(**data)

    result_f32 = result.float()
    torch.testing.assert_close(
        result_f32, reference,
        atol=1e-2, rtol=1e-2,
        msg=f"Mismatch for H_q={H_q}, H_kv={H_kv}, N={N}, Q_LEN={Q_LEN}, D={D}",
    )


@pytest.mark.parametrize("Q_LEN", [0, 1, 3, 16, 63])
def test_queued_len_respected(Q_LEN):
    """
    Ensure that only queued_len entries are used, even when BUF > queued_len.
    The ring buffer has garbage data beyond queued_len which must be ignored.
    """
    H_q, H_kv, N, D, BUF = 8, 2, 256, 128, 64

    data = _make_test_data(H_q=H_q, H_kv=H_kv, N=N, Q_LEN=Q_LEN, D=D, BUF=BUF)

    # Fill beyond queued_len with huge values that would corrupt output
    if Q_LEN < BUF:
        data["queued_keys"][:, Q_LEN:, :] = 1e4
        data["queued_values"][:, Q_LEN:, :] = 1e4

    result = fused_hira_attention_v2(**data)
    reference = _reference_hira_attention(**data)

    result_f32 = result.float()
    torch.testing.assert_close(
        result_f32, reference,
        atol=1e-2, rtol=1e-2,
        msg=f"queued_len={Q_LEN} not respected (data beyond ring buffer leaked)",
    )


def test_zero_indexed_keys():
    """When N=0, only queued keys should contribute."""
    H_q, H_kv, D, Q_LEN, BUF = 8, 2, 128, 16, 64

    data = _make_test_data(H_q=H_q, H_kv=H_kv, N=0, Q_LEN=Q_LEN, D=D, BUF=BUF)
    # N=0 means index_weights is (H_q, 0)
    data["index_weights"] = torch.empty(H_q, 0, device="cuda", dtype=torch.float16)
    data["index_values"] = torch.empty(H_kv, 0, D, device="cuda", dtype=torch.float16)

    result = fused_hira_attention_v2(**data)
    reference = _reference_hira_attention(**data)

    torch.testing.assert_close(result.float(), reference, atol=1e-2, rtol=1e-2)


def test_zero_queued_keys():
    """When queued_len=0, only indexed keys should contribute."""
    H_q, H_kv, N, D, BUF = 8, 2, 512, 128, 64

    data = _make_test_data(H_q=H_q, H_kv=H_kv, N=N, Q_LEN=0, D=D, BUF=BUF)

    result = fused_hira_attention_v2(**data)
    reference = _reference_hira_attention(**data)

    torch.testing.assert_close(result.float(), reference, atol=1e-2, rtol=1e-2)


def test_all_pruned():
    """When all index weights are 0 and queued_len=0, output should be zeros."""
    H_q, H_kv, N, D, BUF = 8, 2, 256, 128, 64

    data = _make_test_data(H_q=H_q, H_kv=H_kv, N=N, Q_LEN=0, D=D, BUF=BUF)
    data["index_weights"].zero_()

    result = fused_hira_attention_v2(**data)
    assert torch.allclose(result.float(), torch.zeros_like(result.float()), atol=1e-6)


def test_splitkv_heuristic():
    """Verify the split heuristic makes reasonable choices."""
    # With H_q=28 and 170 SMs, we need ~12 splits to fill
    splits = _compute_num_splits(H_q=28, total_kv=8192, num_sms=170)
    assert splits > 1, "Should use SplitKV for H_q=28 on 170 SMs"
    assert splits * 28 >= 170 * 0.5, "Should achieve reasonable SM utilization"

    # With H_q=340 (2x SMs), single block is fine
    splits = _compute_num_splits(H_q=340, total_kv=8192, num_sms=170)
    assert splits == 1, "Should not split when H_q already exceeds SMs"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dtype_support(dtype):
    """Verify both fp16 and bf16 work."""
    H_q, H_kv, N, Q_LEN, D, BUF = 8, 2, 256, 16, 128, 64

    data = _make_test_data(H_q=H_q, H_kv=H_kv, N=N, Q_LEN=Q_LEN, D=D, BUF=BUF)
    # Cast all tensors to target dtype
    for k in ["index_weights", "index_values", "query", "queued_keys", "queued_values"]:
        data[k] = data[k].to(dtype)

    result = fused_hira_attention_v2(**data)
    assert result.dtype == dtype
    assert not torch.isnan(result).any(), f"NaN in output for {dtype}"
    assert not torch.isinf(result).any(), f"Inf in output for {dtype}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

