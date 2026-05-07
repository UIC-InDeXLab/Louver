import os
import sys
from pathlib import Path

import pytest
import torch


os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("MAX_JOBS", "4")
Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

HIRA_ROOT = Path(__file__).resolve().parents[1]
if str(HIRA_ROOT) not in sys.path:
    sys.path.insert(0, str(HIRA_ROOT))

from indexer.cpu import CPUIndexer
from indexer.cuda import CUDAIndexer
from searcher.cpu import CPUSearcher
from searcher.cuda import CUDASearcher


def _normalized_positive_keys(h: int, n: int, d: int, seed: int, device: str):
    if device == "cuda":
        g = torch.Generator(device="cuda").manual_seed(seed)
    else:
        g = torch.Generator().manual_seed(seed)
    x = torch.rand((1, h, n, d), generator=g, device=device, dtype=torch.float32)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _normalized_positive_query(h: int, d: int, seed: int, device: str):
    if device == "cuda":
        g = torch.Generator(device="cuda").manual_seed(seed)
    else:
        g = torch.Generator().manual_seed(seed)
    q = torch.rand((1, h, 1, d), generator=g, device=device, dtype=torch.float32)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _bruteforce_scores(keys_hnd: torch.Tensor, query_1h1d: torch.Tensor):
    q = query_1h1d.squeeze(0).squeeze(-2).float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.einsum("hd,hnd->hn", q, keys_hnd.float())


def _head_to_kv(num_query_heads: int, num_kv_heads: int, device: torch.device):
    if num_query_heads == num_kv_heads:
        return torch.arange(num_query_heads, device=device, dtype=torch.long)
    if (num_query_heads % num_kv_heads) != 0:
        raise ValueError(
            f"H_q={num_query_heads} must be divisible by H_kv={num_kv_heads}"
        )
    group_size = num_query_heads // num_kv_heads
    return torch.arange(num_query_heads, device=device, dtype=torch.long) // group_size


def _grouped_bruteforce_scores(
    keys_kv_hnd: torch.Tensor, query_1h1d: torch.Tensor, q_head_to_kv: torch.Tensor
):
    q = query_1h1d.squeeze(0).squeeze(-2).float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    keys_for_query = keys_kv_hnd.index_select(0, q_head_to_kv.long())
    return torch.einsum("hd,hnd->hn", q, keys_for_query.float())


def _query_layout(query_1h1d: torch.Tensor, layout: str) -> torch.Tensor:
    """Return query as (H, D) — both layouts now collapse to 2-D for the
    current CUDA searcher API."""
    if layout in ("4d", "2d"):
        return query_1h1d.squeeze(0).squeeze(-2).contiguous()
    raise ValueError(f"Unsupported query layout: {layout}")


def _half_threshold(scores: torch.Tensor) -> torch.Tensor:
    n = scores.shape[-1]
    assert n % 2 == 0, "n must be even to split exactly in half"
    s = torch.sort(scores, dim=-1).values
    lo = s[:, (n // 2) - 1]
    hi = s[:, n // 2]
    return 0.5 * (lo + hi)


def _assert_exact_set_and_scores(
    pred_scores: torch.Tensor,
    gt_scores: torch.Tensor,
    threshold: torch.Tensor,
    atol: float = 1e-4,
):
    gt_mask = gt_scores > threshold.unsqueeze(-1)
    pred_mask = pred_scores != 0
    assert torch.equal(pred_mask, gt_mask)

    expected = torch.where(gt_mask, gt_scores, torch.zeros_like(gt_scores))
    torch.testing.assert_close(pred_scores, expected, atol=atol, rtol=1e-4)


def _choose_block_c(branching_factor: int) -> int:
    for c in (16, 8, 4, 2, 1):
        if c <= branching_factor and branching_factor % c == 0:
            return c
    raise ValueError(f"No valid BLOCK_C for branching_factor={branching_factor}")


@pytest.mark.parametrize(
    "h,n,d,seed",
    [
        (1, 128, 32, 11),
        (2, 256, 64, 12),
        (4, 192, 32, 13),
    ],
)
def test_cpu_indexer_build_then_search_recall(h, n, d, seed):
    """
    Build a cpu indexer with a set of keys.
    then randomly choose a normalized query vector q.
    for a threshold that returns half of the keys, the cpuindexer built on the keys
    returns exactly the same set of keys (100 recall)
    """
    keys = _normalized_positive_keys(h, n, d, seed=seed, device="cpu")

    indexer = CPUIndexer(num_levels=1, branching_factor=8, max_iterations=1).build(keys)
    query = _normalized_positive_query(h, d, seed=seed + 100, device="cpu")

    gt_scores = _bruteforce_scores(indexer.keys, query)
    threshold = _half_threshold(gt_scores)

    searcher = CPUSearcher()
    pred_scores = searcher.search(query, threshold, indexer)

    _assert_exact_set_and_scores(pred_scores, gt_scores, threshold)


@pytest.mark.parametrize(
    "h,total,d,seed",
    [
        (1, 128, 32, 21),
        (2, 256, 64, 22),
        (3, 192, 32, 23),
    ],
)
def test_cpu_indexer_build_update_search_recall(h, total, d, seed):
    """
    Build a cpu indexer with a set of keys.
    then randomly choose a normalized query vector q.
    for a threshold that returns half of the keys, the cpuindexer build on small keys then updated incrementally
    returns exactly the same set of keys (100 recall)
    """
    n0 = total // 2

    all_keys = _normalized_positive_keys(h, total, d, seed=seed, device="cpu")
    base_keys = all_keys[:, :, :n0, :].contiguous()
    new_keys = all_keys[:, :, n0:, :].contiguous()

    indexer = CPUIndexer(num_levels=1, branching_factor=8, max_iterations=1)
    indexer.build(base_keys)
    indexer.update(new_keys)

    assert indexer.num_keys == total

    query = _normalized_positive_query(h, d, seed=seed + 100, device="cpu")
    gt_scores = _bruteforce_scores(indexer.keys, query)
    threshold = _half_threshold(gt_scores)

    searcher = CPUSearcher()
    pred_scores = searcher.search(query, threshold, indexer)

    _assert_exact_set_and_scores(pred_scores, gt_scores, threshold)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "h_kv,h_q,n,d,query_layout,seed",
    [
        (1, 1, 128, 32, "4d", 31),
        (2, 2, 256, 64, "2d", 32),
        (2, 4, 256, 64, "4d", 33),
    ],
)
def test_cuda_indexer_build_then_search_recall(h_kv, h_q, n, d, query_layout, seed):
    """
    Build a cuda indexer with a set of keys.
    then randomly choose a normalized query vector q.
    for a threshold that returns half of the keys, the cuda indexer built on the keys
    returns exactly the same set of keys (100 recall)
    """
    bf = 1

    keys = _normalized_positive_keys(h_kv, n, d, seed=seed, device="cuda")
    indexer = CUDAIndexer(
        num_levels=CUDAIndexer.DEPTH.TWO_LEVELS,
        max_iterations=1,
        branching_factor=bf,
        pad_value=0.0,
    ).build(keys)

    assert indexer.children is not None

    query = _normalized_positive_query(h_q, d, seed=seed + 100, device="cuda")
    q_head_to_kv = _head_to_kv(
        num_query_heads=h_q,
        num_kv_heads=h_kv,
        device=indexer.children.device,
    )
    gt_scores = _grouped_bruteforce_scores(indexer.children, query, q_head_to_kv)
    threshold = _half_threshold(gt_scores)
    query_for_search = _query_layout(query, query_layout)

    searcher = CUDASearcher(block_c=_choose_block_c(bf), output_fill_value=0.0)
    pred_scores = searcher.search(query_for_search, threshold, indexer)

    _assert_exact_set_and_scores(pred_scores, gt_scores, threshold, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "h_kv,h_q,total,d,query_layout,seed",
    [
        (1, 1, 128, 32, "4d", 41),
        (2, 2, 256, 64, "4d", 42),
        (2, 4, 256, 64, "2d", 43),
    ],
)
def test_cuda_indexer_build_update_search_recall(
    h_kv, h_q, total, d, query_layout, seed
):
    """
    Build a cuda indexer with a set of keys.
    then randomly choose a normalized query vector q.
    for a threshold that returns half of the keys, the cuda indexer build on small keys then updated incrementally
    returns exactly the same set of keys (100 recall)
    """
    n0 = total // 2
    bf = 1

    all_keys = _normalized_positive_keys(h_kv, total, d, seed=seed, device="cuda")
    base_keys = all_keys[:, :, :n0, :].contiguous()
    new_keys = all_keys[:, :, n0:, :].squeeze(0).contiguous()  # (H, M, D)

    indexer = CUDAIndexer(
        num_levels=CUDAIndexer.DEPTH.TWO_LEVELS,
        max_iterations=1,
        branching_factor=bf,
        pad_value=0.0,
    )
    indexer.build(base_keys)
    indexer.update(new_keys)

    assert indexer.children is not None
    assert indexer.children.shape[1] == total

    query = _normalized_positive_query(h_q, d, seed=seed + 100, device="cuda")
    q_head_to_kv = _head_to_kv(
        num_query_heads=h_q,
        num_kv_heads=h_kv,
        device=indexer.children.device,
    )
    gt_scores = _grouped_bruteforce_scores(indexer.children, query, q_head_to_kv)
    threshold = _half_threshold(gt_scores)
    query_for_search = _query_layout(query, query_layout)

    searcher = CUDASearcher(block_c=_choose_block_c(bf), output_fill_value=0.0)
    pred_scores = searcher.search(query_for_search, threshold, indexer)

    _assert_exact_set_and_scores(pred_scores, gt_scores, threshold, atol=1e-3)


def test_cpu_searcher_scaling_outputs():
    """
    The output of searcher should be exactly the same as when scaling=1, but when the scaling is provided, only the numbers are multiplied by scaling
    """
    h, n, d = 2, 128, 32
    scaling = torch.tensor([0.5, 0.8], dtype=torch.float32)

    keys = _normalized_positive_keys(h, n, d, seed=501, device="cpu")
    query = _normalized_positive_query(h, d, seed=502, device="cpu")
    indexer = CPUIndexer(num_levels=1, branching_factor=8, max_iterations=1).build(keys)

    gt_scores = _bruteforce_scores(indexer.keys, query)
    threshold = _half_threshold(gt_scores)

    searcher = CPUSearcher(search_strategy="fused_v1")
    baseline = searcher.search(
        query, threshold, indexer, scaling=torch.ones((h,), dtype=torch.float32)
    )
    scaled = searcher.search(query, threshold, indexer, scaling=scaling)

    assert torch.equal(scaled != 0, baseline != 0)
    torch.testing.assert_close(
        scaled, baseline * scaling.unsqueeze(-1), atol=1e-5, rtol=1e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_searcher_scaling_outputs():
    """
    The output of searcher should be exactly the same as when scaling=1, but when the scaling is provided, only the numbers are multiplied by scaling
    """
    h, n, d = 2, 256, 64
    scaling = torch.tensor([0.6, 0.9], device="cuda", dtype=torch.float32)
    bf = 1

    keys = _normalized_positive_keys(h, n, d, seed=601, device="cuda")
    query = _normalized_positive_query(h, d, seed=602, device="cuda")
    query_2d = query.squeeze(0).squeeze(-2).contiguous()  # (H, D)

    indexer = CUDAIndexer(
        num_levels=CUDAIndexer.DEPTH.TWO_LEVELS,
        max_iterations=1,
        branching_factor=bf,
        pad_value=0.0,
    ).build(keys)

    assert indexer.children is not None
    gt_scores = _bruteforce_scores(indexer.children, query)
    threshold = _half_threshold(gt_scores)

    searcher = CUDASearcher(block_c=_choose_block_c(bf), output_fill_value=0.0)
    baseline = searcher.search(
        query_2d,
        threshold,
        indexer,
        scaling=torch.ones((h,), device="cuda", dtype=torch.float32),
    ).clone()
    scaled = searcher.search(query_2d, threshold, indexer, scaling=scaling)

    assert torch.equal(scaled != 0, baseline != 0)
    torch.testing.assert_close(
        scaled, baseline * scaling.unsqueeze(-1), atol=1e-4, rtol=1e-4
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_indexer_reorders_values():
    """
    Given a list of keys, build cudaindexer on a couple of them and then update it by rest.
    the ordering of values and keys should be exactly the same, meaning that if the keys are shuffled, the same
    shuffling should be applied to values.
    """
    h, total, d = 2, 320, 32
    n0 = 40
    bf = 8

    all_keys = _normalized_positive_keys(h, total, d, seed=701, device="cuda")
    all_values = all_keys * 3.0 - 0.7

    base_keys = all_keys[:, :, :n0, :].contiguous()
    base_values = all_values[:, :, :n0, :].contiguous()
    new_keys = all_keys[:, :, n0:, :].squeeze(0).contiguous()   # (H, M, D)
    new_values = all_values[:, :, n0:, :].squeeze(0).contiguous()  # (H, M, D)

    indexer = CUDAIndexer(
        num_levels=CUDAIndexer.DEPTH.TWO_LEVELS,
        max_iterations=1,
        branching_factor=bf,
        pad_value=0.0,
    ).build(base_keys, base_values)
    indexer.update(new_keys, new_values)

    assert indexer.children is not None
    assert indexer.values is not None

    # values & children are both (H, N, D)
    expected_values = indexer.children * 3.0 - 0.7
    valid_rows = ~torch.all(indexer.children == indexer.pad_value, dim=-1)
    valid_mask = valid_rows.unsqueeze(-1).expand_as(indexer.values)

    torch.testing.assert_close(
        indexer.values[valid_mask], expected_values[valid_mask], atol=1e-6, rtol=0.0
    )

    invalid_mask = ~valid_mask
    if invalid_mask.any():
        torch.testing.assert_close(
            indexer.values[invalid_mask],
            torch.zeros_like(indexer.values[invalid_mask]),
            atol=0.0,
            rtol=0.0,
        )
