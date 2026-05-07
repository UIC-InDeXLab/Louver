"""Pruning regression tests for the CUDA indexer/searcher.

These tests verify that the tree-based index actually prunes — i.e.
``synthetic_scanned_fraction`` is meaningfully below 1.0 for realistic
workloads.  They are designed to catch regressions in the build or
update logic that would silently inflate radii and destroy pruning.

The tests use **clustered** synthetic keys (mixtures-of-Gaussians)
which resemble real LLM key distributions far better than pure random
keys.  With enough keys and moderate branching factor the index should
prune at least a configurable fraction of the tree.
"""

import os
import sys
from pathlib import Path

import pytest
import torch


os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

HIRA_ROOT = Path(__file__).resolve().parents[1]
if str(HIRA_ROOT) not in sys.path:
    sys.path.insert(0, str(HIRA_ROOT))

from indexer.cuda import CUDAIndexer
from searcher.cuda import CUDASearcher


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clustered_keys(
    H: int, N: int, D: int, n_clusters: int = 20, seed: int = 0,
    *, centre_norm: float = 15.0, noise_std: float = 0.15,
) -> torch.Tensor:
    """Generate (H, N, D) keys drawn from a mixture of Gaussians.

    Cluster centres are scaled to ``centre_norm`` (default 15, matching
    real LLM key norms of ~14–25), with ``noise_std`` spread around each
    centre.  This gives a realistic dot-product-to-radius ratio so that
    the tree bound actually prunes.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # cluster centres on sphere of radius `centre_norm`
    centres = torch.randn(H, n_clusters, D, device="cuda", generator=gen)
    centres = centres / centres.norm(dim=-1, keepdim=True) * centre_norm
    # assign each key to a random cluster
    assign = torch.randint(0, n_clusters, (H, N), device="cuda", generator=gen)
    keys = centres.gather(
        1, assign.unsqueeze(-1).expand(-1, -1, D)
    ) + noise_std * torch.randn(H, N, D, device="cuda", generator=gen)
    return keys.float()                           # (H, N, D)


def _random_queries(H: int, n_queries: int, D: int, seed: int) -> torch.Tensor:
    """Return (H, n_queries, D) normalised queries on GPU."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(H, n_queries, D, device="cuda", generator=gen)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _avg_scan_fraction(
    indexer: CUDAIndexer,
    queries: torch.Tensor,      # (H_kv, n_queries, D) normalised
    q_head_to_kv: torch.Tensor,
) -> float:
    """Average scanned fraction across all heads and several queries.

    Uses the SampleMaxThreshold strategy (max dot product over a small
    sample) which matches real inference usage.
    """
    searcher = CUDASearcher(block_c=min(8, indexer.branching_factor))
    H_kv, nq, D = queries.shape
    H_q = q_head_to_kv.shape[0]

    # Build a small sample for threshold (like SampleMaxThreshold)
    children = indexer.children                    # (H_kv, N, D)
    N = children.shape[1]
    sample_size = min(100, N)
    sample_idx = torch.randperm(N, device=children.device)[:sample_size]
    sample = children[:, sample_idx, :]            # (H_kv, sample_size, D)
    sample_q = sample.index_select(0, q_head_to_kv)  # (H_q, sample_size, D)

    fracs = []
    for qi in range(nq):
        q = queries[:, qi, :]                         # (H_kv, D)
        q_hq = q.index_select(0, q_head_to_kv)       # (H_q, D)
        # Threshold = max dot product with the sample (SampleMaxThreshold)
        scores_sample = torch.einsum("hd,hsd->hs", q_hq, sample_q.float())
        th = scores_sample.max(dim=-1).values         # (H_q,)
        stats = searcher.synthetic_scanned_fraction(
            q_hq, th, indexer, q_head_to_kv=q_head_to_kv,
        )
        fracs.append(stats["scanned_fraction_mean"])
    return sum(fracs) / len(fracs)


# ---------------------------------------------------------------------------
# 1) Build-only pruning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "depth, N, bf, max_scan",
    [
        (CUDAIndexer.DEPTH.TWO_LEVELS, 2000, 8, 0.75),
        (CUDAIndexer.DEPTH.TWO_LEVELS, 5000, 8, 0.55),
        (CUDAIndexer.DEPTH.THREE_LEVELS, 5000, 8, 0.55),
    ],
    ids=["2L-2000", "2L-5000", "3L-5000"],
)
def test_build_prunes_clustered_keys(depth, N, bf, max_scan):
    """After a full build the scan fraction should be well below 1.0."""
    H, D = 4, 64
    keys = _clustered_keys(H, N, D, seed=100)
    indexer = CUDAIndexer(
        num_levels=depth, branching_factor=bf, max_iterations=10,
    ).build(keys)

    queries = _random_queries(H, 8, D, seed=200)
    q_map = torch.arange(H, device="cuda")
    scan = _avg_scan_fraction(indexer, queries, q_map)
    assert scan < max_scan, (
        f"scan fraction {scan:.4f} >= {max_scan}; pruning may be broken"
    )


# ---------------------------------------------------------------------------
# 2) Build + single update — pruning should not degrade much
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "depth, N, bf, max_scan_delta",
    [
        (CUDAIndexer.DEPTH.TWO_LEVELS, 3000, 8, 0.15),
        (CUDAIndexer.DEPTH.THREE_LEVELS, 3000, 8, 0.15),
    ],
    ids=["2L-update", "3L-update"],
)
def test_update_preserves_pruning(depth, N, bf, max_scan_delta):
    """Build on 70% of keys, update with 30% — scan should stay close to
    the full-build baseline."""
    H, D = 4, 64
    keys = _clustered_keys(H, N, D, seed=300)
    n0 = int(N * 0.7)

    full = CUDAIndexer(
        num_levels=depth, branching_factor=bf, max_iterations=10,
    ).build(keys)

    inc = CUDAIndexer(
        num_levels=depth, branching_factor=bf, max_iterations=10,
    ).build(keys[:, :n0, :].contiguous())
    inc.update(keys[:, n0:, :].contiguous())

    queries = _random_queries(H, 8, D, seed=400)
    q_map = torch.arange(H, device="cuda")

    scan_full = _avg_scan_fraction(full, queries, q_map)
    scan_inc = _avg_scan_fraction(inc, queries, q_map)
    assert scan_inc < scan_full + max_scan_delta, (
        f"incremental scan {scan_inc:.4f} much worse than full "
        f"{scan_full:.4f} (delta={scan_inc - scan_full:.4f} > {max_scan_delta})"
    )


# ---------------------------------------------------------------------------
# 3) Multiple incremental updates — cumulative pruning quality
# ---------------------------------------------------------------------------

def test_multiple_updates_preserve_pruning():
    """Build on 30% of keys, then 7 incremental updates.  Scan fraction
    should remain within a reasonable margin of the full-build baseline."""
    H, D, N, bf = 4, 64, 4000, 8
    depth = CUDAIndexer.DEPTH.TWO_LEVELS
    keys = _clustered_keys(H, N, D, seed=500)

    full = CUDAIndexer(
        num_levels=depth, branching_factor=bf, max_iterations=10,
    ).build(keys)

    n0 = int(N * 0.3)
    inc = CUDAIndexer(
        num_levels=depth, branching_factor=bf, max_iterations=10,
    ).build(keys[:, :n0, :].contiguous())
    chunk = (N - n0) // 7
    pos = n0
    for _ in range(7):
        end = min(pos + chunk, N)
        if end <= pos:
            break
        inc.update(keys[:, pos:end, :].contiguous())
        pos = end
    if pos < N:
        inc.update(keys[:, pos:N, :].contiguous())

    queries = _random_queries(H, 8, D, seed=600)
    q_map = torch.arange(H, device="cuda")

    scan_full = _avg_scan_fraction(full, queries, q_map)
    scan_inc = _avg_scan_fraction(inc, queries, q_map)
    # Many small updates from 30% base realistically degrades pruning;
    # allow up to 2.5x the full-build scan but cap at 0.85 absolute.
    assert scan_inc < min(scan_full * 2.5, 0.85), (
        f"multi-update scan {scan_inc:.4f} too far from full "
        f"{scan_full:.4f} (ratio={scan_inc / max(scan_full, 1e-9):.2f}x)"
    )


# ---------------------------------------------------------------------------
# 4) Radii sanity — parent radii should tightly bound their children
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "depth",
    [CUDAIndexer.DEPTH.TWO_LEVELS, CUDAIndexer.DEPTH.THREE_LEVELS],
    ids=["2L", "3L"],
)
def test_parent_radii_are_tight(depth):
    """Parent radii should be the exact max-L2-distance to children (no
    inflation).  This catches overflow-placement bugs that were the
    original root-cause of broken pruning."""
    H, N, D, bf = 4, 2000, 64, 8
    keys = _clustered_keys(H, N, D, seed=700)
    indexer = CUDAIndexer(
        num_levels=depth, branching_factor=bf, max_iterations=10,
    ).build(keys)

    parents = indexer.parents.float()             # (H, m, D)
    children = indexer.children.float()           # (H, m*bf, D)
    radii = indexer.parent_radii                  # (H, m)
    m = parents.shape[1]

    children_grouped = children.view(H, m, bf, D)
    diffs = children_grouped - parents.unsqueeze(2)
    dists = torch.linalg.norm(diffs, dim=-1)     # (H, m, bf)

    # Mask padded children (they should not be counted)
    pad = float(indexer.pad_value)
    valid = ~torch.all(children_grouped == pad, dim=-1)
    dists_valid = torch.where(valid, dists, torch.tensor(0.0, device=dists.device))
    recomputed = dists_valid.max(dim=2).values   # (H, m)

    # Radii should match recomputed values exactly (up to float rounding)
    torch.testing.assert_close(radii, recomputed, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# 5) Scan fraction decreases as N grows (more keys → more pruning)
# ---------------------------------------------------------------------------

def test_pruning_improves_with_more_keys():
    """With the same branching factor, adding more keys should yield
    better (lower) scan fractions — the tree structure provides more
    opportunity to prune."""
    H, D, bf = 4, 64, 8
    depth = CUDAIndexer.DEPTH.TWO_LEVELS
    queries = _random_queries(H, 8, D, seed=800)
    q_map = torch.arange(H, device="cuda")

    scans = []
    for N in [1000, 3000, 6000]:
        keys = _clustered_keys(H, N, D, seed=900)
        idx = CUDAIndexer(
            num_levels=depth, branching_factor=bf, max_iterations=10,
        ).build(keys)
        scans.append(_avg_scan_fraction(idx, queries, q_map))

    # Each step should prune more (or at least not regress significantly)
    for i in range(1, len(scans)):
        assert scans[i] < scans[i - 1] + 0.05, (
            f"scan fraction did not improve: N-series scans = {scans}"
        )


# ---------------------------------------------------------------------------
# 6) GQA pruning — grouped-query attention should prune similarly
# ---------------------------------------------------------------------------

def test_gqa_pruning():
    """With GQA (H_q > H_kv), pruning should still be effective.  This
    catches regressions where q_head_to_kv mapping breaks the tree bound
    semantics."""
    H_kv, H_q, D, N, bf = 4, 16, 64, 3000, 8
    depth = CUDAIndexer.DEPTH.TWO_LEVELS

    keys = _clustered_keys(H_kv, N, D, seed=1000)
    indexer = CUDAIndexer(
        num_levels=depth, branching_factor=bf, max_iterations=10,
    ).build(keys)

    queries = _random_queries(H_kv, 8, D, seed=1100)
    q_map = (torch.arange(H_q, device="cuda") // (H_q // H_kv))

    # Expand queries to H_q by replicating per KV head
    scan = _avg_scan_fraction(indexer, queries, q_map)
    assert scan < 0.75, (
        f"GQA scan fraction {scan:.4f} >= 0.75; pruning broken with GQA mapping"
    )
