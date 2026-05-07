"""Pruning regression tests for the CPU indexer/searcher.

These tests verify that the CPU tree-based index actually prunes — i.e.
the scanned fraction is meaningfully below 1.0 for realistic workloads.
They mirror the CUDA pruning tests (``test_pruning.py``) but target the
:class:`CPUIndexer` / :class:`CPUSearcher` path.

The tests use clustered synthetic keys (mixtures-of-Gaussians) with
realistic norms (~15, matching real LLM key distributions) and
sample-max thresholding.
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

from indexer.cpu import CPUIndexer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clustered_keys(
    H: int, N: int, D: int, n_clusters: int = 20, seed: int = 0,
    *, centre_norm: float = 15.0, noise_std: float = 0.15,
) -> torch.Tensor:
    """Generate (1, H, N, D) keys from a mixture of Gaussians on CPU."""
    gen = torch.Generator().manual_seed(seed)
    centres = torch.randn(H, n_clusters, D, generator=gen)
    centres = centres / centres.norm(dim=-1, keepdim=True) * centre_norm
    assign = torch.randint(0, n_clusters, (H, N), generator=gen)
    keys = centres.gather(
        1, assign.unsqueeze(-1).expand(-1, -1, D)
    ) + noise_std * torch.randn(H, N, D, generator=gen)
    return keys.unsqueeze(0).float()


def _random_queries(H: int, n_queries: int, D: int, seed: int) -> torch.Tensor:
    """Return (H, n_queries, D) normalised queries on CPU."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(H, n_queries, D, generator=gen)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _synthetic_scanned_fraction_cpu(
    indexer: CPUIndexer,
    query: torch.Tensor,       # (H, D)  normalised
    threshold: torch.Tensor,   # (H,)
) -> float:
    """Analytically compute scanned fraction for the CPU index.

    Walks each tree level (top → bottom, excluding level-0 = leaves).
    At each level the gate is: ``dot(q, center) + radius >= threshold``.
    Returns the mean fraction of leaf keys scanned across heads.
    """
    depth = len(indexer.levels)
    H = query.shape[0]
    N = indexer.levels[0].size     # total leaf keys

    if depth == 1:
        return 1.0                 # single level → no pruning

    # Top level (index depth-1)
    top = indexer.levels[depth - 1]
    scores = torch.einsum("hd,hmd->hm", query, top.ball_centers.float())
    surviving = (scores + top.ball_radii.float()) >= threshold.unsqueeze(-1)  # (H, m_top)

    # Walk down through intermediate levels
    for lvl_idx in range(depth - 2, 0, -1):
        lvl = indexer.levels[lvl_idx]
        # child2parent maps each node at lvl_idx to its parent at lvl_idx+1
        c2p = lvl.child2parent                           # (H, m_lvl)
        # A node survives if its parent survived
        parent_pass = surviving.gather(1, c2p)            # (H, m_lvl)
        # Apply this level's own gate
        scores_lvl = torch.einsum("hd,hmd->hm", query, lvl.ball_centers.float())
        own_pass = (scores_lvl + lvl.ball_radii.float()) >= threshold.unsqueeze(-1)
        surviving = parent_pass & own_pass                # (H, m_lvl)

    # Level 0 = leaves.  child2parent maps each leaf to its parent at level 1.
    leaf = indexer.levels[0]
    c2p_leaf = leaf.child2parent                          # (H, N)
    leaf_pass = surviving.gather(1, c2p_leaf)             # (H, N)

    scanned_per_head = leaf_pass.float().sum(dim=1)       # (H,)
    frac = (scanned_per_head / float(N)).mean().item()
    return frac


def _avg_scan_fraction(
    indexer: CPUIndexer,
    queries: torch.Tensor,       # (H, n_queries, D) normalised
) -> float:
    """Average scanned fraction over several queries using sample-max
    threshold (matches real inference)."""
    H, nq, D = queries.shape
    N = indexer.levels[0].size
    keys = indexer.levels[0].ball_centers               # (H, N, D)

    sample_size = min(100, N)
    sample_idx = torch.randperm(N)[:sample_size]
    sample = keys[:, sample_idx, :]                     # (H, sample_size, D)

    fracs = []
    for qi in range(nq):
        q = queries[:, qi, :]                           # (H, D)
        scores_sample = torch.einsum("hd,hsd->hs", q, sample.float())
        th = scores_sample.max(dim=-1).values           # (H,)
        fracs.append(_synthetic_scanned_fraction_cpu(indexer, q, th))
    return sum(fracs) / len(fracs)


# ---------------------------------------------------------------------------
# 1)  Build-only pruning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "num_levels, N, bf, max_scan",
    [
        (2, 2000, 8, 0.75),
        (2, 5000, 8, 0.55),
        (3, 5000, 8, 0.55),
    ],
    ids=["2L-2000", "2L-5000", "3L-5000"],
)
def test_build_prunes_clustered_keys(num_levels, N, bf, max_scan):
    """After a full build the scan fraction should be well below 1.0."""
    H, D = 4, 64
    keys = _clustered_keys(H, N, D, seed=100)
    indexer = CPUIndexer(
        num_levels=num_levels, branching_factor=bf, max_iterations=10,
    ).build(keys)

    queries = _random_queries(H, 8, D, seed=200)
    scan = _avg_scan_fraction(indexer, queries)
    assert scan < max_scan, (
        f"scan fraction {scan:.4f} >= {max_scan}; pruning may be broken"
    )


# ---------------------------------------------------------------------------
# 2)  Build + single update — pruning should not degrade much
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "num_levels, N, bf, max_scan_delta",
    [
        (2, 3000, 8, 0.15),
        (3, 3000, 8, 0.15),
    ],
    ids=["2L-update", "3L-update"],
)
def test_update_preserves_pruning(num_levels, N, bf, max_scan_delta):
    """Build on 70% of keys, update with 30% — scan stays close to
    the full-build baseline."""
    H, D = 4, 64
    keys = _clustered_keys(H, N, D, seed=300)
    n0 = int(N * 0.7)

    full = CPUIndexer(
        num_levels=num_levels, branching_factor=bf, max_iterations=10,
    ).build(keys)

    inc = CPUIndexer(
        num_levels=num_levels, branching_factor=bf, max_iterations=10,
    ).build(keys[:, :, :n0, :].contiguous())
    inc.update(keys[:, :, n0:, :].contiguous())

    queries = _random_queries(H, 8, D, seed=400)
    scan_full = _avg_scan_fraction(full, queries)
    scan_inc = _avg_scan_fraction(inc, queries)
    assert scan_inc < scan_full + max_scan_delta, (
        f"incremental scan {scan_inc:.4f} much worse than full "
        f"{scan_full:.4f} (delta={scan_inc - scan_full:.4f} > {max_scan_delta})"
    )


# ---------------------------------------------------------------------------
# 3)  Multiple incremental updates — cumulative pruning quality
# ---------------------------------------------------------------------------

def test_multiple_updates_preserve_pruning():
    """Build on 30%, then 7 incremental updates. Scan should remain
    within a reasonable margin of full build."""
    H, D, N, bf = 4, 64, 4000, 8
    num_levels = 2
    keys = _clustered_keys(H, N, D, seed=500)

    full = CPUIndexer(
        num_levels=num_levels, branching_factor=bf, max_iterations=10,
    ).build(keys)

    n0 = int(N * 0.3)
    inc = CPUIndexer(
        num_levels=num_levels, branching_factor=bf, max_iterations=10,
    ).build(keys[:, :, :n0, :].contiguous())
    chunk = (N - n0) // 7
    pos = n0
    for _ in range(7):
        end = min(pos + chunk, N)
        if end <= pos:
            break
        inc.update(keys[:, :, pos:end, :].contiguous())
        pos = end
    if pos < N:
        inc.update(keys[:, :, pos:N, :].contiguous())

    queries = _random_queries(H, 8, D, seed=600)
    scan_full = _avg_scan_fraction(full, queries)
    scan_inc = _avg_scan_fraction(inc, queries)
    # Many small updates from 30% base realistically degrades pruning;
    # allow up to 2.5x the full-build scan but cap at 0.85 absolute.
    assert scan_inc < min(scan_full * 2.5, 0.85), (
        f"multi-update scan {scan_inc:.4f} too far from full "
        f"{scan_full:.4f} (ratio={scan_inc / max(scan_full, 1e-9):.2f}x)"
    )


# ---------------------------------------------------------------------------
# 4)  Radii sanity — parent radii tightly bound children
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "num_levels",
    [2, 3],
    ids=["2L", "3L"],
)
def test_parent_radii_bound_children(num_levels):
    """For every parent, its radius must be >= the max (dist-to-child-center
    + child-radius) over assigned children.  This is the triangle-inequality
    bound used by the CPU indexer."""
    H, N, D, bf = 4, 2000, 64, 8
    keys = _clustered_keys(H, N, D, seed=700)
    indexer = CPUIndexer(
        num_levels=num_levels, branching_factor=bf, max_iterations=10,
    ).build(keys)

    for lvl_idx in range(len(indexer.levels) - 1):
        child = indexer.levels[lvl_idx]
        parent = indexer.levels[lvl_idx + 1]
        c2p = child.child2parent                         # (H, C)

        # Gather parent centre for each child
        parent_for_child = parent.ball_centers.gather(
            1, c2p.unsqueeze(-1).expand(-1, -1, D)
        )                                                # (H, C, D)
        dist = torch.linalg.norm(
            (child.ball_centers - parent_for_child).float(), dim=-1
        )                                                # (H, C)
        contrib = dist + child.ball_radii.float()        # (H, C)

        # Recompute expected radius per parent via scatter-max
        K = parent.ball_centers.shape[1]
        expected = torch.full((H, K), 0.0)
        expected.scatter_reduce_(1, c2p, contrib, reduce="amax", include_self=True)

        # Parent radii should be >= expected (allowing float tolerance)
        too_small = (parent.ball_radii.float() < expected - 1e-3)
        assert not too_small.any(), (
            f"Level {lvl_idx+1} has parent radii smaller than the child bound "
            f"(max deficit = {(expected - parent.ball_radii.float())[too_small].max():.4f})"
        )


# ---------------------------------------------------------------------------
# 5)  Scan fraction decreases as N grows
# ---------------------------------------------------------------------------

def test_pruning_improves_with_more_keys():
    """More keys with the same branching factor should yield better
    (lower) scan fractions."""
    H, D, bf = 4, 64, 8
    num_levels = 2
    queries = _random_queries(H, 8, D, seed=800)

    scans = []
    for N in [1000, 3000, 6000]:
        keys = _clustered_keys(H, N, D, seed=900)
        idx = CPUIndexer(
            num_levels=num_levels, branching_factor=bf, max_iterations=10,
        ).build(keys)
        scans.append(_avg_scan_fraction(idx, queries))

    for i in range(1, len(scans)):
        assert scans[i] < scans[i - 1] + 0.05, (
            f"scan fraction did not improve: N-series scans = {scans}"
        )


# ---------------------------------------------------------------------------
# 6)  3-level index prunes at least as well as 2-level
# ---------------------------------------------------------------------------

def test_three_levels_prune_at_least_as_well_as_two():
    """A 3-level tree should scan no more than the 2-level tree (the extra
    level provides finer-grained pruning)."""
    H, D, N, bf = 4, 64, 5000, 8
    keys = _clustered_keys(H, N, D, seed=1000)
    queries = _random_queries(H, 8, D, seed=1100)

    idx2 = CPUIndexer(num_levels=2, branching_factor=bf, max_iterations=10).build(keys)
    idx3 = CPUIndexer(num_levels=3, branching_factor=bf, max_iterations=10).build(keys)

    scan2 = _avg_scan_fraction(idx2, queries)
    scan3 = _avg_scan_fraction(idx3, queries)
    # 3-level should be at least as good (allow small tolerance)
    assert scan3 <= scan2 + 0.05, (
        f"3-level scan {scan3:.4f} worse than 2-level {scan2:.4f}"
    )
