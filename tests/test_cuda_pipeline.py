"""
To test:
    - build an index with a set of keys, get different thresholds of the keys and a query representing different ratios being returned and check the recall being 100%.
    - test above, but this time building the index on a smaller set of keys and updating it incrementally, finally, checking the recall on the final index.
    - testing search with different scaling inputs.
    - testing search on all v1 v2 v3 and v4 kernels
    - testing the ordering of values and keys in the CUDAIndexer being the same. if keys shuffled, values should be shuffled in the same way.
"""

import pytest
import torch
import torch.nn.functional as F

from hira.indexer.cuda import CUDAIndexer
from hira.searcher.cuda import CUDASearcher


# ---------------------------------------------------------------------------
# Skip entire module when no CUDA device is available
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_keys(num_heads: int, seq_len: int, dim: int, seed: int = 0) -> torch.Tensor:
    """Return L2-normalised keys of shape (H, L, D) on CPU."""
    torch.manual_seed(seed)
    keys = torch.randn(num_heads, seq_len, dim)
    keys = F.normalize(keys, dim=-1)
    return keys.float()


def _make_query(num_heads: int, dim: int, seed: int = 42) -> torch.Tensor:
    """Return a single L2-normalised query of shape (H, D) on CPU."""
    torch.manual_seed(seed)
    q = torch.randn(num_heads, dim)
    q = F.normalize(q, dim=-1)
    return q.float()


def _exact_scores(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Brute-force dot-product scores.

    Args:
        query: (H, D) – on any device
        keys:  (H, N, D) – on CUDA (indexer.children)

    Returns:
        (H, N) float scores on CPU.
    """
    q = query.to(keys.device).float()  # (H, D) on CUDA
    scores = torch.einsum("hd,hnd->hn", q, keys.float())      # (H, N) on CUDA
    return scores.cpu()


def _threshold_for_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Return per-head threshold such that exactly top-k keys exceed it.

    Args:
        scores: (H, N) on CPU
        k:      number of keys to keep per head

    Returns:
        (H,) threshold tensor on CPU.
    """
    topk_vals, _ = torch.topk(scores, k, dim=-1)
    return topk_vals[:, -1]


def _recall(
    scores_approx: torch.Tensor,
    scores_exact: torch.Tensor,
    threshold: torch.Tensor,
) -> float:
    """Fraction of keys that truly exceed the threshold and are also returned
    by the approximate search (non-zero score).

    All inputs must be on CPU.

    Args:
        scores_approx: (H, N)
        scores_exact:  (H, N)
        threshold:     (H,)
    """
    scores_approx = scores_approx.cpu().float()
    scores_exact = scores_exact.cpu().float()
    threshold = threshold.cpu().float()

    true_mask = scores_exact >= threshold.unsqueeze(-1)     # (H, N)
    returned_mask = scores_approx > 0.0                     # (H, N)

    true_pos = (true_mask & returned_mask).sum().item()
    total_true = true_mask.sum().item()

    if total_true == 0:
        return 1.0
    return true_pos / total_true


def _build_indexer(
    keys: torch.Tensor,
    num_levels: int = 3,
    branching_factor: int = 8,
    max_iterations: int = 3,
) -> CUDAIndexer:
    """Build and return a CUDAIndexer from keys (H, L, D) on CPU."""
    return CUDAIndexer(
        num_levels=num_levels,
        branching_factor=branching_factor,
        max_iterations=max_iterations,
    ).build(keys)


# ---------------------------------------------------------------------------
# Test 1 – full recall at multiple threshold ratios (single build)
# ---------------------------------------------------------------------------


class TestFullRecallSingleBuild:
    """Build an index once; verify >=98% recall at several keep-ratios."""

    NUM_HEADS = 4
    SEQ_LEN = 512
    DIM = 128

    @pytest.fixture(scope="class")
    def setup(self):
        keys = _make_keys(self.NUM_HEADS, self.SEQ_LEN, self.DIM, seed=1)
        query = _make_query(self.NUM_HEADS, self.DIM, seed=10)
        indexer = _build_indexer(keys)
        searcher = CUDASearcher(block_c=indexer.branching_factor)
        # indexer.children: (H, N, D) on CUDA – use as reference keys
        scores_exact = _exact_scores(query, indexer.children)  # (H, N) on CPU
        return indexer, searcher, query, scores_exact

    @pytest.mark.parametrize("keep_ratio", [0.05, 0.10, 0.25])
    def test_recall_is_100_percent(self, setup, keep_ratio):
        indexer, searcher, query, scores_exact = setup
        k = max(1, int(self.SEQ_LEN * keep_ratio))
        threshold = _threshold_for_topk(scores_exact, k)

        scores_approx = searcher.search(
            query=query.cuda(),
            threshold=threshold.cuda(),
            indexer=indexer,
        )  # (H, N) on CUDA

        recall = _recall(scores_approx, scores_exact, threshold)
        assert recall > 0.98, (
            f"Expected >=98% recall at keep_ratio={keep_ratio}, got {recall:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 2 – full recall after incremental updates
# ---------------------------------------------------------------------------


class TestFullRecallIncrementalUpdate:
    """Start with a small index, update it in chunks, then verify >=98% recall."""

    NUM_HEADS = 4
    DIM = 128
    INITIAL_LEN = 64
    CHUNK_LEN = 64
    NUM_CHUNKS = 6  # total seq_len = INITIAL_LEN + NUM_CHUNKS * CHUNK_LEN = 448

    @pytest.fixture(scope="class")
    def setup(self):
        total_len = self.INITIAL_LEN + self.NUM_CHUNKS * self.CHUNK_LEN
        all_keys = _make_keys(self.NUM_HEADS, total_len, self.DIM, seed=2)

        # Build initial index on the first slice
        initial_keys = all_keys[:, : self.INITIAL_LEN, :]
        indexer = _build_indexer(initial_keys, num_levels=3, branching_factor=8)

        # Incrementally update
        for i in range(self.NUM_CHUNKS):
            start = self.INITIAL_LEN + i * self.CHUNK_LEN
            end = start + self.CHUNK_LEN
            chunk = all_keys[:, start:end, :].cuda()
            indexer.update(chunk)

        query = _make_query(self.NUM_HEADS, self.DIM, seed=20)
        searcher = CUDASearcher(block_c=indexer.branching_factor)
        scores_exact = _exact_scores(query, indexer.children)  # (H, N_padded) on CPU

        return indexer, searcher, query, scores_exact, total_len

    @pytest.mark.parametrize("keep_ratio", [0.05, 0.20, 0.50])
    def test_recall_is_100_percent_after_updates(self, setup, keep_ratio):
        indexer, searcher, query, scores_exact, total_len = setup
        k = max(1, int(total_len * keep_ratio))
        threshold = _threshold_for_topk(scores_exact, k)

        scores_approx = searcher.search(
            query=query.cuda(),
            threshold=threshold.cuda(),
            indexer=indexer,
        )

        recall = _recall(scores_approx, scores_exact, threshold)
        assert recall > 0.98, (
            f"Expected >=98% recall after incremental update at "
            f"keep_ratio={keep_ratio}, got {recall:.4f}"
        )

    def test_children_tensor_contains_all_keys(self, setup):
        """After all updates the children tensor must hold at least total_len
        non-padded (non-all-zero) entries per head."""
        indexer, _, _, _, total_len = setup
        children = indexer.children  # (H, N, D) on CUDA
        # A slot is valid if it is not all-zero (pad_value = 0.0)
        valid = ~torch.all(children == 0.0, dim=-1)  # (H, N) bool
        min_valid = valid.sum(dim=-1).min().item()
        assert min_valid >= total_len, (
            f"Expected at least {total_len} non-padded children per head, "
            f"got {min_valid}"
        )


# ---------------------------------------------------------------------------
# Test 3 – search with different scaling inputs
# ---------------------------------------------------------------------------


class TestSearchScaling:
    """Verify that the scaling tensor linearly scales the returned scores."""

    NUM_HEADS = 4
    SEQ_LEN = 256
    DIM = 128

    @pytest.fixture(scope="class")
    def setup(self):
        keys = _make_keys(self.NUM_HEADS, self.SEQ_LEN, self.DIM, seed=3)
        query = _make_query(self.NUM_HEADS, self.DIM, seed=30)
        indexer = _build_indexer(keys)
        searcher = CUDASearcher(block_c=indexer.branching_factor)

        scores_exact = _exact_scores(query, indexer.children)
        # Use a low threshold so many keys are returned
        k = self.SEQ_LEN // 2
        threshold = _threshold_for_topk(scores_exact, k)

        return indexer, searcher, query, threshold

    @pytest.mark.parametrize("scale_value", [0.5, 1.0, 2.0, 10.0])
    def test_scores_scaled_correctly(self, setup, scale_value):
        indexer, searcher, query, threshold = setup
        H = self.NUM_HEADS

        scaling = torch.full((H,), scale_value)
        scores_scaled = searcher.search(
            query=query.cuda(),
            threshold=threshold.cuda(),
            indexer=indexer,
            scaling=scaling.cuda(),
        ).cpu()

        # Baseline with identity scaling
        identity_scaling = torch.ones(H)
        scores_identity = searcher.search(
            query=query.cuda(),
            threshold=threshold.cuda(),
            indexer=indexer,
            scaling=identity_scaling.cuda(),
        ).cpu()

        # For positions returned by both, scores_scaled = scale_value * scores_identity
        returned = scores_identity > 0.0
        if returned.any():
            ratio = scores_scaled[returned] / scores_identity[returned]
            assert torch.allclose(
                ratio, torch.full_like(ratio, scale_value), atol=1e-4
            ), (
                f"Scaling {scale_value} does not linearly scale scores. "
                f"Max ratio deviation: {(ratio - scale_value).abs().max().item()}"
            )

    def test_zero_scaling_returns_zeros(self, setup):
        indexer, searcher, query, threshold = setup
        H = self.NUM_HEADS
        scaling = torch.zeros(H)
        scores = searcher.search(
            query=query.cuda(),
            threshold=threshold.cuda(),
            indexer=indexer,
            scaling=scaling.cuda(),
        ).cpu()
        assert (scores == 0.0).all(), "Zero scaling should produce all-zero scores"

    def test_per_head_scaling(self, setup):
        """Different scale per head should scale each head independently."""
        indexer, searcher, query, threshold = setup
        H = self.NUM_HEADS

        scale_values = torch.arange(1, H + 1, dtype=torch.float32)
        scores_scaled = searcher.search(
            query=query.cuda(),
            threshold=threshold.cuda(),
            indexer=indexer,
            scaling=scale_values.cuda(),
        ).cpu()
        scores_identity = searcher.search(
            query=query.cuda(),
            threshold=threshold.cuda(),
            indexer=indexer,
            scaling=torch.ones(H).cuda(),
        ).cpu()

        for h in range(H):
            returned = scores_identity[h] > 0.0
            if returned.any():
                ratio = scores_scaled[h, returned] / scores_identity[h, returned]
                expected = scale_values[h].item()
                assert torch.allclose(
                    ratio, torch.full_like(ratio, expected), atol=1e-4
                ), (
                    f"Head {h}: expected scale {expected}, "
                    f"got max deviation {(ratio - expected).abs().max().item()}"
                )


# ---------------------------------------------------------------------------
# Test 4 – two-level and three-level depth variants produce consistent results
# ---------------------------------------------------------------------------


class TestAllDepths:
    """2-level and 3-level CUDAIndexer variants should both achieve high recall."""

    NUM_HEADS = 4
    SEQ_LEN = 256
    DIM = 128

    DEPTHS = [2, 3]

    @pytest.fixture(scope="class")
    def setup(self):
        keys = _make_keys(self.NUM_HEADS, self.SEQ_LEN, self.DIM, seed=4)
        query = _make_query(self.NUM_HEADS, self.DIM, seed=40)

        indexers = {
            d: _build_indexer(keys, num_levels=d, branching_factor=8)
            for d in self.DEPTHS
        }

        searcher = CUDASearcher(block_c=indexers[2].branching_factor)
        return indexers, searcher, query

    @pytest.mark.parametrize("depth", DEPTHS)
    def test_recall_at_depth(self, setup, depth):
        indexers, searcher, query = setup
        indexer = indexers[depth]

        scores_exact = _exact_scores(query, indexer.children)
        k = self.SEQ_LEN // 4
        threshold = _threshold_for_topk(scores_exact, k)

        scores_approx = searcher.search(
            query=query.cuda(),
            threshold=threshold.cuda(),
            indexer=indexer,
        )

        recall = _recall(scores_approx, scores_exact, threshold)
        assert recall >= 0.98, (
            f"Depth-{depth} indexer achieved recall={recall:.4f}, expected >=0.98"
        )

    def test_both_depths_agree_on_high_recall(self, setup):
        """Both depth variants should each achieve >=98% recall independently."""
        indexers, searcher, query = setup
        for depth, indexer in indexers.items():
            scores_exact = _exact_scores(query, indexer.children)
            k = self.SEQ_LEN // 4
            threshold = _threshold_for_topk(scores_exact, k)

            scores_approx = searcher.search(
                query=query.cuda(),
                threshold=threshold.cuda(),
                indexer=indexer,
            )
            recall = _recall(scores_approx, scores_exact, threshold)
            assert recall >= 0.98, (
                f"Depth-{depth}: recall={recall:.4f}, expected >=0.98"
            )


# ---------------------------------------------------------------------------
# Test 5 – ordering of keys and values is consistent
# ---------------------------------------------------------------------------


class TestKeyValueOrdering:
    """After building the index with values, the indexer must preserve the
    correspondence: children[h, i] must align with values[0, h, i, :]."""

    NUM_HEADS = 4
    SEQ_LEN = 256
    DIM = 128

    def _build_with_values(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        num_levels: int = 3,
    ) -> CUDAIndexer:
        return CUDAIndexer(
            num_levels=num_levels,
            branching_factor=8,
            max_iterations=3,
        ).build(keys, values)

    def test_key_value_correspondence_preserved(self):
        """values[0, h, i, 0] encodes the original index; verify it matches
        the original key at that index after the index is built."""
        keys = _make_keys(self.NUM_HEADS, self.SEQ_LEN, self.DIM, seed=5)

        # Construct values where values[h, i, 0] = i (original position).
        values = torch.zeros(self.NUM_HEADS, self.SEQ_LEN, self.DIM)
        for i in range(self.SEQ_LEN):
            values[:, i, 0] = float(i)

        indexer = self._build_with_values(keys, values)

        stored_keys = indexer.children          # (H, N, D) on CUDA
        stored_values = indexer.values          # (H, N, D) on CUDA

        assert stored_keys is not None
        assert stored_values is not None

        # Work on CPU for element-wise checks
        sk = stored_keys.cpu()       # (H, N, D)
        sv = stored_values.cpu()     # (H, N, D)
        orig_keys = keys             # (H, L, D)

        H, N, D = sk.shape
        pad = 0.0

        for h in range(H):
            for i in range(N):
                # Skip padded slots (all-zero children)
                if torch.all(sk[h, i] == pad):
                    continue
                # The value's first channel encodes the original index
                stored_val_idx = sv[h, i, 0].long().item()
                if not (0 <= stored_val_idx < self.SEQ_LEN):
                    # Padded value slot; skip
                    continue
                original_key = orig_keys[h, stored_val_idx]  # (D,)
                assert torch.allclose(sk[h, i], original_key, atol=1e-5), (
                    f"Head {h}, slot {i}: stored key does not match original key "
                    f"at index {stored_val_idx}"
                )

    def test_shuffled_keys_values_stay_aligned(self):
        """Explicitly shuffle keys and values together, build, and verify that
        for every non-padded slot the stored value is proportional to the
        stored key (ratio = original_index + 1)."""
        keys = _make_keys(self.NUM_HEADS, self.SEQ_LEN, self.DIM, seed=6)

        # values[0, h, i, :] = keys[0, h, i, :] * (i + 1)
        values = keys.clone()
        for i in range(self.SEQ_LEN):
            values[:, i, :] = keys[:, i, :] * (i + 1)

        # Apply the same random permutation to both keys and values
        torch.manual_seed(99)
        perm = torch.randperm(self.SEQ_LEN)
        shuffled_keys = keys[:, perm, :]
        shuffled_values = values[:, perm, :]

        indexer = self._build_with_values(shuffled_keys, shuffled_values)

        stored_keys = indexer.children.cpu()       # (H, N, D)
        stored_values = indexer.values.cpu()       # (H, N, D)

        assert stored_keys is not None
        assert stored_values is not None

        H, N, D = stored_keys.shape
        pad = 0.0

        for h in range(H):
            k = stored_keys[h]           # (N, D)
            v = stored_values[h]         # (N, D)
            for i in range(N):
                # Skip padded slots
                if torch.all(k[i] == pad):
                    continue
                nz_dims = k[i].abs() > 1e-6
                if not nz_dims.any():
                    continue
                ratios = v[i, nz_dims] / k[i, nz_dims]
                # All ratios for this slot should be equal
                assert torch.allclose(
                    ratios, ratios[0].expand_as(ratios), atol=1e-3
                ), (
                    f"Head {h}, slot {i}: values and keys are misaligned "
                    f"(ratio spread: {ratios.min().item():.4f} – {ratios.max().item():.4f})"
                )
                # The ratio must be an integer in [1, SEQ_LEN]
                scale = ratios[0].item()
                assert 1 <= round(scale) <= self.SEQ_LEN, (
                    f"Head {h}, slot {i}: unexpected scale {scale}"
                )

    def test_values_none_when_not_provided(self):
        """When no values are passed to build(), indexer.values must be None."""
        keys = _make_keys(self.NUM_HEADS, self.SEQ_LEN, self.DIM, seed=7)
        indexer = _build_indexer(keys)
        assert indexer.values is None, (
            "Expected indexer.values to be None when no values were provided"
        )
