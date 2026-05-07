"""Fast approximate nearest-neighbor pairing for bf=2.

The exact nn_greedy is O(N²) per head — too slow for large N.
This implements fast approximations:

1. block_nn: Partition into blocks of size B, do NN-greedy within blocks.
   Cost: O(N * B) per head. For B=64: ~32x faster than exact at N=2000.

2. proj_nn: Sort by multiple random projections, pair nearest in sorted order.
   Cost: O(N * R * log N) per head. Very fast for large N.

3. kdtree_nn: Use recursive median-split (simplified KD-tree) for NN.
   Cost: O(N * log N * log N) per head.
"""

from __future__ import annotations

import math
import torch


def cluster_block_nn(keys: torch.Tensor, bf: int, block_size: int = 64):
    """
    Block-based NN greedy: partition into blocks, pair within blocks.

    For bf=2: O(N * block_size * D) per head.
    """
    H, N, D = keys.shape
    device = keys.device
    K = max(1, math.ceil(N / bf))

    assign = torch.full((H, N), -1, dtype=torch.long, device=device)

    for h in range(H):
        keys_h = keys[h]

        # Random permutation to avoid systematic bias
        perm = torch.randperm(N, device=device)
        keys_perm = keys_h[perm]

        group_id = 0
        paired = torch.zeros(N, dtype=torch.bool, device=device)

        # Process blocks
        n_blocks = math.ceil(N / block_size)
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, N)
            block_idx = torch.arange(start, end, device=device)

            # Get unpaired points in this block
            unpaired_mask = ~paired[block_idx]
            unpaired_local = block_idx[unpaired_mask]

            if len(unpaired_local) < 2:
                continue

            block_keys = keys_perm[unpaired_local]
            n_block = len(unpaired_local)

            # Pairwise distances within block
            dists = torch.cdist(block_keys.unsqueeze(0), block_keys.unsqueeze(0)).squeeze(0)
            dists.diagonal().fill_(float("inf"))

            # Greedy NN pairing within block
            avail = torch.ones(n_block, dtype=torch.bool, device=device)
            for _ in range(n_block // bf):
                if avail.sum() < bf:
                    break

                # Mask unavailable
                d_masked = dists.clone()
                d_masked[~avail] = float("inf")
                d_masked[:, ~avail] = float("inf")

                flat_min = d_masked.argmin()
                i, j = flat_min // n_block, flat_min % n_block

                # Assign pair
                for idx in [unpaired_local[i], unpaired_local[j]]:
                    assign[h, perm[idx]] = group_id
                    paired[idx] = True

                avail[i] = False
                avail[j] = False
                group_id += 1

        # Handle remaining unpaired points
        remaining = (~paired).nonzero(as_tuple=True)[0]
        for i in range(0, len(remaining), bf):
            chunk = remaining[i:i+bf]
            for idx in chunk:
                assign[h, perm[idx]] = min(group_id, K-1)
            group_id += 1

    assign = assign.clamp(0, K-1)
    centers = _compute_centers(keys, assign, K)
    return assign, centers


def cluster_proj_nn(keys: torch.Tensor, bf: int, n_probes: int = 8):
    """
    Multi-probe projection NN: sort by R random directions, find nearest
    pairs in each sorted order, then greedily match closest pairs.

    For bf=2: O(N * R * log N + N * R) per head. Very fast.
    """
    H, N, D = keys.shape
    device = keys.device
    K = max(1, math.ceil(N / bf))

    # Generate random projection directions
    probes = torch.randn(H, n_probes, D, device=device, dtype=keys.dtype)
    probes = probes / probes.norm(dim=-1, keepdim=True)

    # Project keys onto each probe direction: (H, R, N)
    projections = torch.bmm(probes, keys.transpose(1, 2))  # (H, n_probes, N)

    assign = torch.full((H, N), -1, dtype=torch.long, device=device)

    for h in range(H):
        # Collect candidate pairs from all probes
        # For each probe, consecutive elements in sorted order are candidates
        candidates = []  # list of (distance, i, j)

        for r in range(n_probes):
            sorted_idx = projections[h, r].argsort()
            # Consecutive pairs in sorted order
            for s in range(0, N - 1, 1):
                i, j = sorted_idx[s].item(), sorted_idx[s+1].item()
                dist = (keys[h, i] - keys[h, j]).square().sum().item()
                candidates.append((dist, min(i, j), max(i, j)))

        # Deduplicate and sort by distance
        seen = set()
        unique_candidates = []
        for d, i, j in candidates:
            key = (i, j)
            if key not in seen:
                seen.add(key)
                unique_candidates.append((d, i, j))
        unique_candidates.sort()

        # Greedy matching
        used = set()
        group_id = 0
        for _, i, j in unique_candidates:
            if i in used or j in used:
                continue
            assign[h, i] = group_id
            assign[h, j] = group_id
            used.add(i)
            used.add(j)
            group_id += 1

            if group_id >= K:
                break

        # Assign remaining
        for idx in range(N):
            if idx not in used:
                assign[h, idx] = min(group_id, K - 1)
                group_id += 1

    assign = assign.clamp(0, K - 1)
    centers = _compute_centers(keys, assign, K)
    return assign, centers


def cluster_recursive_proj(keys: torch.Tensor, bf: int):
    """
    Recursive projection bisection: at each level, split along the direction
    of maximum variance. Continue until leaf size ≤ bf.

    This creates a balanced binary tree of tight clusters.
    Cost: O(N * D * log(N/bf)) per head.
    """
    H, N, D = keys.shape
    device = keys.device
    K = max(1, math.ceil(N / bf))

    assign = torch.zeros(H, N, dtype=torch.long, device=device)

    for h in range(H):
        keys_h = keys[h]  # (N, D)
        # Start with all points in one group
        groups = [torch.arange(N, device=device)]
        final_groups = []

        while groups:
            new_groups = []
            for indices in groups:
                if len(indices) <= bf:
                    final_groups.append(indices)
                    continue

                # Split along direction of max variance
                subset = keys_h[indices]
                mu = subset.mean(dim=0, keepdim=True)
                centered = subset - mu

                # Power iteration for top eigenvector (2 iterations suffice)
                v = torch.randn(D, 1, device=device, dtype=keys.dtype)
                for _ in range(3):
                    v = centered.T @ (centered @ v)
                    v = v / v.norm().clamp_min(1e-12)

                proj = (centered @ v).squeeze(-1)
                median = proj.median()

                left_mask = proj <= median
                right_mask = ~left_mask

                # Handle edge case: all on one side
                if left_mask.all() or right_mask.all():
                    half = len(indices) // 2
                    sorted_proj = proj.argsort()
                    left_mask = torch.zeros_like(left_mask)
                    left_mask[sorted_proj[:half]] = True
                    right_mask = ~left_mask

                new_groups.append(indices[left_mask])
                new_groups.append(indices[right_mask])

            groups = new_groups

        # Assign group IDs
        for gid, indices in enumerate(final_groups):
            assign[h, indices] = min(gid, K - 1)

    # Remap to contiguous IDs
    for h in range(H):
        _, inverse = assign[h].unique(sorted=True, return_inverse=True)
        assign[h] = inverse.clamp(max=K - 1)

    centers = _compute_centers(keys, assign, K)
    return assign, centers


def _compute_centers(keys, assign, K):
    H, N, D = keys.shape
    device = keys.device
    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts = torch.zeros(H, K, device=device, dtype=keys.dtype)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device, dtype=keys.dtype))
    centers /= counts.clamp_min(1).unsqueeze(-1)
    return centers
