from abc import ABC
import torch
from typing import Optional, Tuple
from enum import Enum

from .base import BaseIndexer
from hira.kernels.triton_update_kernels import (
    fill_existing_children_atomic_batched,
    update_parent_radii_atomic_batched_masked,
    nearest_l2_triton_batched,
)

DEFAULT_PAD_VALUE = 0.0  # used building index for aligning index to (parent * BF)


class CUDAIndexer(BaseIndexer):

    class DEPTH(Enum):
        # currently only up to three levels on GPU
        TWO_LEVELS = 2
        THREE_LEVELS = 3

    def __init__(
        self,
        num_levels: DEPTH | int,
        max_iterations: int,
        branching_factor: int,
        verbose: bool = False,
        pad_value: float = DEFAULT_PAD_VALUE,
    ):
        self.depth = (
            num_levels
            if type(num_levels) == CUDAIndexer.DEPTH
            else CUDAIndexer.DEPTH(num_levels)
        )
        self.max_iterations = max_iterations
        self.branching_factor = branching_factor
        self.verbose = verbose
        self.pad_value = pad_value  # value used for padding unfilled slots

        # to build
        self.dim: int = 0
        self.children: Optional[torch.Tensor] = None  # padded keys
        self.values: Optional[torch.Tensor] = None  # (H,N,V), aligned with children
        self.parents: Optional[torch.Tensor] = None  # level 1
        self.parent_radii: Optional[torch.Tensor] = None  # level 1
        self.grand_parents: Optional[torch.Tensor] = None  # level 2
        self.grand_parent_radii: Optional[torch.Tensor] = None  # level 2

        # ====== UPDATE_V2 CACHE ======
        # Per-parent count of filled children slots (assumes contiguous fill from slot 0).
        self._child_counts: Optional[torch.Tensor] = None  # (num_parents,) int32 CUDA
        # Valid parent rows (needed for THREE_LEVELS where layout may contain padded parents).
        self._parent_valid: Optional[torch.Tensor] = None  # (num_parents,) bool CUDA

    @torch.no_grad()
    def build(self, keys: torch.Tensor, values: Optional[torch.Tensor] = None):
        # keys: (1, H, L, D)
        # values: optional (1, H, L, D) aligned with keys
        # make sure keys are on GPU
        keys = keys.to("cuda").squeeze(0).contiguous()  # (H, L, D)
        self.num_heads, num_keys, self.dim = keys.shape

        if values is not None:
            values = values.squeeze(0).to("cuda").contiguous()
            assert values.shape == keys.shape, "values must have the same shape as keys"

        if self.depth == CUDAIndexer.DEPTH.TWO_LEVELS:
            (
                self.parents,
                self.children,
                values_layout,
            ) = self._build_parents_children_from_keys(
                keys,
                self.branching_factor,
                values=values,
            )
            self.values = None if values_layout is None else values_layout.contiguous()
            self.parent_radii = self._compute_parent_radii_from_layout()
        elif self.depth == CUDAIndexer.DEPTH.THREE_LEVELS:
            (
                self.grand_parents,
                self.parents,
                self.children,
                values_layout,
            ) = self._build_grandparents_parents_children_from_keys(
                keys,
                self.branching_factor,
                values=values,
            )
            self.values = None if values_layout is None else values_layout.contiguous()
            self.parent_radii = self._compute_parent_radii_from_layout()
            self.grand_parent_radii = self._compute_grandparent_radii_from_layout()
        else:
            raise ValueError(f"Unsupported depth {self.depth}")

        # Initialize update_v2 cache state.
        self._init_update_state()

        return self

    # ------------------------------------------------------------------
    # Build Helpers
    # ------------------------------------------------------------------

    def _compute_parent_radii_from_layout(self) -> torch.Tensor:
        """
        Returns:
        radii: (H, m) float32 CUDA
        """
        if self.parents.ndim != 3:
            raise ValueError(
                f"parents must be (H,m,d), got {tuple(self.parents.shape)}"
            )
        if self.children.ndim != 3:
            raise ValueError(
                f"children must be (H,m*bf,d), got {tuple(self.children.shape)}"
            )

        H, m, d = self.parents.shape
        bf = int(self.branching_factor)

        parents_f = self.parents.float().contiguous()  # (H,m,d)
        children_f = self.children.float().contiguous().view(H, m, bf, d)  # (H,m,bf,d)

        diffs = children_f - parents_f[:, :, None, :]  # (H,m,bf,d)
        dists = torch.linalg.norm(diffs, dim=-1)  # (H,m,bf)

        if self.pad_value is not None:
            pad = float(self.pad_value)
            valid = ~torch.all(children_f == pad, dim=-1)  # (H,m,bf)
            dists = torch.where(
                valid,
                dists,
                torch.tensor(float("-inf"), device=dists.device, dtype=dists.dtype),
            )

        radii = torch.max(dists, dim=2).values  # (H,m)
        radii = torch.where(torch.isfinite(radii), radii, torch.zeros_like(radii))

        assert radii.is_cuda
        return radii

    def _compute_grandparent_radii_from_layout(self) -> torch.Tensor:
        """
        Returns:
            radii : (H, g) float32 CUDA
        """
        if self.grand_parents.ndim != 3:
            raise ValueError(
                f"grand_parents must be (H,g,d), got {tuple(self.grand_parents.shape)}"
            )

        H, g, d = self.grand_parents.shape
        bf = int(self.branching_factor)

        gp_f = self.grand_parents.float().contiguous()  # (H,g,d)
        parents_f = self.parents.float().contiguous().view(H, g, bf, d)  # (H,g,bf,d)

        pr = self.parent_radii.float().contiguous().view(H, g, bf)  # (H,g,bf)

        # Distance between grandparent and each parent
        dists = torch.linalg.norm(parents_f - gp_f[:, :, None, :], dim=-1)  # (H,g,bf)

        totals = dists + pr  # (H,g,bf)

        if self.pad_value is not None:
            pad = float(self.pad_value)
            valid = ~torch.all(parents_f == pad, dim=-1)  # (H,g,bf)

            totals = torch.where(
                valid,
                totals,
                torch.tensor(
                    float("-inf"),
                    device=totals.device,
                    dtype=totals.dtype,
                ),
            )

        radii = torch.max(totals, dim=2).values  # (H,g)

        radii = torch.where(
            torch.isfinite(radii),
            radii,
            torch.zeros_like(radii),
        )

        assert radii.is_cuda
        return radii

    # ------------------------------------------------------------------
    # GPU-native nearest-centroid helpers (BLAS-based, auto-chunked)
    # ------------------------------------------------------------------

    @staticmethod
    def _gpu_nearest(
        x: torch.Tensor,  # (H, n, d)
        C: torch.Tensor,  # (H, K, d)
    ):
        """Return ``(dist1, assign)`` — L2 distance and index of nearest
        centroid for every point.  Automatically chunks the ``(H, n, K)``
        distance matrix when it would exceed ~512 MB."""
        H, n, _ = x.shape
        K = C.shape[1]
        device = x.device
        limit = 512 * 1024 * 1024 // 4  # 512 MB in float32 elements
        if H * n * K <= limit:
            d = torch.cdist(x, C)  # (H, n, K)
            return d.min(dim=2)  # namedtuple (values, indices)
        chunk = max(1, limit // (H * K))
        dists = torch.empty(H, n, device=device, dtype=torch.float32)
        assign = torch.empty(H, n, device=device, dtype=torch.int64)
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            dc = torch.cdist(x[:, i:j], C)
            dists[:, i:j], assign[:, i:j] = dc.min(dim=2)
        return dists, assign

    @staticmethod
    def _gpu_assign_sq(
        x: torch.Tensor,  # (H, n, d)
        C: torch.Tensor,  # (H, K, d)
        x_sq: torch.Tensor,  # (H, n) — precomputed ||x||²
    ) -> torch.Tensor:
        """Assign each point to nearest centroid using BLAS-based squared
        L2 distance.  Returns ``assign`` of shape ``(H, n)``."""
        H, n, _ = x.shape
        K = C.shape[1]
        device = x.device
        limit = 512 * 1024 * 1024 // 4

        c_sq = (C * C).sum(dim=-1)  # (H, K)
        CT = C.transpose(1, 2)  # (H, d, K)

        if H * n * K <= limit:
            dots = torch.bmm(x, CT)  # (H, n, K)
            d2 = x_sq.unsqueeze(2) - 2 * dots + c_sq.unsqueeze(1)
            return d2.argmin(dim=2)

        chunk = max(1, limit // (H * K))
        assign = torch.empty(H, n, device=device, dtype=torch.int64)
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            dots_c = torch.bmm(x[:, i:j], CT)
            d2_c = x_sq[:, i:j].unsqueeze(2) - 2 * dots_c + c_sq.unsqueeze(1)
            assign[:, i:j] = d2_c.argmin(dim=2)
        return assign

    # ------------------------------------------------------------------
    # One-level builder (pure GPU)
    # ------------------------------------------------------------------

    def _build_random_bf_level_batched(
        self,
        x: torch.Tensor,  # (H, n, d) float32 CUDA
        bf: int,
        *,
        seed: int = 1234,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU-native one-level builder using random init + Lloyd's K-means
        with empty-cluster recovery.  Everything stays on GPU.

        Steps
        -----
        1) **Random init** — select K distinct points per head.
        2) **Lloyd's refinement** with BLAS-based squared-distance
           (``bmm``) and empty-cluster splitting (re-seed dead centroids
           with the point farthest from its own centroid).
        3) **bf-bounded composite-sort selection** — keep the ``bf``
           closest points per centroid.
        4) **Overflow placement** — leftovers go to nearest centroid with
           remaining room.
        5) **Centroid recomputation** — mean of final children.

        Returns
        -------
        C        : (H, K, d)  float32 centroids on CUDA.
        selected : (H, K, bf) int64 indices into ``x[h]``, or -1 if padded.
        """
        assert x.is_cuda and x.dtype == torch.float32 and x.ndim == 3
        device = x.device
        H, n, d = x.shape
        assert bf >= 1
        K = max(1, (n + bf - 1) // bf)
        niter = max(int(self.max_iterations), 1)

        # ==============================================================
        # 1)  Random initialisation  (GPU, O(n) — no sequential loop)
        # ==============================================================
        gen = torch.Generator(device=device).manual_seed(seed)
        perm = torch.argsort(torch.rand(H, n, device=device, generator=gen), dim=1)
        C = x.gather(
            1, perm[:, :K].unsqueeze(-1).expand(-1, -1, d)
        ).clone()  # (H, K, d)

        # ==============================================================
        # 2)  Lloyd's refinement  (BLAS-based, GPU)
        # ==============================================================
        x_sq = (x * x).sum(dim=-1)  # (H, n)

        for _ in range(niter):
            assign = CUDAIndexer._gpu_assign_sq(x, C, x_sq)  # (H, n)

            idx_e = assign.unsqueeze(-1).expand(H, n, d)
            sums = torch.zeros(H, K, d, device=device)
            sums.scatter_add_(1, idx_e, x)
            cnt = torch.zeros(H, K, device=device)
            cnt.scatter_add_(1, assign, torch.ones(H, n, device=device))

            # --- empty-cluster recovery ---
            empty = cnt == 0
            if empty.any():
                # distance of each point to its assigned centroid
                C_asgn = C.gather(
                    1, assign.unsqueeze(-1).expand(-1, -1, d)
                )  # (H, n, d)
                own_d = ((x - C_asgn) ** 2).sum(dim=-1)  # (H, n)
                for h in range(H):
                    ek = empty[h].nonzero(as_tuple=False).view(-1)
                    if ek.numel() == 0:
                        continue
                    _, far = own_d[h].topk(ek.numel())
                    sums[h, ek] = x[h, far]
                    cnt[h, ek] = 1

            ne = cnt > 0
            sums[ne] /= cnt[ne].unsqueeze(-1)
            sums[~ne] = C[~ne]
            C = sums

        # ==============================================================
        # 3)  bf-bounded composite-sort selection  (GPU)
        # ==============================================================
        dist1, idx1 = CUDAIndexer._gpu_nearest(x, C)

        # Global distance rank per head
        dist_order = dist1.argsort(dim=1)
        dist_rank = torch.empty_like(dist_order)
        n_range = torch.arange(n, device=device).view(1, n).expand(H, -1)
        dist_rank.scatter_(1, dist_order, n_range)

        # Composite key: (centroid, dist-rank) → groups by centroid,
        # closest-first within group.
        composite = idx1.to(torch.int64) * (n + 1) + dist_rank.to(torch.int64)
        order = composite.argsort(dim=1)
        idx_grp = idx1.gather(1, order)  # centroid id in grouped order
        pts_grp = order  # original point indices

        counts = torch.zeros(H, K, device=device, dtype=torch.int64)
        counts.scatter_add_(1, idx1, torch.ones(H, n, device=device, dtype=torch.int64))
        offsets = torch.zeros(H, K + 1, device=device, dtype=torch.int64)
        offsets[:, 1:] = counts.cumsum(dim=1)

        pos = torch.arange(n, device=device, dtype=torch.int64).view(1, n).expand(H, -1)
        start = offsets[:, :-1].gather(1, idx_grp)
        local_rank = pos - start

        mask = local_rank < bf
        selected = torch.full((H, K, bf), -1, device=device, dtype=torch.int64)

        h_idx = (
            torch.arange(H, device=device, dtype=torch.int64).view(H, 1).expand(H, n)
        )
        c_idx = idx_grp.to(torch.int64)
        r_idx = local_rank.to(torch.int64)
        h_m, c_m, r_m = h_idx[mask], c_idx[mask], r_idx[mask]
        p_m = pts_grp[mask].to(torch.int64)
        flat = h_m * (K * bf) + c_m * bf + r_m
        selected.view(-1).scatter_(0, flat, p_m)

        # ==============================================================
        # 4)  Overflow — assign leftovers to nearest centroid w/ room
        # ==============================================================
        if (selected == -1).any():
            for h in range(H):
                sel_h = selected[h]  # (K, bf)
                empty_mask = sel_h == -1
                if not empty_mask.any():
                    continue

                used = torch.zeros(n, device=device, dtype=torch.bool)
                taken = sel_h[sel_h != -1]
                if taken.numel():
                    used[taken] = True
                leftovers = (~used).nonzero(as_tuple=False).view(-1)
                if leftovers.numel() == 0:
                    continue

                room = empty_mask.sum(dim=1)  # (K,)
                has_room = (room > 0).nonzero(as_tuple=False).view(-1)
                if has_room.numel() == 0:
                    continue

                X_left = x[h, leftovers]  # (L, d)
                C_room = C[h, has_room]  # (R, d)
                d_lr = torch.cdist(X_left.unsqueeze(0), C_room.unsqueeze(0)).squeeze(
                    0
                )  # (L, R)

                pt_order = d_lr.min(dim=1).values.argsort()
                cap = room[has_room].clone()
                fp = (bf - room[has_room]).clone()  # fill-pointer

                for li in pt_order:
                    li_v = li.item()
                    pt = leftovers[li_v].item()
                    for ri in d_lr[li_v].argsort():
                        ri_v = ri.item()
                        if cap[ri_v] > 0:
                            c = has_room[ri_v].item()
                            sel_h[c, fp[ri_v].item()] = pt
                            fp[ri_v] += 1
                            cap[ri_v] -= 1
                            break
                selected[h] = sel_h

        # ==============================================================
        # 5)  Recompute centroids from final children (vectorised)
        # ==============================================================
        safe = selected.clamp_min(0)  # (H, K, bf)
        g = x.gather(1, safe.view(H, K * bf).unsqueeze(-1).expand(-1, -1, d)).view(
            H, K, bf, d
        )
        valid = (selected != -1).unsqueeze(-1)  # (H, K, bf, 1)
        g = g * valid.float()
        sums = g.sum(dim=2)  # (H, K, d)
        cnt = valid.squeeze(-1).sum(dim=2, keepdim=True).clamp_min(1).float()
        C_new = sums / cnt
        has_any = (selected != -1).any(dim=2)  # (H, K)
        C[has_any] = C_new[has_any]

        return C, selected

    def _build_parents_children_from_keys(
        self,
        keys: torch.Tensor,  # (H, n, d)
        bf: int,
        values: Optional[torch.Tensor] = None,  # (H, n, v)
        *,
        seed: int = 1234,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Build one level from raw keys using ONLY _build_faiss_kmeans_bf_level.

        Returns:
        parents  : (H, m, d) float32 CUDA, where m = max(1, ceil(n / bf))
        children : (H, m*bf, d) float32 CUDA, where children[i*bf:(i+1)*bf] are parent i's children.
                    If a slot is unfilled (selected == -1), it is padded with pad_value.

        Ordering:
        - For each parent i, children[i*bf:(i+1)*bf] follow the selection order from _build_faiss_kmeans_bf_level
            (centroid-grouped, closest-first within group).
        """
        if keys.ndim != 3:
            raise ValueError(f"keys must be (H,n,d), got {tuple(keys.shape)}")
        if not keys.is_cuda:
            raise ValueError("keys must be on CUDA")
        if bf <= 0:
            raise ValueError("bf must be positive")

        x = keys.detach()
        if x.dtype != torch.float32:
            x = x.float()
        x = x.contiguous()

        device = x.device
        H, n, d = x.shape
        if values is not None:
            assert values.ndim == 3
            assert (
                values.shape[0] == H and values.shape[1] == n
            ), "values must be (H,n,v)"
            values = values.contiguous()
            vdim = int(values.shape[-1])
        else:
            vdim = 0

        # Build one level (batched over heads)
        parents, selected = self._build_random_bf_level_batched(x, bf, seed=seed)
        # parents : (H, m, d)  where m = max(1, ceil(n/bf))
        # selected: (H, m, bf) indices into x[h] or -1

        # Derive m from the actual output shape (ceil-division, not floor).
        m = parents.shape[1]

        # Materialize children: (H, m*bf, d)
        children = torch.full(
            (H, m * bf, d),
            float(self.pad_value),
            device=device,
            dtype=torch.float32,
        )
        children_values = (
            torch.zeros((H, m * bf, vdim), device=device, dtype=values.dtype)
            if values is not None
            else None
        )

        sel_flat = selected.reshape(H, m * bf)  # (H, m*bf)
        valid = sel_flat != -1

        if valid.any():
            # For invalid positions, set index to 0 so gather is safe; we'll mask them out anyway.
            safe_idx = sel_flat.clamp_min(0)  # (H, m*bf)

            # Gather chosen children from x: (H, m*bf, d)
            gathered = x.gather(
                1,
                safe_idx.unsqueeze(-1).expand(-1, -1, d),
            )

            # Write only valid slots; keep pad_value in invalid slots
            children[valid] = gathered[valid]
            if children_values is not None:
                gathered_values = values.gather(
                    1,
                    safe_idx.unsqueeze(-1).expand(-1, -1, vdim),
                )
                children_values[valid] = gathered_values[valid]

        assert parents.is_cuda and children.is_cuda
        return parents, children, children_values

    def _build_grandparents_parents_children_from_keys(
        self,
        keys: torch.Tensor,  # (H,n,d) CUDA float/half ok
        bf: int,
        values: Optional[torch.Tensor] = None,  # (H,n,v)
        *,
        seed: int = 1234,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Batched (over heads) two-level build.

        Returns:
        grand_parents      : (H, g, d) float32 CUDA, g = max(1, m // bf) where m=max(1, n // bf)
        parents_reordered  : (H, g*bf, d) float32 CUDA, contiguous bf-block per grandparent
        children_reordered : (H, g*bf*bf, d) float32 CUDA, contiguous bf-block per parent (in reordered parent order)

        Invariants (per head h):
        - parents_reordered[h, i*bf:(i+1)*bf] are the parents of grand_parent[h, i] (padding possible)
        - children_reordered[h, p*bf:(p+1)*bf] are children of parent p in parents_reordered[h]
        """
        assert keys.ndim == 3
        assert keys.is_cuda
        assert bf >= 1

        # --- first level: keys -> parents, children ---
        parents, children, children_values = self._build_parents_children_from_keys(
            keys,
            bf,
            values=values,
            seed=seed,
        )
        # parents:  (H, m, d)
        # children: (H, m*bf, d)
        device = parents.device
        H, m, d = parents.shape

        # --- second level: parents -> grand_parents, select parent-ids per grandparent ---
        gp, sel_par = self._build_random_bf_level_batched(
            parents.contiguous(), bf, seed=seed + 1
        )
        # gp:      (H, g, d)
        # sel_par: (H, g, bf)
        _, g, _ = gp.shape

        # Flatten selection: (H, g*bf)
        sel_flat = sel_par.reshape(H, g * bf)
        valid = sel_flat != -1  # (H, g*bf)

        # ----------------------------
        # Reorder parents into contiguous bf-blocks per grandparent
        # ----------------------------
        parents_reordered = torch.full(
            (H, g * bf, d),
            float(self.pad_value),
            device=device,
            dtype=torch.float32,
        )

        if valid.any():
            safe_idx = sel_flat.clamp_min(0)  # make gather safe
            gathered_parents = parents.gather(
                1,
                safe_idx.unsqueeze(-1).expand(-1, -1, d),
            )  # (H, g*bf, d)
            parents_reordered[valid] = gathered_parents[valid]

        # ----------------------------
        # Reorder children by permuting whole parent child-blocks
        # children aligned with original parents: children.view(H, m, bf, d)
        # ----------------------------
        child_blocks = children.view(H, m, bf, d)  # (H, m, bf, d)

        children_reordered_blocks = torch.full(
            (H, g * bf, bf, d),
            float(self.pad_value),
            device=device,
            dtype=torch.float32,
        )

        if valid.any():
            safe_idx = sel_flat.clamp_min(0)  # (H, g*bf)
            gathered_blocks = child_blocks.gather(
                1,
                safe_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, bf, d),
            )  # (H, g*bf, bf, d)
            # mask-fill only valid parent slots
            children_reordered_blocks[valid] = gathered_blocks[valid]

        children_reordered = children_reordered_blocks.view(H, g * bf * bf, d)
        children_values_reordered = None
        if children_values is not None:
            vdim = int(children_values.shape[-1])
            child_value_blocks = children_values.view(H, m, bf, vdim)
            children_values_reordered_blocks = torch.zeros(
                (H, g * bf, bf, vdim),
                device=device,
                dtype=children_values.dtype,
            )
            if valid.any():
                safe_idx = sel_flat.clamp_min(0)  # (H, g*bf)
                gathered_value_blocks = child_value_blocks.gather(
                    1,
                    safe_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, bf, vdim),
                )  # (H, g*bf, bf, vdim)
                children_values_reordered_blocks[valid] = gathered_value_blocks[valid]
            children_values_reordered = children_values_reordered_blocks.view(
                H, g * bf * bf, vdim
            )

        assert gp.is_cuda and parents_reordered.is_cuda and children_reordered.is_cuda
        return gp, parents_reordered, children_reordered, children_values_reordered

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, new_keys: torch.Tensor, new_values: Optional[torch.Tensor] = None):
        """Incremental update of the index with new key vectors.

        Args:
            new_keys: (H, M, D) new key vectors per head.
            new_values: optional (1, H, M, V) value vectors per head.

        Algorithm:
            1. Find nearest parent for each new key; fill if the parent
               has a free children slot.
            2. Orphan keys (parent full) -> randomly select O//bf new parents
               from orphans, assign orphans to nearest new parent, build
               children blocks. Append new parents + children.
            3. THREE_LEVELS only: repeat the same pattern for the newly
               created parents at the grandparent level, then append new
               grandparent blocks for any overflow.
            4. Update all radii and refresh internal state.
        """
        assert new_values is None or new_keys.shape == new_values.shape

        new_keys = new_keys.contiguous()
        H, M, D = new_keys.shape
        assert D == self.dim

        has_values = self.values is not None
        if has_values:
            new_values = new_values.contiguous()

        bf = int(self.branching_factor)
        device = new_keys.device

        self._ensure_update_state()

        # ==============================================================
        # Phase 1 - place new keys into existing parent children blocks
        # ==============================================================
        nearest_parent, _ = nearest_l2_triton_batched(
            new_keys, self.parents, valid_mask=self._parent_valid
        )
        nearest_parent = nearest_parent.to(torch.int32)

        placed_mask, placed_flat = fill_existing_children_atomic_batched(
            x=new_keys,
            parent_idx=nearest_parent,
            child_counts=self._child_counts,
            children=self.children,
            bf=bf,
        )
        if has_values and placed_mask.any():
            h_idx = torch.arange(H, device=device, dtype=torch.long)[:, None].expand(
                H, M
            )
            placed_rows = placed_flat[placed_mask].to(torch.long)
            values_3d = self.values
            values_3d[h_idx[placed_mask], placed_rows] = new_values[placed_mask]

        # Update parent radii for placed keys (atomic max).
        if placed_mask.any() and self.parent_radii.dtype == torch.float32:
            update_parent_radii_atomic_batched_masked(
                new_keys,
                nearest_parent,
                placed_mask.to(torch.uint8),
                self.parents,
                self.parent_radii,
            )

        # ==============================================================
        # Phase 2 - build new parents from orphan keys
        # ==============================================================
        overflow_mask = ~placed_mask  # (H, M)
        orphan_counts = overflow_mask.sum(dim=1)  # (H,)
        total_orphans = orphan_counts.sum().item()

        if total_orphans == 0:
            if self.depth == CUDAIndexer.DEPTH.THREE_LEVELS and placed_mask.any():
                self._update_gp_radii_for_placed_children(placed_mask, nearest_parent)
            self._refresh_after_update()
            return self

        (
            new_parents,
            new_children_flat,
            new_parent_radii,
            new_children_values_flat,
        ) = self._build_level_from_orphans(
            all_items=new_keys,
            all_item_values=new_values,
            overflow_mask=overflow_mask,
            orphan_counts=orphan_counts,
        )
        # new_parents:       (H, K_max, D)
        # new_children_flat: (H, K_max * bf, D)
        # new_children_values_flat: (H, K_max * bf, V) or None
        # new_parent_radii:  (H, K_max)

        # ==============================================================
        # Phase 3 - incorporate new parents
        # ==============================================================
        if self.depth == CUDAIndexer.DEPTH.TWO_LEVELS:
            self.parents = torch.cat([self.parents, new_parents], dim=1).contiguous()
            self.parent_radii = torch.cat(
                [self.parent_radii, new_parent_radii], dim=1
            ).contiguous()
            self.children = torch.cat(
                [self.children, new_children_flat], dim=1
            ).contiguous()
            if has_values:
                self.values = torch.cat(
                    [self.values, new_children_values_flat], dim=-2
                ).contiguous()
            self._refresh_after_update()
            return self

        # ---------- THREE_LEVELS ----------
        assert self.grand_parents is not None and self.grand_parent_radii is not None
        H, G_old, _ = self.grand_parents.shape
        P_old = self.parents.shape[1]
        assert P_old == G_old * bf

        # Update GP radii for keys placed in Phase 1.
        if placed_mask.any():
            self._update_gp_radii_for_placed_children(placed_mask, nearest_parent)

        # Try to place new parents into existing GP blocks.
        pad = float(self.pad_value)
        new_parent_valid = ~torch.all(new_parents == pad, dim=-1)  # (H, K_max)
        K_max = new_parents.shape[1]

        parent_placed_mask = torch.zeros((H, K_max), device=device, dtype=torch.bool)

        if new_parent_valid.any():
            self._place_parents_into_gp_blocks(
                new_parents,
                new_parent_radii,
                new_children_flat,
                new_children_values_flat,
                new_parent_valid,
                parent_placed_mask,
            )

        # Orphan parents -> create new grandparent blocks.
        parent_overflow_mask = new_parent_valid & (~parent_placed_mask)
        parent_orphan_counts = parent_overflow_mask.sum(dim=1)

        if parent_orphan_counts.sum().item() > 0:
            self._grow_three_levels_from_orphan_parents(
                new_parents,
                new_parent_radii,
                new_children_flat,
                new_children_values_flat,
                parent_overflow_mask,
                parent_orphan_counts,
            )

        self._refresh_after_update()
        return self

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------

    def _build_level_from_orphans(
        self,
        all_items: torch.Tensor,  # (H, M, D)
        all_item_values: Optional[torch.Tensor],  # (H, M, V)
        overflow_mask: torch.Tensor,  # (H, M) bool
        orphan_counts: torch.Tensor,  # (H,)
    ):
        """Build new parents + children blocks from orphan keys.

        Returns (all padded to max K across heads):
            new_parents       : (H, K_max, D)
            new_children_flat : (H, K_max * bf, D)
            new_children_values_flat : (H, K_max * bf, V) or None
            new_parent_radii  : (H, K_max)
        """
        H, M, D = all_items.shape
        bf = int(self.branching_factor)
        device = all_items.device
        pad = float(self.pad_value)
        has_values = all_item_values is not None
        vdim = int(all_item_values.shape[-1]) if has_values else 0

        # Ceiling division so K_h * bf >= O_h -- every orphan key gets a slot.
        K_per_head = torch.where(
            orphan_counts > 0,
            torch.clamp((orphan_counts + bf - 1) // bf, min=1),
            torch.zeros_like(orphan_counts),
        )
        K_max = K_per_head.max().item()

        new_parents = torch.full((H, K_max, D), pad, device=device, dtype=torch.float32)
        new_children_flat = torch.full(
            (H, K_max * bf, D), pad, device=device, dtype=torch.float32
        )
        new_children_values_flat = (
            torch.zeros(
                (H, K_max * bf, vdim),
                device=device,
                dtype=all_item_values.dtype,
            )
            if has_values
            else None
        )
        new_parent_radii = torch.zeros((H, K_max), device=device, dtype=torch.float32)

        for h in range(H):
            O_h = orphan_counts[h].item()
            if O_h == 0:
                continue
            K_h = K_per_head[h].item()

            orphans_h = all_items[h][overflow_mask[h]]  # (O_h, D)
            orphan_values_h = (
                all_item_values[h][overflow_mask[h]] if has_values else None
            )

            parents_h, children_h, radii_h, children_values_h = (
                self._build_one_level_from_points(
                    orphans_h,
                    K_h,
                    bf,
                    pad,
                    device,
                    D,
                    point_values=orphan_values_h,
                )
            )

            new_parents[h, :K_h] = parents_h
            new_children_flat[h, : K_h * bf] = children_h
            if has_values:
                new_children_values_flat[h, : K_h * bf] = children_values_h
            new_parent_radii[h, :K_h] = radii_h

        return (
            new_parents,
            new_children_flat,
            new_parent_radii,
            new_children_values_flat,
        )

    @staticmethod
    def _build_one_level_from_points(
        points: torch.Tensor,  # (N, D)
        K: int,  # number of parents to create
        bf: int,
        pad: float,
        device: torch.device,
        D: int,
        point_values: Optional[torch.Tensor] = None,  # (N, V)
    ):
        """Build K parents and K*bf children from N points (single head).

        Returns:
            parents  : (K, D)
            children : (K*bf, D), padded with *pad*
            radii    : (K,) float32
            children_values : (K*bf, V) or None
        """
        N = points.shape[0]
        has_values = point_values is not None
        if has_values:
            if point_values.ndim != 2 or point_values.shape[0] != N:
                raise ValueError(
                    f"point_values must be (N,V), got {tuple(point_values.shape)}"
                )
            point_values = point_values.contiguous()
            V = int(point_values.shape[1])
        else:
            V = 0

        # Random parent selection
        perm = torch.randperm(N, device=device)
        parents = points[perm[:K]].contiguous()  # (K, D)

        # Assign every point to nearest parent
        dist = torch.cdist(points.unsqueeze(0), parents.unsqueeze(0)).squeeze(
            0
        )  # (N, K)
        assign = dist.argmin(dim=1)  # (N,)
        dist_assigned = dist.gather(1, assign.unsqueeze(1)).squeeze(1)  # (N,)

        # Group by (parent, distance) - closest first
        dist_order = torch.argsort(dist_assigned)
        dist_rank = torch.empty_like(dist_order)
        dist_rank[dist_order] = torch.arange(N, device=device, dtype=dist_order.dtype)

        composite = assign.to(torch.int64) * (N + 1) + dist_rank.to(torch.int64)
        order = torch.argsort(composite)

        idx_sorted = assign[order]
        pts_sorted = points[order]
        vals_sorted = point_values[order] if has_values else None

        counts = torch.bincount(idx_sorted, minlength=K).to(torch.int64)
        offsets = torch.zeros(K + 1, device=device, dtype=torch.int64)
        offsets[1:] = torch.cumsum(counts, dim=0)

        pos = torch.arange(N, device=device, dtype=torch.int64)
        start_pos = offsets[:-1].gather(0, idx_sorted)
        local_rank = pos - start_pos

        placed = local_rank < bf
        children = torch.full((K * bf, D), pad, device=device, dtype=torch.float32)
        children_values = (
            torch.zeros((K * bf, V), device=device, dtype=point_values.dtype)
            if has_values
            else None
        )

        if placed.any():
            dst_idx = idx_sorted[placed] * bf + local_rank[placed]
            children.index_copy_(0, dst_idx, pts_sorted[placed])
            if has_values:
                children_values.index_copy_(0, dst_idx, vals_sorted[placed])

        # Leftovers (overflow beyond bf per centroid) go into any remaining
        # free slots.  With ceiling-division K, K*bf >= N guarantees
        # free.numel() >= leftovers.shape[0] always holds; the min() guards
        # against any unexpected edge case without silently dropping keys.
        if not placed.all():
            leftovers = pts_sorted[~placed]
            empty_slots = torch.all(children == pad, dim=-1)
            free = torch.nonzero(empty_slots, as_tuple=False).view(-1)
            n_fill = min(free.numel(), leftovers.shape[0])
            if n_fill > 0:
                children[free[:n_fill]] = leftovers[:n_fill]
                if has_values:
                    children_values[free[:n_fill]] = vals_sorted[~placed][:n_fill]

        # Compute parent radii
        c_view = children.view(K, bf, D)
        valid = ~torch.all(c_view == pad, dim=-1)
        dists_r = torch.linalg.norm((c_view - parents[:, None, :]).float(), dim=-1)
        dists_r = torch.where(
            valid,
            dists_r,
            torch.tensor(float("-inf"), device=device),
        )
        radii = torch.max(dists_r, dim=1).values
        radii = torch.where(torch.isfinite(radii), radii, torch.zeros_like(radii))

        return parents, children, radii, children_values

    def _place_parents_into_gp_blocks(
        self,
        new_parents: torch.Tensor,  # (H, K, D)
        new_parent_radii: torch.Tensor,  # (H, K)
        new_children_flat: torch.Tensor,  # (H, K*bf, D)
        new_children_values_flat: Optional[torch.Tensor],  # (H, K*bf, V)
        new_parent_valid: torch.Tensor,  # (H, K) bool
        parent_placed_mask: torch.Tensor,  # (H, K) bool - updated in-place
    ):
        """Try to place new parents into empty slots of existing GP blocks.

        Modifies self.parents, self.parent_radii, self.children,
        self.grand_parent_radii, and *parent_placed_mask* in place.
        """
        bf = int(self.branching_factor)
        device = self.parents.device
        H, G_old, D = self.grand_parents.shape
        K = new_parents.shape[1]
        if (self.values is None) != (new_children_values_flat is None):
            raise ValueError(
                "Value tensors must be present for both source and destination"
            )
        has_values = new_children_values_flat is not None
        V = int(new_children_values_flat.shape[-1]) if has_values else 0
        dst_values = self.values if has_values else None

        # Find nearest GP for each new parent.
        nearest_gp, _ = nearest_l2_triton_batched(
            new_parents, self.grand_parents, valid_mask=None
        )
        nearest_gp = nearest_gp.to(torch.int64)

        # Current fill per GP: (H, G_old)
        gp_counts = self._gp_child_counts.to(torch.int64)

        for h in range(H):
            valid_k = torch.nonzero(new_parent_valid[h], as_tuple=False).view(-1)
            if valid_k.numel() == 0:
                continue
            n_valid = valid_k.numel()
            gp_assign = nearest_gp[h, valid_k]  # (n_valid,)

            avail = (bf - gp_counts[h]).clamp_min(0)  # (G_old,)

            # Rank within GP groups
            order = torch.argsort(gp_assign)
            gp_sorted = gp_assign[order]

            counts = torch.zeros(G_old, device=device, dtype=torch.int64)
            counts.scatter_add_(
                0,
                gp_sorted,
                torch.ones(n_valid, device=device, dtype=torch.int64),
            )
            offsets = torch.zeros(G_old + 1, device=device, dtype=torch.int64)
            offsets[1:] = torch.cumsum(counts, dim=0)

            pos = torch.arange(n_valid, device=device, dtype=torch.int64)
            start_pos = offsets[:-1].gather(0, gp_sorted)
            rank = pos - start_pos

            avail_sorted = avail.gather(0, gp_sorted)
            can_place = rank < avail_sorted

            if not can_place.any():
                continue

            placed_o = order[can_place]
            placed_k = valid_k[placed_o]
            placed_gp = gp_sorted[can_place]
            placed_slot = gp_counts[h].gather(0, placed_gp) + rank[can_place]
            dst_p = placed_gp * bf + placed_slot  # position in parents array

            # Write parent vectors + radii
            self.parents[h, dst_p] = new_parents[h, placed_k]
            self.parent_radii[h, dst_p] = new_parent_radii[h, placed_k].to(
                self.parent_radii.dtype
            )

            # Write children blocks
            P_old = self.parents.shape[1]
            src_c = new_children_flat[h].view(K, bf, D)
            dst_c = self.children[h].view(P_old, bf, D)
            dst_c[dst_p] = src_c[placed_k]
            if has_values:
                src_v = new_children_values_flat[h].view(K, bf, V)
                dst_v = dst_values[h].view(P_old, bf, V)
                dst_v[dst_p] = src_v[placed_k]

            parent_placed_mask[h, placed_k] = True

            # Update GP radii (monotonic increase)
            gp_centers = self.grand_parents[h, placed_gp].float()
            par_vecs = self.parents[h, dst_p].float()
            dist_gp = torch.linalg.norm(par_vecs - gp_centers, dim=1)
            total = dist_gp + self.parent_radii[h, dst_p].float()

            upd = torch.full(
                (G_old,), float("-inf"), device=device, dtype=torch.float32
            )
            upd.scatter_reduce_(0, placed_gp, total, reduce="amax", include_self=True)
            upd = torch.where(torch.isfinite(upd), upd, torch.zeros_like(upd))
            self.grand_parent_radii[h] = torch.maximum(
                self.grand_parent_radii[h].float(), upd
            ).to(self.grand_parent_radii.dtype)

    def _grow_three_levels_from_orphan_parents(
        self,
        new_parents: torch.Tensor,  # (H, K, D)
        new_parent_radii: torch.Tensor,  # (H, K)
        new_children_flat: torch.Tensor,  # (H, K*bf, D)
        new_children_values_flat: Optional[torch.Tensor],  # (H, K*bf, V)
        parent_overflow_mask: torch.Tensor,  # (H, K) bool
        parent_orphan_counts: torch.Tensor,  # (H,)
    ):
        """Create new GP blocks from orphan parents and append to index."""
        bf = int(self.branching_factor)
        pad = float(self.pad_value)
        device = self.parents.device
        H = new_parents.shape[0]
        D = new_parents.shape[2]
        K = new_parents.shape[1]
        if (self.values is None) != (new_children_values_flat is None):
            raise ValueError(
                "Value tensors must be present for both source and destination"
            )
        has_values = new_children_values_flat is not None
        V = int(new_children_values_flat.shape[-1]) if has_values else 0

        # Ceiling division so K_gp * bf >= n_orphan_parents -- none dropped.
        K_gp_per_head = torch.where(
            parent_orphan_counts > 0,
            torch.clamp((parent_orphan_counts + bf - 1) // bf, min=1),
            torch.zeros_like(parent_orphan_counts),
        )
        K_gp_max = K_gp_per_head.max().item()

        new_gps = torch.full((H, K_gp_max, D), pad, device=device, dtype=torch.float32)
        new_parents_block = torch.full(
            (H, K_gp_max * bf, D), pad, device=device, dtype=torch.float32
        )
        new_pr_block = torch.zeros(
            (H, K_gp_max * bf), device=device, dtype=torch.float32
        )
        new_children_block = torch.full(
            (H, K_gp_max * bf * bf, D),
            pad,
            device=device,
            dtype=torch.float32,
        )
        new_children_values_block = (
            torch.zeros(
                (H, K_gp_max * bf * bf, V),
                device=device,
                dtype=new_children_values_flat.dtype,
            )
            if has_values
            else None
        )
        new_gp_radii = torch.zeros((H, K_gp_max), device=device, dtype=torch.float32)

        for h in range(H):
            n_orphan = parent_orphan_counts[h].item()
            if n_orphan == 0:
                continue
            K_gp = K_gp_per_head[h].item()

            orphan_mask_h = parent_overflow_mask[h]
            orphan_parents_h = new_parents[h][orphan_mask_h]  # (n_orphan, D)
            orphan_radii_h = new_parent_radii[h][orphan_mask_h]  # (n_orphan,)
            orphan_children_h = new_children_flat[h].view(K, bf, D)[
                orphan_mask_h
            ]  # (n_orphan, bf, D)
            orphan_children_values_h = (
                new_children_values_flat[h].view(K, bf, V)[orphan_mask_h]
                if has_values
                else None
            )  # (n_orphan, bf, V)

            # Random GP selection from orphan parents
            perm = torch.randperm(n_orphan, device=device)
            gps_h = orphan_parents_h[perm[:K_gp]]  # (K_gp, D)

            # Assign orphan parents to nearest GP
            dist = torch.cdist(
                orphan_parents_h.unsqueeze(0), gps_h.unsqueeze(0)
            ).squeeze(0)
            gp_assign = dist.argmin(dim=1)
            dist_assigned = dist.gather(1, gp_assign.unsqueeze(1)).squeeze(1)

            # Group by (GP, distance)
            dist_order = torch.argsort(dist_assigned)
            dist_rank = torch.empty_like(dist_order)
            dist_rank[dist_order] = torch.arange(
                n_orphan, device=device, dtype=dist_order.dtype
            )
            composite = gp_assign.to(torch.int64) * (n_orphan + 1) + dist_rank.to(
                torch.int64
            )
            order = torch.argsort(composite)

            gp_sorted = gp_assign[order]
            parents_sorted = orphan_parents_h[order]
            radii_sorted = orphan_radii_h[order]
            children_sorted = orphan_children_h[order]  # (n_orphan, bf, D)
            children_values_sorted = (
                orphan_children_values_h[order] if has_values else None
            )  # (n_orphan, bf, V)

            counts = torch.bincount(gp_sorted, minlength=K_gp).to(torch.int64)
            offsets = torch.zeros(K_gp + 1, device=device, dtype=torch.int64)
            offsets[1:] = torch.cumsum(counts, dim=0)

            pos = torch.arange(n_orphan, device=device, dtype=torch.int64)
            start_pos = offsets[:-1].gather(0, gp_sorted)
            local_rank = pos - start_pos
            placed_gp = local_rank < bf

            parents_blk = torch.full(
                (K_gp * bf, D), pad, device=device, dtype=torch.float32
            )
            radii_blk = torch.zeros(K_gp * bf, device=device, dtype=torch.float32)
            children_blk = torch.full(
                (K_gp * bf * bf, D),
                pad,
                device=device,
                dtype=torch.float32,
            )
            children_blk_view = children_blk.view(K_gp * bf, bf, D)
            children_values_blk = (
                torch.zeros(
                    (K_gp * bf, bf, V),
                    device=device,
                    dtype=children_values_sorted.dtype,
                )
                if has_values
                else None
            )

            if placed_gp.any():
                dst_p = gp_sorted[placed_gp] * bf + local_rank[placed_gp]
                parents_blk[dst_p] = parents_sorted[placed_gp]
                radii_blk[dst_p] = radii_sorted[placed_gp]
                children_blk_view[dst_p] = children_sorted[placed_gp]
                if has_values:
                    children_values_blk[dst_p] = children_values_sorted[placed_gp]

            # Leftovers: orphan parents that overflowed their assigned GP.
            # With ceiling-division K_gp, free slots >= leftovers always.
            if not placed_gp.all():
                left_p = parents_sorted[~placed_gp]
                left_r = radii_sorted[~placed_gp]
                left_c = children_sorted[~placed_gp]
                left_v = children_values_sorted[~placed_gp] if has_values else None

                empty = torch.all(parents_blk == pad, dim=-1)
                free = torch.nonzero(empty, as_tuple=False).view(-1)
                n_fill = min(free.numel(), left_p.shape[0])
                if n_fill > 0:
                    parents_blk[free[:n_fill]] = left_p[:n_fill]
                    radii_blk[free[:n_fill]] = left_r[:n_fill]
                    children_blk_view[free[:n_fill]] = left_c[:n_fill]
                    if has_values:
                        children_values_blk[free[:n_fill]] = left_v[:n_fill]

            # GP radii
            gp_f = gps_h.float()
            pv = parents_blk.float().view(K_gp, bf, D)
            rv = radii_blk.float().view(K_gp, bf)
            valid_p = ~torch.all(pv == pad, dim=-1)
            dists_gp = torch.linalg.norm(pv - gp_f[:, None, :], dim=-1)
            totals = dists_gp + rv
            totals = torch.where(
                valid_p,
                totals,
                torch.tensor(float("-inf"), device=device),
            )
            gpr = torch.max(totals, dim=1).values
            gpr = torch.where(torch.isfinite(gpr), gpr, torch.zeros_like(gpr))

            new_gps[h, :K_gp] = gps_h
            new_gp_radii[h, :K_gp] = gpr
            new_parents_block[h, : K_gp * bf] = parents_blk
            new_pr_block[h, : K_gp * bf] = radii_blk
            new_children_block[h, : K_gp * bf * bf] = children_blk
            if has_values:
                new_children_values_block[h, : K_gp * bf * bf] = (
                    children_values_blk.view(K_gp * bf * bf, V)
                )

        # Append
        self.grand_parents = torch.cat(
            [self.grand_parents, new_gps], dim=1
        ).contiguous()
        self.grand_parent_radii = torch.cat(
            [self.grand_parent_radii, new_gp_radii], dim=1
        ).contiguous()
        self.parents = torch.cat([self.parents, new_parents_block], dim=1).contiguous()
        self.parent_radii = torch.cat(
            [self.parent_radii, new_pr_block], dim=1
        ).contiguous()
        self.children = torch.cat(
            [self.children, new_children_block], dim=1
        ).contiguous()
        if has_values:
            self.values = torch.cat(
                [self.values, new_children_values_block], dim=-2
            ).contiguous()

    def _update_gp_radii_for_placed_children(
        self,
        placed_mask: torch.Tensor,  # (H, M) bool
        nearest_parent: torch.Tensor,  # (H, M) int32
    ):
        """Update grand_parent_radii after placing children under existing parents."""
        H = placed_mask.shape[0]
        bf = int(self.branching_factor)
        G = self.grand_parents.shape[1]
        device = placed_mask.device

        h_idx = torch.arange(H, device=device)[:, None].expand_as(placed_mask)
        h_m = h_idx[placed_mask]
        p_m = nearest_parent[placed_mask].to(torch.int64)
        gp_m = p_m // bf

        gp_centers = self.grand_parents[h_m, gp_m].float()
        par_vecs = self.parents[h_m, p_m].float()
        dist = torch.linalg.norm(par_vecs - gp_centers, dim=1)
        total = dist + self.parent_radii[h_m, p_m].float()

        upd_flat = torch.full(
            (H * G,), float("-inf"), device=device, dtype=torch.float32
        )
        upd_flat.scatter_reduce_(
            0, h_m * G + gp_m, total, reduce="amax", include_self=True
        )
        upd = upd_flat.view(H, G)
        upd = torch.where(torch.isfinite(upd), upd, torch.zeros_like(upd))
        self.grand_parent_radii = torch.maximum(
            self.grand_parent_radii.float(), upd
        ).to(self.grand_parent_radii.dtype)

    def _refresh_after_update(self):
        """Cached state after an update."""
        self._init_update_state()

    # ------------------------------------------------------------------
    # Update state management
    # ------------------------------------------------------------------

    def _init_update_state(self) -> None:
        if self.parents is None or self.children is None:
            self._child_counts = None
            self._parent_valid = None
            self._gp_child_counts = None
            return

        bf = int(self.branching_factor)

        if self.parents.ndim != 3:
            raise ValueError(
                f"parents must be (H,P,d), got {tuple(self.parents.shape)}"
            )
        if self.children.ndim != 3:
            raise ValueError(
                f"children must be (H,P*bf,d), got {tuple(self.children.shape)}"
            )

        H, P, d = self.parents.shape

        # Parent validity (padded parents are all pad_value).
        if self.pad_value is None:
            self._parent_valid = torch.ones(
                (H, P), device=self.parents.device, dtype=torch.bool
            )
        else:
            pad = float(self.pad_value)
            self._parent_valid = ~torch.all(self.parents == pad, dim=-1)  # (H,P)

        # Child counts per parent.
        children3 = self.children.view(H, P, bf, d)
        if self.pad_value is None:
            valid_child = torch.ones(
                (H, P, bf), device=self.children.device, dtype=torch.bool
            )
        else:
            pad = float(self.pad_value)
            valid_child = ~torch.all(children3 == pad, dim=-1)

        self._child_counts = valid_child.sum(dim=2).to(torch.int32).contiguous()

        # GP child counts (parent fill per GP) for THREE_LEVELS.
        if (
            self.depth == CUDAIndexer.DEPTH.THREE_LEVELS
            and self.grand_parents is not None
        ):
            G = self.grand_parents.shape[1]
            parents4 = self.parents.view(H, G, bf, d)
            if self.pad_value is None:
                gp_valid = torch.ones(
                    (H, G, bf), device=self.parents.device, dtype=torch.bool
                )
            else:
                pad = float(self.pad_value)
                gp_valid = ~torch.all(parents4 == pad, dim=-1)
            self._gp_child_counts = gp_valid.sum(dim=2).to(torch.int32).contiguous()
        else:
            self._gp_child_counts = None

    def _ensure_update_state(self) -> None:
        if self._child_counts is None or self._parent_valid is None:
            self._init_update_state()
