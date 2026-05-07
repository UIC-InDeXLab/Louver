import torch
from typing import Optional
import math

from .base import BaseSearcher
from hira.indexer import CUDAIndexer
from hira.kernels.triton_search_wrappers import (
    triton_two_level_filter,
    triton_three_level_filter,
)

DEFAULT_OUTPUT_FILL_VALUE = 0.0  # used for searching, filtered-out values in the output


class CUDASearcher(BaseSearcher):

    def __init__(self, block_c, output_fill_value: float = DEFAULT_OUTPUT_FILL_VALUE):
        super().__init__()
        self.block_c = block_c
        self.output_fill_value = float(output_fill_value)
        self._tmp_out: Optional[torch.Tensor] = None

    @staticmethod
    def _resolve_q_head_to_kv(
        *,
        num_query_heads: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        if num_query_heads == num_kv_heads:
            return torch.arange(num_query_heads, dtype=torch.long)
        assert num_query_heads % num_kv_heads == 0
        group_size = num_query_heads // num_kv_heads
        return torch.arange(num_query_heads, dtype=torch.long) // group_size

    def _prepare_output_buffer(
        self, *, num_query_heads: int, num_keys: int, indexer: CUDAIndexer
    ) -> torch.Tensor:
        fill_value = self.output_fill_value
        needs_new = (
            self._tmp_out is None
            or self._tmp_out.device != indexer.children.device
            or self._tmp_out.dtype != indexer.children.dtype
            or tuple(self._tmp_out.shape) != (num_query_heads, num_keys)
        )
        if needs_new:
            self._tmp_out = torch.full(
                (num_query_heads, num_keys),
                fill_value,
                device=indexer.children.device,
                dtype=indexer.children.dtype,
            )
        else:
            self._tmp_out.fill_(fill_value)
        return self._tmp_out

    def search(
        self,
        query,
        threshold,
        indexer: CUDAIndexer,
        q_head_to_kv: Optional[torch.Tensor] = None,
        scaling: Optional[torch.Tensor] = None,
    ):
        """GPU tree-pruned inner-product search using Triton kernels.

        High-level behavior:
        - Parent (and grandparent for 3-level) traversal uses
          ``dot(q, center) + radius`` against ``threshold`` to decide which
          branches are scanned.
        - Child scores are emitted for surviving branches and scaled by
          ``scaling``.
        - Output has shape ``(H_q, N)`` and is initialized with
          ``output_fill_value`` in non-scanned/rejected positions.

        ``q`` is expected to be L2-normalized for the tree bound semantics.

        Args:
            query:     ``(1, H, 1, D)`` query tensor.
            threshold: ``(H,)`` per-head threshold tensor.
            scaling:   Optional ``(H,)`` per-head scaling tensor for returned scores.
        Returns:
            ``(H_q, N)`` float tensor -- dot-product scores for qualifying
            keys, ``output_fill_value`` for pruned / below-threshold keys.
        """
        if indexer.children is None:
            raise ValueError("Indexer is not built: missing children tensor.")
        num_kv_heads = int(indexer.children.shape[0])
        num_keys = int(indexer.children.shape[1])

        # query = query.squeeze(0).squeeze(-2).contiguous()
        num_query_heads = int(query.shape[0])

        assert threshold.shape == (
            query.shape[0],
        ), "threshold shape must match number of heads in query"

        if scaling is None:
            scaling = torch.ones(
                (query.shape[0],), device=query.device, dtype=torch.float32
            )
        assert scaling.shape == threshold.shape

        if q_head_to_kv is None:
            q_head_to_kv = self._resolve_q_head_to_kv(
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
            )

        out = self._prepare_output_buffer(
            num_query_heads=num_query_heads, num_keys=num_keys, indexer=indexer
        )

        depth_value = getattr(indexer.depth, "value", indexer.depth)

        if depth_value == CUDAIndexer.DEPTH.TWO_LEVELS.value:
            output = triton_two_level_filter(
                indexer.children,
                indexer.parents,
                indexer.parent_radii,
                query,
                threshold,
                q_head_to_kv=q_head_to_kv,
                out=out,
                BLOCK_C=self.block_c,
                branch=indexer.branching_factor,
                scaling=scaling,
            )
        elif depth_value == CUDAIndexer.DEPTH.THREE_LEVELS.value:
            output = triton_three_level_filter(
                indexer.children,
                indexer.parents,
                indexer.parent_radii,
                indexer.grand_parents,
                indexer.grand_parent_radii,
                query,
                threshold,
                q_head_to_kv=q_head_to_kv,
                out=out,
                branch=indexer.branching_factor,
                BLOCK_C=self.block_c,
                scaling=scaling,
            )
        else:
            raise ValueError(f"Unsupported index depth: {indexer.depth}")
        return output

    def synthetic_scanned_fraction(
        self,
        query,
        threshold,
        indexer: CUDAIndexer,
        q_head_to_kv: Optional[torch.Tensor] = None,
        scaling: Optional[torch.Tensor] = None,
    ):
        """
        Synthetic estimate of how many child rows are scanned by the CUDA traversal.

        Returns a dict with:
        - scanned_fraction_per_head: (H_q,) tensor in [0, 1]
        - scanned_children_per_head: (H_q,) tensor (counts)
        - total_children: int
        - scanned_children_total: int -- sum of scanned children across all query heads
        - scanned_fraction_mean: float
        - output_size_per_head: (H_q,) tensor -- children with dot(q, child) > threshold
        - output_size_total: int -- sum of output_size_per_head across all query heads
        """
        if indexer.children is None:
            raise ValueError("Indexer is not built: missing children tensor.")
        num_kv_heads = int(indexer.children.shape[0])
        num_keys = int(indexer.children.shape[1])

        # query = query.squeeze(0).squeeze(-2).contiguous()
        num_query_heads = int(query.shape[0])

        assert threshold.shape == (
            query.shape[0],
        ), "threshold shape must match number of heads in query"

        if scaling is None:
            scaling = torch.ones(
                (query.shape[0],), device=query.device, dtype=torch.float32
            )
        assert scaling.shape == threshold.shape

        if q_head_to_kv is None:
            q_head_to_kv = self._resolve_q_head_to_kv(
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
            )

        depth_value = getattr(indexer.depth, "value", indexer.depth)
        bf = int(indexer.branching_factor)

        parents = indexer.parents.index_select(0, q_head_to_kv)
        parent_radii = indexer.parent_radii.index_select(0, q_head_to_kv)

        if depth_value == CUDAIndexer.DEPTH.TWO_LEVELS.value:
            # Parent gate: (q·p + r) > t
            parent_scores = torch.einsum("hmd,hd->hm", parents, query)
            parent_pass = (parent_scores + parent_radii) > threshold.unsqueeze(-1)
            scanned_children_per_head = parent_pass.sum(dim=1).to(torch.int64) * bf
        elif depth_value == CUDAIndexer.DEPTH.THREE_LEVELS.value:
            if indexer.grand_parents is None or indexer.grand_parent_radii is None:
                raise ValueError(
                    "Three-level search requires grand_parents and grand_parent_radii."
                )
            grand_parents = indexer.grand_parents.index_select(0, q_head_to_kv)
            grand_parent_radii = indexer.grand_parent_radii.index_select(
                0, q_head_to_kv
            )
            # Grandparent gate (P2 -> P1 mask pass)
            gp_scores = torch.einsum("hgd,hd->hg", grand_parents, query)
            gp_pass = (gp_scores + grand_parent_radii) > threshold.unsqueeze(-1)

            # Expand gp mask to level-1 parents exactly as branch-grouped layout.
            gp_mask_on_p1 = (
                gp_pass.unsqueeze(-1).expand(-1, -1, bf).reshape(gp_pass.shape[0], -1)
            )

            # Parent gate (P1 -> K), masked by grandparent pass.
            p1_scores = torch.einsum("hmd,hd->hm", parents, query)
            p1_pass = gp_mask_on_p1 & (
                (p1_scores + parent_radii) > threshold.unsqueeze(-1)
            )
            scanned_children_per_head = p1_pass.sum(dim=1).to(torch.int64) * bf
        else:
            raise ValueError(f"Unsupported index depth: {indexer.depth}")

        total_children = int(indexer.children.shape[1])
        denom = max(1, total_children)
        scanned_fraction_per_head = scanned_children_per_head.to(torch.float32) / float(
            denom
        )

        scanned_children_total = int(scanned_children_per_head.sum().item())

        # Output size: children whose dot product with the query exceeds the threshold.
        children = indexer.children.index_select(0, q_head_to_kv)  # (H_q, N, D)
        child_scores = torch.einsum("hnd,hd->hn", children, query)  # (H_q, N)
        output_size_per_head = (child_scores > threshold.unsqueeze(-1)).sum(
            dim=1
        )  # (H_q,)
        output_size_mean = float(output_size_per_head.float().mean().item())

        return {
            "scanned_fraction_per_head": scanned_fraction_per_head,
            "scanned_children_per_head": scanned_children_per_head,
            "total_children": total_children,
            "scanned_fraction_mean": float(scanned_fraction_per_head.mean().item()),
            "output_size_mean": output_size_mean,
        }
