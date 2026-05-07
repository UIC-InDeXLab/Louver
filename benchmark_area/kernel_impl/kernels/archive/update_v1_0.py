"""update_v1.0 — index update kernels (full rebuild + incremental append).

Both variants return a new state dict (the same shape as build's output).

full: rebuild index on concat(old_keys, buffer_keys).
inc:  build a small index on buffer_keys only, then append its children and
      parent centroids/radii to the existing state along dim=1 (N dim) and
      dim=1 (K dim) respectively. Assignments from buffer are offset by the
      old K so they index into the concatenated parent layer.
"""

from __future__ import annotations

import torch

from .build_v1_0 import _parent_major_layout, build as _build_v1_0

KERNEL_VERSION = "v1.0"


def update(
    state: dict,
    old_keys: torch.Tensor,       # (H, N_old, D) — already in index
    buffer_keys: torch.Tensor,    # (H, B, D)
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    mode: str = "inc",
) -> tuple[dict, torch.Tensor]:
    """Return (new_state, new_keys_children) where new_keys_children is the
    concatenated key set used by the updated index (needed by search for
    dot-product evaluation).
    """
    if mode == "full":
        new_keys = torch.cat([old_keys, buffer_keys], dim=1).contiguous()
        new_state = _build_v1_0(new_keys, bf, n_subspaces, refine_iter)
        return new_state, new_keys

    if mode == "inc":
        # Build a small index on the buffer alone and concatenate with the
        # existing one along point (N) and parent (K) axes per subspace.
        sub_state = _build_v1_0(buffer_keys, bf, n_subspaces, refine_iter)

        K_old = state["K"]
        new_dim_slices = state["dim_slices"]  # same splits
        new_assigns, new_centers, new_radii = [], [], []
        new_child_order, new_child_offsets, new_child_counts = [], [], []

        for s in range(n_subspaces):
            # Offset buffer assigns into the appended parent range.
            a_new = torch.cat(
                [state["assigns"][s], sub_state["assigns"][s] + K_old], dim=1
            )
            c_new = torch.cat([state["centers"][s], sub_state["centers"][s]], dim=1)
            r_new = torch.cat([state["radii"][s], sub_state["radii"][s]], dim=1)
            order, offsets, counts = _parent_major_layout(a_new, K_old + sub_state["K"])
            new_assigns.append(a_new)
            new_centers.append(c_new)
            new_radii.append(r_new)
            new_child_order.append(order)
            new_child_offsets.append(offsets)
            new_child_counts.append(counts)

        new_state = {
            "dim_slices": new_dim_slices,
            "assigns": new_assigns,
            "centers": new_centers,
            "radii": new_radii,
            "child_order": new_child_order,
            "child_offsets": new_child_offsets,
            "child_counts": new_child_counts,
            "K": K_old + sub_state["K"],
            "N": state["N"] + sub_state["N"],
        }
        new_keys = torch.cat([old_keys, buffer_keys], dim=1).contiguous()
        return new_state, new_keys

    raise ValueError(f"Unknown update mode: {mode!r}")


KERNEL = update
