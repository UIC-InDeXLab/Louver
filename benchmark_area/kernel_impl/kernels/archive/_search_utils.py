"""Shared helpers for search_v* kernels."""

from __future__ import annotations

import torch

_NEG_INF = float("-inf")


def _mapping_mode(
    q_head_to_kv: torch.Tensor | None,
    h_q: int,
    h_kv: int,
) -> tuple[str, int, tuple[int, ...] | None]:
    if q_head_to_kv is None:
        return "identity", 1, None

    if h_q % h_kv == 0:
        groups = h_q // h_kv
        expected = torch.arange(h_q, device=q_head_to_kv.device, dtype=q_head_to_kv.dtype) // groups
        if torch.equal(q_head_to_kv, expected):
            return "grouped", groups, None

    return "expanded", 1, tuple(int(x) for x in q_head_to_kv.tolist())


def get_layout_cache(
    state: dict,
    keys_children: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    cache_name: str,
) -> dict:
    """Build or fetch packed search layout tensors for a given head mapping."""
    h_kv = int(keys_children.shape[0])
    h_q = int(q_head_to_kv.shape[0]) if q_head_to_kv is not None else h_kv
    mode, groups, mapping_sig = _mapping_mode(q_head_to_kv, h_q, h_kv)
    cache_key = (mode, groups, mapping_sig)

    root = state.setdefault(cache_name, {})
    if (
        root.get("cache_key") == cache_key
        and root.get("keys_ptr") == keys_children.data_ptr()
        and root.get("keys_shape") == tuple(keys_children.shape)
        and root.get("keys_dtype") == keys_children.dtype
    ):
        return root["layout"]

    dim_slices = state["dim_slices"]
    widths = [end - start for start, end in dim_slices]
    max_d = max(widths)
    num_subspaces = len(dim_slices)

    if mode == "expanded":
        assert q_head_to_kv is not None
        centers_src = [t.index_select(0, q_head_to_kv).contiguous() for t in state["centers"]]
        radii_src = [t.index_select(0, q_head_to_kv).contiguous() for t in state["radii"]]
        assigns_src = [t.index_select(0, q_head_to_kv).contiguous() for t in state["assigns"]]
        child_order_src = [t.index_select(0, q_head_to_kv).contiguous() for t in state["child_order"]]
        child_offsets_src = [t.index_select(0, q_head_to_kv).contiguous() for t in state["child_offsets"]]
        child_counts_src = [t.index_select(0, q_head_to_kv).contiguous() for t in state["child_counts"]]
        keys_base = keys_children.index_select(0, q_head_to_kv).contiguous()
    else:
        centers_src = state["centers"]
        radii_src = state["radii"]
        assigns_src = state["assigns"]
        child_order_src = state.get("child_order")
        child_offsets_src = state.get("child_offsets")
        child_counts_src = state.get("child_counts")
        keys_base = keys_children

    base_heads = int(centers_src[0].shape[0])
    k_clusters = int(centers_src[0].shape[1])

    centers = torch.zeros(
        num_subspaces,
        base_heads,
        k_clusters,
        max_d,
        device=keys_base.device,
        dtype=centers_src[0].dtype,
    )
    for s, center_s in enumerate(centers_src):
        centers[s, :, :, : center_s.shape[-1]] = center_s

    layout = {
        "mode": mode,
        "groups": groups,
        "base_heads": base_heads,
        "num_subspaces": num_subspaces,
        "num_points": int(keys_base.shape[1]),
        "max_d": max_d,
        "dim_slices": tuple(dim_slices),
        "widths": torch.tensor(widths, device=keys_base.device, dtype=torch.int32),
        "keys": keys_base,
        "centers": centers.contiguous(),
        "centers_t": centers.reshape(num_subspaces * base_heads, k_clusters, max_d)
        .transpose(1, 2)
        .contiguous(),
        "radii": torch.stack(radii_src, dim=0).contiguous(),
        "assigns": torch.stack(assigns_src, dim=0).contiguous(),
        "assigns_i32": torch.stack(assigns_src, dim=0).to(torch.int32).contiguous(),
        "keys_t": keys_base.transpose(1, 2).contiguous(),
    }

    if child_order_src is not None and child_offsets_src is not None and child_counts_src is not None:
        layout["child_order"] = torch.stack(child_order_src, dim=0).contiguous()
        layout["child_offsets"] = torch.stack(child_offsets_src, dim=0).contiguous()
        layout["child_counts"] = torch.stack(child_counts_src, dim=0).contiguous()

    root["cache_key"] = cache_key
    root["keys_ptr"] = keys_children.data_ptr()
    root["keys_shape"] = tuple(keys_children.shape)
    root["keys_dtype"] = keys_children.dtype
    root["layout"] = layout
    return layout


def _group_query(q: torch.Tensor, layout: dict) -> torch.Tensor:
    return q.view(layout["base_heads"], layout["groups"], q.shape[-1])


def pack_query_subspaces(
    q: torch.Tensor,
    layout: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_grouped = _group_query(q, layout)
    q_packed = q.new_zeros(
        layout["num_subspaces"],
        layout["base_heads"],
        layout["groups"],
        layout["max_d"],
    )
    for s, (start, end) in enumerate(layout["dim_slices"]):
        q_packed[s, :, :, : end - start] = q_grouped[:, :, start:end]

    q_norm = q_packed.norm(dim=-1)
    return q_grouped, q_packed, q_norm


def cluster_pass_only(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    layout: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-subspace cluster-pass mask and grouped query view."""
    q_grouped, q_packed, q_norm = pack_query_subspaces(q, layout)
    center_dots = torch.bmm(
        q_packed.reshape(
            layout["num_subspaces"] * layout["base_heads"],
            layout["groups"],
            layout["max_d"],
        ),
        layout["centers_t"],
    ).reshape(
        layout["num_subspaces"],
        layout["base_heads"],
        layout["groups"],
        -1,
    )

    cluster_ub = center_dots + layout["radii"][:, :, None, :] * q_norm[:, :, :, None]
    cluster_pass = cluster_ub >= th_per_subspace.view(
        layout["num_subspaces"],
        layout["base_heads"],
        layout["groups"],
    ).unsqueeze(-1)
    return cluster_pass, q_grouped


def flatten_cluster_pass(cluster_pass: torch.Tensor, layout: dict) -> torch.Tensor:
    return cluster_pass.reshape(
        layout["num_subspaces"],
        layout["base_heads"] * layout["groups"],
        cluster_pass.shape[-1],
    ).contiguous()


def gate_and_group_query(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    layout: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return survive mask over index children and grouped query view."""
    cluster_pass, q_grouped = cluster_pass_only(q, th_per_subspace, layout)
    point_pass = cluster_pass.gather(
        -1,
        layout["assigns"][:, :, None, :].expand(-1, -1, layout["groups"], -1),
    )
    survive = point_pass.all(dim=0).reshape(q.shape[0], layout["num_points"])
    return survive, q_grouped


def dense_index_search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    layout: dict,
) -> torch.Tensor:
    survive, q_grouped = gate_and_group_query(q, th_per_subspace, layout)
    dots = torch.bmm(q_grouped, layout["keys_t"]).reshape(q.shape[0], layout["num_points"])
    return dots.masked_fill(~survive, _NEG_INF)


def make_compiled_dense_core(layout: dict):
    def core(q: torch.Tensor, th_per_subspace: torch.Tensor) -> torch.Tensor:
        return dense_index_search(q, th_per_subspace, layout)

    return torch.compile(core, fullgraph=False, dynamic=False)


def buffer_dot(
    q: torch.Tensor,
    buffer_keys: torch.Tensor | None,
    q_head_to_kv: torch.Tensor | None,
    layout: dict,
) -> torch.Tensor | None:
    if buffer_keys is None or buffer_keys.shape[1] == 0:
        return None

    if layout["mode"] == "grouped":
        q_grouped = _group_query(q, layout)
        return torch.bmm(q_grouped, buffer_keys.transpose(1, 2)).reshape(q.shape[0], buffer_keys.shape[1])

    if layout["mode"] == "expanded":
        assert q_head_to_kv is not None
        keys_base = buffer_keys.index_select(0, q_head_to_kv)
    else:
        keys_base = buffer_keys

    return torch.bmm(q.unsqueeze(1), keys_base.transpose(1, 2)).squeeze(1)
