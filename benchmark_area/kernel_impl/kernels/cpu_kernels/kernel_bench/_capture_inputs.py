from __future__ import annotations

from pathlib import Path

import torch

try:
    from hira.benchmark_area.quick_pruning.pruning_bench_utils import CaptureState
except ModuleNotFoundError:
    from benchmark_area.quick_pruning.pruning_bench_utils import CaptureState


def q_to_kv_map(num_q_heads: int, num_kv_heads: int) -> torch.Tensor:
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads must be divisible by num_kv_heads, got "
            f"{num_q_heads} and {num_kv_heads}."
        )
    groups = num_q_heads // num_kv_heads
    return torch.arange(num_q_heads, dtype=torch.int64) // groups


def load_capture(path: Path, requested_layer: int | None) -> tuple[int, CaptureState, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    cap = CaptureState.load(path)
    layer_ids = cap.layer_ids()
    if not layer_ids:
        raise ValueError(f"Capture {path} does not contain any layers.")
    if requested_layer is not None and requested_layer in layer_ids:
        layer = requested_layer
    else:
        layer = layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, values_cpu = cap.to_layer_tensors(layer)
    return layer, cap, queries_cpu, keys_cpu, values_cpu


def real_build_case(path: Path, requested_layer: int | None, n: int | None) -> dict:
    layer, cap, queries_cpu, keys_cpu, values_cpu = load_capture(path, requested_layer)
    total = int(keys_cpu.shape[1])
    n_used = total if n is None else min(int(n), total)
    return {
        "layer": layer,
        "prompt_length": cap.prompt_length,
        "queries_cpu": queries_cpu,
        "keys": keys_cpu[:, :n_used, :].to(dtype=torch.float32).contiguous(),
        "values": (
            values_cpu[:, :n_used, :].to(dtype=torch.float32).contiguous()
            if values_cpu is not None
            else None
        ),
        "total_keys": total,
    }


def real_update_case(
    path: Path,
    requested_layer: int | None,
    n: int | None,
    b: int,
) -> dict:
    layer, cap, queries_cpu, keys_cpu, values_cpu = load_capture(path, requested_layer)
    total = int(keys_cpu.shape[1])
    prompt_length = int(cap.prompt_length or 0)
    if b <= 0:
        raise ValueError("--B must be positive")
    if n is None:
        old_len = max(prompt_length, total - b)
    else:
        old_len = int(n)
    if old_len < 1 or old_len + b > total:
        raise ValueError(
            f"Need 1 <= N and N + B <= captured keys ({total}), got N={old_len}, B={b}."
        )
    buf = slice(old_len, old_len + b)
    return {
        "layer": layer,
        "prompt_length": cap.prompt_length,
        "keys": keys_cpu[:, :old_len, :].to(dtype=torch.float32).contiguous(),
        "values": (
            values_cpu[:, :old_len, :].to(dtype=torch.float32).contiguous()
            if values_cpu is not None
            else None
        ),
        "buffer_keys": keys_cpu[:, buf, :].to(dtype=torch.float32).contiguous(),
        "buffer_values": (
            values_cpu[:, buf, :].to(dtype=torch.float32).contiguous()
            if values_cpu is not None
            else None
        ),
        "total_keys": total,
    }


def real_attention_case(
    path: Path,
    requested_layer: int | None,
    n: int | None,
    n_queries: int,
    buffer_len: int,
) -> dict:
    layer, cap, queries_cpu, keys_cpu, values_cpu = load_capture(path, requested_layer)
    if values_cpu is None:
        values_cpu = keys_cpu

    total_keys = int(keys_cpu.shape[1])
    if buffer_len < 0 or buffer_len >= total_keys:
        raise ValueError(f"--buffer-len must be in [0, {total_keys - 1}], got {buffer_len}")

    n_index_max = total_keys - buffer_len
    n_index = n_index_max if n is None else min(int(n), n_index_max)
    if n_index < 1:
        raise ValueError(f"Need at least one indexed key, got N={n_index}")

    total_q = int(queries_cpu.shape[1])
    n_q = min(max(1, int(n_queries)), total_q)
    stride = max(1, total_q // n_q)
    q_indices = list(range(total_q - 1, -1, -stride))[:n_q]
    q_batch = torch.stack(
        [
            queries_cpu[:, qi, :].to(dtype=torch.float32).contiguous()
            for qi in q_indices
        ],
        dim=0,
    )
    q_batch = q_batch / q_batch.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    keys_index = keys_cpu[:, :n_index, :].to(dtype=torch.float32).contiguous()
    values_index = values_cpu[:, :n_index, :].to(dtype=torch.float32).contiguous()
    keys_eval = keys_cpu[:, : n_index + buffer_len, :].to(dtype=torch.float32).contiguous()
    values_eval = values_cpu[:, : n_index + buffer_len, :].to(dtype=torch.float32).contiguous()
    buffer_keys = keys_cpu[:, n_index : n_index + buffer_len, :].to(dtype=torch.float32).contiguous()
    buffer_values = values_cpu[:, n_index : n_index + buffer_len, :].to(dtype=torch.float32).contiguous()

    h_q = int(q_batch.shape[1])
    h_kv = int(keys_index.shape[0])
    q_head_to_kv = q_to_kv_map(h_q, h_kv) if h_q != h_kv else None
    return {
        "layer": layer,
        "prompt_length": cap.prompt_length,
        "keys": keys_index,
        "values": values_index,
        "keys_eval": keys_eval,
        "values_eval": values_eval,
        "buffer_keys": buffer_keys,
        "buffer_values": buffer_values,
        "q_batch": q_batch,
        "q_head_to_kv": q_head_to_kv,
        "q_indices": q_indices,
        "total_keys": total_keys,
    }
