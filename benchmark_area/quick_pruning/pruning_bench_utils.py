#!/usr/bin/env python3
"""
Run a CUDA simulated HIRA pruning benchmark by capturing Q/K/V directly from
Transformers' AttentionInterface (no observer hooks, no save/load round-trip).

Output CSV columns:
  layer_idx, token_idx, token_pos, num_keys, scanned_fraction, output_size_mean
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
from transformers.models.llama.modeling_llama import eager_attention_forward as _llama_eager_attn

_FLASH_ONLY_KWARGS = frozenset({"sliding_window"})

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency

    def _tqdm(iterable, **kwargs):
        return iterable


def _repo_root() -> Path:
    # .../hira/attention/benchs/simulated_pruning_bench.py -> repo root
    return Path(__file__).resolve().parents[3]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.indexer.cuda import CUDAIndexer
from hira.searcher.cuda import CUDASearcher
from hira.threshold.algs import (
    FullSearchThreshold,
    SampleMaxThreshold,
    SampleMeanMaxThreshold,
    TopKThreshold,
)


ATTN_CAPTURE_IMPL = "sim_capture_attention_ref"
_CAPTURE_STATE: "CaptureState | None" = None
_QWEN2_CAPTURE_PATCHED = False


@dataclass
class CaptureState:
    prompt_length: int | None = None
    prefill_keys: dict[int, torch.Tensor] = field(default_factory=dict)
    prefill_values: dict[int, torch.Tensor] = field(default_factory=dict)
    generated_queries: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    generated_keys: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    generated_values: dict[int, list[torch.Tensor]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_length": self.prompt_length,
            "prefill_keys": self.prefill_keys,
            "prefill_values": self.prefill_values,
            "generated_queries": self.generated_queries,
            "generated_keys": self.generated_keys,
            "generated_values": self.generated_values,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "CaptureState":
        data = torch.load(path, map_location="cpu")
        return cls(
            prompt_length=data["prompt_length"],
            prefill_keys={int(k): v.contiguous() for k, v in data["prefill_keys"].items()},
            prefill_values={
                int(k): v.contiguous() for k, v in data["prefill_values"].items()
            },
            generated_queries={
                int(k): [x.contiguous() for x in values]
                for k, values in data["generated_queries"].items()
            },
            generated_keys={
                int(k): [x.contiguous() for x in values]
                for k, values in data["generated_keys"].items()
            },
            generated_values={
                int(k): [x.contiguous() for x in values]
                for k, values in data["generated_values"].items()
            },
        )

    def _to_cpu_half(self, x: torch.Tensor) -> torch.Tensor:
        return x.detach().to(device="cpu", dtype=torch.float16).contiguous()

    def record(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor | None,
    ) -> None:
        layer_idx = int(getattr(module, "layer_idx", -1))
        if layer_idx < 0:
            return

        q_len = int(query.shape[-2])
        if q_len > 1:
            # Prefill pass: keep prompt keys/values for this layer.
            if self.prompt_length is None:
                self.prompt_length = q_len
            elif self.prompt_length != q_len:
                raise RuntimeError(
                    f"Inconsistent prompt lengths captured: {self.prompt_length} vs {q_len}"
                )

            self.prefill_keys[layer_idx] = self._to_cpu_half(key[0])
            if value is not None:
                self.prefill_values[layer_idx] = self._to_cpu_half(value[0])

            self.generated_queries.setdefault(layer_idx, [])
            self.generated_keys.setdefault(layer_idx, [])
            self.generated_values.setdefault(layer_idx, [])
            return

        # Decode pass (single token): capture query + newly appended key/value.
        self.generated_queries.setdefault(layer_idx, []).append(
            self._to_cpu_half(query[0, :, 0, :])
        )
        self.generated_keys.setdefault(layer_idx, []).append(
            self._to_cpu_half(key[0, :, -1, :])
        )
        if value is not None:
            self.generated_values.setdefault(layer_idx, []).append(
                self._to_cpu_half(value[0, :, -1, :])
            )

    def layer_ids(self) -> list[int]:
        return sorted(self.prefill_keys.keys())

    def generated_token_count(self) -> int:
        counts = [len(v) for v in self.generated_queries.values()]
        return min(counts) if counts else 0

    def to_layer_tensors(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if layer_idx not in self.prefill_keys:
            raise ValueError(f"Layer {layer_idx} missing prefill keys.")

        prefill_k = self.prefill_keys[layer_idx]
        q_list = self.generated_queries.get(layer_idx, [])
        if not q_list:
            raise ValueError(f"Layer {layer_idx} has no generated queries.")

        k_list = self.generated_keys.get(layer_idx, [])
        if len(k_list) != len(q_list):
            raise ValueError(
                f"Layer {layer_idx} has mismatched query/key counts: "
                f"{len(q_list)} vs {len(k_list)}"
            )

        queries = torch.stack(q_list, dim=1)  # (H_q, T, D)
        generated_k = torch.stack(k_list, dim=1) if k_list else None  # (H_kv, T, D)
        keys = (
            torch.cat([prefill_k, generated_k], dim=1)
            if generated_k is not None
            else prefill_k
        )

        values: torch.Tensor | None = None
        if layer_idx in self.prefill_values:
            prefill_v = self.prefill_values[layer_idx]
            v_list = self.generated_values.get(layer_idx, [])
            if len(v_list) == len(q_list) and v_list:
                generated_v = torch.stack(v_list, dim=1)
                values = torch.cat([prefill_v, generated_v], dim=1)
            else:
                values = prefill_v

        return queries, keys, values


def _capture_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float,
    **kwargs,
):
    global _CAPTURE_STATE
    if _CAPTURE_STATE is not None:
        _CAPTURE_STATE.record(module=module, query=query, key=key, value=value)

    eager_attn = _resolve_model_eager_attention(type(module).__module__)
    filtered = {k: v for k, v in kwargs.items() if k not in _FLASH_ONLY_KWARGS}
    return eager_attn(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout,
        **filtered,
    )


@lru_cache(maxsize=None)
def _resolve_model_eager_attention(module_name: str):
    if module_name.startswith("transformers.models.llama."):
        return _llama_eager_attn

    module = importlib.import_module(module_name)
    eager_attn = getattr(module, "eager_attention_forward", None)
    if eager_attn is None:
        raise RuntimeError(
            f"Capture attention does not know how to dispatch eager attention for "
            f"module '{module_name}'."
        )
    return eager_attn


def _register_capture_attention_impl() -> None:
    try:
        AttentionInterface.register(ATTN_CAPTURE_IMPL, _capture_attention_forward)
    except ValueError:
        # Already registered in this process.
        pass


def _install_qwen2_capture_forward() -> None:
    global _QWEN2_CAPTURE_PATCHED
    if _QWEN2_CAPTURE_PATCHED:
        return

    qwen2_mod = importlib.import_module("transformers.models.qwen2.modeling_qwen2")
    original_forward = qwen2_mod.Qwen2Attention.forward

    def _capture_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = qwen2_mod.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        if _CAPTURE_STATE is not None:
            _CAPTURE_STATE.record(
                module=self,
                query=query_states,
                key=key_states,
                value=value_states,
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            qwen2_mod.eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    qwen2_mod.Qwen2Attention.forward = _capture_forward
    qwen2_mod.Qwen2Attention._capture_original_forward = original_forward
    _QWEN2_CAPTURE_PATCHED = True

    try:
        ALL_MASK_ATTENTION_FUNCTIONS.register(
            ATTN_CAPTURE_IMPL, ALL_MASK_ATTENTION_FUNCTIONS["eager"]
        )
    except ValueError:
        # Already registered in this process.
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate CUDA index/search pruning from directly captured Q/K tensors."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HF model id for direct capture.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10000,
        help="Number of generated tokens/queries to capture (default: 10000).",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional prompt text file. Defaults to PROMPT from run_attn.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("simulated_attention_stats.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help='Layer selection: "all" or comma-separated list, e.g. "0,1,2".',
    )
    parser.add_argument(
        "--start-token",
        type=int,
        default=0,
        help="First generated token index to process.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="Max generated tokens to process per selected layer.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Token stride (1 = every token).",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        choices=[2, 3],
        default=3,
        help="CUDAIndexer depth.",
    )
    parser.add_argument(
        "--branching-factor",
        type=int,
        default=8,
        help="CUDAIndexer branching factor.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="CUDAIndexer max k-means iterations.",
    )
    parser.add_argument(
        "--threshold-alg",
        choices=["topk", "sample_max", "sample_mean_max", "full_search"],
        default="topk",
        help="Threshold algorithm used during simulated search.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Sample size for sample_* threshold algorithms.",
    )
    parser.add_argument(
        "--block-c",
        type=int,
        default=8,
        help="CUDASearcher block_c parameter.",
    )
    parser.add_argument(
        "--run-search",
        action="store_true",
        help="If set, also call searcher.search(...) for each token.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    parser.add_argument(
        "--update-every",
        type=int,
        default=256,
        help="Update indexer every N tokens (1 = every token, 256 = batch like attention_stats_v2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device for capture/simulation. Must be "cuda".',
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Model dtype for direct capture pass.",
    )
    return parser.parse_args()


def _parse_layers(spec: str, available_layers: list[int]) -> list[int]:
    if spec == "all":
        return available_layers
    selected = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        selected.append(int(part))
    missing = sorted(set(selected) - set(available_layers))
    if missing:
        raise ValueError(f"Requested layers not present in capture: {missing}")
    return sorted(selected)


def _load_prompt_text(prompt_file: Path | None) -> str:
    if prompt_file is not None:
        return prompt_file.read_text()
    from hira.attention.benchs.run_attn import PROMPT

    return PROMPT


def _make_threshold(name: str, sample_size: int):
    if name == "topk":
        return TopKThreshold()
    if name == "sample_max":
        return SampleMaxThreshold(sample_size=sample_size)
    if name == "sample_mean_max":
        return SampleMeanMaxThreshold(sample_size=sample_size)
    if name == "full_search":
        return FullSearchThreshold()
    raise ValueError(f"Unsupported threshold algorithm: {name}")


def _threshold_for_query(
    threshold_alg: str,
    threshold_obj,
    q_normal: torch.Tensor,
    indexer: CUDAIndexer,
) -> torch.Tensor:
    if threshold_alg == "topk":
        return threshold_obj.get_threshold(q_normal, indexer)
    return threshold_obj.get_threshold(q_normal)


def _q_to_kv_map(num_q_heads: int, num_kv_heads: int, device: str) -> torch.Tensor:
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"GQA mapping requires num_q_heads % num_kv_heads == 0, got "
            f"{num_q_heads} and {num_kv_heads}."
        )
    groups = num_q_heads // num_kv_heads
    return torch.arange(num_q_heads, device=device, dtype=torch.int64) // groups


def _capture_qkv(
    *,
    model_name: str,
    prompt_text: str,
    n: int,
    device: str,
    torch_dtype: torch.dtype,
    show_progress: bool,
    show_tokens: bool = False,
) -> CaptureState:
    global _CAPTURE_STATE

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if config.model_type == "qwen2":
        _install_qwen2_capture_forward()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch_dtype,
        )
    else:
        _register_capture_attention_impl()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch_dtype,
            attn_implementation=ATTN_CAPTURE_IMPL,
        )
    model.eval()

    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generated_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    past_key_values = None

    capture = CaptureState()
    _CAPTURE_STATE = capture

    if show_tokens:
        print(f"\n{'=' * 60}")
        print("GENERATED TEXT (progressive)")
        print(f"{'=' * 60}")

    try:
        steps = range(n + 1)
        # Disable tqdm when showing tokens — use inline progress instead.
        use_tqdm = show_progress and not show_tokens
        iterator = _tqdm(steps, desc="Capture QKV", disable=not use_tqdm)

        with torch.no_grad():
            for step in iterator:
                input_ids = generated_ids if step == 0 else generated_ids[:, -1:]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values

                # Extra forward pass at step == n captures Q/K/V for token n-1.
                if step >= n:
                    break

                next_token = torch.argmax(
                    outputs.logits[:, -1, :], dim=-1, keepdim=True
                )
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=1,
                )

                if show_tokens:
                    tok_str = tokenizer.decode(
                        next_token[0, 0].item(), skip_special_tokens=False
                    )
                    print(tok_str, end="", flush=True)
                    if step % 200 == 0:
                        print(f"\n\033[90m--- [{step}/{n}] ---\033[0m", flush=True)

    finally:
        _CAPTURE_STATE = None

    gen_count = capture.generated_token_count()

    if show_tokens:
        print(f"\n\033[90m--- [{gen_count}/{n} done] ---\033[0m")
        print(f"{'=' * 60}\n")

    if capture.prompt_length is None:
        raise RuntimeError("Failed to capture prompt length from attention calls.")

    if gen_count < n:
        raise RuntimeError(
            f"Captured only {gen_count} generated queries, expected at least {n}."
        )

    return capture


def main() -> None:
    args = parse_args()

    if args.device != "cuda":
        raise ValueError('This benchmark is CUDA-only. Use "--device cuda".')
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")

    torch_dtype = torch.float16 if args.torch_dtype == "float16" else torch.float32

    prompt_text = _load_prompt_text(args.prompt_file)

    capture_start = time.perf_counter()
    capture = _capture_qkv(
        model_name=args.model_name,
        prompt_text=prompt_text,
        n=args.n,
        device=args.device,
        torch_dtype=torch_dtype,
        show_progress=not args.no_progress,
    )
    capture_elapsed = time.perf_counter() - capture_start

    layer_ids = capture.layer_ids()
    if not layer_ids:
        raise RuntimeError("No layers were captured.")

    selected_layers = _parse_layers(args.layers, layer_ids)

    token_start = max(0, args.start_token)
    max_generated = capture.generated_token_count()
    token_stop = min(max_generated, token_start + max(0, args.max_tokens))
    stride = max(1, args.stride)
    selected_token_indices = set(range(token_start, token_stop, stride))
    if not selected_token_indices:
        raise ValueError(
            "No tokens selected. Check --start-token/--max-tokens/--stride."
        )

    prompt_len = int(capture.prompt_length)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    searcher = CUDASearcher(block_c=args.block_c)
    rows: list[dict[str, float | int]] = []

    print(
        f"Captured layers={len(layer_ids)}, prompt_length={prompt_len}, "
        f"generated_queries={max_generated}, capture_time={capture_elapsed:.2f}s"
    )
    print(
        f"Selected layers={selected_layers}, token_count={len(selected_token_indices)}, "
        f"token_span={token_start}..{token_stop - 1}"
    )

    sim_start = time.perf_counter()

    layer_iter = _tqdm(
        selected_layers,
        desc="Sim layers",
        disable=args.no_progress,
    )
    for layer_idx in layer_iter:
        layer_start = time.perf_counter()

        queries_cpu, keys_cpu, _ = capture.to_layer_tensors(layer_idx)

        num_q_heads = int(queries_cpu.shape[0])
        num_kv_heads = int(keys_cpu.shape[0])
        q_head_to_kv = _q_to_kv_map(num_q_heads, num_kv_heads, args.device)

        queries = queries_cpu.to(
            device=args.device, dtype=torch.float32, non_blocking=True
        )
        keys = keys_cpu.to(device=args.device, dtype=torch.float32, non_blocking=True)

        initial_len = prompt_len + token_start
        if initial_len <= 0:
            raise ValueError(
                "prompt_length must be positive in captured attention data."
            )
        if initial_len > int(keys.shape[1]):
            raise ValueError(
                f"Requested start_token={token_start} exceeds available keys."
            )

        indexer = CUDAIndexer(
            num_levels=args.num_levels,
            max_iterations=args.max_iterations,
            branching_factor=args.branching_factor,
        ).build(keys[:, :initial_len, :].contiguous())

        thresholder = _make_threshold(args.threshold_alg, args.sample_size)
        if args.threshold_alg in {"sample_max", "sample_mean_max"}:
            thresholder.prefill_prep(keys[:, :initial_len, :].unsqueeze(0).contiguous())

        # Match hira_attention_stats_isolated token_pos convention:
        # first token position is based on indexer's effective child count (+1).
        token_pos_base = int(indexer.children.shape[-2] + 1 - token_start)

        token_iter = _tqdm(
            range(token_start, token_stop),
            total=max(0, token_stop - token_start),
            desc=f"Layer {layer_idx} tokens",
            leave=False,
            disable=args.no_progress,
        )
        update_every = max(1, args.update_every)
        pending_since = 0  # count of tokens not yet flushed to the indexer

        for token_idx in token_iter:
            # Keep token_pos aligned with hira_attention_stats_isolated CSV.
            num_keys_before_update = int(prompt_len + token_idx)
            token_pos = int(token_pos_base + token_idx)

            if token_idx in selected_token_indices:
                q = queries[:, token_idx, :]
                q_norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
                q_normal = q / q_norm

                th = _threshold_for_query(
                    args.threshold_alg, thresholder, q_normal, indexer
                )

                stats = searcher.synthetic_scanned_fraction(
                    query=q_normal,
                    threshold=th,
                    indexer=indexer,
                    q_head_to_kv=q_head_to_kv,
                )

                if args.run_search:
                    _ = searcher.search(
                        query=q_normal,
                        threshold=th,
                        indexer=indexer,
                        q_head_to_kv=q_head_to_kv,
                    )

                rows.append(
                    {
                        "layer_idx": int(layer_idx),
                        "token_idx": int(token_idx),
                        "token_pos": int(token_pos),
                        "num_keys": int(token_pos),
                        "scanned_fraction": float(stats["scanned_fraction_mean"]),
                        "output_size_mean": float(stats["output_size_mean"]),
                    }
                )

            # Advance index state: batch updates every `update_every` tokens.
            pending_since += 1
            if pending_since >= update_every:
                start_idx = num_keys_before_update + 1 - pending_since
                end_idx = num_keys_before_update + 1
                new_keys = keys[:, start_idx:end_idx, :].contiguous()
                if new_keys.shape[1] > 0:
                    indexer.update(new_keys)
                    if args.threshold_alg in {"sample_max", "sample_mean_max"}:
                        thresholder.update(new_keys.unsqueeze(0), cache_len=token_pos)
                pending_since = 0

        # Flush any remaining pending tokens after the loop.
        if pending_since > 0:
            flush_end = int(prompt_len + token_stop - 1) + 1
            flush_start = flush_end - pending_since
            remaining = keys[:, flush_start:flush_end, :].contiguous()
            if remaining.shape[1] > 0:
                indexer.update(remaining)

        layer_elapsed = time.perf_counter() - layer_start
        print(
            f"Layer {layer_idx} done in {layer_elapsed:.2f}s "
            f"(token span {token_start}..{token_stop - 1})"
        )

    fieldnames = [
        "layer_idx",
        "token_idx",
        "token_pos",
        "num_keys",
        "scanned_fraction",
        "output_size_mean",
    ]
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {args.output_csv}")

    by_token = {}
    for r in rows:
        by_token.setdefault(r["token_idx"], []).append(float(r["scanned_fraction"]))
    token_means = [sum(v) / len(v) for _, v in sorted(by_token.items())]

    if token_means:
        mean_scan = sum(token_means) / len(token_means)
        mean_pruning = 1.0 - mean_scan
        print(
            f"Simulated scanned_fraction mean={mean_scan:.6f}, "
            f"min={min(token_means):.6f}, max={max(token_means):.6f}"
        )
        print(f"Simulated pruning mean={mean_pruning:.6f}")

    print(f"Simulation wall time: {time.perf_counter() - sim_start:.2f}s")


if __name__ == "__main__":
    main()
