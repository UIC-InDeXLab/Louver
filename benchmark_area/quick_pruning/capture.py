#!/usr/bin/env python3
"""
Debug script: capture QKV and display generated tokens progressively.

Usage:
    python debug_capture.py --n-tokens 200
    python debug_capture.py --n-tokens 8000 --model Qwen/Qwen2.5-7B-Instruct
    python debug_capture.py --n-tokens 500 --prompt "Explain quantum computing in detail."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pruning_bench_utils import _capture_qkv

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
# DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# DEFAULT_PROMPT = (
#     "Solve the following problem step by step, showing all intermediate "
#     "reasoning, calculations, and verification.\n\n"
#     "A research lab is designing a distributed computing cluster. They have "
#     "a budget for 120 machines. Each machine can be configured as a CPU node "
#     "(32 cores, 128 GB RAM, $5000), a GPU node (8 cores, 64 GB RAM, 4×A100 "
#     "GPUs, $35000), or a storage node (8 cores, 256 GB RAM, 100 TB disk, "
#     "$12000). Determine the optimal allocation of the 120 machines across all "
#     "three node types for a mixed training, data processing, and inference "
#     "workload. Show all math and reasoning."
# )
DEFAULT_PROMPT = (
    "Solve the following problem step by step, showing all intermediate "
    "reasoning, calculations, and verification.\n\n"
    "A research lab is designing a distributed computing cluster. They have "
    "a budget for 120 machines. Each machine can be configured as a CPU node "
    "(32 cores, 128 GB RAM, $5000), a GPU node (8 cores, 64 GB RAM, 4×A100 "
    "GPUs, $35000), or a storage node (8 cores, 256 GB RAM, 100 TB disk, "
    "$12000). The workload consists of three phases that repeat in a cycle:\n\n"
    "Phase 1 (Training): Requires at least 200 A100 GPUs running in parallel. "
    "Each training job needs 4 GPUs and 48 GB RAM. Communication overhead "
    "between nodes adds 12% latency per additional node beyond the first. "
    "Calculate the optimal GPU node count to minimize total training time for "
    "a 500-epoch run where each epoch takes 45 minutes on a single 4-GPU node.\n\n"
    "Phase 2 (Data Processing): Must process 50 PB of raw data. Each CPU core "
    "can process 2 TB/hour. Storage nodes can serve data at 20 GB/s each but "
    "need 3 replicas for fault tolerance. Calculate the minimum storage and "
    "CPU nodes needed to finish processing within 72 hours.\n\n"
    "Phase 3 (Inference): Must serve 10,000 requests/second with p99 latency "
    "under 100ms. Each GPU can handle 150 requests/second. Each CPU core can "
    "handle 8 requests/second as fallback. The system must maintain 99.99% "
    "uptime, requiring N+2 redundancy.\n\n"
    "Determine the optimal allocation of the 120 machines across all three "
    "node types. Then analyze: What happens if the budget increases by 20%? "
    "What if training data doubles? What if inference load triples? For each "
    "scenario, re-derive the full allocation from scratch, show the math, "
    "compare trade-offs, and explain your reasoning at every step. Finally, "
    "prove mathematically that your allocation is Pareto-optimal across the "
    "three phases, or explain why no single allocation can be."
)


def _default_output_qkv_path(model_name: str, n_tokens: int) -> Path:
    safe_model_name = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in model_name
    )
    return Path(f"capture_qkv_{n_tokens}_{safe_model_name}.pt")


def main():
    parser = argparse.ArgumentParser(description="Debug QKV capture with token display")
    parser.add_argument(
        "--n-tokens", type=int, default=200, help="Number of tokens to generate"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF model id")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt text")
    parser.add_argument(
        "--prompt-file", type=Path, default=None, help="Read prompt from file"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--output-qkv",
        type=Path,
        default=None,
        help="Where to save the captured QKV tensors. Defaults to capture_qkv_<n_tokens>_<model>.pt",
    )
    args = parser.parse_args()

    if args.prompt_file is not None:
        prompt_text = args.prompt_file.read_text()
    elif args.prompt is not None:
        prompt_text = args.prompt
    else:
        prompt_text = DEFAULT_PROMPT

    output_qkv = args.output_qkv or _default_output_qkv_path(args.model, args.n_tokens)
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    print(f"Model:  {args.model}")
    print(f"Tokens: {args.n_tokens}")
    print(f"Device: {args.device}, dtype: {args.dtype}")
    print(f"Prompt: {prompt_text[:120]}...")

    capture = _capture_qkv(
        model_name=args.model,
        prompt_text=prompt_text,
        n=args.n_tokens,
        device=args.device,
        torch_dtype=torch_dtype,
        show_progress=True,
        show_tokens=True,
    )

    print(f"Prompt length:    {capture.prompt_length}")
    print(f"Generated tokens: {capture.generated_token_count()}")
    print(f"Layers captured:  {capture.layer_ids()}")

    capture.save(output_qkv)
    print(f"Saved capture:    {output_qkv}")

    # Print shapes for first layer as a sanity check
    layer0 = capture.layer_ids()[0]
    queries, keys, values = capture.to_layer_tensors(layer0)
    print(f"\nLayer {layer0} shapes:")
    print(f"  queries: {queries.shape}")
    print(f"  keys:    {keys.shape}")
    if values is not None:
        print(f"  values:  {values.shape}")


if __name__ == "__main__":
    main()
