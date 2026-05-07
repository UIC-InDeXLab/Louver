import argparse
import random

import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

PROMPT = """
You are an expert systems thinker, research scientist, and software architect. I want you to produce a deep, structured, and carefully reasoned analysis of the following multi-layered scenario. You must organize your response with clearly labeled sections, bullet points where appropriate, and concise but dense explanations. Avoid fluff. Prioritize clarity, depth, and structured thinking.

SCENARIO:

You are tasked with designing a large-scale AI-assisted knowledge platform for researchers in computational science. The platform must support the following capabilities:

1. Semantic search across millions of academic papers.
2. Real-time conversational querying over indexed documents.
3. Automatic summarization at multiple abstraction levels (short, medium, detailed).
4. Cross-paper reasoning (detecting contradictions, agreements, and novel connections).
5. Citation-grounded answer generation.
6. Adaptive memory that stores user-specific research interests.
7. Efficient retrieval over vector embeddings with low latency.
8. Distributed deployment across multiple data centers.
9. Strong privacy guarantees for proprietary documents.
10. Continuous model improvement without catastrophic forgetting.

CONSTRAINTS:

- The system must handle 10 million documents.
- Average document length is 8,000 tokens.
- Users expect sub-second response latency for search.
- Conversational responses must remain under 2,000 generated tokens.
- The system must operate under realistic GPU budget constraints.
- Some documents cannot leave specific geographic regions due to regulation.
- You must assume adversarial users may attempt prompt injection attacks.
- Memory storage must scale to 1 million concurrent active users.

TASK:

Provide a structured answer covering the following sections:

SECTION 1: High-Level Architecture
- Propose the system architecture.
- Separate components (indexing, retrieval, LLM layer, memory layer, caching).
- Explain how data flows through the system.

SECTION 2: Embedding & Retrieval Strategy
- Discuss embedding dimensionality tradeoffs.
- ANN indexing methods (HNSW, IVF, PQ).
- Hybrid search (sparse + dense).
- Handling freshness and document updates.
- Tradeoffs between recall and latency.

SECTION 3: Conversational Reasoning Layer
- How would you structure prompts?
- How do you prevent hallucinations?
- How do you enforce citation grounding?
- How do you handle multi-hop reasoning across documents?

SECTION 4: Summarization Design
- Hierarchical summarization strategy.
- Chunking methods.
- Recursive abstraction.
- Quality evaluation methods.
- Avoiding information loss.

SECTION 5: Distributed Systems Considerations
- Sharding strategy.
- Geo-partitioning.
- Caching layers.
- Fault tolerance.
- Model serving scaling strategy.

REQUIREMENTS FOR YOUR ANSWER:

- Be structured and labeled clearly.
- Use bullet points where helpful.
- Provide technical depth.
- Avoid generic textbook explanations.
- Prioritize engineering realism.
- Be concise but information-dense.
- Include tradeoff analysis throughout.
- Do not repeat the prompt.
- Do not include disclaimers.
- Do not include meta commentary.

Additionally, at the end of your response, provide a brief summary table listing:

Component | Key Challenge | Core Tradeoff | Proposed Solution

Your final answer should resemble a high-quality technical design document suitable for internal review at a major AI research lab.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["hira", "baseline", "hira_v2"],
        default="hira",
        help="Select custom hira attention+cache or baseline attention.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-7B-Instruct",
        # default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model id to load from Hugging Face.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Max generated tokens.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--display-mode",
        choices=["stream", "progress"],
        default="stream",
        help="`stream`: print decoded text, `progress`: show generation progress only.",
    )
    return parser.parse_args()


class ProgressBarStreamer:
    def __init__(self, total_tokens):
        self.total_tokens = total_tokens
        self.generated_tokens = 0
        self._seen_prompt = False
        self._bar = None

        if tqdm is not None:
            self._bar = tqdm(total=total_tokens, desc="Generating", unit="tok")

    def put(self, value):
        if not self._seen_prompt:
            self._seen_prompt = True
            return

        new_tokens = int(value.numel())
        remaining = max(self.total_tokens - self.generated_tokens, 0)
        increment = min(new_tokens, remaining)
        self.generated_tokens += increment

        if self._bar is not None:
            self._bar.update(increment)
        else:
            left = max(self.total_tokens - self.generated_tokens, 0)
            print(
                f"\rGenerated {self.generated_tokens}/{self.total_tokens} tokens "
                f"(remaining: {left})",
                end="",
                flush=True,
            )

    def end(self):
        if self._bar is not None:
            self._bar.close()
        else:
            print()


def main():
    args = parse_args()

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    if args.mode == "hira":
        # Side-effect import registers "hira_attention_v1".
        from hira.attention.hira_attention_v1 import (
            hira_attention_v1_forward,
        )  # noqa: F401
        from hira.cache.hira_cache import HiraCache
        from hira.cache.hira_config import DeviceMode, HiraConfig

        attn_impl = "hira_attention_v1"
    elif args.mode == "hira_v2":
        # Side-effect import registers "hira_attention_v2".
        from hira.attention.hira_attention_v2 import (
            hira_attention_v2_forward,
        )  # noqa: F401
        from hira.cache.hira_cache import HiraCache
        from hira.cache.hira_config import DeviceMode, HiraConfig

        attn_impl = "hira_attention_v2"
    else:
        # Side-effect import registers "sdpa_attention_ref".
        from hira.attention.baseline_attn import sdp_attention_ref  # noqa: F401

        attn_impl = "sdpa_attention_ref"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device,
        torch_dtype=torch.float32,
        attn_implementation=attn_impl,
    )
    model.eval()

    if args.display_mode == "stream":
        streamer = TextStreamer(tokenizer)
    else:
        streamer = ProgressBarStreamer(total_tokens=args.max_new_tokens)

    messages = [
        {
            "role": "user",
            "content": PROMPT,
        }
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # text = """
    # What is the capital of France?
    # """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.max_new_tokens,
        do_sample=False,
        streamer=streamer,
        num_beams=1,
    )

    if "hira" in args.mode:
        cache = HiraCache(
            cache_config=model.config,
            hira_config=HiraConfig(
                device_mode=(
                    DeviceMode.CPU_ONLY
                    if "cpu" in args.device
                    else DeviceMode.CUDA_ONLY
                ),
                update_every=512,
                num_levels=2,
                branching_factor=8,
            ),
        )
        generate_kwargs["past_key_values"] = cache

    torch.cuda.synchronize()
    start = time.perf_counter()

    out = model.generate(**generate_kwargs)

    torch.cuda.synchronize()
    end = time.perf_counter()

    print(f"Elapsed time: {end - start:.6f} seconds")

    # print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
