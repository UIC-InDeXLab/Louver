#!/usr/bin/env python3
"""Stream the LLM output for the benchmark prompt to see what it generates."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

PROMPT = (
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=5000, help="Max tokens to generate")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda", torch_dtype=torch.float16
    )
    model.eval()

    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"Prompt tokens: {inputs.input_ids.shape[1]}")
    print("=" * 60)

    generated = inputs.input_ids
    past = None
    with torch.no_grad():
        for i in range(args.n):
            out = model(
                input_ids=generated[:, -1:] if past else generated,
                use_cache=True,
                past_key_values=past,
            )
            past = out.past_key_values
            tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, tok], dim=1)

            word = tokenizer.decode(tok[0], skip_special_tokens=False)
            print(word, end="", flush=True)

    print(f"\n{'=' * 60}")
    print(f"Generated {generated.shape[1] - inputs.input_ids.shape[1]} tokens total")


if __name__ == "__main__":
    main()