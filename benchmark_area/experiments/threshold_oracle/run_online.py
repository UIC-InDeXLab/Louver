"""
run_online.py — Run model on-the-fly, compute threshold oracle stats per decode step.

No full KV capture saved. Hooks into attention layers to grab Q and K at each
decode step, computes cov50_weight + oracle taus on the fly, saves timeseries CSV.

Uses same prompt as score_distribution/capture.py.

Usage:
    python run_online.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --max-new-tokens 2000 \
        --output-dir results/
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE  = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "fixed_k_chal"))

from threshold import LouverThreshold
from bench    import ORACLES, ORACLE_NAMES, _compute_tau

RESULTS_DIR = _HERE / "results"

REASONING_PROMPT = (Path(_HERE).parents[1]
                    / "experiments" / "score_distribution" / "capture.py"
                    ).read_text().split('REASONING_PROMPT = """')[1].split('"""')[0]


# ── Rope helper (same as ObserveAttentionHelper) ──────────────────────────────

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _apply_rope(module, q, k, position_ids, hidden_states):
    if not hasattr(module, "rotary_emb"):
        return q, k
    cos, sin = module.rotary_emb(k, position_ids)
    if cos.dim() == 2: cos = cos.unsqueeze(0); sin = sin.unsqueeze(0)
    if cos.dim() == 4 and cos.shape[1] == 1:
        cos = cos.permute(0, 2, 1, 3); sin = sin.permute(0, 2, 1, 3)
    if cos.dim() == 3: cos = cos.unsqueeze(2); sin = sin.unsqueeze(2)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


# ── Online stats collector ────────────────────────────────────────────────────

class OnlineStatsCollector:
    """
    Registers forward_pre_hooks on selected attention layers.
    Per decode step: computes cov50_weight_mean + oracle tau means across layers.
    """

    MAX_N = 4096  # pre-alloc limit; resize if needed

    def __init__(self, model, layer_indices: list[int], sample_size: int = 256):
        self.model         = model
        self.layer_indices = layer_indices
        self.sample_size   = sample_size
        self.timeseries: list[dict] = []

        # Per-layer state (allocated on first prefill)
        self._H_kv:   dict[int, int] = {}
        self._H_q:    dict[int, int] = {}
        self._D:      dict[int, int] = {}
        self._N:      dict[int, int] = {}       # keys stored so far

        self._keys:   dict[int, torch.Tensor] = {}   # (H_kv, MAX_N, D) float32
        self._reservoir: dict[int, torch.Tensor] = {} # (H_kv, M, D) float16
        self._filled: dict[int, int] = {}
        self._cur_q:  dict[int, torch.Tensor] = {}   # (H_q, D) float32 for this step

        self._prompt_len  = 0
        self._decode_step = 0
        self._layers_done: set[int] = set()
        self._hooks: list = []

    def register_hooks(self):
        for li in self.layer_indices:
            layer = self.model.model.layers[li]
            h = layer.self_attn.register_forward_pre_hook(
                self._make_hook(li), with_kwargs=True
            )
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Hook ─────────────────────────────────────────────────────────────────

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None:
                return
            bsz, seq_len, _ = hidden.shape
            position_ids = kwargs.get("position_ids") if isinstance(kwargs, dict) else None

            with torch.no_grad():
                q_raw = module.q_proj(hidden)
                k_raw = module.k_proj(hidden)

            H_q  = module.config.num_attention_heads
            H_kv = module.config.num_key_value_heads
            D    = module.head_dim

            q_raw = q_raw.view(bsz, seq_len, H_q,  D)
            k_raw = k_raw.view(bsz, seq_len, H_kv, D)
            q_raw, k_raw = _apply_rope(module, q_raw, k_raw, position_ids, hidden)

            # (seq_len, H_*, D)  →  (H_*, seq_len, D)  on CPU float32
            q_cpu = q_raw[0].permute(1, 0, 2).float().cpu()   # (H_q, seq_len, D)
            k_cpu = k_raw[0].permute(1, 0, 2).float().cpu()   # (H_kv, seq_len, D)

            if seq_len > 1:
                # ── Prefill ───────────────────────────────────────────────
                self._prompt_len = seq_len
                self._H_q[layer_idx]  = H_q
                self._H_kv[layer_idx] = H_kv
                self._D[layer_idx]    = D

                max_n = max(self.MAX_N, seq_len + 2048)
                self._keys[layer_idx] = torch.zeros(H_kv, max_n, D, dtype=torch.float32)
                self._keys[layer_idx][:, :seq_len, :] = k_cpu
                self._N[layer_idx] = seq_len

                # Init reservoir
                M = min(self.sample_size, seq_len)
                idx = torch.randperm(seq_len)[:M]
                self._reservoir[layer_idx] = k_cpu[:, idx, :].half()  # (H_kv, M, D)
                self._filled[layer_idx] = M

            else:
                # ── Decode step ───────────────────────────────────────────
                self._cur_q[layer_idx] = q_cpu[:, 0, :]  # (H_q, D)

                n = self._N.get(layer_idx, 0)
                # Grow buffer if needed
                if n >= self._keys[layer_idx].shape[1]:
                    ext = torch.zeros(H_kv, n, D, dtype=torch.float32)
                    self._keys[layer_idx] = torch.cat(
                        [self._keys[layer_idx], ext], dim=1
                    )
                self._keys[layer_idx][:, n, :] = k_cpu[:, 0, :]
                self._N[layer_idx] = n + 1

                # Reservoir update (streaming)
                total = self._prompt_len + self._decode_step + 1
                filled = self._filled[layer_idx]
                if filled < self.sample_size:
                    self._reservoir[layer_idx][:, filled, :] = k_cpu[:, 0, :].half()
                    self._filled[layer_idx] += 1
                else:
                    j = int(torch.randint(0, total, (1,)).item())
                    if j < self.sample_size:
                        self._reservoir[layer_idx][:, j, :] = k_cpu[:, 0, :].half()

                self._layers_done.add(layer_idx)
                if len(self._layers_done) == len(self.layer_indices):
                    self._compute_step_stats()
                    self._layers_done.clear()
                    self._decode_step += 1

        return hook_fn

    # ── Per-step stats ────────────────────────────────────────────────────────

    def _compute_step_stats(self):
        cov50_vals = []
        tau_accum: dict[str, list] = {n: [] for n in ORACLE_NAMES}

        for li in self.layer_indices:
            q    = self._cur_q.get(li)
            N    = self._N.get(li, 0)
            if q is None or N == 0:
                continue

            H_kv = self._H_kv[li]
            H_q  = q.shape[0]
            D    = q.shape[1]
            g    = H_q // H_kv
            keys = self._keys[li][:, :N, :]  # (H_kv, N, D) float32

            scale = D ** -0.5

            # Exact scores (H_q, N)
            exact = torch.empty(H_q, N)
            for h in range(H_q):
                exact[h] = (keys[h // g] @ q[h]) * scale

            # cov50_weight (mean across heads)
            probs   = torch.softmax(exact, dim=-1)
            sorted_p = probs.sort(dim=-1, descending=True).values
            k50     = (sorted_p.cumsum(-1) < 0.5).sum(-1).float() + 1
            cov50_vals.append(float((k50 / N).mean().item()))

            # Oracle taus
            M       = self._filled[li]
            samp_f16 = self._reservoir[li][:, :M, :]
            q_f16    = q.half()
            for name, kw in ORACLES:
                tau = _compute_tau(samp_f16, q_f16, self.sample_size, kw)
                tau_accum[name].append(float(tau.mean().item()))

        N_total = self._prompt_len + self._decode_step + 1
        row: dict = {
            "step":              self._decode_step,
            "N_total":           N_total,
            "cov50_weight_mean": float(np.mean(cov50_vals)) if cov50_vals else float("nan"),
        }
        for name in ORACLE_NAMES:
            vals = tau_accum[name]
            row[f"tau_{name}_mean"] = float(np.mean(vals)) if vals else float("nan")

        self.timeseries.append(row)

    # ── Save ─────────────────────────────────────────────────────────────────

    def save_csv(self, path: Path):
        if not self.timeseries:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(self.timeseries[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in self.timeseries:
                w.writerow({
                    k: f"{v:.8f}" if isinstance(v, float) else v
                    for k, v in row.items()
                })
        print(f"  → {path}  ({len(self.timeseries)} steps)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=2000)
    ap.add_argument("--sample-size",    type=int, default=256)
    ap.add_argument("--output-dir",     default=str(RESULTS_DIR))
    ap.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype",          default="bfloat16")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"Loading {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device, trust_remote_code=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    # Top-25% of layers (same as score_distribution/tail_metrics.py)
    start = (n_layers * 3) // 4
    layer_indices = list(range(start, n_layers))
    print(f"Capturing layers {layer_indices}  ({len(layer_indices)} layers)")

    collector = OnlineStatsCollector(model, layer_indices, args.sample_size)
    collector.register_hooks()

    inputs = tokenizer(REASONING_PROMPT, return_tensors="pt").to(args.device)
    print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")
    print(f"Generating up to {args.max_new_tokens} tokens …")

    pbar = tqdm(total=args.max_new_tokens, desc="generating", unit="tok", dynamic_ncols=True)

    original_compute = collector._compute_step_stats
    def _compute_with_pbar():
        original_compute()
        pbar.update(1)
    collector._compute_step_stats = _compute_with_pbar

    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
        )

    pbar.close()

    collector.remove_hooks()
    print(f"Collected {len(collector.timeseries)} decode steps")

    model_tag = args.model.replace("/", "_")
    ts_path = output_dir / f"{model_tag}_online_timeseries.csv"
    collector.save_csv(ts_path)

    # Plot immediately
    import subprocess
    subprocess.run([
        sys.executable, str(_HERE / "plot_oracle_oscillation.py"),
        "--results-dir", str(output_dir),
        "--out-dir",     str(output_dir / "figs"),
    ], check=False)


if __name__ == "__main__":
    main()
