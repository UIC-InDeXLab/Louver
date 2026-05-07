"""
Capture QKV tensors at a single model layer from one AIME 2024 problem.

Only the target layer is recorded during generation — all other layers are
skipped — so peak CPU memory scales with one layer, not all 48.

Output: captures/<model_slug>_layer<L>_N<gen_count>.pt
        (CaptureState-compatible, loadable by TA_filter_alg/bench.py)

Usage
-----
    python capture_aime.py --model meta-llama/Llama-3.2-3B-Instruct --max-tokens 20000
    python capture_aime.py --model Qwen/Qwen2.5-7B-Instruct          --max-tokens 20000
    python capture_aime.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --max-tokens 32000
    python capture_aime.py --model Qwen/Qwen2.5-14B-Instruct          --max-tokens 32000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import types as _types
_hira = _types.ModuleType('hira')
_hira.__path__ = [str(REPO_ROOT)]
_hira.__package__ = 'hira'
sys.modules['hira'] = _hira

import benchmark_area.quick_pruning.pruning_bench_utils as _pbu
from benchmark_area.quick_pruning.pruning_bench_utils import CaptureState  # noqa: E402

# ── Fallback problem (used when HuggingFace datasets unavailable) ─────────────
_FALLBACK_PROBLEM = (
    "In triangle $ABC$, $AB = 10$, $BC = 12$, and $CA = 14$. "
    "The angle bisector from $A$ meets $BC$ at $D$. "
    "Point $E$ is the midpoint of $AD$. "
    "Line $BE$ extended meets $CA$ at $F$. "
    "Compute $\\frac{CF}{FA}$.\n\n"
    "Please reason step by step, showing all your work, and provide the final answer."
)


def _load_aime_problem(idx: int = 0) -> str:
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        row = ds[idx]
        for key in ("Problem", "problem", "question", "Question"):
            if key in row and isinstance(row[key], str):
                return row[key]
        for v in row.values():
            if isinstance(v, str) and len(v) > 20:
                return v
        raise ValueError(f"Cannot find problem text in row keys: {list(row.keys())}")
    except Exception as exc:
        print(f"[capture_aime] Dataset load failed ({exc}). Using fallback problem.")
        return _FALLBACK_PROBLEM


def _model_slug(name: str) -> str:
    return name.replace("/", "_").replace("-", "_")


def _mid_layer(model_name: str) -> int:
    cfg = AutoConfig.from_pretrained(model_name)
    return cfg.num_hidden_layers // 2


def capture_with_layer_filter(
    model_name: str,
    prompt_text: str,
    n: int,
    target_layers: list[int],
) -> CaptureState:
    """
    Run _capture_qkv but only record the given layers.

    Monkey-patches CaptureState.record so that non-target layers are silently
    discarded before any tensor is stored, keeping CPU RAM low.
    """
    target_set = set(target_layers)
    orig_record = CaptureState.record

    def _filtered_record(self, module, query, key, value):  # type: ignore[override]
        layer_idx = int(getattr(module, "layer_idx", -1))
        if layer_idx in target_set:
            orig_record(self, module, query, key, value)

    CaptureState.record = _filtered_record  # type: ignore[method-assign]
    try:
        cap = _pbu._capture_qkv(
            model_name=model_name,
            prompt_text=prompt_text,
            n=n,
            device="cuda",
            torch_dtype=torch.float16,
            show_progress=True,
        )
    finally:
        CaptureState.record = orig_record  # type: ignore[method-assign]
    return cap


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Capture QKV at one layer from a single AIME 2024 problem.",
    )
    ap.add_argument(
        "--model", default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model ID.",
    )
    ap.add_argument(
        "--layer", type=int, default=None,
        help="Layer index to capture. Default: middle layer of the model.",
    )
    ap.add_argument(
        "--max-tokens", type=int, default=40000,
        help="Maximum number of tokens to generate.",
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "captures",
        help="Directory for the output .pt file.",
    )
    ap.add_argument(
        "--problem-idx", type=int, default=0,
        help="AIME 2024 problem index (0-based).",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for capture.")

    layer = args.layer if args.layer is not None else _mid_layer(args.model)

    print(f"Model       : {args.model}")
    print(f"Layer       : {layer}  (pass --layer N to override)")
    print(f"Max tokens  : {args.max_tokens}")
    print(f"Output dir  : {args.output_dir}")
    print()

    problem = _load_aime_problem(args.problem_idx)
    print(f"Problem [{args.problem_idx}] (first 160 chars):")
    print(" ", problem[:160].replace("\n", " "))
    print()

    cap = capture_with_layer_filter(
        model_name=args.model,
        prompt_text=problem,
        n=args.max_tokens,
        target_layers=[layer],
    )

    gen_count = cap.generated_token_count()
    prefill_n = cap.prefill_keys.get(layer)
    prefill_len = prefill_n.shape[1] if prefill_n is not None else "?"
    print(f"\nCaptured {gen_count} generated tokens  (prefill={prefill_len})  layer={layer}")

    slug = _model_slug(args.model)
    out_path = args.output_dir / f"{slug}_layer{layer}_N{gen_count}.pt"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap.save(out_path)
    size_mb = out_path.stat().st_size // 1024 ** 2
    print(f"Saved → {out_path}  ({size_mb} MB)")


if __name__ == "__main__":
    main()
