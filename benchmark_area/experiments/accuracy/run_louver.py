"""
Unified entry point for Louver accuracy experiments.

Examples:

  # LongBench, TA filter, oracle threshold
  python run_louver.py longbench \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --louver_variant ta --threshold_mode oracle

  # LongBench, full-subspace, 10% budget
  python run_louver.py longbench \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --louver_variant full --threshold_mode budget --budget_fraction 0.1

  # RULER, seq_len=32k, TA filter
  python run_louver.py ruler \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --louver_variant ta --seq_len 32768

  # AIME on DeepSeek-R1
  python run_louver.py aime \
      --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
      --louver_variant ta --max_new_tokens 8192

  # Dense SDPA baseline (no Louver)
  python run_louver.py longbench \
      --model meta-llama/Llama-3.1-8B-Instruct --method dense_sdpa
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--model", required=True)
    p.add_argument("--method", default="louver_ta",
                   choices=["dense_sdpa", "dense_eager", "louver_full", "louver_ta"])
    p.add_argument("--louver_variant", default="ta", choices=["full", "ta"])
    p.add_argument("--threshold_mode", default="oracle", choices=["oracle", "budget"])
    p.add_argument("--oracle", default="sample_max", choices=["sample_max", "sample_mean_max"])
    p.add_argument("--budget_fraction", type=float, default=0.1)
    p.add_argument("--sample_size", type=int, default=256)
    p.add_argument("--update_interval", type=int, default=256)
    p.add_argument("--output_dir", default=None)


def main():
    parser = argparse.ArgumentParser(description="Louver accuracy experiments")
    sub = parser.add_subparsers(dest="benchmark", required=True)

    # LongBench
    lb = sub.add_parser("longbench")
    add_common_args(lb)
    lb.add_argument("--tasks", default="hotpotqa,2wikimqa,musique,qasper,narrativeqa,gov_report,trec,triviaqa,passage_retrieval_en")
    lb.add_argument("--max_samples", type=int, default=None)

    # RULER
    rl = sub.add_parser("ruler")
    add_common_args(rl)
    rl.add_argument("--tasks", default="niah_single,niah_multi,vt,cwe")
    rl.add_argument("--seq_len", type=int, default=32768)
    rl.add_argument("--n_samples", type=int, default=50)
    rl.add_argument("--seed", type=int, default=42)

    # AIME
    ai = sub.add_parser("aime")
    add_common_args(ai)
    ai.add_argument("--year", type=int, default=2024)
    ai.add_argument("--max_new_tokens", type=int, default=8192)

    args = parser.parse_args()

    # Patch method to match louver_variant when method not explicitly set
    if args.method in ("louver_full", "louver_ta") and args.louver_variant:
        args.method = f"louver_{args.louver_variant}"

    if args.output_dir is None:
        model_tag = args.model.split("/")[-1]
        args.output_dir = f"results/{args.benchmark}/{model_tag}"

    if args.benchmark == "longbench":
        from eval.longbench import main as run
    elif args.benchmark == "ruler":
        from eval.ruler import main as run
    elif args.benchmark == "aime":
        from eval.aime import main as run

    # Override sys.argv so the sub-module's argparse sees the right flags
    sys.argv = [sys.argv[0]] + [f"--{k}={v}" for k, v in vars(args).items()
                                if k != "benchmark" and v is not None]
    run()


if __name__ == "__main__":
    main()
