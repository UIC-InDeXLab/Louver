"""Run exp1/2/3 across multiple models. Tags reports with model short name."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
REP = ROOT / "reports"


def short(name): return name.split("/")[-1].replace(".", "_")


def run_one(model, trials, py):
    env = os.environ.copy()
    env["EXP_MODEL"] = model
    tag = short(model)
    REP.mkdir(exist_ok=True)
    for script, args in [
        ("exp1_relevant_vs_irrelevant.py", ["--trials", str(trials)]),
        ("exp2_fixed_k.py", ["--trials", str(trials)]),
        ("exp3_variable_list.py", ["--trials", str(trials)]),
        ("exp4_fixed_sum_threshold.py", ["--trials", str(trials)]),
    ]:
        print(f"=== {tag}: {script} ===", flush=True)
        subprocess.run([py, str(ROOT / script), *args], env=env, check=True)
        # rename CSVs to keep per-model copies
        base = script.replace(".py", "").replace("exp", "exp")
        src = REP / f"{base}.csv"
        dst = REP / f"{base}__{tag}.csv"
        shutil.copy(src, dst)
        print(f"  saved {dst.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    for m in args.models:
        run_one(m, args.trials, args.python)
