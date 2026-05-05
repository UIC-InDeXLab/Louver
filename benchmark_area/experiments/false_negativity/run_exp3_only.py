"""Run only exp3 across multiple models."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
REP = ROOT / "reports"


def short(name): return name.split("/")[-1].replace(".", "_")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    REP.mkdir(exist_ok=True)
    for m in args.models:
        env = os.environ.copy()
        env["EXP_MODEL"] = m
        tag = short(m)
        print(f"=== {tag} ===", flush=True)
        subprocess.run(
            [args.python, str(ROOT / "exp3_variable_list.py"),
             "--trials", str(args.trials)],
            env=env, check=True,
        )
        src = REP / "exp3_variable_list.csv"
        dst = REP / f"exp3_variable_list__{tag}.csv"
        shutil.copy(src, dst)
        print(f"  saved {dst.name}")
