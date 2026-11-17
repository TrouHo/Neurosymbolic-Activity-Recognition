"""Run a grid of experiments in parallel worker processes (skips runs that already have metrics.json).

    python -m emob.run_grid --phase 1 --workers 6 --out ~/emob/results/phase1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def phase1_jobs(inputs: str) -> list[list[str]]:
    jobs = []
    for frac in (1.0, 0.25):
        for seed in (0, 1, 2):
            for fold in range(8):
                common = ["--fold", str(fold), "--seed", str(seed), "--fraction", str(frac), "--inputs", inputs]
                jobs.append(["emob.baselines.rf", *common])
                jobs.append(["emob.train", "--model", "B1", *common])
                jobs.append(["emob.train", "--model", "B1+S", *common])
                if frac == 1.0:                          # one comparison: S trained on the masked variant
                    jobs.append(["emob.train", "--model", "B1+S", "--s-variant", "s", *common])
    return jobs


def run_name(job: list[str]) -> str:
    args = job[1:]
    get = lambda k, d=None: args[args.index(k) + 1] if k in args else d
    model = "B0" if job[0] == "emob.baselines.rf" else get("--model")
    return (f"{model}_{get('--inputs', 'MP')}_frac{float(get('--fraction', 1.0)):g}_"
            f"{get('--s-variant', 's_all')}_fold{get('--fold')}_seed{get('--seed', 0)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--inputs", default="MP")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    jobs = [j for j in phase1_jobs(a.inputs) if not (out / run_name(j) / "metrics.json").exists()]
    print(f"{len(jobs)} runs to do", flush=True)
    logdir = out / "_logs"
    logdir.mkdir(parents=True, exist_ok=True)

    def run(job):
        name = run_name(job)
        with open(logdir / f"{name}.log", "w") as log:
            r = subprocess.run([sys.executable, "-m", job[0], *job[1:], "--out", str(out)],
                               stdout=log, stderr=subprocess.STDOUT)
        print(f"{'OK ' if r.returncode == 0 else 'ERR'} {name}", flush=True)
        return r.returncode

    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(run, jobs))
    print(f"done: {codes.count(0)} ok, {len(codes) - codes.count(0)} failed", flush=True)


if __name__ == "__main__":
    main()
