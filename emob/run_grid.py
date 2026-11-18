"""Run the experiment grid phase by phase in parallel worker processes.

Runs that already have metrics.json are skipped, so an interrupted grid can simply be restarted.
Phases (Explanation.md §1; experiment plan §4; TCN first, DeepConvLSTM after gate G1):
  1   B0 (Random Forest), B1, B2                          8 folds x 3 seeds
  2a  NS1, lam_hier in GRID                               -> lam_hier selected per fold on validation
  2b  NS2, lam_hier = selected for NS1 in that fold,
      (lam_hard, lam_soft) in GRID_RULES^2 with lam_soft <= lam_hard; plus the fixed set (fairness check)
  3   NS3 with the lambdas selected for NS2 in that fold; plus the fixed set
Selection uses emob.summarize.select_lambda (validation participant only).

    python -m emob.run_grid --phase 1 --workers 4 --out ~/emob/results
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GRID = [0.1, 0.3, 1.0]
# lam_hard / lam_soft grid of NS2, extended 2026-11-18 with 3 and 10: the rule losses of A2, A3, A6
# are non-zero only around rare events (posture-transition onsets, short walking runs) but are
# averaged over every step, so lambdas <= 1 moved the arg-max violations only a little
# (diagnostics/rule_loss_train_vs_test.py). lam_hier keeps GRID (A1 acts on every step).
GRID_RULES = [0.1, 0.3, 1.0, 3.0, 10.0]
SEEDS = [0, 1, 2]
FOLDS = range(8)


def ns_fixed():
    from emob.summarize import FIXED
    return FIXED


def common(fold, seed, frac):
    return ["--fold", str(fold), "--seed", str(seed), "--fraction", f"{frac:g}"]


def jobs_for(phase: str, backbones: list[str], fractions: list[float], out: Path) -> list[list[str]]:
    jobs = []
    rows = None
    if phase in ("2b", "3"):
        from emob.summarize import collect
        rows = collect(out)
    from emob.summarize import select_lambda
    for frac in fractions:
        for fold in FOLDS:
            for seed in SEEDS:
                c = common(fold, seed, frac)
                if phase == "1":
                    if "tcn" in backbones:                       # B0 once, independent of the backbone
                        jobs.append(["emob.baselines.rf", *c])
                    for bb in backbones:
                        jobs += [["emob.train", "--model", m, "--backbone", bb, *c] for m in ("B1", "B2")]
                elif phase == "2a":
                    for bb in backbones:
                        jobs += [["emob.train", "--model", "NS1", "--backbone", bb, *c, "--lam-hier", f"{h:g}"] for h in GRID]
                elif phase in ("2b", "3"):
                    for bb in backbones:
                        name = {"tcn": "TCN", "lstm": "LSTM"}[bb]
                        model = "NS2" if phase == "2b" else "NS3"
                        fixed = ns_fixed()[model]
                        lam_sets = [fixed]
                        if phase == "2b":
                            sel = select_lambda(rows, name, "NS1", fold, frac)
                            if sel is None:
                                print(f"skip NS2 {name} fold {fold} frac {frac:g}: NS1 not finished", flush=True)
                            else:
                                lam_sets += [(sel[0], d, s) for d in GRID_RULES for s in GRID_RULES if s <= d]
                        else:
                            sel = select_lambda(rows, name, "NS2", fold, frac)
                            if sel is None:
                                print(f"skip NS3 {name} fold {fold} frac {frac:g}: NS2 not finished", flush=True)
                            else:
                                lam_sets.append(sel)
                        for h, d, s in dict.fromkeys(lam_sets):
                            jobs.append(["emob.train", "--model", model, "--backbone", bb, *c,
                                         "--lam-hier", f"{h:g}", "--lam-hard", f"{d:g}", "--lam-soft", f"{s:g}"])
                else:
                    raise ValueError(phase)
    return jobs


def run_name(job: list[str]) -> str:
    if job[0] == "emob.baselines.rf":
        args = job[1:]
        get = lambda k: args[args.index(k) + 1]
        return f"RF-B0_frac{float(get('--fraction')):g}_fold{get('--fold')}_seed{get('--seed')}"
    from emob.train import parse, run_id
    return run_id(parse(job[1:]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "2a", "2b", "3"], required=True)
    ap.add_argument("--backbones", nargs="+", choices=["tcn", "lstm"], default=["tcn"])
    ap.add_argument("--fractions", nargs="+", type=float, default=[1.0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="list the jobs without running them")
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    jobs = [j for j in jobs_for(a.phase, a.backbones, a.fractions, out)
            if not (out / run_name(j) / "metrics.json").exists()]
    print(f"{len(jobs)} runs to do", flush=True)
    if a.dry_run:
        for j in jobs:
            print(run_name(j))
        return
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
