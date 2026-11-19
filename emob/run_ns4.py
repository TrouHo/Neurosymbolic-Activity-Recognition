"""Run NS4: constrained Viterbi decoding (emob.decode) on the saved test logits of trained runs.

Reference: Explanation.md §9 (approved 2026-11-19). No training: each source run's test_logits.npz
is decoded under every sensor configuration and evaluated like a trained run, so emob.summarize
reads the outputs unchanged.

  NS4    = B2 logits + decoding with all constraints (main: NS4 - B2)
  NS4 with "no_rule" = leave one constraint out (A2, A45, A6, link), exploratory
  NS2V   = NS2 logits (lambdas selected on validation) + decoding (are training-time and
           decoding-time rules additive?)
The validation score of the source run is copied, so lambda selection is unchanged.

    python -m emob.run_ns4 --results ~/emob/results/v2 --summary ~/emob/results/v2/summary_phase3.json \
        --out ~/emob/results/v2_ns4 --workers 10
"""
from __future__ import annotations

import argparse
import json
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from emob import config as C
from emob.data.dataset import load_npz, targets
from emob.decode import CONSTRAINTS, decode_logits
from emob.evaluate import decide, evaluate

B2_RUN = re.compile(r"^TCN-B2_frac[\d.]+_fold\d+_seed\d+$")


def git_commit() -> str | None:
    """Same as emob.train.git_commit, without importing torch."""
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=C.CODE, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def jobs(results: Path, summary: Path | None, ablations: bool) -> list[tuple]:
    out = []
    for d in sorted(results.iterdir()):
        if B2_RUN.match(d.name) and (d / "test_logits.npz").exists():
            out.append((d, "NS4", ()))
            if ablations:
                out += [(d, "NS4", (c,)) for c in CONSTRAINTS]
    if summary is not None:
        sel = json.loads(summary.read_text(encoding="utf-8"))["selected_lambda"]
        for key, lam in sel.items():
            name, frac, fold = key.split("|")
            if name != "TCN-NS2":
                continue
            lam_s = "-".join(f"{x:g}" for x in lam)
            for seed in range(3):
                d = results / f"TCN-NS2_{frac}_{fold}_seed{seed}_lam{lam_s}"
                if (d / "test_logits.npz").exists():
                    out.append((d, "NS2V", ()))
    return out


def run_one(job: tuple, out_root: str) -> str:
    src, model, drop = job
    m = json.loads((src / "metrics.json").read_text(encoding="utf-8"))
    a = m["args"]
    use = tuple(c for c in CONSTRAINTS if c not in drop)
    rid = src.name.replace("TCN-B2_", f"TCN-{model}_").replace("TCN-NS2_", f"TCN-{model}_")
    if drop:
        rid += "_no" + "".join(drop)
    out = Path(out_root) / rid
    if (out / "metrics.json").exists():
        return f"SKIP {rid}"
    lg = np.load(src / "test_logits.npz")
    gt = {h: v.astype(np.int64) for h, v in targets(load_npz(m["fold"]["test"]), a.get("s_variant", "s")).items()}
    t0 = time.time()
    test, changes = {}, {}
    for cfg in C.EVAL_CONFIGS:
        logits = {h: lg[f"{cfg}_{h}"].astype(np.float64) for h in C.HEADS if f"{cfg}_{h}" in lg.files}
        if not logits:
            continue
        pred = decode_logits(logits, use)
        base = decide(logits)
        test[cfg] = evaluate(pred, gt)
        changes[cfg] = {h: float((pred[h] != base[h]).mean()) for h in ("l2", "s", "w")}
    out.mkdir(parents=True, exist_ok=True)
    res = {"run_id": rid, "source_run": src.name, "git_commit": git_commit(), "constraints": list(use),
           "args": dict(a, no_rule=list(drop)), "fold": m["fold"], "heads": m["heads"],
           "val_l2_macro_f1": m["val_l2_macro_f1"], "train_time_s": None, "best_epoch": m.get("best_epoch"),
           "decode_time_s": time.time() - t0, "changed_vs_argmax": changes, "test": test}
    (out / "metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    ch = ", ".join(f"{c} L2 {v['l2']:.3%}" for c, v in changes.items())
    f1 = ", ".join(f"{c} {t['l2_macro_f1']:.3f}" for c, t in test.items())
    return f"OK {rid}: {f1}; changed {ch}"


def _run(args: tuple) -> str:
    return run_one(*args)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="folder of the trained runs")
    ap.add_argument("--summary", help="summary json with selected_lambda (for NS2V)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    js = jobs(Path(a.results).expanduser(), Path(a.summary).expanduser() if a.summary else None, not a.no_ablations)
    print(f"{len(js)} decodings to do", flush=True)
    out = str(Path(a.out).expanduser())
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(_run, [(j, out) for j in js]):
            print(msg, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
