"""Knowledge-free smoothing controls for NS4 (emob.smooth), on the saved test logits of B2 runs.

Decision 2026-11-21. Only the activity (L2) is smoothed; the other heads stay arg-max, so the rule
violations are computed as for B2 with the smoothed activity. Models (run ids TCN-<model>_..., read
by emob.summarize like any run):
  SMV    viterbi_switch with beta matched per recording and configuration to the share of L2 steps
         that NS4 changed on the same B2 run (needs the NS4 output in --ns4)
  SMM05, SMM10, SMM20   mode filter of the arg-max over 0.5, 1, 2 s (5, 11, 21 steps), fixed a priori

    python -m emob.run_smooth --results ~/emob/results/v2 --ns4 ~/emob/results/v2_ns4 \
        --out ~/emob/results/v2_smooth --workers 12
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
from emob.decode import log_softmax
from emob.evaluate import decide, evaluate
from emob.run_ns4 import git_commit
from emob.smooth import match_budget, mode_filter

B2_RUN = re.compile(r"^(TCN|LSTM)-B2_frac[\d.]+_fold\d+_seed\d+$")
MODE_WIDTHS = {"SMM05": 5, "SMM10": 11, "SMM20": 21}


def run_one(args: tuple) -> str:
    src, ns4_root, out_root = args
    m = json.loads((src / "metrics.json").read_text(encoding="utf-8"))
    bb, rest = src.name.split("-", 1)
    tail = rest.split("_", 1)[1]
    ns4_file = ns4_root / f"{bb}-NS4_{tail}" / "metrics.json"
    if not ns4_file.exists():
        return f"NONS4 {src.name}"
    ns4 = json.loads(ns4_file.read_text(encoding="utf-8"))
    lg = np.load(src / "test_logits.npz")
    gt = {h: v.astype(np.int64) for h, v in targets(load_npz(m["fold"]["test"]), m["args"].get("s_variant", "s")).items()}
    msgs = []
    for model in ["SMV", *MODE_WIDTHS]:
        rid = f"{bb}-{model}_{tail}"
        out = out_root / rid
        if (out / "metrics.json").exists():
            continue
        t0 = time.time()
        test, info = {}, {}
        for cfg in m["test"]:
            logits = {h: lg[f"{cfg}_{h}"].astype(np.float64) for h in C.HEADS if f"{cfg}_{h}" in lg.files}
            pred = decide(logits)
            am = pred["l2"]
            if model == "SMV":
                target = ns4["changed_vs_argmax"][cfg]["l2"]
                lab, beta, share = match_budget(log_softmax(logits["l2"]), target)
                info[cfg] = {"beta": beta, "changed": share, "ns4_changed": target}
            else:
                lab = mode_filter(am, MODE_WIDTHS[model], len(C.L2_CLASSES))
                info[cfg] = {"width_steps": MODE_WIDTHS[model], "changed": float((lab != am).mean())}
            pred["l2"] = lab
            test[cfg] = evaluate(pred, gt)
        out.mkdir(parents=True, exist_ok=True)
        res = {"run_id": rid, "source_run": src.name, "smoothing": model, "git_commit": git_commit(),
               "args": m["args"], "fold": m["fold"], "heads": m["heads"], "val_l2_macro_f1": m["val_l2_macro_f1"],
               "train_time_s": None, "best_epoch": m.get("best_epoch"), "smooth_time_s": time.time() - t0,
               "smoothing_info": info, "test": test,
               "segments_rescored": {"convention": "per supervised interval, pooled (2026-11-21)"}}
        (out / "metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        msgs.append(f"{model} " + ", ".join(f"{c} {t['l2_macro_f1']:.3f}" for c, t in test.items()))
    return f"OK {src.name}: " + "; ".join(msgs) if msgs else f"SKIP {src.name}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--ns4", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    res = Path(a.results).expanduser()
    js = [(d, Path(a.ns4).expanduser(), Path(a.out).expanduser()) for d in sorted(res.iterdir())
          if B2_RUN.match(d.name) and (d / "test_logits.npz").exists()]
    print(f"{len(js)} B2 runs", flush=True)
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(run_one, js):
            print(msg, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
