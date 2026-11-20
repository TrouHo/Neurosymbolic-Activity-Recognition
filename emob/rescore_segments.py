"""Re-score the segment metrics of saved runs with the per-interval convention (2026-11-21).

emob.metrics.segment_metrics now scores each interval of consecutive supervised steps separately
(no segment or match across an unknown step). This script recomputes test[cfg]["l2_segments"] of
existing metrics.json files from the saved outputs, without retraining:
  - trained runs (emob.train): arg-max of test_logits.npz;
  - decoded runs (emob.run_ns4: NS4, NS2V, ablations): the source run's logits decoded again with the
    same constraints;
  - runs without saved outputs (RF-B0): segment metrics set to None (not comparable any more).
The old values are kept under "l2_segments_concat". As a check, the old convention is recomputed from
the same predictions and compared with the stored values (mismatches are reported, e.g. float16 ties).
Idempotent: files that already have "l2_segments_concat" are skipped. Files are replaced atomically.

    python -m emob.rescore_segments --results ~/emob/results/v2 ~/emob/results/v2_ns4 \
        --source ~/emob/results/v2 --workers 12
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from emob import config as C
from emob.data.dataset import load_npz, targets
from emob.evaluate import decide
from emob.metrics import segment_metrics, segment_metrics_concat

KEYS = ("edit", "f1@10", "f1@25", "f1@50")


def predictions(f: Path, m: dict, source: Path) -> dict[str, np.ndarray] | None:
    """{config: (T,) L2 predictions}, or None when the run saved no outputs."""
    if "source_run" in m:                                    # decoded run (run_ns4)
        from emob.decode import decode_logits
        lg = np.load(source / m["source_run"] / "test_logits.npz")
        use = tuple(m["constraints"])
        dec = lambda logits: decode_logits(logits, use)["l2"]
    else:
        p = f.parent / "test_logits.npz"
        if not p.exists():
            return None
        lg = np.load(p)
        dec = lambda logits: decide(logits)["l2"]
    out = {}
    for cfg in m["test"]:
        logits = {h: lg[f"{cfg}_{h}"].astype(np.float64) for h in C.HEADS if f"{cfg}_{h}" in lg.files}
        out[cfg] = dec(logits)
    return out


def rescore(args: tuple) -> str:
    f, source = args
    m = json.loads(f.read_text(encoding="utf-8"))
    if "smoothing" in m or all("l2_segments_concat" in t for t in m["test"].values()):
        return f"SKIP {f.parent.name}"                       # smoothing controls: new convention already
    gt = targets(load_npz(m["fold"]["test"]), m["args"].get("s_variant", "s"))["l2"].astype(np.int64)
    preds = predictions(f, m, source)
    diff = 0.0
    for cfg, t in m["test"].items():
        old = t["l2_segments"]
        if preds is None:
            new = {k: None for k in KEYS}
        else:
            p = preds[cfg]
            if len(p) != len(gt):
                raise ValueError(f"{f}: {cfg} prediction length {len(p)} != labels {len(gt)}")
            check = segment_metrics_concat(gt, p)
            diff = max(diff, max(abs(check[k] - old[k]) for k in KEYS))
            new = segment_metrics(gt, p)
        t["l2_segments_concat"] = old
        t["l2_segments"] = new
    # Trained runs saved float16 logits: a few near-ties change the arg-max, so the old convention
    # recomputed here can differ slightly from the stored value (computed on float32 logits).
    m["segments_rescored"] = {"convention": "per supervised interval, pooled (2026-11-21)",
                              "concat_check_max_abs_diff": diff, "outputs_saved": preds is not None}
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=1), encoding="utf-8")
    os.replace(tmp, f)
    tag = "NOOUT" if preds is None else ("BIGDIFF" if diff > 2.0 else "OK")
    return f"{tag} {diff:.3f} {f.parent.name}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--source", required=True, help="folder of the trained runs (source of decoded runs)")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    source = Path(a.source).expanduser()
    files = [f for r in a.results for f in sorted(Path(r).expanduser().glob("*/metrics.json"))]
    print(f"{len(files)} metrics files", flush=True)
    counts: dict[str, int] = {}
    diffs = []
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(rescore, [(f, source) for f in files], chunksize=4):
            tag = msg.split()[0]
            counts[tag] = counts.get(tag, 0) + 1
            if tag in ("OK", "BIGDIFF"):
                diffs.append(float(msg.split()[1]))
            if tag not in ("OK", "SKIP"):
                print(msg, flush=True)
    print("counts", counts, flush=True)
    if diffs:
        d = np.array(diffs)
        print(f"old-convention check, max |diff| per file (points): median {np.median(d):.3f}, "
              f"95% {np.quantile(d, 0.95):.3f}, max {d.max():.3f}, exact {np.mean(d < 1e-6):.0%}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
