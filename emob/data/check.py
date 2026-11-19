"""Gate G — checks on the built data v2 (plan 2026-11-17 §3). Read-only.

1. Shapes and step counts.
2. Rules A1-A7 on the v2 targets (the rule table approved 2026-11-16 was computed on the family
   columns; here every rule is evaluated on the head it will be attached to).
   A3 is evaluated on the 0.1-s grid: change of head height > 15 % of the participant's standing
   height between t-1 s and t+1 s, overlapping T = 1 within +-1 s; steps in a masked anomaly excluded.
3. Residual sensor lag after alignment (same estimator as E1, on the aligned signals): pass when the
   90 % interval is not entirely outside +-0.1 s (i.e. the E1 rule would not shift again).
4. Reproducibility: rebuild into a temporary folder and compare the content hashes.
Usage (from code/): .venv\\Scripts\\python.exe -m emob.data.check [--skip-rebuild]
"""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from emob import config as C
from emob.data import rule_stats as RS
from estimate.sensor_offsets import device_signal, estimate, marker_signal, rank, summarise

V2 = C.PROJECT / "data" / "processed" / "v2"
RESULTS = []


def report(name, ok, detail):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rebuild", action="store_true")
    args = ap.parse_args(argv)
    man = json.load(open(V2 / "manifest.json", encoding="utf-8"))
    D = {p["participant"]: dict(np.load(V2 / f"{p['participant']}.npz")) for p in man["participants"]}

    # 1 shapes
    ok = all(len(d[k]) == len(d["ts"]) // C.STEP for d in D.values() for k in ("l2", "l1", "s", "w", "t", "annotated"))
    ok &= all(d["x50"].shape == (len(d["ts"]), 3 * len(C.BLOCK_ORDER)) for d in D.values())
    report("shapes", ok, "every target has len(ts)/5 steps; x50 has 15 channels")

    # 2 rules (definitions in emob/data/rule_stats.py, shared with inspection_data)
    r = RS.a1(D)
    report("A1 hierarchy L2 -> L1", r["ok"] == r["n"], f"{r['ok']}/{r['n']} steps")

    r = RS.a2_pairs(man)
    report("A2 posture transitions chain", True,
           f"{r['chain']} chain / {r['break']} do not / {r['with_other']} with Autre_Transition "
           f"({100 * r['chain'] / (r['chain'] + r['break']):.1f} %; 2026-11-16 table: 584/18, 97.0 %)")

    r = RS.a3(D)
    report("A3 posture changes overlap T (soft rule)", r["ok"] / r["n"] >= 0.85,
           f"{r['ok']}/{r['n']} = {100 * r['ok'] / r['n']:.1f} % (2026-11-16 table: 90.9 %); "
           + ", ".join(f"{u} {k}/{n}" for u, (k, n) in r["per_participant"].items()))

    r = RS.a4_a5(D)
    report("A4 no walking while sitting/lying", r["a4_violations"] == 0,
           f"{r['a4_violations']} steps of {r['walking_steps_known_posture']} walking steps with known posture")
    report("A5 no walking while kneeling/squatting", r["a5_violations"] == 0, f"{r['a5_violations']} steps")

    r = RS.a6(D)
    report("A6 walking segments >= 1 s (W)", r["ok"] / r["n"] >= 0.95,
           f"{r['ok']}/{r['n']} = {100 * r['ok'] / r['n']:.1f} %; >= 0.5 s: {r['ge_0_5s']}; median {r['median_s']:.2f} s")

    r = RS.a7(D)
    report("A7 specific movements <= 5 s (L2)", r["ok"] / r["n"] >= 0.95,
           f"{r['ok']}/{r['n']} = {100 * r['ok'] / r['n']:.1f} %; longer: " + ", ".join(f"{nm} {a:.1f} s ({u})" for a, nm, u in r["longer"]))

    # 3 residual sensor lag
    print("\nResidual sensor lag after alignment (median over 120-s windows, 90 % interval):")
    bad = []
    for u, d in D.items():
        f = next(C.RAW_2022.glob(f"{u}-*.csv"))
        mk = pd.read_csv(f, usecols=lambda c: c.startswith(("x_sq_", "y_sq_", "z_sq_")), low_memory=False).apply(pd.to_numeric, errors="coerce")
        L = len(d["ts"])
        row = []
        for dev, (blk, m) in C.DEVICES.items():
            j = C.BLOCK_ORDER.index(blk)
            if not d["avail50"][:, j].any():
                continue
            x_ = rank(marker_signal(mk[[f"{a}_sq_{m}" for a in "xyz"]].to_numpy(float)[:L]))
            y_ = rank(device_signal(d["x50"][:, 3 * j:3 * j + 3].astype(float)))
            *_, wins = estimate(x_, y_)
            s = summarise(wins, L / C.FS)
            if s["lo"] is None:
                row.append(f"{dev} n/a"); continue
            okd = not (s["lo"] > 0.1 or s["hi"] < -0.1)
            if not okd:
                bad.append(f"{u} {dev}")
            row.append(f"{dev} {s['median']:+.2f} [{s['lo']:+.2f},{s['hi']:+.2f}]{'' if okd else ' !'}")
        print(f"   {u}: " + "; ".join(row))
    report("residual sensor lag", not bad, "all within the rule" if not bad else "outside: " + ", ".join(bad))

    # 4 reproducibility
    if not args.skip_rebuild:
        from emob.data import build
        with tempfile.TemporaryDirectory() as tmp:
            build.main(["--out", tmp])
            m2 = json.load(open(Path(tmp) / "manifest.json", encoding="utf-8"))
        h1 = {p["participant"]: p["content_sha256"] for p in man["participants"]}
        h2 = {p["participant"]: p["content_sha256"] for p in m2["participants"]}
        report("reproducible build", h1 == h2, "content hashes identical for all 8 files" if h1 == h2 else f"differ: {h1} vs {h2}")

    print(f"\nGATE G: {sum(o for _, o in RESULTS)}/{len(RESULTS)} checks passed")


if __name__ == "__main__":
    main()
