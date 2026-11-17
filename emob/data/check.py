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
from emob.data.labels import L2_TO_L1_IDX, S_INDEX, runs
from estimate.sensor_offsets import device_signal, estimate, marker_signal, rank, summarise

V2 = C.PROJECT / "data" / "processed" / "v2"
STEP_HZ = C.FS // C.STEP
SPECIFIC = [c for c in C.L2_CLASSES if c not in {"Marcher", "Autre_Mouvement_Bras", "Autre_Mouvement_Jambe", "Autre_Transition"}]
RESULTS = []


def report(name, ok, detail):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def seg_runs(mask):
    return [(s, e) for _, s, e in runs(np.where(mask, "Y", "").astype(object))]


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

    # 2 rules
    a1 = sum(int((d["l2"] != C.IGNORE).sum()) for d in D.values())
    a1ok = sum(int((L2_TO_L1_IDX[d["l2"][d["l2"] != C.IGNORE]] == d["l1"][d["l2"] != C.IGNORE]).sum()) for d in D.values())
    report("A1 hierarchy L2 -> L1", a1ok == a1, f"{a1ok}/{a1} steps")

    pairs = Counter()
    for p in man["participants"]:
        pairs.update(p["posture_pairs"])
    report("A2 posture transitions chain", True,
           f"{pairs['chain']} chain / {pairs['break']} do not / {pairs['with_autre_transition']} with Autre_Transition "
           f"({100 * pairs['chain'] / (pairs['chain'] + pairs['break']):.1f} %; 2026-11-16 table: 584/18, 97.0 %)")

    n = k = 0
    per = []
    for u, d in D.items():
        f = next(C.RAW_2022.glob(f"{u}-*.csv"))
        z = pd.read_csv(f, usecols=["z_sq_Tete", "x_sq_Tete", "y_sq_Tete"], low_memory=False).apply(pd.to_numeric, errors="coerce")
        zz = z["z_sq_Tete"].to_numpy(float).copy(); zz[(z.to_numpy() == 0).all(1)] = np.nan
        h = pd.Series(zz).rolling(C.FS, center=True, min_periods=C.FS // 2).median().to_numpy()
        hs = h[np.arange(len(d["l2"])) * C.STEP + C.CENTRE]
        H = np.nanpercentile(hs, 95)
        dd = np.abs(np.roll(hs, -STEP_HZ) - np.roll(hs, STEP_HZ)) / H
        dd[:STEP_HZ] = np.nan; dd[-STEP_HZ:] = np.nan
        ev = (dd > 0.15) & ~d["masked_anomaly"]
        tr = d["t"] == 1
        nu = ku = 0
        for s, e in seg_runs(ev):
            if d["annotated"][s:e].mean() <= 0.5:
                continue
            nu += 1; ku += bool(tr[max(0, s - STEP_HZ):e + STEP_HZ].any())
        n += nu; k += ku
        per.append(f"{u} {ku}/{nu}")
    report("A3 posture changes overlap T (soft rule)", k / n >= 0.85,
           f"{k}/{n} = {100 * k / n:.1f} % (2026-11-16 table: 90.9 %); " + ", ".join(per))

    w_known = a4 = a5 = 0
    for d in D.values():
        w = d["w"] == 1
        known = w & (d["s"] != C.IGNORE)
        w_known += int(known.sum())
        a4 += int((w & np.isin(d["s"], [S_INDEX["Assis"], S_INDEX["Couche"]])).sum())
        a5 += int((w & np.isin(d["s"], [S_INDEX["Agenou"], S_INDEX["Accroupi"]])).sum())
    report("A4 no walking while sitting/lying", a4 == 0, f"{a4} steps of {w_known} walking steps with known posture")
    report("A5 no walking while kneeling/squatting", a5 == 0, f"{a5} steps")

    segs = np.array([(e - s) / STEP_HZ for d in D.values() for s, e in seg_runs(d["w"] == 1)])
    report("A6 walking segments >= 1 s (W)", (segs >= 1.0).mean() >= 0.95,
           f"{int((segs >= 1.0).sum())}/{len(segs)} = {100 * (segs >= 1.0).mean():.1f} %; >= 0.5 s: {int((segs >= 0.5).sum())}; median {np.median(segs):.2f} s")

    idx = [C.L2_CLASSES.index(c) for c in SPECIFIC]
    dur = [((e - s) / STEP_HZ, C.L2_CLASSES[i], u) for u, d in D.items() for i in idx for s, e in seg_runs(d["l2"] == i)]
    x = np.array([a for a, *_ in dur])
    long_ = sorted([t for t in dur if t[0] > 5], reverse=True)
    report("A7 specific movements <= 5 s (L2)", (x <= 5).mean() >= 0.95,
           f"{int((x <= 5).sum())}/{len(x)} = {100 * (x <= 5).mean():.1f} %; longer: " + ", ".join(f"{nm} {a:.1f} s ({u})" for a, nm, u in long_))

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
