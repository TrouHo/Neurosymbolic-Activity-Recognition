"""Agreement of rules A1-A7 with the prepared targets (data v2). Single definition used by both the
gate check (emob.data.check) and the inspection numbers (inspection_data/04_rules.py).

Every function takes D = {participant: dict of v2 arrays at the 0.1-s step grid} and returns plain
numbers; nothing is printed here.
  A1  L2 -> L1 hierarchy, on labelled L2 steps.
  A2  successive named transitions of the body track chain (X_Y followed by Y_Z); counted by the
      build (manifest "posture_pairs"), so it is passed in.
  A3  posture changes measured independently by motion capture (head height change > 15 % of the
      participant's standing height between t - 1 s and t + 1 s, i.e. within 2 s), outside masked
      anomalies, mostly in annotated time, overlapping T = 1 within +-1 s.
  A4  walking (W = 1) while sitting or lying;  A5  while kneeling or squatting.
  A6  maximal runs of W = 1 (any neighbour that is not W = 1 ends a run), duration >= 1 s.
  A7  maximal runs of each of the 16 specific L2 classes, duration <= 5 s.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from emob import config as C
from emob.data.labels import L2_TO_L1_IDX, S_INDEX, runs

STEP_HZ = C.FS // C.STEP                       # steps per second (10)
SPECIFIC = [c for c in C.L2_CLASSES if c not in {"Marcher", "Autre_Mouvement_Bras", "Autre_Mouvement_Jambe", "Autre_Transition"}]


def seg_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs (start, end_exclusive) where mask is True."""
    return [(s, e) for _, s, e in runs(np.where(mask, "Y", "").astype(object))]


def a1(D: dict) -> dict:
    n = ok = 0
    for d in D.values():
        lab = d["l2"] != C.IGNORE
        n += int(lab.sum())
        ok += int((L2_TO_L1_IDX[d["l2"][lab]] == d["l1"][lab]).sum())
    return {"ok": ok, "n": n}


def a2_pairs(manifest: dict) -> dict:
    c = Counter()
    for p in manifest["participants"]:
        c.update(p["posture_pairs"])
    return {"chain": c["chain"], "break": c["break"], "with_other": c["with_autre_transition"]}


def a3(D: dict) -> dict:
    n = k = 0
    per = {}
    for u, d in D.items():
        f = next(C.RAW_2022.glob(f"{u}-*.csv"))
        z = pd.read_csv(f, usecols=["z_sq_Tete", "x_sq_Tete", "y_sq_Tete"], low_memory=False).apply(pd.to_numeric, errors="coerce")
        zz = z["z_sq_Tete"].to_numpy(float).copy()
        zz[(z.to_numpy() == 0).all(1)] = np.nan
        h = pd.Series(zz).rolling(C.FS, center=True, min_periods=C.FS // 2).median().to_numpy()
        hs = h[np.arange(len(d["l2"])) * C.STEP + C.CENTRE]
        H = np.nanpercentile(hs, 95)
        dd = np.abs(np.roll(hs, -STEP_HZ) - np.roll(hs, STEP_HZ)) / H
        dd[:STEP_HZ] = np.nan
        dd[-STEP_HZ:] = np.nan
        ev = (dd > 0.15) & ~d["masked_anomaly"]
        tr = d["t"] == 1
        nu = ku = 0
        for s, e in seg_runs(ev):
            if d["annotated"][s:e].mean() <= 0.5:
                continue
            nu += 1
            ku += bool(tr[max(0, s - STEP_HZ):e + STEP_HZ].any())
        n += nu
        k += ku
        per[u] = (ku, nu)
    return {"ok": k, "n": n, "per_participant": per}


def a4_a5(D: dict) -> dict:
    known = a4 = a5 = 0
    for d in D.values():
        w = d["w"] == 1
        known += int((w & (d["s"] != C.IGNORE)).sum())
        a4 += int((w & np.isin(d["s"], [S_INDEX["Assis"], S_INDEX["Couche"]])).sum())
        a5 += int((w & np.isin(d["s"], [S_INDEX["Agenou"], S_INDEX["Accroupi"]])).sum())
    return {"walking_steps_known_posture": known, "a4_violations": a4, "a5_violations": a5}


def a6(D: dict) -> dict:
    segs = np.array([(e - s) / STEP_HZ for d in D.values() for s, e in seg_runs(d["w"] == 1)])
    return {"ok": int((segs >= 1.0).sum()), "n": len(segs), "ge_0_5s": int((segs >= 0.5).sum()),
            "median_s": float(np.median(segs)), "durations_s": segs}


def a7(D: dict) -> dict:
    idx = [C.L2_CLASSES.index(c) for c in SPECIFIC]
    dur = [((e - s) / STEP_HZ, C.L2_CLASSES[i], u) for u, d in D.items() for i in idx for s, e in seg_runs(d["l2"] == i)]
    x = np.array([a for a, *_ in dur])
    return {"ok": int((x <= 5).sum()), "n": len(x), "longer": sorted([t for t in dur if t[0] > 5], reverse=True),
            "longest_within_5s": max(t for t in dur if t[0] <= 5)}
