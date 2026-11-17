"""Shared: 2022 recordings with shifted labels, and the PROPOSED treatment of the six long runs.

The treatment is the proposal of reports/data/2026-11-13_label_errors_check.md §2.4 (not decided):
the first and the last real movement of a long run keep the label, the middle is "not annotated"
for that family. The boundaries are recomputed here with the same method as check_label_errors.py
(signed vertical velocity of the family's marker: > 150 mm/s = movement, ends when < 50 mm/s for
0.2 s; Autre_Transition: head and torso below 100 mm/s for 0.5 s). Nothing is written to the data.

Labels are shifted by the approved per-participant offsets when EMOB_LABEL_SHIFT=1 (labelshift.py).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import labelshift
from emob import config as C
from emob.data.labels import annotated_mask, posture_states, runs

FS = 50
SUSPECT_S = 12.0
GENERIC = {"inactif", "Marcher", "Autre_Mouvement_Bras", "Autre_Mouvement_Jambe", "Debut", "Fin", ""}
COLS = {"body": "etiq mvt_corps", "arm": "etiq mvt_bras", "leg": "etiq mvt_jambe"}
GROUP = {"body": ("Tete", "Torse"), "arm": ("D_main", "G_main"), "leg": ("D_pied", "G_pied")}
MARKERS = ("Tete", "Torse", "D_main", "G_main", "D_pied", "G_pied")
STILL = 100.0
SIGN = {**{k: -1 for k in ("Debout_Assis", "Assis_Couche", "Debout_Penche", "Debout_Agenou", "Debout_Accroupi", "Baisser_Bras", "Baisser_2_Bras", "Baisser_Jambe")},
        **{k: 1 for k in ("Assis_Debout", "Couche_Assis", "Penche_Debout", "Agenou_Debout", "Accroupi_Debout", "Lever_Bras", "Lever_2_Bras", "Lever_Jambe")}}


def _xyz(df, k):
    p = df[[f"x_sq_{k}", f"y_sq_{k}", f"z_sq_{k}"]].apply(pd.to_numeric, errors="coerce").to_numpy(float).copy()
    p[(p == 0).all(1) | ~np.isfinite(p).all(1)] = np.nan
    return p


def sm(x, n):
    return pd.Series(x).rolling(n, center=True, min_periods=max(2, n // 2)).mean().to_numpy()


def _speed(p):
    q = np.stack([sm(p[:, a], 5) for a in range(3)], 1)
    v = np.linalg.norm(np.gradient(q, axis=0) * FS, axis=1)
    v[v > 3000] = np.nan
    return sm(v, FS // 2)


def _movement_end(v, s, limit, n):
    moved = False
    for i in range(s, min(limit, n - FS // 2)):
        if np.isfinite(v[i]) and v[i] >= STILL:
            moved = True
        if moved and np.nanmax(v[i:i + FS // 2]) < STILL:
            return i
    return None


def _first_movement(sv, s, limit):
    on = next((i for i in range(s, min(s + 3 * FS, limit)) if np.isfinite(sv[i]) and sv[i] > 150), None)
    if on is None:
        return None
    for j in range(on, limit):
        seg = sv[j:j + 10]
        if np.isfinite(seg).any() and np.nanmax(seg) < 50:
            return j
    return None


def _last_movement(sv, e, floor):
    k = next((i for i in range(min(e + FS // 2, len(sv) - 1), max(e - 3 * FS, floor), -1) if np.isfinite(sv[i]) and sv[i] > 150), None)
    if k is None:
        return None
    a = k
    while a > floor and np.isfinite(sv[a]) and sv[a] > 50:
        a -= 1
    b = k
    while b < len(sv) - 1 and np.isfinite(sv[b]) and sv[b] > 50:
        b += 1
    return a, b


def load():
    """One dict per 2022 recording: shifted label families, activite, markers, long runs, repaired families."""
    recs = []
    for f in sorted(glob.glob(str(C.RAW_2022 / "*.csv"))):
        user = os.path.basename(f).split("-")[0]
        want = set(C.FAMILY_COLS + [C.CAT_COL, C.ACT_COL] + [f"{a}_sq_{k}" for k in MARKERS for a in "xyz"])
        df = pd.read_csv(f, low_memory=False, usecols=lambda c: c in want)
        n = len(df)
        k = int(round(labelshift.shift_s(user) * FS))
        fam = labelshift.families(df, user)
        act = labelshift.shifted(df[C.ACT_COL].astype("string").fillna("").to_numpy(object), k)
        cat = labelshift.shifted(df[C.CAT_COL].astype("string").fillna("").to_numpy(object), k)
        ann = annotated_mask(fam)
        idx = np.flatnonzero(ann)
        P = {m: _xyz(df, m) for m in MARKERS}
        vz = lambda a: np.gradient(sm(P[a][:, 2], 5)) * FS
        nsum = lambda a, b: np.where(np.isfinite(a) | np.isfinite(b), np.nan_to_num(a) + np.nan_to_num(b), np.nan)
        svz = {"body": sm(vz("Tete"), 10), "arm": sm(nsum(vz("D_main"), vz("G_main")), 10), "leg": sm(nsum(vz("D_pied"), vz("G_pied")), 10)}
        spd = {g: np.fmax(_speed(P[a]), _speed(P[b])) for g, (a, b) in GROUP.items()}

        # ---- long runs and their first / last real movement
        long_runs = []
        for famn, col in COLS.items():
            for name, s, e in runs(fam[col]):
                if name in GENERIC or (e - s) / FS <= SUSPECT_S:
                    continue
                if name in SIGN:
                    sv = SIGN[name] * svz[famn]
                    fe = _first_movement(sv, s, e)
                    lm = _last_movement(sv, e, (fe or s) + FS)
                else:
                    fe, lm = _movement_end(spd[famn], s, e, n), None
                long_runs.append(dict(user=user, family=famn, col=col, name=name, s=s, e=e, fe=fe, lm=lm))

        # ---- proposed repair: the middle of each long run is "not annotated" for that family
        rep = {c: v.copy() for c, v in fam.items()}
        unann = {c: np.zeros(n, bool) for c in COLS.values()}
        for lr in long_runs:
            a = lr["fe"] if lr["fe"] is not None else lr["s"]
            b = min(lr["lm"][0], lr["e"]) if lr["lm"] else lr["e"]
            lr["mid"] = (a, b)
            rep[lr["col"]][a:b] = ""
            unann[lr["col"]][a:b] = True
        ps = posture_states(fam[C.BODY_COL], ann)
        ps_rep = posture_states(rep[C.BODY_COL], ann)
        st_rep = ps_rep.state_all.copy()
        for _, a, b in runs(np.where(unann[C.BODY_COL], "U", "").astype(object)):
            nxt = next((s2 for _, s2, _ in ps_rep.transitions if s2 >= b), n)
            st_rep[a:nxt] = C.IGNORE                       # posture unknown until the next annotated transition
        recs.append(dict(user=user, n=n, fam=fam, act=act, cat=cat, ann=ann, lo=int(idx[0]), hi=int(idx[-1]) + 1,
                         head=P["Tete"][:, 2], long_runs=long_runs, rep=rep, unann=unann,
                         ps=ps, ps_rep=ps_rep, st=ps.state_all, st_rep=st_rep))
    return recs


def print_long_runs(recs):
    print("Long runs (a specific movement longer than 12 s) and the middle part treated as 'not annotated' (proposal):")
    print(f"{'user':6s} {'label':18s} {'family':6s} {'labelled':>16s} | {'first movement':>16s} | {'last movement':>16s} | {'middle':>16s} {'s':>6s}")
    tot = 0.0
    for r in recs:
        for lr in r["long_runs"]:
            a, b = lr["mid"]
            tot += (b - a) / FS
            f1 = f"{lr['s'] / FS:.1f}-{lr['fe'] / FS:.1f}" if lr["fe"] is not None else "not found"
            f2 = f"{lr['lm'][0] / FS:.1f}-{lr['lm'][1] / FS:.1f}" if lr["lm"] else "none"
            print(f"{lr['user']:6s} {lr['name']:18s} {lr['family']:6s} {lr['s'] / FS:7.1f}-{lr['e'] / FS:<8.1f} | {f1:>16s} | {f2:>16s} | {a / FS:7.1f}-{b / FS:<8.1f} {(b - a) / FS:6.1f}")
    print(f"   sum of the middle parts: {tot:.1f} s (they overlap for user1)\n")
    print("What follows each long run in the same family column (next run that is not 'inactif'):")
    for r in recs:
        for lr in r["long_runs"]:
            nxt = next(((nm, s, e) for nm, s, e in runs(r["fam"][lr["col"]]) if s >= lr["e"] and nm != "inactif"), None)
            print(f"   {lr['user']:6s} {lr['name']:18s} ends at {lr['e'] / FS:7.2f} s; next: "
                  + (f"{nxt[0]} {nxt[1] / FS:.2f}-{nxt[2] / FS:.2f} s, {(nxt[1] - lr['e']) / FS:.2f} s after the end"
                     + ("  <- same label" if nxt[0] == lr["name"] else "") if nxt else "none"))
    print()
