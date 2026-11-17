"""Agreement of the rules A1-A4, B1, B2, B5a with the 2022 labels, recomputed for the meeting of 2026-11-16.

Same definitions as code/emob/data/rules.py (A1-A4) and check_rules_B.py (B1, B2, B5a). Two versions:
  "as labelled"  the label columns (shifted when EMOB_LABEL_SHIFT=1); the six long runs are still there,
                 and are excluded where check_rules_B.py excluded them (B2, B5);
  "repaired"     the proposed treatment of the long runs (labelrepair.py): middle = not annotated,
                 posture unknown from a not-annotated body part until the next annotated transition.
Read-only. Rows at 50 Hz, shown in seconds.

Usage (from code/):  $env:PYTHONPATH="."; $env:EMOB_LABEL_SHIFT="1"; python -W ignore diagnostics\\check_rules_shifted.py
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

import labelrepair as LR
import labelshift
from emob import config as C
from emob.data.labels import S_INDEX, runs
from emob.data.rules import hierarchy_agreement

FS = 50
TRANS = C.POSTURE_TRANSITIONS
SPEC = ["Lever_Bras", "Baisser_Bras", "Lever_2_Bras", "Baisser_2_Bras", "Lever_Jambe", "Baisser_Jambe"] + TRANS
BODY, ARM, LEG = "etiq mvt_corps", "etiq mvt_bras", "etiq mvt_jambe"


def pct(a, b):
    return f"{a} / {b} ({100 * a / b:.1f} %)" if b else "0 / 0"


print(labelshift.banner() + "\n")
recs = LR.load()
LR.print_long_runs(recs)

for version in ("as labelled", "repaired"):
    print("=" * 110 + f"\nVERSION: {version}\n")
    rep = version == "repaired"
    V = []
    for r in recs:
        fam = r["rep"] if rep else r["fam"]
        ps = r["ps_rep"] if rep else r["ps"]
        st = r["st_rep"] if rep else r["st"]
        sus = np.zeros(r["n"], bool)
        if rep:
            sus_body = r["unann"][BODY]
        else:
            for lr in r["long_runs"]:
                sus[lr["s"]:lr["e"]] = True
            sus_body = sus
        V.append(dict(r=r, fam=fam, ps=ps, st=st, sus=sus, sus_body=sus_body))

    # ---- A1
    if not rep:
        n = k = 0
        for v in V:
            a, b = hierarchy_agreement(v["r"]["act"], v["r"]["cat"], v["fam"])
            n += a; k += b
        print(f"A1  `activite` equals the label of the family named in `categorie`: {pct(k, n)} rows")
        print("    (in the model the coarse class is the parent of the fine class by definition: the rule holds by construction)")
    else:
        print("A1  unchanged (does not depend on the repair)")

    # ---- A2
    c = Counter()
    for v in V:
        c.update(ok=v["ps"].consistent_pairs, bad=v["ps"].inconsistent_pairs, at=v["ps"].pairs_with_other)
    print(f"A2  consecutive transitions: {c['ok']} chain (X_Y then Y_Z), {c['bad']} do not, {c['at']} involve Autre_Transition (not checkable)"
          f" -> {pct(c['ok'], c['ok'] + c['bad'])} of the checkable pairs")
    print("    per participant (chain / do not): " + ", ".join(f"{v['r']['user']} {v['ps'].consistent_pairs}/{v['ps'].inconsistent_pairs}" for v in V))

    # ---- A3
    d = np.array([(e - s) / FS for v in V for name, s, e in runs(v["fam"][LEG]) if name == "Marcher"])
    print(f"A3  walking segments of at least 1.0 s: {pct(int((d >= 1.0).sum()), len(d))}; at least 0.5 s: {pct(int((d >= 0.5).sum()), len(d))}; "
          f"median {np.median(d):.2f} s; shortest {np.sort(d)[:3].round(2).tolist()}")

    # ---- A4 / B5a
    cnt = Counter(); tot = 0
    for v in V:
        w = (v["fam"][LEG] == "Marcher") & ~v["sus"]
        tot += int(w.sum())
        for i in range(-1, len(C.S_CLASSES)):
            cnt[i] += int((w & (v["st"] == i)).sum())
    known = tot - cnt[-1]
    print(f"A4 / B5a  Marcher: {tot / FS:.1f} s; posture state known on {known / FS:.1f} s: "
          + ", ".join(f"{C.S_ENGLISH[i].lower()} {cnt[i] / FS:.1f}" for i in range(len(C.S_CLASSES))))
    a4 = cnt[S_INDEX["Assis"]] + cnt[S_INDEX["Couche"]]; b5 = cnt[S_INDEX["Agenou"]] + cnt[S_INDEX["Accroupi"]]
    print(f"    A4  walking while sitting or lying: {a4} rows ({a4 / FS:.1f} s) -> holds on {pct(known - a4, known)} rows")
    print(f"    B5a walking while kneeling or squatting: {b5} rows ({b5 / FS:.1f} s) -> holds on {pct(known - b5, known)} rows")
    print(f"    (rejected B5b: walking while bent {cnt[S_INDEX['Penche']] / FS:.1f} s)")

    # ---- B1
    dur = [((e - s) / FS, name, v["r"]["user"]) for v in V for col in (BODY, ARM, LEG) for name, s, e in runs(v["fam"][col]) if name in SPEC]
    d = np.array([x[0] for x in dur])
    print(f"B1  specific movements (4 arm, 2 leg raise/lower, 10 named transitions): " + "; ".join(f"<= {th} s: {pct(int((d <= th).sum()), len(d))}" for th in (3, 5, 8)))
    print("    longer than 5 s: " + ", ".join(f"{n} {x:.1f} s ({u})" for x, n, u in sorted(dur, reverse=True) if x > 5))

    # ---- B2
    T = Counter(); per = []
    for v in V:
        r = v["r"]
        h = pd.Series(r["head"]).rolling(FS, center=True, min_periods=FS // 2).median().to_numpy()
        H = np.nanpercentile(h, 95)
        dd = np.abs(np.roll(h, -FS) - np.roll(h, FS)) / H
        dd[:FS] = np.nan; dd[-FS:] = np.nan
        ev = (dd > 0.15) & ~v["sus_body"]
        ev[:r["lo"]] = False; ev[r["hi"]:] = False
        tr = np.isin(v["fam"][BODY], TRANS + [C.OTHER_TRANSITION])
        n = k = na = ka = 0
        for _, s, e in runs(np.where(ev, "E", "").astype(object)):
            ok = bool(tr[max(0, s - FS): e + FS].any())
            n += 1; k += ok
            if r["ann"][s:e].mean() > 0.5:
                na += 1; ka += ok
        T.update(n=n, k=k, na=na, ka=ka)
        per.append(f"{r['user']} {ka}/{na} ({100 * ka / max(na, 1):.0f} %)")
    print(f"B2  changes of head height > 15 % within 2 s that overlap an annotated body transition (+-1 s): all rows {pct(T['k'], T['n'])}; "
          f"on annotated rows {pct(T['ka'], T['na'])}")
    print("    per participant (annotated rows): " + ", ".join(per))
    print()
