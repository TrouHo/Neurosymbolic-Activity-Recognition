"""Are long gaps without any label motionless? (read-only, 2026-11-13)

Reading of the data contributor (2026-11-11): a gap without label shorter than about 1 s can be an
imprecise boundary between two labels; a gap longer than 1 s, with consistent posture labels before and
after, means that the participant does not move. This script tests it against motion capture.

Gap = maximal run of rows where every label family is empty, inside the labelled region.
Movement = 3-D speed of six markers (both hands, both feet, head, torso), from positions smoothed over
0.1 s, then averaged over 0.5 s. A row is "still" when the fastest available marker is below a threshold.

Labels are first shifted by the per-participant offsets approved on 2026-11-13
(reports/data/2026-11-13_label_timing.json; applied when the whole 90 % interval is beyond +-0.1 s).

Reference groups for comparison:
  'all inactif'  rows where mvt_corps, mvt_bras and mvt_jambe are all 'inactif' (explicitly no movement)
  'movement'     rows with at least one movement label (anything other than 'inactif' / empty)

Usage (from code/):  $env:PYTHONPATH="."; python -W ignore diagnostics\\check_empty_rows.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from emob import config as C
from emob.data.labels import annotated_mask, posture_states, runs

FS = 50
THS = (50, 100, 200)        # mm/s
T = 100
PARTS = {"hands": ("D_main", "G_main"), "feet": ("D_pied", "G_pied"), "head/torso": ("Tete", "Torse")}
TIMING = json.load(open(C.PROJECT / "reports" / "data" / "2026-11-13_label_timing.json", encoding="utf-8"))


def label_shift(user):
    r = next(x for x in TIMING["estimates"][user] if x["estimator"].startswith("signed: 10 named"))
    return r["d"] if (r["lo"] > 0.1 or r["hi"] < -0.1) else 0.0


def shifted(a, k, fill=""):
    out = np.full(len(a), fill, dtype=a.dtype)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def speed(df, k):
    p = df[[f"x_sq_{k}", f"y_sq_{k}", f"z_sq_{k}"]].apply(pd.to_numeric, errors="coerce").to_numpy(float).copy()
    p[(p == 0).all(1) | ~np.isfinite(p).all(1)] = np.nan
    q = np.stack([pd.Series(p[:, a]).rolling(5, center=True, min_periods=2).mean().to_numpy() for a in range(3)], 1)
    v = np.linalg.norm(np.gradient(q, axis=0) * FS, axis=1)
    v[v > 3000] = np.nan
    return pd.Series(v).rolling(FS // 2, center=True, min_periods=FS // 4).mean().to_numpy()


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.1f} %)" if b else "0/0"


recs = []
for f in sorted(glob.glob(str(C.RAW_2022 / "*.csv"))):
    user = os.path.basename(f).split("-")[0]
    df = pd.read_csv(f, low_memory=False)
    d = label_shift(user); k = int(round(d * FS))
    fam = {c: shifted(df[c].astype("string").fillna("").to_numpy(object), k) for c in C.FAMILY_COLS}
    ann = annotated_mask(fam)
    idx = np.flatnonzero(ann); lo, hi = int(idx[0]), int(idx[-1]) + 1
    reg = np.zeros(len(df), bool); reg[lo:hi] = True
    body, arm, leg = fam["etiq mvt_corps"], fam["etiq mvt_bras"], fam["etiq mvt_jambe"]
    st = posture_states(body, ann).state_all
    part = {p: np.fmax(speed(df, a), speed(df, b)) for p, (a, b) in PARTS.items()}
    allv = np.stack(list(part.values()))
    vmax = np.nanmax(np.where(np.isfinite(allv), allv, -1), axis=0); vmax[vmax < 0] = np.nan
    nvalid = np.isfinite(allv).sum(0)
    which = np.array(list(PARTS))[np.argmax(np.where(np.isfinite(allv), allv, -1), axis=0)]
    head = pd.to_numeric(df["z_sq_Tete"], errors="coerce").to_numpy(float).copy(); head[head == 0] = np.nan
    inact = (body == "inactif") & (arm == "inactif") & (leg == "inactif")
    move = np.zeros(len(df), bool)
    for c in (body, arm, leg):
        move |= ~np.isin(c, ["", "inactif", "Debut", "Fin"])
    gaps = [(s, e) for name, s, e in runs(np.where(reg & ~ann, "G", "").astype(object))]
    recs.append(dict(user=user, shift=d, reg=reg, ann=ann, st=st, vmax=vmax, nvalid=nvalid, which=which, part=part, head=head,
                     H=np.nanpercentile(head, 95), inact=inact & reg, move=move & reg, gaps=gaps, body=body))

print("Label shifts applied (s):", {r["user"]: r["shift"] for r in recs})

# ------------------------------------------------------------------ 1
print("\n" + "=" * 110 + "\n1. Gaps without any label inside the labelled region\n")
print(f"{'user':6s} {'region s':>9s} {'gaps':>5s} {'gap s':>7s} {'% region':>9s} | {'< 1 s: gaps':>12s} {'seconds':>8s} | {'>= 1 s: gaps':>13s} {'seconds':>8s} {'median s':>9s} {'max s':>7s} | {'all inactif s':>14s} {'movement s':>11s}")
TOT = Counter()
for r in recs:
    d = np.array([(e - s) / FS for s, e in r["gaps"]])
    R = r["reg"].sum() / FS
    a, b = d[d < 1], d[d >= 1]
    TOT.update(R=R, n=len(d), s=d.sum(), na=len(a), sa=a.sum(), nb=len(b), sb=b.sum(), inact=r["inact"].sum() / FS, move=r["move"].sum() / FS)
    print(f"{r['user']:6s} {R:9.0f} {len(d):5d} {d.sum():7.1f} {100 * d.sum() / R:8.1f}% | {len(a):12d} {a.sum():8.1f} | {len(b):13d} {b.sum():8.1f} "
          f"{np.median(b) if len(b) else 0:9.1f} {b.max() if len(b) else 0:7.1f} | {r['inact'].sum() / FS:14.1f} {r['move'].sum() / FS:11.1f}")
print(f"{'ALL':6s} {TOT['R']:9.0f} {TOT['n']:5d} {TOT['s']:7.1f} {100 * TOT['s'] / TOT['R']:8.1f}% | {TOT['na']:12d} {TOT['sa']:8.1f} | {TOT['nb']:13d} {TOT['sb']:8.1f} {'':9s} {'':7s} | {TOT['inact']:14.1f} {TOT['move']:11.1f}")


def mask_gaps(r, lo_s, hi_s):
    m = np.zeros(len(r["vmax"]), bool)
    for s, e in r["gaps"]:
        if lo_s <= (e - s) / FS < hi_s:
            m[s:e] = True
    return m


# ------------------------------------------------------------------ 2
print("\n" + "=" * 110 + "\n2. Speed of the fastest marker (mm/s), by group of rows — all participants\n")
groups = {"gap < 1 s": lambda r: mask_gaps(r, 0, 1), "gap 1-3 s": lambda r: mask_gaps(r, 1, 3), "gap 3-10 s": lambda r: mask_gaps(r, 3, 10),
          "gap >= 10 s": lambda r: mask_gaps(r, 10, 1e9), "gap >= 1 s (all)": lambda r: mask_gaps(r, 1, 1e9),
          "all inactif (reference: still)": lambda r: r["inact"], "movement label (reference: moving)": lambda r: r["move"]}
print(f"{'group':36s} {'seconds':>8s} {'markers ok':>11s} {'median':>7s} {'p75':>6s} {'p90':>6s} " + " ".join(f"{'< ' + str(t):>9s}" for t in THS))
store = {}
for g, fn in groups.items():
    v = np.concatenate([r["vmax"][fn(r)] for r in recs]); n = len(v); v = v[np.isfinite(v)]
    store[g] = v
    if len(v):
        print(f"{g:36s} {n / FS:8.1f} {100 * len(v) / n:10.1f}% {np.median(v):7.0f} {np.percentile(v, 75):6.0f} {np.percentile(v, 90):6.0f} "
              + " ".join(f"{100 * (v < t).mean():8.1f}%" for t in THS))


def auc(a, b):
    """P(a random value of a is below a random value of b)."""
    x = np.concatenate([a, b]); rk = pd.Series(x).rank().to_numpy()
    return 1 - (rk[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


for g in ("gap < 1 s", "gap >= 1 s (all)", "all inactif (reference: still)"):
    print(f"   P(speed in '{g}' < speed in 'movement label') = {auc(store[g], store['movement label (reference: moving)']):.3f}")
print(f"   P(speed in 'gap >= 1 s' < speed in 'all inactif') = {auc(store['gap >= 1 s (all)'], store['all inactif (reference: still)']):.3f}   (0.5 = same distribution)")

# ------------------------------------------------------------------ 3
print("\n" + "=" * 110 + f"\n3. Per participant: share of the time with the fastest marker below {T} mm/s\n")
print(f"{'user':6s} " + " ".join(f"{g[:22]:>22s}" for g in ("gap < 1 s", "gap >= 1 s (all)", "all inactif (reference: still)", "movement label (reference: moving)")))
for r in recs:
    row = []
    for g in ("gap < 1 s", "gap >= 1 s (all)", "all inactif (reference: still)", "movement label (reference: moving)"):
        v = r["vmax"][groups[g](r)]; v = v[np.isfinite(v)]
        row.append(f"{100 * (v < T).mean():5.1f}% of {len(v) / FS:6.1f} s" if len(v) > FS // 2 else "-")
    print(f"{r['user']:6s} " + " ".join(f"{x:>22s}" for x in row))

# ------------------------------------------------------------------ 4
print("\n" + "=" * 110 + f"\n4. Per gap >= 1 s: still for at least 80 % of its rows (threshold {T} mm/s)? Posture known (the two bounding transitions chain)?\n")
G = []
for r in recs:
    for s, e in r["gaps"]:
        if (e - s) / FS < 1:
            continue
        v = r["vmax"][s:e]; ok = np.isfinite(v)
        stt = r["st"][s:e]
        known = bool((stt >= 0).mean() > 0.9)
        h = r["head"][s:e]; h = h[np.isfinite(h)]
        g = dict(user=r["user"], t=s / FS, dur=(e - s) / FS, known=known, state=C.S_ENGLISH[int(np.bincount(stt[stt >= 0]).argmax())] if known else "unknown")
        if ok.sum() < 0.5 * (e - s):
            g["cls"] = "no marker"
        else:
            sh = float((v[ok] < T).mean()); g["share"] = sh
            g["cls"] = "still" if sh >= 0.8 else ("moving" if sh < 0.2 else "mixed")
            w = r["which"][s:e][ok & (np.nan_to_num(v) >= T)]
            g["mover"] = Counter(w).most_common(1)[0][0] if len(w) else ""
            g["hspread"] = float((np.percentile(h, 95) - np.percentile(h, 5)) / r["H"]) if len(h) > 5 else np.nan
        G.append(g)
for lab, sel in (("all gaps >= 1 s", G), ("posture known", [g for g in G if g["known"]]), ("posture unknown", [g for g in G if not g["known"]])):
    c = Counter(g["cls"] for g in sel); sec = defaultdict(float)
    for g in sel:
        sec[g["cls"]] += g["dur"]
    print(f"   {lab}: {len(sel)} gaps, {sum(g['dur'] for g in sel):.1f} s")
    for k in ("still", "mixed", "moving", "no marker"):
        print(f"      {k:10s} {pct(c[k], len(sel)):>18s} {sec[k]:8.1f} s")
print("\n   By duration (gaps with markers): still / mixed / moving")
for a, b in ((1, 2), (2, 5), (5, 10), (10, 1e9)):
    sel = [g for g in G if a <= g["dur"] < b and g["cls"] != "no marker"]
    c = Counter(g["cls"] for g in sel)
    print(f"      {a:>2}-{b if b < 1e8 else 'inf':>3} s: {len(sel):4d} gaps   {c['still']} / {c['mixed']} / {c['moving']}")
print("\n   By posture state (gaps with markers): gaps, seconds, still / mixed / moving, seconds still")
for stn in C.S_ENGLISH + ["unknown"]:
    sel = [g for g in G if g["state"] == stn and g["cls"] != "no marker"]
    if sel:
        c = Counter(g["cls"] for g in sel)
        print(f"      {stn:10s} {len(sel):4d} {sum(g['dur'] for g in sel):8.1f} s   {c['still']} / {c['mixed']} / {c['moving']}   {sum(g['dur'] for g in sel if g['cls'] == 'still'):.1f} s")
print("\n   Per participant (gaps with markers): still / mixed / moving, seconds in each")
for r in recs:
    sel = [g for g in G if g["user"] == r["user"] and g["cls"] != "no marker"]
    c = Counter(g["cls"] for g in sel); sec = defaultdict(float)
    for g in sel:
        sec[g["cls"]] += g["dur"]
    print(f"      {r['user']}: {c['still']} / {c['mixed']} / {c['moving']}   {sec['still']:.1f} / {sec['mixed']:.1f} / {sec['moving']:.1f} s")
mv = [g for g in G if g["cls"] in ("mixed", "moving")]
print(f"\n   In the {len(mv)} gaps that are not still, the fastest marker is most often: " + ", ".join(f"{k} {v}" for k, v in Counter(g["mover"] for g in mv).most_common()))
hs = np.array([g["hspread"] for g in G if g["cls"] != "no marker" and np.isfinite(g.get("hspread", np.nan))])
print(f"   Head-height spread inside a gap >= 1 s (p95 - p5, fraction of standing height): median {np.median(hs):.3f}; > 0.10 in {pct(int((hs > 0.10).sum()), len(hs))}; > 0.15 in {pct(int((hs > 0.15).sum()), len(hs))}")
hk = np.array([g["hspread"] for g in G if g["cls"] != "no marker" and g["known"] and np.isfinite(g.get("hspread", np.nan))])
print(f"   Same, gaps with known posture only: > 0.10 in {pct(int((hk > 0.10).sum()), len(hk))}; > 0.15 in {pct(int((hk > 0.15).sum()), len(hk))}")
print("\n   The 15 longest gaps that are 'moving' or 'mixed' (t = start, s from the first CSV row, after the label shift):")
for g in sorted(mv, key=lambda g: -g["dur"])[:15]:
    print(f"      {g['user']} t={g['t']:7.1f}  {g['dur']:5.1f} s  still {100 * g['share']:3.0f} % of the time  posture {g['state']:9s} fastest: {g['mover']}")

# ------------------------------------------------------------------ 5
print("\n" + "=" * 110 + "\n5. Sensitivity of section 4 to the speed threshold (gaps >= 1 s with markers: still / mixed / moving)\n")
for t in THS:
    c = Counter()
    for r in recs:
        for s, e in r["gaps"]:
            if (e - s) / FS >= 1:
                v = r["vmax"][s:e]; ok = np.isfinite(v)
                if ok.sum() >= 0.5 * (e - s):
                    sh = (v[ok] < t).mean(); c["still" if sh >= 0.8 else ("moving" if sh < 0.2 else "mixed")] += 1
    print(f"   threshold {t:3d} mm/s: {c['still']} / {c['mixed']} / {c['moving']}")
