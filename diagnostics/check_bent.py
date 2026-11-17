"""Why does the derived posture "bent" (Penche) agree poorly with motion capture? (read-only, 2026-11-13)

A fixed head height is a weak test for "bent", because the depth of a bend varies. Here the test is the
TRUNK ANGLE: angle between the vertical and the vector mid-hip -> mid-shoulder (Qualisys markers
D/G_hanche, D/G_epaule). Upright = small angle; bent forward = large angle.

A. trunk angle per derived posture state (is the measure meaningful?)
B. rows whose derived posture is "bent": how much of that time is the trunk really inclined?
C. rows whose derived posture is "standing": how much time is the trunk inclined (bends without label)?
D. each "bent" episode (after Debout_Penche, until the next body transition): bent all along / comes
   back upright before the end / never bent
E. the transitions Debout_Penche and Penche_Debout themselves: does the angle go up / down?
F. bends without a stay (Debout_Penche immediately followed by Penche_Debout): time of the largest
   angle relative to the boundary between the two labels = timing error of the body labels

Usage (from code/):  $env:PYTHONPATH="."; python -W ignore diagnostics\\check_bent.py
"""
from __future__ import annotations

import glob
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import labelshift
print(labelshift.banner() + chr(10))
from emob import config as C
from emob.data.labels import annotated_mask, posture_states, runs

FS = 50
SUSPECT_S = 12.0
AT = C.OTHER_TRANSITION
NAMED = set(C.POSTURE_TRANSITIONS)
THS = (20, 30, 40)         # degrees
T = 30                     # main threshold for "trunk inclined"


def xyz(df, k):
    p = df[[f"x_sq_{k}", f"y_sq_{k}", f"z_sq_{k}"]].apply(pd.to_numeric, errors="coerce").to_numpy(float).copy()
    p[(p == 0).all(1) | ~np.isfinite(p).all(1)] = np.nan
    return p


def smooth(x, n=FS // 4):
    return pd.Series(x).rolling(n, center=True, min_periods=max(2, n // 2)).median().to_numpy()


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.1f} %)" if b else "0/0"


def sec(n):
    return n / FS


recs = []
for f in sorted(glob.glob(str(C.RAW_2022 / "*.csv"))):
    user = os.path.basename(f).split("-")[0]
    df = pd.read_csv(f, low_memory=False)
    fam = labelshift.families(df, user)
    ann = annotated_mask(fam)
    body = fam[C.BODY_COL]
    idx = np.flatnonzero(ann); lo, hi = int(idx[0]), int(idx[-1]) + 1
    ps = posture_states(body, ann)
    sh = np.nanmean(np.stack([xyz(df, "D_epaule"), xyz(df, "G_epaule")]), axis=0)
    hp = np.nanmean(np.stack([xyz(df, "D_hanche"), xyz(df, "G_hanche")]), axis=0)
    v = sh - hp
    L = np.linalg.norm(v, axis=1)
    ang = np.degrees(np.arccos(np.clip(v[:, 2] / L, -1, 1)))
    Lmed = np.nanmedian(L)
    ang[(L < 0.6 * Lmed) | (L > 1.4 * Lmed)] = np.nan          # marker swap / artefact: implausible trunk length
    ang = smooth(ang)
    reg = np.zeros(len(body), bool); reg[lo:hi] = True
    sus = np.zeros(len(body), bool)
    for name, s, e in runs(body):
        if name in NAMED | {AT} and (e - s) / FS > SUSPECT_S:
            sus[s:e] = True
    recs.append(dict(user=user, body=body, fam=fam, ann=ann, reg=reg, sus=sus, ps=ps, st=ps.state_all, ang=ang,
                     hipz=smooth(hp[:, 2]), tr=ps.transitions))

# ---------------------------------------------------------------- A
print("=" * 100 + "\nA. Trunk angle (degrees from vertical) per derived posture state: median [p10 - p90], seconds with valid markers\n")
print(f"{'user':6s} " + " ".join(f"{s:>22s}" for s in C.S_ENGLISH))
allv = defaultdict(list)
for r in recs:
    row = []
    for k in range(len(C.S_CLASSES)):
        a = r["ang"][(r["st"] == k) & r["reg"] & ~np.isin(r["body"], list(NAMED | {AT}))]
        a = a[np.isfinite(a)]
        allv[k].append(a)
        row.append(f"{np.median(a):4.0f} [{np.percentile(a, 10):3.0f}-{np.percentile(a, 90):3.0f}] {sec(len(a)):5.0f}s" if len(a) > FS else f"{'-':>22s}")
    print(f"{r['user']:6s} " + " ".join(f"{x:>22s}" for x in row))
row = []
for k in range(len(C.S_CLASSES)):
    a = np.concatenate(allv[k])
    row.append(f"{np.median(a):4.0f} [{np.percentile(a, 10):3.0f}-{np.percentile(a, 90):3.0f}] {sec(len(a)):5.0f}s")
print(f"{'ALL':6s} " + " ".join(f"{x:>22s}" for x in row))

# ---------------------------------------------------------------- B, C
print("\n" + "=" * 100 + "\nB. Rows with derived posture BENT: share of the time with the trunk inclined more than the threshold\n")
print(f"{'user':6s} {'seconds':>8s} " + " ".join(f"{'> ' + str(t) + ' deg':>18s}" for t in THS) + f" {'< 20 deg (upright)':>20s}")
tot = Counter()
for r in recs:
    a = r["ang"][(r["st"] == 3) & r["reg"] & ~np.isin(r["body"], list(NAMED | {AT}))]; a = a[np.isfinite(a)]
    tot["n"] += len(a); tot["lt"] += int((a < 20).sum())
    for t in THS:
        tot[t] += int((a > t).sum())
    if len(a):
        print(f"{r['user']:6s} {sec(len(a)):8.1f} " + " ".join(f"{pct(int((a > t).sum()), len(a)):>18s}" for t in THS) + f" {pct(int((a < 20).sum()), len(a)):>20s}")
print(f"{'ALL':6s} {sec(tot['n']):8.1f} " + " ".join(f"{100 * tot[t] / tot['n']:17.1f}%" for t in THS) + f" {100 * tot['lt'] / tot['n']:19.1f}%")

print("\n" + "=" * 100 + "\nC. Rows with derived posture STANDING: time with the trunk inclined (a bend that has no label)\n")
print(f"{'user':6s} {'seconds':>8s} " + " ".join(f"{'> ' + str(t) + ' deg (s)':>16s}" for t in THS) + f"   episodes > {T} deg lasting >= 0.5 s (count, seconds)")
tot = Counter()
for r in recs:
    m = (r["st"] == 0) & r["reg"] & ~np.isin(r["body"], list(NAMED | {AT})) & np.isfinite(r["ang"])
    a = r["ang"][m]
    ep = [(s, e) for name, s, e in runs(np.where(m & (r["ang"] > T), "B", "").astype(object)) if (e - s) / FS >= 0.5]
    tot["n"] += len(a); tot["ep"] += len(ep); tot["eps"] += sum(e - s for s, e in ep)
    for t in THS:
        tot[t] += int((a > t).sum())
    r["hidden"] = ep
    print(f"{r['user']:6s} {sec(len(a)):8.1f} " + " ".join(f"{sec((a > t).sum()):16.1f}" for t in THS) + f"   {len(ep)}, {sec(sum(e - s for s, e in ep)):.1f} s")
print(f"{'ALL':6s} {sec(tot['n']):8.1f} " + " ".join(f"{sec(tot[t]):16.1f}" for t in THS) + f"   {tot['ep']}, {sec(tot['eps']):.1f} s")

# ---------------------------------------------------------------- D
print("\n" + "=" * 100 + "\nD. BENT episodes: rows between the end of Debout_Penche and the start of the next body transition\n")
print(f"   Classes (threshold {T} deg): 'bent all along' = inclined >= 80 % of the episode; 'never bent' = inclined < 20 %;")
print(f"   'partly' = in between. 'upright at the end' = median angle of the last 0.5 s (or of the episode if shorter) < 20 deg.\n")
eps = []
for r in recs:
    tr = r["tr"]
    for (n1, s1, e1), (n2, s2, e2) in zip(tr[:-1], tr[1:]):
        if n1 != "Debout_Penche" or r["sus"][s1:e1].any():
            continue
        a = r["ang"][e1:s2]; ok = np.isfinite(a)
        d = dict(user=r["user"], dur=(s2 - e1) / FS, nxt=n2, t=e1 / FS)
        if s2 - e1 == 0:
            d["cls"] = "no stay (0 s)"
        elif ok.sum() < max(3, 0.5 * (s2 - e1)):
            d["cls"] = "no marker"
        else:
            share = float((a[ok] > T).mean())
            d["share"] = share
            d["cls"] = "bent all along" if share >= 0.8 else ("never bent" if share < 0.2 else "partly")
            tail = a[-FS // 2:]; tail = tail[np.isfinite(tail)]
            d["up_end"] = bool(len(tail) and np.median(tail) < 20)
        eps.append(d)
du = np.array([d["dur"] for d in eps])
print(f"   Debout_Penche runs: {len(eps)}; the bent posture lasts 0 s (next transition starts at once) for {int((du == 0).sum())}; "
      f"< 0.5 s for {int((du < 0.5).sum())}; >= 0.5 s for {int((du >= 0.5).sum())}; >= 2 s for {int((du >= 2).sum())}")
print(f"   Duration of the stay: median {np.median(du):.2f} s, p75 {np.percentile(du, 75):.2f} s, p90 {np.percentile(du, 90):.2f} s, max {du.max():.1f} s; total {du.sum():.1f} s")
print("   Next transition: " + ", ".join(f"{k} {v}" for k, v in Counter(d["nxt"] for d in eps).most_common()))
for lab, sel in (("all episodes with a stay", [d for d in eps if d["dur"] > 0]), ("stay < 0.5 s", [d for d in eps if 0 < d["dur"] < 0.5]),
                 ("stay 0.5-2 s", [d for d in eps if 0.5 <= d["dur"] < 2]), ("stay >= 2 s", [d for d in eps if d["dur"] >= 2])):
    c = Counter(d["cls"] for d in sel); s = defaultdict(float)
    for d in sel:
        s[d["cls"]] += d["dur"]
    print(f"\n   {lab}: {len(sel)} episodes, {sum(d['dur'] for d in sel):.1f} s")
    for k in ("bent all along", "partly", "never bent", "no marker"):
        print(f"      {k:16s} {pct(c[k], len(sel)):>18s}   {s[k]:7.1f} s")
    te = [d for d in sel if "up_end" in d]
    print(f"      upright at the end: {pct(sum(d['up_end'] for d in te), len(te))}")
print("\n   Per participant (episodes with a stay >= 0.5 s): bent all along / partly / never bent / no marker")
for r in recs:
    c = Counter(d["cls"] for d in eps if d["user"] == r["user"] and d["dur"] >= 0.5)
    print(f"      {r['user']}: {c['bent all along']} / {c['partly']} / {c['never bent']} / {c['no marker']}")
print(f"\n   The 12 longest episodes that are 'never bent' or 'partly' (t = start of the stay, s from the first CSV row):")
for d in sorted([d for d in eps if d["cls"] in ("never bent", "partly")], key=lambda d: -d["dur"])[:12]:
    print(f"      {d['user']} t={d['t']:7.1f}  stay {d['dur']:5.1f} s  inclined {100 * d['share']:3.0f} % of the time  next: {d['nxt']}")

# ---------------------------------------------------------------- E
print("\n" + "=" * 100 + "\nE. The transitions themselves: trunk angle 0.2 s before the start and 0.2 s after the end of the label\n")
print(f"{'transition':16s} {'runs':>5s} {'no marker':>10s} {'median start':>13s} {'median end':>11s} {'median change':>14s} {'expected sign':>14s} {'change > 15 deg in the expected direction':>42s}")
for name, sign in (("Debout_Penche", 1), ("Penche_Debout", -1)):
    ch = []; a0 = []; a1 = []; nm = 0; n = 0
    for r in recs:
        for nm_, s, e in r["tr"]:
            if nm_ != name or r["sus"][s:e].any():
                continue
            n += 1
            x = np.nanmedian(r["ang"][max(s - 15, 0): max(s - 5, 1)]); y = np.nanmedian(r["ang"][e + 5: e + 15])
            if not (np.isfinite(x) and np.isfinite(y)):
                nm += 1; continue
            ch.append(y - x); a0.append(x); a1.append(y)
    ch = np.array(ch)
    print(f"{name:16s} {n:5d} {nm:10d} {np.median(a0):13.1f} {np.median(a1):11.1f} {np.median(ch):+14.1f} {'up' if sign > 0 else 'down':>14s} {pct(int((sign * ch > 15).sum()), len(ch)):>42s}")

# ---------------------------------------------------------------- F
print("\n" + "=" * 100 + "\nF. Bends without a stay: Debout_Penche followed within 0.3 s by Penche_Debout\n")
print("   The largest trunk angle should fall on the boundary between the two labels. Offset = time of the largest")
print("   angle (searched from 2 s before the first label to 2 s after the second) minus the time of the boundary.")
print("   Negative = the movement happens BEFORE the labels (labels are late).\n")
print(f"{'user':6s} {'pairs':>6s} {'usable':>7s} {'median offset (s)':>18s} {'p25':>7s} {'p75':>7s} {'peak inside the two labels':>27s} {'median peak angle':>18s}")
allo = []
for r in recs:
    off = []; inside = 0; peak = []; n = 0
    tr = r["tr"]
    for (n1, s1, e1), (n2, s2, e2) in zip(tr[:-1], tr[1:]):
        if n1 == "Debout_Penche" and n2 == "Penche_Debout" and (s2 - e1) / FS <= 0.3 and not r["sus"][s1:e2].any():
            n += 1
            a, b = max(s1 - 2 * FS, 0), min(e2 + 2 * FS, len(r["ang"]))
            seg = r["ang"][a:b]
            if np.isfinite(seg).mean() < 0.7 or np.nanmax(seg) < T:
                continue
            j = a + int(np.nanargmax(seg))
            off.append((j - (e1 + s2) / 2) / FS); inside += s1 <= j < e2; peak.append(np.nanmax(seg))
    allo += off
    if off:
        print(f"{r['user']:6s} {n:6d} {len(off):7d} {np.median(off):+18.2f} {np.percentile(off, 25):+7.2f} {np.percentile(off, 75):+7.2f} {pct(inside, len(off)):>27s} {np.median(peak):18.0f}")
    else:
        print(f"{r['user']:6s} {n:6d} {0:7d}")
print(f"{'ALL':6s} {'':6s} {len(allo):7d} {np.median(allo):+18.2f} {np.percentile(allo, 25):+7.2f} {np.percentile(allo, 75):+7.2f}")
