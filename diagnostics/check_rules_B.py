"""Agreement of candidate rules B1-B6 with the 2022 labels (read-only, 2026-11-10).

B1 maximum duration of a specific movement          (labels only)
B2 posture changes only through a transition         (labels vs Qualisys head height; the label-only
                                                      version is true by construction, see report)
B3 raise arm => hand goes up, lower => hand goes down (labels vs Qualisys hand height; hand-sensor tilt)
B4 hand still => no specific arm movement             (labels vs Qualisys hand speed; hand-sensor activity)
B5 no walking outside STANDING, none during a posture transition   (labels only)
B6 leg movements per posture state                    (labels only)

Usage (from code/):  $env:PYTHONPATH="."; python diagnostics\\check_rules_B.py
Sensor time shifts are read from reports/data/2026-11-10_observability_matrix.json.
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
SUSPECT_S = 12.0
ARM_SPEC = ["Lever_Bras", "Baisser_Bras", "Lever_2_Bras", "Baisser_2_Bras"]
LEG_SPEC = ["Lever_Jambe", "Baisser_Jambe"]
TRANS = C.POSTURE_TRANSITIONS
SHIFTS = json.load(open(C.PROJECT / "reports" / "data" / "2026-11-10_observability_matrix.json", encoding="utf-8"))["meta"]["sensors"]
LOW = {"Assis": 1, "Couche": 2, "Penche": 1, "Agenou": 1, "Accroupi": 1, "Debout": 0}   # 0 high, 2 lowest


def marker(df, k):
    p = df[[f"x_sq_{k}", f"y_sq_{k}", f"z_sq_{k}"]].to_numpy(float).copy()
    p[(p == 0).all(1) | ~np.isfinite(p).all(1)] = np.nan
    return p


def roll(x, n, f="mean"):
    r = pd.Series(x).rolling(n, center=True, min_periods=max(2, n // 2))
    return getattr(r, f)().to_numpy()


def speed(p):
    v = np.linalg.norm(np.diff(p, axis=0, prepend=p[:1]), axis=1) * FS
    v[v > 3000] = np.nan
    return roll(v, FS // 2)


def at(z, i, half=2):
    seg = z[max(0, i - half): i + half + 1]
    return np.nanmean(seg) if np.isfinite(seg).any() else np.nan


def shifted(x, sec):
    k = int(round(sec * FS)); out = np.full_like(x, np.nan)
    if k >= 0:
        out[: len(x) - k] = x[k:]
    else:
        out[-k:] = x[: len(x) + k]
    return out


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.1f} %)" if b else "0/0"


recs = []
for f in sorted(glob.glob(str(C.RAW_2022 / "*.csv"))):
    user = os.path.basename(f).split("-")[0]
    df = pd.read_csv(f, low_memory=False)
    cols = list(df.columns)
    fam = {c: df[c].astype("string").fillna("").to_numpy(object) for c in C.FAMILY_COLS}
    ann = annotated_mask(fam)
    body, arm, leg = fam["etiq mvt_corps"], fam["etiq mvt_bras"], fam["etiq mvt_jambe"]
    st = posture_states(body, ann).state_all
    i = cols.index("ts_SB - Main droite - ACC")
    hand = df[cols[i + 1:i + 4]].apply(pd.to_numeric, errors="coerce").to_numpy(float).copy()
    info = SHIFTS[user]["hand"]
    if info["used"]:
        hand[(hand == 0).all(1) | ~np.isfinite(hand).all(1)] = np.nan
        hand = np.stack([shifted(hand[:, k], info["shift"]) for k in range(3)], 1)
    else:
        hand = None
    recs.append(dict(user=user, df=df, body=body, arm=arm, leg=leg, ann=ann, st=st, hand=hand,
                     zD=marker(df, "D_main")[:, 2], zG=marker(df, "G_main")[:, 2],
                     vD=speed(marker(df, "D_main")), vG=speed(marker(df, "G_main")),
                     head=marker(df, "Tete")[:, 2]))

# =============================================================================== B1
print("=" * 100, "\nB1  Duration of specific movements (runs in the family column; seconds)\n")
dur = defaultdict(list)
for r in recs:
    for col in ("body", "arm", "leg"):
        for name, s, e in runs(r[col]):
            if name not in ("inactif", "Debut", "Fin"):
                dur[name].append(((e - s) / FS, r["user"]))
print(f"{'class':24s} {'runs':>5s} {'median':>7s} {'p90':>6s} {'p95':>6s} {'max':>7s}  {'>3s':>5s} {'>5s':>5s} {'>8s':>5s} {'>12s':>5s}  longest (user)")
tot = Counter()
for name in ARM_SPEC + LEG_SPEC + TRANS + ["Autre_Transition", "Autre_Mouvement_Bras", "Autre_Mouvement_Jambe", "Marcher"]:
    d = np.array([x[0] for x in dur[name]])
    top = sorted(dur[name], reverse=True)[:3]
    print(f"{name:24s} {len(d):5d} {np.median(d):7.2f} {np.percentile(d, 90):6.2f} {np.percentile(d, 95):6.2f} {d.max():7.1f}  "
          f"{(d > 3).sum():5d} {(d > 5).sum():5d} {(d > 8).sum():5d} {(d > 12).sum():5d}  " + ", ".join(f"{a:.1f} ({u})" for a, u in top))
    if name in ARM_SPEC + LEG_SPEC + TRANS:
        tot["n"] += len(d)
        for th in (3, 5, 8, 12):
            tot[th] += int((d > th).sum())
print("\nSpecific movements (arm raise/lower, leg raise/lower, 10 posture transitions): "
      + "; ".join(f"<= {th} s: {pct(tot['n'] - tot[th], tot['n'])}" for th in (3, 5, 8, 12)))

# suspect mask for the other checks
for r in recs:
    sus = np.zeros(len(r["arm"]), bool)
    for col in ("body", "arm", "leg"):
        for name, s, e in runs(r[col]):
            if name not in ("inactif", "Marcher", "Autre_Mouvement_Bras", "Autre_Mouvement_Jambe", "Debut", "Fin") and (e - s) / FS > SUSPECT_S:
                sus[s:e] = True
    r["sus"] = sus

# =============================================================================== B2
print("\n" + "=" * 100, "\nB2  Posture changes only through a transition\n")
print("(a) Label-only version: the posture state is DERIVED from the transition names, so the rule holds by construction.")
print("(b) Independent check against Qualisys: every large change of head height should overlap an annotated")
print("    transition (any mvt_corps label, +-1 s). Event = |head(t+1s) - head(t-1s)| > 15 % of standing head height.\n")
print(f"{'user':6s} {'events':>6s} {'with transition':>22s} | {'events on annotated rows':>24s} {'with transition':>22s} | events on unannotated rows")
T2 = Counter()
for r in recs:
    h = roll(r["head"], FS, "median")
    H = np.nanpercentile(h, 95)
    d = np.abs(np.roll(h, -FS) - np.roll(h, FS)) / H
    d[:FS] = np.nan; d[-FS:] = np.nan
    idx = np.flatnonzero(r["ann"]); lo, hi = idx[0], idx[-1] + 1
    ev = (d > 0.15) & ~r["sus"]
    ev[:lo] = False; ev[hi:] = False
    tr = np.isin(r["body"], TRANS + ["Autre_Transition"])
    n = k = na = ka = nu = 0
    for _, s, e in runs(np.where(ev, "E", "").astype(object)):
        c = (s + e) // 2
        ok = tr[max(0, s - FS): e + FS].any()
        n += 1; k += ok
        if r["ann"][s:e].mean() > 0.5:
            na += 1; ka += ok
        else:
            nu += 1
    T2.update(n=n, k=k, na=na, ka=ka, nu=nu)
    print(f"{r['user']:6s} {n:6d} {pct(k, n):>22s} | {na:24d} {pct(ka, na):>22s} | {nu}")
print(f"{'ALL':6s} {T2['n']:6d} {pct(T2['k'], T2['n']):>22s} | {T2['na']:24d} {pct(T2['ka'], T2['na']):>22s} | {T2['nu']}")

print("\n(c) Direction of each annotated transition: head height 0.5 s after the end minus 0.5 s before the start")
print("    (fraction of standing head height). Expected sign: down for Debout_X and Assis_Couche, up for the reverse.\n")
print(f"{'transition':18s} {'runs':>5s} {'median dh':>10s} {'expected sign':>14s} {'agree':>18s} {'|dh|<0.03':>10s}")
T2c = Counter()
for name in TRANS:
    a, b = name.split("_")
    exp = np.sign(LOW[a] - LOW[b])            # +1 = head goes up
    vals = []
    for r in recs:
        H = np.nanpercentile(r["head"], 95)
        for nm, s, e in runs(r["body"]):
            if nm == name and not r["sus"][s:e].any():
                dh = (at(r["head"], min(e + FS // 2, len(r["head"]) - 1)) - at(r["head"], max(s - FS // 2, 0))) / H
                if np.isfinite(dh):
                    vals.append(dh)
    v = np.array(vals)
    ag = int((np.sign(v) == exp).sum())
    T2c.update(n=len(v), k=ag)
    print(f"{name:18s} {len(v):5d} {np.median(v):+10.3f} {'up' if exp > 0 else 'down':>14s} {pct(ag, len(v)):>18s} {int((np.abs(v) < 0.03).sum()):10d}")
print(f"{'ALL':18s} {T2c['n']:5d} {'':10s} {'':14s} {pct(T2c['k'], T2c['n']):>18s}")

# =============================================================================== B3
print("\n" + "=" * 100, "\nB3  Raise arm => hand goes up; lower arm => hand goes down (Qualisys hand height, end minus start of the run)\n")
print("One-arm labels do not say which arm: the arm with the larger |dz| is taken as the moving arm.")
print("Two-arm labels: both hands must move in the expected direction.\n")
rows3 = defaultdict(list)
for r in recs:
    for name, s, e in runs(r["arm"]):
        if name in ARM_SPEC + ["Autre_Mouvement_Bras"] and not r["sus"][s:e].any():
            dD, dG = at(r["zD"], e - 1) - at(r["zD"], s), at(r["zG"], e - 1) - at(r["zG"], s)
            rows3[name].append((dD, dG, (e - s) / FS, r["user"], s, e))
print(f"{'label':22s} {'runs':>5s} {'no marker':>9s} {'median dz (mm)':>15s} | agree at threshold  {'50 mm':>16s} {'100 mm':>16s} {'200 mm':>16s} | wrong direction (>100 mm) | moving arm right/left")
T3 = Counter()
for name in ARM_SPEC:
    sign = 1 if name.startswith("Lever") else -1
    two = "_2_" in name
    A = np.array([(a, b) for a, b, *_ in rows3[name]], float)
    ok = np.isfinite(A).all(1) if two else np.isfinite(A).any(1)
    A = A[ok]
    if two:
        key = sign * np.nanmin(sign * A, axis=1)          # the hand that moved least in the expected direction
        side = ""
    else:
        j = np.nanargmax(np.abs(np.nan_to_num(A)), axis=1)
        key = A[np.arange(len(A)), j]
        side = f"{int((j == 0).sum())}/{int((j == 1).sum())}"
    res = [int((sign * key > th).sum()) for th in (50, 100, 200)]
    wrong = int((sign * (A[np.arange(len(A)), np.nanargmax(np.abs(np.nan_to_num(A)), axis=1)]) < -100).sum())
    T3.update(n=len(A), k=res[1], w=wrong)
    print(f"{name:22s} {len(ok):5d} {int((~ok).sum()):9d} {np.median(key):+15.0f} | {'':18s} " + " ".join(f"{pct(x, len(A)):>16s}" for x in res) + f" | {wrong:25d} | {side}")
print(f"{'ALL specific':22s} {T3['n']:5d} {'':9s} {'':15s} | {'':18s} {'':16s} {pct(T3['k'], T3['n']):>16s} {'':16s} | {T3['w']:25d}")
A = np.array([(a, b) for a, b, *_ in rows3["Autre_Mouvement_Bras"]], float)
A = A[np.isfinite(A).any(1)]
key = A[np.arange(len(A)), np.nanargmax(np.abs(np.nan_to_num(A)), axis=1)]
print(f"\nFor comparison, Autre_Mouvement_Bras ({len(A)} runs): hand up > 100 mm in {pct(int((key > 100).sum()), len(A))}, "
      f"down > 100 mm in {pct(int((key < -100).sum()), len(A))}, |dz| <= 100 mm in {pct(int((np.abs(key) <= 100).sum()), len(A))}")
print("\nPer participant, one-arm + two-arm specific runs agreeing at 100 mm:")
for r in recs:
    n = k = 0
    for name in ARM_SPEC:
        sign = 1 if name.startswith("Lever") else -1
        for dD, dG, _, u, *_ in rows3[name]:
            if u != r["user"]:
                continue
            v = np.array([dD, dG])
            if "_2_" in name:
                if not np.isfinite(v).all():
                    continue
                key = sign * np.min(sign * v)
            else:
                if not np.isfinite(v).any():
                    continue
                key = v[np.nanargmax(np.abs(np.nan_to_num(v)))]
            n += 1; k += sign * key > 100
    print(f"   {r['user']}: {pct(int(k), n)}")

print("\nB3 sensitivity to label timing: the same test (100 mm) with all arm labels of a participant shifted by d seconds,")
print("and with the heights taken 0.5 s outside the run (before the start / after the end).\n")


def b3_agree(r, d=0.0, pad=0):
    n = k = 0
    N = len(r["zD"])
    for name in ARM_SPEC:
        sign = 1 if name.startswith("Lever") else -1
        for _, _, _, u, s, e in rows3[name]:
            if u != r["user"]:
                continue
            s2, e2 = int(s + d * FS) - pad, int(e - 1 + d * FS) + pad
            if s2 < 0 or e2 >= N:
                continue
            v = np.array([at(r["zD"], e2) - at(r["zD"], s2), at(r["zG"], e2) - at(r["zG"], s2)])
            if "_2_" in name:
                if not np.isfinite(v).all():
                    continue
                key = sign * np.min(sign * v)
            else:
                if not np.isfinite(v).any():
                    continue
                key = v[np.nanargmax(np.abs(np.nan_to_num(v)))]
            n += 1; k += int(sign * key > 100)
    return k, n


grid = np.round(np.arange(-2, 2.01, 0.25), 2)
print(f"{'user':6s} {'runs':>4s} {'d=0':>6s} {'d=0, 0.5 s outside':>19s} {'best d (s)':>11s} {'agree at best d':>16s}   agreement for d = -2 ... +2 s (step 0.5)")
S = Counter()
for r in recs:
    k0, n0 = b3_agree(r)
    kp, _ = b3_agree(r, pad=FS // 2)
    res = [b3_agree(r, d) for d in grid]
    j = int(np.argmax([a / max(b, 1) for a, b in res]))
    S.update(n=n0, k0=k0, kp=kp, kb=res[j][0], nb=res[j][1])
    print(f"{r['user']:6s} {n0:4d} {k0:6d} {kp:19d} {grid[j]:+11.2f} {pct(res[j][0], res[j][1]):>16s}   " + " ".join(f"{a}/{b}" for (a, b), d in zip(res, grid) if abs(d * 2 - round(d * 2)) < 1e-9))
print(f"{'ALL':6s} {S['n']:4d} {S['k0']:6d} {S['kp']:19d} {'per user':>11s} {pct(S['kb'], S['nb']):>16s}   (best d chosen per participant on the same data: optimistic)")

print("\nB3 with the right-hand SENSOR (participants with a working hand sensor): angle between the gravity direction")
print("0.25 s around the start and around the end of the run (degrees).\n")
ang = defaultdict(list)
for r in recs:
    if r["hand"] is None:
        continue
    g = np.stack([roll(r["hand"][:, k], FS // 2) for k in range(3)], 1)
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-9

    def angle(s, e):
        a, b = g[s], g[min(e, len(g) - 1)]
        return np.degrees(np.arccos(np.clip(a @ b, -1, 1))) if np.isfinite(a).all() and np.isfinite(b).all() else np.nan
    for name, s, e in runs(r["arm"]):
        if r["sus"][s:e].any():
            continue
        if name in ARM_SPEC:
            dD, dG = at(r["zD"], e - 1) - at(r["zD"], s), at(r["zG"], e - 1) - at(r["zG"], s)
            right = np.nan_to_num(abs(dD)) >= np.nan_to_num(abs(dG))
            ang[name + (" (right arm moves)" if ("_2_" in name or right) else " (left arm moves)")].append(angle(s, e))
        elif name == "Autre_Mouvement_Bras":
            ang["Autre_Mouvement_Bras"].append(angle(s, e))
        elif name == "inactif":
            for c in range(s, e - FS, FS):
                ang["arm inactif (1-s pieces)"].append(angle(c, c + FS))
print(f"{'group':40s} {'runs':>6s} {'median angle':>13s} {'> 30 deg':>18s} {'> 60 deg':>18s}")
for k in sorted(ang):
    v = np.array(ang[k]); v = v[np.isfinite(v)]
    if len(v):
        print(f"{k:40s} {len(v):6d} {np.median(v):13.1f} {pct(int((v > 30).sum()), len(v)):>18s} {pct(int((v > 60).sum()), len(v)):>18s}")

# =============================================================================== B4
print("\n" + "=" * 100, "\nB4  Hand still => no specific arm movement\n")
print("(a) Qualisys: speed of the faster hand (0.5-s mean). Share of rows where it is below the threshold.\n")
groups = {"specific arm (Lever/Baisser)": lambda r: np.isin(r["arm"], ARM_SPEC), "Autre_Mouvement_Bras": lambda r: r["arm"] == "Autre_Mouvement_Bras",
          "arm inactif": lambda r: r["arm"] == "inactif"}
print(f"{'rows':32s} {'seconds':>8s} {'< 50 mm/s':>20s} {'< 100 mm/s':>20s} {'< 200 mm/s':>20s} {'median speed':>13s}")
for gname, fn in groups.items():
    V = np.concatenate([np.fmax(r["vD"], r["vG"])[fn(r) & ~r["sus"]] for r in recs]); V = V[np.isfinite(V)]
    print(f"{gname:32s} {len(V) / FS:8.0f} " + " ".join(f"{pct(int((V < th).sum()), len(V)):>20s}" for th in (50, 100, 200)) + f" {np.median(V):13.0f}")
for th in (50, 100):
    still = np.concatenate([(np.fmax(r["vD"], r["vG"]) < th)[~r["sus"] & r["ann"]] for r in recs])
    spec = np.concatenate([np.isin(r["arm"], ARM_SPEC)[~r["sus"] & r["ann"]] for r in recs])
    print(f"Rule as an implication (hands slower than {th} mm/s => not a specific arm movement): "
          f"holds on {pct(int((still & ~spec).sum()), int(still.sum()))} of still rows; base rate of 'not specific' = {100 * (~spec).mean():.1f} %")

print("\n(b) Right-hand SENSOR: 1-s std of |a| (m/s2). Share of rows below the threshold.\n")
print(f"{'rows':32s} {'seconds':>8s} {'< 0.1':>20s} {'< 0.2':>20s} {'< 0.4':>20s} {'median':>8s}")
for gname, fn in groups.items():
    V = []
    for r in recs:
        if r["hand"] is not None:
            a = roll(np.linalg.norm(r["hand"], axis=1), FS, "std")
            V.append(a[fn(r) & ~r["sus"]])
    V = np.concatenate(V); V = V[np.isfinite(V)]
    print(f"{gname:32s} {len(V) / FS:8.0f} " + " ".join(f"{pct(int((V < th).sum()), len(V)):>20s}" for th in (0.1, 0.2, 0.4)) + f" {np.median(V):8.2f}")

# =============================================================================== B5
print("\n" + "=" * 100, "\nB5  Walking only while STANDING, and not during a posture transition (rows at 50 Hz, shown in seconds)\n")
cnt = Counter(); tr_cnt = Counter(); tot_walk = 0
for r in recs:
    w = (r["leg"] == "Marcher") & ~r["sus"]
    tot_walk += int(w.sum())
    for k in range(-1, len(C.S_CLASSES)):
        cnt[k] += int((w & (r["st"] == k)).sum())
    for name in TRANS + ["Autre_Transition"]:
        tr_cnt[name] += int((w & (r["body"] == name)).sum())
known = tot_walk - cnt[-1]
print(f"Marcher rows: {tot_walk / FS:.0f} s; posture state known on {known / FS:.0f} s")
for k, name in enumerate(C.S_ENGLISH):
    print(f"   state {name:10s} {cnt[k] / FS:8.1f} s  ({100 * cnt[k] / known:.2f} % of walking with known state)")
print(f"   => walking while STANDING: {pct(cnt[0], known)} rows; outside STANDING: {known - cnt[0]} rows ({(known - cnt[0]) / FS:.1f} s)")
per = Counter()
for r in recs:
    w = (r["leg"] == "Marcher") & ~r["sus"] & (r["st"] > 0)
    for k in np.unique(r["st"][w]):
        per[(r["user"], C.S_ENGLISH[k])] = round(float((w & (r["st"] == k)).sum()) / FS, 1)
print("   seconds of walking outside STANDING per participant/state:", dict(per))
tw = sum(tr_cnt.values())
print(f"\nMarcher during a body transition: {pct(tw, tot_walk)} rows = {tw / FS:.1f} s")
for name, v in tr_cnt.most_common():
    if v:
        print(f"   {name:18s} {v / FS:6.1f} s")
segs = n_ov = 0
for r in recs:
    tr = np.isin(r["body"], TRANS)
    for name, s, e in runs(r["leg"]):
        if name == "Marcher" and not r["sus"][s:e].any():
            segs += 1; n_ov += tr[s:e].any()
print(f"   walking segments overlapping a named transition: {pct(int(n_ov), segs)}")

# =============================================================================== B6
print("\n" + "=" * 100, "\nB6  Leg movements per posture state (seconds; posture state known)\n")
names = LEG_SPEC + ["Autre_Mouvement_Jambe", "Marcher"] + ARM_SPEC + ["Autre_Mouvement_Bras"]
tab = {n: np.zeros(len(C.S_CLASSES) + 1) for n in names}
state_tot = np.zeros(len(C.S_CLASSES))
for r in recs:
    okr = ~r["sus"] & r["ann"]
    for k in range(len(C.S_CLASSES)):
        state_tot[k] += (okr & (r["st"] == k)).sum()
    for n in names:
        col = r["leg"] if n in LEG_SPEC + ["Autre_Mouvement_Jambe", "Marcher"] else r["arm"]
        m = (col == n) & okr
        for k in range(len(C.S_CLASSES)):
            tab[n][k] += (m & (r["st"] == k)).sum()
        tab[n][-1] += (m & (r["st"] < 0)).sum()
print(f"{'label':24s} " + " ".join(f"{s:>10s}" for s in C.S_ENGLISH) + f" {'unknown':>10s}")
print(f"{'(annotated time)':24s} " + " ".join(f"{v / FS:10.0f}" for v in state_tot))
for n in names:
    print(f"{n:24s} " + " ".join(f"{v / FS:10.1f}" for v in tab[n]))
print("\nRuns of Lever_Jambe / Baisser_Jambe per posture state at the start of the run:")
c6 = Counter()
for r in recs:
    for name, s, e in runs(r["leg"]):
        if name in LEG_SPEC and not r["sus"][s:e].any():
            c6[(name, C.S_ENGLISH[r["st"][s]] if r["st"][s] >= 0 else "unknown")] += 1
for k, v in sorted(c6.items()):
    print(f"   {k[0]:14s} {k[1]:10s} {v}")
