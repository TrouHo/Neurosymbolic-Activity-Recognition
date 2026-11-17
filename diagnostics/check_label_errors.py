"""Three remaining label problems, checked against motion capture (read-only, 2026-11-13).

A. Labels opened and never closed: runs of a specific movement longer than 12 s. Where does the movement
   really end? Compared with the contributor's rule "the run ends when the next annotated movement begins".
B. Pairs of named transitions that do not chain (X_Y followed by X'_Z with X' != Y): what happens in
   between? A transition without label, or a wrong name?
C. Rows whose derived posture is "standing" while the trunk is inclined by more than 30 deg: next to a
   labelled bend (imprecise boundary) or isolated (a bend without label)?

Labels are shifted by the approved per-participant offsets when EMOB_LABEL_SHIFT=1.
Usage (from code/):  $env:PYTHONPATH="."; $env:EMOB_LABEL_SHIFT="1"; python -W ignore diagnostics\\check_label_errors.py
"""
from __future__ import annotations

import glob
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import labelshift
from emob import config as C
from emob.data.labels import annotated_mask, posture_states, runs

FS = 50
SUSPECT_S = 12.0
AT = C.OTHER_TRANSITION
NAMED = set(C.POSTURE_TRANSITIONS)
GENERIC = {"inactif", "Marcher", "Autre_Mouvement_Bras", "Autre_Mouvement_Jambe", "Debut", "Fin", ""}
COLS = {"body": "etiq mvt_corps", "arm": "etiq mvt_bras", "leg": "etiq mvt_jambe"}
GROUP = {"body": ("Tete", "Torse"), "arm": ("D_main", "G_main"), "leg": ("D_pied", "G_pied")}
STILL = 100.0   # mm/s

print(labelshift.banner() + "\n")


def xyz(df, k):
    p = df[[f"x_sq_{k}", f"y_sq_{k}", f"z_sq_{k}"]].apply(pd.to_numeric, errors="coerce").to_numpy(float).copy()
    p[(p == 0).all(1) | ~np.isfinite(p).all(1)] = np.nan
    return p


def sm(x, n):
    return pd.Series(x).rolling(n, center=True, min_periods=max(2, n // 2)).mean().to_numpy()


def speed(p):
    q = np.stack([sm(p[:, a], 5) for a in range(3)], 1)
    v = np.linalg.norm(np.gradient(q, axis=0) * FS, axis=1)
    v[v > 3000] = np.nan
    return sm(v, FS // 2)


def med(z, s, e):
    seg = z[max(s, 0):max(e, 0)]
    return float(np.nanmedian(seg)) if np.isfinite(seg).any() else np.nan


recs = []
for f in sorted(glob.glob(str(C.RAW_2022 / "*.csv"))):
    user = os.path.basename(f).split("-")[0]
    df = pd.read_csv(f, low_memory=False)
    fam = labelshift.families(df, user)
    ann = annotated_mask(fam)
    idx = np.flatnonzero(ann); lo, hi = int(idx[0]), int(idx[-1]) + 1
    ps = posture_states(fam[C.BODY_COL], ann)
    P = {k: xyz(df, k) for k in ("Tete", "Torse", "D_main", "G_main", "D_pied", "G_pied", "D_epaule", "G_epaule", "D_hanche", "G_hanche")}
    spd = {g: np.fmax(speed(P[a]), speed(P[b])) for g, (a, b) in GROUP.items()}
    z = {"body": sm(P["Tete"][:, 2], 5), "arm": np.fmax(sm(P["D_main"][:, 2], 5), sm(P["G_main"][:, 2], 5)),
         "leg": np.fmax(sm(P["D_pied"][:, 2], 5), sm(P["G_pied"][:, 2], 5))}
    zmin = {"arm": np.fmin(sm(P["D_main"][:, 2], 5), sm(P["G_main"][:, 2], 5))}
    sh = np.nanmean(np.stack([P["D_epaule"], P["G_epaule"]]), axis=0); hp = np.nanmean(np.stack([P["D_hanche"], P["G_hanche"]]), axis=0)
    v = sh - hp; L = np.linalg.norm(v, axis=1)
    ang = np.degrees(np.arccos(np.clip(v[:, 2] / L, -1, 1))); Lm = np.nanmedian(L)
    ang[(L < 0.6 * Lm) | (L > 1.4 * Lm)] = np.nan
    ang = pd.Series(ang).rolling(FS // 4, center=True, min_periods=3).median().to_numpy()
    head = sm(P["Tete"][:, 2], 5); hip = sm(hp[:, 2], 5)
    st = ps.state_all
    ref = {}
    for k, name in enumerate(C.S_CLASSES):
        m = (st == k) & ann
        if m.sum() >= FS:
            ref[name] = (float(np.nanmedian(head[m])), float(np.nanmedian(hip[m])), float(np.nanmedian(ang[m])))
    vz = lambda a: np.gradient(sm(P[a][:, 2], 5)) * FS
    nsum = lambda a, b: np.where(np.isfinite(a) | np.isfinite(b), np.nan_to_num(a) + np.nan_to_num(b), np.nan)
    svz = {"body": sm(vz("Tete"), 10), "arm": sm(nsum(vz("D_main"), vz("G_main")), 10), "leg": sm(nsum(vz("D_pied"), vz("G_pied")), 10)}
    act = labelshift.shifted(df[C.ACT_COL].astype("string").fillna("").to_numpy(object), int(round(labelshift.shift_s(user) * FS)))
    recs.append(dict(user=user, fam=fam, ann=ann, lo=lo, hi=hi, ps=ps, st=st, spd=spd, z=z, zmin=zmin, ang=ang, head=head, hip=hip, svz=svz, act=act,
                     H=np.nanpercentile(head, 95), ref=ref, n=len(df)))

# ================================================================================================ A
print("=" * 112 + "\nA. Labels opened and never closed (a specific movement longer than 12 s)\n")
dur = defaultdict(list)
for r in recs:
    for fam_name, col in COLS.items():
        for name, s, e in runs(r["fam"][col]):
            if name not in GENERIC and (e - s) / FS <= SUSPECT_S:
                dur[name].append((e - s) / FS)


def movement_end(r, family, s, limit):
    """First time after the start where the markers of the family stay below STILL for 0.5 s, once they have moved."""
    v = r["spd"][family]
    moved = False
    for i in range(s, min(limit, r["n"] - FS // 2)):
        if np.isfinite(v[i]) and v[i] >= STILL:
            moved = True
        if moved and np.nanmax(v[i:i + FS // 2]) < STILL:
            return i
    return None


cases = []
for r in recs:
    for fam_name, col in COLS.items():
        for name, s, e in runs(r["fam"][col]):
            if name in GENERIC or (e - s) / FS <= SUSPECT_S:
                continue
            others = []
            for f2, c2 in COLS.items():
                if f2 == fam_name:
                    continue
                for n2, s2, e2 in runs(r["fam"][c2]):
                    if s2 > s and n2 not in ("inactif", "Debut", "Fin", ""):
                        others.append((s2, f2, n2)); break
            nxt = min(others) if others else None
            qe = movement_end(r, fam_name, s, e)
            d = np.array(dur[name])
            cases.append(dict(r=r, family=fam_name, name=name, s=s, e=e, nxt=nxt, qe=qe, typ=d))
            print(f"--- {r['user']}  {name}  ({fam_name} family)  labelled {s / FS:.1f} -> {e / FS:.1f} s = {(e - s) / FS:.1f} s")
            print(f"    usual duration of {name} (other runs, {len(d)}): median {np.median(d):.2f} s, p95 {np.percentile(d, 95):.2f} s, max {d.max():.2f} s")
            print(f"    contributor's rule (next annotated movement in another family): "
                  + (f"{nxt[2]} ({nxt[1]}) at {nxt[0] / FS:.1f} s -> run of {(nxt[0] - s) / FS:.2f} s" if nxt else "none"))
            print(f"    motion capture ({'/'.join(GROUP[fam_name])} below {STILL:.0f} mm/s for 0.5 s after having moved): "
                  + (f"at {qe / FS:.1f} s -> run of {(qe - s) / FS:.2f} s" if qe else "not found inside the run"))
            what_next = r["fam"][col][e] if e < r["n"] else ""
            print(f"    label that follows in the same family at {e / FS:.1f} s: {what_next or '(empty)'}")
            zz = r["z"][fam_name]
            print(f"    {'t (s)':>7s} {'mvt_corps':18s} {'mvt_bras':22s} {'mvt_jambe':22s} {'z mm':>6s} {'speed':>6s} {'head':>6s} {'hip':>5s} {'trunk':>6s}")
            stop = min(e, s + 9 * FS)
            for i in list(range(max(s - FS, 0), stop, FS // 2)) + list(range(max(stop, e - 2 * FS), min(e + 2 * FS, r["n"]), FS // 2)):
                if i == max(stop, e - 2 * FS) and stop < e - 2 * FS:
                    print(f"    {'...':>7s}")
                c = min(i + FS // 4, r["n"] - 1)
                print(f"    {i / FS:7.1f} {r['fam'][COLS['body']][c] or '(empty)':18s} {r['fam'][COLS['arm']][c] or '(empty)':22s} {r['fam'][COLS['leg']][c] or '(empty)':22s} "
                      f"{med(zz, i, i + FS // 2):6.0f} {med(r['spd'][fam_name], i, i + FS // 2):6.0f} {med(r['head'], i, i + FS // 2):6.0f} {med(r['hip'], i, i + FS // 2):5.0f} {med(r['ang'], i, i + FS // 2):6.0f}")
            print()

print("Summary A (run length in s under each rule):")
print(f"{'user':6s} {'label':18s} {'labelled':>9s} {'usual median':>13s} {'usual max':>10s} {'contributor rule':>17s} {'motion capture':>15s}")
for c in cases:
    print(f"{c['r']['user']:6s} {c['name']:18s} {(c['e'] - c['s']) / FS:9.1f} {np.median(c['typ']):13.2f} {c['typ'].max():10.2f} "
          f"{(c['nxt'][0] - c['s']) / FS if c['nxt'] else float('nan'):17.2f} {(c['qe'] - c['s']) / FS if c['qe'] else float('nan'):15.2f}")

# ---- A2: is each long run "one real movement at the start + one real movement at the end, same label"?
print("\nA2. Structure of each long run. Signed vertical velocity of the family's marker in the direction the label")
print("    implies (up for Lever / X_Debout, down for Baisser / Debout_X). A movement = velocity above 150 mm/s; it ends")
print("    when the velocity stays below 50 mm/s for 0.2 s. For Autre_Transition (no direction) the speed rule of A is used.\n")
SIGN = {**{k: -1 for k in ("Debout_Assis", "Assis_Couche", "Debout_Penche", "Debout_Agenou", "Debout_Accroupi", "Baisser_Bras", "Baisser_2_Bras", "Baisser_Jambe")},
        **{k: 1 for k in ("Assis_Debout", "Couche_Assis", "Penche_Debout", "Agenou_Debout", "Accroupi_Debout", "Lever_Bras", "Lever_2_Bras", "Lever_Jambe")}}


def first_movement(sv, s, limit):
    on = next((i for i in range(s, min(s + 3 * FS, limit)) if np.isfinite(sv[i]) and sv[i] > 150), None)
    if on is None:
        return None
    for j in range(on, limit):
        seg = sv[j:j + 10]
        if np.isfinite(seg).any() and np.nanmax(seg) < 50:
            return j
    return None


def last_movement(sv, e, floor):
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


print(f"{'user':6s} {'label':18s} | {'first movement':>32s} | {'last movement':>32s} | {'middle s':>8s} {'markers > 100 mm/s':>19s} {'head events':>12s} | activite in the middle (top 3) ")
A2 = []
for c in cases:
    r, s, e, famn, name = c["r"], c["s"], c["e"], c["family"], c["name"]
    zz = r["z"][famn] if not (famn == "arm" and name.startswith("Baisser")) else r["z"][famn]
    if name in SIGN:
        sv = SIGN[name] * r["svz"][famn]
        fe = first_movement(sv, s, e)
        lm = last_movement(sv, e, (fe or s) + FS)
    else:
        fe, lm = c["qe"], None
    mid_s, mid_e = (fe or s), (lm[0] if lm else e)
    v = r["spd"][famn][mid_s:mid_e]; v = v[np.isfinite(v)]
    h = r["head"]; d = np.abs(np.roll(h, -FS // 2) - np.roll(h, FS // 2)) / r["H"]
    ev = len([1 for _, a, b in runs(np.where(np.nan_to_num(d[mid_s:mid_e]) > 0.15, "E", "").astype(object))])
    vc = Counter(r["act"][mid_s:mid_e]).most_common(3)
    A2.append(dict(user=r["user"], name=name, family=famn, s=s, e=e, fe=fe, lm=lm))
    f1 = f"{s / FS:.1f} -> {fe / FS:.1f} s ({(fe - s) / FS:.2f} s)" if fe else "not found"
    f2 = f"{lm[0] / FS:.1f} -> {lm[1] / FS:.1f} s ({(lm[1] - lm[0]) / FS:.2f} s)" if lm else ("not determined" if name not in SIGN else "none found")
    print(f"{r['user']:6s} {name:18s} | {f1:>32s} | {f2:>32s} | {(mid_e - mid_s) / FS:8.1f} {100 * (v > 100).mean() if len(v) else float('nan'):18.0f}% {ev:12d} | "
          + ", ".join(f"{k or '(empty)'} {100 * n_ / max(mid_e - mid_s, 1):.0f}%" for k, n_ in vc))
print("\n    'head events' = number of changes of head height larger than 15 % of the standing height within 1 s, in the middle part")
print("    (each one is a change of posture that the body family cannot show while its column is occupied by the long run).")

# ================================================================================================ B
print("\n" + "=" * 112 + "\nB. Pairs of named transitions that do not chain: X_Y, then X'_Z with X' != Y\n")
print("   'start' / 'end' = median over the first / last 0.5 s of the rows between the two transitions.")
print("   'jump' = largest change of head height over 1 s between the two transitions (fraction of standing height) and its time.")
print("   Reference per participant (head / hip mm, trunk deg) is printed first.\n")
for r in recs:
    print(f"   {r['user']}: " + " | ".join(f"{k} {v[0]:.0f}/{v[1]:.0f}/{v[2]:.0f}" for k, v in r["ref"].items()))
print()
print(f"{'user':6s} {'t':>7s} {'first':>16s} {'gap s':>6s} {'second':>16s} {'unlabelled %':>12s} | {'start head/hip/trunk':>21s} | {'end head/hip/trunk':>21s} | {'jump':>6s} {'at t':>7s} | body-other labels in the gap")
tot = Counter(); pairs = Counter()
for r in recs:
    tr = r["ps"].transitions
    body = r["fam"][COLS["body"]]
    for (n1, s1, e1), (n2, s2, e2) in zip(tr[:-1], tr[1:]):
        if n1 in NAMED and n2 in NAMED and C.transition_states(n1)[1] != C.transition_states(n2)[0]:
            g = (s2 - e1)
            h = r["head"]
            if g > FS:
                d = np.abs(h[e1 + FS:s2] - h[e1:s2 - FS]) / r["H"]
                j = int(np.nanargmax(d)) if np.isfinite(d).any() else 0
                jump, jt = (float(d[j]), (e1 + j + FS // 2) / FS) if np.isfinite(d).any() else (np.nan, np.nan)
            else:
                jump, jt = np.nan, np.nan
            oth = Counter()
            for cname in ("arm", "leg"):
                v = r["fam"][COLS[cname]][e1:s2]
                for k2, n_ in Counter(v).items():
                    if k2 not in ("", "inactif"):
                        oth[k2] += n_
            pairs[(n1, n2)] += 1; tot["n"] += 1; tot["s"] += g
            print(f"{r['user']:6s} {e1 / FS:7.1f} {n1:>16s} {g / FS:6.1f} {n2:>16s} {100 * (1 - r['ann'][e1:s2].mean()) if g else 0:11.0f}% | "
                  f"{med(h, e1, e1 + FS // 2):6.0f}/{med(r['hip'], e1, e1 + FS // 2):5.0f}/{med(r['ang'], e1, e1 + FS // 2):4.0f}     | "
                  f"{med(h, s2 - FS // 2, s2):6.0f}/{med(r['hip'], s2 - FS // 2, s2):5.0f}/{med(r['ang'], s2 - FS // 2, s2):4.0f}     | "
                  f"{jump:6.2f} {jt:7.1f} | " + ", ".join(f"{k2} {v / FS:.1f}s" for k2, v in oth.most_common(3)))
print(f"\n   {tot['n']} pairs, {tot['s'] / FS:.1f} s between them. By type: " + ", ".join(f"{a}->{b} {v}" for (a, b), v in pairs.most_common()))

# ================================================================================================ C
print("\n" + "=" * 112 + "\nC. Derived posture STANDING with the trunk inclined > 30 deg for at least 0.5 s\n")
print("   'to bend label' = time from the episode to the nearest Debout_Penche / Penche_Debout label (0 = touching).")
print("   'hip' = lowest hip height in the episode / standing hip height (a low value means squatting or kneeling, not bending).\n")
print(f"{'user':6s} {'t':>7s} {'dur':>5s} {'max deg':>8s} {'hip':>5s} {'to bend label':>14s} {'to any transition':>18s} {'rows unlabelled':>16s}  arm / leg labels")
E = []
for r in recs:
    body = r["fam"][COLS["body"]]
    intr = np.isin(body, list(NAMED | {AT}))
    m = (r["st"] == 0) & ~intr & np.isfinite(r["ang"]) & (r["ang"] > 30)
    m[:r["lo"]] = False; m[r["hi"]:] = False
    bend = [(s, e) for n_, s, e in r["ps"].transitions if n_ in ("Debout_Penche", "Penche_Debout")]
    anyt = [(s, e) for n_, s, e in r["ps"].transitions]
    hipref = r["ref"]["Debout"][1]

    def dist(lst, s, e):
        return min([max(0, s - b, a - e) for a, b in lst] or [10 ** 9]) / FS
    for _, s, e in runs(np.where(m, "B", "").astype(object)):
        if (e - s) / FS < 0.5:
            continue
        lab = Counter()
        for cname in ("arm", "leg"):
            for k2, n_ in Counter(r["fam"][COLS[cname]][s:e]).items():
                if k2 not in ("", "inactif"):
                    lab[k2] += n_
        d = dict(user=r["user"], t=s / FS, dur=(e - s) / FS, mx=float(np.nanmax(r["ang"][s:e])), hip=float(np.nanmin(r["hip"][s:e]) / hipref),
                 db=dist(bend, s, e), da=dist(anyt, s, e), un=float(1 - r["ann"][s:e].mean()), lab=lab)
        E.append(d)
        print(f"{d['user']:6s} {d['t']:7.1f} {d['dur']:5.1f} {d['mx']:8.0f} {d['hip']:5.2f} {d['db']:14.1f} {d['da']:18.1f} {100 * d['un']:15.0f}%  "
              + ", ".join(f"{k2} {v / FS:.1f}s" for k2, v in d["lab"].most_common(2)))


def cls(d):
    if d["hip"] < 0.75:
        return "hip low (squat / kneel, not a bend)"
    if d["db"] <= 0.5:
        return "touches a labelled bend (boundary)"
    if d["da"] <= 0.5:
        return "touches another transition"
    return "isolated (bend without label)"


print("\n   Classes:")
c = Counter(cls(d) for d in E); sec = defaultdict(float)
for d in E:
    sec[cls(d)] += d["dur"]
for k, v in c.most_common():
    print(f"      {k:40s} {v:3d} episodes {sec[k]:6.1f} s")
print("   Per participant: " + "; ".join(f"{u}: " + ", ".join(f"{k.split(' (')[0]} {v}" for k, v in Counter(cls(d) for d in E if d['user'] == u).items()) for u in sorted({d['user'] for d in E})))
iso = [d for d in E if cls(d).startswith("isolated")]
print(f"   Isolated episodes: {len(iso)}, {sum(d['dur'] for d in iso):.1f} s; with Marcher labelled: {sum('Marcher' in d['lab'] for d in iso)}; "
      f"with an arm label: {sum(any(k.endswith('Bras') for k in d['lab']) for d in iso)}; on rows without any label (> 50 %): {sum(d['un'] > 0.5 for d in iso)}")
