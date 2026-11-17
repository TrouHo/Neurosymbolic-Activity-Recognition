"""How often is the posture state unknown because of Autre_Transition? (read-only, 2026-11-13)

Question of the supervisor (meeting 2026-11-12): is the case "Autre_Transition followed by rows whose
posture cannot be derived" real in the 2022 data, and how frequent is it?

For every Autre_Transition (AT) run in `etiq mvt_corps`:
  prev = last NAMED transition before it (gives posture Y_prev), next = first NAMED transition after it
  (gives posture X_next). The "unknown interval" is from the end of the AT to the start of the next
  transition of any kind (this is what emob.data.labels.posture_states masks).
Reported:
  1. counts and durations of AT runs, per participant;
  2. size of the unknown interval, compared with the labelled region and with the other cause of
     unknown posture (inconsistent pairs X_Y -> X'_Z);
  3. context of each AT: is Y_prev == X_next (the named posture is the same before and after)?
  4. check against Qualisys: head and hip height in the unknown interval compared with the participant's
     own reference heights for Y_prev and X_next (taken on rows where the posture is known);
  5. does the AT itself change the body height?
  6. what the three options (backfill / forward fill / not scored) would do.

Usage (from code/):  $env:PYTHONPATH="."; python diagnostics\\check_autre_transition.py
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
SUSPECT_S = 12.0           # same threshold as check_rules_B.py: a specific movement longer than this is a label error
AT = C.OTHER_TRANSITION
NAMED = set(C.POSTURE_TRANSITIONS)
TOL = 0.10                 # height tolerance, fraction of standing head height
MIN_GAP_S = 0.5            # unknown intervals shorter than this are not tested against Qualisys


def col(df, k):
    z = pd.to_numeric(df[k], errors="coerce").to_numpy(float).copy()
    z[z == 0] = np.nan
    return z


def med(z, s, e):
    seg = z[s:e]
    return float(np.nanmedian(seg)) if np.isfinite(seg).any() else np.nan


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.1f} %)" if b else "0/0"


recs = []
for f in sorted(glob.glob(str(C.RAW_2022 / "*.csv"))):
    user = os.path.basename(f).split("-")[0]
    df = pd.read_csv(f, low_memory=False)
    fam = labelshift.families(df, user)
    ann = annotated_mask(fam)
    body = fam[C.BODY_COL]
    idx = np.flatnonzero(ann)
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    ps = posture_states(body, ann)
    head = col(df, "z_sq_Tete")
    hip = np.nanmean(np.stack([col(df, "z_sq_D_hanche"), col(df, "z_sq_G_hanche")]), axis=0)
    recs.append(dict(user=user, fam=fam, body=body, ann=ann, lo=lo, hi=hi, ps=ps, st=ps.state_all,
                     head=head, hip=hip, H=np.nanpercentile(head, 95)))

print("Values in etiq mvt_corps (seconds, all participants):")
vc = Counter()
for r in recs:
    for name, s, e in runs(r["body"]):
        vc[name] += e - s
for k, v in vc.most_common():
    print(f"   {k:20s} {v / FS:8.1f}")

# ------------------------------------------------------------------ 1. AT runs
print("\n" + "=" * 100 + "\n1. Autre_Transition runs in etiq mvt_corps\n")
print(f"{'user':6s} {'runs':>5s} {'seconds':>8s} {'median s':>9s} {'max s':>7s} {'runs > 12 s (label error)':>26s} {'all transitions (runs)':>23s} {'AT share of runs':>17s}")
events = []                # one dict per AT run
TOT = Counter()
for r in recs:
    tr = r["ps"].transitions
    at = [(i, t) for i, t in enumerate(tr) if t[0] == AT]
    d = np.array([(t[2] - t[1]) / FS for _, t in at]) if at else np.array([0.0])
    nsus = int((d > SUSPECT_S).sum())
    TOT.update(n=len(at), s=float(d.sum()) if at else 0, sus=nsus, tr=len(tr))
    print(f"{r['user']:6s} {len(at):5d} {d.sum() if at else 0:8.1f} {np.median(d):9.2f} {d.max():7.1f} {nsus:26d} {len(tr):23d} {pct(len(at), len(tr)):>17s}")
    for i, (name, s, e) in at:
        prev = next((t for t in reversed(tr[:i]) if t[0] in NAMED), None)
        nxt = next((t for t in tr[i + 1:] if t[0] in NAMED), None)
        end_unknown = tr[i + 1][1] if i + 1 < len(tr) else r["hi"]
        events.append(dict(r=r, user=r["user"], s=s, e=e, dur=(e - s) / FS, sus=(e - s) / FS > SUSPECT_S,
                           prev=prev, nxt=nxt, next_any=tr[i + 1][0] if i + 1 < len(tr) else None,
                           prev_any=tr[i - 1][0] if i > 0 else None,
                           gs=e, ge=min(end_unknown, r["hi"]),
                           y=C.transition_states(prev[0])[1] if prev else None,
                           x=C.transition_states(nxt[0])[0] if nxt else None))
print(f"{'ALL':6s} {TOT['n']:5d} {TOT['s']:8.1f} {'':9s} {'':7s} {TOT['sus']:26d} {TOT['tr']:23d} {pct(TOT['n'], TOT['tr']):>17s}")

# ------------------------------------------------------------------ 2. size of the unknown interval
print("\n" + "=" * 100 + "\n2. Posture unknown: how much of the labelled region (first to last annotated row)\n")
print("   'after AT'      = rows between the end of an Autre_Transition and the start of the next transition")
print("   'inconsistent'  = rows between X_Y and X'_Z with X' != Y (the other cause of unknown posture)")
print("   'in transition' = rows inside any body transition (posture is changing, never scored)\n")
print(f"{'user':6s} {'region s':>9s} {'in transition':>14s} {'posture known':>14s} {'after AT':>10s} {'% region':>9s} {'of which all families empty':>28s} {'inconsistent':>13s} {'other unknown':>14s}")
T = Counter()
for r in recs:
    n = len(r["body"])
    reg = np.zeros(n, bool); reg[r["lo"]:r["hi"]] = True
    intr = np.isin(r["body"], list(NAMED | {AT})) & reg
    known = (r["st"] >= 0) & reg & ~intr
    after = np.zeros(n, bool)
    for ev in events:
        if ev["r"] is r:
            after[ev["gs"]:ev["ge"]] = True
    after &= reg & ~intr
    incons = np.zeros(n, bool)
    tr = r["ps"].transitions
    for (n1, s1, e1), (n2, s2, e2) in zip(tr[:-1], tr[1:]):
        if n1 in NAMED and n2 in NAMED and C.transition_states(n1)[1] != C.transition_states(n2)[0]:
            incons[e1:s2] = True
    incons &= reg
    other = reg & ~intr & ~known & ~after & ~incons
    empty = after & ~r["ann"]
    r["after"] = after
    R = int(reg.sum())
    T.update(R=R, intr=int(intr.sum()), known=int(known.sum()), after=int(after.sum()), empty=int(empty.sum()),
             incons=int(incons.sum()), other=int(other.sum()))
    print(f"{r['user']:6s} {R / FS:9.0f} {intr.sum() / FS:14.1f} {known.sum() / FS:14.1f} {after.sum() / FS:10.1f} {100 * after.sum() / R:8.1f}% "
          f"{empty.sum() / FS:28.1f} {incons.sum() / FS:13.1f} {other.sum() / FS:14.1f}")
print(f"{'ALL':6s} {T['R'] / FS:9.0f} {T['intr'] / FS:14.1f} {T['known'] / FS:14.1f} {T['after'] / FS:10.1f} {100 * T['after'] / T['R']:8.1f}% "
      f"{T['empty'] / FS:28.1f} {T['incons'] / FS:13.1f} {T['other'] / FS:14.1f}")

g = np.array([(ev["ge"] - ev["gs"]) / FS for ev in events])
print(f"\nLength of the unknown interval after one AT ({len(g)} intervals): median {np.median(g):.1f} s, p90 {np.percentile(g, 90):.1f} s, "
      f"max {g.max():.1f} s; zero length (next transition starts immediately): {int((g == 0).sum())}; "
      f"< 1 s: {int((g < 1).sum())}; >= 1 s: {int((g >= 1).sum())}")
print("What is labelled in the other families on 'after AT' rows (seconds):")
c = Counter()
for r in recs:
    m = r["after"]
    for k in ("etiq mvt_bras", "etiq mvt_jambe"):
        for v in np.unique(r["fam"][k][m]):
            c[(k, v or "(empty)")] += int((r["fam"][k][m] == v).sum())
for (k, v), nrow in sorted(c.items(), key=lambda kv: -kv[1]):
    print(f"   {k:16s} {v:24s} {nrow / FS:7.1f}")

# ------------------------------------------------------------------ 3. context
print("\n" + "=" * 100 + "\n3. Context of each Autre_Transition: named posture before (Y_prev) and after (X_next)\n")
ctx = Counter(); pairs = Counter()
for ev in events:
    if ev["y"] is None or ev["x"] is None:
        k = "no named transition on one side"
    elif ev["y"] == ev["x"]:
        k = "same posture before and after (Y_prev == X_next)"
    else:
        k = "different posture (Y_prev != X_next)"
    ev["ctx"] = k
    ctx[k] += 1
    pairs[(ev["y"], ev["x"])] += 1
for k, v in ctx.most_common():
    print(f"   {k:50s} {pct(v, len(events))}")
print("\n   Y_prev -> X_next (runs):")
for (y, x), v in pairs.most_common():
    print(f"      {str(y):10s} -> {str(x):10s} {v}")
print("\n   Neighbouring transition of any kind:")
print("      previous is also AT:", sum(ev["prev_any"] == AT for ev in events), "| next is also AT:", sum(ev["next_any"] == AT for ev in events))
print("\n   Per participant:")
for r in recs:
    cc = Counter(ev["ctx"] for ev in events if ev["r"] is r)
    print(f"      {r['user']}: " + "; ".join(f"{k.split(' (')[0]}: {v}" for k, v in cc.items()))

# ------------------------------------------------------------------ 4. Qualisys check
print("\n" + "=" * 100 + "\n4. Qualisys: body height in the unknown interval against the participant's own reference\n")
print(f"   Reference = median head and hip height on rows where the posture state is known (per participant, per state).")
print(f"   An interval 'matches' a state when head AND hip height are within {TOL:.2f} x standing head height of that reference.")
print(f"   Only intervals >= {MIN_GAP_S} s with markers visible are tested.\n")
print("   Reference heights (mm), head / hip:")
for r in recs:
    ref = {}
    row = []
    for k, name in enumerate(C.S_CLASSES):
        m = (r["st"] == k) & r["ann"]
        if m.sum() >= FS:
            ref[name] = (float(np.nanmedian(r["head"][m])), float(np.nanmedian(r["hip"][m])))
            row.append(f"{name} {ref[name][0]:.0f}/{ref[name][1]:.0f}")
    r["ref"] = ref
    print(f"      {r['user']}: " + " | ".join(row))


def match(r, h, p, state):
    if state not in r["ref"] or not np.isfinite(h) or not np.isfinite(p):
        return None
    rh, rp = r["ref"][state]
    return abs(h - rh) / r["H"] < TOL and abs(p - rp) / r["H"] < TOL


def nearest(r, h, p):
    best, bd = None, 9e9
    for name, (rh, rp) in r["ref"].items():
        d = max(abs(h - rh), abs(p - rp)) / r["H"]
        if d < bd:
            best, bd = name, d
    return best, bd


res = defaultdict(Counter)
near = Counter()
detail = []
for ev in events:
    r = ev["r"]
    if (ev["ge"] - ev["gs"]) / FS < MIN_GAP_S:
        res[ev["ctx"]]["too short"] += 1
        continue
    h, p = med(r["head"], ev["gs"], ev["ge"]), med(r["hip"], ev["gs"], ev["ge"])
    if not (np.isfinite(h) and np.isfinite(p)):
        res[ev["ctx"]]["no marker"] += 1
        continue
    my, mx = match(r, h, p, ev["y"]), match(r, h, p, ev["x"])
    nn, nd = nearest(r, h, p)
    res[ev["ctx"]]["tested"] += 1
    res[ev["ctx"]]["matches Y_prev (forward fill right)"] += bool(my)
    res[ev["ctx"]]["matches X_next (backfill right)"] += bool(mx)
    res[ev["ctx"]]["matches neither"] += not (my or mx)
    res[ev["ctx"]]["nearest reference is within tolerance"] += nd < TOL
    near[(ev["ctx"].split(" (")[0], nn if nd < TOL else "none of the six")] += 1
    detail.append((ev["user"], ev["s"] / FS, ev["dur"], (ev["ge"] - ev["gs"]) / FS, ev["y"], ev["x"], h, p, nn, nd, my, mx, ev["sus"]))
for k, cc in res.items():
    print(f"\n   {k}:")
    n = cc["tested"]
    for kk in ("too short", "no marker", "tested"):
        print(f"      {kk:42s} {cc[kk]}")
    for kk in ("matches Y_prev (forward fill right)", "matches X_next (backfill right)", "matches neither", "nearest reference is within tolerance"):
        print(f"      {kk:42s} {pct(cc[kk], n)}")
print("\n   Nearest reference posture of the tested intervals:")
for (k, nn), v in sorted(near.items(), key=lambda kv: -kv[1]):
    print(f"      {k:40s} {nn:16s} {v}")

print("\n   Every tested interval (t = start of the AT in s from the first row; heights in mm; d = distance to the nearest reference / standing head height):")
print(f"   {'user':6s} {'t':>7s} {'AT s':>6s} {'gap s':>6s} {'Y_prev':>9s} {'X_next':>9s} {'head':>6s} {'hip':>6s} {'nearest':>9s} {'d':>5s} {'=Y':>4s} {'=X':>4s}")
for u, t, d, gdur, y, x, h, p, nn, nd, my, mx, sus in detail:
    print(f"   {u:6s} {t:7.1f} {d:6.1f} {gdur:6.1f} {str(y):>9s} {str(x):>9s} {h:6.0f} {p:6.0f} {nn:>9s} {nd:5.2f} {str(bool(my))[0]:>4s} {str(bool(mx))[0]:>4s}" + ("   AT > 12 s" if sus else ""))

# ------------------------------------------------------------------ 5. does the AT change the height?
print("\n" + "=" * 100 + "\n5. Does Autre_Transition change the body height? (median height 0.5 s after the end minus 0.5 s before the start)\n")
dh = []
for ev in events:
    if ev["sus"]:
        continue
    r = ev["r"]
    a = med(r["head"], max(ev["s"] - FS // 2, 0), ev["s"]); b = med(r["head"], ev["e"], ev["e"] + FS // 2)
    if np.isfinite(a) and np.isfinite(b):
        dh.append((b - a) / r["H"])
dh = np.array(dh)
print(f"   AT runs tested (<= 12 s, head marker visible): {len(dh)}")
for th in (0.05, 0.10, 0.15):
    print(f"   |head change| > {th:.2f} x standing height: {pct(int((np.abs(dh) > th).sum()), len(dh))}")
ref = []
for r in recs:
    for name, s, e in r["ps"].transitions:
        if name in NAMED and (e - s) / FS <= SUSPECT_S:
            a = med(r["head"], max(s - FS // 2, 0), s); b = med(r["head"], e, e + FS // 2)
            if np.isfinite(a) and np.isfinite(b):
                ref.append(abs(b - a) / r["H"])
ref = np.array(ref)
print(f"   For comparison, the {len(ref)} named transitions: > 0.05: {pct(int((ref > 0.05).sum()), len(ref))}; > 0.10: {pct(int((ref > 0.10).sum()), len(ref))}; > 0.15: {pct(int((ref > 0.15).sum()), len(ref))}")

# ------------------------------------------------------------------ 6. the options
print("\n" + "=" * 100 + "\n6. What each option does with the 'after AT' rows\n")
same = sum((ev["ge"] - ev["gs"]) for ev in events if ev["ctx"].startswith("same")) / FS
diff = sum((ev["ge"] - ev["gs"]) for ev in events if ev["ctx"].startswith("different")) / FS
none = sum((ev["ge"] - ev["gs"]) for ev in events if ev["ctx"].startswith("no named")) / FS
print(f"   Seconds after AT where Y_prev == X_next: {same:.1f}; Y_prev != X_next: {diff:.1f}; one side missing: {none:.1f}")
print(f"   Posture-known time today: {T['known'] / FS:.0f} s. Recovering the 'same' intervals would add {100 * same * FS / T['known']:.1f} %.")

# ------------------------------------------------------------------ 7. rows BEFORE an AT (after a named transition)
print("\n" + "=" * 100 + "\n7. Rows between a named transition X_Y and a following Autre_Transition (posture set to Y)\n")
print("   The pipeline sets the posture to Y there (the name X_Y says where the transition ends).")
print("   Same Qualisys test as in section 4, on intervals >= 0.5 s. Baseline = intervals between two named")
print("   transitions that chain (X_Y then Y_Z), where the posture Y is given by both names.\n")


def test_pairs(kind):
    out = Counter(); rows = []
    for r in recs:
        tr = r["ps"].transitions
        for (n1, s1, e1), (n2, s2, e2) in zip(tr[:-1], tr[1:]):
            if n1 not in NAMED or (e1 - s1) / FS > SUSPECT_S:
                continue
            y = C.transition_states(n1)[1]
            if kind == "AT" and n2 != AT:
                continue
            if kind == "chain" and not (n2 in NAMED and C.transition_states(n2)[0] == y):
                continue
            out["n"] += 1; out["rows"] += s2 - e1
            if (s2 - e1) / FS < MIN_GAP_S:
                out["short"] += 1; continue
            h, p = med(r["head"], e1, s2), med(r["hip"], e1, s2)
            if not (np.isfinite(h) and np.isfinite(p)) or h < 150:
                out["nomarker"] += 1; continue
            m = match(r, h, p, y)
            if m is None:
                out["noref"] += 1; continue
            out["tested"] += 1; out["ok"] += bool(m)
            rows.append((r["user"], e1 / FS, (s2 - e1) / FS, n1, y, h, p, bool(m)))
    return out, rows


for kind, label in (("AT", "named X_Y -> Autre_Transition"), ("chain", "named X_Y -> named Y_Z (baseline)")):
    o, rows = test_pairs(kind)
    print(f"   {label}: {o['n']} intervals, {o['rows'] / FS:.1f} s; shorter than {MIN_GAP_S} s: {o['short']}; no valid marker: {o['nomarker']}; "
          f"no reference for Y: {o['noref']}; tested: {o['tested']}; height matches Y: {pct(o['ok'], o['tested'])}")
    if kind == "AT":
        d = np.array([x[2] for x in rows])
        print(f"      per participant (intervals): " + ", ".join(f"{u} {v}" for u, v in sorted(Counter(x[0] for x in rows).items())))
        print(f"      by Y: " + ", ".join(f"{k} {sum(x[7] for x in rows if x[4] == k)}/{v}" for k, v in sorted(Counter(x[4] for x in rows).items())))
        print(f"      {'user':6s} {'t':>7s} {'gap s':>6s} {'transition':>16s} {'head':>6s} {'hip':>6s} {'=Y':>4s}")
        for u, t, gdur, n1, y, h, p, m in rows:
            print(f"      {u:6s} {t:7.1f} {gdur:6.1f} {n1:>16s} {h:6.0f} {p:6.0f} {str(m)[0]:>4s}")

# ------------------------------------------------------------------ 8. cost of a two-sided rule
print("\n" + "=" * 100 + "\n8. Posture-known seconds per state, by how the state is determined (labelled region, rows outside transitions)\n")
print("   two-sided  = between X_Y and Y_Z (both names give Y)")
print("   before AT  = between a named X_Y and a following Autre_Transition (only the previous name gives Y)")
print("   edge       = before the first / after the last transition of the recording (only one name)\n")
tab = {k: np.zeros(len(C.S_CLASSES)) for k in ("two-sided", "before AT", "edge")}
per_user = defaultdict(lambda: np.zeros(len(C.S_CLASSES)))
for r in recs:
    n = len(r["body"]); kind = np.full(n, "", dtype=object)
    tr = r["ps"].transitions
    for (n1, s1, e1), (n2, s2, e2) in zip(tr[:-1], tr[1:]):
        if n1 in NAMED:
            kind[e1:s2] = "before AT" if n2 == AT else "two-sided"
    kind[:tr[0][1]] = "edge"; kind[tr[-1][2]:] = "edge"
    ok = (r["st"] >= 0) & (np.arange(n) >= r["lo"]) & (np.arange(n) < r["hi"])
    for k in tab:
        for i in range(len(C.S_CLASSES)):
            v = int((ok & (kind == k) & (r["st"] == i)).sum())
            tab[k][i] += v
            if k == "before AT":
                per_user[r["user"]][i] += v
print(f"   {'':12s} " + " ".join(f"{s:>10s}" for s in C.S_ENGLISH) + f" {'total':>10s}")
for k, v in tab.items():
    print(f"   {k:12s} " + " ".join(f"{x / FS:10.1f}" for x in v) + f" {v.sum() / FS:10.1f}")
tot = sum(tab.values())
print(f"   {'before AT %':12s} " + " ".join(f"{100 * a / b if b else 0:9.1f}%" for a, b in zip(tab['before AT'], tot)) + f" {100 * tab['before AT'].sum() / tot.sum():9.1f}%")
print("\n   'before AT' seconds per participant:")
for u, v in sorted(per_user.items()):
    print(f"   {u:12s} " + " ".join(f"{x / FS:10.1f}" for x in v))

# ------------------------------------------------------------------ 9. one-sided vs two-sided, per posture; real timelines
print("\n" + "=" * 100 + "\n9. Is 'before AT' less reliable because it is one-sided, or because of which postures it contains?\n")
print("   Same height test, split by the posture Y. Intervals >= 0.5 s with valid markers.\n")


def collect(kind):
    rows = []
    for r in recs:
        tr = r["ps"].transitions
        for i, ((n1, s1, e1), (n2, s2, e2)) in enumerate(zip(tr[:-1], tr[1:])):
            if n1 not in NAMED or (e1 - s1) / FS > SUSPECT_S:
                continue
            y = C.transition_states(n1)[1]
            if kind == "AT" and n2 != AT:
                continue
            if kind == "chain" and not (n2 in NAMED and C.transition_states(n2)[0] == y):
                continue
            if (s2 - e1) / FS < MIN_GAP_S:
                continue
            h, p = med(r["head"], e1, s2), med(r["hip"], e1, s2)
            if not (np.isfinite(h) and np.isfinite(p)) or h < 150:
                continue
            m = match(r, h, p, y)
            if m is None:
                continue
            # spread of the head height inside the interval: does the posture stay the same?
            seg = r["head"][e1:s2]; seg = seg[np.isfinite(seg) & (seg > 150)]
            spread = (np.percentile(seg, 95) - np.percentile(seg, 5)) / r["H"] if len(seg) > 5 else np.nan
            rows.append(dict(r=r, user=r["user"], y=y, ok=bool(m), dur=(s2 - e1) / FS, s1=s1, e1=e1, s2=s2, e2=e2, n1=n1, n2=n2, h=h, p=p, spread=spread))
    return rows


A, B = collect("AT"), collect("chain")
print(f"   {'posture Y':10s} | {'two-sided: intervals ok':>24s} {'seconds ok':>22s} | {'before AT: intervals ok':>24s} {'seconds ok':>22s}")
for y in C.S_CLASSES + ["ALL"]:
    out = []
    for rows in (B, A):
        sel = [x for x in rows if y == "ALL" or x["y"] == y]
        n, k = len(sel), sum(x["ok"] for x in sel)
        sec, seck = sum(x["dur"] for x in sel), sum(x["dur"] for x in sel if x["ok"])
        out.append(f"{pct(k, n):>24s} {seck:8.1f}/{sec:7.1f} ({100 * seck / sec if sec else 0:3.0f} %)")
    print(f"   {y:10s} | " + " | ".join(out))
# expected agreement of 'before AT' if it behaved like two-sided intervals of the same posture
exp = 0.0
for x in A:
    sel = [b for b in B if b["y"] == x["y"]]
    exp += sum(b["ok"] for b in sel) / len(sel)
print(f"\n   If 'before AT' intervals behaved like two-sided intervals of the same posture, expected matches: {exp:.1f} of {len(A)}; observed: {sum(x['ok'] for x in A)}")
print(f"   Head-height spread inside the interval (p95 - p5, fraction of standing height), median: two-sided {np.nanmedian([b['spread'] for b in B]):.3f}; before AT {np.nanmedian([a['spread'] for a in A]):.3f}")
print(f"   Intervals with spread > 0.15 (the height changes inside the interval, although no transition is labelled): "
      f"two-sided {pct(sum(b['spread'] > 0.15 for b in B), len(B))}; before AT {pct(sum(a['spread'] > 0.15 for a in A), len(A))}")
for y in C.S_CLASSES:
    sb = [b for b in B if b["y"] == y]; sa = [a for a in A if a["y"] == y]
    print(f"      {y:10s} two-sided {pct(sum(b['spread'] > 0.15 for b in sb), len(sb)):>18s}; before AT {pct(sum(a['spread'] > 0.15 for a in sa), len(sa))}")


def timeline(x, title):
    r = x["r"]
    a, b = max(x["s1"] - FS, 0), min(x["e2"] + FS, len(r["body"]))
    step = FS // 2 if (b - a) / FS <= 14 else FS
    ref = r["ref"]
    print(f"\n   --- {title}: {x['user']}, {x['n1']} at {x['s1'] / FS:.1f} s, then {x['dur']:.1f} s without body label, then {x['n2']}")
    print(f"       pipeline posture for the gap: {x['y']} (reference head/hip {ref[x['y']][0]:.0f}/{ref[x['y']][1]:.0f} mm); "
          f"standing reference {ref['Debout'][0]:.0f}/{ref['Debout'][1]:.0f} mm; gap median {x['h']:.0f}/{x['p']:.0f} mm -> {'matches' if x['ok'] else 'does NOT match'}")
    print(f"       {'t (s)':>7s} {'mvt_corps':18s} {'mvt_bras':22s} {'mvt_jambe':22s} {'head mm':>8s} {'hip mm':>7s}")
    for s in range(a, b, step):
        c = min(s + step // 2, len(r["body"]) - 1)
        mark = " <- gap" if x["e1"] <= c < x["s2"] else ""
        print(f"       {s / FS:7.1f} {r['body'][c] or '(empty)':18s} {r['fam']['etiq mvt_bras'][c] or '(empty)':22s} {r['fam']['etiq mvt_jambe'][c] or '(empty)':22s} "
              f"{med(r['head'], s, s + step):8.0f} {med(r['hip'], s, s + step):7.0f}{mark}")


print("\n   Real cases 'named transition -> gap -> Autre_Transition' (time from the first CSV row; one line per 0.5 s or 1 s):")
for x in A:
    if not x["ok"]:
        timeline(x, "NO MATCH")
good = sorted([x for x in A if x["ok"]], key=lambda x: -x["dur"])
for x in good[:3]:
    timeline(x, "MATCH")
