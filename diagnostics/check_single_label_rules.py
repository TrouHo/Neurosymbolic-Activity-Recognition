"""One label per time step from three label families: candidate rules, seconds kept per class (2026-11-15).

For the meeting of 2026-11-16. Read-only. Rows at 50 Hz, shown in seconds.

0. How often several families are active, and how the existing column `activite` chooses.
1. Candidate rules; seconds per class under each rule (long runs treated as proposed on 2026-11-13).
2. What to do with rows where one family is "not annotated" (the middle of the six long runs):
   U1 use the other families / U2 no label on these rows / U3 keep the label only where it does not
   depend on the missing family.
3. Fragmentation of the derived label, and classes per participant.

Usage (from code/):  $env:PYTHONPATH="."; $env:EMOB_LABEL_SHIFT="1"; python -W ignore diagnostics\\check_single_label_rules.py
"""
from __future__ import annotations

from collections import Counter

import numpy as np

import labelrepair as LR
import labelshift
from emob import config as C
from emob.data.labels import runs

FS = 50
FAMS = ["body", "leg", "arm"]
COL = LR.COLS
CLASSES = C.L2_CLASSES
CLS_FAM = {c: ("arm" if c.endswith("Bras") else "leg" if (c.endswith("Jambe") or c == "Marcher") else "body") for c in CLASSES}
CATCHALL = {"Autre_Transition", "Autre_Mouvement_Bras", "Autre_Mouvement_Jambe"}
TIER = {c: (2 if c in CATCHALL else 1 if c == "Marcher" else 0) for c in CLASSES}
CAT2FAM = {"mvt_corps": "body", "mvt_bras": "arm", "mvt_jambe": "leg"}
INF = 10 ** 12

print(labelshift.banner() + "\n")
recs = LR.load()
LR.print_long_runs(recs)

for r in recs:
    r["lab"] = {f: r["rep"][COL[f]] for f in FAMS}                               # after the proposed repair
    r["on"] = {f: np.isin(r["lab"][f], CLASSES) for f in FAMS}
    r["raw_on"] = {f: np.isin(r["fam"][COL[f]], CLASSES) for f in FAMS}
    r["onset"] = {}
    for f in FAMS:
        o = np.full(r["n"], -1)
        for name, s, e in runs(r["lab"][f]):
            o[s:e] = s
        r["onset"][f] = o
    r["un"] = {f: r["unann"][COL[f]] for f in FAMS}
    r["any_un"] = np.any([r["un"][f] for f in FAMS], axis=0)

# ================================================================================================ 0
print("=" * 118 + "\n0. Overlap of the families, and how `activite` chooses (labels as in the CSV columns, shifted; long runs included)\n")
print(f"{'user':6s} {'1 family':>9s} {'2':>7s} {'3':>7s} {'>=2 / labelled':>15s} | activite on rows with >= 2 families: share explained by ...")
print(f"{'':6s} {'':9s} {'':7s} {'':7s} {'':15s} | {'label started first':>20s} {'best fixed order':>22s} {'body>leg>arm':>13s} {'arm>leg>body':>13s}")
T = Counter()
for r in recs:
    nb = sum(r["raw_on"][f].astype(int) for f in FAMS)
    cat = np.array([CAT2FAM.get(x, "") for x in r["cat"]], dtype=object)
    ov = (nb >= 2) & (cat != "")
    ons = {}
    for f in FAMS:
        o = np.full(r["n"], INF)
        for name, s, e in runs(r["fam"][COL[f]]):
            if name in CLASSES:
                o[s:e] = s
        ons[f] = o
    first = np.array(FAMS, dtype=object)[np.stack([ons[f] for f in FAMS], 1).argmin(1)]
    res = {}
    for order in (("body", "leg", "arm"), ("body", "arm", "leg"), ("arm", "leg", "body"), ("arm", "body", "leg"), ("leg", "arm", "body"), ("leg", "body", "arm")):
        pick = np.array(order, dtype=object)[np.stack([r["raw_on"][f] for f in order], 1).argmax(1)]
        res[">".join(order)] = int((pick == cat)[ov].sum())
    best = max(res, key=res.get)
    k1 = int((first == cat)[ov].sum()); n = int(ov.sum())
    T.update(n=n, first=k1, bla=res["body>leg>arm"], alb=res["arm>leg>body"], one=int((nb == 1).sum()), two=int((nb == 2).sum()), three=int((nb == 3).sum()))
    print(f"{r['user']:6s} {(nb == 1).sum() / FS:9.1f} {(nb == 2).sum() / FS:7.1f} {(nb == 3).sum() / FS:7.1f} {100 * (nb >= 2).sum() / max((nb >= 1).sum(), 1):14.1f}% | "
          f"{100 * k1 / n:19.1f}% {best + f' {100 * res[best] / n:.1f}%':>22s} {100 * res['body>leg>arm'] / n:12.1f}% {100 * res['arm>leg>body'] / n:12.1f}%")
print(f"{'ALL':6s} {T['one'] / FS:9.1f} {T['two'] / FS:7.1f} {T['three'] / FS:7.1f} {100 * (T['two'] + T['three']) / (T['one'] + T['two'] + T['three']):14.1f}% | "
      f"{100 * T['first'] / T['n']:19.1f}% {'':>22s} {100 * T['bla'] / T['n']:12.1f}% {100 * T['alb'] / T['n']:12.1f}%")

# ================================================================================================ 1
# seconds of each class in its own family column (after repair) -> upper bound, and rarity order
own = Counter()
for r in recs:
    for f in FAMS:
        for c, k in Counter(r["lab"][f][r["on"][f]]).items():
            own[c] += k
RARE = {c: i for i, c in enumerate(sorted(CLASSES, key=lambda c: own[c]))}        # 0 = rarest


def keys(r, rule):
    """Per family: key per row (lower wins; INF where the family has no label) and the best key the family could have."""
    out, best = {}, {}
    for f in FAMS:
        lab, on = r["lab"][f], r["on"][f]
        if rule[0] == "prio":
            k = np.full(r["n"], float(rule[1].index(f))); b = float(rule[1].index(f))
        elif rule[0] == "tier":
            rank = rule[1].index(f)
            k = np.array([TIER.get(x, 9) * 10 + rank for x in lab], float)
            b = min(TIER[c] for c in CLASSES if CLS_FAM[c] == f) * 10 + rank
        elif rule[0] == "rare":
            k = np.array([RARE.get(x, 99) for x in lab], float)
            b = min(RARE[c] for c in CLASSES if CLS_FAM[c] == f)
        elif rule[0] == "first":
            k = r["onset"][f].astype(float) + 0.1 * FAMS.index(f); b = -INF          # an unseen label could have started any time
        elif rule[0] == "last":
            k = -r["onset"][f].astype(float) + 0.1 * FAMS.index(f); b = -INF
        out[f] = np.where(on, k, INF); best[f] = b
    return out, best


def derive(r, rule):
    """Derived label, and the mask of rows whose label does not depend on a not-annotated family."""
    if rule[0] == "activite":
        lab = np.where(np.isin(r["act"], CLASSES), r["act"], "").astype(object)
        # the stuck label is the value of activite in the middle of the long runs: not usable there
        return lab, ~r["any_un"]
    k, best = keys(r, rule)
    K = np.stack([k[f] for f in FAMS], 1)
    j = K.argmin(1); win = K.min(1)
    lab = np.array([r["lab"][f] for f in FAMS], dtype=object)[j, np.arange(r["n"])]
    lab = np.where(win < INF, lab, "").astype(object)
    safe = np.ones(r["n"], bool)
    for f in FAMS:
        safe &= ~(r["un"][f] & ~(win < best[f]))
    return lab, safe


RULES = [
    ("R0 activite as given", ("activite",)),
    ("R1 body>leg>arm", ("prio", ("body", "leg", "arm"))),
    ("R2 body>arm>leg", ("prio", ("body", "arm", "leg"))),
    ("R3 arm>leg>body", ("prio", ("arm", "leg", "body"))),
    ("R4 specific>walk>other", ("tier", ("body", "leg", "arm"))),
    ("R5 rarest class", ("rare",)),
    ("R6 started first", ("first",)),
    ("R7 started last", ("last",)),
]
SHORT = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]

print("\n" + "=" * 118 + "\n1. Seconds per class under each rule\n")
print("   R4 tiers: specific movements (10 named transitions, 4 arm raise/lower, 2 leg raise/lower) > Marcher > Autre_*; ties body > leg > arm.")
print("   R5 order (rarest first, seconds in the own family column): " + ", ".join(f"{c} {own[c] / FS:.0f}" for c in sorted(CLASSES, key=lambda c: own[c])))
print("   Long runs: first / last movement keep the label, the middle is 'not annotated' for that family.")
print("   This table uses U3 (see section 2): a row keeps its label only if the label does not depend on the missing family.\n")
res = {}
for (name, rule), sh in zip(RULES, SHORT):
    cnt = {u: Counter() for u in ("U1", "U2", "U3")}; per_user = {}; segs = []
    for r in recs:
        lab, safe = derive(r, rule)
        m1 = lab != ""
        if rule[0] == "activite":
            m1u1 = m1                                        # U1 does not exist for activite: the stuck label stays
        cnt["U1"].update(Counter(lab[m1]))
        cnt["U2"].update(Counter(lab[m1 & ~r["any_un"]]))
        cnt["U3"].update(Counter(lab[m1 & safe]))
        final = np.where(safe, lab, "").astype(object)
        per_user[r["user"]] = Counter(final[final != ""])
        segs += [(e - s) / FS for _, s, e in runs(final)]
        r.setdefault("derived", {})[sh] = final
    res[sh] = dict(cnt=cnt, per_user=per_user, segs=np.array(segs))

hdr = f"{'class':24s} {'own col.':>9s} " + " ".join(f"{s:>7s}" for s in SHORT)
print(hdr)
for c in CLASSES:
    print(f"{c:24s} {own[c] / FS:9.1f} " + " ".join(f"{res[s]['cnt']['U3'][c] / FS:7.1f}" for s in SHORT))
print(f"{'TOTAL labelled':24s} {'':9s} " + " ".join(f"{sum(res[s]['cnt']['U3'].values()) / FS:7.0f}" for s in SHORT))
for th in (10, 20, 30):
    print(f"{'classes below ' + str(th) + ' s':24s} {sum(own[c] / FS < th for c in CLASSES):9d} " + " ".join(f"{sum(res[s]['cnt']['U3'][c] / FS < th for c in CLASSES):7d}" for s in SHORT))
print(f"{'smallest class (s)':24s} {min(own[c] for c in CLASSES) / FS:9.1f} " + " ".join(f"{min(res[s]['cnt']['U3'][c] for c in CLASSES) / FS:7.1f}" for s in SHORT))
grp = {"16 specific classes": [c for c in CLASSES if TIER[c] == 0], "Marcher": ["Marcher"], "3 catch-all (Autre_*)": [c for c in CLASSES if TIER[c] == 2]}
for g, cl in grp.items():
    print(f"{g:24s} {sum(own[c] for c in cl) / FS:9.1f} " + " ".join(f"{sum(res[s]['cnt']['U3'][c] for c in cl) / FS:7.1f}" for s in SHORT))
print(f"{'share of own col., spec.':24s} {'':9s} " + " ".join(f"{100 * sum(res[s]['cnt']['U3'][c] for c in grp['16 specific classes']) / sum(own[c] for c in grp['16 specific classes']):6.0f}%" for s in SHORT))
print("\n   Names of the rules: " + "; ".join(n for n, _ in RULES))

# ================================================================================================ 2
print("\n" + "=" * 118 + "\n2. Rows where one family is 'not annotated' (middle of the long runs)\n")
tot_un = sum(int(r["any_un"].sum()) for r in recs)
print("   Rows concerned: " + ", ".join(f"{r['user']} {r['any_un'].sum() / FS:.1f} s" for r in recs if r["any_un"].any()) + f"; total {tot_un / FS:.1f} s")
for r in recs:
    if r["any_un"].any():
        print(f"      {r['user']}: not annotated per family: " + ", ".join(f"{f} {r['un'][f].sum() / FS:.1f} s" for f in FAMS if r["un"][f].any())
              + f"; `activite` equals a stuck label on {np.mean([any(r['act'][i] == lr['name'] and lr['mid'][0] <= i < lr['mid'][1] for lr in r['long_runs']) for i in np.flatnonzero(r['any_un'])[::5]]) * 100:.0f} % of these rows")
print("\n   U1 = use the other families as if the missing one had no label; U2 = no label on any of these rows;")
print("   U3 = keep the label only where it cannot be changed by the missing family (the winner outranks every label that family could have).\n")
print(f"{'rule':26s} {'labelled s (U1)':>16s} {'U2':>8s} {'U3':>8s} | {'on the rows concerned: U1':>26s} {'U3 keeps':>9s} {'U3 removes':>11s}")
for (name, rule), sh in zip(RULES, SHORT):
    c = res[sh]["cnt"]
    t1, t2, t3 = (sum(c[u].values()) / FS for u in ("U1", "U2", "U3"))
    print(f"{name:26s} {t1:16.1f} {t2:8.1f} {t3:8.1f} | {t1 - t2:26.1f} {t3 - t2:9.1f} {t1 - t3:11.1f}")
print("\n   R0: U1 would keep the stuck label (wrong label), so U1 is not an option for `activite`.")
print("\n   Classes most affected by U3 instead of U1 (seconds U1 -> U3), rules R1 and R4:")
for sh in ("R1", "R4"):
    c = res[sh]["cnt"]
    d = sorted(((c["U1"][k] - c["U3"][k]) / FS, k) for k in CLASSES)[::-1][:6]
    print(f"      {sh}: " + ", ".join(f"{k} {c['U1'][k] / FS:.1f} -> {c['U3'][k] / FS:.1f}" for v, k in d if v > 0))

# ================================================================================================ 3
print("\n" + "=" * 118 + "\n3. Shape of the derived label (U3)\n")
print(f"{'rule':26s} {'segments':>9s} {'median s':>9s} {'< 0.3 s':>8s} {'< 0.5 s':>8s} | classes with >= 5 s in at least 4 participants | classes absent (< 1 s) in >= 4 participants")
for (name, rule), sh in zip(RULES, SHORT):
    sg = res[sh]["segs"]; pu = res[sh]["per_user"]
    ok = sum(sum(pu[u][c] / FS >= 5 for u in pu) >= 4 for c in CLASSES)
    absent = [c for c in CLASSES if sum(pu[u][c] / FS < 1 for u in pu) >= 4]
    print(f"{name:26s} {len(sg):9d} {np.median(sg):9.2f} {int((sg < 0.3).sum()):8d} {int((sg < 0.5).sum()):8d} | {ok:46d} | " + (", ".join(absent) or "none"))

print("\n   Participants (of 8) with at least 1 s of the class, per rule (U3):")
print(f"{'class':24s} " + " ".join(f"{s:>4s}" for s in SHORT))
for c in CLASSES:
    print(f"{c:24s} " + " ".join(f"{sum(res[s]['per_user'][u][c] / FS >= 1 for u in res[s]['per_user']):4d}" for s in SHORT))

print("\n   Seconds per participant, rule R4 (U3):")
print(f"{'class':24s} " + " ".join(f"{r['user']:>7s}" for r in recs))
for c in CLASSES:
    print(f"{c:24s} " + " ".join(f"{res['R4']['per_user'][r['user']][c] / FS:7.1f}" for r in recs))
print(f"{'TOTAL':24s} " + " ".join(f"{sum(res['R4']['per_user'][r['user']].values()) / FS:7.1f}" for r in recs))

print("\n   Agreement between rules (share of rows labelled by both that get the same label, U3):")
print(f"{'':6s} " + " ".join(f"{s:>6s}" for s in SHORT))
for a in SHORT:
    row = []
    for b in SHORT:
        same = both = 0
        for r in recs:
            x, y = r["derived"][a], r["derived"][b]
            m = (x != "") & (y != "")
            both += int(m.sum()); same += int((x[m] == y[m]).sum())
        row.append(100 * same / max(both, 1))
    print(f"{a:6s} " + " ".join(f"{v:5.1f}%" for v in row))
