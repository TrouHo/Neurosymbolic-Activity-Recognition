"""Time offset of the manual labels against motion capture, per participant and per label family
(read-only, 2026-11-13).

Labels and Qualisys markers are on the same rows of the CSV, so any offset found here is an offset of
the ANNOTATION, not of a sensor clock.

Principle. Each label implies a movement that a marker must show:
  body   signed    transitions that lower the head (Debout_X, Assis_Couche) / raise it (reverse)  -> vertical velocity of the head
  body   unsigned  any named transition                                                             -> |vertical velocity of the head|
  arm    signed    Lever_(2_)Bras up / Baisser_(2_)Bras down                                        -> vertical velocity of the hands (sum)
  arm    unsigned  Autre_Mouvement_Bras                                                             -> speed of the faster hand
  leg    signed    Lever_Jambe up / Baisser_Jambe down                                              -> vertical velocity of the feet (sum)
  leg    unsigned  Marcher                                                                          -> horizontal speed of the torso
  leg    unsigned  Autre_Mouvement_Jambe                                                            -> speed of the faster foot
For a shift d, every labelled run [start, end) is moved to [start + d, end + d] and the score is the
mean of the signal over the moved runs (signed: multiplied by the expected sign; unsigned: the signal
is replaced by its rank, so the score is equivalent to an AUC). The estimate is the d with the highest
score, searched from -3 to +3 s in steps of 0.02 s.

  d < 0 : the movement happens BEFORE the label -> labels are late and must be moved earlier by |d|.

Reliability of each estimate:
  - 90 % interval from a bootstrap over the labelled runs (1 000 resamples);
  - the estimate on odd-numbered and on even-numbered runs separately (two independent halves).
An estimate is called reliable when it has >= 8 runs, the 90 % interval is <= 0.4 s wide and the two
halves differ by <= 0.3 s.

Usage (from code/):  $env:PYTHONPATH="."; python -W ignore diagnostics\\check_label_timing.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

import labelshift
print(labelshift.banner() + chr(10))
from emob import config as C
from emob.data.labels import annotated_mask, runs

FS = 50
SUSPECT_S = 12.0
MAXS = 3.0
K = np.arange(-int(MAXS * FS), int(MAXS * FS) + 1)       # shifts in samples (0.02 s)
D = K / FS
NB = 1000
RNG = np.random.default_rng(0)
MIN_RUNS, MAX_CI, MAX_HALF = 8, 0.4, 0.3

DOWN = {"Debout_Assis", "Assis_Couche", "Debout_Penche", "Debout_Agenou", "Debout_Accroupi"}
UP = {"Assis_Debout", "Couche_Assis", "Penche_Debout", "Agenou_Debout", "Accroupi_Debout"}


def xyz(df, k):
    p = df[[f"x_sq_{k}", f"y_sq_{k}", f"z_sq_{k}"]].apply(pd.to_numeric, errors="coerce").to_numpy(float).copy()
    p[(p == 0).all(1) | ~np.isfinite(p).all(1)] = np.nan
    return p


def sm(x, n=5):
    return pd.Series(x).rolling(n, center=True, min_periods=2).mean().to_numpy()


def vel(p, axes):
    """Velocity (mm/s) along the given axes, from positions smoothed over 0.1 s; marker jumps dropped."""
    q = np.stack([sm(p[:, a]) for a in axes], 1)
    v = np.gradient(q, axis=0) * FS
    bad = np.linalg.norm(np.gradient(np.stack([sm(p[:, a]) for a in range(3)], 1), axis=0) * FS, axis=1) > 3000
    v[bad] = np.nan
    return v


def nsum(a, b):
    """Sum of two signals; NaN only where both are missing."""
    out = np.nansum(np.stack([a, b]), axis=0)
    out[~np.isfinite(a) & ~np.isfinite(b)] = np.nan
    return out


def nmax(a, b):
    out = np.fmax(a, b)
    return out


def ranked(x, lo, hi):
    """Rank of x inside the labelled region, scaled to [-0.5, 0.5]; NaN kept."""
    out = np.full(len(x), np.nan)
    seg = x[lo:hi]
    ok = np.isfinite(seg)
    r = pd.Series(seg[ok]).rank().to_numpy()
    seg2 = np.full(len(seg), np.nan); seg2[ok] = r / ok.sum() - 0.5
    out[lo:hi] = seg2
    return out


def curves(x, events):
    """Per run: sum of w * x over the run moved by each shift, and the number of valid rows. (E, len(K)) each."""
    n = len(x)
    ok = np.isfinite(x)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))])
    cn = np.concatenate([[0.0], np.cumsum(ok.astype(float))])
    s = np.array([e[0] for e in events])[:, None] + K[None, :]
    e = np.array([e[1] for e in events])[:, None] + K[None, :]
    w = np.array([e[2] for e in events], float)[:, None]
    s = s.clip(0, n); e = e.clip(0, n)
    return w * (cs[e] - cs[s]), cn[e] - cn[s]


def best(S, N):
    tot, cnt = S.sum(0), N.sum(0)
    sc = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    j = int(np.nanargmax(sc))
    return D[j], sc


def estimate(x, events):
    if len(events) < 3:
        return None
    S, N = curves(x, events)
    keep = N[:, len(K) // 2] > 0                        # runs with at least one valid row at d = 0
    S, N = S[keep], N[keep]
    E = len(S)
    if E < 3:
        return None
    d, sc = best(S, N)
    boot = np.empty(NB)
    for b in range(NB):
        i = RNG.integers(0, E, E)
        boot[b] = best(S[i], N[i])[0]
    d_odd = best(S[0::2], N[0::2])[0]
    d_even = best(S[1::2], N[1::2])[0]
    z = len(K) // 2
    lo, hi = np.percentile(boot, [5, 95])
    t3 = [float(best(S[a:b], N[a:b])[0]) for a, b in ((0, E // 3), (E // 3, 2 * E // 3), (2 * E // 3, E))] if E >= 12 else None
    return dict(n=E, sec=float(N[:, z].sum() / FS), d=float(d), lo=float(lo), hi=float(hi), d_odd=float(d_odd), d_even=float(d_even), thirds=t3,
                s0=float(sc[z]), sbest=float(np.nanmax(sc)), edge=bool(abs(d) >= MAXS - 0.05),
                reliable=bool(E >= MIN_RUNS and hi - lo <= MAX_CI and abs(d_odd - d_even) <= MAX_HALF and abs(d) < MAXS - 0.05))


def events_of(col, mapping, sus):
    """Runs of the given labels: (start, end, weight)."""
    out = []
    for name, s, e in runs(col):
        if name in mapping and not sus[s:e].any():
            out.append((s, e, mapping[name]))
    return out


SPEC = [  # family, estimator name, label column, {label: weight}, signal key
    ("body", "signed: 10 named transitions vs head vertical velocity", C.BODY_COL, {**{k: -1 for k in DOWN}, **{k: 1 for k in UP}}, "head_vz"),
    ("body", "unsigned: named transitions vs |head vertical velocity|", C.BODY_COL, {k: 1 for k in DOWN | UP}, "head_absvz_rank"),
    ("arm", "signed: Lever/Baisser (2_)Bras vs hand vertical velocity", "etiq mvt_bras",
     {"Lever_Bras": 1, "Lever_2_Bras": 1, "Baisser_Bras": -1, "Baisser_2_Bras": -1}, "hand_vz"),
    ("arm", "unsigned: Autre_Mouvement_Bras vs speed of the faster hand", "etiq mvt_bras", {"Autre_Mouvement_Bras": 1}, "hand_speed_rank"),
    ("leg", "signed: Lever/Baisser_Jambe vs foot vertical velocity", "etiq mvt_jambe", {"Lever_Jambe": 1, "Baisser_Jambe": -1}, "foot_vz"),
    ("leg", "unsigned: Marcher vs horizontal speed of the torso", "etiq mvt_jambe", {"Marcher": 1}, "torso_hspeed_rank"),
    ("leg", "unsigned: Autre_Mouvement_Jambe vs speed of the faster foot", "etiq mvt_jambe", {"Autre_Mouvement_Jambe": 1}, "foot_speed_rank"),
]

results = {}
for f in sorted(glob.glob(str(C.RAW_2022 / "*.csv"))):
    user = os.path.basename(f).split("-")[0]
    df = pd.read_csv(f, low_memory=False)
    fam = labelshift.families(df, user)
    ann = annotated_mask(fam)
    idx = np.flatnonzero(ann); lo, hi = int(idx[0]), int(idx[-1]) + 1
    n = len(df)
    sus = np.zeros(n, bool)
    for c in ("etiq mvt_corps", "etiq mvt_bras", "etiq mvt_jambe"):
        for name, s, e in runs(fam[c]):
            if name not in ("inactif", "Marcher", "Autre_Mouvement_Bras", "Autre_Mouvement_Jambe", "Debut", "Fin") and (e - s) / FS > SUSPECT_S:
                sus[s:e] = True
    head, torso = xyz(df, "Tete"), xyz(df, "Torse")
    hD, hG, fD, fG = xyz(df, "D_main"), xyz(df, "G_main"), xyz(df, "D_pied"), xyz(df, "G_pied")
    sp = lambda p: np.linalg.norm(vel(p, (0, 1, 2)), axis=1)
    sig = {
        "head_vz": vel(head, (2,))[:, 0],
        "hand_vz": nsum(vel(hD, (2,))[:, 0], vel(hG, (2,))[:, 0]),
        "foot_vz": nsum(vel(fD, (2,))[:, 0], vel(fG, (2,))[:, 0]),
    }
    sig["head_absvz_rank"] = ranked(np.abs(sig["head_vz"]), lo, hi)
    sig["hand_speed_rank"] = ranked(nmax(sp(hD), sp(hG)), lo, hi)
    sig["foot_speed_rank"] = ranked(nmax(sp(fD), sp(fG)), lo, hi)
    sig["torso_hspeed_rank"] = ranked(np.linalg.norm(vel(torso, (0, 1)), axis=1), lo, hi)
    results[user] = []
    for family, name, col, mapping, key in SPEC:
        ev = events_of(fam[col], mapping, sus)
        r = estimate(sig[key], ev)
        results[user].append(dict(family=family, estimator=name, **(r or dict(n=len(ev), reliable=False, d=None))))

# ------------------------------------------------------------------ output
print("Shift d (s) that best aligns the labels with motion capture. d < 0: the movement happens before the label")
print("(labels late; move them earlier). 90 % interval from a bootstrap over runs; 'odd' / 'even' = estimate on each half.\n")
for family in ("body", "arm", "leg"):
    for name in [s[1] for s in SPEC if s[0] == family]:
        print("=" * 110 + f"\n{family.upper()} — {name}\n")
        print(f"{'user':6s} {'runs':>5s} {'seconds':>8s} {'d':>7s} {'90 % interval':>18s} {'width':>6s} {'odd':>7s} {'even':>7s} {'score at 0':>11s} {'score at d':>11s}  reliable")
        for user, lst in results.items():
            r = next(x for x in lst if x["estimator"] == name)
            if r["d"] is None:
                print(f"{user:6s} {r['n']:5d} {'too few runs':>40s}")
                continue
            flag = "yes" if r["reliable"] else ("no (peak at the edge of the search)" if r["edge"] else "no")
            print(f"{user:6s} {r['n']:5d} {r['sec']:8.1f} {r['d']:+7.2f} {'[' + format(r['lo'], '+.2f') + ', ' + format(r['hi'], '+.2f') + ']':>18s} {r['hi'] - r['lo']:6.2f} "
                  f"{r['d_odd']:+7.2f} {r['d_even']:+7.2f} {r['s0']:11.3f} {r['sbest']:11.3f}  {flag}")
        print()

print("=" * 110 + "\nSUMMARY — reliable estimates only (d in s, [90 % interval]); '-' = no reliable estimate\n")
short = {s[1]: f"{s[0]} {'signed' if s[1].startswith('signed') else s[1].split(':')[1].split(' vs')[0].strip()}" for s in SPEC}
names = [s[1] for s in SPEC]
print(f"{'user':6s} " + " ".join(f"{short[nm][:26]:>26s}" for nm in names))
for user, lst in results.items():
    row = []
    for nm in names:
        r = next(x for x in lst if x["estimator"] == nm)
        row.append(f"{r['d']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}]" if r["reliable"] else "-")
    print(f"{user:6s} " + " ".join(f"{x:>26s}" for x in row))

print("\nPer participant and family: the reliable estimate with the narrowest interval\n")
print(f"{'user':6s} {'body':>24s} {'arm':>24s} {'leg':>24s}   spread between families (s)")
chosen = {}
for user, lst in results.items():
    row, vals = [], []
    chosen[user] = {}
    for family in ("body", "arm", "leg"):
        c = [x for x in lst if x["family"] == family and x["reliable"]]
        if c:
            b = min(c, key=lambda x: x["hi"] - x["lo"])
            row.append(f"{b['d']:+.2f} [{b['lo']:+.2f},{b['hi']:+.2f}]"); vals.append(b["d"])
            chosen[user][family] = dict(d=b["d"], lo=b["lo"], hi=b["hi"], estimator=b["estimator"], runs=b["n"])
        else:
            row.append("-"); chosen[user][family] = None
    print(f"{user:6s} " + " ".join(f"{x:>24s}" for x in row) + (f"   {max(vals) - min(vals):.2f}" if len(vals) > 1 else "   -"))

print("\n" + "=" * 110 + "\nIs the offset constant inside a recording? Estimate on the first, middle and last third of the runs (in time order)\n")
print(f"{'user':6s} {'body signed':>26s} {'leg signed':>26s} {'leg Marcher':>26s}")
for user, lst in results.items():
    row = []
    for nm in (SPEC[0][1], SPEC[4][1], SPEC[5][1]):
        r = next(x for x in lst if x["estimator"] == nm)
        row.append(" ".join(f"{v:+.2f}" for v in r["thirds"]) if r.get("thirds") else "-")
    print(f"{user:6s} " + " ".join(f"{x:>26s}" for x in row))

out = C.PROJECT / "reports" / "data" / ("2026-11-13_label_timing_after_shift.json" if labelshift.ON else "2026-11-13_label_timing.json")
json.dump(dict(meta=dict(convention="d < 0: movement before the label; shift labels by d seconds", search_s=MAXS, step_s=1 / FS, bootstrap=NB,
                         reliable=dict(min_runs=MIN_RUNS, max_interval_s=MAX_CI, max_half_difference_s=MAX_HALF)),
               estimates=results, chosen=chosen), open(out, "w", encoding="utf-8"), indent=1)
print(f"\nSaved: {out}")
