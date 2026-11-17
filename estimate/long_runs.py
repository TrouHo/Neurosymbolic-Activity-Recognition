"""E3 — verify the six long label runs and decide what is kept and what is masked (plan 2026-11-17 §E3).

A "long run" is a run of one specific movement label (not `inactif`, `Marcher`, `Autre_Mouvement_*`,
`Debut`, `Fin`) longer than 12 s in a family column; the usual duration of these labels is
0.5-4.5 s (reports/data/2026-11-13_label_errors_check.md). Six exist in 2022 (user1 x3, user3 x2,
user4 x1). Inside them the participant keeps moving and `activite` carries the stuck label.

Decision rule (meeting 2026-11-16: verify with Qualisys and/or against the other participants; what
cannot be verified is dropped). The first part of a run, from the label start to the end of the
movement measured by Qualisys, is KEPT only if all three conditions hold:
  1. Qualisys sees the movement: signed vertical velocity of the family's marker, in the direction
     the label implies, exceeds 150 mm/s within 3 s of the label start, and the movement has a
     clear end (< 50 mm/s for 0.2 s). Labels without a direction (`Autre_Transition`) cannot pass.
  2. Its duration is not longer than the longest run of the same label in the 7 other participants
     (runs > 12 s excluded).
  3. The label that precedes the run in the same column (previous non-`inactif` run) is followed by
     this label in at least half of the other participants (same protocol, same order).
     Only the preceding side can be checked: in 5/6 runs the label that follows is the same label
     again (2026-11-15 finding), which never occurs in the other participants.
Everything else in the run is MASKED. Labels are not shifted here: users 1, 3, 4 have no label
shift, and shifting the others does not change durations or order.

Output: params/long_runs.json (row indices in the label time of the CSV, before alignment).
Usage (from code/):  .venv\\Scripts\\python.exe -m estimate.long_runs
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from emob import config as C
from emob.data.labels import runs

FS = C.FS
LONG_S = 12.0
GENERIC = {"inactif", "Marcher", "Autre_Mouvement_Bras", "Autre_Mouvement_Jambe", "Debut", "Fin", ""}
COLS = {"body": "etiq mvt_corps", "arm": "etiq mvt_bras", "leg": "etiq mvt_jambe"}
MARK = {"body": ("Tete",), "arm": ("D_main", "G_main"), "leg": ("D_pied", "G_pied")}
DOWN = {"Debout_Assis", "Assis_Couche", "Debout_Penche", "Debout_Agenou", "Debout_Accroupi",
        "Baisser_Bras", "Baisser_2_Bras", "Baisser_Jambe"}
UP = {"Assis_Debout", "Couche_Assis", "Penche_Debout", "Agenou_Debout", "Accroupi_Debout",
      "Lever_Bras", "Lever_2_Bras", "Lever_Jambe"}
ON_MM_S, OFF_MM_S, ONSET_WINDOW_S, OFF_HOLD = 150.0, 50.0, 3.0, 10


def sm(x, n):
    return pd.Series(x).rolling(n, center=True, min_periods=max(2, n // 2)).mean().to_numpy()


def vz(df, m):
    z = pd.to_numeric(df[f"z_sq_{m}"], errors="coerce").to_numpy(float).copy()
    xyz = df[[f"x_sq_{m}", f"y_sq_{m}", f"z_sq_{m}"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    z[(xyz == 0).all(1)] = np.nan
    return np.gradient(sm(z, 5)) * FS


def family_velocity(df, fam):
    vs = [vz(df, m) for m in MARK[fam]]
    if len(vs) == 1:
        return sm(vs[0], 10)
    a, b = vs
    return sm(np.where(np.isfinite(a) | np.isfinite(b), np.nan_to_num(a) + np.nan_to_num(b), np.nan), 10)


def first_movement_end(sv, s, limit):
    on = next((i for i in range(s, min(s + int(ONSET_WINDOW_S * FS), limit)) if np.isfinite(sv[i]) and sv[i] > ON_MM_S), None)
    if on is None:
        return None, None
    for j in range(on, limit):
        seg = sv[j:j + OFF_HOLD]
        if np.isfinite(seg).any() and np.nanmax(seg) < OFF_MM_S:
            return on, j
    return on, None


def label_seq(col):
    return [(nm, s, e) for nm, s, e in runs(col) if nm != "inactif"]


def main():
    data = {}
    for f in sorted(C.RAW_2022.glob("user*-*.csv")):
        user = f.name.split("-")[0]
        want = set(COLS.values()) | {C.ACT_COL} | {f"{a}_sq_{m}" for ms in MARK.values() for m in ms for a in "xyz"}
        df = pd.read_csv(f, usecols=lambda c: c in want, low_memory=False)
        data[user] = dict(df=df, fam={k: df[c].astype("string").fillna("").to_numpy(object) for k, c in COLS.items()},
                          act=df[C.ACT_COL].astype("string").fillna("").to_numpy(object))

    # durations and bigrams of every participant (long runs excluded from durations)
    dur, bigr = {}, {}
    for u, d in data.items():
        for fam in COLS:
            seq = label_seq(d["fam"][fam])
            for nm, s, e in seq:
                if (e - s) / FS <= LONG_S:
                    dur.setdefault((u, fam, nm), []).append((e - s) / FS)
            for (a, *_), (b, *_) in zip(seq[:-1], seq[1:]):
                bigr.setdefault((u, fam), set()).add((a, b))

    out = {"created": datetime.now().isoformat(timespec="seconds"), "script": "estimate/long_runs.py",
           "rule": __doc__.split("Decision rule")[1].split("Output")[0].strip(),
           "time_base": "row indices (50 Hz) in the label time of the CSV, before label alignment", "runs": []}
    for u, d in data.items():
        for fam, col in COLS.items():
            seq = label_seq(d["fam"][fam])
            for k, (nm, s, e) in enumerate(seq):
                if nm in GENERIC or (e - s) / FS <= LONG_S:
                    continue
                others = [x for x in data if x != u]
                # 1. Qualisys
                if nm in UP or nm in DOWN:
                    sv = (1 if nm in UP else -1) * family_velocity(d["df"], fam)
                    on, end = first_movement_end(sv, s, e)
                    c1 = end is not None
                    c1_note = (f"onset {on / FS:.2f} s, end {end / FS:.2f} s" if c1 else
                               "no onset within 3 s" if on is None else "no clear end")
                else:
                    on, end, c1, c1_note = None, None, False, "label has no direction: not checkable"
                kept_s = (end - s) / FS if c1 else None
                # 2. duration against the other participants
                omax = max((x for o in others for x in dur.get((o, fam, nm), [])), default=None)
                c2 = bool(c1 and omax is not None and kept_s <= omax)
                # 3. preceding label, same order in the others
                prev = seq[k - 1][0] if k > 0 else None
                support = sum((prev, nm) in bigr.get((o, fam), set()) for o in others) if prev else 0
                c3 = bool(prev is not None and support >= len(others) / 2)
                keep = c1 and c2 and c3
                a = end if keep else s                      # masked part: [a, e)
                stuck = d["act"][a:e] == nm
                rec = dict(user=u, family=fam, column=col, label=nm, start_row=int(s), end_row=int(e),
                           start_s=s / FS, end_s=e / FS, length_s=(e - s) / FS,
                           c1_qualisys=bool(c1), c1_note=c1_note, kept_length_s=kept_s,
                           c2_duration=c2, others_max_s=omax,
                           c3_order=c3, previous_label=prev, previous_followed_by_label_in=f"{support}/{len(others)}",
                           decision="keep first movement, mask the rest" if keep else "mask the whole run",
                           kept_rows=[int(s), int(end)] if keep else None, masked_rows=[int(a), int(e)],
                           masked_s=(e - a) / FS, activite_rows_masked=int(stuck.sum()),
                           activite_s_masked=float(stuck.sum() / FS))
                out["runs"].append(rec)
                print(f"{u} {nm:16s} {fam:4s} {s / FS:6.1f}-{e / FS:6.1f} s ({(e - s) / FS:5.1f} s)")
                print(f"    1 Qualisys : {'yes' if c1 else 'NO '}  {c1_note}" + (f"  -> kept part {kept_s:.2f} s" if c1 else ""))
                print(f"    2 duration : {'yes' if c2 else 'NO '}  kept {kept_s if kept_s is None else round(kept_s, 2)} s vs max in others {omax} s")
                print(f"    3 order    : {'yes' if c3 else 'NO '}  previous '{prev}' followed by '{nm}' in {support}/{len(others)} others")
                print(f"    => {rec['decision']}; masked {rec['masked_s']:.1f} s; activite masked {rec['activite_s_masked']:.1f} s")
    tot = sum(r["masked_s"] for r in out["runs"])
    act = sum(r["activite_s_masked"] for r in out["runs"])
    kept = sum(r["kept_length_s"] or 0 for r in out["runs"] if r["kept_rows"])
    print(f"\n{len(out['runs'])} runs; kept {kept:.1f} s; masked {tot:.1f} s in the family columns "
          f"(user1 runs overlap in time); activite masked {act:.1f} s (overlaps counted once per run)")
    C.PARAMS.mkdir(exist_ok=True)
    p = C.PARAMS / "long_runs.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"written {p}")


if __name__ == "__main__":
    main()
