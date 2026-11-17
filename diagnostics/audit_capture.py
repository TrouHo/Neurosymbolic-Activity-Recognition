"""Timing + label audit for the synchronised captures (2021, 2022), read-only, 2026-11-09.

Per recording:
  * every IMU block (ACC and GYR with signal): presence, distinct values, median |v|, and its lag
    vs the best of a few candidate Qualisys markers (+-60 s scan, then per 120-s window);
  * walking labels vs torso speed and arm labels vs hand speed: best label shift (+-60 s), AUC;
  * posture states (from etiq mvt_corps, same rules as the pipeline) vs head height;
  * label names not in the 2022 class list.
Usage (from code/):  $env:PYTHONPATH="."; .venv\\Scripts\\python.exe diagnostics\\audit_capture.py 2021
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import lagtools as T  # noqa: E402
from emob import config as C  # noqa: E402
from emob.data.labels import annotated_mask, posture_states  # noqa: E402

YEAR = sys.argv[1] if len(sys.argv) > 1 else "2021"
FOLDER = {"2021": "capture_2021-05-31", "2022": "capture_2022-01-26"}[YEAR]
CAND = {
    "SF - poche droite": ["D_hanche", "D_genou"], "SF - pg": ["G_hanche", "G_genou"],
    "SB - Main droite": ["D_main", "D_coude"], "SB - Ceinture": ["Bassin", "D_hanche", "G_hanche", "Torse"],
    "SB - Jambe droite": ["D_genou", "D_pied"], "SB - Jambe gauche": ["G_genou", "G_pied"],
    "SB - Bras droit": ["D_main", "D_coude"], "SB - Bras gauche": ["G_main", "G_coude"],
    "SB - Torse": ["Torse", "Tete"],
}
ARM = {"Lever_Bras", "Baisser_Bras", "Lever_2_Bras", "Baisser_2_Bras", "Autre_Mouvement_Bras"}
FS = 50
out_all = []

files = sorted(glob.glob(os.path.join(r"D:\Projects\EmoB\data\raw", FOLDER, "*.csv")),
               key=lambda p: int(re.search(r"\d+", os.path.basename(p)).group()))
for f in files:
    name = os.path.basename(f)
    df = pd.read_csv(f, low_memory=False)
    t = pd.to_numeric(df["ts_SQ"], errors="coerce").to_numpy(float) / 1000
    fam = [c for c in ["etiq standard", "etiq mvt_corps", "etiq mvt_bras", "etiq mvt_jambe", "etiq activite"] if c in df]
    lab = df[fam].astype("string").fillna("")
    idx = np.flatnonzero((lab != "").any(axis=1).to_numpy())
    lo, hi = (idx[0], idx[-1] + 1) if len(idx) else (0, len(df))
    rec = {"file": name, "labelled_s": (hi - lo) / FS if len(idx) else 0, "blocks": {}}
    g = np.arange(t[lo], t[hi - 1], 1 / T.HZ)

    sq = {}
    for c in df.columns:
        m = re.fullmatch(r"x_sq_(.+)", c)
        if m:
            k = m.group(1)
            ts_, v = T.marker_speed(t[lo:hi], df[[f"x_sq_{k}", f"y_sq_{k}", f"z_sq_{k}"]].to_numpy(float)[lo:hi])
            sq[k] = T.regular(ts_, v, T.HZ, g[0], g[-1] + 1e-9)[1]

    cols = list(df.columns)
    for i, c in enumerate(cols):
        if not (c.startswith("ts_S") and c != "ts_SQ"):
            continue
        blk = c[3:]
        site = " - ".join(blk.split(" - ")[:2])
        kind = blk.split(" - ")[-1]
        xyz = df[cols[i + 1:i + 4]].apply(pd.to_numeric, errors="coerce").to_numpy(float)[lo:hi]
        good = np.isfinite(xyz).all(1) & ~(xyz == 0).all(1)
        distinct = max(len(np.unique(xyz[good, k])) for k in range(3)) if good.any() else 0
        b = {"good": round(float(good.mean()), 3), "distinct": int(distinct),
             "norm_median": round(float(np.median(np.linalg.norm(xyz[good], axis=1))), 3) if good.any() else None}
        rec["blocks"][blk] = b
        if kind in ("LAC", "MAG") or site not in CAND or good.mean() < 0.3 or distinct < 50:
            continue
        tt, e = T.imu_activity(t[lo:hi], xyz)
        e = T.regular(tt, e, T.HZ, g[0], g[-1] + 1e-9)[1]
        best = None
        for mk in CAND[site]:
            if mk not in sq:
                continue
            s = T.summarize(*T.xcorr(sq[mk], e, 60))
            if s["r"] is not None and (best is None or s["r"] > best[1]["r"]):
                best = (mk, s)
        if best is None:
            continue
        w = T.windowed(sq[best[0]], e, best[1]["lag"])
        b.update({"marker": best[0], **best[1],
                  "win_lags": [x[1] for x in w], "win_r": [round(x[2], 2) for x in w]})

    # labels vs Qualisys
    act = lab["etiq activite"].to_numpy(object)[lo:hi] if "etiq activite" in lab else np.array([])
    chk = {}
    if len(idx) and "Torse" in sq:
        walk = T.regular(t[lo:hi], (act == "Marcher").astype(float), T.HZ, g[0], g[-1] + 1e-9)[1]
        if np.nansum(walk) > 20:
            s = T.summarize(*T.xcorr(sq["Torse"], walk, 60))   # positive lag = labels late
            k = int(round(s["lag"] * T.HZ))
            chk["walk"] = {**s, "auc0": T.auc(sq["Torse"], walk > 0.5),
                           "auc_at_lag": T.auc(sq["Torse"], np.roll(walk, -k) > 0.5),
                           "win_lags": [x[1] for x in T.windowed(sq["Torse"], walk, s["lag"], search_s=20)]}
        armi = T.regular(t[lo:hi], np.isin(act, list(ARM)).astype(float), T.HZ, g[0], g[-1] + 1e-9)[1]
        for mk in ("D_main", "G_main"):
            if mk in sq and np.nansum(armi) > 20:
                s = T.summarize(*T.xcorr(sq[mk], armi, 60))
                chk[f"arm_vs_{mk}"] = {**s, "auc0": T.auc(sq[mk], armi > 0.5)}
        names = sorted(set(act) - set(C.L2_CLASSES) - {"", "Debut", "Fin"})
        chk["activite_classes"] = sorted(set(act) - {""})
        chk["not_in_2022_list"] = names
        # posture vs head height
        fams = {c: df[c].astype("string").fillna("").to_numpy(object) for c in C.FAMILY_COLS if c in df}
        if len(fams) == 4:
            st = posture_states(fams[C.BODY_COL], annotated_mask(fams)).state_all[lo:hi]
            head = df["z_sq_Tete"].to_numpy(float)[lo:hi].copy(); head[head == 0] = np.nan
            chk["head_mm_by_state"] = {C.S_CLASSES[k]: [round((st == k).sum() / FS), round(float(np.nanmedian(head[st == k])))]
                                       for k in range(len(C.S_CLASSES)) if (st == k).sum() > 0}
    rec["labels"] = chk
    out_all.append(rec)

    # ---- print
    print(f"\n=== {name}: labelled {rec['labelled_s']:.0f} s")
    for blk, b in rec["blocks"].items():
        if blk.endswith(("LAC", "MAG")):
            continue
        line = f"  {blk:28s} good {b['good']:.2f} distinct {b['distinct']:6d} |v| {b['norm_median']}"
        if "marker" in b:
            wl = b["win_lags"]
            line += (f" | vs {b['marker']:8s} lag {b['lag']:+6.1f}s r {b['r']:.2f} (r@0 {b['r0']}, other peak {b['r_far']:.2f})"
                     f" windows {min(wl) if wl else 'na'}..{max(wl) if wl else 'na'}")
        print(line)
    for k, v in chk.items():
        if k in ("walk",) or k.startswith("arm_"):
            print(f"  LABEL {k}: lag {v['lag']:+.1f}s r {v['r']:.2f} (r@0 {v['r0']:.2f}, other peak {v['r_far']:.2f}) AUC@0 {v['auc0']:.2f}"
                  + (f" AUC@lag {v['auc_at_lag']:.2f} windows {v['win_lags']}" if k == "walk" else ""))
    if "head_mm_by_state" in chk:
        print("  posture [seconds, median head mm]:", chk["head_mm_by_state"])
    if chk.get("not_in_2022_list"):
        print("  activite names not in 2022 list:", chk["not_in_2022_list"])

json.dump(out_all, open(os.path.join(os.environ.get("AUDIT_OUT", "."), f"audit_capture_{YEAR}.json"), "w"), indent=1)
