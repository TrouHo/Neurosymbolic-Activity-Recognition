"""Which sensor "sees" which label family? (2022 capture, read-only diagnostic, 2026-11-10)

For every sensor configuration and every label task, a Random Forest on orientation-invariant
2-s window features is evaluated leave-one-subject-out. A Qualisys (motion capture) feature set
is the ceiling: what the labels allow when body geometry is fully known.

This is a DIAGNOSTIC, not a baseline of the study:
  * one seed, no tuning, no validation participant;
  * per-recording unsupervised normalisation (reference gravity, standing head height) uses the
    test participant's unlabelled signal;
  * sensor time shifts are estimated here against Qualisys (same method as the 2026-11-09 audit).

Usage (from code/, needs scikit-learn):
    $env:PYTHONPATH="."; python diagnostics\\observability_matrix.py [out_dir]
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import lagtools as T  # noqa: E402
from emob import config as C  # noqa: E402
from emob.data.labels import annotated_mask, posture_states, runs  # noqa: E402

warnings.filterwarnings("ignore")
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
FS, STEP, WIN = 50, 5, 100            # 50 Hz, 0.1-s steps, 2-s windows
BASELINE_LAG = 0.2                    # lag measured on the best-synchronised pairs (audit §2)
SUSPECT_RUN_S = 12.0                  # a single specific movement longer than this is not credible
TREES, SEED = 150, 0

BLOCKS = {  # sensor -> (timestamp column, candidate Qualisys markers for the lag)
    "phone_acc": ("ts_SF - poche droite - ACC", ["D_hanche", "D_genou"]),
    "phone_gyr": ("ts_SF - poche droite - GYR", None),          # same device: lag of phone_acc
    "hand": ("ts_SB - Main droite - ACC", ["D_main", "D_coude"]),
    "belt": ("ts_SB - Ceinture - ACC", ["Bassin", "D_hanche", "G_hanche", "Torse"]),
    "leg": ("ts_SB - Jambe droite - ACC", ["D_genou", "D_pied"]),
}
MARKERS = ["Tete", "Torse", "D_epaule", "G_epaule", "D_coude", "G_coude", "D_main", "G_main",
           "D_hanche", "G_hanche", "D_genou", "G_genou", "D_pied", "G_pied"]
ARM = ["Lever_Bras", "Baisser_Bras", "Lever_2_Bras", "Baisser_2_Bras", "Autre_Mouvement_Bras"]
LEG = ["Lever_Jambe", "Baisser_Jambe", "Autre_Mouvement_Jambe"]
TRANS = C.POSTURE_TRANSITIONS + [C.OTHER_TRANSITION]
NOT_SUSPECT = {"inactif", "Marcher", "Autre_Mouvement_Bras", "Autre_Mouvement_Jambe", "Debut", "Fin"}


# ----------------------------------------------------------------------------- features
def windows(x: np.ndarray) -> np.ndarray:
    """(N,) 50-Hz signal -> (T, WIN): the 2-s window centred on every 0.1-s step."""
    n_steps = len(x) // STEP
    pad = np.pad(x.astype(float), (WIN // 2, WIN // 2), constant_values=np.nan)
    return sliding_window_view(pad, WIN)[np.arange(n_steps) * STEP + C.CENTRE]


def stats(w, names, which):
    f = {"mean": np.nanmean, "std": np.nanstd, "min": np.nanmin, "max": np.nanmax,
         "p25": lambda a, axis: np.nanpercentile(a, 25, axis=axis),
         "p75": lambda a, axis: np.nanpercentile(a, 75, axis=axis)}
    return [f[k](w, axis=1) for k in which], [f"{names}_{k}" for k in which]


def acc_features(xyz: np.ndarray):
    """Orientation-invariant features of one accelerometer. Returns (T, F), valid (T,)."""
    a = xyz.copy()
    m = np.linalg.norm(a, axis=1)
    g = pd.DataFrame(a).rolling(FS, center=True, min_periods=FS // 2).mean().to_numpy()   # gravity estimate
    gn = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
    dyn = a - g
    vert = (dyn * gn).sum(1)
    hor = np.linalg.norm(dyn - vert[:, None] * gn, axis=1)
    ref = np.nanmedian(gn, axis=0); ref /= np.linalg.norm(ref) + 1e-9      # usual orientation of this recording
    tilt = np.degrees(np.arccos(np.clip(gn @ ref, -1, 1)))
    tilt[~np.isfinite(gn).all(1)] = np.nan

    wm, wv, wh, wt = windows(m), windows(vert), windows(hor), windows(tilt)
    cols, names = [], []
    for w, nm, which in [(wm, "mag", ["mean", "std", "min", "max", "p25", "p75"]),
                         (wv, "vert", ["mean", "std", "min", "max"]),
                         (wh, "hor", ["mean", "std", "max"]),
                         (wt, "tilt", ["mean", "std", "min", "max"])]:
        c, n = stats(w, nm, which); cols += c; names += n
    cols.append(np.nanmean(np.abs(np.diff(wm, axis=1)), axis=1)); names.append("mag_absdiff")
    q = WIN // 4
    cols.append(np.nanmean(wt[:, -q:], axis=1) - np.nanmean(wt[:, :q], axis=1)); names.append("tilt_delta")
    z = np.where(np.isfinite(wm), wm, np.nanmean(wm, axis=1, keepdims=True))
    z = np.nan_to_num(z - z.mean(1, keepdims=True))
    spec = np.abs(np.fft.rfft(z, axis=1)) ** 2
    fr = np.fft.rfftfreq(WIN, 1 / FS)
    band = (fr >= 0.5) & (fr <= 3.0)
    cols.append(fr[spec[:, 1:].argmax(1) + 1]); names.append("mag_domfreq")
    cols.append(spec[:, band].sum(1) / (spec.sum(1) + 1e-9)); names.append("mag_band")
    valid = np.isfinite(wm).mean(1) >= 0.5
    return np.stack(cols, 1), valid, names


def gyr_features(xyz: np.ndarray):
    w = windows(np.linalg.norm(xyz, axis=1))
    c, n = stats(w, "gyr", ["mean", "std", "max", "p75"])
    c.append(np.nanmean(np.abs(np.diff(w, axis=1)), axis=1)); n.append("gyr_absdiff")
    return np.stack(c, 1), np.isfinite(w).mean(1) >= 0.5, n


def mocap_features(df: pd.DataFrame):
    """Body-geometry features from Qualisys (ceiling). Heights are divided by standing head height."""
    P = {}
    for k in MARKERS:
        p = df[[f"x_sq_{k}", f"y_sq_{k}", f"z_sq_{k}"]].to_numpy(float).copy()
        p[(p == 0).all(1) | ~np.isfinite(p).all(1)] = np.nan
        P[k] = p
    scale = np.nanpercentile(P["Tete"][:, 2], 95)
    cols, names = [], []
    for k in MARKERS:
        wz = windows(P[k][:, 2] / scale)
        cols += [np.nanmean(wz, 1), np.nanmax(wz, 1) - np.nanmin(wz, 1)]; names += [f"{k}_z", f"{k}_zrange"]
        v = np.linalg.norm(np.diff(P[k], axis=0, prepend=P[k][:1]), axis=1) * FS
        v[v > 3000] = np.nan
        cols.append(np.nanmean(windows(v), 1) / scale); names.append(f"{k}_speed")
    for s in "DG":
        cols.append(np.nanmean(windows((P[f"{s}_main"][:, 2] - P[f"{s}_epaule"][:, 2]) / scale), 1)); names.append(f"{s}_hand_minus_shoulder")
        cols.append(np.nanmean(windows((P[f"{s}_genou"][:, 2] - P[f"{s}_hanche"][:, 2]) / scale), 1)); names.append(f"{s}_knee_minus_hip")
    hips = np.nanmean(np.stack([P["D_hanche"], P["G_hanche"]]), axis=0)
    trunk = P["Tete"] - hips
    incl = np.degrees(np.arccos(np.clip(trunk[:, 2] / (np.linalg.norm(trunk, axis=1) + 1e-9), -1, 1)))
    cols.append(np.nanmean(windows(incl), 1)); names.append("trunk_incl")
    wh = windows(P["Tete"][:, 2] / scale)
    q = WIN // 4
    cols.append(np.nanmean(wh[:, -q:], 1) - np.nanmean(wh[:, :q], 1)); names.append("head_z_delta")
    vt = np.linalg.norm(np.diff(P["Torse"][:, :2], axis=0, prepend=P["Torse"][:1, :2]), axis=1) * FS
    vt[vt > 3000] = np.nan
    cols.append(np.nanmean(windows(vt), 1) / scale); names.append("torso_hspeed")
    X = np.stack(cols, 1)
    return X, np.isfinite(X[:, 0]), names


# ----------------------------------------------------------------------------- loading
def estimate_lag(t, xyz, df, lo, hi, cands):
    g = np.arange(t[lo], t[hi - 1], 1 / T.HZ)
    tt, e = T.imu_activity(t[lo:hi], xyz[lo:hi])
    e = T.regular(tt, e, T.HZ, g[0], g[-1] + 1e-9)[1]
    best = None
    for mk in cands:
        p = df[[f"x_sq_{mk}", f"y_sq_{mk}", f"z_sq_{mk}"]].to_numpy(float)[lo:hi]
        if (np.isfinite(p).all(1) & ~(p == 0).all(1)).mean() < 0.3:
            continue
        ts_, v = T.marker_speed(t[lo:hi], p)
        s = T.summarize(*T.xcorr(T.regular(ts_, v, T.HZ, g[0], g[-1] + 1e-9)[1], e, 30))
        if s["r"] is not None and (best is None or s["r"] > best[1]["r"]):
            best = (mk, s)
    return best


def shift(x, seconds):
    """Sensor is late by `seconds` -> take the value recorded `seconds` later."""
    k = int(round(seconds * FS))
    out = np.full_like(x, np.nan)
    if k >= 0:
        out[: len(x) - k] = x[k:]
    else:
        out[-k:] = x[: len(x) + k]
    return out


def load(path):
    user = os.path.basename(path).split("-")[0]
    df = pd.read_csv(path, low_memory=False)
    cols = list(df.columns)
    t = pd.to_numeric(df["ts_SQ"], errors="coerce").to_numpy(float) / 1000
    fam = {c: df[c].astype("string").fillna("").to_numpy(object) for c in C.FAMILY_COLS}
    act = df[C.ACT_COL].astype("string").fillna("").to_numpy(object)
    ann = annotated_mask(fam)
    idx = np.flatnonzero(ann); lo, hi = idx[0], idx[-1] + 1

    rec = {"user": user, "feat": {}, "valid": {}, "lag": {}, "names": {}}
    for key, (ts_col, cands) in BLOCKS.items():
        i = cols.index(ts_col)
        xyz = df[cols[i + 1:i + 4]].apply(pd.to_numeric, errors="coerce").to_numpy(float).copy()
        bad =~np.isfinite(xyz).all(1) | (xyz == 0).all(1)
        good = ~bad[lo:hi]
        distinct = max(len(np.unique(xyz[lo:hi][good, k])) for k in range(3)) if good.any() else 0
        norm = float(np.median(np.linalg.norm(xyz[lo:hi][good], axis=1))) if good.any() else 0.0
        ok = good.mean() >= 0.5 and distinct >= C.MIN_UNIQUE and (key == "phone_gyr" or 8.0 <= norm <= 11.5)
        info = {"good": round(float(good.mean()), 3), "distinct": int(distinct), "norm": round(norm, 2), "used": bool(ok)}
        if ok:
            xyz[bad] = np.nan
            if cands is None:
                lag = rec["lag"]["phone_acc"].get("shift", 0.0)
                info.update({"shift": lag})
            else:
                b = estimate_lag(t, np.nan_to_num(xyz), df, lo, hi, cands)
                lag = (b[1]["lag"] - BASELINE_LAG) if b else 0.0
                info.update({"marker": b[0] if b else None, "lag": b[1]["lag"] if b else None,
                             "r": round(b[1]["r"], 2) if b else None, "r0": b[1]["r0"], "shift": round(lag, 2)})
            xs = np.stack([shift(xyz[:, k], lag) for k in range(3)], 1)
            X, v, n = (gyr_features if key == "phone_gyr" else acc_features)(xs)
            rec["feat"][key], rec["valid"][key], rec["names"][key] = X, v, n
        rec["lag"][key] = info
    rec["feat"]["mocap"], rec["valid"]["mocap"], rec["names"]["mocap"] = mocap_features(df)

    # ---- labels on the step grid
    suspect = np.zeros(len(df), bool); flagged = []
    for c in ["etiq mvt_corps", "etiq mvt_bras", "etiq mvt_jambe"]:
        for name, s, e in runs(fam[c]):
            if name not in NOT_SUSPECT and (e - s) / FS > SUSPECT_RUN_S:
                suspect[s:e] = True; flagged.append((c, name, round((e - s) / FS, 1)))
    rec["suspect_runs"] = flagged
    st = posture_states(fam[C.BODY_COL], ann).state_all.copy()
    region = np.zeros(len(df), bool); region[lo:hi] = True
    centres = np.arange(len(df) // STEP) * STEP + C.CENTRE
    body, arm, leg = fam["etiq mvt_corps"][centres], fam["etiq mvt_bras"][centres], fam["etiq mvt_jambe"][centres]
    a, ok = act[centres], (ann & ~suspect)[centres]

    def enc(v, classes, mask):
        y = np.full(len(v), -1)
        for k, c in enumerate(classes):
            y[(v == c) & mask] = k
        return y

    y = {}
    y["walk"] = np.where(ok, (leg == "Marcher").astype(int), -1)
    y["leg_move"] = np.where(ok, np.isin(leg, LEG).astype(int), -1)
    y["arm_move"] = np.where(ok, np.isin(arm, ARM).astype(int), -1)
    y["trans_detect"] = np.where(ok, np.isin(body, TRANS).astype(int), -1)
    y["arm_type"] = enc(arm, ARM, ok)
    y["leg_type"] = enc(leg, LEG + ["Marcher"], ok)
    y["trans_type"] = enc(body, C.POSTURE_TRANSITIONS, ok)
    y["posture"] = np.where((region & ~suspect)[centres], st[centres], -1)
    y["L2"] = enc(a, C.L2_CLASSES, ok)
    l1 = np.array([C.L1_CLASSES.index(C.L2_TO_L1[c]) for c in C.L2_CLASSES])
    y["L1"] = np.where(y["L2"] >= 0, l1[np.clip(y["L2"], 0, None)], -1)
    rec["y"] = y
    return rec


# ----------------------------------------------------------------------------- evaluation
BINARY = ["walk", "leg_move", "arm_move", "trans_detect"]
MULTI = {"posture": C.S_CLASSES, "trans_type": C.POSTURE_TRANSITIONS, "arm_type": ARM,
         "leg_type": LEG + ["Marcher"], "L1": C.L1_CLASSES, "L2": C.L2_CLASSES}


def loso(recs, sensors, task):
    users = [r["user"] for r in recs]
    X, Y = {}, {}
    for r in recs:
        x = np.concatenate([r["feat"][s] for s in sensors], 1)
        v = np.all([r["valid"][s] for s in sensors], 0)
        X[r["user"]] = np.nan_to_num(x, nan=-999.0, posinf=-999.0, neginf=-999.0)
        Y[r["user"]] = np.where(v, r["y"][task], -1)
    aucs, yt, yp, yd = [], [], [], []
    for te in users:
        Xtr = np.concatenate([X[u][::2] for u in users if u != te])
        ytr = np.concatenate([Y[u][::2] for u in users if u != te])
        m = ytr >= 0
        mte = Y[te] >= 0
        if len(np.unique(ytr[m])) < 2 or mte.sum() == 0:
            continue
        rf = RandomForestClassifier(TREES, min_samples_leaf=3, class_weight="balanced_subsample",
                                    n_jobs=-1, random_state=SEED).fit(Xtr[m], ytr[m])
        proba = rf.predict_proba(X[te][mte])
        if task in BINARY:
            if len(np.unique(Y[te][mte])) == 2:
                aucs.append(roc_auc_score(Y[te][mte], proba[:, list(rf.classes_).index(1)]))
        yt.append(Y[te][mte]); yp.append(rf.classes_[proba.argmax(1)])
        yd.append(DummyClassifier(strategy="stratified", random_state=SEED).fit(Xtr[m], ytr[m]).predict(X[te][mte]))
    yt, yp, yd = np.concatenate(yt), np.concatenate(yp), np.concatenate(yd)
    out = {"n_test": int(len(yt))}
    if task in BINARY:
        out.update({"auc": float(np.mean(aucs)), "auc_min": float(np.min(aucs)),
                    "f1_pos": float(f1_score(yt, yp)), "pos_rate": float(yt.mean())})
    else:
        labels = sorted(np.unique(yt))
        per = f1_score(yt, yp, labels=labels, average=None)
        out.update({"macro_f1": float(per.mean()),
                    "dummy_macro_f1": float(f1_score(yt, yd, labels=labels, average="macro")),
                    "per_class": {MULTI[task][k]: [round(float(f), 3), int((yt == k).sum())] for k, f in zip(labels, per)}})
    return out


def main():
    t0 = time.time()
    recs = [load(f) for f in sorted(glob.glob(str(C.RAW_2022 / "*.csv")))]
    print("=== sensors (labelled region): good fraction, distinct values, median |v|, lag vs Qualisys, shift applied")
    for r in recs:
        print(r["user"], "| suspect runs masked:", r["suspect_runs"])
        for k, v in r["lag"].items():
            print(f"   {k:10s} {v}")
    have = lambda s: [r for r in recs if all(k in r["feat"] for k in s)]  # noqa: E731
    U8 = [r["user"] for r in recs]
    S5 = [r["user"] for r in have(["hand", "leg"])]
    S4 = [r["user"] for r in have(["hand", "belt", "leg"])]
    print("\nparticipants: all", U8, "| hand+leg", S5, "| hand+belt+leg", S4)
    PH = ["phone_acc", "phone_gyr"]
    configs = [("phone ACC", ["phone_acc"], U8), ("phone ACC+GYR", PH, U8), ("Qualisys (ceiling)", ["mocap"], U8),
               ("phone ACC+GYR", PH, S5), ("hand ACC", ["hand"], S5), ("leg ACC", ["leg"], S5),
               ("hand+leg ACC", ["hand", "leg"], S5), ("phone+hand", PH + ["hand"], S5),
               ("Qualisys (ceiling)", ["mocap"], S5),
               ("phone ACC+GYR", PH, S4), ("belt ACC", ["belt"], S4), ("hand+belt ACC", ["hand", "belt"], S4),
               ("phone+hand+belt", PH + ["hand", "belt"], S4), ("Qualisys (ceiling)", ["mocap"], S4)]
    results = []
    for name, sensors, users in configs:
        sub = [r for r in recs if r["user"] in users]
        row = {"config": name, "sensors": sensors, "users": users, "tasks": {}}
        for task in BINARY + list(MULTI):
            row["tasks"][task] = loso(sub, sensors, task)
        results.append(row)
        tk = row["tasks"]
        print(f"{name:20s} n={len(users)} | AUC walk {tk['walk']['auc']:.2f} leg {tk['leg_move']['auc']:.2f} arm {tk['arm_move']['auc']:.2f}"
              f" trans {tk['trans_detect']['auc']:.2f} | F1 posture {tk['posture']['macro_f1']:.2f} trans_type {tk['trans_type']['macro_f1']:.2f}"
              f" arm_type {tk['arm_type']['macro_f1']:.2f} leg_type {tk['leg_type']['macro_f1']:.2f} L1 {tk['L1']['macro_f1']:.2f} L2 {tk['L2']['macro_f1']:.2f}"
              f" | {time.time() - t0:.0f}s", flush=True)
    meta = {"sensors": {r["user"]: r["lag"] for r in recs}, "suspect_runs": {r["user"]: r["suspect_runs"] for r in recs},
            "n_features": {k: len(v) for k, v in recs[-1]["names"].items()}, "trees": TREES, "seed": SEED}
    with open(os.path.join(OUT, "observability_matrix.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
