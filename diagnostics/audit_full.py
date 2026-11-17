"""Full read-only audit of every sensor block, timestamp column and label column (2026-11-09).

For each recording CSV (2022 by default, 2021 with --year 2021) and each block
(Qualisys markers, SF phone ACC/LAC/GYR, every SB ACC/GYR/MAG) it reports, inside the
labelled region (first..last row with any non-empty label):

  timestamps : NaN fraction, fraction equal to ts_SQ, median offset to ts_SQ (ms),
               fraction of rows whose block timestamp repeats the previous row (held sample)
  values     : NaN fraction, all-zero fraction, max distinct values per axis,
               median and IQR of the vector norm, fraction of rows identical to the previous row
  sync       : correlation of the block's 1-s activity envelope with the speed of the nearest
               Qualisys marker, best lag in [-5, 5] s, whole recording and per quarter
  labels     : walking-label vs torso-speed correlation; arm-label vs hand-marker speed ratio

Writes a JSON with every number and prints a compact table. Modifies nothing.
Usage (from D:\\Projects\\EmoB\\code):  .venv\\Scripts\\python.exe diagnostics\\audit_full.py --out <file.json>
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(r"D:\Projects\EmoB\data\raw")
FS = 50
LAB_PREFIX = "etiq "

# IMU block -> Qualisys marker used as movement reference for the sync check
REF_MARKER = {
    "SF - poche droite": "D_hanche",
    "SB - Ceinture": "Bassin",
    "SB - Main droite": "D_main",
    "SB - Jambe droite": "D_genou",
    "SB - Bras droit": "D_main",
    "SB - Bras gauche": "G_main",
    "SB - Jambe gauche": "G_genou",
    "SB - Torse": "Torse",
}


def blocks(df: pd.DataFrame):
    """Yield (block_name, ts_col or None, [x, y, z]) in file order."""
    cols = list(df.columns)
    for i, c in enumerate(cols):
        if c.startswith("ts_S") and c != "ts_SQ":
            yield c[3:], c, cols[i + 1:i + 4]
    for c in cols:
        m = re.fullmatch(r"x_sq_(.+)", c)
        if m:
            yield "SQ " + m.group(1), None, [c, f"y_sq_{m.group(1)}", f"z_sq_{m.group(1)}"]


def envelope(x: np.ndarray, win: int = FS) -> np.ndarray:
    """1-s moving std of a 1-D signal (NaN-safe via pandas)."""
    return pd.Series(x).rolling(win, center=True, min_periods=win // 2).std().to_numpy()


def speed(xyz: np.ndarray, clip: float = 3000.0) -> np.ndarray:
    """Marker speed (mm/s); values above `clip` (marker swaps / reappearance) set to NaN."""
    v = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * FS
    v = np.r_[v[:1], v]
    v[v > clip] = np.nan
    return v


def auc(score: np.ndarray, pos: np.ndarray) -> float:
    """P(score of a positive row > score of a negative row) (Mann-Whitney), NaN rows dropped."""
    ok = np.isfinite(score)
    s, p = score[ok], pos[ok]
    n1, n0 = p.sum(), (~p).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _rank(x: np.ndarray) -> np.ndarray:
    """Rank-transform (NaN kept) so correlations are Spearman, robust to marker spikes."""
    return pd.Series(x).rank().to_numpy()


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 5 * FS, step: int = 5):
    """Lag (s) maximising Spearman corr(a[t], b[t+lag]); positive = b later than a."""
    a, b = _rank(a), _rank(b)
    best = (np.nan, np.nan)
    for lag in range(-max_lag, max_lag + 1, step):
        if lag >= 0:
            x, y = a[:len(a) - lag], b[lag:]
        else:
            x, y = a[-lag:], b[:len(b) + lag]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 10 * FS or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            continue
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        if np.isnan(best[0]) or r > best[0]:
            best = (float(r), lag / FS)
    r0 = np.nan
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() > 10 * FS and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
        r0 = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return {"r_at_0": r0, "r_best": best[0], "lag_best_s": best[1]}


def audit_file(path: Path) -> dict:
    df = pd.read_csv(path, sep=",", low_memory=False)
    ts = pd.to_numeric(df["ts_SQ"], errors="coerce").to_numpy(float)
    # 'etiq tst 1' is the timestamp of the label track, not a label
    labcols = [c for c in df.columns if c.startswith(LAB_PREFIX) and c != "etiq tst 1"]
    lab = df[labcols].astype("string").fillna("")
    anylab = (lab != "").any(axis=1).to_numpy()
    idx = np.flatnonzero(anylab)
    lo, hi = (idx[0], idx[-1] + 1) if len(idx) else (0, len(df))
    R = slice(lo, hi)

    out = {
        "file": path.name,
        "n_rows": int(len(df)),
        "ts_SQ_start_utc": datetime.fromtimestamp(ts[0] / 1000, timezone.utc).isoformat(),
        "ts_SQ_end_utc": datetime.fromtimestamp(ts[-1] / 1000, timezone.utc).isoformat(),
        "duration_s": float((ts[-1] - ts[0]) / 1000),
        "ts_SQ_steps": {str(k): int(v) for k, v in zip(*np.unique(np.diff(ts), return_counts=True))},
        "ts_SQ_whole_seconds_frac": float(np.mean(ts % 1000 == 0)),
        "label_region_rows": [int(lo), int(hi)],
        "label_region_s": float((hi - lo) / FS),
        "label_annotated_frac_in_region": float(anylab[R].mean()),
        "label_cols_nonempty_in_region": {c: int((lab[c].iloc[lo:hi] != "").sum()) for c in labcols},
        "blocks": {},
    }
    tl = pd.to_numeric(df["etiq tst 1"], errors="coerce").to_numpy(float)
    out["label_ts_equals_SQ_frac"] = float(np.mean(tl == ts))
    act_r = lab["etiq activite"].to_numpy(object)[R]
    std_r = lab["etiq standard"].to_numpy(object)[R]
    empty = act_r == ""
    out["activite_empty_frac_in_region"] = float(empty.mean())
    out["standard_values_where_activite_empty"] = {
        str(k): int(v) for k, v in pd.Series(std_r[empty]).value_counts().items()}
    out["standard_values_in_region"] = {str(k): int(v) for k, v in pd.Series(std_r).value_counts().items()}

    sq = {}
    for name, tcol, xyz in blocks(df):
        v = df[xyz].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if name.startswith("SQ "):
            sq[name[3:]] = v
        vr = v[R]
        fin = np.isfinite(vr).all(axis=1)
        zero = (vr == 0).all(axis=1)
        good = fin & ~zero
        norm = np.linalg.norm(vr[good], axis=1)
        same_prev = np.r_[False, (np.diff(vr, axis=0) == 0).all(axis=1)]
        b = {
            "nan_frac": float(1 - fin.mean()),
            "allzero_frac": float(zero.mean()),
            "good_frac": float(good.mean()),
            "distinct_max": int(max(len(np.unique(vr[good, k])) for k in range(3))) if good.any() else 0,
            "norm_median": float(np.median(norm)) if len(norm) else None,
            "norm_iqr": [float(np.percentile(norm, 25)), float(np.percentile(norm, 75))] if len(norm) else None,
            "repeat_prev_frac": float(same_prev[good].mean()) if good.any() else None,
        }
        if tcol is not None:
            t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float)[R]
            tf = np.isfinite(t)
            b["ts_nan_frac"] = float(1 - tf.mean())
            b["ts_equals_SQ_frac"] = float(np.mean(t[tf] == ts[R][tf])) if tf.any() else None
            b["ts_offset_median_ms"] = float(np.median(t[tf] - ts[R][tf])) if tf.any() else None
            dt = np.diff(t[tf])
            b["ts_repeat_frac"] = float(np.mean(dt == 0)) if len(dt) else None
            b["ts_backwards_frac"] = float(np.mean(dt < 0)) if len(dt) else None
        out["blocks"][name] = b

    # sync check: IMU activity envelope vs marker speed
    for name, tcol, xyz in blocks(df):
        if name.startswith("SQ "):
            continue
        site = " - ".join(name.split(" - ")[:2])
        ref = REF_MARKER.get(site)
        if ref is None or ref not in sq:
            continue
        b = out["blocks"][name]
        if b["good_frac"] < 0.5 or b["distinct_max"] < 50 or name.endswith("MAG") or name.endswith("LAC"):
            continue
        v = df[xyz].apply(pd.to_numeric, errors="coerce").to_numpy(float).copy()
        v[(v == 0).all(axis=1)] = np.nan
        a = envelope(np.linalg.norm(v, axis=1))
        s = pd.Series(speed(sq[ref])).rolling(FS, center=True, min_periods=FS // 2).mean().to_numpy()
        a, s = a[R], s[R]
        res = {"ref_marker": ref, "whole": best_lag(s, a)}
        q = len(a) // 4
        res["quarters"] = [best_lag(s[k * q:(k + 1) * q], a[k * q:(k + 1) * q]) for k in range(4)]
        b["sync"] = res

    # label checks against Qualisys
    act = df["etiq activite"].astype("string").fillna("").to_numpy(object)[R]
    chk = {}
    if "Torse" in sq:
        tsp = pd.Series(speed(sq["Torse"])).rolling(FS, center=True, min_periods=FS // 2).mean().to_numpy()[R]
        w = (act == "Marcher").astype(float)
        ok = np.isfinite(tsp)
        chk["walk_vs_torso_speed"] = best_lag(tsp, w) if w.sum() else None
        chk["walk_auc_torso_speed"] = auc(tsp, act == "Marcher")
        chk["torso_speed_mm_s_walk_vs_not"] = [float(np.nanmedian(tsp[act == "Marcher"])),
                                               float(np.nanmedian(tsp[(act != "Marcher") & ok]))] if w.sum() else None
        # same AUC with the labels shifted by +-k s: if the best shift is far from 0, labels are offset
        shifts = {}
        for k in (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3):
            n = int(k * FS)
            wl = np.roll(act == "Marcher", n)
            shifts[str(k)] = auc(tsp, wl)
        chk["walk_auc_vs_label_shift_s"] = shifts
    if "D_main" in sq:
        hsp = pd.Series(speed(sq["D_main"])).rolling(FS // 2, center=True, min_periods=5).mean().to_numpy()[R]
        arm = np.isin(act, ["Lever_Bras", "Baisser_Bras", "Lever_2_Bras", "Baisser_2_Bras", "Autre_Mouvement_Bras"])
        still = act == ""
        if arm.any() and still.any():
            chk["hand_speed_mm_s_arm_vs_empty"] = [float(np.nanmedian(hsp[arm])), float(np.nanmedian(hsp[still]))]
            chk["arm_auc_hand_speed_vs_empty"] = auc(hsp[arm | still], arm[arm | still])
    out["label_checks"] = chk
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2022", choices=["2021", "2022"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    folder = RAW / ("capture_2022-01-26" if args.year == "2022" else "capture_2021-05-31")
    files = sorted(folder.glob("*.csv"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
    res = []
    for p in files:
        print("...", p.name, flush=True)
        res.append(audit_file(p))
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("written", args.out)


if __name__ == "__main__":
    main()
