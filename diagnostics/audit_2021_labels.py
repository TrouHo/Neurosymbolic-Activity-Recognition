"""2021: do the label tracks belong to the recording they sit in? (read-only, 2026-11-09)

For every pair (labels of file A, Qualisys of file B): walking indicator of A vs torso speed of B,
best rank correlation over shifts of +-900 s. If A's labels belong to A, the diagonal wins.
Also: label drift = slope of the per-window walking-label lag (s per 10 min).
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import lagtools as T  # noqa: E402

files = sorted(glob.glob(r"D:\Projects\EmoB\data\raw\capture_2021-05-31\*.csv"),
               key=lambda p: int(re.search(r"\d+", os.path.basename(p)).group()))
walk, torso = {}, {}
for f in files:
    k = os.path.basename(f)[:5]
    df = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("ts_SQ", "etiq activite") or "Torse" in c and "sq" in c)
    t = df["ts_SQ"].to_numpy(float) / 1000
    g = np.arange(t[0], t[-1], 1 / T.HZ)
    ts_, v = T.marker_speed(t, df[["x_sq_Torse", "y_sq_Torse", "z_sq_Torse"]].to_numpy(float))
    torso[k] = T.regular(ts_, v, T.HZ, g[0], g[-1] + 1e-9)[1]
    w = (df["etiq activite"].astype("string").fillna("") == "Marcher").to_numpy(float)
    if w.sum() > 100:
        walk[k] = T.regular(t, w, T.HZ, g[0], g[-1] + 1e-9)[1]

print("best r (lag s) of walking labels of ROW vs torso speed of COLUMN, shifts +-900 s")
print("labels\\mocap " + " ".join(f"{k:>14s}" for k in torso))
for a, w in walk.items():
    cells = []
    for b, s in torso.items():
        S = T.summarize(*T.xcorr(s, w, 900, min_overlap_s=120))
        cells.append(f"{S['r']:.2f} ({S['lag']:+6.1f})" if S["r"] is not None else "   -")
    print(f"{a:12s} " + " ".join(f"{c:>14s}" for c in cells))

print("\nlabel drift (own recording): per-120-s-window lag of walking labels vs torso speed")
for a, w in walk.items():
    S = T.summarize(*T.xcorr(torso[a], w, 120))
    win = T.windowed(torso[a], w, S["lag"], search_s=20)
    if len(win) >= 3:
        x = np.array([v[0] for v in win]); y = np.array([v[1] for v in win])
        slope = np.polyfit(x, y, 1)[0] * 600
        print(f"{a}: lags {[round(v, 1) for v in y]}  -> slope {slope:+.2f} s per 10 min")
