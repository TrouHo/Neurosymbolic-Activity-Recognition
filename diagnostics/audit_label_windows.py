"""Windowed check of walking-label timing against Qualisys torso speed (read-only, 2026-11-09).

Re-tests the 2026-11-05 claim (dbg_drift.py) that user1-4 walking labels have lags jumping
between -39 and +32 s. Here: 120-s windows (step 60 s); in each window with >= 5 s of walking,
the label track is shifted by -20..+20 s (0.2-s steps) and the shift maximising AUC of torso
speed (walking vs not) is kept. Torso speed: 3-D marker speed, values > 2000 mm/s dropped
(marker swaps), 1-s mean. AUC is rank-based, so isolated spikes cannot dominate it.
"""
import glob
import os

import numpy as np
import pandas as pd

FS = 50


def auc(score, pos):
    ok = np.isfinite(score); s, p = score[ok], pos[ok]
    n1, n0 = p.sum(), (~p).sum()
    if n1 < FS or n0 < FS:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


for f in sorted(glob.glob(r"D:\Projects\EmoB\data\raw\capture_2022-01-26\*.csv")):
    u = os.path.basename(f)[:5]
    df = pd.read_csv(f, low_memory=False)
    tor = df[["x_sq_Torse", "y_sq_Torse", "z_sq_Torse"]].to_numpy(float).copy()
    tor[(tor == 0).all(1)] = np.nan
    sp = np.r_[np.nan, np.linalg.norm(np.diff(tor, axis=0), axis=1) * FS]
    sp[sp > 2000] = np.nan
    sp = pd.Series(sp).rolling(FS, center=True, min_periods=FS // 2).mean().to_numpy()
    walk = (df["etiq activite"].astype("string").fillna("") == "Marcher").to_numpy()
    idx = np.flatnonzero(walk)
    out = []
    for a in range(0, len(df) - 120 * FS + 1, 60 * FS):
        b = a + 120 * FS
        if walk[a:b].sum() < 5 * FS:
            continue
        best = (np.nan, -1)
        for L in np.arange(-20, 20.01, 0.2):
            k = int(round(L * FS))
            w = np.roll(walk, k)[a:b]
            v = auc(sp[a:b], w)
            if v == v and v > best[1]:
                best = (L, v)
        a0 = auc(sp[a:b], walk[a:b])
        out.append(f"{a/FS:4.0f}s:{best[0]:+5.1f}({best[1]:.2f}|0:{a0:.2f})")
    print(u, " ".join(out))
