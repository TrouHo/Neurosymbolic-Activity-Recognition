"""Is each sensor's lag vs Qualisys constant over the recording? (read-only, 2026-11-09)

Lag of the IMU activity envelope against the nearest marker's speed, in 120-s windows
(step 60 s), searched over +-20 s. Constant lag -> fixable by one shift; changing lag -> not.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from audit_followup import env, mspeed, lag_scan  # noqa: E402  (module has no side effects on import? see guard)

FS = 50
PAIRS = {"phone": (["x_sf", "y_sf", "z_sf"], "D_hanche"),
         "SB hand": (["x_sb.6", "y_sb.6", "z_sb.6"], "D_main"),
         "SB belt": (["x_sb", "y_sb", "z_sb"], "Bassin"),
         "SB leg": (["x_sb.3", "y_sb.3", "z_sb.3"], "D_genou")}

for f in sorted(glob.glob(r"D:\Projects\EmoB\data\raw\capture_2022-01-26\*.csv")):
    u = os.path.basename(f)[:5]
    df = pd.read_csv(f, low_memory=False)
    fam = ["etiq standard", "etiq mvt_corps", "etiq mvt_bras", "etiq mvt_jambe", "etiq activite"]
    idx = np.flatnonzero((df[fam].astype("string").fillna("") != "").any(axis=1).to_numpy())
    lo, hi = idx[0], idx[-1] + 1
    for name, (cols, mk) in PAIRS.items():
        v = df[cols].to_numpy(float).copy()
        ok = np.isfinite(v).all(1) & ~(v == 0).all(1)
        if ok.mean() < 0.5 or max(len(np.unique(v[ok, k])) for k in range(3)) < 50:
            continue
        e = env(v)
        s = mspeed(df[[f"x_sq_{mk}", f"y_sq_{mk}", f"z_sq_{mk}"]].to_numpy(float), 2000)
        out = []
        for a in range(lo, hi - 120 * FS + 1, 60 * FS):
            L, rb, _, rfar = lag_scan(s[a:a + 120 * FS], e[a:a + 120 * FS], max_s=20)
            out.append(f"{L:+5.1f}({rb:.2f})")
        print(f"{u} {name:8s} lag per 120-s window, s (r): {' '.join(out)}")
