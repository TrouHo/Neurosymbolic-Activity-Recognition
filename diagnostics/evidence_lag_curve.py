"""How well an IMU matches Qualisys when the IMU is shifted by X seconds (read-only, 2026-11-09).

r = Spearman correlation, whole labelled region, between the 1-s activity of the IMU and the
1-s speed of the nearest Qualisys marker. The X giving the highest r is the 'lag' of the audit.
Usage: .venv\\Scripts\\python.exe diagnostics\\evidence_lag_curve.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from audit_followup import env, mspeed, corr  # noqa: E402

FS = 50
CASES = [("user3", "SB leg", ["x_sb.3", "y_sb.3", "z_sb.3"], "D_genou", [-20, -15, -14, -13.6, -13, -10, -5, -1, 0, 1, 5]),
         ("user8", "SB leg", ["x_sb.3", "y_sb.3", "z_sb.3"], "D_genou", [-20, -15, -14, -13.6, -13, -10, -5, -1, 0, 1, 5]),
         ("user2", "phone", ["x_sf", "y_sf", "z_sf"], "D_hanche", [-1, -0.4, 0, 0.4, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 3.0]),
         ("user8", "phone", ["x_sf", "y_sf", "z_sf"], "D_hanche", [-1, -0.4, 0, 0.2, 0.4, 0.8, 1.0, 1.2, 1.4, 2.0, 3.0])]

for u, name, cols, mk, shifts in CASES:
    df = pd.read_csv(glob.glob(rf"D:\Projects\EmoB\data\raw\capture_2022-01-26\{u}-*.csv")[0], low_memory=False)
    fam = ["etiq standard", "etiq mvt_corps", "etiq mvt_bras", "etiq mvt_jambe", "etiq activite"]
    idx = np.flatnonzero((df[fam].astype("string").fillna("") != "").any(axis=1).to_numpy())
    R = slice(idx[0], idx[-1] + 1)
    e = env(df[cols].to_numpy(float).copy())
    s = mspeed(df[[f"x_sq_{mk}", f"y_sq_{mk}", f"z_sq_{mk}"]].to_numpy(float), 2000)
    print(f"\n{u} {name} vs Qualisys marker {mk}")
    print("  X = sensor time shift (s); r = match with Qualisys after reading the sensor X s later")
    for X in shifts:
        k = int(round(X * FS))
        e_shift = np.roll(e, -k)          # sensor value at t+X placed at t
        r = corr(s[R], e_shift[R], rank=True)
        print(f"  X = {X:+6.1f}  r = {r:.2f} {'#' * int(max(r, 0) * 40)}")
