"""Timeline around the start of a walking bout: label vs Qualisys vs IMU (read-only, 2026-11-09).

Shows what the 'lag' of the audit means. Columns per time step:
  label        : etiq activite (W = Marcher, . = other / empty)
  torso mm/s   : speed of the Qualisys torso marker (camera system = independent reference)
  <sensor>     : activity of the IMU = std of |a| over the step (m/s^2); ~0.0x when still
Usage: .venv\\Scripts\\python.exe diagnostics\\evidence_lag_timeline.py user3 leg 1.0 -20 8
       (participant, sensor phone|hand|belt|leg, step s, from s, to s relative to label onset)
"""
import glob
import sys

import numpy as np
import pandas as pd

FS = 50
COLS = {"phone": ["x_sf", "y_sf", "z_sf"], "belt": ["x_sb", "y_sb", "z_sb"],
        "leg": ["x_sb.3", "y_sb.3", "z_sb.3"], "hand": ["x_sb.6", "y_sb.6", "z_sb.6"]}
u, sensor, step, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
f = glob.glob(rf"D:\Projects\EmoB\data\raw\capture_2022-01-26\{u}-*.csv")[0]
df = pd.read_csv(f, low_memory=False)
walk = (df["etiq activite"].astype("string").fillna("") == "Marcher").to_numpy()
# walking bout of >= 2 s with the longest walking-free stretch before it
starts = np.flatnonzero(walk & ~np.r_[False, walk[:-1]])
ok = [s for s in starts if walk[s:s + 2 * FS].all()]
gap = [s - (np.flatnonzero(walk[:s])[-1] if walk[:s].any() else 0) for s in ok]
onset = ok[int(np.argmax(gap))]
print(f"(walking-free time before this bout: {max(gap) / FS:.1f} s)")
tor = df[["x_sq_Torse", "y_sq_Torse", "z_sq_Torse"]].to_numpy(float)
sp = np.r_[np.nan, np.linalg.norm(np.diff(tor, axis=0), axis=1) * FS]
sp[sp > 2000] = np.nan
mag = np.linalg.norm(df[COLS[sensor]].to_numpy(float), axis=1)
n = int(round(step * FS))
print(f"{u}: walking label starts at file row {onset} (t = 0); sensor = {sensor}")
print(f"{'t (s)':>6} | label | torso mm/s (Qualisys) | {sensor} activity (IMU)")
for a in range(onset + int(t0 * FS), onset + int(t1 * FS), n):
    lab = "W" if walk[a:a + n].mean() > 0.5 else "."
    s = np.nanmean(sp[a:a + n]); e = np.nanstd(mag[a:a + n])
    print(f"{(a - onset) / FS:+6.1f} |   {lab}   | {s:6.0f} {'#' * int(min(s, 1200) / 60):20s} | {e:5.2f} {'#' * int(min(e, 3) * 7)}")
