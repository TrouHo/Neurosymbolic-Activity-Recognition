"""Do posture-state labels (S, s_all) agree with body geometry from Qualisys? (read-only, 2026-11-09)

For each participant and posture state: median head height (z of 'Tete', mm) and pelvis height
('Bassin', or mean of the two hip markers when Bassin is missing). If labels are right, standing
must be highest, lying lowest, etc. Also lists the share of each state's rows where head height
is implausible for that state (standing with head < 1200 mm; lying with head > 900 mm).
Assumption checked first: z is the vertical axis (standing head height ~1.5-1.8 m).
"""
import glob
import os

import numpy as np
import pandas as pd

from emob import config as C
from emob.data.io import load_recording
from emob.data.labels import annotated_mask, posture_states

print("state order:", C.S_CLASSES)
for f in sorted(glob.glob(str(C.RAW_2022 / "*.csv"))):
    u = os.path.basename(f)[:5]
    rec = load_recording(f)
    df = pd.read_csv(f, low_memory=False, usecols=lambda c: c.startswith(("x_sq", "y_sq", "z_sq")))
    head = df["z_sq_Tete"].to_numpy(float).copy()
    head[head == 0] = np.nan
    hips = df[["z_sq_D_hanche", "z_sq_G_hanche"]].to_numpy(float).copy()
    hips[hips == 0] = np.nan
    pelvis = np.nanmean(hips, axis=1)
    ps = posture_states(rec.families[C.BODY_COL], annotated_mask(rec.families))
    st = ps.state_all
    row = []
    for k, name in enumerate(C.S_CLASSES):
        m = st == k
        if m.sum() == 0:
            row.append(f"{name}: -"); continue
        h = np.nanmedian(head[m]); p = np.nanmedian(pelvis[m])
        row.append(f"{name}: n={m.sum()/50:.0f}s head {h:.0f} hip {p:.0f}")
    bad_stand = np.nanmean(head[st == 0] < 1200) if (st == 0).any() else np.nan
    bad_lie = np.nanmean(head[st == 2] > 900) if (st == 2).any() else np.nan
    print(f"{u} | " + " | ".join(row) + f" || standing&head<1200: {bad_stand:.3f}  lying&head>900: {bad_lie:.3f}")
