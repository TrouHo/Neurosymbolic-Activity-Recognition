"""run2: which time scale makes the sensor match Qualisys? (read-only, 2026-11-09)

The log says '24576 ticks (614 ms)', i.e. a 40 kHz tick. If that is wrong, sensor time is stretched
and no single offset can align it. Sensor time is multiplied by a factor c (c = 1: 40 kHz is right)
and the best correlation over all offsets is recorded for each c. A clear maximum well above the
c = 1 value, with a sharp peak, identifies the real clock rate.
Usage: .venv\\Scripts\\python.exe diagnostics\\audit_run2_clock.py 1_poignet 1_ceinture 7_poignet ...
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import lagtools as T  # noqa: E402
from run2io import read_log, read_qtm, sample_times, signed, RAW, CAND  # noqa: E402

cases = sys.argv[1:] or ["1_poignet", "1_ceinture", "7_poignet", "12_poignet"]
COMPACT = cases == ["all"]
if COMPACT:
    import glob
    import re
    cases = sorted((os.path.basename(p)[6:-4] for p in glob.glob(os.path.join(RAW, "acc_data", "*.txt"))),
                   key=lambda s: (s.split("_")[1], int(s.split("_")[0])))
    print("per log: best clock factor c (sensor time x c), r, lag, other peak; then per-120-s-window lags at that c")
qcache = {}
for case in cases:
    k, site = case.split("_"); k = int(k)
    kinds, ts, blk, raw, scaled, mal, mism = read_log(os.path.join(RAW, "acc_data", f"output{k}_{site}.txt"))
    t, per = sample_times(ts, blk, len(raw))
    t = t - t[0]
    if k not in qcache:
        _, _, names, q = read_qtm(os.path.join(RAW, "trajectoires", f"patient{k:02d}.tsv"))
        qt = q["Time"].to_numpy(float)
        qcache[k] = (qt[-1], {m: T.marker_speed(qt, q[[f"{m} X", f"{m} Y", f"{m} Z"]].to_numpy(float)) for m in names})
    qdur, spd = qcache[k]
    rows = []
    grid_c = np.arange(0.96, 1.121, 0.01) if COMPACT else np.r_[np.arange(0.80, 1.30, 0.02), 40000 / 32768, 32768 / 40000]
    for c in grid_c:
        ti, e = T.imu_activity(t * c, signed(raw), fs=50)
        T0 = min(0, ti[0]) - 1; T1 = max(ti[-1], qdur) + 1
        eg = T.regular(ti, e, T.HZ, T0, T1)[1]
        best = max((T.summarize(*T.xcorr(T.regular(*spd[m], T.HZ, T0, T1)[1], eg, T1 - T0 - 60, min_overlap_s=120)) | {"m": m}
                    for m in CAND[site]), key=lambda s: s["r"] or -1)
        rows.append((c, best))
    top = max(rows, key=lambda x: x[1]["r"])
    if COMPACT:
        c, s = top
        ti, e = T.imu_activity(t * c, signed(raw), fs=50)
        T0 = min(0, ti[0]) - 1; T1 = max(ti[-1], qdur) + 1
        eg = T.regular(ti, e, T.HZ, T0, T1)[1]
        sg = T.regular(*spd[s["m"]], T.HZ, T0, T1)[1]
        w = T.windowed(sg, eg, s["lag"] - T0 * 0, search_s=5)
        r1 = [x for x in rows if abs(x[0] - 1) < 1e-9][0][1]["r"]
        print(f"output{k}_{site}: c {c:.2f} r {s['r']:.2f} (c=1: {r1:.2f}) lag {s['lag']:+7.1f} s other peak {s['r_far']:.2f} "
              f"marker {s['m']}; overlap with Qualisys {max(0, min(qdur, ti[-1] - s['lag']) - max(0, ti[0] - s['lag'])):.0f} s; "
              f"window lags {[round(x[1], 1) for x in w]}")
        continue
    print(f"\noutput{k}_{site}: sensor span {t[-1]:.1f} s at 40 kHz, Qualisys {qdur:.1f} s")
    for c, s in sorted(rows):
        mark = " <== best" if c == top[0] else (" (c=1: 40 kHz as stated)" if abs(c - 1) < 1e-9 else "")
        print(f"  c {c:.4f}: r {s['r']:.2f} lag {s['lag']:+7.1f} s other peak {s['r_far']:.2f} {s['m']}{mark}")
