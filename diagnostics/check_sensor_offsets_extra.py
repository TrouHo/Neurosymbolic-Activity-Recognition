"""Follow-up checks for E1 (2026-11-17), read-only.

1. Phone GYR vs phone ACC: same device, so the lag between them should be 0 (then one offset per
   device is justified).
2. user3 SB devices (failed the E1 gate on interval width / drift): window lags with 60-s windows
   and a wider search (+-2 s), to tell a real drift from a bad first window.
3. Estimator check: shift the phone signal of user6 by a known amount and recover it.
Usage (from code/): .venv\\Scripts\\python.exe -m diagnostics.check_sensor_offsets_extra
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from emob import config as C
from estimate.sensor_offsets import FS, device_signal, marker_signal, rank, search


def load(user, cols):
    f = next(C.RAW_2022.glob(f"{user}-*.csv"))
    return pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False).apply(pd.to_numeric, errors="coerce")


def gyr_signal(xyz):
    xyz = np.asarray(xyz, float)
    m = np.linalg.norm(xyz, axis=1)
    m[~np.isfinite(xyz).all(1)] = np.nan
    return pd.Series(m).rolling(10, center=True, min_periods=5).mean().to_numpy()


print("1. Phone GYR against phone ACC (lag > 0: GYR later than ACC)")
acc_c, gyr_c = C.BLOCKS["phone_acc"][1], C.BLOCKS["phone_gyr"][1]
for u in [f"user{i}" for i in range(1, 9)]:
    df = load(u, set(acc_c + gyr_c))
    a = rank(device_signal(df[acc_c].to_numpy(float)))
    g = rank(gyr_signal(df[gyr_c].to_numpy(float)))
    lag, r = search(a, g, 0, 1.0 * FS, 1)
    print(f"   {u}: {lag / FS:+.2f} s (r {r:.2f})")

print("\n2. user3 SB devices: 60-s windows, search +-2 s around -14 s")
dev = {d: C.DEVICES[d] for d in ("hand", "belt", "leg")}
cols = set()
for blk, mk in dev.values():
    cols |= set(C.BLOCKS[blk][1]) | {f"{a}_sq_{mk}" for a in "xyz"}
df = load("user3", cols)
n = len(df)
for d, (blk, mk) in dev.items():
    x = rank(marker_signal(df[[f"{a}_sq_{mk}" for a in "xyz"]].to_numpy(float)))
    y = rank(device_signal(df[C.BLOCKS[blk][1]].to_numpy(float)))
    row = []
    for s in range(0, n - 60 * FS + 1, 60 * FS):
        xs = np.full(n, np.nan)
        xs[s:s + 60 * FS] = x[s:s + 60 * FS]
        lag, r = search(xs, y, -14.0 * FS, 2.0 * FS, 1)
        row.append(f"{(s + 30 * FS) / FS:.0f}:{lag / FS:+.2f}/{r:.2f}")
    print(f"   {d:5s} " + "  ".join(row))

print("\n3. Estimator recovers a known shift (user6 phone, true lag +0.20 s measured)")
blk, mk = C.DEVICES["phone"]
df = load("user6", set(C.BLOCKS[blk][1]) | {f"{a}_sq_{mk}" for a in "xyz"})
x = rank(marker_signal(df[[f"{a}_sq_{mk}" for a in "xyz"]].to_numpy(float)))
y0 = rank(device_signal(df[C.BLOCKS[blk][1]].to_numpy(float)))
for k in (-25, -7, 0, 3, 50):
    y = np.roll(y0, k)
    lag, r = search(x, y, 0, 2.0 * FS, 1)
    print(f"   added {k / FS:+.2f} s -> measured {lag / FS:+.2f} s (expected {0.20 + k / FS:+.2f})")
