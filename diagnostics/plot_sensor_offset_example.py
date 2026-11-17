"""Figure for E1 (2026-11-17): what a sensor time offset looks like, before and after alignment.

Two examples, each a short stretch of one recording: the device signal (|a| minus gravity,
0.2-s mean) and the Qualisys signal of the nearest marker (acceleration magnitude, 0.2-s mean),
both scaled to [0, 1] for display. Top: as in the CSV. Bottom: device shifted by the estimated
lag from params/sensor_offsets.json.
Usage (from code/): .venv\\Scripts\\python.exe -m diagnostics.plot_sensor_offset_example <out.png>
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emob import config as C
from estimate.sensor_offsets import FS, device_signal, marker_signal

EXAMPLES = [("user3", "hand", 300, 330), ("user2", "phone", 300, 315)]
OFF = json.load(open(C.PARAMS / "sensor_offsets.json", encoding="utf-8"))["participants"]


def scale(x, s, e):
    seg = x[s:e]
    lo, hi = np.nanpercentile(seg, 2), np.nanpercentile(seg, 98)
    return np.clip((x - lo) / (hi - lo), 0, 1.2)


fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharey=True)
for col, (u, dev, t0, t1) in enumerate(EXAMPLES):
    blk, mk = C.DEVICES[dev]
    f = next(C.RAW_2022.glob(f"{u}-*.csv"))
    df = pd.read_csv(f, usecols=lambda c: c in set(C.BLOCKS[blk][1]) | {f"{a}_sq_{mk}" for a in "xyz"},
                     low_memory=False).apply(pd.to_numeric, errors="coerce")
    m = marker_signal(df[[f"{a}_sq_{mk}" for a in "xyz"]].to_numpy(float))
    d = device_signal(df[C.BLOCKS[blk][1]].to_numpy(float))
    lag = OFF[u]["devices"][dev]["median"]
    k = int(round(lag * FS))
    d_al = np.full_like(d, np.nan)
    if k >= 0:
        d_al[:len(d) - k] = d[k:]
    else:
        d_al[-k:] = d[:k]
    t = np.arange(len(d)) / FS
    s, e = t0 * FS, t1 * FS
    for row, (dd, title) in enumerate([(d, "as in the CSV"), (d_al, f"device shifted by {lag:+.2f} s")]):
        ax = axes[row, col]
        ax.plot(t[s:e], scale(m, s, e)[s:e], color="#1f6feb", lw=1.4, label=f"Qualisys marker {mk} (acceleration)")
        ax.plot(t[s:e], scale(dd, s, e)[s:e], color="#d1495b", lw=1.2, label=f"{dev} sensor (|a| − gravity)")
        ax.set_title(f"{u}, {dev}: {title}", fontsize=10)
        ax.set_xlim(t0, t1)
        ax.grid(alpha=0.3)
        if row == 1:
            ax.set_xlabel("time in the recording (s)")
        if col == 0:
            ax.set_ylabel("scaled 0–1")
    axes[0, col].legend(fontsize=8, loc="upper right")
fig.suptitle("Sensor time offset against Qualisys: before (top) and after (bottom) alignment", fontsize=11)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=130)
print("written", sys.argv[1])
