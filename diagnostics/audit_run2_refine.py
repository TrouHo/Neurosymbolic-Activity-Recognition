"""run2: refined clock factor per log, and wrist-belt consistency (read-only, 2026-11-09).

1. For each log, c is searched in steps of 0.002 around 1.00-1.10; reported: best c, r, other peak,
   overlap with Qualisys, and the r of each 120-s window at that alignment (all windows should be high
   if the alignment is right).
2. For participants with wrist and belt: offset implied by their two Qualisys alignments vs offset
   found by matching wrist activity directly to belt activity (both rescaled by their c).
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import lagtools as T  # noqa: E402
from run2io import read_log, read_qtm, sample_times, signed, RAW, CAND  # noqa: E402

logs = sorted((os.path.basename(p)[6:-4] for p in glob.glob(os.path.join(RAW, "acc_data", "*.txt"))),
              key=lambda s: (int(s.split("_")[0]), s.split("_")[1]))
qcache, res = {}, {}
for case in logs:
    k, site = case.split("_"); k = int(k)
    _, ts, blk, raw, _, _, _ = read_log(os.path.join(RAW, "acc_data", f"output{k}_{site}.txt"))
    t, _ = sample_times(ts, blk, len(raw)); t = t - t[0]
    if k not in qcache:
        _, _, names, q = read_qtm(os.path.join(RAW, "trajectoires", f"patient{k:02d}.tsv"))
        qt = q["Time"].to_numpy(float)
        qcache[k] = (qt[-1], {m: T.marker_speed(qt, q[[f"{m} X", f"{m} Y", f"{m} Z"]].to_numpy(float)) for m in names})
    qdur, spd = qcache[k]

    def score(c):
        ti, e = T.imu_activity(t * c, signed(raw), fs=50)
        T0 = min(0, ti[0]) - 1; T1 = max(ti[-1], qdur) + 1
        eg = T.regular(ti, e, T.HZ, T0, T1)[1]
        out = []
        for m in CAND[site]:
            sg = T.regular(*spd[m], T.HZ, T0, T1)[1]
            s = T.summarize(*T.xcorr(sg, eg, T1 - T0 - 60, min_overlap_s=120))
            out.append((s["r"] or -1, s, m, sg, eg, T0, ti))
        return max(out, key=lambda x: x[0])

    coarse = max(np.arange(1.00, 1.101, 0.01), key=lambda c: score(c)[0])
    best_c = max(np.arange(coarse - 0.012, coarse + 0.0121, 0.002), key=lambda c: score(c)[0])
    r, s, m, sg, eg, T0, ti = score(best_c)
    w = T.windowed(sg, eg, s["lag"], search_s=2)
    overlap = max(0, min(qdur, ti[-1] - s["lag"]) - max(0, ti[0] - s["lag"]))
    res[(k, site)] = (best_c, s["lag"], t, raw)
    print(f"output{k}_{site:8s} c {best_c:.3f} r {r:.2f} other peak {s['r_far']:.2f} marker {m:9s} lag {s['lag']:+7.1f} s "
          f"overlap {overlap:4.0f} s | window r {[round(x[2], 2) for x in w]}", flush=True)

print("\nwrist vs belt of the same participant")
for k in sorted({k for k, _ in res}):
    if (k, "poignet") not in res or (k, "ceinture") not in res:
        continue
    cw, lw, tw, rw = res[(k, "poignet")]; cb, lb, tb, rb = res[(k, "ceinture")]
    tiw, ew = T.imu_activity(tw * cw, signed(rw), fs=50)
    tib, eb = T.imu_activity(tb * cb, signed(rb), fs=50)
    T0 = min(tiw[0], tib[0]) - 1; T1 = max(tiw[-1], tib[-1]) + 1
    s = T.summarize(*T.xcorr(T.regular(tiw, ew, T.HZ, T0, T1)[1], T.regular(tib, eb, T.HZ, T0, T1)[1], T1 - T0 - 60, min_overlap_s=120))
    print(f" participant {k:2d}: via Qualisys belt-wrist = {lb - lw:+7.1f} s | direct wrist-belt match {s['lag']:+7.1f} s "
          f"(r {s['r']:.2f}, other peak {s['r_far']:.2f})")
