"""Audit of run2 = Session 4 (2024-06-04), read-only, 2026-11-09.

Sensor logs (outputN_poignet / outputN_ceinture .txt): rows by kind, malformed rows, the device clock
('Timestamp' rows, 24576 ticks = 614 ms -> 40 kHz), samples per clock block, sampling rate,
signed vs supplied scaled values, |a| in counts. Each sample's time = its position inside its clock
block (uniform spacing).
Qualisys (patientNN.tsv, 100 Hz): duration, marker coverage. Video: duration (mp4 header).
Timing: sensor activity vs wrist / hip / torso marker speed over all shifts (device clock has no
wall-clock reference), best marker (gives the side), per-window lags (clock drift).
File pairing: each wrist log against both wrist markers of every Qualisys file.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import lagtools as T  # noqa: E402

RAW = r"D:\Projects\EmoB\data\raw\run2"
TICK = 40000.0
CAND = {"poignet": ["G poignet", "D poignet"], "ceinture": ["Torse", "G hanche", "D hanche"]}


def read_log(path):
    kinds, ts, blocks, vals, scaled, malformed, mism = {}, [], [], [], [], [], 0
    cur = -1
    for line in open(path, encoding="utf-8", errors="replace").read().splitlines()[2:]:
        p = line.split(";")
        kind = p[4] if len(p) > 4 else "?"
        if len(p) < 6:
            malformed.append(line); continue
        if kind == "Timestamp":
            ts.append(int(p[5])); cur += 1
        elif kind in ("Accelerometer", ""):
            try:
                v = [int(p[5]), int(p[6]), int(p[7])]; s = [float(p[8]), float(p[9]), float(p[10])]
            except (ValueError, IndexError):
                malformed.append(line); continue
            if kind == "Accelerometer":
                b = p[2].split()
                if len(b) == 6 and [int(b[i + 1] + b[i], 16) for i in (0, 2, 4)] != v:
                    mism += 1
            vals.append(v); scaled.append(s); blocks.append(cur)
            kind = "Accelerometer (continuation row)" if kind == "" else kind
        else:
            malformed.append(line)
        kinds[kind] = kinds.get(kind, 0) + 1
    return kinds, np.array(ts, float), np.array(blocks), np.array(vals, float), np.array(scaled, float), malformed, mism


def read_qtm(path):
    blob = open(path, "rb").read(20000)
    enc = "utf-8"
    try:
        blob.decode("utf-8")
    except UnicodeDecodeError:
        enc = "cp1252"
    hdr, n = {}, 0
    with open(path, encoding=enc) as fh:
        for line in fh:
            n += 1
            p = line.rstrip("\n").split("\t")
            if p[0] == "Frame":
                break
            hdr[p[0]] = p[1:]
    df = pd.read_csv(path, sep="\t", skiprows=n - 1, encoding=enc)
    return hdr["TIME_STAMP"][0], int(hdr["FREQUENCY"][0]), hdr["MARKER_NAMES"], df


# ---------- Qualisys + video
Q = {}
print("=== Qualisys and video")
for f in sorted(glob.glob(os.path.join(RAW, "trajectoires", "patient*.tsv"))):
    k = int(re.search(r"(\d+)", os.path.basename(f)).group(1))
    t0, fs, names, q = read_qtm(f)
    qt = q["Time"].to_numpy(float)
    cov = {m: round(float(((q[[f"{m} X", f"{m} Y", f"{m} Z"]] != 0).any(axis=1)).mean()), 2) for m in names}
    Q[k] = {m: T.marker_speed(qt, q[[f"{m} X", f"{m} Y", f"{m} Z"]].to_numpy(float)) for m in names}
    Q[k]["_dur"] = qt[-1]
    v = os.path.join(RAW, "videos", f"part{k}_video.mp4")
    vi = T.mp4_info(v) if os.path.exists(v) else None
    print(f" patient{k:02d}: start {t0}, {fs} Hz, {qt[-1]:.1f} s | video {vi['duration_s']:.1f} s | coverage {cov}")

# ---------- sensor logs
print("\n=== sensor logs")
S = {}
for f in sorted(glob.glob(os.path.join(RAW, "acc_data", "*.txt")),
                key=lambda p: (re.search(r"_(\w+)\.txt", p).group(1), int(re.search(r"output(\d+)", p).group(1)))):
    k = int(re.search(r"output(\d+)", f).group(1)); site = re.search(r"_(\w+)\.txt", f).group(1)
    kinds, ts, blk, raw, scaled, malformed, mism = read_log(f)
    per = np.bincount(blk[blk >= 0], minlength=len(ts))
    dts = np.diff(ts)
    # sample times: uniform inside each clock block
    t = np.full(len(raw), np.nan)
    step = np.median(dts) / np.median(per[per > 0])
    for b in range(len(ts)):
        i = np.flatnonzero(blk == b)
        if len(i) == 0:
            continue
        span = (ts[b + 1] - ts[b]) if b + 1 < len(ts) else step * len(i)
        t[i] = (ts[b] + span * np.arange(len(i)) / len(i)) / TICK
    sgn = np.where(raw >= 32768, raw - 65536, raw)
    mag_s, mag_u = np.linalg.norm(sgn, axis=1), np.linalg.norm(raw, axis=1)
    ratio = scaled / np.where(raw == 0, np.nan, raw)
    dur = (ts[-1] - ts[0]) / TICK
    print(f" output{k}_{site}: kinds {kinds}; malformed {len(malformed)}; byte/value mismatches {mism}")
    print(f"   clock steps {sorted(set(dts.astype(int)))[:4]} ticks, {len(ts)} blocks, {dur:.1f} s, "
          f"samples {len(raw)} -> {len(raw) / dur:.1f} Hz; samples per block median {np.median(per):.0f} "
          f"(min {per.min()}, max {per.max()}, blocks with 0: {(per == 0).sum()})")
    print(f"   scaled/raw ratio {np.nanmedian(ratio):.5f} (spread {np.nanpercentile(ratio, 1):.5f}-{np.nanpercentile(ratio, 99):.5f}); "
          f"|a| counts: signed median {np.median(mag_s):.0f} IQR {np.subtract(*np.percentile(mag_s, [75, 25])):.0f} | "
          f"as supplied (unsigned) median {np.median(mag_u):.0f} IQR {np.subtract(*np.percentile(mag_u, [75, 25])):.0f}; "
          f"rows with a value >= 32768: {np.mean((raw >= 32768).any(1)):.2f}")
    if malformed:
        print("   malformed examples:", malformed[:2])
    S[(k, site)] = (t, sgn)

# ---------- timing vs Qualisys of the same number
print("\n=== timing vs Qualisys (same participant number); lag = sensor clock zero relative to Qualisys zero")
res = {}
for (k, site), (t, sgn) in S.items():
    if k not in Q:
        print(f" output{k}_{site}: no Qualisys file"); continue
    ti, e = T.imu_activity(t, sgn, fs=50)
    T0 = min(ti[0], 0) - 1; T1 = max(ti[-1], Q[k]["_dur"]) + 1
    eg = T.regular(ti, e, T.HZ, T0, T1)[1]
    best = None
    for m in CAND[site]:
        sg = T.regular(*Q[k][m], T.HZ, T0, T1)[1]
        s = T.summarize(*T.xcorr(sg, eg, T1 - T0 - 60, min_overlap_s=120))
        if s["r"] is not None and (best is None or s["r"] > best[1]["r"]):
            best = (m, s, sg)
    m, s, sg = best
    w = T.windowed(sg, eg, s["lag"], search_s=10)
    y = np.array([v[1] for v in w]); x = np.array([v[0] for v in w])
    slope = np.polyfit(x, y, 1)[0] * 600 if len(w) >= 3 else np.nan
    res[(k, site)] = s["lag"]
    print(f" output{k}_{site}: best marker {m}, lag {s['lag']:+.1f} s, r {s['r']:.2f} (other peak {s['r_far']:.2f}); "
          f"window lags {[round(v, 1) for v in y]} -> drift {slope:+.2f} s/10 min")

print("\n=== wrist vs belt offset of the same participant (from their lags vs Qualisys)")
for k in sorted({k for k, _ in S}):
    if (k, "poignet") in res and (k, "ceinture") in res:
        print(f" participant {k}: belt lag - wrist lag = {res[(k, 'ceinture')] - res[(k, 'poignet')]:+.1f} s")

print("\n=== pairing: best r of each wrist log vs the wrist markers of every Qualisys file")
keys = sorted(Q)
print("log \\ qtm " + " ".join(f"{k:>5d}" for k in keys))
for (k, site), (t, sgn) in S.items():
    if site != "poignet":
        continue
    ti, e = T.imu_activity(t, sgn, fs=50)
    row = []
    for j in keys:
        T0 = min(ti[0], 0) - 1; T1 = max(ti[-1], Q[j]["_dur"]) + 1
        eg = T.regular(ti, e, T.HZ, T0, T1)[1]
        r = max((T.summarize(*T.xcorr(T.regular(*Q[j][m], T.HZ, T0, T1)[1], eg, T1 - T0 - 60, min_overlap_s=120))["r"] or -1)
                for m in CAND["poignet"])
        row.append(r)
    best = keys[int(np.argmax(row))]
    print(f"output{k:<3d}   " + " ".join(f"{r:5.2f}" for r in row) + f"   best: patient{best:02d}{'' if best == k else '  <-- not the same number'}")
