"""Audit of run1 = Session 3 (2023-03-13), read-only, 2026-11-09.

Per participant:
  sensors (Bras / Jambe / Dos CSV): rows, malformed rows, sampling, gaps, PC-clock vs device-clock
      rate, |acc| (m/s^2), gyro, and the lag vs Qualisys using the wall-clock times written in the
      files (sensor: 'temps initial' + timestamp_pc; Qualisys: TIME_STAMP + Time), +-900 s scan,
      per-window lags (clock drift);
  Qualisys (.tsv, 300 Hz): duration, marker coverage;
  labels (video time): names, counts, labelled time; offset video -> Qualisys found by matching
      walking intervals to torso speed and arm intervals to hand speed (+-900 s);
  video: duration and creation time from the mp4 header.
"""
import glob
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import lagtools as T  # noqa: E402

RAW = r"D:\Projects\EmoB\data\raw\run1"
CAND = {"Bras": ["Main_Dr", "Main_Gc", "Coude_Dr", "Coude_Gc"],
        "Jambe": ["Genou_Dr", "Genou_Gc", "Pied_Dr", "Pied_Gc"],
        "Dos": ["Torse", "Epaule_Dr", "Epaule_Gc", "Hanche_Dr", "Hanche_Gc"]}


def epoch(dt):
    return (dt - datetime(1970, 1, 1)).total_seconds()   # naive local time, same convention both sides


def read_sensor(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        l1, l2, l3 = fh.readline().strip(), fh.readline().strip(), fh.readline().strip()
    pos = re.search(r"Position (\w+)", l1).group(1)
    t0 = datetime.strptime(l3.split(": ", 1)[1].strip(), "%Y-%m-%d %H:%M:%S.%f")
    raw = pd.read_csv(path, skiprows=3, header=None, dtype=str)
    num = raw.apply(pd.to_numeric, errors="coerce")
    bad = num.isna().any(axis=1) | (raw.shape[1] != 8)
    num = num[~bad].to_numpy(float)
    return pos, t0, raw, bad, num


def read_qtm(path):
    hdr = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        n = 0
        for line in fh:
            n += 1
            p = line.rstrip("\n").split("\t")
            if p[0] == "Frame":
                break
            hdr[p[0]] = p[1:]
    t0 = datetime.strptime(hdr["TIME_STAMP"][0], "%Y-%m-%d, %H:%M:%S.%f")
    df = pd.read_csv(path, sep="\t", skiprows=n - 1)
    return t0, int(hdr["FREQUENCY"][0]), hdr["MARKER_NAMES"], df


def tosec(s):
    h, m, rest = s.split(":"); sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


def read_labels(path):
    rows = []
    blob = open(path, "rb").read()
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        text = blob.decode("cp1252")      # label files are not UTF-8 (e.g. 't\xeate')
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        times = re.findall(r"\d\d:\d\d:\d\d,\d{3}", line)
        rows.append((line.split(",")[0].strip('"').strip(), tosec(times[0]), tosec(times[1])))
    return pd.DataFrame(rows, columns=["name", "start", "end"])


for k in (1, 2, 3, 4):
    print(f"\n{'=' * 90}\nparticipant{k}")
    qt0, qfs, names, q = read_qtm(os.path.join(RAW, "trajectoires", f"participant_{k}_trajectoire.tsv"))
    qt = epoch(qt0) + q["Time"].to_numpy(float)
    print(f" Qualisys: start {qt0}  {qfs} Hz  {len(q)} frames = {len(q) / qfs:.1f} s")
    cov = {m: float(((q[[f'{m} X', f'{m} Y', f'{m} Z']] != 0).any(axis=1) & q[[f'{m} X', f'{m} Y', f'{m} Z']].notna().all(axis=1)).mean())
           for m in names}
    print("  marker coverage:", {m: round(v, 2) for m, v in cov.items()})
    spd = {m: T.marker_speed(qt, q[[f"{m} X", f"{m} Y", f"{m} Z"]].to_numpy(float)) for m in names}

    vid = T.mp4_info(os.path.join(RAW, "videos", f"participant{k}_video.mp4"))
    print(f" video: duration {vid['duration_s']:.1f} s, mp4 creation time {vid['creation_utc']}")

    for f in sorted(glob.glob(os.path.join(RAW, "acc_data", f"participant{k}_data_*.csv"))):
        pos, st0, raw, bad, a = read_sensor(f)
        tpc, mil = a[:, 0] / 1000, a[:, 1] / 1000
        acc, gyr = a[:, 2:5], a[:, 5:8]
        d = np.diff(tpc)
        dur_pc, dur_dev = tpc[-1] - tpc[0], mil[-1] - mil[0]
        same = np.r_[False, (np.diff(acc, axis=0) == 0).all(1)]
        print(f" {os.path.basename(f)}: position {pos}, start {st0}, rows {len(raw)} (malformed {int(bad.sum())}), "
              f"{dur_pc:.1f} s, {len(a) / dur_pc:.1f} Hz, step median {np.median(d) * 1000:.0f} ms, "
              f"p99 {np.percentile(d, 99) * 1000:.0f} ms, max gap {d.max():.2f} s, backward steps {(d < 0).sum()}, "
              f"device/PC duration ratio {dur_dev / dur_pc:.5f}")
        print(f"   |acc| median {np.median(np.linalg.norm(acc, axis=1)):.2f} m/s2 (IQR {np.subtract(*np.percentile(np.linalg.norm(acc, axis=1), [75, 25])):.2f}), "
              f"acc distinct {max(len(np.unique(acc[:, j])) for j in range(3))}, repeated rows {same.mean():.3f}; "
              f"gyro std per axis {gyr.std(0).round(2)}, gyro distinct {max(len(np.unique(gyr[:, j])) for j in range(3))}")
        if bad.any():
            print("   malformed row examples:", [", ".join(map(str, r)) for r in raw[bad].head(3).fillna("").to_numpy()])
        # timing vs Qualisys on the wall-clock axis
        st = epoch(st0) + tpc
        T0, T1 = min(st[0], qt[0]) - 1, max(st[-1], qt[-1]) + 1
        ti, e = T.imu_activity(st, acc)
        eg = T.regular(ti, e, T.HZ, T0, T1)[1]
        best = None
        for m in CAND[pos]:
            sg = T.regular(*spd[m], T.HZ, T0, T1)[1]
            S = T.summarize(*T.xcorr(sg, eg, 900))
            if S["r"] is not None and (best is None or S["r"] > best[1]["r"]):
                best = (m, S, sg)
        if best is None:
            print("   timing vs Qualisys: not measurable (recording too short for the 60-s minimum overlap)")
            continue
        m, S, sg = best
        w = T.windowed(sg, eg, S["lag"], search_s=10)
        x = np.array([v[0] for v in w]); y = np.array([v[1] for v in w])
        slope = np.polyfit(x, y, 1)[0] * 600 if len(w) >= 3 else np.nan
        print(f"   vs Qualisys {m}: lag {S['lag']:+.1f} s (r {S['r']:.2f}; r at the wall-clock alignment {S['r0']}; "
              f"other peak {S['r_far']:.2f}); per-window lags {[round(v, 1) for v in y]} -> drift {slope:+.2f} s/10 min")

    lab = read_labels(os.path.join(RAW, "labels", f"participant{k}_label.csv"))
    print(f" labels: {len(lab)} intervals, {lab['name'].nunique()} names, video time {lab.start.min():.1f}-{lab.end.max():.1f} s, "
          f"labelled {np.sum(lab.end - lab.start):.1f} s, negative/zero-length {(lab.end <= lab.start).sum()}")
    print("  names:", lab.groupby("name").size().sort_values(ascending=False).to_dict())
    # label offset video -> Qualisys (both axes start at 0 here)
    qrel = qt - qt[0]
    G = np.arange(0, max(qrel[-1], lab.end.max()) + 1, 1 / T.HZ)

    def ind(mask):
        y = np.zeros(len(G))
        for s, e in lab.loc[mask, ["start", "end"]].to_numpy():
            y[(G >= s) & (G < e)] = 1
        return y

    tests = {"any label (all intervals)": (lab.name.notna(), ["Torse", "Main_Dr", "Main_Gc", "Genou_Dr", "Genou_Gc"])}
    # protocol repeats each movement, so a free search per half is ambiguous; test offset 0 per half instead
    mk5 = ["Torse", "Main_Dr", "Main_Gc", "Genou_Dr", "Genou_Gc"]
    sp0 = np.nanmean(np.vstack([T.regular(spd[m][0] - qt[0], spd[m][1], T.HZ, G[0], G[-1] + 1e-9)[1] for m in mk5]), axis=0)
    y0 = ind(lab.name.notna())
    mid = lab.start.median()
    for hname, sel in (("first half", G < mid), ("second half", G >= mid)):
        print(f"  any label, {hname}: AUC at offset 0 = {T.auc(sp0[sel], y0[sel] > 0.5):.2f}")
    for tname, (mask, mks) in tests.items():
        if mask.sum() < 2:
            print(f"  {tname}: too few intervals"); continue
        sp = np.nanmean(np.vstack([T.regular(spd[m][0] - qt[0], spd[m][1], T.HZ, G[0], G[-1] + 1e-9)[1] for m in mks]), axis=0)
        y = ind(mask)
        S = T.summarize(*T.xcorr(sp, y, 900, min_overlap_s=60))
        kk = int(round(S["lag"] * T.HZ))
        print(f"  {tname}: {int(mask.sum())} intervals -> label time = Qualisys time {S['lag']:+.1f} s "
              f"(r {S['r']:.2f}, other peak {S['r_far']:.2f}); AUC at that offset {T.auc(sp, np.roll(y, -kk) > 0.5):.2f}")
