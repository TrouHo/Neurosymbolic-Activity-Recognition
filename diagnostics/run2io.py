"""Readers for run2 (Session 4) raw files, shared by the audit scripts (read-only)."""
import numpy as np
import pandas as pd

RAW = r"D:\Projects\EmoB\data\raw\run2"
TICK = 40000.0   # ticks per second as stated in the log ('24576 ticks (614 ms)')
CAND = {"poignet": ["G poignet", "D poignet"], "ceinture": ["Torse", "G hanche", "D hanche"]}


def read_log(path):
    """Rows of a decoded sensor log. Returns kinds, clock values, block index of each sample,
    raw X/Y/Z (as supplied, unsigned), supplied scaled X/Y/Z, malformed lines, byte/value mismatches."""
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


def sample_times(ts, blk, n):
    """Time (s, device clock at TICK) of each sample, uniformly spaced inside its clock block."""
    per = np.bincount(blk[blk >= 0], minlength=len(ts))
    step = np.median(np.diff(ts)) / np.median(per[per > 0])
    t = np.full(n, np.nan)
    for b in range(len(ts)):
        i = np.flatnonzero(blk == b)
        if len(i) == 0:
            continue
        span = (ts[b + 1] - ts[b]) if b + 1 < len(ts) else step * len(i)
        t[i] = (ts[b] + span * np.arange(len(i)) / len(i)) / TICK
    return t, per


def signed(raw):
    return np.where(raw >= 32768, raw - 65536, raw)


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
