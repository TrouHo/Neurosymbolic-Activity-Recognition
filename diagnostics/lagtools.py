"""Shared helpers for the timing audits (read-only, 2026-11-09).

Principle (same as the 2022 audit): an inertial sensor and the Qualisys marker on the same body
part must become active at the same time. Activity of the sensor = 1-s moving std of |a|;
activity of the marker = 1-s mean speed (jumps > 3000 mm/s dropped). Both are put on one
10-Hz grid and compared with a rank (Spearman-type) correlation for every time shift.
Lag L > 0 means the second signal (the sensor) is late by L seconds.
"""
from __future__ import annotations

import struct
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

HZ = 10  # comparison grid


def regular(t, x, hz, t0, t1, max_gap=0.5):
    """Interpolate x(t) onto np.arange(t0, t1, 1/hz); NaN where no valid sample within max_gap s."""
    t = np.asarray(t, float); x = np.asarray(x, float)
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    o = np.argsort(t, kind="stable"); t, x = t[o], x[o]
    g = np.arange(t0, t1, 1.0 / hz)
    if len(t) < 2:
        return g, np.full(len(g), np.nan)
    y = np.interp(g, t, x, left=np.nan, right=np.nan)
    j = np.searchsorted(t, g).clip(1, len(t) - 1)
    d = np.minimum(np.abs(g - t[j - 1]), np.abs(t[j] - g))
    y[d > max_gap] = np.nan
    return g, y


def imu_activity(t, xyz, fs=50):
    """1-s moving std of |a| on a regular fs grid. Rows that are non-finite or all-zero are gaps."""
    xyz = np.asarray(xyz, float)
    m = np.linalg.norm(xyz, axis=1)
    m[~np.isfinite(xyz).all(1) | (xyz == 0).all(1)] = np.nan
    g, y = regular(t, m, fs, np.nanmin(t), np.nanmax(t))
    e = pd.Series(y).rolling(fs, center=True, min_periods=fs // 2).std().to_numpy()
    return g, e


def marker_speed(t, xyz, clip=3000.0):
    """1-s mean speed (mm/s) of a marker; all-zero / NaN positions are gaps, jumps > clip dropped."""
    p = np.asarray(xyz, float).copy()
    p[(p == 0).all(1)] = np.nan
    t = np.asarray(t, float)
    dt = np.diff(t)
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    v[(v > clip) | ~np.isfinite(v)] = np.nan
    fs = int(round(1 / np.median(dt)))
    v = pd.Series(v).rolling(fs, center=True, min_periods=fs // 2).mean().to_numpy()
    return (t[1:] + t[:-1]) / 2, v


def _prep(x):
    r = pd.Series(x).rank().to_numpy()
    m = np.isfinite(r)
    z = np.zeros(len(r))
    if m.sum() > 1 and r[m].std() > 0:
        z[m] = (r[m] - r[m].mean()) / r[m].std()
    return z, m.astype(float)


def xcorr(a, b, max_lag_s, hz=HZ, min_overlap_s=60):
    """r(L) ~ rank corr(a[t], b[t+L]) for L in [-max_lag_s, max_lag_s] (FFT). Returns lags (s), r."""
    za, ma = _prep(a); zb, mb = _prep(b)
    n = len(a) + len(b)
    nfft = 1 << (n - 1).bit_length()
    fa, fb = np.fft.rfft(za, nfft), np.fft.rfft(zb, nfft)
    num = np.fft.irfft(np.conj(fa) * fb, nfft)
    cnt = np.fft.irfft(np.conj(np.fft.rfft(ma, nfft)) * np.fft.rfft(mb, nfft), nfft)
    L = int(round(max_lag_s * hz))
    lags = np.arange(-L, L + 1)
    idx = lags % nfft
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num[idx] / cnt[idx]
    r[cnt[idx] < min_overlap_s * hz] = np.nan
    return lags / hz, r


def summarize(lags, r, far_s=5.0):
    """Best lag, best r, r at 0, and the best r more than far_s away from the peak (ambiguity)."""
    if np.all(~np.isfinite(r)):
        return {"lag": None, "r": None, "r0": None, "r_far": None}
    j = int(np.nanargmax(r))
    z = int(np.argmin(np.abs(lags)))
    far = np.abs(lags - lags[j]) > far_s
    return {"lag": float(lags[j]), "r": float(r[j]), "r0": float(r[z]) if np.isfinite(r[z]) else None,
            "r_far": float(np.nanmax(r[far])) if np.isfinite(r[far]).any() else None}


def windowed(a, b, centre_lag, win_s=120, step_s=60, search_s=10, hz=HZ):
    """Best lag per window of a (search centre_lag +- search_s). Returns list of (t_start, lag, r)."""
    out = []
    W, S = int(win_s * hz), int(step_s * hz)
    ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
    ks = np.arange(int((centre_lag - search_s) * hz), int((centre_lag + search_s) * hz) + 1)
    for s in range(0, len(a) - W + 1, S):
        best = (np.nan, -2.0)
        x = ra[s:s + W]
        for k in ks:
            if s + k < 0 or s + k + W > len(b):
                continue
            y = rb[s + k:s + k + W]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 0.5 * W or x[m].std() == 0 or y[m].std() == 0:
                continue
            c = np.corrcoef(x[m], y[m])[0, 1]
            if c > best[1]:
                best = (k / hz, c)
        if np.isfinite(best[0]):
            out.append((s / hz, float(best[0]), float(best[1])))
    return out


def auc(score, pos):
    score = np.asarray(score, float); pos = np.asarray(pos, bool)
    ok = np.isfinite(score)
    s, p = score[ok], pos[ok]
    n1, n0 = p.sum(), (~p).sum()
    if n1 < 10 or n0 < 10:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mp4_info(path):
    """Duration (s) and creation time (UTC) from the mp4 'mvhd' box; no external tools needed."""
    with open(path, "rb") as f:
        data_len = f.seek(0, 2); f.seek(0)

        def boxes(start, end):
            pos = start
            while pos + 8 <= end:
                f.seek(pos)
                size, kind = struct.unpack(">I4s", f.read(8))
                hdr = 8
                if size == 1:
                    size = struct.unpack(">Q", f.read(8))[0]; hdr = 16
                elif size == 0:
                    size = end - pos
                yield kind, pos + hdr, pos + size
                pos += size

        for kind, s, e in boxes(0, data_len):
            if kind == b"moov":
                for k2, s2, e2 in boxes(s, e):
                    if k2 == b"mvhd":
                        f.seek(s2)
                        ver = f.read(1)[0]; f.read(3)
                        if ver == 1:
                            c, m, ts, du = struct.unpack(">QQIQ", f.read(28))
                        else:
                            c, m, ts, du = struct.unpack(">IIII", f.read(16))
                        ct = datetime(1904, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=c)
                        return {"duration_s": du / ts, "creation_utc": ct.isoformat()}
    return None
