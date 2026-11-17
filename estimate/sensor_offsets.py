"""E1 — time offset of each inertial device against Qualisys (pipeline v2, plan 2026-11-17 §E1).

Why: labels and Qualisys markers share the CSV rows, so a device that is late against Qualisys is
also late against the labels. The 2026-11-09 audit found the phone 0.4-1.4 s late in 6/8 recordings
and the SB sensors of user3 about 14 s early, at a resolution of only 0.2 s (coarser than the 0.1-s
output step). This script re-estimates every offset at 0.02 s (one sample).

Method (acceleration against acceleration, 50 Hz):
- device: |a| minus its 1-s moving median (removes gravity), absolute value, 0.2-s moving mean;
- marker: position smoothed over 0.1 s, second derivative, magnitude, 0.2-s moving mean; samples
  next to a marker jump (speed > 3000 mm/s) are dropped;
- both rank-transformed (Spearman correlation, robust to marker spikes);
- coarse search +-20 s on a 10-Hz copy, then fine search +-0.5 s at 0.02 s on the whole recording;
- the same fine search on every 120-s window (+-1 s around the global value) -> median over windows,
  90 % interval by bootstrap over windows (1 000 resamples), drift = slope of window lag vs time
  multiplied by the recording length.
Sign: lag > 0 means the device is LATE; aligning means device_aligned[t] = device[t + lag].

Rule for applying (same rule as the label shift approved 2026-11-13): shift only when the whole
90 % interval lies outside +-0.1 s; the shift is rounded to whole samples.

Gate (decided before running, plan §E1):
  (a) global peak r >= 0.5; (b) 90 % interval width <= 0.2 s; (c) drift <= 0.1 s;
  (d) same sign and size as the 2026-11-09 audit (checked by reading, printed side by side).
A device that fails is reported, never fixed silently.

Fallback for a device that fails the gate (rule added 2026-11-17 AFTER the first run, when user3's
SB devices failed; approved by the user; both results are reported):
  the recording is cut into 60-s windows and the lag is searched in each (+-2 s around the global
  value). A window is VERIFIED when r >= 0.5 and its lag is within 0.1 s of the median lag of the
  verified windows (median recomputed until stable). The device is shifted by that median and is
  usable only inside verified windows (availability flag 0 elsewhere). If fewer than half of the
  windows that contain data are verified, the device is not used at all.
Same principle as the long label runs (meeting 2026-11-16): what can be verified is used, the rest
is dropped.

Output: params/sensor_offsets.json (+ the printed report, saved by the caller as evidence).
Usage (from code/):  .venv\\Scripts\\python.exe -m estimate.sensor_offsets
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from emob import config as C

FS = C.FS
WIN_S = 120
COARSE_S, FINE_S, WIN_SEARCH_S = 20.0, 0.5, 1.0
GATE_R, GATE_CI, GATE_DRIFT, APPLY_MIN = 0.5, 0.2, 0.1, 0.1
MIN_UNIQUE = C.MIN_UNIQUE
AUDIT_0911 = {  # lag reported by the 2026-11-09 audit (0.2-s resolution; +0.2 s method baseline)
    "phone": {"user1": 0.8, "user2": 1.4, "user3": 1.0, "user4": 1.2, "user5": 0.4, "user6": 0.0, "user7": 1.0, "user8": 0.0},
    "hand": {"user3": -14.6, "user5": 0.6, "user6": 0.0, "user7": -0.6, "user8": 0.0},
    "belt": {"user3": -14.2, "user6": 0.5, "user7": None, "user8": 0.0},
    "leg": {"user3": -13.6, "user5": 0.6, "user6": 0.0, "user7": -0.6, "user8": 0.0},
}
RNG = np.random.default_rng(20261117)


def roll(x, n, how="mean"):
    s = pd.Series(x).rolling(n, center=True, min_periods=max(1, n // 2))
    return getattr(s, how)().to_numpy()


def device_signal(xyz):
    xyz = np.asarray(xyz, float)
    bad = ~np.isfinite(xyz).all(1) | (xyz == 0).all(1)
    m = np.linalg.norm(xyz, axis=1)
    m[bad] = np.nan
    d = np.abs(m - roll(m, FS, "median"))
    return roll(d, 10)


def marker_signal(p):
    p = np.asarray(p, float).copy()
    p[(p == 0).all(1) | ~np.isfinite(p).all(1)] = np.nan
    ps = np.stack([roll(p[:, k], 5) for k in range(3)], 1)
    v = np.gradient(ps, axis=0) * FS
    sp = np.linalg.norm(np.gradient(p, axis=0) * FS, axis=1)
    a = np.linalg.norm(np.gradient(v, axis=0) * FS, axis=1)
    jump = roll((sp > 3000).astype(float), 25, "max") > 0      # +-0.25 s around a marker jump
    a[jump] = np.nan
    return roll(a, 10)


def rank(x):
    return pd.Series(x).rank().to_numpy()


def corr_at(x, y, k):
    """Pearson of x[t] and y[t+k] over finite pairs."""
    n = len(x)
    if k >= 0:
        a, b = x[:n - k], y[k:]
    else:
        a, b = x[-k:], y[:n + k]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 * FS:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())


def search(x, y, centre, half, step):
    lags = np.arange(int(round((centre - half) / step * 1)), int(round((centre + half) / step)) + 1)
    rs = np.array([corr_at(x, y, k)[0] for k in lags])
    if not np.isfinite(rs).any():
        return np.nan, np.nan
    i = int(np.nanargmax(rs))
    return lags[i] * step, rs[i]


def estimate(x, y):
    """x: marker signal, y: device signal (both 50 Hz, ranked)."""
    xc, yc = x[::5], y[::5]                                     # 10 Hz copy for the coarse search
    lag_c, r_c = search(xc, yc, 0.0, COARSE_S * 10, 1)          # in 10-Hz samples
    lag_c = lag_c / 10
    lag_f, r_f = search(x, y, lag_c * FS, FINE_S * FS, 1)
    lag_f = lag_f / FS
    wins = []
    n = len(x)
    for s in range(0, n - WIN_S * FS + 1, WIN_S * FS):
        xs, ys = np.full(n, np.nan), y
        xs[s:s + WIN_S * FS] = x[s:s + WIN_S * FS]
        lw, rw = search(xs, ys, lag_f * FS, WIN_SEARCH_S * FS, 1)
        if np.isfinite(lw) and rw >= 0.3:
            wins.append(((s + WIN_S * FS / 2) / FS, lw / FS, rw))
    return lag_c, r_c, lag_f, r_f, wins


FB_WIN_S, FB_SEARCH_S, FB_R, FB_TOL = 60, 2.0, 0.5, 0.1


def fallback(x, y, centre_s):
    """Window-level verification for a device that failed the gate (see module docstring)."""
    n = len(x)
    rows = []
    for s in range(0, n - FB_WIN_S * FS + 1, FB_WIN_S * FS):
        xs = np.full(n, np.nan)
        xs[s:s + FB_WIN_S * FS] = x[s:s + FB_WIN_S * FS]
        lag, r = search(xs, y, centre_s * FS, FB_SEARCH_S * FS, 1)
        rows.append(dict(start_row=s, end_row=s + FB_WIN_S * FS, lag_s=None if not np.isfinite(lag) else lag / FS,
                         r=None if not np.isfinite(r) else float(r)))
    with_data = [w for w in rows if w["lag_s"] is not None]
    cand = [w for w in with_data if w["r"] >= FB_R]
    med = float(np.median([w["lag_s"] for w in cand])) if cand else None
    for _ in range(10):
        ok = [w for w in cand if abs(w["lag_s"] - med) <= FB_TOL + 1e-9]
        new = float(np.median([w["lag_s"] for w in ok])) if ok else None
        if new == med:
            break
        med = new
    for w in rows:
        w["verified"] = bool(med is not None and w["lag_s"] is not None and w["r"] >= FB_R and abs(w["lag_s"] - med) <= FB_TOL + 1e-9)
    nv = sum(w["verified"] for w in rows)
    usable = nv >= len(with_data) / 2
    return dict(window_s=FB_WIN_S, median_lag_s=med, n_windows_with_data=len(with_data), n_verified=nv,
                usable=bool(usable), apply_shift_s=(round(med * FS) / FS if usable else None),
                valid_rows=[[w["start_row"], w["end_row"]] for w in rows if w["verified"]] if usable else [],
                windows=rows)


def summarise(wins, dur_s):
    if len(wins) < 3:
        return dict(n_windows=len(wins), median=None, lo=None, hi=None, drift=None)
    t, l, _ = map(np.array, zip(*wins))
    boot = [np.median(RNG.choice(l, len(l))) for _ in range(1000)]
    slope = np.polyfit(t, l, 1)[0]
    return dict(n_windows=len(wins), median=float(np.median(l)), lo=float(np.percentile(boot, 5)),
                hi=float(np.percentile(boot, 95)), drift=float(slope * dur_s),
                windows=[dict(t_s=round(a, 1), lag_s=round(b, 2), r=round(c, 3)) for a, b, c in wins])


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def main():
    out = {"created": datetime.now().isoformat(timespec="seconds"), "script": "estimate/sensor_offsets.py",
           "method": __doc__.split("Method")[1].split("Rule")[0].strip(),
           "rule": f"shift only if the whole 90 % interval is outside +-{APPLY_MIN} s; rounded to samples",
           "gate": dict(r_min=GATE_R, ci_width_max_s=GATE_CI, drift_max_s=GATE_DRIFT),
           "sign": "lag > 0: device late; aligned[t] = raw[t + lag]", "participants": {}}
    files = sorted(C.RAW_2022.glob("user*-*.csv"))
    for f in files:
        user = f.name.split("-")[0]
        want = {C.BLOCKS[C.DEVICES[d][0]][0] for d in C.DEVICES}
        hdr = list(pd.read_csv(f, nrows=0).columns)
        use = set()
        for d, (blk, mk) in C.DEVICES.items():
            use |= set(C.BLOCKS[blk][1]) | {f"{a}_sq_{mk}" for a in "xyz"}
        df = pd.read_csv(f, usecols=lambda c: c in use, low_memory=False).apply(pd.to_numeric, errors="coerce")
        dur = len(df) / FS
        res = {"source": f.name, "sha256": sha(f), "duration_s": dur, "devices": {}}
        print(f"\n=== {user} ({dur:.0f} s) ===")
        for d, (blk, mk) in C.DEVICES.items():
            xyz = df[C.BLOCKS[blk][1]].to_numpy(float)
            ok = np.isfinite(xyz).all(1) & ~(xyz == 0).all(1)
            nu = max((len(np.unique(xyz[ok, k])) for k in range(3)), default=0) if ok.any() else 0
            if nu < MIN_UNIQUE:
                res["devices"][d] = dict(signal=False, reason=f"no signal ({nu} distinct values per axis)", apply_shift_s=None)
                print(f"  {d:6s} no signal ({nu} distinct values) -> not usable")
                continue
            x = rank(marker_signal(df[[f"{a}_sq_{mk}" for a in "xyz"]].to_numpy(float)))
            y = rank(device_signal(xyz))
            lag_c, r_c, lag_f, r_f, wins = estimate(x, y)
            s = summarise(wins, dur)
            gate = dict(r=bool(r_f >= GATE_R),
                        ci=bool(s["lo"] is not None and s["hi"] - s["lo"] <= GATE_CI),
                        drift=bool(s["drift"] is not None and abs(s["drift"]) <= GATE_DRIFT))
            passed = all(gate.values())
            shift = 0.0
            if passed and (s["lo"] > APPLY_MIN or s["hi"] < -APPLY_MIN):
                shift = round(s["median"] * FS) / FS
            res["devices"][d] = dict(signal=True, marker=mk, coarse_lag_s=lag_c, coarse_r=r_c, global_lag_s=lag_f,
                                     global_r=r_f, **s, gate=gate, gate_passed=passed,
                                     apply_shift_s=shift if passed else None, audit_2026_11_09=AUDIT_0911[d].get(user))
            ci = f"[{s['lo']:+.2f}, {s['hi']:+.2f}]" if s["lo"] is not None else "n/a"
            dr = f"{s['drift']:+.2f}" if s["drift"] is not None else "n/a"
            med = f"{s['median']:+.2f}" if s["median"] is not None else "n/a"
            print(f"  {d:6s} coarse {lag_c:+6.2f} s (r {r_c:.2f}) | global {lag_f:+6.2f} s (r {r_f:.2f}) | windows {s['n_windows']:2d} "
                  f"median {med} 90% {ci} drift {dr} | audit 09/11 {AUDIT_0911[d].get(user)} | "
                  f"gate {'PASS' if passed else 'FAIL ' + str([k for k, v in gate.items() if not v])} | shift {shift if passed else '-'}")
            if s.get("windows"):
                print("         windows (t: lag/r): " + "  ".join(f"{w['t_s']:.0f}:{w['lag_s']:+.2f}/{w['r']:.2f}" for w in s["windows"]))
            if not passed:
                fb = fallback(x, y, lag_f)
                res["devices"][d]["fallback"] = fb
                res["devices"][d]["apply_shift_s"] = fb["apply_shift_s"]
                res["devices"][d]["valid_rows"] = fb["valid_rows"]
                kept = sum(b - a for a, b in fb["valid_rows"]) / FS
                print(f"         FALLBACK (60-s windows): median {fb['median_lag_s']}, verified {fb['n_verified']}/{fb['n_windows_with_data']} "
                      f"-> {'USABLE in verified windows, ' + f'{kept:.0f} s' if fb['usable'] else 'NOT USABLE'}; shift {fb['apply_shift_s']}")
                print("         " + "  ".join(
                    f"{w['start_row'] / FS:.0f}-{w['end_row'] / FS:.0f}:" + ("no data" if w["lag_s"] is None else
                                                                             f"{w['lag_s']:+.2f}/{w['r']:.2f}{'*' if w['verified'] else ''}")
                    for w in fb["windows"]))
        out["participants"][user] = res
    C.PARAMS.mkdir(exist_ok=True)
    p = C.PARAMS / "sensor_offsets.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten {p}")


if __name__ == "__main__":
    main()
