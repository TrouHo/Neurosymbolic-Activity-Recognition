"""S2 — sensor time alignment (pipeline v2).

Applies the frozen offsets of params/sensor_offsets.json (estimated by estimate/sensor_offsets.py,
method and decisions: reports/data/2026-11-17_sensor_alignment_and_long_runs.md).

- One offset per device (phone ACC and GYR share the phone's clock).
- aligned[t] = raw[t + k], k = offset in samples (k > 0: device was late). Rows left without a
  sample after the shift are unavailable; nothing is interpolated.
- A device that failed the gate is used only inside its verified windows (window-level fallback);
  a device without signal, or whose fallback failed, is unavailable everywhere.
"""
from __future__ import annotations

import json

import numpy as np

from emob import config as C


def load_offsets() -> dict:
    return json.load(open(C.PARAMS / "sensor_offsets.json", encoding="utf-8"))


def shift_rows(a: np.ndarray, k: int, fill) -> np.ndarray:
    """out[t] = a[t + k] (k > 0 moves the content earlier); `fill` where no source row exists."""
    out = np.full_like(a, fill)
    n = len(a)
    if k > 0:
        out[:n - k] = a[k:]
    elif k < 0:
        out[-k:] = a[:n + k]
    else:
        out[:] = a
    return out


def device_plan(offsets: dict, user: str, device: str) -> dict:
    """{'usable': bool, 'shift_s': float, 'valid_rows': list | None, 'status': str} for one device."""
    d = offsets["participants"][user]["devices"][device]
    if not d.get("signal", False):
        return dict(usable=False, shift_s=0.0, valid_rows=None, status="no signal")
    if d.get("gate_passed"):
        return dict(usable=True, shift_s=float(d["apply_shift_s"]), valid_rows=None, status="gate passed")
    fb = d.get("fallback") or {}
    if fb.get("usable"):
        return dict(usable=True, shift_s=float(fb["apply_shift_s"]), valid_rows=fb["valid_rows"],
                    status=f"gate failed; fallback: {fb['n_verified']}/{fb['n_windows_with_data']} windows verified")
    return dict(usable=False, shift_s=0.0, valid_rows=None, status="gate failed; fallback failed")


def align_block(xyz: np.ndarray, ok: np.ndarray, plan: dict) -> tuple[np.ndarray, np.ndarray]:
    """Shift one sensor block; return (aligned xyz with NaN where unavailable, availability)."""
    n = len(xyz)
    if not plan["usable"]:
        return np.full_like(xyz, np.nan), np.zeros(n, bool)
    k = int(round(plan["shift_s"] * C.FS))
    x = shift_rows(xyz, k, np.nan)
    a = shift_rows(ok, k, False)
    if plan["valid_rows"] is not None:
        keep = np.zeros(n, bool)
        for s, e in plan["valid_rows"]:
            keep[s:e] = True
        a &= keep
    x[~a] = np.nan
    return x, a
