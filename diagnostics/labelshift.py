"""Label families of a 2022 recording, optionally shifted by the approved per-participant offset.

Set the environment variable EMOB_LABEL_SHIFT=1 to apply the shifts decided on 2026-11-13
(reports/data/2026-11-13_label_timing.json: body-signed estimate, applied only when its whole 90 %
interval is more than 0.1 s from zero). Without the variable the labels are returned as in the CSV.
The raw files are never modified.
"""
from __future__ import annotations

import json
import os

import numpy as np

from emob import config as C

FS = 50
ON = os.environ.get("EMOB_LABEL_SHIFT", "0") == "1"
_JSON = C.PROJECT / "reports" / "data" / "2026-11-13_label_timing.json"
_T = json.load(open(_JSON, encoding="utf-8")) if ON else None


def shift_s(user: str) -> float:
    """Shift (s) to apply to the labels of this participant; 0.0 when shifting is off."""
    if not ON:
        return 0.0
    r = next(x for x in _T["estimates"][user] if x["estimator"].startswith("signed: 10 named"))
    return float(r["d"]) if (r["lo"] > 0.1 or r["hi"] < -0.1) else 0.0


def shifted(a: np.ndarray, k: int, fill="") -> np.ndarray:
    """Move the content of a by k rows (k > 0: later)."""
    out = np.full(len(a), fill, dtype=a.dtype)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def families(df, user: str) -> dict:
    k = int(round(shift_s(user) * FS))
    return {c: shifted(df[c].astype("string").fillna("").to_numpy(object), k) for c in C.FAMILY_COLS}


def banner() -> str:
    return "LABELS SHIFTED by the approved per-participant offsets (EMOB_LABEL_SHIFT=1)" if ON else "Labels as in the CSV (no shift)"
