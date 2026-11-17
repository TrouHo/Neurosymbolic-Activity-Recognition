"""S3 — label time alignment (pipeline v2).

One shift per participant, the same for every label column, read from params/label_offsets.json
(frozen by estimate/label_offsets.py; decided 2026-11-13, confirmed 2026-11-17 as an alignment of
the labels with the signals, not a change of labels). label_aligned[t] = label_raw[t - d].
Rows left without a label after the shift are empty ("").
"""
from __future__ import annotations

import json

import numpy as np

from emob import config as C


def load_offsets() -> dict:
    return json.load(open(C.PARAMS / "label_offsets.json", encoding="utf-8"))


def shift_s(offsets: dict, user: str) -> float:
    return float(offsets["participants"][user]["shift_s"])


def shift_labels(a: np.ndarray, d_s: float, fill="") -> np.ndarray:
    """Move the content of `a` by d_s seconds (d_s < 0: earlier). Works for strings and booleans."""
    k = int(round(d_s * C.FS))
    out = np.full(len(a), fill, dtype=a.dtype)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out
