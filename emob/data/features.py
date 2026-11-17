"""Input representation (pipeline v2 -> model). Decision 2026-11-05 (orientation-invariant inputs),
layout 2026-11-17 (reports/design/2026-11-17_model_architecture.md §1.3).

Why invariant: each sensor is worn with a different orientation in each recording, and per-axis
inputs did not transfer between participants (chance-level LOSO, 2026-11-05).

Per device (one availability flag each; a device has one clock and one placement):
  ACC block -> 3 channels:
     |a|                      magnitude
     vertical dynamic acc.    (a - g) . g_hat, g = 1-s centred moving mean of a (gravity estimate)
     horizontal dynamic acc.  |(a - g) - vertical * g_hat|
  GYR block -> 1 channel: |w|
  phone = ACC (3) + GYR (1); hand, belt, leg = ACC (3)        -> 13 channels + 4 flags = 17
Representation "raw" (ablation): the 15 x/y/z channels + the same 4 flags = 19.

Unavailable samples are NaN here; normalisation sets them to 0 (normalize.py). Nothing is
interpolated. The phone flag follows the phone ACC; where only the phone GYR is missing (<= 2 % of
rows) its channel is 0 after normalisation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from emob import config as C

DEVICES = ["phone", "hand", "belt", "leg"]
DEVICE_BLOCKS = {"phone": ["phone_acc", "phone_gyr"], "hand": ["hand"], "belt": ["belt"], "leg": ["leg"]}
INV_CHANNELS = (["phone_amag", "phone_avert", "phone_ahor", "phone_wmag"]
                + [f"{d}_{k}" for d in ("hand", "belt", "leg") for k in ("amag", "avert", "ahor")])
RAW_CHANNELS = [f"{b}_{a}" for b in C.BLOCK_ORDER for a in "xyz"]
FLAG_CHANNELS = [f"{d}_available" for d in DEVICES]
GRAVITY_WIN = C.FS                     # 1 s


def _block(x50: np.ndarray, b: str) -> np.ndarray:
    j = C.BLOCK_ORDER.index(b)
    return x50[:, 3 * j:3 * j + 3].astype(np.float64)


def acc_invariant(a: np.ndarray) -> np.ndarray:
    """(N, 3) acceleration -> (N, 3): |a|, vertical and horizontal dynamic acceleration."""
    g = pd.DataFrame(a).rolling(GRAVITY_WIN, center=True, min_periods=GRAVITY_WIN // 2).mean().to_numpy()
    gn = np.linalg.norm(g, axis=1, keepdims=True)
    gh = np.divide(g, gn, out=np.full_like(g, np.nan), where=gn > 0)
    dyn = a - g
    vert = np.sum(dyn * gh, axis=1)
    hor = np.linalg.norm(dyn - vert[:, None] * gh, axis=1)
    return np.stack([np.linalg.norm(a, axis=1), vert, hor], axis=1)


def device_flags(avail50: np.ndarray) -> np.ndarray:
    """(N, 5) block availability -> (N, 4) device availability (phone follows its ACC)."""
    a = {b: avail50[:, i] for i, b in enumerate(C.BLOCK_ORDER)}
    return np.stack([a["phone_acc"], a["hand"], a["belt"], a["leg"]], axis=1)


def represent(x50: np.ndarray, avail50: np.ndarray, kind: str = "invariant") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (channels (N, K) with NaN where unavailable, device flags (N, 4), channel names).

    The device of each channel is given by `channel_device` for normalisation and dropout."""
    flags = device_flags(avail50)
    if kind == "raw":
        return x50.astype(np.float64), flags, RAW_CHANNELS
    if kind != "invariant":
        raise ValueError(kind)
    w = _block(x50, "phone_gyr")
    cols = [acc_invariant(_block(x50, "phone_acc")), np.linalg.norm(w, axis=1)[:, None]]
    cols += [acc_invariant(_block(x50, b)) for b in ("hand", "belt", "leg")]
    return np.concatenate(cols, axis=1), flags, INV_CHANNELS


def channel_device(names: list[str]) -> np.ndarray:
    """Index in DEVICES of the device that each channel belongs to."""
    return np.array([DEVICES.index(n.split("_")[0]) for n in names])
