"""Motion-capture measures used to check the labels. Same definitions as the diagnostics that first
produced the numbers (code/diagnostics/check_empty_rows.py, check_bent.py, check_label_errors.py);
moved here so that the reported values have one source.

Markers: Qualisys 3-D positions (mm) on the 50-Hz grid of the CSV; a row with all-zero or
non-finite coordinates is missing.
  speed(k)        3-D speed of marker k: positions smoothed over 5 rows (0.1 s), central gradient,
                  values > 3000 mm/s dropped (marker swaps), averaged over 25 rows (0.5 s).
  fastest speed   maximum over both hands, both feet, head and torso (each group = max of its two).
  trunk angle     angle (deg) between the vertical and the vector mid-hip -> mid-shoulder; rows whose
                  trunk length is outside 0.6-1.4 x its median are dropped; median over 12 rows.
Label versions: "raw" = the CSV; "aligned" = shifted by the per-participant annotation offsets of
params/label_offsets.json, exactly as in the build (no masking).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from emob import config as C
from emob.data import alignment

FS = C.FS
GROUPS = {"hands": ("D_main", "G_main"), "feet": ("D_pied", "G_pied"), "head/torso": ("Tete", "Torse")}


@lru_cache(maxsize=None)
def markers(path: str) -> pd.DataFrame:
    return pd.read_csv(path, usecols=lambda c: c.startswith(("x_sq_", "y_sq_", "z_sq_")), low_memory=False).apply(pd.to_numeric, errors="coerce")


def xyz(mk: pd.DataFrame, k: str) -> np.ndarray:
    p = mk[[f"x_sq_{k}", f"y_sq_{k}", f"z_sq_{k}"]].to_numpy(float).copy()
    p[(p == 0).all(1) | ~np.isfinite(p).all(1)] = np.nan
    return p


def speed(mk: pd.DataFrame, k: str) -> np.ndarray:
    p = xyz(mk, k)
    q = np.stack([pd.Series(p[:, a]).rolling(5, center=True, min_periods=2).mean().to_numpy() for a in range(3)], 1)
    v = np.linalg.norm(np.gradient(q, axis=0) * FS, axis=1)
    v[v > 3000] = np.nan
    return pd.Series(v).rolling(FS // 2, center=True, min_periods=FS // 4).mean().to_numpy()


def fastest_speed(mk: pd.DataFrame) -> np.ndarray:
    allv = np.stack([np.fmax(speed(mk, a), speed(mk, b)) for a, b in GROUPS.values()])
    vmax = np.nanmax(np.where(np.isfinite(allv), allv, -1), axis=0)
    vmax[vmax < 0] = np.nan
    return vmax


def trunk_angle(mk: pd.DataFrame) -> np.ndarray:
    sh = np.nanmean(np.stack([xyz(mk, "D_epaule"), xyz(mk, "G_epaule")]), axis=0)
    hp = np.nanmean(np.stack([xyz(mk, "D_hanche"), xyz(mk, "G_hanche")]), axis=0)
    v = sh - hp
    L = np.linalg.norm(v, axis=1)
    ang = np.degrees(np.arccos(np.clip(v[:, 2] / L, -1, 1)))
    Lm = np.nanmedian(L)
    ang[(L < 0.6 * Lm) | (L > 1.4 * Lm)] = np.nan
    # check_bent.py used min_periods = 6, check_label_errors.py 3; one definition is kept here (6)
    return pd.Series(ang).rolling(FS // 4, center=True, min_periods=6).median().to_numpy()


def hip_height(mk: pd.DataFrame) -> np.ndarray:
    hp = np.nanmean(np.stack([xyz(mk, "D_hanche"), xyz(mk, "G_hanche")]), axis=0)
    return pd.Series(hp[:, 2]).rolling(5, center=True, min_periods=2).mean().to_numpy()


def families(raw: dict, lab_off: dict, user: str, aligned: bool) -> dict:
    d = alignment.shift_s(lab_off, user) if aligned else 0.0
    return {c: alignment.shift_labels(raw["fam"][c], d) for c in C.FAMILY_COLS}
