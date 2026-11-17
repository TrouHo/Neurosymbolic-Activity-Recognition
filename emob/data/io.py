"""Read one 2022 recording into numpy arrays (no cleaning, no interpolation)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from emob import config as C


@dataclass
class Recording:
    participant: str               # e.g. "user1"
    source: str                    # CSV file name
    sha256: str                    # hash of the CSV (traceability)
    ts: np.ndarray                 # (N,) ts_SQ
    acc: np.ndarray                # (N, 6) float64, hand xyz then belt xyz, raw values (NaN kept)
    usable: np.ndarray             # (N, 2) bool, per sensor: finite, not all-zero, block not degenerate
    families: dict[str, np.ndarray]  # family column -> (N,) object array, "" = empty
    categorie: np.ndarray          # (N,) object
    activite: np.ndarray           # (N,) object
    extra: np.ndarray | None = None          # (N, 6) phone ACC xyz, phone GYR xyz
    extra_usable: np.ndarray | None = None   # (N, 2)
    degenerate: dict[str, bool] = field(default_factory=dict)   # sensor -> block has no signal
    n_unique: dict[str, int] = field(default_factory=dict)      # sensor -> max distinct values per axis

    @property
    def n(self) -> int:
        return len(self.ts)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _str_col(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].astype("string").fillna("").to_numpy(dtype=object)


def usable_rows(xyz: np.ndarray) -> np.ndarray:
    """A sensor row is usable if all three axes are finite and not all exactly zero."""
    finite = np.isfinite(xyz).all(axis=1)
    allzero = (xyz == 0).all(axis=1)
    return finite & ~allzero


def _read_block(df: pd.DataFrame, name: str, ts_col: str, xyz_cols: list[str]):
    i = list(df.columns).index(ts_col)
    if list(df.columns[i + 1:i + 4]) != xyz_cols:
        raise ValueError(f"unexpected columns after {ts_col}: {list(df.columns[i+1:i+4])}")
    xyz = df[xyz_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = usable_rows(xyz)
    n_unique = max((len(np.unique(xyz[ok, k])) for k in range(3)), default=0) if ok.any() else 0
    degenerate = n_unique < C.MIN_UNIQUE
    if degenerate:
        ok = np.zeros_like(ok)
    return xyz, ok, degenerate, n_unique


def load_recording(path: str | Path) -> Recording:
    path = Path(path)
    df = pd.read_csv(path, sep=None, engine="python")
    ts = pd.to_numeric(df["ts_SQ"], errors="coerce").to_numpy(float)
    if not np.all(np.diff(ts) == C.TS_STEP):
        raise ValueError(f"{path.name}: ts_SQ is not a regular grid of step {C.TS_STEP}")

    blocks, usable, extra, extra_usable, degen, nuniq = [], [], [], [], {}, {}
    for key in C.SENSOR_ORDER:
        xyz, ok, d, nu = _read_block(df, key, *C.SENSORS[key])
        blocks.append(xyz); usable.append(ok); degen[key] = d; nuniq[key] = nu
    for key in C.EXTRA_ORDER:
        xyz, ok, d, nu = _read_block(df, key, *C.EXTRA_SENSORS[key])
        extra.append(xyz); extra_usable.append(ok); degen[key] = d; nuniq[key] = nu

    return Recording(
        participant=path.name.split("-")[0],
        source=path.name,
        sha256=_sha256(path),
        ts=ts,
        acc=np.concatenate(blocks, axis=1),
        usable=np.stack(usable, axis=1),
        families={c: _str_col(df, c) for c in C.FAMILY_COLS},
        categorie=_str_col(df, C.CAT_COL),
        activite=_str_col(df, C.ACT_COL),
        extra=np.concatenate(extra, axis=1),
        extra_usable=np.stack(extra_usable, axis=1),
        degenerate=degen,
        n_unique=nuniq,
    )
