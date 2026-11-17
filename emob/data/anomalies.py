"""S4 — mask the anomalous parts of the six long label runs (pipeline v2).

Reads params/long_runs.json (estimate/long_runs.py; rule decided at the meeting of 2026-11-16 and
approved 2026-11-17; reports/data/2026-11-17_sensor_alignment_and_long_runs.md §4). For each run,
the rows `masked_rows` = [a, e) are, in the label time of the CSV (before alignment):
- the family column of the run: set to "" (not annotated) and flagged;
- `activite`: flagged (no label) where it equals the stuck label;
Posture after a masked body run is unknown until the next named transition (handled in labels.py).
"""
from __future__ import annotations

import json

import numpy as np

from emob import config as C


def load_runs() -> list[dict]:
    return json.load(open(C.PARAMS / "long_runs.json", encoding="utf-8"))["runs"]


def masks(runs: list[dict], user: str, n: int, activite: np.ndarray) -> tuple[dict, np.ndarray]:
    """Return ({family column: bool mask}, activite mask) in the label time of the CSV."""
    fam = {c: np.zeros(n, bool) for c in C.FAMILY_COLS}
    act = np.zeros(n, bool)
    for r in runs:
        if r["user"] != user:
            continue
        a, e = r["masked_rows"]
        fam[r["column"]][a:e] = True
        act[a:e] |= activite[a:e] == r["label"]
    return fam, act
