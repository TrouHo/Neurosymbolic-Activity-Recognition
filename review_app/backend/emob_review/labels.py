"""Label segments built directly from per-row label strings.

* A segment is a maximal run of consecutive rows with the same raw string.
* Sample i covers [ts_i, ts_{i+1}); the last sample's end uses the median step and is flagged.
* Segments are split at timestamp anomalies (duplicate, reversed or gap step) and flagged.
* The empty string is kept as its own value ("" -> shown as "(empty)"); it is never mapped to rest.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .csvload import LoadedCsv


@dataclass
class Segment:
    start_row: int
    end_row: int          # exclusive
    value: str
    start_ts: float
    end_ts: float
    end_extrapolated: bool
    split_at_anomaly: bool


def _row_end_ts(ts, i: int, med: float) -> tuple[float, bool]:
    """End timestamp of sample i (exclusive)."""
    if i + 1 < len(ts):
        return ts[i + 1], False
    return ts[i] + med, True


def _is_anomaly(ts, i: int, med: float, gap_factor: float) -> bool:
    """True when the step from row i-1 to row i is not a regular forward step."""
    a, b = ts[i - 1], ts[i]
    if math.isnan(a) or math.isnan(b):
        return True
    d = b - a
    return d <= 0 or d > gap_factor * med


def segments_for(values: list[str], ts, med: float, gap_factor: float = 1.5) -> list[Segment]:
    out: list[Segment] = []
    n = len(values)
    if n == 0:
        return out
    start = 0
    split_flag = False
    for i in range(1, n + 1):
        anomaly = i < n and _is_anomaly(ts, i, med, gap_factor)
        boundary = i == n or values[i] != values[start]
        if boundary or anomaly:
            if anomaly:  # never let a segment bridge a timestamp anomaly
                end_ts, extrap = ts[i - 1] + med, True
            else:
                end_ts, extrap = _row_end_ts(ts, i - 1, med)
            forced_split = anomaly and not boundary
            out.append(Segment(start, i, values[start], ts[start], end_ts, extrap, split_flag or forced_split))
            split_flag = forced_split
            start = i
    return out


def summarize(segs: list[Segment]) -> list[dict]:
    stats: dict[str, dict] = {}
    for s in segs:
        d = stats.setdefault(s.value, {"value": s.value, "rows": 0, "segments": 0, "duration_ms": 0.0})
        d["rows"] += s.end_row - s.start_row
        d["segments"] += 1
        d["duration_ms"] += s.end_ts - s.start_ts
    return sorted(stats.values(), key=lambda d: -d["rows"])


def row_runs(mask: bytes | bytearray) -> list[tuple[int, int]]:
    """[start, end) runs where mask is non-zero (one byte per row)."""
    return [(m.start(), m.end()) for m in re.finditer(rb"[^\x00]+", bytes(mask))]


def label_overview(lc: LoadedCsv, ts, med: float, family_cols: list[int], derived_cols: list[int],
                   neutral_values: tuple[str, ...] = ("", "inactif")) -> dict:
    """Segments for each label column plus overlap / unlabeled intervals across family columns.

    overlap   = rows where >= 2 family columns hold a value not in `neutral_values`
    unlabeled = rows where every family column is the empty string
    The definitions are reported with the result so the UI can display them.
    """
    cols = {}
    for c in [*family_cols, *derived_cols]:
        vals = lc.text.get(c)
        if vals is None:  # numeric/empty column chosen as label: show raw numbers as strings
            vals = ["" if math.isnan(v) else repr(v) for v in lc.numeric[c]]
        segs = segments_for(vals, ts, med)
        cols[c] = {"column": c, "name": lc.header[c], "role": "family" if c in family_cols else "derived",
                   "segments": [s.__dict__ for s in segs], "summary": summarize(segs)}
    n = lc.n_rows
    overlap_mask = bytearray(n)
    unlabeled_mask = bytearray(n)
    if family_cols:
        fam_vals = [lc.text.get(c) or [""] * n for c in family_cols]
        for r in range(n):
            active = sum(1 for v in fam_vals if v[r] not in neutral_values)
            overlap_mask[r] = active >= 2
            unlabeled_mask[r] = all(v[r] == "" for v in fam_vals)

    def to_intervals(mask):
        out = []
        for a, b in row_runs(mask):
            if b < len(ts) and _is_anomaly(ts, b, med, 1.5):
                end_ts, extrap = ts[b - 1] + med, True
            else:
                end_ts, extrap = _row_end_ts(ts, b - 1, med)
            out.append({"start_row": a, "end_row": b, "start_ts": ts[a], "end_ts": end_ts,
                        "end_extrapolated": extrap})
        return out

    return {
        "columns": cols,
        "overlap": to_intervals(overlap_mask),
        "unlabeled": to_intervals(unlabeled_mask),
        "definitions": {
            "overlap": f">= 2 family columns hold a value other than {list(neutral_values)!r} on the same row",
            "unlabeled": "every family column is the empty string on that row (not interpreted as rest)",
            "sample_interval": "sample i covers [ts_i, ts_(i+1)); last sample end = ts_last + median step",
        },
        "family_columns": family_cols,
        "derived_columns": derived_cols,
    }
