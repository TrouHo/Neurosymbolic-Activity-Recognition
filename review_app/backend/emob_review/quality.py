"""Data-quality candidates for review. Nothing here modifies data; zeros are reported as
"suspicious candidates", never as missing.

Per-row masks are built with bytes.translate and combined as big integers (one byte per row,
value 0/1), which keeps whole-recording scans fast without numpy.
"""
from __future__ import annotations

from .csvload import CLASS_NAMES, EMPTY, INF, NANSTR, NUM, TEXT, ZERO, LoadedCsv
from .labels import row_runs


def _table(classes: set[int]) -> bytes:
    return bytes(1 if i in classes else 0 for i in range(256))


_T_NUM = _table({NUM})
_T_ZERO = _table({ZERO})
_T_MISSING = _table({EMPTY, NANSTR, INF, TEXT})


def _mask(lc: LoadedCsv, cols: list[int], table: bytes, combine: str) -> int:
    acc = None
    for c in cols:
        m = int.from_bytes(bytes(lc.cls[c]).translate(table), "big")
        acc = m if acc is None else (acc | m if combine == "any" else acc & m)
    return acc or 0


def _to_bytes(mask: int, n: int) -> bytes:
    return mask.to_bytes(n, "big") if n else b""


def column_counts(lc: LoadedCsv, c: int) -> dict:
    k = lc.cls[c]
    return {name: k.count(i) for i, name in enumerate(CLASS_NAMES)}


def group_flags(lc: LoadedCsv, axes: dict[str, int]) -> dict:
    """Per-group summary used for availability badges."""
    cols = list(axes.values())
    nonzero = _mask(lc, cols, _T_NUM, "any").bit_count()
    all_zero = _mask(lc, cols, _T_ZERO, "all").bit_count()
    missing = _mask(lc, cols, _T_MISSING, "any").bit_count()
    return {
        "rows": lc.n_rows,
        "rows_with_nonzero_number": nonzero,
        "rows_all_axes_zero": all_zero,
        "rows_with_missing_or_nonnumeric": missing,
        "no_nonzero_data": nonzero == 0,
    }


def _runs_as_dicts(runs, ts, min_len: int, limit: int):
    kept = [(a, b) for a, b in runs if b - a >= min_len]
    return {"count": len(kept), "rows": sum(b - a for a, b in kept),
            "runs": [{"start_row": a, "end_row": b, "start_ts": ts[a], "last_ts": ts[b - 1]}
                     for a, b in kept[:limit]]}


def group_runs(lc: LoadedCsv, axes: dict[str, int], ts, min_len: int = 1, limit: int = 500) -> dict:
    cols = list(axes.values())
    n = lc.n_rows
    zero_mask = _to_bytes(_mask(lc, cols, _T_ZERO, "all"), n)
    missing_mask = _to_bytes(_mask(lc, cols, _T_MISSING, "any"), n)
    return {
        "all_axes_zero": _runs_as_dicts(row_runs(zero_mask), ts, min_len, limit),
        "missing_or_nonnumeric": _runs_as_dicts(row_runs(missing_mask), ts, min_len, limit),
    }


def constant_runs(lc: LoadedCsv, c: int, ts, min_len: int, limit: int = 500) -> dict:
    """Runs of identical non-zero numeric values (possible stuck sensor / heavy smoothing)."""
    v = lc.numeric[c]
    k = lc.cls[c]
    runs = []
    start = 0
    for i in range(1, lc.n_rows + 1):
        if i == lc.n_rows or not (k[i] == NUM and k[start] == NUM and v[i] == v[start]):
            if k[start] == NUM and i - start >= min_len:
                runs.append((start, i))
            start = i
    return _runs_as_dicts(runs, ts, min_len, limit)
