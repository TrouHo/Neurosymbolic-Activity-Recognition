"""Read-only CSV inspection.

Every cell is classified (number / exact zero / empty / NaN-like string / infinite / text) and the
original line text is kept, so that exports can reproduce source rows byte-for-byte. Nothing is
replaced, interpolated, resampled or smoothed.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import statistics
from array import array
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

# Cell classes (stored per column in a bytearray)
NUM, ZERO, EMPTY, NANSTR, INF, TEXT = range(6)
CLASS_NAMES = ["number", "zero", "empty", "nan_string", "infinite", "text"]
_NAN_TOKENS = {"nan", "-nan", "na", "n/a", "null", "none"}

_GROUP_TS_RE = re.compile(r"^ts_(?P<source>[A-Za-z]+)\s*-\s*(?P<placement>.+?)\s*-\s*(?P<modality>[A-Za-z]+)\s*$")
_AXIS_RE = re.compile(r"^(?P<axis>[xyz])_(?P<rest>.+)$")
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


@dataclass
class ChannelGroup:
    id: str
    source: str            # as written in the header (e.g. "SB", "SF", "sq")
    placement: str         # original text (e.g. "Main droite", "D_main")
    modality: str          # original text (e.g. "ACC", "GYR") or "XYZ" for Qualisys markers
    header: str            # the header text the group was parsed from
    ts_col: int | None     # per-group timestamp column, if any
    axes: dict[str, int]   # axis -> column index
    provenance: str = "parsed from header text"


@dataclass
class LoadedCsv:
    path: Path
    size: int
    mtime: float
    sha256: str
    encoding: str
    delimiter: str
    decimal: str
    header: list[str]
    lines: list[str]                  # original data lines, without line terminator
    numeric: list[array]              # float64 per column (NaN where not a number)
    cls: list[bytearray]              # cell class per column
    text: dict[int, list[str]]        # raw strings for text-like columns
    bad_rows: list[int]               # rows whose field count differs from the header
    lines_match_records: bool         # physical lines == CSV records (needed for verbatim export)
    groups: list[ChannelGroup] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.cls[0]) if self.cls else 0

    def column_kind(self, i: int) -> str:
        c = self.cls[i]
        n = len(c)
        n_text = c.count(TEXT)
        if n and c.count(EMPTY) == n:
            return "empty"
        if n and n_text and n_text + c.count(EMPTY) == n:
            return "text"
        return "numeric" if n_text == 0 else "mixed"


def classify(s: str, decimal: str = ".") -> tuple[int, float]:
    t = s.strip()
    if not t:
        return EMPTY, math.nan
    if t.lower() in _NAN_TOKENS:
        return NANSTR, math.nan
    if decimal != ".":
        t = t.replace(decimal, ".")
    try:
        v = float(t)
    except ValueError:
        return TEXT, math.nan
    if math.isnan(v):
        return NANSTR, math.nan
    if math.isinf(v):
        return INF, v
    return (ZERO if v == 0.0 else NUM), v


def decode_bytes(data: bytes, encoding: str | None = None) -> tuple[str, str]:
    for enc in ([encoding] if encoding else ENCODINGS):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode file with " + ", ".join(ENCODINGS))


def sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def parse_groups(header: list[str]) -> list[ChannelGroup]:
    """Group columns using only the structure visible in the header.

    * "ts_<SRC> - <placement> - <MOD>" followed by x_/y_/z_ columns  -> one IMU-style group
    * "x_<rest>", "y_<rest>", "z_<rest>" with an identical <rest>    -> one XYZ group (e.g. sq markers)
    """
    groups: list[ChannelGroup] = []
    used: set[int] = set()
    i = 0
    while i < len(header):
        m = _GROUP_TS_RE.match(header[i])
        if m:
            axes = {}
            j = i + 1
            while j < len(header) and len(axes) < 3:
                am = _AXIS_RE.match(header[j])
                if not am or am.group("axis") in axes:
                    break
                axes[am.group("axis")] = j
                j += 1
            if axes:
                gid = f"g{len(groups)}"
                groups.append(ChannelGroup(gid, m.group("source"), m.group("placement"), m.group("modality"),
                                           header[i], i, axes))
                used.update([i, *axes.values()])
                i = j
                continue
        i += 1
    # XYZ triplets sharing a suffix (Qualisys markers)
    by_rest: dict[str, dict[str, int]] = {}
    for idx, h in enumerate(header):
        if idx in used:
            continue
        am = _AXIS_RE.match(h)
        if am:
            by_rest.setdefault(am.group("rest"), {}).setdefault(am.group("axis"), idx)
    for rest, axes in by_rest.items():
        if len(axes) < 2:
            continue
        parts = rest.split("_", 1)
        source, placement = (parts[0], parts[1]) if len(parts) == 2 else ("", rest)
        gid = f"g{len(groups)}"
        groups.append(ChannelGroup(gid, source, placement, "XYZ", rest, None, axes,
                                   provenance="parsed from header text (x_/y_/z_ suffix)"))
    return groups


def load_csv(path: Path, encoding: str | None = None, delimiter: str | None = None,
             decimal: str = ".") -> LoadedCsv:
    st = path.stat()
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    text, enc = decode_bytes(data, encoding)
    del data
    delim = delimiter or sniff_delimiter(text[:65536])

    raw_lines = text.split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()
    raw_lines = [ln[:-1] if ln.endswith("\r") else ln for ln in raw_lines]

    reader = csv.reader(io.StringIO(text), delimiter=delim)
    header = next(reader)
    ncol = len(header)
    numeric = [array("d") for _ in range(ncol)]
    cls = [bytearray() for _ in range(ncol)]
    bad_rows: list[int] = []
    n_records = 1
    r = 0
    for rec in reader:
        n_records += 1
        if not rec:  # blank line; only tolerated at the end of the file
            continue
        if len(rec) != ncol:
            bad_rows.append(r)
        for c in range(ncol):
            k, v = classify(rec[c] if c < len(rec) else "", decimal)
            cls[c].append(k)
            numeric[c].append(v)
        r += 1
    n_rows = r
    lines_match = len(raw_lines) == n_records and len(raw_lines) - 1 == n_rows

    # Second pass: keep raw strings only for columns that contain text (labels etc.)
    text_cols = [c for c in range(ncol) if cls[c].count(TEXT)]
    texts: dict[int, list[str]] = {c: [] for c in text_cols}
    if text_cols:
        reader = csv.reader(io.StringIO(text), delimiter=delim)
        next(reader)
        for rec in reader:
            if not rec:
                continue
            for c in text_cols:
                texts[c].append(rec[c] if c < len(rec) else "")
    del text

    out = LoadedCsv(path=path, size=st.st_size, mtime=st.st_mtime, sha256=sha, encoding=enc, delimiter=delim,
                    decimal=decimal, header=header, lines=raw_lines[1:] if lines_match else [],
                    numeric=numeric, cls=cls, text=texts, bad_rows=bad_rows, lines_match_records=lines_match)
    out.groups = parse_groups(header)
    return out


# ---------------------------------------------------------------- timestamps

UNIT_TO_MS = {"ms": 1.0, "s": 1000.0, "us": 1e-3, "ns": 1e-6}


@dataclass
class TimestampReport:
    column: int
    n: int
    first: float | None
    last: float | None
    n_invalid: int
    invalid_rows: list[int]
    n_fractional: int
    n_duplicates: int
    duplicate_rows: list[int]
    n_nonmonotonic: int
    nonmonotonic_rows: list[int]
    median_step: float | None
    step_histogram: list[tuple[float, int]]
    gaps: list[dict]
    effective_rate_hz: float | None


def timestamp_report(lc: LoadedCsv, col: int, unit: str = "ms", gap_factor: float = 1.5,
                     max_list: int = 200) -> TimestampReport:
    vals = lc.numeric[col]
    k = lc.cls[col]
    n = len(vals)
    invalid = [i for i in range(n) if k[i] not in (NUM, ZERO)]
    fractional = 0
    # A timestamp written as "1643189820000.0" is integral; flag any with a non-zero fraction.
    for i in range(n):
        v = vals[i]
        if k[i] in (NUM, ZERO) and v != math.floor(v):
            fractional += 1
    steps: list[float] = []
    dup, nonmono = [], []
    for i in range(1, n):
        a, b = vals[i - 1], vals[i]
        if math.isnan(a) or math.isnan(b):
            continue
        d = b - a
        steps.append(d)
        if d == 0:
            dup.append(i)
        elif d < 0:
            nonmono.append(i)
    pos = [d for d in steps if d > 0]
    med = statistics.median(pos) if pos else None
    hist: dict[float, int] = {}
    for d in steps:
        hist[d] = hist.get(d, 0) + 1
    gaps = []
    if med:
        for i in range(1, n):
            a, b = vals[i - 1], vals[i]
            if not (math.isnan(a) or math.isnan(b)) and (b - a) > gap_factor * med:
                gaps.append({"row": i, "from": a, "to": b, "step": b - a})
    valid = [vals[i] for i in range(n) if k[i] in (NUM, ZERO)]
    first = valid[0] if valid else None
    last = valid[-1] if valid else None
    rate = None
    if first is not None and last is not None and last > first and len(valid) > 1:
        rate = (len(valid) - 1) / ((last - first) * UNIT_TO_MS[unit] / 1000.0)
    return TimestampReport(
        column=col, n=n, first=first, last=last, n_invalid=len(invalid), invalid_rows=invalid[:max_list],
        n_fractional=fractional, n_duplicates=len(dup), duplicate_rows=dup[:max_list],
        n_nonmonotonic=len(nonmono), nonmonotonic_rows=nonmono[:max_list], median_step=med,
        step_histogram=sorted(hist.items(), key=lambda kv: -kv[1])[:10], gaps=gaps[:max_list],
        effective_rate_hz=rate)


def exact_cell(lc: LoadedCsv, row: int, col: int) -> str:
    """Original cell text for one cell (re-parsed from the stored line)."""
    if lc.lines_match_records:
        rec = next(csv.reader([lc.lines[row]], delimiter=lc.delimiter))
        return rec[col] if col < len(rec) else ""
    if col in lc.text:
        return lc.text[col][row]
    return repr(lc.numeric[col][row])


def exact_timestamp_ms(raw: str, unit: str = "ms") -> Decimal | None:
    """Parse a timestamp string exactly (no binary float rounding) and convert to milliseconds."""
    try:
        return Decimal(raw.strip()) * Decimal(str(UNIT_TO_MS[unit]))
    except (InvalidOperation, KeyError):
        return None
