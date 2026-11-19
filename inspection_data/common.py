"""Shared loading, definitions and the number registry of the inspection.

Data versions (named in every registered number):
  raw   the 2022 CSV files in data/raw (read-only), read with the same reader as the build
        (emob.data.build.read), so "usable sensor" has one definition everywhere;
  v2    the prepared data in data/processed/v2. Before any number is computed, v2 is rebuilt in
        memory from raw + params/*.json with emob.data.build.process and its content hash is compared
        with the files on disk: the inspection refuses to run if they differ.
  params the frozen estimates in code/params/*.json (sensor offsets, label offsets, long runs).

Units (named in every registered number):
  row   one CSV row, 1/50 s;  step  one output step, 0.1 s (label = centre row of the step).

Segment = maximal run of consecutive rows (or steps) with the same non-empty value.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path

import numpy as np

from emob import config as C
from emob.data import alignment, anomalies, build, sync
from emob.data.splits import PARTICIPANTS

V2 = C.PROCESSED
OUT = C.PROJECT / "reports" / "inspection"
ROW_S = 1 / C.FS          # seconds per CSV row
STEP_S = C.STEP / C.FS    # seconds per output step (0.1)
PNAME = {u: f"P{u[4:]}" for u in PARTICIPANTS}      # user3 -> P3, the name used in reports


# ------------------------------------------------------------------ segments
def segments(a: np.ndarray, empty=None) -> list[tuple[object, int, int]]:
    """Maximal runs (value, start, end_exclusive) of equal values, skipping `empty` values.
    For string arrays empty defaults to "" ; for integer label arrays to C.IGNORE (-1)."""
    a = np.asarray(a)
    if empty is None:
        empty = "" if a.dtype == object or a.dtype.kind in "US" else C.IGNORE
    if len(a) == 0:
        return []
    cut = np.flatnonzero(a[1:] != a[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(a)]
    return [(a[s], int(s), int(e)) for s, e in zip(starts, ends) if a[s] != empty]


def lengths(a, unit_s: float, empty=None, keep=None) -> np.ndarray:
    """Durations (s) of the segments of `a`; `keep(value)` filters the values."""
    return np.array([(e - s) * unit_s for v, s, e in segments(a, empty) if keep is None or keep(v)])


# ------------------------------------------------------------------ registry
@dataclass
class Entry:
    id: str
    value: object
    unit: str
    data: str
    definition: str
    section: str


@dataclass
class Registry:
    entries: dict = field(default_factory=dict)
    section: str = ""

    def add(self, id: str, value, unit: str, data: str, definition: str):
        if id in self.entries:
            raise KeyError(f"duplicate number id {id}")
        if isinstance(value, (np.floating, float)):
            value = round(float(value), 4)
        elif isinstance(value, np.integer):
            value = int(value)
        self.entries[id] = Entry(id, value, unit, data, definition, self.section)
        return value


# ------------------------------------------------------------------ data context
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Context:
    """Loads raw, rebuilds v2 in memory and checks it against the files on disk."""

    def __init__(self):
        self.users = list(PARTICIPANTS)
        self.files = {u: next(C.RAW_2022.glob(f"{u}-*.csv")) for u in self.users}
        self.raw = {u: build.read(self.files[u]) for u in self.users}
        self.sens_off, self.lab_off, self.long_runs = sync.load_offsets(), alignment.load_offsets(), anomalies.load_runs()
        vocab = sorted({v for r in self.raw.values() for c in C.FAMILY_COLS for v in r["fam"][c]})
        vindex = {v: i for i, v in enumerate(vocab)}
        self.proc = {u: build.process(u, self.raw[u], self.sens_off, self.lab_off, self.long_runs, vindex) for u in self.users}
        self.manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
        self.v2 = {u: dict(np.load(V2 / f"{u}.npz")) for u in self.users}
        self.integrity = self._check_integrity()

    def _check_integrity(self) -> dict:
        disk = {p["participant"]: p["content_sha256"] for p in self.manifest["participants"]}
        mem = {u: self.proc[u][1]["content_sha256"] for u in self.users}
        files = {u: build.content_sha256(self.v2[u]) for u in self.users}
        if not (disk == mem == files):
            raise RuntimeError("data/processed/v2 differs from a rebuild of raw + params: rebuild v2 first "
                               f"(manifest {disk}, rebuilt {mem}, files {files})")
        return {"v2_matches_rebuild": True, "content_sha256": mem}

    def stats(self, u: str) -> dict:
        """Build statistics of participant u (seconds counted on 50-Hz rows)."""
        return self.proc[u][1]

    @cached_property
    def raw_sha256(self) -> dict:
        return {u: build.sha256_file(self.files[u]) for u in self.users}


# ------------------------------------------------------------------ output
def git_state() -> dict:
    return build.git_state()


def write(reg: Registry, ctx: Context, out: Path = OUT) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    meta = {"created": datetime.now().isoformat(timespec="seconds"), "git": git_state(),
            "raw_sha256": ctx.raw_sha256, "params_sha256": {f.name: build.sha256_file(f) for f in sorted(C.PARAMS.glob("*.json"))},
            "v2": ctx.integrity}
    js = out / "inspection_numbers.json"
    js.write_text(json.dumps({"meta": meta, "numbers": [vars(e) for e in reg.entries.values()]}, indent=1,
                             ensure_ascii=False, default=str), encoding="utf-8")
    L = ["# eMOB 2022 — inspection numbers", "",
         "Generated by `code/inspection_data` (`python -m inspection_data.run_all` from `code/`). "
         "**Do not edit by hand.** Every number about the data cited in a report must appear here; "
         "cite it by its ID in a comment next to the number.", "",
         f"- Created: {meta['created']}; code commit `{meta['git']['commit']}`"
         + (" (**uncommitted changes**)" if meta["git"]["dirty"] else ""),
         "- Data: raw = `data/raw/capture_2022-01-26` CSV; v2 = `data/processed/v2`, rebuilt in memory from raw + "
         "`code/params/*.json` and identical to the files on disk (content hashes match); params = `code/params/*.json`.",
         "- Units: *row* = one CSV row (1/50 s); *step* = one output step (0.1 s, label of the centre row). "
         "A *segment* is a maximal run of equal non-empty values.", ""]
    sec = None
    for e in reg.entries.values():
        if e.section != sec:
            sec = e.section
            L += ["", f"## {sec}", "", "| ID | Value | Unit | Data | Definition |", "|---|---|---|---|---|"]
        v = e.value if not isinstance(e.value, float) else f"{e.value:g}"
        L.append(f"| `{e.id}` | {v} | {e.unit} | {e.data} | {e.definition} |")
    L += ["", "## Data fingerprints", "", "| File | SHA-256 |", "|---|---|"]
    L += [f"| raw {u} | `{h}` |" for u, h in ctx.raw_sha256.items()]
    L += [f"| params {k} | `{h}` |" for k, h in meta["params_sha256"].items()]
    L += [f"| v2 {u} (content) | `{h}` |" for u, h in ctx.integrity["content_sha256"].items()]
    md = out / "inspection_numbers.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    return md, js
