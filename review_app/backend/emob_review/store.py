"""Recording registry, CSV cache, and review outputs (profiles, notes, exports).

All writes go to cfg.output_dir. Source files are only ever opened for reading.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from .config import Config, resolve_read_path, resolve_write_path
from .csvload import LoadedCsv, load_csv
from .quality import group_flags

SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class Recording:
    id: str
    csv_path: Path
    video_path: Path | None
    pairing_rule: str | None   # None = no proposal / user-defined
    origin: str                # "scan" | "custom"


class Store:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out = cfg.output_dir
        for sub in ("profiles", "notes", "verdicts", "exports", "cache"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self._cache: "OrderedDict[tuple, LoadedCsv]" = OrderedDict()
        self._lock = threading.Lock()
        self._recordings: dict[str, Recording] = {}
        self.rescan()

    # ------------------------------------------------------------ recordings
    def rescan(self) -> None:
        recs: dict[str, Recording] = {}
        for d in self.cfg.scan_dirs:
            if not d.is_dir():
                continue
            files = sorted(p for p in d.iterdir() if p.is_file())
            vids = [p for p in files if p.suffix.lower() in (".mp4", ".m4v", ".mov", ".webm")]
            for c in (p for p in files if p.suffix.lower() == ".csv"):
                video, rule = None, None
                for r in self.cfg.pairing_rules:
                    m = re.match(r.csv_regex, c.name)
                    if not m:
                        continue
                    cands = [v for v in vids if (vm := re.match(r.video_regex, v.name))
                             and vm.group("key") == m.group("key")]
                    if len(cands) == 1:
                        video, rule = cands[0], r.name
                        break
                rid = SAFE_ID.sub("_", f"{d.name}__{c.stem}")
                recs[rid] = Recording(rid, c, video, rule, "scan")
        for item in self._read_json(self.out / "custom_recordings.json", []):
            rid = item["id"]
            recs[rid] = Recording(rid, Path(item["csv_path"]),
                                  Path(item["video_path"]) if item.get("video_path") else None, None, "custom")
        self._recordings = recs

    def recordings(self) -> list[Recording]:
        return list(self._recordings.values())

    def get(self, rid: str) -> Recording:
        if rid not in self._recordings:
            raise KeyError(rid)
        return self._recordings[rid]

    def add_custom(self, csv_path: str, video_path: str | None) -> Recording:
        c = resolve_read_path(self.cfg, csv_path)
        v = resolve_read_path(self.cfg, video_path) if video_path else None
        if not c.is_file():
            raise FileNotFoundError(c)
        if v is not None and not v.is_file():
            raise FileNotFoundError(v)
        rid = SAFE_ID.sub("_", f"custom__{c.parent.name}__{c.stem}" + (f"__{v.stem}" if v else ""))
        items = [i for i in self._read_json(self.out / "custom_recordings.json", []) if i["id"] != rid]
        items.append({"id": rid, "csv_path": str(c), "video_path": str(v) if v else None, "added": now_iso()})
        self._write_json(self.out / "custom_recordings.json", items)
        self.rescan()
        return self._recordings[rid]

    # ------------------------------------------------------------ CSV cache
    def load(self, rec: Recording, encoding: str | None = None, delimiter: str | None = None,
             decimal: str = ".") -> LoadedCsv:
        st = rec.csv_path.stat()
        key = (str(rec.csv_path), st.st_size, st.st_mtime, encoding, delimiter, decimal)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        lc = load_csv(rec.csv_path, encoding=encoding, delimiter=delimiter, decimal=decimal)
        with self._lock:
            self._cache[key] = lc
            while len(self._cache) > 3:
                self._cache.popitem(last=False)
        return lc

    # ------------------------------------------------------------ summaries (availability flags)
    def _summary_path(self, rec: Recording) -> Path:
        st = rec.csv_path.stat()
        h = hashlib.sha1(f"{rec.csv_path}|{st.st_size}|{st.st_mtime}".encode()).hexdigest()[:16]
        return self.out / "cache" / f"summary_{h}.json"

    def cached_summary(self, rec: Recording) -> dict | None:
        return self._read_json(self._summary_path(rec), None)

    def compute_summary(self, rec: Recording) -> dict:
        lc = self.load(rec)
        label_cols = [i for i in range(len(lc.header)) if lc.column_kind(i) == "text"]
        all_label_cols_empty = [i for i in range(len(lc.header))
                                if lc.header[i].lower().startswith("etiq") and lc.cls[i].count(2) == lc.n_rows]
        groups = []
        for g in lc.groups:
            groups.append({"source": g.source, "placement": g.placement, "modality": g.modality,
                           **group_flags(lc, g.axes)})
        s = {
            "computed": now_iso(), "csv_sha256": lc.sha256, "rows": lc.n_rows, "cols": len(lc.header),
            "text_columns": [lc.header[i] for i in label_cols],
            "label_columns_entirely_empty": [lc.header[i] for i in all_label_cols_empty],
            "groups": groups,
        }
        self._write_json(self._summary_path(rec), s)
        return s

    # ------------------------------------------------------------ profiles & notes
    def profile_path(self, rid: str) -> Path:
        return resolve_write_path(self.cfg, self.out / "profiles" / f"{SAFE_ID.sub('_', rid)}.json")

    def get_profile(self, rid: str) -> dict | None:
        return self._read_json(self.profile_path(rid), None)

    def put_profile(self, rid: str, profile: dict) -> dict:
        profile = dict(profile)
        profile["recording_id"] = rid
        profile["saved_at"] = now_iso()
        self._write_json(self.profile_path(rid), profile)
        return profile

    def notes_path(self, rid: str) -> Path:
        return resolve_write_path(self.cfg, self.out / "notes" / f"{SAFE_ID.sub('_', rid)}.json")

    def get_notes(self, rid: str) -> list[dict]:
        return self._read_json(self.notes_path(rid), [])

    def add_note(self, rid: str, note: dict) -> dict:
        notes = self.get_notes(rid)
        note = dict(note)
        note["id"] = f"n{dt.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        note["created"] = now_iso()
        note["recording_id"] = rid
        notes.append(note)
        self._write_json(self.notes_path(rid), notes)
        return note

    def delete_note(self, rid: str, nid: str) -> None:
        self._write_json(self.notes_path(rid), [n for n in self.get_notes(rid) if n.get("id") != nid])

    # ------------------------------------------------------------ label verdicts
    # A verdict is the reviewer's judgement of ONE label segment (column + row range + raw value).
    # Stored separately; never written to the source CSV. Re-judging a segment keeps the old values in history.
    VERDICT_KINDS = ("agree", "disagree", "unsure")
    _VKEY = ("column", "start_row", "end_row", "label_value")

    def verdicts_path(self, rid: str) -> Path:
        return resolve_write_path(self.cfg, self.out / "verdicts" / f"{SAFE_ID.sub('_', rid)}.json")

    def get_verdicts(self, rid: str) -> list[dict]:
        return self._read_json(self.verdicts_path(rid), [])

    def put_verdict(self, rid: str, v: dict) -> dict:
        missing = [k for k in (*self._VKEY, "column_name", "verdict") if k not in v]
        if missing:
            raise ValueError(f"Verdict is missing fields: {missing}")
        if v["verdict"] not in self.VERDICT_KINDS:
            raise ValueError(f"verdict must be one of {self.VERDICT_KINDS}")
        items = self.get_verdicts(rid)
        key = tuple(v[k] for k in self._VKEY)
        now = now_iso()
        existing = next((x for x in items if tuple(x.get(k) for k in self._VKEY) == key), None)
        if existing:
            prev = {k: existing.get(k) for k in ("verdict", "observed", "comment", "updated", "alignment_snapshot")}
            existing.update({k: val for k, val in v.items() if k not in ("id", "created", "history", "recording_id")})
            existing["updated"] = now
            existing["history"] = [*existing.get("history", []), prev]
            out = existing
        else:
            out = {**v, "id": f"v{dt.datetime.now().strftime('%Y%m%d%H%M%S%f')}", "created": now, "updated": now,
                   "recording_id": rid, "history": []}
            items.append(out)
        self._write_json(self.verdicts_path(rid), items)
        return out

    def delete_verdict(self, rid: str, vid: str) -> None:
        self._write_json(self.verdicts_path(rid), [x for x in self.get_verdicts(rid) if x.get("id") != vid])

    # ------------------------------------------------------------ exports
    def export(self, rec: Recording, lc: LoadedCsv, req: dict) -> Path:
        r0, r1 = int(req["row_start"]), int(req["row_end"])
        if not (0 <= r0 < r1 <= lc.n_rows):
            raise ValueError(f"Invalid row range [{r0}, {r1}) for {lc.n_rows} rows")
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = resolve_write_path(self.cfg, self.out / "exports" / SAFE_ID.sub("_", rec.id) / f"{stamp}_rows{r0}-{r1}")
        folder.mkdir(parents=True, exist_ok=False)

        ts_col = int(req["timestamp"]["column"])
        unit_ms = {"ms": 1.0, "s": 1000.0, "us": 1e-3, "ns": 1e-6}[req["timestamp"]["unit"]]
        ref = float(req["timestamp"]["reference_value"])
        derived = ["derived_row_index", "derived_t_csv_s"]
        with open(folder / "rows.csv", "w", encoding="utf-8", newline="") as f:
            if lc.lines_match_records:
                # Original header and lines are copied verbatim after the derived columns.
                hdr_buf = io.StringIO()
                csv.writer(hdr_buf, delimiter=lc.delimiter, lineterminator="").writerow(lc.header)
                f.write(lc.delimiter.join(derived) + lc.delimiter + hdr_buf.getvalue() + "\n")
                for r in range(r0, r1):
                    t = (lc.numeric[ts_col][r] - ref) * unit_ms / 1000.0
                    f.write(f"{r}{lc.delimiter}{t!r}{lc.delimiter}{lc.lines[r]}\n")
                verbatim = True
            else:
                w = csv.writer(f, delimiter=lc.delimiter)
                w.writerow(derived + lc.header)
                for r in range(r0, r1):
                    t = (lc.numeric[ts_col][r] - ref) * unit_ms / 1000.0
                    cells = [lc.text[c][r] if c in lc.text else repr(lc.numeric[c][r]) for c in range(len(lc.header))]
                    w.writerow([r, repr(t), *cells])
                verbatim = False

        self._write_json(folder / "label_segments.json", req.get("label_segments", []))
        self._write_json(folder / "notes.json", req.get("notes", []))
        self._write_json(folder / "label_verdicts.json", req.get("verdicts", []))
        self._write_json(folder / "channel_mapping.json", req.get("channel_mapping", {}))
        self._write_json(folder / "alignment.json", req.get("alignment", {}))
        self._write_json(folder / "manifest.json", {
            "created": now_iso(),
            "recording_id": rec.id,
            "source_csv": str(rec.csv_path), "source_csv_sha256": lc.sha256,
            "source_video": str(rec.video_path) if rec.video_path else None,
            "rows": [r0, r1], "row_index_convention": "0-based data row (file line = row + 2)",
            "timestamp": req["timestamp"],
            "derived_columns": {
                "derived_row_index": "0-based data row index in the source CSV (added by the review app)",
                "derived_t_csv_s": "(timestamp - reference_value) converted to seconds (added by the review app)",
            },
            "source_rows_verbatim": verbatim,
            "verbatim_note": ("Columns after the derived ones are the original header and line text, copied verbatim."
                              if verbatim else
                              "Source physical lines did not map 1:1 to CSV records; cells were re-serialised. "
                              "Numeric cells are float repr of parsed values; text cells are original strings."),
            "warning": "Derived review output. Not an annotation source. Original files were not modified.",
        })
        return folder

    # ------------------------------------------------------------ helpers
    def _read_json(self, p: Path, default):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default

    def _write_json(self, p: Path, data) -> None:
        p = resolve_write_path(self.cfg, p)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
