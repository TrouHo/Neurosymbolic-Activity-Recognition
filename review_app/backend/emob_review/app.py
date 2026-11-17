"""FastAPI backend for eMOB Data Review (local only)."""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import APP_DIR, Config, PathNotAllowed, load_config, resolve_read_path
from .csvload import CLASS_NAMES, LoadedCsv, decode_bytes, exact_cell, sniff_delimiter, timestamp_report
from .labels import label_overview
from .mp4meta import read_mp4_metadata
from .quality import column_counts, constant_runs, group_flags, group_runs
from .store import Recording, Store

CONTRIBUTOR_TS_NOTE = ("Proposed from column name 'ts_SQ'. Contributor states timestamps are Unix milliseconds "
                       "and the combined CSV is built on 50 Hz Qualisys data (not independently verified).")
DERIVED_LABEL_NOTE = ("Proposed as derived because explanation_data.docx describes category/activity labels as "
                      "derived from the family columns (construction not verified).")


def _fmt_floats(vals) -> str:
    return "[" + ",".join("null" if (v != v or v in (math.inf, -math.inf)) else repr(v) for v in vals) + "]"


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    store = Store(cfg)
    app = FastAPI(title="eMOB Data Review", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.exception_handler(PathNotAllowed)
    async def _forbidden(_: Request, exc: PathNotAllowed):
        return JSONResponse({"detail": str(exc)}, status_code=403)

    @app.exception_handler(KeyError)
    async def _missing(_: Request, exc: KeyError):
        return JSONResponse({"detail": f"Not found: {exc}"}, status_code=404)

    @app.exception_handler(ValueError)
    async def _bad(_: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    def rec_and_csv(rid: str, encoding: str | None, delimiter: str | None, decimal: str) -> tuple[Recording, LoadedCsv]:
        rec = store.get(rid)
        return rec, store.load(rec, encoding or None, delimiter or None, decimal or ".")

    def rec_json(r: Recording) -> dict:
        return {
            "id": r.id, "csv_path": str(r.csv_path), "video_path": str(r.video_path) if r.video_path else None,
            "csv_size": r.csv_path.stat().st_size if r.csv_path.exists() else None,
            "pairing_rule": r.pairing_rule, "origin": r.origin,
            "summary": store.cached_summary(r) if r.csv_path.exists() else None,
        }

    # ------------------------------------------------------------ general
    @app.get("/api/config")
    def get_config():
        return {"data_roots": [str(p) for p in cfg.data_roots], "scan_dirs": [str(p) for p in cfg.scan_dirs],
                "output_dir": str(cfg.output_dir)}

    @app.get("/api/recordings")
    def list_recordings():
        return [rec_json(r) for r in store.recordings()]

    @app.post("/api/recordings/{rid}/summary")
    def summarize_recording(rid: str):
        return store.compute_summary(store.get(rid))

    @app.post("/api/recordings/custom")
    def add_custom(body: dict = Body(...)):
        r = store.add_custom(body["csv_path"], body.get("video_path"))
        return rec_json(r)

    @app.get("/api/browse")
    def browse(path: str | None = None):
        if not path:
            return {"path": None, "entries": [{"name": str(p), "path": str(p), "dir": True} for p in cfg.data_roots]}
        p = resolve_read_path(cfg, path)
        if not p.is_dir():
            raise HTTPException(400, f"Not a directory: {p}")
        entries = [{"name": c.name, "path": str(c), "dir": c.is_dir(), "size": None if c.is_dir() else c.stat().st_size}
                   for c in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower()))]
        parent = str(p.parent) if any(p != r and p.is_relative_to(r) for r in cfg.data_roots) else None
        return {"path": str(p), "parent": parent, "entries": entries}

    # ------------------------------------------------------------ recording data
    @app.get("/api/rec/{rid}/info")
    def info(rid: str, encoding: str = "", delimiter: str = "", decimal: str = "."):
        rec, lc = rec_and_csv(rid, encoding, delimiter, decimal)
        cols = []
        for i, name in enumerate(lc.header):
            cols.append({"index": i, "name": name, "kind": lc.column_kind(i), "counts": column_counts(lc, i)})
        groups = []
        for g in lc.groups:
            groups.append({**g.__dict__, "flags": group_flags(lc, g.axes)})
        # Proposals (always shown to the user as proposals with provenance)
        ts_prop = None
        if "ts_SQ" in lc.header:
            ts_prop = {"column": lc.header.index("ts_SQ"), "unit": "ms", "provenance": CONTRIBUTOR_TS_NOTE}
        else:
            for i, n in enumerate(lc.header):
                if (n.lower().startswith("ts") or "time" in n.lower()) and lc.column_kind(i) == "numeric":
                    ts_prop = {"column": i, "unit": None,
                               "provenance": f"Proposed from column name '{n}'. Unit unknown: select it."}
                    break
        label_props = []
        for i, n in enumerate(lc.header):
            if lc.column_kind(i) in ("text", "empty") and not any(i in g.axes.values() for g in lc.groups):
                derived = any(k in n.lower() for k in ("categorie", "activite"))
                label_props.append({"column": i, "name": n, "role": "derived" if derived else "family",
                                    "provenance": DERIVED_LABEL_NOTE if derived else
                                    "Proposed: text column (values are non-numeric strings)"})
        return {
            "recording": rec_json(rec),
            "csv": {"path": str(lc.path), "size": lc.size, "sha256": lc.sha256, "encoding": lc.encoding,
                    "delimiter": lc.delimiter, "decimal": lc.decimal, "rows": lc.n_rows, "columns": len(lc.header),
                    "bad_rows": lc.bad_rows[:200], "n_bad_rows": len(lc.bad_rows),
                    "lines_match_records": lc.lines_match_records,
                    "row_index_convention": "0-based data row (file line = row + 2)"},
            "columns": cols, "class_names": CLASS_NAMES, "groups": groups,
            "proposals": {"timestamp": ts_prop, "labels": label_props},
        }

    @app.get("/api/rec/{rid}/timestamps")
    def timestamps(rid: str, col: int, unit: str = "ms", encoding: str = "", delimiter: str = "", decimal: str = "."):
        _, lc = rec_and_csv(rid, encoding, delimiter, decimal)
        rep = timestamp_report(lc, col, unit)
        body = ('{"report":' + JSONResponse(content=rep.__dict__).body.decode()
                + ',"first_raw":' + JSONResponse(content=exact_cell(lc, 0, col)).body.decode()
                + ',"values":' + _fmt_floats(lc.numeric[col]) + "}")
        return Response(body, media_type="application/json")

    @app.get("/api/rec/{rid}/series")
    def series(rid: str, cols: str, encoding: str = "", delimiter: str = "", decimal: str = "."):
        _, lc = rec_and_csv(rid, encoding, delimiter, decimal)
        idx = [int(c) for c in cols.split(",") if c != ""]
        parts = []
        for c in idx:
            if not 0 <= c < len(lc.header):
                raise HTTPException(400, f"Column out of range: {c}")
            parts.append(f'"{c}":' + _fmt_floats(lc.numeric[c]))
        return Response("{" + ",".join(parts) + "}", media_type="application/json")

    @app.get("/api/rec/{rid}/row/{row}")
    def raw_row(rid: str, row: int, encoding: str = "", delimiter: str = "", decimal: str = "."):
        _, lc = rec_and_csv(rid, encoding, delimiter, decimal)
        if not 0 <= row < lc.n_rows:
            raise HTTPException(400, "Row out of range")
        if lc.lines_match_records:
            rec = next(csv.reader([lc.lines[row]], delimiter=lc.delimiter))
            cells = [rec[c] if c < len(rec) else "" for c in range(len(lc.header))]
        else:
            cells = [exact_cell(lc, row, c) for c in range(len(lc.header))]
        return {"row": row, "file_line": row + 2,
                "cells": [{"index": c, "name": lc.header[c], "raw": cells[c], "class": CLASS_NAMES[lc.cls[c][row]]}
                          for c in range(len(lc.header))]}

    @app.get("/api/rec/{rid}/labels")
    def labels(rid: str, ts_col: int, family: str = "", derived: str = "", encoding: str = "", delimiter: str = "",
               decimal: str = "."):
        _, lc = rec_and_csv(rid, encoding, delimiter, decimal)
        fam = [int(c) for c in family.split(",") if c]
        der = [int(c) for c in derived.split(",") if c]
        rep = timestamp_report(lc, ts_col)
        if rep.median_step is None:
            raise HTTPException(400, "Timestamp column has no positive steps")
        return label_overview(lc, lc.numeric[ts_col], rep.median_step, fam, der)

    @app.get("/api/rec/{rid}/quality")
    def quality(rid: str, ts_col: int, unit: str = "ms", min_len: int = Query(1, ge=1),
                const_min_len: int = Query(25, ge=2), encoding: str = "", delimiter: str = "", decimal: str = "."):
        _, lc = rec_and_csv(rid, encoding, delimiter, decimal)
        ts = lc.numeric[ts_col]
        groups = []
        for g in lc.groups:
            groups.append({"id": g.id, "source": g.source, "placement": g.placement, "modality": g.modality,
                           "header": g.header, "flags": group_flags(lc, g.axes),
                           "runs": group_runs(lc, g.axes, ts, min_len),
                           "constant_runs": {ax: constant_runs(lc, c, ts, const_min_len) for ax, c in g.axes.items()}})
        return {"timestamp": timestamp_report(lc, ts_col, unit).__dict__, "groups": groups,
                "thresholds": {"min_run_samples": min_len, "constant_min_samples": const_min_len},
                "notes": ["Zeros are suspicious candidates for review, not assumed missing.",
                          "Runs are counted on the raw parsed values; nothing was interpolated or smoothed."]}

    # ------------------------------------------------------------ video
    @app.get("/api/rec/{rid}/video_meta")
    def video_meta(rid: str):
        rec = store.get(rid)
        if not rec.video_path:
            raise HTTPException(404, "No video paired with this recording")
        return {"path": str(rec.video_path), **read_mp4_metadata(rec.video_path)}

    @app.get("/api/rec/{rid}/video")
    def video(rid: str):
        rec = store.get(rid)
        if not rec.video_path:
            raise HTTPException(404, "No video paired with this recording")
        p = resolve_read_path(cfg, rec.video_path)
        media = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}
        return FileResponse(p, media_type=media.get(p.suffix.lower(), "application/octet-stream"))

    # ------------------------------------------------------------ review outputs
    @app.get("/api/rec/{rid}/profile")
    def get_profile(rid: str):
        store.get(rid)
        return {"profile": store.get_profile(rid), "path": str(store.profile_path(rid))}

    @app.put("/api/rec/{rid}/profile")
    def put_profile(rid: str, profile: dict = Body(...)):
        store.get(rid)
        return {"profile": store.put_profile(rid, profile), "path": str(store.profile_path(rid))}

    @app.get("/api/rec/{rid}/notes")
    def get_notes(rid: str):
        store.get(rid)
        return store.get_notes(rid)

    @app.post("/api/rec/{rid}/notes")
    def add_note(rid: str, note: dict = Body(...)):
        store.get(rid)
        return store.add_note(rid, note)

    @app.delete("/api/rec/{rid}/notes/{nid}")
    def delete_note(rid: str, nid: str):
        store.delete_note(rid, nid)
        return {"ok": True}

    @app.get("/api/rec/{rid}/verdicts")
    def get_verdicts(rid: str):
        store.get(rid)
        return store.get_verdicts(rid)

    @app.put("/api/rec/{rid}/verdicts")
    def put_verdict(rid: str, verdict: dict = Body(...)):
        store.get(rid)
        return store.put_verdict(rid, verdict)

    @app.delete("/api/rec/{rid}/verdicts/{vid}")
    def delete_verdict(rid: str, vid: str):
        store.delete_verdict(rid, vid)
        return {"ok": True}

    @app.post("/api/rec/{rid}/export")
    def export(rid: str, req: dict = Body(...), encoding: str = "", delimiter: str = "", decimal: str = "."):
        rec, lc = rec_and_csv(rid, encoding, delimiter, decimal)
        folder = store.export(rec, lc, req)
        return {"folder": str(folder), "files": sorted(p.name for p in folder.iterdir())}

    # ------------------------------------------------------------ external label files
    @app.post("/api/external_labels/preview")
    def ext_preview(body: dict = Body(...)):
        p = resolve_read_path(cfg, body["path"])
        text, enc = decode_bytes(p.read_bytes(), body.get("encoding") or None)
        delim = body.get("delimiter") or sniff_delimiter(text[:65536])
        rows = list(csv.reader(io.StringIO(text), delimiter=delim))
        return {"path": str(p), "encoding": enc, "delimiter": delim, "header": rows[0] if rows else [],
                "rows": rows[1:21], "n_rows": max(0, len(rows) - 1)}

    @app.post("/api/external_labels/load")
    def ext_load(body: dict = Body(...)):
        """Return raw rows with parsed start/end numbers; time mapping is done in the UI with an
        explicit unit and reference. Rows without an end column are point events (never forward-filled)."""
        p = resolve_read_path(cfg, body["path"])
        text, enc = decode_bytes(p.read_bytes(), body.get("encoding") or None)
        delim = body.get("delimiter") or sniff_delimiter(text[:65536])
        rows = list(csv.reader(io.StringIO(text), delimiter=delim))
        sc, ec, lc_, fc = body["start_col"], body.get("end_col"), body["label_col"], body.get("family_col")

        def num(s):
            try:
                v = float(s.strip())
                return v if math.isfinite(v) else None
            except ValueError:
                return None

        items = []
        for i, r in enumerate(rows[1:]):
            get = lambda c: r[c] if c is not None and c < len(r) else None  # noqa: E731
            items.append({"row": i, "start_raw": get(sc), "end_raw": get(ec), "label": get(lc_),
                          "family": get(fc) if fc is not None else None,
                          "start": num(get(sc) or ""), "end": num(get(ec) or "") if ec is not None else None})
        return {"path": str(p), "encoding": enc, "delimiter": delim, "header": rows[0] if rows else [],
                "kind": "intervals" if ec is not None else "points", "items": items}

    # ------------------------------------------------------------ static frontend
    dist = APP_DIR / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")

    return app
