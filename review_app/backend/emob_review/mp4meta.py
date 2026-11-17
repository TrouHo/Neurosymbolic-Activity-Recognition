"""Minimal ISO-BMFF (MP4) metadata reader (no ffprobe on this machine).

Reads only box headers and the moov box. Reports what the container states; FPS is reported
as nominal (timescale / constant sample delta) with its provenance, never guessed.
"""
from __future__ import annotations

import struct
from pathlib import Path

_CONTAINERS = {"moov", "trak", "mdia", "minf", "stbl", "edts"}


def _iter_boxes(buf: bytes, off: int, end: int):
    while off + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[off:off + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", buf[off + 8:off + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - off
        if size < hdr:
            return
        yield typ.decode("latin1"), off + hdr, off + size
        off += size


def _find_moov(f, file_size: int) -> bytes | None:
    off = 0
    while off + 8 <= file_size:
        f.seek(off)
        h = f.read(16)
        size, typ = struct.unpack(">I4s", h[:8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", h[8:16])[0]
            hdr = 16
        elif size == 0:
            size = file_size - off
        if size < hdr:
            return None
        if typ == b"moov":
            f.seek(off)
            return f.read(size)
        off += size
    return None


def read_mp4_metadata(path: Path) -> dict:
    size = path.stat().st_size
    with open(path, "rb") as f:
        moov = _find_moov(f, size)
    if moov is None:
        return {"ok": False, "error": "No moov box found (not an MP4/ISO-BMFF file?)"}
    res: dict = {"ok": True, "file_size": size, "tracks": []}
    track: dict = {}

    def walk(off, end):
        nonlocal track
        for typ, body, bend in _iter_boxes(moov, off, end):
            if typ == "trak":
                track = {}
                res["tracks"].append(track)
                walk(body, bend)
            elif typ in _CONTAINERS:
                walk(body, bend)
            elif typ == "mvhd":
                v = moov[body]
                ts, du = (struct.unpack(">IQ", moov[body + 20:body + 32]) if v == 1
                          else struct.unpack(">II", moov[body + 12:body + 20]))
                res["movie_timescale"] = ts
                res["movie_duration_s"] = du / ts if ts else None
            elif typ == "tkhd":
                w, h = struct.unpack(">II", moov[bend - 8:bend])
                track["display_wh"] = [w >> 16, h >> 16]
            elif typ == "mdhd":
                v = moov[body]
                ts, du = (struct.unpack(">IQ", moov[body + 20:body + 32]) if v == 1
                          else struct.unpack(">II", moov[body + 12:body + 20]))
                track["timescale"] = ts
                track["duration_s"] = du / ts if ts else None
            elif typ == "hdlr":
                track["handler"] = moov[body + 8:body + 12].decode("latin1")
            elif typ == "stsd":
                track["codec"] = moov[body + 12:body + 16].decode("latin1")
                if track.get("handler") == "vide":
                    w, h = struct.unpack(">HH", moov[body + 8 + 32:body + 8 + 36])
                    track["coded_wh"] = [w, h]
            elif typ == "stts":
                cnt = struct.unpack(">I", moov[body + 4:body + 8])[0]
                ents = [list(struct.unpack(">II", moov[body + 8 + 8 * i:body + 16 + 8 * i])) for i in range(cnt)]
                track["stts_entry_count"] = cnt
                track["stts_first_entries"] = ents[:6]
                track["sample_count"] = sum(e[0] for e in ents)
                nonzero = {e[1] for e in ents if e[1] != 0}
                track["zero_duration_samples"] = sum(e[0] for e in ents if e[1] == 0)
                track["constant_delta"] = next(iter(nonzero)) if len(nonzero) == 1 else None
            elif typ == "ctts":
                track["has_composition_offsets"] = True  # B-frames / reordering present
            elif typ == "elst":
                v = moov[body]
                cnt = struct.unpack(">I", moov[body + 4:body + 8])[0]
                ents = []
                for i in range(cnt):
                    if v == 1:
                        sd, mt = struct.unpack(">Qq", moov[body + 8 + 20 * i:body + 24 + 20 * i])
                    else:
                        sd, mt = struct.unpack(">Ii", moov[body + 8 + 12 * i:body + 16 + 12 * i])
                    ents.append({"segment_duration_movie_units": sd, "media_time_track_units": mt})
                track["edit_list"] = ents

    walk(0, len(moov))
    for t in res["tracks"]:
        if t.get("handler") == "vide" and t.get("constant_delta") and t.get("timescale"):
            t["nominal_fps"] = t["timescale"] / t["constant_delta"]
            t["fps_provenance"] = ("MP4 container: mdhd timescale / constant stts sample delta"
                                   + (f"; {t['zero_duration_samples']} zero-duration sample(s) excluded"
                                      if t.get("zero_duration_samples") else ""))
        if t.get("edit_list") and t.get("timescale"):
            for e in t["edit_list"]:
                if e["media_time_track_units"] > 0:
                    e["media_time_s"] = e["media_time_track_units"] / t["timescale"]
    video = [t for t in res["tracks"] if t.get("handler") == "vide"]
    res["video_track"] = video[0] if video else None
    res["has_audio"] = any(t.get("handler") == "soun" for t in res["tracks"])
    return res
