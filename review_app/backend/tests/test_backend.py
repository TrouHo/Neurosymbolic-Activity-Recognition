"""Backend tests on SYNTHETIC files (generated in a temp dir). Real-data checks live in test_real_data.py."""
import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from emob_review.app import create_app
from emob_review.config import Config, PairingRule, PathNotAllowed, resolve_write_path
from emob_review.csvload import EMPTY, NANSTR, NUM, TEXT, ZERO, load_csv, parse_groups, timestamp_report
from emob_review.labels import segments_for
from emob_review.mp4meta import read_mp4_metadata

HEADER = ['ts_SQ', 'x_sq_D_main', 'y_sq_D_main', 'z_sq_D_main',
          'ts_SB - Main droite - ACC', 'x_sb', 'y_sb', 'z_sb',
          'ts_SB - Main droite - GYR', 'x_sb', 'y_sb', 'z_sb',
          'etiq standard', 'etiq mvt_bras', 'etiq mvt_jambe', 'etiq activite']


def make_csv(path: Path, ts: list[str], labels_bras: list[str], labels_jambe: list[str], gyr_nan_rows=()):
    lines = [",".join(f'"{h}"' for h in HEADER)]
    for i, t in enumerate(ts):
        acc = ["0", "0", "0"] if i == 1 else [f"{i * 0.1:.2f}", "-9.81", "0.5"]
        gyr = ["NaN", "NaN", "NaN"] if i in gyr_nan_rows else ["0.01", "0", "-0.02"]
        lines.append(",".join([t, "1.5", "0.0", "2.0", t, *acc, t, *gyr,
                               '"inactif"', f'"{labels_bras[i]}"', f'"{labels_jambe[i]}"', '""']))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def env(tmp_path):
    raw = tmp_path / "raw" / "cap"
    raw.mkdir(parents=True)
    out = tmp_path / "out"
    base = 1643189820000
    ts = [f"{base + 20 * i}.0" for i in range(10)]
    ts[6] = f"{base + 20 * 5}.0"            # duplicate timestamp at row 6
    ts[8] = f"{base + 20 * 8 + 200}.0"      # gap before row 8
    ts[9] = f"{base + 20 * 9 + 200}.0"
    bras = ["", "", "Lever_Bras", "Lever_Bras", "Lever_Bras", "inactif", "inactif", "", "", ""]
    jambe = ["", "", "Marcher", "Marcher", "", "", "", "", "", ""]
    make_csv(raw / "part1_data.csv", ts, bras, jambe, gyr_nan_rows={3, 4})
    (raw / "part1_video.mp4").write_bytes(b"\x00" * 64)
    cfg = Config(data_roots=[(tmp_path / "raw").resolve()], scan_dirs=[raw.resolve()], output_dir=out.resolve(),
                 pairing_rules=[PairingRule("part", r"^part(?P<key>\d+)_data\.csv$", r"^part(?P<key>\d+)_video\.mp4$")])
    return cfg, raw


def test_classification_and_groups(env):
    cfg, raw = env
    lc = load_csv(raw / "part1_data.csv")
    assert lc.n_rows == 10 and lc.delimiter == "," and lc.lines_match_records
    g = {(x.source, x.placement, x.modality): x for x in lc.groups}
    assert set(g) == {("SB", "Main droite", "ACC"), ("SB", "Main droite", "GYR"), ("sq", "D_main", "XYZ")}
    acc = g[("SB", "Main droite", "ACC")]
    assert lc.cls[acc.axes["x"]][1] == ZERO           # observed zero kept as zero, not missing
    assert lc.numeric[acc.axes["x"]][1] == 0.0
    gyr = g[("SB", "Main droite", "GYR")]
    assert lc.cls[gyr.axes["x"]][3] == NANSTR and math.isnan(lc.numeric[gyr.axes["x"]][3])
    assert lc.cls[HEADER.index("etiq mvt_bras")][2] == TEXT
    assert lc.cls[HEADER.index("etiq activite")][0] == EMPTY
    assert lc.column_kind(HEADER.index("etiq activite")) == "empty"
    assert lc.cls[1][0] == NUM


def test_timestamp_report_flags_anomalies(env):
    _, raw = env
    lc = load_csv(raw / "part1_data.csv")
    rep = timestamp_report(lc, 0, "ms")
    assert rep.first == 1643189820000.0          # exact in float64
    assert rep.n_duplicates == 1 and rep.duplicate_rows == [6]
    assert rep.n_nonmonotonic == 0
    assert rep.median_step == 20
    assert [g["row"] for g in rep.gaps] == [7, 8]   # row 7: 40 ms step after the duplicate; row 8: 220 ms
    assert rep.n_fractional == 0


def test_label_segments_keep_empty_and_split_at_anomaly(env):
    _, raw = env
    lc = load_csv(raw / "part1_data.csv")
    vals = lc.text[HEADER.index("etiq mvt_bras")]
    segs = segments_for(vals, lc.numeric[0], 20.0)
    assert [(s.start_row, s.end_row, s.value) for s in segs] == [
        (0, 2, ""), (2, 5, "Lever_Bras"), (5, 6, "inactif"), (6, 7, "inactif"), (7, 8, ""), (8, 10, "")]
    assert not segs[1].split_at_anomaly
    assert segs[2].split_at_anomaly and segs[3].split_at_anomaly   # duplicate ts at row 6
    assert segs[4].split_at_anomaly and segs[4].end_extrapolated   # gap after row 7 -> end not bridged
    assert segs[-1].end_extrapolated             # last sample end extrapolated by median step


def test_write_guard(env, tmp_path):
    cfg, raw = env
    with pytest.raises(PathNotAllowed):
        resolve_write_path(cfg, raw / "x.json")
    with pytest.raises(PathNotAllowed):
        resolve_write_path(cfg, tmp_path / "elsewhere.json")


def test_api_end_to_end(env):
    cfg, raw = env
    src_before = (raw / "part1_data.csv").read_bytes()
    c = TestClient(create_app(cfg))
    recs = c.get("/api/recordings").json()
    assert len(recs) == 1 and recs[0]["video_path"].endswith("part1_video.mp4")
    rid = recs[0]["id"]
    info = c.get(f"/api/rec/{rid}/info").json()
    assert info["proposals"]["timestamp"]["column"] == 0 and info["proposals"]["timestamp"]["unit"] == "ms"
    roles = {p["name"]: p["role"] for p in info["proposals"]["labels"]}
    assert roles == {"etiq standard": "family", "etiq mvt_bras": "family", "etiq mvt_jambe": "family",
                     "etiq activite": "derived"}
    s = c.get(f"/api/rec/{rid}/series", params={"cols": "5,9"}).json()
    assert s["5"][1] == 0 and s["9"][3] is None   # zero stays 0, NaN string -> null (gap)

    lab = c.get(f"/api/rec/{rid}/labels", params={"ts_col": 0, "family": "12,13,14", "derived": "15"}).json()
    assert [(o["start_row"], o["end_row"]) for o in lab["overlap"]] == [(2, 4)]
    assert [(o["start_row"], o["end_row"]) for o in lab["unlabeled"]] == []  # 'etiq standard' is 'inactif'

    q = c.get(f"/api/rec/{rid}/quality", params={"ts_col": 0}).json()
    gyr = next(g for g in q["groups"] if g["modality"] == "GYR")
    assert gyr["runs"]["missing_or_nonnumeric"]["runs"][0]["start_row"] == 3
    acc = next(g for g in q["groups"] if g["modality"] == "ACC")
    assert acc["runs"]["all_axes_zero"]["runs"][0]["start_row"] == 1

    row = c.get(f"/api/rec/{rid}/row/3").json()
    assert row["cells"][9]["raw"] == "NaN" and row["cells"][0]["raw"] == "1643189820060.0"

    prof = {"schema_version": 1, "alignment": {"a": 1.0, "b": -2.5, "status": "unverified", "anchors": []}}
    assert c.put(f"/api/rec/{rid}/profile", json=prof).status_code == 200
    back = c.get(f"/api/rec/{rid}/profile").json()["profile"]
    assert back["alignment"] == prof["alignment"] and back["recording_id"] == rid

    n = c.post(f"/api/rec/{rid}/notes", json={"text": "check", "row_start": 2, "row_end": 5}).json()
    assert c.get(f"/api/rec/{rid}/notes").json()[0]["id"] == n["id"]

    ex = c.post(f"/api/rec/{rid}/export", json={
        "row_start": 2, "row_end": 5,
        "timestamp": {"column": 0, "unit": "ms", "reference_value": 1643189820000.0},
        "alignment": prof["alignment"]}).json()
    folder = Path(ex["folder"])
    lines = (folder / "rows.csv").read_text(encoding="utf-8").splitlines()
    src_lines = src_before.decode().splitlines()
    assert lines[1] == "2,0.04," + src_lines[3]        # verbatim source line after derived cols
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert man["source_rows_verbatim"] is True
    assert (raw / "part1_data.csv").read_bytes() == src_before   # source untouched

    assert c.get("/api/browse", params={"path": str(cfg.output_dir.parent)}).status_code == 403


def test_verdicts_upsert_history_and_validation(env):
    cfg, raw = env
    src_before = (raw / "part1_data.csv").read_bytes()
    c = TestClient(create_app(cfg))
    rid = c.get("/api/recordings").json()[0]["id"]
    base = {"column": 13, "column_name": "etiq mvt_bras", "start_row": 2, "end_row": 5, "label_value": "Lever_Bras"}
    v1 = c.put(f"/api/rec/{rid}/verdicts", json={**base, "verdict": "agree", "observed": "", "comment": ""}).json()
    v2 = c.put(f"/api/rec/{rid}/verdicts", json={**base, "verdict": "disagree", "observed": "Debout_Assis",
                                                  "comment": "sits down"}).json()
    assert v2["id"] == v1["id"] and v2["verdict"] == "disagree" and v2["history"][0]["verdict"] == "agree"
    other = c.put(f"/api/rec/{rid}/verdicts", json={**base, "start_row": 0, "end_row": 2, "label_value": "",
                                                     "verdict": "unsure"}).json()
    assert len(c.get(f"/api/rec/{rid}/verdicts").json()) == 2
    assert c.put(f"/api/rec/{rid}/verdicts", json={**base, "verdict": "maybe"}).status_code == 400
    assert c.put(f"/api/rec/{rid}/verdicts", json={"verdict": "agree"}).status_code == 400
    c.delete(f"/api/rec/{rid}/verdicts/{other['id']}")
    assert [x["id"] for x in c.get(f"/api/rec/{rid}/verdicts").json()] == [v1["id"]]
    assert (raw / "part1_data.csv").read_bytes() == src_before


def test_video_range_request(env, tmp_path):
    cfg, raw = env
    (raw / "part1_video.mp4").write_bytes(bytes(range(256)) * 4)
    c = TestClient(create_app(cfg))
    rid = c.get("/api/recordings").json()[0]["id"]
    r = c.get(f"/api/rec/{rid}/video", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == bytes(range(10, 20))


def test_parse_groups_handles_reordered_markers():
    hdr = ["ts_SQ", "x_sq_G_genou", "y_sq_G_genou", "z_sq_G_genou", "x_sq_D_main", "y_sq_D_main", "z_sq_D_main"]
    g = {x.placement: x.axes for x in parse_groups(hdr)}
    assert g == {"G_genou": {"x": 1, "y": 2, "z": 3}, "D_main": {"x": 4, "y": 5, "z": 6}}


def test_mp4_reader_rejects_non_mp4(tmp_path):
    p = tmp_path / "x.mp4"
    p.write_bytes(b"\x00\x00\x00\x08free")
    assert read_mp4_metadata(p)["ok"] is False
