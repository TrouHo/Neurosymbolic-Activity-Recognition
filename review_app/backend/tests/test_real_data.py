"""REAL-DATA checks on two provided recordings (skipped when the files are not present).

Expected numbers come from the independent read-only inventory run on 2026-10-23 (plain csv module,
not this app's loader) and from the 2026-10-21 audit log.
"""
import time
from pathlib import Path

import pytest

from emob_review.csvload import load_csv, timestamp_report
from emob_review.labels import label_overview, summarize, segments_for
from emob_review.mp4meta import read_mp4_metadata
from emob_review.quality import group_flags

RAW = Path(r"D:\Projects\EmoB\data\raw")
P2 = RAW / "capture_2021-05-31" / "part2_data.csv"
U3 = RAW / "capture_2022-01-26" / "user3-22-06-17.csv"
pytestmark = pytest.mark.skipif(not (P2.exists() and U3.exists()), reason="real data not available")


@pytest.fixture(scope="module")
def p2():
    t = time.perf_counter()
    lc = load_csv(P2)
    print(f"\npart2 load {time.perf_counter() - t:.1f}s")
    return lc


@pytest.fixture(scope="module")
def u3():
    return load_csv(U3)


def test_part2_shape_and_timing(p2):
    assert (p2.n_rows, len(p2.header)) == (45001, 116)
    assert p2.lines_match_records and not p2.bad_rows
    rep = timestamp_report(p2, p2.header.index("ts_SQ"), "ms")
    assert (rep.first, rep.last) == (1622464500000.0, 1622465400000.0)
    assert rep.median_step == 20 and rep.step_histogram == [(20.0, 45000)]
    assert rep.n_duplicates == rep.n_nonmonotonic == rep.n_invalid == rep.n_fractional == 0 and not rep.gaps
    assert abs(rep.effective_rate_hz - 50.0) < 1e-9


def test_part2_labels_match_inventory(p2):
    c = p2.header.index("etiq mvt_bras")
    ts = p2.numeric[0]
    summ = {d["value"]: d["rows"] for d in summarize(segments_for(p2.text[c], ts, 20.0))}
    assert summ == {"": 35513, "inactif": 5566, "Autre_Mouvement_Bras": 3017, "Baisser_Bras": 566, "Lever_Bras": 339}


def test_part2_sb_gyr_has_no_nonzero_data(p2):
    gyr = [g for g in p2.groups if g.source == "SB" and g.modality == "GYR"]
    assert len(gyr) == 5
    assert all(group_flags(p2, g.axes)["no_nonzero_data"] for g in gyr)


def test_user3_reordered_markers_and_nan_strings(u3):
    g = {(x.source, x.placement, x.modality): x for x in u3.groups}
    assert g[("sq", "D_main", "XYZ")].axes == {"x": 16, "y": 17, "z": 18}   # mapped by name, not position
    fl = group_flags(u3, g[("SB", "Main droite", "GYR")].axes)
    assert fl["rows_with_missing_or_nonnumeric"] == 35405 and fl["no_nonzero_data"]
    assert not group_flags(u3, g[("SB", "Main droite", "ACC")].axes)["no_nonzero_data"]


def test_user3_overview_runs(u3):
    fam = [u3.header.index(h) for h in ("etiq standard", "etiq mvt_corps", "etiq mvt_bras", "etiq mvt_jambe")]
    der = [u3.header.index(h) for h in ("etiq categorie", "etiq activite")]
    ov = label_overview(u3, u3.numeric[0], 20.0, fam, der)
    unl = sum(o["end_row"] - o["start_row"] for o in ov["unlabeled"])
    assert unl == 24366                       # rows where all four family columns are ""
    debut = [s for s in ov["columns"][fam[0]]["segments"] if s["value"] == "Debut"]
    assert [(s["start_row"], s["end_row"]) for s in debut] == [(5419, 5423)]


def test_video_metadata():
    m2 = read_mp4_metadata(RAW / "capture_2021-05-31" / "part2_video.mp4")["video_track"]
    # container states 10_000_000 / 333_333 = 30.00003 fps (reported as-is, not rounded)
    assert m2["sample_count"] == 12636 and abs(m2["nominal_fps"] - 30.0) < 1e-4 and m2["codec"] == "avc1"
    m3 = read_mp4_metadata(RAW / "capture_2022-01-26" / "Measurement_user3.mp4")["video_track"]
    assert m3["sample_count"] == 29508 and m3["nominal_fps"] == 50.0 and m3["zero_duration_samples"] == 1
