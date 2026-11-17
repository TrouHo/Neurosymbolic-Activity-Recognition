"""Gate G0 on the real 2022 data (skipped if the processed files are absent).

Expected values were recomputed on 2026-11-05 with the definitions in emob/data/labels.py.
Hierarchy (195 870 rows) and walking (261/263) match the earlier analysis exactly; the
posture-state-machine count and the A4 figure differ from the earlier, non-reproducible
script (see knowledge_base/04_changelog.md, 2026-11-05).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from emob import config as C

V1 = C.PROJECT / "data" / "processed" / "v1"       # historical pilot build (emob/data/build_v1.py)
REPORT = V1 / "phase0_report.json"
pytestmark = pytest.mark.skipif(not REPORT.exists(), reason="run `python -m emob.data.build_v1` first")


@pytest.fixture(scope="module")
def report():
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_hierarchy_holds_on_every_labelled_row(report):
    t = report["totals"]
    assert t["hierarchy_rows"] == 195870 and t["hierarchy_agree"] == 195870


def test_walk_minimum_duration(report):
    t = report["totals"]
    assert (t["walk_segments"], t["walk_segments_ge_1s"]) == (263, 261)


def test_posture_state_machine_counts(report):
    t = report["totals"]
    assert (t["sm_consistent_pairs"], t["sm_inconsistent_pairs"], t["sm_pairs_with_other"]) == (584, 18, 80)


def test_no_walking_in_sitting_or_lying_state(report):
    t = report["totals"]
    assert t["walking_rows_sitting_or_lying"] == 0


def test_sensor_availability(report):
    """SB hand/belt ACC: degenerate (no signal) for user1, 2, 4; belt absent for user5;
    otherwise < 9 % unusable. Phone ACC/GYR usable for everyone (< 2 % unusable)."""
    degenerate = {"user1", "user2", "user4"}
    for p in report["participants"]:
        u, d = p["unusable_in_labelled_region_pct"], p["degenerate_sensor"]
        if p["participant"] in degenerate:
            assert d["hand"] and d["belt"] and u["hand"] == 100.0 and u["belt"] == 100.0
        elif p["participant"] == "user5":
            assert u["belt"] == 100.0 and u["hand"] < 9
        else:
            assert u["hand"] < 9 and u["belt"] < 9
        assert not d["phone_acc"] and not d["phone_gyr"]
        assert u["phone_acc"] < 2 and u["phone_gyr"] < 2


def test_centre_label_almost_always_equals_majority(report):
    t = report["totals"]
    assert t["centre_differs_from_majority"] / t["centre_supervised_steps"] < 0.001


def test_npz_files_are_consistent():
    for p in [f"user{i}" for i in range(1, 9)]:
        d = np.load(V1 / f"{p}.npz")
        T = len(d["l2"])
        assert d["acc50"].shape == (d["ts"].shape[0], 6)
        assert T == d["ts"].shape[0] // C.STEP
        for k in ("l1", "s", "s_all", "annotated"):
            assert len(d[k]) == T
        # L1 is defined exactly where L2 is
        assert ((d["l1"] == C.IGNORE) == (d["l2"] == C.IGNORE)).all()
        # S masked version is a subset of the unmasked one
        assert ((d["s"] == C.IGNORE) | (d["s"] == d["s_all"])).all()
        # supervised L2 steps are annotated steps
        assert d["annotated"][d["l2"] != C.IGNORE].all()
