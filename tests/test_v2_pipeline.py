"""Unit tests of the pipeline v2 steps on small synthetic inputs with known answers."""
import numpy as np
import pytest

from emob import config as C
from emob.data import alignment, anomalies, sync
from emob.data.labels import S_INDEX, mask_posture_after, posture_states, transition_targets, walk_targets


# ---- S2 sensor alignment
def test_shift_rows_late_device_moves_earlier():
    a = np.arange(10.0)
    out = sync.shift_rows(a, 3, np.nan)           # device late by 3 samples
    assert out[0] == 3 and np.isnan(out[-3:]).all()
    out = sync.shift_rows(a, -2, np.nan)          # device early by 2 samples
    assert np.isnan(out[:2]).all() and out[2] == 0 and out[-1] == 7


def test_align_block_respects_valid_windows_and_unusable_devices():
    xyz = np.ones((100, 3)); ok = np.ones(100, bool)
    x, a = sync.align_block(xyz, ok, dict(usable=True, shift_s=0.1, valid_rows=[[20, 40]]))
    assert a.sum() == 20 and a[20:40].all() and np.isnan(x[~a]).all()
    x, a = sync.align_block(xyz, ok, dict(usable=False, shift_s=0.0, valid_rows=None))
    assert not a.any() and np.isnan(x).all()


def test_device_plan_states():
    off = {"participants": {"u": {"devices": {
        "a": {"signal": False},
        "b": {"signal": True, "gate_passed": True, "apply_shift_s": 0.5},
        "c": {"signal": True, "gate_passed": False, "fallback": {"usable": True, "apply_shift_s": -1.0,
                                                               "valid_rows": [[0, 5]], "n_verified": 9, "n_windows_with_data": 11}},
        "d": {"signal": True, "gate_passed": False, "fallback": {"usable": False}}}}}}
    assert not sync.device_plan(off, "u", "a")["usable"]
    assert sync.device_plan(off, "u", "b")["shift_s"] == 0.5
    p = sync.device_plan(off, "u", "c")
    assert p["usable"] and p["valid_rows"] == [[0, 5]]
    assert not sync.device_plan(off, "u", "d")["usable"]


# ---- S3 label alignment: d < 0 means labels late -> moved earlier
def test_shift_labels_sign():
    a = np.array(["", "", "", "X", "X", ""], dtype=object)
    early = alignment.shift_labels(a, -2 / C.FS)
    assert list(early) == ["", "X", "X", "", "", ""]
    late = alignment.shift_labels(a, 1 / C.FS)
    assert list(late) == ["", "", "", "", "X", "X"]
    m = alignment.shift_labels(np.array([False, True, False]), 1 / C.FS, False)
    assert list(m) == [False, False, True]


# ---- S4 anomaly masks
def test_anomaly_masks_only_stuck_activite():
    act = np.array(["L", "L", "L", "Z", "L", "L"], dtype=object)
    runs = [dict(user="u", column=C.BODY_COL, label="L", masked_rows=[1, 5])]
    fam, am = anomalies.masks(runs, "u", 6, act)
    assert fam[C.BODY_COL].tolist() == [False, True, True, True, True, False]
    assert am.tolist() == [False, True, True, False, True, False]     # "Z" is not the stuck label
    assert not fam[C.LEG_COL].any()


# ---- S5 targets
def test_walk_and_transition_targets():
    leg = np.array(["", "inactif", "Marcher", "Lever_Jambe"], dtype=object)
    assert walk_targets(leg).tolist() == [C.IGNORE, 0, 1, 0]
    body = np.array(["", "inactif", "Debout_Assis", "Autre_Transition"], dtype=object)
    assert transition_targets(body).tolist() == [C.IGNORE, 0, 1, 1]
    with pytest.raises(ValueError):
        walk_targets(np.array(["Debout_Assis"], dtype=object))


def test_posture_unknown_after_masked_body_until_next_transition():
    body = np.array(["Debout_Assis"] * 2 + ["inactif"] * 6 + ["Assis_Debout"] * 2 + ["inactif"] * 2, dtype=object)
    ann = np.ones(len(body), bool)
    ps = posture_states(body, ann)
    assert ps.state_all[3] == S_INDEX["Assis"]
    masked = np.zeros(len(body), bool); masked[4:6] = True
    s = mask_posture_after(ps.state_all, masked, ps.transitions)
    assert s[3] == S_INDEX["Assis"]                    # before the masked part: unchanged
    assert (s[4:8] == C.IGNORE).all()                  # from the masked part to the next transition
    assert s[10] == S_INDEX["Debout"]                  # after the next transition: known again
