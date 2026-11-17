"""Unit tests on synthetic sequences (no real data needed)."""
import numpy as np
import pytest

from emob import config as C
from emob.data.labels import (L2_INDEX, S_INDEX, annotated_mask, centre_vs_majority, encode_l2,
                              l1_from_l2, posture_states, runs, step_centres)
from emob.data.rules import hierarchy_agreement, lookback_check, walk_segments, walking_state_counts

I = C.IGNORE


def seq(*parts):
    """seq(("a", 3), ("", 2)) -> ['a','a','a','','']"""
    out = []
    for v, k in parts:
        out += [v] * k
    return np.array(out, dtype=object)


def test_runs_skips_empty_and_splits_on_change():
    v = seq(("", 2), ("a", 3), ("b", 1), ("", 1), ("a", 2))
    assert runs(v) == [("a", 2, 5), ("b", 5, 6), ("a", 7, 9)]


def test_l1_mapping_covers_every_l2_class():
    assert set(C.L2_TO_L1) == set(C.L2_CLASSES)
    assert set(C.L2_TO_L1.values()) == set(C.L1_CLASSES)
    l2 = np.array([L2_INDEX["Marcher"], I, L2_INDEX["Debout_Assis"]])
    l1 = l1_from_l2(l2)
    assert C.L1_CLASSES[l1[0]] == "WALK" and l1[1] == I and C.L1_CLASSES[l1[2]] == "SIT_TRANSITION"


def test_encode_l2_ignores_empty_and_markers_and_rejects_unknown():
    out = encode_l2(np.array(["", "Debut", "Fin", "Marcher"], dtype=object))
    assert list(out[:3]) == [I, I, I] and out[3] == L2_INDEX["Marcher"]
    with pytest.raises(ValueError):
        encode_l2(np.array(["inactif"], dtype=object))


def test_posture_consistent_sequence():
    body = seq(("inactif", 3), ("Debout_Assis", 2), ("inactif", 4), ("Assis_Debout", 2), ("inactif", 3))
    r = posture_states(body, np.ones(len(body), bool))
    D, A = S_INDEX["Debout"], S_INDEX["Assis"]
    assert list(r.state) == [D] * 3 + [I] * 2 + [A] * 4 + [I] * 2 + [D] * 3
    assert (r.consistent_pairs, r.inconsistent_pairs) == (1, 0)


def test_posture_inconsistent_pair_is_masked():
    body = seq(("Debout_Assis", 2), ("inactif", 3), ("Debout_Penche", 2))  # Assis != Debout
    r = posture_states(body, np.ones(len(body), bool))
    assert (r.state[2:5] == I).all()
    assert (r.consistent_pairs, r.inconsistent_pairs) == (0, 1)


def test_posture_after_other_transition_is_masked_and_before_is_kept():
    body = seq(("Debout_Assis", 2), ("inactif", 2), ("Autre_Transition", 2), ("inactif", 3), ("Debout_Penche", 1))
    r = posture_states(body, np.ones(len(body), bool))
    assert (r.state[2:4] == S_INDEX["Assis"]).all()   # X_Y followed by Autre -> Y (DECISION)
    assert (r.state[6:9] == I).all()                  # after Autre -> masked
    assert r.pairs_with_other == 2 and r.consistent_pairs == 0


def test_posture_masked_on_unannotated_rows_but_kept_in_state_all():
    body = seq(("Debout_Assis", 2), ("", 3), ("Assis_Debout", 2))
    ann = body != ""
    r = posture_states(body, ann)
    assert (r.state[2:5] == I).all()
    assert (r.state_all[2:5] == S_INDEX["Assis"]).all()


def test_step_centres_and_majority():
    assert list(step_centres(12)) == [2, 7]          # 12 samples -> 2 complete steps
    l2 = np.array([0, 0, 0, 1, 1,   3, 3, 2, 3, 3])
    n_sup, differ = centre_vs_majority(l2)
    assert n_sup == 2 and differ == 1                 # 2nd step: centre = 2, majority = 3


def test_annotated_mask():
    fam = {c: np.array(["", "x", ""], dtype=object) for c in C.FAMILY_COLS}
    fam[C.FAMILY_COLS[0]] = np.array(["", "", "y"], dtype=object)
    assert list(annotated_mask(fam)) == [False, True, True]


def test_rule_checks():
    fams = {c: seq(("", 4)) for c in C.FAMILY_COLS}
    fams["etiq mvt_jambe"] = seq(("Marcher", 2), ("inactif", 2))
    act = seq(("Marcher", 2), ("", 2))
    cat = seq(("mvt_jambe", 2), ("", 2))
    assert hierarchy_agreement(act, cat, fams) == (2, 2)
    leg = seq(("Marcher", 60), ("inactif", 5), ("Marcher", 20))   # 1.2 s and 0.4 s at 50 Hz
    assert walk_segments(leg) == (2, 1)
    state = np.array([S_INDEX["Assis"]] * 10 + [I] * 75)
    assert walking_state_counts(leg, state)["walking_rows_sitting_or_lying"] == 10


def test_lookback_counts():
    body = seq(("inactif", 50), ("Debout_Assis", 10), ("inactif", 50))
    r = posture_states(body, np.ones(len(body), bool))
    out = lookback_check(r.transitions, r.state, [0.1, 2.0])
    assert out[0.1]["equals_X"] == 1 and out[2.0]["out_of_range"] == 1
