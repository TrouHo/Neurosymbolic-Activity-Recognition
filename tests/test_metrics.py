import numpy as np

from emob import config as C
from emob.metrics import (confusion, edit_score, f1_at_k, macro_f1, rule_violations, segment_metrics,
                          segments)

I = C.IGNORE


def test_macro_f1_only_over_present_classes():
    gt = np.array([0, 0, 1, 1, I])
    pred = np.array([0, 0, 1, 2, 2])
    cm = confusion(gt, pred, 3)
    assert cm.sum() == 4                                  # ignored step excluded
    # class 0: F1 = 1; class 1: tp=1 fn=1 -> 2/3; class 2 absent in GT -> excluded
    assert abs(macro_f1(cm) - (1 + 2 / 3) / 2) < 1e-9
    assert abs(macro_f1(cm, classes=[0]) - 1.0) < 1e-9


def test_segments_and_edit():
    assert segments(np.array([1, 1, 2, 2, 2, 1])) == [(1, 0, 2), (2, 2, 5), (1, 5, 6)]
    gt = np.array([1, 1, 2, 2, 1, 1])
    assert edit_score(gt, gt) == 100.0
    pred = np.array([1, 1, 2, 1, 2, 1])                  # segments 1,2,1,2,1 vs 1,2,1 -> distance 2
    assert abs(edit_score(gt, pred) - 100 * (1 - 2 / 5)) < 1e-9


def test_f1_at_k():
    gt = np.array([0] * 10 + [1] * 10)
    assert f1_at_k(gt, gt, 0.5) == 100.0
    pred = np.array([0] * 4 + [1] * 16)                  # IoU(0)=0.4, IoU(1)=10/16
    assert f1_at_k(gt, pred, 0.5) == 50.0                # only class-1 segment matches
    assert f1_at_k(gt, pred, 0.25) == 100.0


def test_segment_metrics_drop_ignored_steps():
    gt = np.array([0, 0, I, I, 0, 0])
    pred = np.array([0, 0, 1, 1, 0, 0])
    m = segment_metrics(gt, pred)
    assert m["edit"] == 100.0 and m["n_seg_gt"] == 1


def test_rule_violations():
    l2_to_l1 = np.array([0, 1, 2])                       # toy: class 1 = walk, class 2 = transition from state 0
    l2 = np.array([1, 1, 1, 0, 2, 2, 1])
    l1 = np.array([1, 1, 1, 0, 2, 0, 1])                 # one A1 violation (step 5)
    s = np.array([5, 5, 5, 0, 0, 0, 3])                  # state 5 = sitting in this toy
    v = rule_violations(l1, l2, s, walk_idx=1, l2_to_l1=l2_to_l1, sit_lie_idx=[5],
                        transition_from={2: 0}, min_walk_steps=3)
    assert abs(v["A1"] - 1 / 7) < 1e-9
    assert v["A3"] == 0.5                                # walk segments of length 3 and 1
    assert v["A4"] == 0.75                               # 3 of 4 walking steps sitting
    assert v["A2"] == 0.0                                # onset at 4, state at 3 is 0 = X
