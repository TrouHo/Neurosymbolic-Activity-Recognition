import numpy as np

from emob import config as C
from emob.metrics import (binary_f1, confusion, edit_score, f1_at_k, macro_f1, rule_violations,
                          segment_metrics, segment_metrics_concat, segments)

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


def test_segment_metrics_score_each_supervised_interval():
    # predictions in the unknown gap are ignored; the two sides are separate segments, not merged
    gt = np.array([0, 0, I, I, 0, 0])
    pred = np.array([0, 0, 1, 1, 0, 0])
    m = segment_metrics(gt, pred)
    assert m["edit"] == 100.0 and m["f1@50"] == 100.0
    assert (m["n_seg_gt"], m["n_intervals"]) == (2, 2)
    assert segment_metrics_concat(gt, pred)["n_seg_gt"] == 1          # old convention merged them


def test_segment_metrics_pool_counts_over_intervals():
    # interval 1: gt 0,0 | pred 0,1  -> segments [0] vs [0,1]: edit distance 1 of 2; F1@50: tp 1 fp 1 fn 0
    # interval 2: gt 1,1,1,1 | pred 1,1,1,1 -> distance 0 of 1; tp 1
    gt = np.array([0, 0, I, 1, 1, 1, 1])
    pred = np.array([0, 1, 2, 1, 1, 1, 1])
    m = segment_metrics(gt, pred)
    assert abs(m["edit"] - 100 * (1 - 1 / 3)) < 1e-9                  # micro: (1 + 0) / (2 + 1)
    assert abs(m["f1@50"] - 100 * 4 / 5) < 1e-9                       # 2 tp, 1 fp, 0 fn
    # a segment cannot match across the gap: one long prediction over both sides
    gt2 = np.array([0, 0, I, 0, 0])
    m2 = segment_metrics(gt2, np.zeros(5, int))
    assert m2["f1@50"] == 100.0 and m2["n_seg_pred"] == 2


def test_segment_metrics_all_unknown():
    m = segment_metrics(np.array([I, I]), np.array([0, 1]))
    assert m["edit"] == 100.0 and m["n_intervals"] == 0


def test_rule_violations_a1_a2_a4_a5():
    from emob.data.labels import L1_INDEX, L2_INDEX, L2_TO_L1_IDX, S_INDEX
    arm, sit_down = L2_INDEX["Autre_Mouvement_Bras"], L2_INDEX["Debout_Assis"]
    l2 = np.array([arm, arm, sit_down, sit_down, arm, sit_down, sit_down, I])
    l1 = L2_TO_L1_IDX[np.where(l2 == I, 0, l2)]
    l1[1] = L1_INDEX["WALK"]                                              # one A1 violation
    l1[-1] = I
    st, si = S_INDEX["Debout"], S_INDEX["Assis"]
    s = np.array([st, st, I, I, si, I, I, I])                              # 2nd onset starts from sitting
    v = rule_violations(l2=l2, l1=l1, s=s)
    assert (v["A1"]["viol"], v["A1"]["n"]) == (1, 7)
    assert (v["A2"]["viol"], v["A2"]["n"]) == (1, 2)
    w = np.array([1, 1, 0, 0, 1, 1, I, 1])
    s2 = np.array([st, si, si, st, S_INDEX["Agenou"], I, si, si])
    v = rule_violations(s=s2, w=w)
    assert (v["A4"]["viol"], v["A4"]["n"]) == (2, 4)                      # steps 1 and 7; step 5 unknown
    assert (v["A5"]["viol"], v["A5"]["n"]) == (1, 4)


def test_rule_violations_a3_a6_a7():
    from emob.data.labels import L2_INDEX
    from emob.rules_spec import MAX_SPECIFIC_STEPS
    s = np.zeros(100, int); s[30:] = 1; s[70:] = 2                       # changes at 30 and 70
    t = np.zeros(100, int); t[25:28] = 1                                  # T near the first change only
    v = rule_violations(s=s, t=t)
    assert (v["A3"]["viol"], v["A3"]["n"]) == (1, 2)
    w = np.zeros(100, int); w[10:15] = 1; w[20:40] = 1; w[95:] = 1       # short run, long run, run at the edge
    v = rule_violations(w=w)
    assert (v["A6"]["viol"], v["A6"]["n"]) == (1, 2)
    up, walk = L2_INDEX["Lever_Bras"], L2_INDEX["Marcher"]
    l2 = np.full(200, walk); l2[0:60] = up; l2[100:120] = up             # one run of 60 > 50 steps
    v = rule_violations(l2=l2)
    assert (v["A7"]["viol"], v["A7"]["n"], v["A7"]["excess_steps"]) == (1, 2, 60 - MAX_SPECIFIC_STEPS)
