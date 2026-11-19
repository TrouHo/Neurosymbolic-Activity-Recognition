"""Per-step rule violations v_t of Eq. (3.1)-(3.7) on predicted probabilities, in numpy.

Same formulas as emob/rules_loss.py (which needs torch); used only to draw figures. Checked against
rules_loss in tests/test_report_figures.py where torch is available.
"""
from __future__ import annotations

import numpy as np

from emob import config as C
from emob.data.labels import L2_TO_L1_IDX
from emob.rules_spec import (A3_HALF_WINDOW, KNEEL_SQUAT, MAX_SPECIFIC_STEPS, MIN_WALK_STEPS, SIT_LIE, SPECIFIC_IDX,
                             TRANSITION_FROM, TRANSITION_IDX)


def softmax(z):
    z = z - z.max(0, keepdims=True)
    e = np.exp(z)
    return e / e.sum(0, keepdims=True)


def probabilities(logits: dict) -> dict:
    return {h: (1 / (1 + np.exp(-z[0])) if h in C.BINARY_HEADS else softmax(z)) for h, z in logits.items()}


def violations(p: dict) -> dict:
    T = p["l2"].shape[1]
    v = {}
    v["A1"] = -np.log(np.clip((p["l2"] * p["l1"][L2_TO_L1_IDX]).sum(0), 1e-6, None))
    pl = p["l2"][TRANSITION_IDX]
    a2 = np.zeros(T)
    a2[1:] = (pl[:, 1:] * (1 - pl[:, :-1]) * (1 - p["s"][TRANSITION_FROM, :-1])).sum(0)
    v["A2"] = a2
    ps = p["s"]
    change = np.zeros(T)
    change[1:] = 1 - (ps[:, 1:] * ps[:, :-1]).sum(0)
    h = A3_HALF_WINDOW
    pt = np.pad(p["t"], h, constant_values=0.0)
    near = np.array([pt[t:t + 2 * h + 1].max() for t in range(T)])
    v["A3"] = change * (1 - near)
    v["A4"] = p["w"] * p["s"][SIT_LIE].sum(0)
    v["A5"] = p["w"] * p["s"][KNEEL_SQUAT].sum(0)
    pw = p["w"]
    a6 = np.zeros(T)
    for t in range(1, T):
        run = 1.0
        for k in range(1, MIN_WALK_STEPS):
            if t + k >= T:
                break
            run *= pw[t + k - 1]
            a6[t] += (1 - pw[t - 1]) * run * (1 - pw[t + k])
    v["A6"] = a6
    n = MAX_SPECIFIC_STEPS + 1
    a7 = np.zeros(T)
    sp = p["l2"][SPECIFIC_IDX]
    for t in range(T - n + 1):
        a7[t] = sp[:, t:t + n].min(1).sum()
    v["A7"] = a7
    return v
