"""Evaluation metrics: frame-level F1, segment metrics, rule-violation rates.

Conventions (documented in knowledge_base/02_design_rationale.md):
- Frame macro-F1 is averaged over the classes present in the ground truth of the
  evaluated participant (absent classes have no defined recall).
- Segment metrics are computed on the sequence restricted to supervised steps.
"""
from __future__ import annotations

import numpy as np

from emob import config as C


# ----------------------------------------------------------------------------- frame level
def confusion(gt: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    ok = gt != C.IGNORE
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (gt[ok], pred[ok]), 1)
    return cm


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.full(len(tp), np.nan), where=denom > 0)


def binary_f1(gt: np.ndarray, pred: np.ndarray) -> float | None:
    """F1 of the positive class (1) over supervised steps; None if neither has a positive."""
    ok = gt != C.IGNORE
    g, p = gt[ok] == 1, pred[ok] == 1
    tp, fp, fn = int((g & p).sum()), int((~g & p).sum()), int((g & ~p).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None


def macro_f1(cm: np.ndarray, classes: list[int] | None = None) -> float:
    """Mean F1 over classes present in the ground truth (optionally restricted to `classes`)."""
    present = np.flatnonzero(cm.sum(1) > 0)
    if classes is not None:
        present = np.intersect1d(present, classes)
    if len(present) == 0:
        return float("nan")
    f1 = per_class_f1(cm)
    return float(np.nanmean(np.where(np.isnan(f1[present]), 0.0, f1[present])))


# ----------------------------------------------------------------------------- segments
def segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """(label, start, end_exclusive) runs of a dense label sequence."""
    if len(labels) == 0:
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(labels)]])
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def _levenshtein(a: list[int], b: list[int]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def edit_score(gt: np.ndarray, pred: np.ndarray) -> float:
    """Segmental edit score in % (Lea et al. 2017): 100 * (1 - Levenshtein / max length)."""
    g = [s[0] for s in segments(gt)]
    p = [s[0] for s in segments(pred)]
    if max(len(g), len(p)) == 0:
        return 100.0
    return 100.0 * (1 - _levenshtein(g, p) / max(len(g), len(p)))


def f1_at_k(gt: np.ndarray, pred: np.ndarray, k: float) -> float:
    """Segmental F1@k in % (Farha & Gall 2019): a predicted segment is a true positive if its IoU
    with a not-yet-matched ground-truth segment of the same class is >= k."""
    gs, ps = segments(gt), segments(pred)
    used = np.zeros(len(gs), dtype=bool)
    tp = 0
    for lab, s, e in ps:
        best, best_j = 0.0, -1
        for j, (gl, gs_, ge) in enumerate(gs):
            if gl != lab:
                continue
            inter = max(0, min(e, ge) - max(s, gs_))
            union = max(e, ge) - min(s, gs_)
            iou = inter / union if union > 0 else 0.0
            if iou > best:
                best, best_j = iou, j
        if best >= k and not used[best_j]:
            tp += 1
            used[best_j] = True
    fp, fn = len(ps) - tp, len(gs) - tp
    denom = 2 * tp + fp + fn
    return 100.0 * 2 * tp / denom if denom else 100.0


def segment_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    ok = gt != C.IGNORE
    g, p = gt[ok], pred[ok]
    return {"edit": edit_score(g, p), "f1@10": f1_at_k(g, p, 0.10), "f1@25": f1_at_k(g, p, 0.25),
            "f1@50": f1_at_k(g, p, 0.50), "n_seg_gt": len(segments(g)), "n_seg_pred": len(segments(p))}


# ----------------------------------------------------------------------------- rules A1–A7
def _rate(viol: int, n: int, **extra) -> dict:
    return {"viol": int(viol), "n": int(n), "rate": (viol / n if n else None), **extra}


def rule_violations(l2=None, l1=None, s=None, w=None, t=None) -> dict:
    """Violations of rules A1–A7 in label sequences (predictions or ground truth) at 0.1-s steps.

    Same definitions as the rule losses (emob.rules_loss); C.IGNORE = unknown, never a violation.
    Heads that are None are skipped (e.g. B1 has no S, W, T).
      A1  steps where L1 is not the parent of L2                   / steps with both known
      A2  onsets of a posture transition X_Y where S(t-1) != X     / onsets with S(t-1) known
      A3  posture changes with no T = 1 within +-1 s               / changes (10 <= t <= T-11)
      A4  walking steps with S in {sitting, lying}                 / walking steps with S known
      A5  walking steps with S in {kneeling, squatting}            / same
      A6  walking runs shorter than 1 s                            / runs bounded by W = 0 inside the sequence
      A7  runs of a specific class longer than 5 s                 / runs of the 16 specific classes
    """
    from emob.data.labels import L2_TO_L1_IDX
    from emob.rules_spec import (A3_HALF_WINDOW, KNEEL_SQUAT, MAX_SPECIFIC_STEPS, MIN_WALK_STEPS,
                                 SIT_LIE, SPECIFIC_IDX, TRANSITION_FROM, TRANSITION_IDX)
    I = C.IGNORE
    out = {}
    if l2 is not None and l1 is not None:
        k = (l2 != I) & (l1 != I)
        out["A1"] = _rate((L2_TO_L1_IDX[l2[k]] != l1[k]).sum(), k.sum())
    if l2 is not None and s is not None:
        frm = dict(zip(TRANSITION_IDX, TRANSITION_FROM))
        n = v = 0
        for tt in range(1, len(l2)):
            c = int(l2[tt])
            if c in frm and l2[tt - 1] != c and s[tt - 1] != I:
                n += 1
                v += int(s[tt - 1] != frm[c])
        out["A2"] = _rate(v, n)
    if s is not None and t is not None:
        T, h = len(s), A3_HALF_WINDOW
        n = v = 0
        for tt in range(h, T - h):
            if s[tt] != I and s[tt - 1] != I and s[tt] != s[tt - 1]:
                n += 1
                v += int(not (t[tt - h:tt + h + 1] == 1).any())
        out["A3"] = _rate(v, n)
    if w is not None and s is not None:
        walk = (w == 1) & (s != I)
        out["A4"] = _rate(np.isin(s[walk], SIT_LIE).sum(), walk.sum())
        out["A5"] = _rate(np.isin(s[walk], KNEEL_SQUAT).sum(), walk.sum())
    if w is not None:
        runs = [(a, b) for lab, a, b in segments(np.asarray(w)) if lab == 1
                and a > 0 and b < len(w) and w[a - 1] == 0 and w[b] == 0]
        out["A6"] = _rate(sum(b - a < MIN_WALK_STEPS for a, b in runs), len(runs))
    if l2 is not None:
        runs = [(a, b) for lab, a, b in segments(np.asarray(l2)) if lab in SPECIFIC_IDX]
        long_ = [b - a for a, b in runs if b - a > MAX_SPECIFIC_STEPS]
        out["A7"] = _rate(len(long_), len(runs), excess_steps=int(sum(x - MAX_SPECIFIC_STEPS for x in long_)))
    return out
