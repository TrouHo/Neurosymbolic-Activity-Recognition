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


# ----------------------------------------------------------------------------- rules on predictions
def rule_violations(l1: np.ndarray, l2: np.ndarray, s: np.ndarray | None,
                    walk_idx: int, l2_to_l1: np.ndarray, sit_lie_idx: list[int],
                    transition_from: dict[int, int], min_walk_steps: int = 10) -> dict:
    """Violation rates of rules A1–A4 in a predicted (argmax) sequence over all steps.

    A1: share of steps where L1 is not the parent of L2.
    A2: share of predicted posture-transition onsets whose predicted state one step before is not X.
    A3: share of predicted walking segments shorter than `min_walk_steps`.
    A4: share of predicted walking steps whose predicted state is sitting or lying.
    """
    out = {"A1": float(np.mean(l2_to_l1[l2] != l1))}
    walk_segs = [x for x in segments(l2) if x[0] == walk_idx]
    out["A3"] = float(np.mean([e - s_ < min_walk_steps for _, s_, e in walk_segs])) if walk_segs else 0.0
    if s is not None:
        walk = l2 == walk_idx
        out["A4"] = float(np.mean(np.isin(s[walk], sit_lie_idx))) if walk.any() else 0.0
        onsets = [(lab, st) for lab, st, _ in segments(l2) if lab in transition_from and st > 0]
        out["A2"] = float(np.mean([s[st - 1] != transition_from[lab] for lab, st in onsets])) if onsets else 0.0
    return out
