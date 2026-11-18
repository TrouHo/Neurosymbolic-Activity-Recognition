"""Turn the per-step predictions of one test participant into the metric dictionary.

Reference: reports/design/2026-11-17_model_architecture.md §6 (metrics).
- Main metric: frame-level macro-F1 of L2 over the classes present in the participant's labels.
- Secondary: L2 without the three "Autre_*" classes; L2 macro-F1 over the 10 named posture
  transitions (declared before phase 3, see emob.summarize.SECONDARY); per-class F1; confusion; L1 and S macro-F1;
  F1 of the positive class of W and T; segment metrics of L2 (edit, F1@10/25/50); violations of
  rules A1–A7 in the predictions and, for reference, in the participant's labels.
Heads the model does not have (B0: L2 only; B1: L2, L1) are skipped.
"""
from __future__ import annotations

import numpy as np

from emob import config as C
from emob.data.labels import L2_INDEX
from emob.metrics import binary_f1, confusion, macro_f1, per_class_f1, rule_violations, segment_metrics
from emob.rules_spec import TRANSITION_IDX

AUTRE = [L2_INDEX[c] for c in ("Autre_Mouvement_Bras", "Autre_Mouvement_Jambe", "Autre_Transition")]
NON_AUTRE = [i for i in range(len(C.L2_CLASSES)) if i not in AUTRE]


def decide(logits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """(K, T) logits per head -> (T,) labels: arg-max for softmax heads, logit > 0 (p > 0.5) for W, T."""
    return {h: (z[0] > 0).astype(np.int64) if h in C.BINARY_HEADS else z.argmax(0).astype(np.int64)
            for h, z in logits.items()}


def evaluate(pred: dict[str, np.ndarray], gt: dict[str, np.ndarray]) -> dict:
    """pred: {head: (T,) labels} for the heads of the model; gt: {head: (T,) labels, C.IGNORE = none}."""
    cm2 = confusion(gt["l2"], pred["l2"], len(C.L2_CLASSES))
    out = {
        "l2_macro_f1": macro_f1(cm2),
        "l2_macro_f1_no_autre": macro_f1(cm2, NON_AUTRE),
        "l2_macro_f1_transitions": macro_f1(cm2, TRANSITION_IDX),
        "l2_per_class_f1": {c: (None if np.isnan(f) else float(f)) for c, f in zip(C.L2_CLASSES, per_class_f1(cm2))},
        "l2_present_classes": [C.L2_CLASSES[i] for i in np.flatnonzero(cm2.sum(1) > 0)],
        "l2_confusion": cm2.tolist(),
        "l2_segments": segment_metrics(gt["l2"], pred["l2"]),
        "n_steps_l2": int((gt["l2"] != C.IGNORE).sum()),
    }
    if "l1" in pred:
        cm1 = confusion(gt["l1"], pred["l1"], len(C.L1_CLASSES))
        out["l1_macro_f1"] = macro_f1(cm1)
        out["l1_confusion"] = cm1.tolist()
    if "s" in pred:
        cms = confusion(gt["s"], pred["s"], len(C.S_CLASSES))
        out["s_macro_f1"] = macro_f1(cms)
        out["s_confusion"] = cms.tolist()
    for h in ("w", "t"):
        if h in pred:
            out[f"{h}_f1"] = binary_f1(gt[h], pred[h])
    out["rules_pred"] = rule_violations(**{h: pred.get(h) for h in C.HEADS})
    out["rules_labels"] = rule_violations(**{h: gt.get(h) for h in C.HEADS})
    return out
