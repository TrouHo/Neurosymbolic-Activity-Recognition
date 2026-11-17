"""Turn per-step predictions of one test participant into the metric dictionary."""
from __future__ import annotations

import numpy as np

from emob import config as C
from emob.data.labels import L2_INDEX, L2_TO_L1_IDX, S_INDEX
from emob.metrics import confusion, macro_f1, per_class_f1, rule_violations, segment_metrics

AUTRE = [L2_INDEX[c] for c in ("Autre_Mouvement_Bras", "Autre_Mouvement_Jambe", "Autre_Transition")]
NON_AUTRE = [i for i in range(len(C.L2_CLASSES)) if i not in AUTRE]
TRANSITION_FROM = {L2_INDEX[t]: S_INDEX[C.transition_states(t)[0]] for t in C.POSTURE_TRANSITIONS}
SIT_LIE = [S_INDEX["Assis"], S_INDEX["Couche"]]


def evaluate(pred: dict[str, np.ndarray], gt: dict[str, np.ndarray]) -> dict:
    """pred: argmax per head ('l1', 'l2', optional 's'); gt: 'l1', 'l2', 's', 's_all'."""
    cm2 = confusion(gt["l2"], pred["l2"], len(C.L2_CLASSES))
    cm1 = confusion(gt["l1"], pred["l1"], len(C.L1_CLASSES))
    out = {
        "l2_macro_f1": macro_f1(cm2),
        "l2_macro_f1_no_autre": macro_f1(cm2, NON_AUTRE),
        "l2_per_class_f1": {c: (None if np.isnan(f) else float(f))
                            for c, f in zip(C.L2_CLASSES, per_class_f1(cm2))},
        "l2_confusion": cm2.tolist(),
        "l1_macro_f1": macro_f1(cm1),
        "l1_confusion": cm1.tolist(),
        "l2_segments": segment_metrics(gt["l2"], pred["l2"]),
        "n_steps_l2": int((gt["l2"] != C.IGNORE).sum()),
    }
    if "s" in pred:
        for name in ("s_all", "s"):
            cm = confusion(gt[name], pred["s"], len(C.S_CLASSES))
            out[f"S_macro_f1_vs_{name}"] = macro_f1(cm)
            out[f"S_confusion_vs_{name}"] = cm.tolist()
    out["rule_violations_pred"] = rule_violations(
        pred["l1"], pred["l2"], pred.get("s"), walk_idx=L2_INDEX["Marcher"], l2_to_l1=L2_TO_L1_IDX,
        sit_lie_idx=SIT_LIE, transition_from=TRANSITION_FROM, min_walk_steps=10)
    return out
