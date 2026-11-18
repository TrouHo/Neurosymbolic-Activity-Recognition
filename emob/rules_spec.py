"""Constants of rules A1–A7 shared by the rule losses (rules_loss.py, needs torch) and the
rule-violation metric (metrics.py, numpy only). Approved 2026-11-16 (rules), 2026-11-18 (formulas).

The thresholds are those of the rule table checked on the labels (gate G, data v2):
A6 walking >= 1 s; A7 specific movements and named transitions <= 5 s; A3 window +-1 s.
"""
from __future__ import annotations

from emob import config as C
from emob.data.labels import L2_INDEX, S_INDEX

RULES = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
GROUP = {"A1": "hier", "A2": "hard", "A3": "soft", "A4": "hard", "A5": "hard", "A6": "hard", "A7": "hard"}
HEADS_NEEDED = {"A1": {"l2", "l1"}, "A2": {"l2", "s"}, "A3": {"s", "t"}, "A4": {"w", "s"},
                "A5": {"w", "s"}, "A6": {"w"}, "A7": {"l2"}}

MIN_WALK_STEPS = 10          # A6: walking >= 1 s
MAX_SPECIFIC_STEPS = 50      # A7: <= 5 s, so a run of 51 steps violates
A3_HALF_WINDOW = 10          # A3: +-1 s around a posture change

# A7 applies to the 16 specific classes: all L2 classes except walking and the three "other" classes
# (same list as SPECIFIC in emob/data/check.py, gate G).
SPECIFIC = [c for c in C.L2_CLASSES
            if c not in {"Marcher", "Autre_Mouvement_Bras", "Autre_Mouvement_Jambe", "Autre_Transition"}]
SPECIFIC_IDX = [L2_INDEX[c] for c in SPECIFIC]
# A2 applies to the 10 named posture transitions X_Y (Autre_Transition has no known start posture)
TRANSITION_IDX = [L2_INDEX[c] for c in C.POSTURE_TRANSITIONS]
TRANSITION_FROM = [S_INDEX[C.transition_states(c)[0]] for c in C.POSTURE_TRANSITIONS]
SIT_LIE = [S_INDEX["Assis"], S_INDEX["Couche"]]
KNEEL_SQUAT = [S_INDEX["Agenou"], S_INDEX["Accroupi"]]
