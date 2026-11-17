"""Rule checks on label sequences (gate G0, and later rule-violation metrics)."""
from __future__ import annotations

import numpy as np

from emob import config as C
from emob.data.labels import S_INDEX, runs

CAT_TO_COL = {"standard": "etiq standard", "mvt_corps": "etiq mvt_corps",
              "mvt_bras": "etiq mvt_bras", "mvt_jambe": "etiq mvt_jambe"}


def hierarchy_agreement(activite, categorie, families) -> tuple[int, int]:
    """Rows where activite is non-empty, and rows where it equals the family named in categorie."""
    rows = np.flatnonzero(activite != "")
    agree = 0
    for r in rows:
        col = CAT_TO_COL.get(categorie[r])
        if col is not None and families[col][r] == activite[r]:
            agree += 1
    return len(rows), agree


def walk_segments(leg: np.ndarray, min_s: float = 1.0) -> tuple[int, int]:
    """Number of 'Marcher' runs in the leg column, and how many last >= min_s."""
    segs = [(e - s) / C.FS for v, s, e in runs(leg) if v == "Marcher"]
    return len(segs), int(sum(d >= min_s for d in segs))


def walking_state_counts(leg: np.ndarray, state: np.ndarray) -> dict[str, int]:
    walk = leg == "Marcher"
    sit_lie = np.isin(state, [S_INDEX["Assis"], S_INDEX["Couche"]])
    return {
        "walking_rows": int(walk.sum()),
        "walking_rows_known_state": int((walk & (state != C.IGNORE)).sum()),
        "walking_rows_sitting_or_lying": int((walk & sit_lie).sum()),
    }


def lookback_check(transitions, state: np.ndarray, deltas_s) -> dict[float, dict[str, int]]:
    """For each posture transition X_Y (not Autre) with onset s, look at state[s - Δ].

    Returns counts per Δ: equals X / other known state / masked / before start.
    Used to choose Δ for rule A2 (open item in the rationale register).
    """
    out = {}
    for d in deltas_s:
        k = int(round(d * C.FS))
        c = {"equals_X": 0, "other_state": 0, "masked": 0, "out_of_range": 0}
        for name, s, e in transitions:
            if name == C.OTHER_TRANSITION:
                continue
            x = S_INDEX[C.transition_states(name)[0]]
            j = s - k
            if j < 0:
                c["out_of_range"] += 1
            elif state[j] == C.IGNORE:
                c["masked"] += 1
            elif state[j] == x:
                c["equals_X"] += 1
            else:
                c["other_state"] += 1
        out[d] = c
    return out


def last_valid_state_check(transitions, state: np.ndarray) -> dict[str, int]:
    """Alternative to a fixed Δ: the last known state before the onset."""
    c = {"equals_X": 0, "other_state": 0, "none": 0}
    for name, s, e in transitions:
        if name == C.OTHER_TRANSITION:
            continue
        x = S_INDEX[C.transition_states(name)[0]]
        prev = np.flatnonzero(state[:s] != C.IGNORE)
        if len(prev) == 0:
            c["none"] += 1
        elif state[prev[-1]] == x:
            c["equals_X"] += 1
        else:
            c["other_state"] += 1
    return c
