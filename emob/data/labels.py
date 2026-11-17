"""Label construction: L2 / L1 / S at the 0.1-s step grid.

Definitions follow proposal §3 (approved 2026-11-05). Choices not fixed by the proposal
are marked DECISION and listed in knowledge_base/02_design_rationale.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from emob import config as C

L2_INDEX = {c: i for i, c in enumerate(C.L2_CLASSES)}
L1_INDEX = {c: i for i, c in enumerate(C.L1_CLASSES)}
S_INDEX = {c: i for i, c in enumerate(C.S_CLASSES)}
L2_TO_L1_IDX = np.array([L1_INDEX[C.L2_TO_L1[c]] for c in C.L2_CLASSES], dtype=np.int64)
TRANSITIONS = set(C.POSTURE_TRANSITIONS) | {C.OTHER_TRANSITION}


def runs(values: np.ndarray) -> list[tuple[str, int, int]]:
    """Maximal runs of equal, non-empty values: (value, start, end_exclusive)."""
    v = np.asarray(values, dtype=object)
    n = len(v)
    if n == 0:
        return []
    change = np.flatnonzero(v[1:] != v[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]])
    return [(v[s], int(s), int(e)) for s, e in zip(starts, ends) if v[s] != ""]


def annotated_mask(families: dict[str, np.ndarray]) -> np.ndarray:
    """True where at least one label family is non-empty (row was annotated)."""
    return np.any(np.stack([families[c] != "" for c in C.FAMILY_COLS]), axis=0)


@dataclass
class PostureResult:
    state: np.ndarray                       # (N,) int, IGNORE where unknown (also on unannotated rows)
    state_all: np.ndarray | None = None     # same, but NOT masked on unannotated rows
    consistent_pairs: int = 0               # X_Y followed by Y_Z
    inconsistent_pairs: int = 0             # X_Y followed by X'_Z with X' != Y
    pairs_with_other: int = 0               # pairs involving Autre_Transition (not checkable)
    transitions: list[tuple[str, int, int]] = field(default_factory=list)


def posture_states(body: np.ndarray, annotated: np.ndarray) -> PostureResult:
    """Reconstruct the posture state from the transition names in the body column.

    Rules (proposal §3.4):
    - between X_Y and X'_Y' the state is Y if Y == X', otherwise masked;
    - before the first transition: X of that transition; after the last: Y;
    - during a transition and after Autre_Transition (until the next transition): masked.
    DECISION (not fixed by the proposal): between X_Y and a following Autre_Transition the
    state is Y (it is known from X_Y; the pair just cannot be checked).
    DECISION: the state is also masked where the row is not annotated (all families empty),
    because unlabelled rows only feed the logic losses (D6).
    """
    n = len(body)
    state = np.full(n, C.IGNORE, dtype=np.int64)
    tr = [r for r in runs(body) if r[0] in TRANSITIONS]
    res = PostureResult(state=state, transitions=tr)
    if not tr:
        res.state_all = state.copy()
        return res

    first = tr[0]
    if first[0] != C.OTHER_TRANSITION:
        state[: first[1]] = S_INDEX[C.transition_states(first[0])[0]]

    for (n1, s1, e1), (n2, s2, e2) in zip(tr[:-1], tr[1:]):
        if n1 == C.OTHER_TRANSITION:
            res.pairs_with_other += 1
            continue                                  # masked until the next transition
        y1 = C.transition_states(n1)[1]
        if n2 == C.OTHER_TRANSITION:
            res.pairs_with_other += 1
            state[e1:s2] = S_INDEX[y1]
            continue
        x2 = C.transition_states(n2)[0]
        if y1 == x2:
            res.consistent_pairs += 1
            state[e1:s2] = S_INDEX[y1]
        else:
            res.inconsistent_pairs += 1               # masked

    last = tr[-1]
    if last[0] != C.OTHER_TRANSITION:
        state[last[2]:] = S_INDEX[C.transition_states(last[0])[1]]

    res.state_all = state.copy()
    state[~annotated] = C.IGNORE
    return res


def mask_posture_after(state: np.ndarray, masked_body: np.ndarray,
                       transitions: list[tuple[str, int, int]]) -> np.ndarray:
    """Pipeline v2 (S4/S5): posture is unknown from the start of each masked body interval until the
    start of the next annotated transition (the masked part of a long run hides posture changes)."""
    out = state.copy()
    starts = [s for _, s, _ in transitions]
    for _, a, b in runs(np.where(masked_body, "M", "").astype(object)):
        nxt = next((s for s in starts if s >= b), len(out))
        out[a:nxt] = C.IGNORE
    return out


LEG_VALUES = {"", "inactif", "Marcher", "Lever_Jambe", "Baisser_Jambe", "Autre_Mouvement_Jambe"}


def walk_targets(leg: np.ndarray) -> np.ndarray:
    """Head W (pipeline v2, approved 2026-11-17): 1 = `Marcher` in the leg column, 0 = any other leg
    label (incl. `inactif`), IGNORE = empty (not annotated or masked)."""
    bad = set(np.unique(leg)) - LEG_VALUES
    if bad:
        raise ValueError(f"unexpected leg labels: {bad}")
    out = np.where(leg == "Marcher", 1, 0).astype(np.int64)
    out[leg == ""] = C.IGNORE
    return out


def transition_targets(body: np.ndarray) -> np.ndarray:
    """Head T (pipeline v2, approved 2026-11-17): 1 = a posture transition (10 named +
    `Autre_Transition`) in the body column, 0 = `inactif`, IGNORE = empty (not annotated or masked)."""
    bad = set(np.unique(body)) - TRANSITIONS - {"", "inactif"}
    if bad:
        raise ValueError(f"unexpected body labels: {bad}")
    out = np.isin(body, list(TRANSITIONS)).astype(np.int64)
    out[body == ""] = C.IGNORE
    return out


def step_centres(n_samples: int) -> np.ndarray:
    """Centre sample index of each complete 0.1-s step (5 samples)."""
    n_steps = n_samples // C.STEP
    return np.arange(n_steps) * C.STEP + C.CENTRE


def encode_l2(activite: np.ndarray) -> np.ndarray:
    """activite strings -> L2 index; IGNORE for empty and protocol markers."""
    out = np.full(len(activite), C.IGNORE, dtype=np.int64)
    for i, a in enumerate(activite):
        if a == "" or a in C.PROTOCOL_MARKERS:
            continue
        if a not in L2_INDEX:
            raise ValueError(f"unknown activite value: {a!r}")
        out[i] = L2_INDEX[a]
    return out


def l1_from_l2(l2: np.ndarray) -> np.ndarray:
    out = np.full(len(l2), C.IGNORE, dtype=np.int64)
    ok = l2 != C.IGNORE
    out[ok] = L2_TO_L1_IDX[l2[ok]]
    return out


def centre_vs_majority(l2_samples: np.ndarray) -> tuple[int, int]:
    """Steps whose centre label is supervised, and how many differ from the 5-sample majority."""
    n_steps = len(l2_samples) // C.STEP
    blocks = l2_samples[: n_steps * C.STEP].reshape(n_steps, C.STEP)
    centre = blocks[:, C.CENTRE]
    differ = 0
    for b, c in zip(blocks, centre):
        if c == C.IGNORE:
            continue
        vals, counts = np.unique(b, return_counts=True)
        if vals[np.argmax(counts)] != c:
            differ += 1
    return int((centre != C.IGNORE).sum()), differ
