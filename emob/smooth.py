"""Generic temporal smoothing of the activity sequence: knowledge-free controls for NS4.

Decision 2026-11-21 (methodology review): NS4's gains are on segment metrics (edit, F1@25), which any
smoothing improves by removing fragments. These controls smooth the activity (L2) alone, with no rule
and no other head, so "NS4 - control" isolates what the rules add beyond smoothing:

  mode_filter    majority label in a centred window of w steps (ties keep the arg-max label);
                 w fixed a priori: 5, 11, 21 steps (0.5, 1, 2 s).
  viterbi_switch best sequence under  sum_t log p_t(c_t) - beta * #{t : c_t != c_t-1}  (exact Viterbi).
                 beta = 0 returns the arg-max; larger beta gives fewer segments.
  match_budget   beta chosen per recording so that the share of L2 steps changed w.r.t. the arg-max is as
                 close as possible to NS4's share on the same recording (same amount of editing, no rule).
"""
from __future__ import annotations

import numpy as np


def mode_filter(labels: np.ndarray, width: int, n_classes: int) -> np.ndarray:
    """Centred majority filter (window clipped at the ends); ties are resolved in favour of the
    label at the centre, then the smallest class index."""
    if width % 2 != 1:
        raise ValueError("width must be odd")
    T = len(labels)
    onehot = np.zeros((n_classes, T + 1))
    onehot[labels, np.arange(1, T + 1)] = 1
    cum = onehot.cumsum(1)                                   # cum[:, t] = counts in labels[:t]
    h = width // 2
    lo = np.clip(np.arange(T) - h, 0, T)
    hi = np.clip(np.arange(T) + h + 1, 0, T)
    counts = cum[:, hi] - cum[:, lo]                         # (C, T)
    counts[labels, np.arange(T)] += 0.5                      # tie -> keep the centre label
    return counts.argmax(0).astype(np.int64)


def viterbi_switch(logp: np.ndarray, beta: float) -> np.ndarray:
    """argmax over sequences of sum_t logp[c_t, t] - beta * (number of label changes). logp: (C, T)."""
    C, T = logp.shape
    delta = logp[:, 0].copy()
    back = np.zeros((T, C), np.int64)
    idx = np.arange(C)
    for t in range(1, T):
        j = int(delta.argmax())
        switch = delta[j] - beta
        stay = delta >= switch                               # prefer staying on ties
        back[t] = np.where(stay, idx, j)
        delta = np.where(stay, delta, switch) + logp[:, t]
    out = np.empty(T, np.int64)
    out[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return out


def match_budget(logp: np.ndarray, target: float, iters: int = 30, beta_max: float = 50.0) -> tuple[np.ndarray, float, float]:
    """viterbi_switch with the beta whose share of steps changed w.r.t. the arg-max is closest to
    `target` (bisection; the share grows with beta in practice). Returns (labels, beta, share)."""
    am = logp.argmax(0)
    best = (am, 0.0, 0.0)
    if target <= 0:
        return best
    lo, hi = 0.0, beta_max
    for _ in range(iters):
        mid = (lo + hi) / 2
        lab = viterbi_switch(logp, mid)
        share = float((lab != am).mean())
        if abs(share - target) < abs(best[2] - target):
            best = (lab, mid, share)
        if share < target:
            lo = mid
        else:
            hi = mid
    return best
