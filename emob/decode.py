"""NS4: constrained Viterbi decoding of the saved logits of a trained model (no retraining).

Reference: Explanation.md §9 (approved 2026-11-19). Arg-max decides every head and every step
independently; NS4 instead returns the joint sequence (L2 class, posture, walking) with the highest
total log-probability among the sequences that satisfy the hard constraints:

  A2    onset of a named transition X_Y at step t (L2(t-1) != X_Y)  =>  posture(t-1) = X
  A45   walking (W = 1)  =>  posture in {standing, bent}              (rules A4 and A5)
  A6    a walking run lasts >= 10 steps (1 s); runs touching the start or the end of the
        recording are exempt, as in the violation metric (emob.metrics.rule_violations)
  link  L2 = Marcher  =>  W = 1   (6 258 / 6 258 labelled steps; lets A6 reach L2)

Score of a sequence = sum_t log p_L2(c_t) + log p_S(p_t) + log p_W(w_t). There is no tunable
parameter and no switching penalty: the transitions are free apart from the constraints, so when
the arg-max sequence satisfies every constraint the decoder returns it unchanged; it only edits
the steps where a rule binds, in the cheapest way in probability.

State at each step: (c, p, k) with c the L2 class, p the posture, k the walking counter:
k = 0 not walking, k = 1..K-1 walking for k steps, k = K walking for >= K steps (or exempt run
that started at the beginning of the recording). Without A6 only k in {0, K} is used.
The best predecessor of (c, p, k) does not depend on p (no constraint links p(t) to the past), so
the search costs O(T * C * P * K) with small constant.
"""
from __future__ import annotations

import numpy as np

CONSTRAINTS = ("A2", "A45", "A6", "link")
NEG = -np.inf


def log_softmax(z: np.ndarray, axis: int = 0) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def log_sigmoid(z: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -z)


def viterbi(lp2: np.ndarray, lps: np.ndarray, lw: np.ndarray, *, trans_from: dict[int, int],
            marcher: int, walk_postures: list[int], min_walk: int,
            use: tuple[str, ...] = CONSTRAINTS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """lp2 (C, T), lps (P, T): log-probabilities; lw (2, T): log p(W = 0), log p(W = 1).
    trans_from: {L2 index of a named transition X_Y: posture index of X}.
    Returns (l2, posture, walking) sequences of length T."""
    unknown = set(use) - set(CONSTRAINTS)
    if unknown:
        raise ValueError(f"unknown constraints {unknown}")
    C_, T = lp2.shape
    P = lps.shape[0]
    K = min_walk if "A6" in use else 1          # counter states 0..K
    nk = K + 1

    # forbidden states (c, p, k)
    allowed = np.ones((C_, P, nk), bool)
    if "link" in use:
        allowed[marcher, :, 0] = False
    if "A45" in use:
        bad = np.setdiff1d(np.arange(P), walk_postures)
        allowed[:, bad, 1:] = False

    # counter transitions: prev k' -> allowed next k
    succ = {0: [0, 1] if K > 1 else [0, K]}
    for k in range(1, K):
        succ[k] = [k + 1]
    succ[K] = [K, 0]
    pred_k = {k: [kp for kp in range(nk) if k in succ[kp]] for k in range(nk)}

    def emit(t: int) -> np.ndarray:
        e = lp2[:, t][:, None, None] + lps[:, t][None, :, None] + np.where(np.arange(nk) > 0, lw[1, t], lw[0, t])[None, None, :]
        return np.where(allowed, e, NEG)

    a2 = [(c, x) for c, x in trans_from.items()] if "A2" in use else []
    # start: not walking, or an exempt walking run (counter already full)
    delta = emit(0)
    if K > 1:
        delta[:, :, 1:K] = NEG
    bk = np.zeros((T, C_, P, nk), np.int8)        # best previous counter for (c', p', k)
    bcp = np.zeros((T, C_, nk), np.int32)          # best previous flat (c', p') for (c, k)
    for t in range(1, T):
        # 1. reduce over the previous counter
        Dk = np.full((C_, P, nk), NEG)
        for k in range(nk):
            cand = delta[:, :, pred_k[k]]
            j = cand.argmax(-1)
            Dk[:, :, k] = np.take_along_axis(cand, j[..., None], -1)[..., 0]
            bk[t, :, :, k] = np.asarray(pred_k[k])[j]
        # 2. best previous (c', p') for each target (c, k): free unless c is a transition onset
        flat = Dk.reshape(C_ * P, nk)
        g = flat.argmax(0)                                  # (nk,)
        best = np.broadcast_to(flat[g, np.arange(nk)], (C_, nk)).copy()
        arg = np.broadcast_to(g, (C_, nk)).copy()
        for c, x in a2:
            same = Dk[c]                                    # stay in X_Y: any previous posture
            js = same.argmax(0)
            vs = same[js, np.arange(nk)]
            col = Dk[:, x, :].copy()                        # enter X_Y from another class: posture X
            col[c] = NEG
            jc = col.argmax(0)
            vc = col[jc, np.arange(nk)]
            take_same = vs >= vc
            best[c] = np.where(take_same, vs, vc)
            arg[c] = np.where(take_same, c * P + js, jc * P + x)
        bcp[t] = arg
        delta = best[:, None, :] + emit(t)

    # back-track
    l2 = np.empty(T, np.int64); po = np.empty(T, np.int64); kk = np.empty(T, np.int64)
    c, p, k = np.unravel_index(int(delta.argmax()), delta.shape)
    if not np.isfinite(delta[c, p, k]):
        raise RuntimeError("no feasible sequence")
    for t in range(T - 1, -1, -1):
        l2[t], po[t], kk[t] = c, p, k
        if t == 0:
            break
        cp = bcp[t, c, k]
        c_prev, p_prev = divmod(int(cp), P)
        k = int(bk[t, c_prev, p_prev, k])
        c, p = c_prev, p_prev
    return l2, po, (kk > 0).astype(np.int64)


def decode_logits(logits: dict[str, np.ndarray], use: tuple[str, ...] = CONSTRAINTS) -> dict[str, np.ndarray]:
    """{head: (K, T) logits} of a model with heads L2, L1, S, W, T -> {head: (T,) labels}.
    L2, S, W come from the constrained decoding; L1 and T stay arg-max."""
    from emob import config as C
    from emob.data.labels import L2_INDEX, S_INDEX
    from emob.evaluate import decide
    from emob.rules_spec import MIN_WALK_STEPS, TRANSITION_FROM, TRANSITION_IDX

    lg = {h: np.asarray(v, np.float64) for h, v in logits.items()}
    lw = np.stack([log_sigmoid(-lg["w"][0]), log_sigmoid(lg["w"][0])])
    l2, s, w = viterbi(log_softmax(lg["l2"]), log_softmax(lg["s"]), lw,
                       trans_from=dict(zip(TRANSITION_IDX, TRANSITION_FROM)), marcher=L2_INDEX["Marcher"],
                       walk_postures=[S_INDEX["Debout"], S_INDEX["Penche"]], min_walk=MIN_WALK_STEPS, use=use)
    out = decide(logits)
    out.update({"l2": l2, "s": s, "w": w})
    return out
