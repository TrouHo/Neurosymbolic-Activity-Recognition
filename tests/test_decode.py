"""NS4 constrained Viterbi (emob.decode): optimality against brute force, feasibility, arg-max
fixed point, and behaviour on the real labels of data v2."""
import glob
import itertools

import numpy as np
import pytest

from emob import config as C
from emob.decode import CONSTRAINTS, decode_logits, log_softmax, viterbi

# toy problem: L2 classes 0 other, 1 transition X_Y with X = posture 0, 2 walking, 3 other
TOY = dict(trans_from={1: 0}, marcher=2, walk_postures=[0])


def feasible(c, p, w, min_walk, use=CONSTRAINTS):
    T = len(c)
    for t in range(1, T):
        if "A2" in use and c[t] == 1 and c[t - 1] != 1 and p[t - 1] != 0:
            return False
    for t in range(T):
        if "link" in use and c[t] == 2 and w[t] != 1:
            return False
        if "A45" in use and w[t] == 1 and p[t] not in TOY["walk_postures"]:
            return False
    if "A6" in use:
        t = 0
        while t < T:
            if w[t] == 1:
                b = t
                while b < T and w[b] == 1:
                    b += 1
                if t > 0 and b < T and b - t < min_walk:
                    return False
                t = b
            else:
                t += 1
    return True


def score(lp2, lps, lw, c, p, w):
    return sum(lp2[c[t], t] + lps[p[t], t] + lw[w[t], t] for t in range(len(c)))


def random_problem(rng, T, C_=4, P=2, sharp=2.0):
    lp2 = log_softmax(rng.normal(0, sharp, (C_, T)))
    lps = log_softmax(rng.normal(0, sharp, (P, T)))
    z = rng.normal(0, sharp, T)
    lw = np.stack([-np.logaddexp(0, z), -np.logaddexp(0, -z)])
    return lp2, lps, lw


@pytest.mark.parametrize("min_walk", [2, 3])
@pytest.mark.parametrize("use", [CONSTRAINTS, ("A2",), ("A6", "link"), ("A45",), ()])
def test_optimal_against_brute_force(min_walk, use):
    rng = np.random.default_rng(min_walk * 100 + len(use))
    T = 5
    states = list(itertools.product(range(4), range(2), range(2)))
    for _ in range(3):
        lp2, lps, lw = random_problem(rng, T)
        best = -np.inf
        for seq in itertools.product(states, repeat=T):
            c, p, w = zip(*seq)
            if feasible(c, p, w, min_walk, use):
                best = max(best, score(lp2, lps, lw, c, p, w))
        c, p, w = viterbi(lp2, lps, lw, min_walk=min_walk, use=use, **TOY)
        assert feasible(c, p, w, min_walk, use)
        assert score(lp2, lps, lw, c, p, w) == pytest.approx(best)


def test_feasible_on_long_random_sequences():
    rng = np.random.default_rng(1)
    for _ in range(5):
        lp2, lps, lw = random_problem(rng, 300, sharp=1.0)
        c, p, w = viterbi(lp2, lps, lw, min_walk=10, **TOY)
        assert feasible(c, p, w, 10)


def test_argmax_kept_when_it_satisfies_the_rules():
    rng = np.random.default_rng(2)
    T = 200
    lp2, lps, lw = random_problem(rng, T)
    # make the arg-max feasible: no transition class, no walking anywhere
    lp2[1] -= 50
    lp2[2] -= 50
    lw[1] -= 50
    c, p, w = viterbi(lp2, lps, lw, min_walk=10, **TOY)
    assert (c == lp2.argmax(0)).all() and (p == lps.argmax(0)).all() and (w == 0).all()


V2 = sorted(glob.glob(str(C.PROCESSED / "user*.npz")))


@pytest.mark.skipif(not V2, reason="build data v2 first")
def test_one_hot_labels_are_almost_unchanged():
    """Logits = labels (one-hot, uniform where unknown): decoding keeps the labels except where
    the labels themselves break a rule (A6 on W: 1 run of 259 in data v2)."""
    changed = known = 0
    for f in V2:
        d = np.load(f)
        lg = {}
        for h in ("l2", "l1", "s"):
            y = d[h if h != "s" else "s"].astype(int)
            z = np.zeros((C.HEADS[h], len(y)))
            k = y != C.IGNORE
            z[y[k], np.flatnonzero(k)] = 20.0
            lg[h] = z
        for h in ("w", "t"):
            y = d[h].astype(int)
            lg[h] = np.where(y == C.IGNORE, 0.0, np.where(y == 1, 20.0, -20.0))[None, :]
        out = decode_logits(lg)
        for h in ("l2", "s", "w"):
            y = d[h].astype(int)
            k = y != C.IGNORE
            changed += int((out[h][k] != y[k]).sum())
            known += int(k.sum())
    assert changed / known < 1e-3, (changed, known)
