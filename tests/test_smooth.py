"""Knowledge-free smoothing controls (numpy only)."""
import itertools

import numpy as np

from emob.smooth import match_budget, mode_filter, viterbi_switch


def test_mode_filter_removes_short_fragments_and_keeps_ties():
    lab = np.array([0, 0, 0, 1, 0, 0, 0, 2, 2, 2, 2])
    out = mode_filter(lab, 3, 3)
    assert out.tolist() == [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]
    assert mode_filter(np.array([0, 1]), 3, 2).tolist() == [0, 1]          # 1-1 ties keep the centre
    assert mode_filter(lab, 1, 3).tolist() == lab.tolist()


def _brute(logp, beta):
    C, T = logp.shape
    best, arg = -np.inf, None
    for seq in itertools.product(range(C), repeat=T):
        s = sum(logp[c, t] for t, c in enumerate(seq)) - beta * sum(a != b for a, b in zip(seq, seq[1:]))
        if s > best + 1e-12:
            best, arg = s, seq
    return best


def test_viterbi_switch_is_optimal_and_reduces_to_argmax():
    rng = np.random.default_rng(0)
    for _ in range(20):
        logp = np.log(rng.dirichlet(np.ones(3), size=6).T)
        for beta in (0.0, 0.5, 2.0):
            seq = viterbi_switch(logp, beta)
            score = logp[seq, np.arange(6)].sum() - beta * (seq[1:] != seq[:-1]).sum()
            assert abs(score - _brute(logp, beta)) < 1e-9
        assert (viterbi_switch(logp, 0.0) == logp.argmax(0)).all()
        assert len(set(viterbi_switch(logp, 1e6).tolist())) == 1


def test_match_budget_hits_the_target_share():
    rng = np.random.default_rng(1)
    logp = np.log(rng.dirichlet(np.ones(4) * 0.5, size=400).T)
    lab, beta, share = match_budget(logp, 0.10)
    assert abs(share - 0.10) < 0.02 and beta > 0
    assert abs((lab != logp.argmax(0)).mean() - share) < 1e-12
    assert match_budget(logp, 0.0)[1] == 0.0
