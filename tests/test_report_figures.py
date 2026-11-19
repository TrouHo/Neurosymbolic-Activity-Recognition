"""The numpy rule violations used for the figures equal the training losses (emob/rules_loss.py)."""
import numpy as np
import pytest

from report_figures.rulecalc import probabilities, violations


def test_rulecalc_matches_rules_loss():
    torch = pytest.importorskip("torch")
    from emob.rules_loss import probabilities as tprob, violations as tviol
    rng = np.random.default_rng(0)
    T = 80
    logits = {"l2": rng.normal(size=(20, T)), "l1": rng.normal(size=(7, T)), "s": rng.normal(size=(6, T)),
              "w": rng.normal(size=(1, T)), "t": rng.normal(size=(1, T))}
    v = violations(probabilities(logits))
    p = tprob({h: torch.tensor(z[None], dtype=torch.float64) for h, z in logits.items()})
    for r in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        vt, ok = tviol(p, r)
        m = ok[0].numpy()
        assert np.allclose(v[r][m], vt[0].numpy()[m], atol=1e-9), r
