"""Rule losses A1–A7 (need torch; run on the training server).

Main check: on one-hot probabilities each rule loss counts exactly the violations that the
rule-violation metric counts (emob.metrics.rule_violations), so the loss penalises what the
metric and gate G measure.
"""
import numpy as np
import pytest

from emob import config as C
from emob.data.labels import L2_TO_L1_IDX
from emob.metrics import rule_violations

torch = pytest.importorskip("torch")

from emob import rules_loss as R  # noqa: E402

I = C.IGNORE


def onehot(y: np.ndarray, k: int) -> torch.Tensor:
    """(T,) labels -> (1, k, T) probabilities; C.IGNORE -> all zeros."""
    p = np.zeros((k, len(y)))
    ok = y != I
    p[y[ok], np.flatnonzero(ok)] = 1.0
    return torch.tensor(p[None], dtype=torch.float64)


def binary(y: np.ndarray) -> torch.Tensor:
    return torch.tensor((y == 1).astype(float)[None], dtype=torch.float64)


def probs(y: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {h: binary(v) if h in C.BINARY_HEADS else onehot(v, C.HEADS[h]) for h, v in y.items()}


def runs_sequence(rng, T, n_classes, max_len):
    out, t = [], 0
    while t < T:
        n = int(rng.integers(1, max_len + 1))
        out += [int(rng.integers(0, n_classes))] * n
        t += n
    return np.array(out[:T])


def loss_sum(p, rule):
    v, ok = R.violations(p, rule)
    return float((v * ok).sum()), v, ok


@pytest.mark.parametrize("seed", range(5))
def test_loss_counts_equal_metric_counts_on_one_hot(seed):
    rng = np.random.default_rng(seed)
    T = 1500
    l2 = runs_sequence(rng, T, 20, 80)
    l1 = L2_TO_L1_IDX[l2].copy()
    flip = rng.random(T) < 0.05
    l1[flip] = (l1[flip] + 1) % 7
    y = {"l2": l2, "l1": l1, "s": runs_sequence(rng, T, 6, 40),
         "w": runs_sequence(rng, T, 2, 15), "t": runs_sequence(rng, T, 2, 30)}
    p = probs(y)
    m = rule_violations(**y)
    _, v1, _ = loss_sum(p, "A1")
    assert int((v1 > 1).sum()) == m["A1"]["viol"]                 # -log(1) = 0 vs -log(eps) ~ 13.8
    for r in ("A2", "A3", "A4", "A5", "A6"):
        assert loss_sum(p, r)[0] == pytest.approx(m[r]["viol"]), r
    assert loss_sum(p, "A7")[0] == pytest.approx(m["A7"]["excess_steps"])


def test_examples_from_the_explanation():
    """Numbers used in Explanation.md §3–5."""
    from emob.data.labels import L2_INDEX, S_INDEX
    T = 4
    l2 = torch.zeros(1, 20, T, dtype=torch.float64)
    sd = L2_INDEX["Debout_Assis"]
    l2[0, sd] = torch.tensor([0.1, 0.9, 0.9, 0.9])
    s = torch.zeros(1, 6, T, dtype=torch.float64)
    s[0, S_INDEX["Debout"]] = torch.tensor([0.2, 0.2, 0.3, 0.3])
    v, _ = R.violations({"l2": l2, "s": s}, "A2")
    assert v[0, 1].item() == pytest.approx(0.81 * 0.8) and v[0, 2].item() == pytest.approx(0.09 * 0.8)
    w = torch.tensor([[0.1, 0.9, 0.9, 0.1]], dtype=torch.float64)
    v, _ = R.violations({"w": w}, "A6")
    # anchor t = 1: run of exactly 1 step (0.9 * 0.9 * (1 - 0.9)) + exactly 2 steps (0.9 ** 4, the "blip" of §4)
    assert v[0, 1].item() == pytest.approx(0.9 * 0.9 * 0.1 + 0.9 ** 4)
    l2 = torch.zeros(1, 20, 60, dtype=torch.float64)
    l2[0, L2_INDEX["Lever_Bras"]] = 0.6
    v, _ = R.violations({"l2": l2}, "A7")
    assert v[0, 0].item() == pytest.approx(0.6)                    # min, not 0.6 ** 51


V2 = C.PROCESSED / "manifest.json"


@pytest.mark.skipif(not V2.exists(), reason="build data v2 first")
def test_labels_of_v2_satisfy_hard_rules():
    """Gate G on the losses: A1, A4, A5 are 0 on the labels; A7 counts the two long runs."""
    for u in [f"user{i}" for i in range(1, 9)]:
        d = np.load(C.PROCESSED / f"{u}.npz")
        y = {h: d[h].astype(np.int64) for h in C.HEADS}
        p = probs(y)
        labelled = torch.tensor((y["l2"] != I)[None])
        v, ok = R.violations(p, "A1")
        assert float((v * (ok & labelled)).sum()) == pytest.approx(0.0, abs=1e-9), u
        for r in ("A4", "A5"):
            assert loss_sum(p, r)[0] == 0.0, (u, r)
        assert loss_sum(p, "A7")[0] == pytest.approx(rule_violations(l2=y["l2"])["A7"]["excess_steps"]), u


def test_gradients_reduce_each_rule():
    torch.manual_seed(0)
    logits = {h: torch.randn(2, C.HEADS[h], 300, requires_grad=True) for h in C.HEADS}
    mask = torch.ones(2, 300, dtype=torch.bool)
    for r in R.RULES:
        before = R.rule_losses(logits, mask, [r])[r]
        g = torch.autograd.grad(before, list(logits.values()), allow_unused=True)
        with torch.no_grad():
            stepped = {h: z - 0.1 * (gi if gi is not None else 0) for (h, z), gi in zip(logits.items(), g)}
        after = R.rule_losses(stepped, mask, [r])[r]
        assert after < before, r


def test_mask_and_missing_heads():
    logits = {h: torch.randn(1, C.HEADS[h], 100) for h in ("l2", "l1")}
    out = R.rule_losses(logits, torch.zeros(1, 100, dtype=torch.bool), ["A1"])
    assert out["A1"].item() == 0.0
    with pytest.raises(ValueError):
        R.rule_losses(logits, torch.ones(1, 100, dtype=torch.bool), ["A4"])
    lam = {"hier": 2.0, "hard": 0.0, "soft": 1.0}
    x = {"A1": torch.tensor(1.0), "A4": torch.tensor(5.0), "A3": torch.tensor(3.0)}
    assert R.weighted_sum(x, lam).item() == 5.0
