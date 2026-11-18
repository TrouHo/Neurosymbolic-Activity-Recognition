"""Lambda selection and paired statistics (numpy only)."""
import numpy as np
import pytest

from emob.summarize import holm, select_lambda, wilcoxon_exact


def test_wilcoxon_exact_small_samples():
    assert wilcoxon_exact(np.arange(1, 9) / 100) == pytest.approx(2 / 256)      # 8 of 8 improved
    # Explanation.md §7 example: 6 of 8 improved
    d = np.array([0.05, 0.02, -0.01, 0.04, 0.03, 0.06, 0.01, -0.02])
    p = wilcoxon_exact(d)
    assert 0.02 < p < 0.2
    assert wilcoxon_exact(np.array([0.0, 0.0])) is None
    assert wilcoxon_exact(np.array([0.1, -0.1])) == 1.0


def test_wilcoxon_matches_scipy_when_available():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    for _ in range(5):
        d = rng.normal(0.01, 0.03, 8)
        assert wilcoxon_exact(d) == pytest.approx(stats.wilcoxon(d, method="exact").pvalue)


def test_holm():
    assert holm([0.01, 0.04, 0.03, None]) == [0.03, 0.06, 0.06, None]


def test_select_lambda_uses_mean_over_seeds_and_prefers_smaller_on_ties():
    rows = []
    for lam, vals in {(0.1, 0, 0): [0.5, 0.7], (0.3, 0, 0): [0.6, 0.6], (1.0, 0, 0): [0.55, 0.62]}.items():
        for seed, v in enumerate(vals):
            rows.append({"backbone": "TCN", "model": "NS1", "fold": 0, "fraction": 1.0, "variant": "",
                         "lam": lam, "seed": seed, "val_l2_macro_f1": v})
    assert select_lambda(rows, "TCN", "NS1", 0) == (0.1, 0, 0)               # 0.6 tie -> smaller
    assert select_lambda(rows, "TCN", "NS1", 1) is None
