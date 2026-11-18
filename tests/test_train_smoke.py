"""End-to-end smoke tests on the real data v2 (need torch / sklearn and the data; run on the server)."""
import json

import pytest

from emob import config as C

V2 = C.PROCESSED / "manifest.json"
pytestmark = pytest.mark.skipif(not V2.exists(), reason="build data v2 first")


@pytest.mark.parametrize("model,extra", [("B1", []), ("NS2", ["--lam-hier", "0.3", "--lam-hard", "1", "--lam-soft", "0.3"]),
                                         ("NS3", ["--lam-hier", "0.3", "--lam-hard", "1", "--lam-soft", "0.3"])])
def test_train_two_epochs(tmp_path, model, extra):
    pytest.importorskip("torch")
    from emob import train
    train.main(["--model", model, "--fold", "2", "--max-epochs", "2", "--out", str(tmp_path), *extra])
    (run,) = list(tmp_path.iterdir())
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert m["epochs_run"] == 2 and set(m["test"]) <= set(C.EVAL_CONFIGS) and "ALL" in m["test"]
    assert 0 <= m["test"]["ALL"]["l2_macro_f1"] <= 1
    if model != "B1":
        assert m["rules_on"] == ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
        assert all(m["history"][0][r] is not None for r in m["rules_on"])
    # fold 2: test user3 (no usable belt) -> no HB; validation user4
    assert "HB" not in m["test"]


def test_lstm_one_epoch(tmp_path):
    pytest.importorskip("torch")
    from emob import train
    train.main(["--model", "B2", "--backbone", "lstm", "--fold", "5", "--max-epochs", "1", "--out", str(tmp_path)])
    (run,) = list(tmp_path.iterdir())
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert run.name.startswith("LSTM-B2") and "HB" in m["test"]          # test user6 has hand + belt


def test_random_forest(tmp_path):
    pytest.importorskip("sklearn")
    from emob.baselines import rf
    rf.main(["--fold", "5", "--trees", "10", "--out", str(tmp_path)])
    (run,) = list(tmp_path.iterdir())
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert set(m["test"]) == {"ALL", "PHONE", "HB"}
    assert m["per_config"]["HB"]["n_train_participants"] < m["per_config"]["ALL"]["n_train_participants"]
