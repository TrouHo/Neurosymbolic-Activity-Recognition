"""Shape / behaviour tests that need torch and sklearn (run on the server)."""
import numpy as np
import pytest

from emob import config as C

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from emob.baselines.rf import features  # noqa: E402
from emob.data.dataset import Seq, class_weights, modality_dropout, sample_crops  # noqa: E402
from emob.models.tcn import TCN  # noqa: E402


def test_tcn_output_shapes_and_receptive_field():
    m = TCN(use_s=True)
    out = m(torch.zeros(2, 3000, 8))
    assert out["l1"].shape == (2, 7, 600)
    assert out["l2"].shape == (2, 20, 600)
    assert out["s"].shape == (2, 6, 600)
    assert TCN.receptive_field_steps(8) == 511
    assert "s" not in TCN(use_s=False)(torch.zeros(1, 50, 8))


def test_tcn_is_non_causal_with_expected_reach():
    """An impulse at step 300 influences outputs ~255 steps away but not 300 steps away."""
    torch.manual_seed(0)
    m = TCN(use_s=False, dropout=0.0).eval()
    x = torch.zeros(1, 5000, 8)
    base = m(x)["l2"]
    x[0, 2500:2505, 0] = 10.0                           # step 500
    diff = (m(x)["l2"] - base).abs().sum(1)[0]
    assert diff[500 - 250] > 0 and diff[500 + 250] > 0
    assert diff[500 - 280] == 0 and diff[500 + 280] == 0


def _seq(T, lab):
    return Seq("u", np.ones((T * 5, 8), np.float32), np.full(T, lab), np.full(T, lab), np.full(T, lab),
               np.full(T, lab), np.full(T, lab), np.ones(T, bool))


def test_crops_and_class_weights():
    rng = np.random.default_rng(0)
    seqs = [_seq(1000, 1), _seq(700, 2)]
    x, l1, l2, s = sample_crops(seqs, 5, 600, rng)
    assert x.shape == (5, 3000, 8) and l2.shape == (5, 600)
    w = class_weights(seqs, "l2", 4)
    assert w[0] == 0 and w[3] == 0 and abs(w[1:3].mean() - 1) < 1e-9 and w[2] > w[1]


def test_modality_dropout_never_removes_the_last_sensor():
    rng = np.random.default_rng(0)
    x = np.ones((50, 10, 8), np.float32)
    x[:, :, 7] = 0.0                                    # belt unavailable everywhere (like user5)
    y = modality_dropout(x, 1.0, rng)
    assert (y[:, :, 6] == 1).all()                      # hand never dropped


def test_modality_dropout_with_four_blocks():
    rng = np.random.default_rng(0)
    x = np.ones((200, 10, 16), np.float32)
    x[:100, :, 12:14] = 0.0                              # first half: hand and belt missing (user1/2/4)
    x[:100, :, 0:6] = 0.0
    y = modality_dropout(x, 1.0, rng)
    flags = y[:, 0, 12:]
    assert (flags.sum(1) >= 1).all()                    # never drops the last available block
    assert (flags[100:].sum(1) == 3).all()              # exactly one block dropped when 4 available
    assert ((flags[:100, 2] == 0) | (flags[:100, 3] == 0)).all()


def test_tcn_and_rf_with_16_input_channels():
    assert TCN(in_ch=16)(torch.zeros(1, 500, 16))["l2"].shape == (1, 20, 100)
    f = features(np.zeros((500, 16), np.float32))
    assert f.shape == (100, 9 * 12 + 4)


def test_rf_features_shape():
    x = np.random.default_rng(0).normal(size=(1000, 8)).astype(np.float32)
    f = features(x)
    assert f.shape == (200, 9 * 6 + 2) and np.isfinite(f).all()
