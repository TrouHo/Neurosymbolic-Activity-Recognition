"""Model shape and behaviour tests (need torch; run on the training server)."""
import pytest

from emob import config as C

torch = pytest.importorskip("torch")

from emob.models.deepconvlstm import DeepConvLSTM  # noqa: E402
from emob.models.heads import B1_HEADS  # noqa: E402
from emob.models.tcn import TCN  # noqa: E402

UNITS = {"l2": 20, "l1": 7, "s": 6, "w": 1, "t": 1}


@pytest.mark.parametrize("Model", [TCN, DeepConvLSTM])
def test_output_shapes_all_heads(Model):
    out = Model(in_ch=17)(torch.zeros(2, 3000, 17))
    assert set(out) == set(C.HEADS)
    for h, u in UNITS.items():
        assert out[h].shape == (2, u, 600)


@pytest.mark.parametrize("Model", [TCN, DeepConvLSTM])
def test_b1_heads_only(Model):
    out = Model(in_ch=17, heads=B1_HEADS)(torch.zeros(1, 500, 17))
    assert set(out) == {"l2", "l1"}
    with pytest.raises(ValueError):
        Model(in_ch=17, heads=["s"])


def test_tcn_receptive_field_and_non_causal_reach():
    assert TCN.receptive_field_steps(8) == 511
    torch.manual_seed(0)
    m = TCN(in_ch=17, dropout=0.0).eval()
    x = torch.zeros(1, 5000, 17)
    base = m(x)["l2"]
    x[0, 2500:2505, 0] = 10.0                           # impulse at step 500
    diff = (m(x)["l2"] - base).abs().sum(1)[0]
    assert diff[500 - 250] > 0 and diff[500 + 250] > 0
    assert diff[500 - 280] == 0 and diff[500 + 280] == 0


def test_deepconvlstm_sees_the_whole_sequence():
    """Bidirectional LSTM: an impulse influences steps far before and after it."""
    torch.manual_seed(0)
    m = DeepConvLSTM(in_ch=17, dropout=0.0).eval()
    x = torch.zeros(1, 5000, 17)
    base = m(x)["l2"]
    x[0, 2500:2505, 0] = 10.0
    diff = (m(x)["l2"] - base).abs().sum(1)[0]
    assert diff[10] > 0 and diff[990] > 0


def test_parameter_counts():
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert 130_000 < n(TCN(in_ch=17)) < 145_000
    assert n(DeepConvLSTM(in_ch=17)) > 1_000_000        # the original design is much larger than the TCN
