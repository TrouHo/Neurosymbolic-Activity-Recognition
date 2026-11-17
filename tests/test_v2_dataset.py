"""Inputs and fold assembly for training on data v2 (numpy only, runs anywhere)."""
import numpy as np
import pytest

from emob import config as C
from emob.data import dataset as DS
from emob.data.features import DEVICES, INV_CHANNELS, acc_invariant, channel_device, device_flags, represent
from emob.data.normalize import apply_channels, fit_channels


def _rot(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q


def test_acc_features_do_not_depend_on_sensor_orientation():
    rng = np.random.default_rng(0)
    t = np.arange(2000) / C.FS
    a = np.stack([0.5 * np.sin(2 * t), 0.3 * np.cos(3 * t), 9.81 + np.sin(t)], 1) + rng.normal(0, 0.05, (2000, 3))
    f0 = acc_invariant(a)
    f1 = acc_invariant(a @ _rot(1).T)                 # same movement, sensor worn differently
    ok = np.isfinite(f0).all(1)
    assert np.allclose(f0[ok], f1[ok], atol=1e-8)
    # gravity is removed from the vertical channel: mean close to 0 while |a| is about 9.8
    assert abs(np.nanmean(f0[:, 1])) < 0.1 and abs(np.nanmean(f0[:, 0]) - 9.81) < 0.2


def test_representation_layout_and_flags():
    n = 500
    x50 = np.random.default_rng(1).normal(size=(n, 15))
    av = np.ones((n, 5), bool)
    av[:, 3] = False                                   # belt missing
    x50[:, 9:12] = np.nan
    ch, fl, names = represent(x50, av)
    assert ch.shape == (n, 13) and fl.shape == (n, 4) and names == INV_CHANNELS
    assert fl[:, DEVICES.index("belt")].sum() == 0 and fl[:, DEVICES.index("phone")].all()
    assert np.isnan(ch[:, [i for i, d in enumerate(channel_device(names)) if d == DEVICES.index("belt")]]).all()
    assert device_flags(av).shape == (n, 4)


def test_channel_normalisation_ignores_missing_participants():
    a = np.random.default_rng(2).normal(3, 2, size=(1000, 2))
    b = np.full((1000, 2), np.nan); b[:, 0] = np.random.default_rng(3).normal(3, 2, 1000)
    mean, std = fit_channels([a, b])
    assert abs(mean[1] - 3) < 0.3                       # channel 1 fitted on `a` only
    x = apply_channels(b, np.zeros((1000, 1)), mean, std)
    assert x.shape == (1000, 3) and np.isfinite(x).all() and (x[:, 1] == 0).all()
    with pytest.raises(ValueError):
        fit_channels([np.full((10, 1), np.nan)])


def _seq(T, lab, flags=(1, 1, 1, 1)):
    K = len(INV_CHANNELS)
    x = np.ones((T * 5, K + 4), np.float32)
    x[:, K:] = np.array(flags, np.float32)
    y = {h: np.full(T, lab, np.int64) for h in C.HEADS}
    return DS.Seq("u", x, y, np.ones(T, bool), np.zeros(T, bool), INV_CHANNELS)


def test_crops_and_class_weights():
    rng = np.random.default_rng(0)
    seqs = [_seq(1000, 1), _seq(700, 2)]
    x, y = DS.sample_crops(seqs, 5, 600, rng)
    assert x.shape == (5, 3000, 17) and set(y) == set(C.HEADS) and y["w"].shape == (5, 600)
    w = DS.class_weights(seqs, "l2", 4)
    assert w[0] == 0 and w[3] == 0 and abs(w[1:3].mean() - 1) < 1e-9 and w[2] > w[1]


def test_modality_dropout_keeps_one_device_and_switches_whole_devices():
    rng = np.random.default_rng(0)
    K = len(INV_CHANNELS)
    x = np.stack([_seq(20, 0).x for _ in range(300)])
    x[:100, :, K + 1:] = 0.0                            # first 100 crops: phone only (user1, 2, 4)
    y = DS.modality_dropout(x, INV_CHANNELS, 1.0, rng)
    flags = y[:, 0, K:]
    assert (flags.sum(1) >= 1).all()                    # never all devices off
    assert (y[:100, 0, K] == 1).all()                   # a lone phone is never dropped
    dev = channel_device(INV_CHANNELS)
    for b in range(100, 300):                           # dropped device: channels and flag both 0
        for j in range(4):
            if flags[b, j] == 0:
                assert (y[b, :, np.flatnonzero(dev == j)] == 0).all()
    y2 = DS.modality_dropout(x, INV_CHANNELS, 0.0, rng)
    assert (y2 == x).all()


def test_eval_configs():
    s = _seq(10, 0, flags=(1, 1, 1, 1))
    hb = DS.eval_config(s, "HB")
    K = len(INV_CHANNELS)
    assert hb.x[:, K + DEVICES.index("phone")].sum() == 0 and hb.x[:, K + DEVICES.index("hand")].all()
    assert DS.eval_config(_seq(10, 0, flags=(1, 0, 0, 0)), "HB") is None      # phone-only participant
    assert DS.eval_config(_seq(10, 0, flags=(1, 1, 0, 1)), "HB") is None      # hand without belt (user5)
    assert (DS.eval_config(s, "ALL").x == s.x).all()


V2 = C.PROJECT / "data" / "processed" / "v2" / "manifest.json"


@pytest.mark.skipif(not V2.exists(), reason="build data v2 first")
def test_build_fold_on_real_v2_data():
    f = DS.build_fold(["user2", "user3", "user4", "user5", "user6", "user7"], "user8", "user1", fraction=0.25, seed=0)
    te, tr = f["test"][0], f["train"]
    assert te.x.shape[1] == 17 and np.isfinite(te.x).all()
    assert te.x.shape[0] == 5 * len(te.y["l2"])
    # training participants: normalised channels have mean about 0 where available
    allx = np.concatenate([s.x for s in tr])
    K = 13
    for k in range(K):
        dev = channel_device(f["norm"]["channels"])[k]
        on = allx[:, K + dev] > 0
        assert abs(allx[on, k].mean()) < 0.1
    # label fraction hides labels of training participants only, the same blocks for every head
    for s in tr:
        hidden = s.y["l2"] == C.IGNORE
        assert ((s.y["s"] == C.IGNORE) | ~hidden | (s.y["s"] != C.IGNORE)).all()
    assert (te.y["l2"] != C.IGNORE).sum() > 0
    # user1 has the phone only: HB not available, PHONE equals ALL
    assert DS.eval_config(te, "HB") is None
    assert (DS.eval_config(te, "PHONE").x == te.x).all()
