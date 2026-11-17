import numpy as np

from emob.data.normalize import apply_zscore, fit_zscore
from emob.data.splits import PARTICIPANTS, label_fraction_mask, loso_folds


def test_loso_folds_partition_participants():
    folds = loso_folds()
    assert len(folds) == 8
    for f in folds:
        assert f["test"] != f["val"]
        assert sorted(f["train"] + [f["test"], f["val"]]) == sorted(PARTICIPANTS)
        assert len(f["train"]) == 6
    assert sorted(f["test"] for f in folds) == sorted(PARTICIPANTS)
    assert folds[7]["val"] == "user1"               # cyclic


def test_label_fraction_masks_are_nested_deterministic_and_blockwise():
    labelled = np.ones(6000, bool)
    labelled[:300] = False
    masks = {fr: label_fraction_mask(labelled, fr, "user3", seed=0) for fr in (0.1, 0.25, 0.5, 1.0)}
    assert (masks[1.0] == labelled).all()
    assert not (masks[0.1] & ~masks[0.25]).any()
    assert not (masks[0.25] & ~masks[0.5]).any()
    assert not (masks[0.5] & ~masks[1.0]).any()
    assert (label_fraction_mask(labelled, 0.25, "user3", seed=0) == masks[0.25]).all()
    assert not (label_fraction_mask(labelled, 0.25, "user3", seed=1) == masks[0.25]).all()
    # whole 600-step blocks are kept or hidden
    kept_blocks = masks[0.5].reshape(10, 600).any(axis=1)
    assert kept_blocks.sum() == 5
    assert not (masks[0.5] & ~labelled).any()


def test_zscore_uses_usable_rows_only_and_flags_missing():
    rng = np.random.default_rng(0)
    acc = rng.normal(5.0, 2.0, size=(1000, 6))
    usable = np.ones((1000, 2), bool)
    acc[:100, 3:] = 0.0
    usable[:100, 1] = False
    mean, std = fit_zscore([acc], [usable])
    assert abs(mean[3] - 5.0) < 0.3                  # zeros of unusable rows not included
    x = apply_zscore(acc, usable, mean, std)
    assert x.shape == (1000, 8) and x.dtype == np.float32
    assert (x[:100, 3:6] == 0).all() and (x[:100, 7] == 0).all() and (x[100:, 7] == 1).all()
    assert np.isfinite(x).all()


def test_zscore_handles_a_participant_without_belt():
    rng = np.random.default_rng(1)
    a1, a2 = rng.normal(size=(500, 6)), np.full((500, 6), np.nan)
    a2[:, :3] = rng.normal(size=(500, 3))
    u1 = np.ones((500, 2), bool)
    u2 = np.stack([np.ones(500, bool), np.zeros(500, bool)], axis=1)
    mean, std = fit_zscore([a1, a2], [u1, u2])
    x = apply_zscore(a2, u2, mean, std)
    assert np.isfinite(x).all() and (x[:, 3:6] == 0).all()
