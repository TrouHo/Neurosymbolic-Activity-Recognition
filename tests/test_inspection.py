"""Tests of the inspection package (definitions only; the full run needs the raw data)."""
import importlib

import numpy as np
import pytest

from inspection_data.common import Registry, lengths, segments

models_sizes = importlib.import_module("inspection_data.06_models")


def test_segments_strings_and_ints():
    a = np.array(["", "A", "A", "B", "", "B", "B"], dtype=object)
    assert segments(a) == [("A", 1, 3), ("B", 3, 4), ("B", 5, 7)]
    b = np.array([-1, 2, 2, -1, -1, 3])
    assert segments(b) == [(2, 1, 3), (3, 5, 6)]
    assert lengths(b, 0.1).tolist() == pytest.approx([0.2, 0.1])


def test_unit_changes_count_not_data():
    # 50-Hz rows vs 0.1-s steps (centre row of 5): a 3-row segment can vanish on the step grid
    rows = np.array(["A"] * 13 + ["B"] * 3 + ["A"] * 9, dtype=object)   # B on rows 13-15, no step centre
    steps = rows[np.arange(len(rows) // 5) * 5 + 2]
    assert len(segments(rows)) == 3 and len(segments(steps)) == 1


def test_registry_rejects_duplicates():
    r = Registry()
    r.add("x", np.float64(1.23456789), "s", "v2", "d")
    assert r.entries["x"].value == pytest.approx(1.2346)
    with pytest.raises(KeyError):
        r.add("x", 1, "s", "v2", "d")


def test_parameter_counts_match_models():
    torch = pytest.importorskip("torch")  # noqa: F841
    from emob.models.deepconvlstm import DeepConvLSTM
    from emob.models.tcn import TCN
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(TCN()) == models_sizes.tcn_params() == 139_875
    assert count(DeepConvLSTM()) == models_sizes.deepconvlstm_params() == 1_723_235
