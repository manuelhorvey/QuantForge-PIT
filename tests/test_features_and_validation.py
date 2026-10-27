import numpy as np
import pandas as pd
from quantforge.features import fracdiff_ffd, momentum
from quantforge.models.splits import purged_split
from quantforge.optimization.walkforward import walk_forward_windows


def test_momentum_is_trailing_only():
    s = pd.Series([100., 101., 102., 200.])
    out = momentum(s, 1)
    assert np.isclose(out.iloc[1], .01)


def test_fracdiff_has_no_future_dependence():
    s = pd.Series(np.arange(100, dtype=float))
    a = fracdiff_ffd(s, .5, threshold=.01)
    changed = s.copy(); changed.iloc[-1] = 99999
    b = fracdiff_ffd(changed, .5, threshold=.01)
    assert np.allclose(a.iloc[:-1], b.iloc[:-1], equal_nan=True)


def test_purging_removes_overlapping_labels():
    label_end = np.arange(10)
    label_end[2] = 6
    train, test = purged_split(10, 5, 7, label_end=label_end, embargo=1)
    assert 2 not in train and 5 in test and 7 not in train


def test_walk_forward_is_chronological():
    windows = list(walk_forward_windows(30, 10, 5, 5))
    assert windows[0][0] == slice(0, 10)
    assert windows[0][1] == slice(10, 15)
    assert windows[-1][1].stop <= 30


def test_grid_enumeration_is_deterministic():
    from quantforge.optimization.grid import parameter_grid
    assert list(parameter_grid({"b": [1, 2], "a": [10]})) == [
        {"a": 10, "b": 1}, {"a": 10, "b": 2}]
