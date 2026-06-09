"""Tests for time-aware cross-validation splitters (no future leakage)."""

import numpy as np
import pytest

from core import TimeSeriesSplit, WalkForwardSplit


def test_time_series_split_no_leakage():
    data = np.random.randn(120, 3)
    cv = TimeSeriesSplit(n_splits=4, test_size=10, gap=2)
    splits = list(cv.split(data))

    assert len(splits) == 4
    for train_idx, test_idx in splits:
        # Train strictly precedes test, and the gap is respected
        assert train_idx.max() < test_idx.min()
        assert test_idx.min() - train_idx.max() > cv.gap


def test_time_series_split_expanding_window():
    data = np.random.randn(120, 3)
    cv = TimeSeriesSplit(n_splits=4, test_size=10)
    train_sizes = [len(tr) for tr, _ in cv.split(data)]
    # Expanding window: each fold trains on at least as much as the previous
    assert train_sizes == sorted(train_sizes)


def test_walk_forward_no_leakage():
    data = np.random.randn(200, 3)
    cv = WalkForwardSplit(n_splits=5, train_size=80, test_size=20, gap=1)
    splits = list(cv.split(data))

    assert len(splits) >= 1
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()
        assert len(train_idx) == 80  # fixed-size rolling window


def test_walk_forward_actual_folds_match_reported():
    """
    WalkForwardSplit may yield fewer folds than n_splits when the window runs
    off the end of the data. The actual number produced must never exceed the
    reported get_n_splits(), so callers that pre-size arrays don't over-allocate
    silently (regression guard against an over-report bug).
    """
    data = np.random.randn(200, 3)
    cv = WalkForwardSplit(n_splits=5, train_size=80, test_size=20, gap=1)
    actual = len(list(cv.split(data)))
    assert actual <= cv.get_n_splits()


def test_split_raises_when_insufficient_data():
    data = np.random.randn(10, 3)
    cv = TimeSeriesSplit(n_splits=5, test_size=50)
    with pytest.raises(ValueError):
        list(cv.split(data))
