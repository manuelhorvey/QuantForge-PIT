from __future__ import annotations
from itertools import combinations
import numpy as np


def cpcv_splits(n_samples: int, n_groups: int = 6, n_test_groups: int = 2):
    """Yield contiguous combinatorial test folds and complementary train indices.

    This creates split geometry only. Apply label-interval purging and embargo to each split
    before model fitting when labels overlap in time. CPCV is not ordinary shuffled K-fold CV.
    """
    if n_groups < 2 or not 1 <= n_test_groups < n_groups: raise ValueError("invalid group counts")
    if n_samples < n_groups: raise ValueError("n_samples must be >= n_groups")
    groups = np.array_split(np.arange(n_samples), n_groups)
    for test_group_ids in combinations(range(n_groups), n_test_groups):
        test = np.concatenate([groups[i] for i in test_group_ids])
        mask = np.ones(n_samples, dtype=bool); mask[test] = False
        yield np.flatnonzero(mask), np.sort(test)


def probability_of_backtest_overfitting(performance_in_sample: np.ndarray,
                                        performance_out_sample: np.ndarray) -> float:
    """Estimate PBO from paired trials × splits IS/OOS performance matrices.

    For each split, select the best in-sample trial, rank that same trial's OOS score among
    all trials, and count how often its OOS rank is below the median. Inputs must come from
    valid symmetric splits with correct purging/embargo. This helper cannot verify that.
    """
    ins = np.asarray(performance_in_sample, dtype=float)
    oos = np.asarray(performance_out_sample, dtype=float)
    if ins.ndim != 2 or ins.shape != oos.shape or min(ins.shape) < 2:
        raise ValueError("IS and OOS matrices must have matching 2D trials × splits shapes")
    if not np.isfinite(ins).all() or not np.isfinite(oos).all():
        raise ValueError("performance matrices must be finite")
    below_median = []
    n_trials = ins.shape[0]
    for j in range(ins.shape[1]):
        winner = int(np.argmax(ins[:, j]))
        # Average ranks handle ties; percentile is in [0, 1], higher is better.
        ranks = np.argsort(np.argsort(oos[:, j], kind="stable"), kind="stable")
        percentile = (ranks[winner] + 1) / (n_trials + 1)
        below_median.append(percentile < 0.5)
    return float(np.mean(below_median))
