from __future__ import annotations
import numpy as np


def purged_split(n_samples: int, test_start: int, test_end: int, *, label_end: np.ndarray,
                 embargo: int = 0):
    """Return train/test integer indices after purging labels overlapping test and embargo.

    label_end[i] is the last index whose outcome contributes to sample i's label.
    The caller is responsible for defining label intervals correctly.
    """
    if not (0 <= test_start < test_end <= n_samples): raise ValueError("invalid test interval")
    if len(label_end) != n_samples: raise ValueError("label_end length mismatch")
    if embargo < 0: raise ValueError("embargo cannot be negative")
    test = np.arange(test_start, test_end)
    train = np.concatenate([np.arange(0, test_start), np.arange(test_end, n_samples)])
    # Remove training labels whose information interval overlaps the test interval.
    overlaps = (train <= test_end - 1) & (label_end[train] >= test_start)
    train = train[~overlaps]
    # Remove observations immediately after test to reduce serial leakage.
    train = train[~((train >= test_end) & (train < min(n_samples, test_end + embargo)))]
    return train, test
