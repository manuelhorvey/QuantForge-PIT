from __future__ import annotations


def walk_forward_windows(n_samples: int, train_size: int, test_size: int, step: int | None = None):
    """Yield expanding/rolling chronological (train_slice, test_slice) pairs."""
    if min(n_samples, train_size, test_size) < 1: raise ValueError("sizes must be positive")
    step = test_size if step is None else step
    if step < 1: raise ValueError("step must be positive")
    start = 0
    while start + train_size + test_size <= n_samples:
        yield slice(start, start + train_size), slice(start + train_size, start + train_size + test_size)
        start += step
