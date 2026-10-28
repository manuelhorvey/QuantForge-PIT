from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..data.timeutils import ensure_utc, ensure_utc_series


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


@dataclass(frozen=True)
class TimeSplitResult:
    """Forward-only time split with purged overlap and embargo exclusions."""

    train: "np.ndarray"
    test: "np.ndarray"
    excluded: "np.ndarray"
    excluded_reasons: tuple[str, ...]
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo: pd.Timedelta

    @property
    def train_indices(self) -> "np.ndarray":
        return self.train

    @property
    def test_indices(self) -> "np.ndarray":
        return self.test

    @property
    def excluded_indices(self) -> "np.ndarray":
        return self.excluded

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": self.train.tolist(),
            "test": self.test.tolist(),
            "excluded": self.excluded.tolist(),
            "excluded_reasons": list(self.excluded_reasons),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "embargo": self.embargo.isoformat(),
        }


def _as_utc_series(values: Sequence[Any] | pd.Series, *, name: str) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    return ensure_utc_series(series, name=name).reset_index(drop=True)


def _chronological_indices(samples: pd.Series, indices: np.ndarray) -> np.ndarray:
    """Order indices by sample time, breaking ties by original position."""

    result = np.asarray(indices, dtype=int)
    if result.size == 0:
        return result
    times = samples.iloc[result].to_numpy(dtype="datetime64[ns]")
    return result[np.lexsort((result, times))].astype(int)


def purged_time_split(
    *,
    sample_times: Sequence[Any] | pd.Series,
    label_start_times: Sequence[Any] | pd.Series,
    label_end_times: Sequence[Any] | pd.Series,
    label_available_times: Sequence[Any] | pd.Series,
    test_start: Any,
    test_end: Any,
    embargo: Any = "0D",
) -> TimeSplitResult:
    """Split observations by time, purging training labels that touch the test.

    Training observations must occur strictly before ``test_start``. Test
    observations are ``test_start <= sample_time < test_end``. Any pre-test
    sample whose label interval or availability reaches the test start (or the
    embargo boundary) is excluded. Post-test samples are reported but never
    used for training. Equality at a boundary is treated conservatively.

    Train and test indices are returned in chronological sample-time order, with
    original input position as the deterministic tie-breaker. A label start
    before its sample timestamp is invalid; equality is allowed for callers that
    use the sample timestamp as the label start marker.
    """

    samples = _as_utc_series(sample_times, name="sample_times")
    starts = _as_utc_series(label_start_times, name="label_start_times")
    ends = _as_utc_series(label_end_times, name="label_end_times")
    available = _as_utc_series(label_available_times, name="label_available_times")
    if not (len(samples) == len(starts) == len(ends) == len(available)):
        raise ValueError("all label/sample arrays must have the same length")
    if samples.isna().any() or starts.isna().any() or ends.isna().any() or available.isna().any():
        raise ValueError("time columns cannot contain null values")
    start_boundary = ensure_utc(test_start, name="test_start")
    end_boundary = ensure_utc(test_end, name="test_end")
    if start_boundary >= end_boundary:
        raise ValueError("test_start must be strictly before test_end")
    embargo_delta = pd.Timedelta(embargo)
    if embargo_delta < pd.Timedelta(0):
        raise ValueError("embargo cannot be negative")
    if bool((starts < samples).any()):
        raise ValueError("label_start_times must be >= sample_times")
    if bool((starts > ends).any()):
        raise ValueError("label_start_times must be <= label_end_times")
    if bool((available < ends).any()):
        raise ValueError("label_available_times must be >= label_end_times")

    test_mask = (samples >= start_boundary) & (samples < end_boundary)
    if not bool(test_mask.any()):
        raise ValueError("test interval contains no samples")
    test_indices = _chronological_indices(samples, np.flatnonzero(test_mask.to_numpy()))

    embargo_boundary = start_boundary - embargo_delta
    train_indices: list[int] = []
    excluded_indices: list[int] = []
    excluded_reasons: list[str] = []
    for index in np.flatnonzero(~test_mask.to_numpy()):
        sample = samples.iloc[index]
        label_end = ends.iloc[index]
        label_available = available.iloc[index]
        reason: str | None = None
        if sample >= end_boundary:
            reason = "post_test_not_used"
        elif label_end >= start_boundary or label_available >= start_boundary:
            reason = "label_overlaps_test"
        elif label_end >= embargo_boundary or label_available >= embargo_boundary:
            reason = "embargo_overlap"
        if reason is None:
            train_indices.append(int(index))
        else:
            excluded_indices.append(int(index))
            excluded_reasons.append(reason)

    return TimeSplitResult(
        train=_chronological_indices(samples, np.asarray(train_indices, dtype=int)),
        test=test_indices.astype(int),
        excluded=np.asarray(excluded_indices, dtype=int),
        excluded_reasons=tuple(excluded_reasons),
        test_start=start_boundary,
        test_end=end_boundary,
        embargo=embargo_delta,
    )
