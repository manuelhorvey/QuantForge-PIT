"""Purge-aware walk-forward fold specification and planning."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from ..data.timeutils import TemporalError, ensure_utc, ensure_utc_series
from ..dataset import ResearchDataset, hash_frame, hash_payload
from ..models import purged_time_split
from .errors import ExperimentError, ExperimentIntegrityError


def _utc(value: Any, name: str) -> pd.Timestamp:
    try:
        return ensure_utc(value, name=name)
    except (TemporalError, TypeError, ValueError) as exc:
        raise ExperimentError(str(exc)) from exc


def _duration(value: Any, name: str) -> pd.Timedelta:
    try:
        duration = pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentError(f"{name} must be a valid timedelta") from exc
    if pd.isna(duration):
        raise ExperimentError(f"{name} must be a valid timedelta")
    return duration


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ExperimentError(f"{name} must be an integer >= 1")
    return int(value)


@dataclass(frozen=True)
class WalkForwardSpec:
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    step: pd.Timedelta
    embargo: pd.Timedelta = pd.Timedelta(0)
    max_train_span: pd.Timedelta | None = None
    max_folds: int | None = None
    min_train_rows: int = 1
    min_test_rows: int = 1

    def __post_init__(self) -> None:
        test_start = _utc(self.test_start, "test_start")
        test_end = _utc(self.test_end, "test_end")
        step = _duration(self.step, "step")
        embargo = _duration(self.embargo, "embargo")
        max_train_span = (
            None
            if self.max_train_span is None
            else _duration(self.max_train_span, "max_train_span")
        )
        if test_start >= test_end:
            raise ExperimentError("test_start must be strictly before test_end")
        if step <= pd.Timedelta(0):
            raise ExperimentError("step must be strictly positive")
        if step < (test_end - test_start):
            raise ExperimentError("step must be at least the test-window duration")
        if embargo < pd.Timedelta(0):
            raise ExperimentError("embargo cannot be negative")
        if max_train_span is not None and max_train_span <= pd.Timedelta(0):
            raise ExperimentError("max_train_span must be strictly positive")
        max_folds = (
            None
            if self.max_folds is None
            else _positive_integer(self.max_folds, "max_folds")
        )
        min_train_rows = _positive_integer(self.min_train_rows, "min_train_rows")
        min_test_rows = _positive_integer(self.min_test_rows, "min_test_rows")
        object.__setattr__(self, "test_start", test_start)
        object.__setattr__(self, "test_end", test_end)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "embargo", embargo)
        object.__setattr__(self, "max_train_span", max_train_span)
        object.__setattr__(self, "max_folds", max_folds)
        object.__setattr__(self, "min_train_rows", min_train_rows)
        object.__setattr__(self, "min_test_rows", min_test_rows)

    def to_identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "quantforge.walk_forward.v1",
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "step": self.step.isoformat(),
            "embargo": self.embargo.isoformat(),
            "max_train_span": (
                None if self.max_train_span is None else self.max_train_span.isoformat()
            ),
            "max_folds": self.max_folds,
            "min_train_rows": self.min_train_rows,
            "min_test_rows": self.min_test_rows,
        }


@dataclass(frozen=True)
class PurgedFold:
    fold_id: str
    ordinal: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    excluded_indices: tuple[int, ...]
    excluded_reasons: tuple[str, ...]
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo: pd.Timedelta
    train_keys: tuple[tuple[object, pd.Timestamp], ...]
    test_keys: tuple[tuple[object, pd.Timestamp], ...]
    excluded_keys: tuple[tuple[object, pd.Timestamp], ...]


@dataclass(frozen=True)
class FoldPlan:
    dataset_id: str
    analysis_indices: tuple[int, ...]
    analysis_keys: tuple[tuple[object, pd.Timestamp], ...]
    excluded_status_counts: Mapping[str, int]
    folds: tuple[PurgedFold, ...]
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    fold_plan_hash: str


def _key_frame(
    keys: Sequence[tuple[object, pd.Timestamp]],
    instrument_col: str,
    decision_col: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        list(keys),
        columns=[instrument_col, decision_col],
    )


def _membership_hash(
    keys: Sequence[tuple[object, pd.Timestamp]],
    instrument_col: str,
    decision_col: str,
) -> str:
    return hash_frame(
        _key_frame(keys, instrument_col, decision_col),
        key_cols=[instrument_col, decision_col],
        columns=[instrument_col, decision_col],
        schema_version="quantforge.membership.v1",
    )


def _analysis_exclusion_reason(
    dataset_status: str,
    label_status: str,
    timing_complete: bool,
    label_finite: bool,
) -> str | None:
    if dataset_status != "ok":
        if dataset_status == "label_not_ok":
            return "label_not_ok"
        return dataset_status or "unknown_dataset_status"
    if label_status != "ok":
        return "label_not_ok"
    if not timing_complete:
        return "missing_label_timing"
    if not label_finite:
        return "nonfinite_label"
    return None


def plan_folds(
    dataset: ResearchDataset,
    spec: WalkForwardSpec,
    *,
    dataset_id: str | None = None,
) -> FoldPlan:
    if not isinstance(dataset, ResearchDataset):
        raise ExperimentIntegrityError("dataset must be a ResearchDataset")
    if not isinstance(spec, WalkForwardSpec):
        raise ExperimentError("spec must be a WalkForwardSpec")
    try:
        dataset.verify_integrity()
    except (TypeError, ValueError) as exc:
        raise ExperimentIntegrityError(f"dataset integrity check failed: {exc}") from exc

    actual_dataset_id = dataset.manifest.get("dataset_id")
    if not isinstance(actual_dataset_id, str) or not actual_dataset_id:
        raise ExperimentIntegrityError("dataset manifest has no dataset_id")
    if dataset_id is not None and dataset_id != actual_dataset_id:
        raise ExperimentIntegrityError("supplied dataset_id does not match the dataset")

    frame = dataset.frame
    instrument_col = dataset.instrument_col
    decision_col = dataset.decision_col
    required = [
        "dataset_status",
        "label_status",
        dataset.label_name,
        decision_col,
        "label_start_timestamp",
        "label_end_timestamp",
        "label_available_at",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ExperimentIntegrityError(
            f"dataset is missing planning columns {sorted(set(missing))}"
        )

    try:
        decisions = ensure_utc_series(frame[decision_col], name=decision_col)
        label_starts = ensure_utc_series(
            frame["label_start_timestamp"], name="label_start_timestamp", allow_missing=True
        )
        label_ends = ensure_utc_series(
            frame["label_end_timestamp"], name="label_end_timestamp", allow_missing=True
        )
        label_available = ensure_utc_series(
            frame["label_available_at"], name="label_available_at", allow_missing=True
        )
    except (TemporalError, TypeError, ValueError) as exc:
        raise ExperimentIntegrityError(f"dataset timestamps are invalid: {exc}") from exc

    dataset_status = (
        frame["dataset_status"].astype("string").fillna("").astype(str)
    )
    label_status = frame["label_status"].astype("string").fillna("").astype(str)
    labels = pd.to_numeric(frame[dataset.label_name], errors="coerce")
    finite = np.isfinite(labels.to_numpy(dtype=float, na_value=np.nan))
    timing_complete = (
        label_starts.notna().to_numpy()
        & label_ends.notna().to_numpy()
        & label_available.notna().to_numpy()
        & decisions.notna().to_numpy()
    )
    selected = (
        dataset_status.eq("ok").to_numpy()
        & label_status.eq("ok").to_numpy()
        & finite
        & timing_complete
    )

    positions: np.ndarray = np.arange(len(frame), dtype=int)
    counts: Counter[str] = Counter()
    for position in positions[~selected]:
        reason = _analysis_exclusion_reason(
            dataset_status.iloc[position],
            label_status.iloc[position],
            bool(timing_complete[position]),
            bool(finite[position]),
        )
        if reason is not None:
            counts[reason] += 1

    analysis_positions = positions[selected]
    if analysis_positions.size == 0:
        raise ExperimentError("dataset contains no evaluable analysis rows")

    time_values = decisions.to_numpy(dtype="datetime64[ns]").astype("int64")
    order = np.lexsort((analysis_positions, time_values[analysis_positions]))
    analysis_positions = analysis_positions[order]
    analysis_keys = tuple(
        (
            frame.iloc[int(position)][instrument_col],
            pd.Timestamp(decisions.iloc[int(position)]),
        )
        for position in analysis_positions
    )
    analysis_indices = tuple(int(position) for position in analysis_positions)
    analysis_times = np.asarray([key[1] for key in analysis_keys], dtype=object)

    last_sample_time = max(key[1] for key in analysis_keys)
    duration = spec.test_end - spec.test_start
    embargo = spec.embargo
    folds: list[PurgedFold] = []
    ordinal = 0
    while spec.max_folds is None or len(folds) < spec.max_folds:
        window_start = spec.test_start + (spec.step * ordinal)
        if window_start > last_sample_time:
            break
        window_end = window_start + duration
        try:
            split = purged_time_split(
                sample_times=[key[1] for key in analysis_keys],
                label_start_times=label_starts.iloc[analysis_positions].tolist(),
                label_end_times=label_ends.iloc[analysis_positions].tolist(),
                label_available_times=label_available.iloc[analysis_positions].tolist(),
                test_start=window_start,
                test_end=window_end,
                embargo=embargo,
            )
        except (TypeError, ValueError) as exc:
            raise ExperimentError(
                f"candidate fold {ordinal} [{window_start.isoformat()}, "
                f"{window_end.isoformat()}) failed: {exc}"
            ) from exc

        train_positions = split.train
        if spec.max_train_span is not None:
            cutoff = window_start - spec.max_train_span
            cutoff_ns = cutoff.value
            keep = np.asarray(
                [analysis_times[position].value >= cutoff_ns for position in train_positions],
                dtype=bool,
            )
            train_positions = train_positions[keep]
        if len(train_positions) < spec.min_train_rows:
            raise ExperimentError(
                f"candidate fold {ordinal} has {len(train_positions)} training rows; "
                f"min_train_rows is {spec.min_train_rows}"
            )
        if len(split.test) < spec.min_test_rows:
            raise ExperimentError(
                f"candidate fold {ordinal} has {len(split.test)} test rows; "
                f"min_test_rows is {spec.min_test_rows}"
            )

        train_indices = tuple(int(analysis_positions[position]) for position in train_positions)
        test_indices = tuple(int(analysis_positions[position]) for position in split.test)
        excluded_indices = tuple(
            int(analysis_positions[position]) for position in split.excluded
        )
        train_keys = tuple(analysis_keys[int(position)] for position in train_positions)
        test_keys = tuple(analysis_keys[int(position)] for position in split.test)
        excluded_keys = tuple(
            analysis_keys[int(position)] for position in split.excluded
        )

        train_hash = _membership_hash(train_keys, instrument_col, decision_col)
        test_hash = _membership_hash(test_keys, instrument_col, decision_col)
        excluded_hash = _membership_hash(excluded_keys, instrument_col, decision_col)
        fold_id = hash_payload(
            {
                "schema_version": "quantforge.fold.v1",
                "dataset_id": actual_dataset_id,
                "ordinal": ordinal,
                "test_start": window_start.isoformat(),
                "test_end": window_end.isoformat(),
                "embargo": embargo.isoformat(),
                "train_membership_hash": train_hash,
                "test_membership_hash": test_hash,
                "excluded_membership_hash": excluded_hash,
            }
        )
        folds.append(
            PurgedFold(
                fold_id=fold_id,
                ordinal=ordinal,
                train_indices=train_indices,
                test_indices=test_indices,
                excluded_indices=excluded_indices,
                excluded_reasons=split.excluded_reasons,
                test_start=window_start,
                test_end=window_end,
                embargo=embargo,
                train_keys=train_keys,
                test_keys=test_keys,
                excluded_keys=excluded_keys,
            )
        )
        ordinal += 1

    if not folds:
        raise ExperimentError("fold specification produced no folds")
    evaluation_start = folds[0].test_start
    evaluation_end = folds[-1].test_end
    analysis_hash = _membership_hash(analysis_keys, instrument_col, decision_col)
    fold_plan_hash = hash_payload(
        {
            "schema_version": "quantforge.fold_plan.v1",
            "dataset_id": actual_dataset_id,
            "analysis_membership_hash": analysis_hash,
            "fold_ids": [fold.fold_id for fold in folds],
            "evaluation_start": evaluation_start.isoformat(),
            "evaluation_end": evaluation_end.isoformat(),
        }
    )
    return FoldPlan(
        dataset_id=actual_dataset_id,
        analysis_indices=analysis_indices,
        analysis_keys=analysis_keys,
        excluded_status_counts=dict(sorted(counts.items())),
        folds=tuple(folds),
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        fold_plan_hash=fold_plan_hash,
    )
