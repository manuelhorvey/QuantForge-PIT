"""Exact-coverage out-of-sample evaluation and descriptive metrics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from ..data.timeutils import TemporalError, ensure_utc_series
from ..dataset import ResearchDataset, hash_frame
from .errors import (
    ExperimentError,
    ExperimentIntegrityError,
    PredictionCoverageError,
)
from .folds import PurgedFold
from .spec import ExperimentPlan

_METRICS = frozenset(
    {"mae", "rmse", "mean_error", "pearson", "directional_accuracy"}
)
_SAMPLE_LIMIT = 5


@dataclass(frozen=True)
class FoldResult:
    fold_id: str
    n_train: int
    n_test: int
    n_obs: int
    metrics: Mapping[str, float | None]
    predictions_hash: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    dataset_id: str
    fold_plan_hash: str
    folds: tuple[FoldResult, ...]
    pooled: Mapping[str, float | None]
    pooled_n_obs: int
    predictions_hash: str
    metric_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "dataset_id": self.dataset_id,
            "fold_plan_hash": self.fold_plan_hash,
            "folds": [
                {
                    "fold_id": fold.fold_id,
                    "n_train": fold.n_train,
                    "n_test": fold.n_test,
                    "n_obs": fold.n_obs,
                    "metrics": dict(fold.metrics),
                    "predictions_hash": fold.predictions_hash,
                    "test_start": fold.test_start.isoformat(),
                    "test_end": fold.test_end.isoformat(),
                }
                for fold in self.folds
            ],
            "pooled": dict(self.pooled),
            "pooled_n_obs": self.pooled_n_obs,
            "predictions_hash": self.predictions_hash,
            "metric_names": list(self.metric_names),
        }


def _metric_names(
    requested: Sequence[str] | None,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    source = default if requested is None else requested
    if isinstance(source, str):
        raise ExperimentError("metric_names must be a sequence of names")
    names = tuple(source)
    if not names:
        raise ExperimentError("metric_names must not be empty")
    if len(set(names)) != len(names):
        raise ExperimentError("metric_names must be unique")
    unknown = [name for name in names if name not in _METRICS]
    if unknown:
        raise ExperimentError(f"unknown metric names {unknown}")
    return names


def _prepare_dataset(dataset: ResearchDataset, plan: ExperimentPlan) -> None:
    if not isinstance(dataset, ResearchDataset):
        raise ExperimentIntegrityError("dataset must be a ResearchDataset")
    if not isinstance(plan, ExperimentPlan):
        raise ExperimentError("plan must be an ExperimentPlan")
    try:
        dataset.verify_integrity()
    except (TypeError, ValueError) as exc:
        raise ExperimentIntegrityError(f"dataset integrity check failed: {exc}") from exc
    actual_id = dataset.manifest.get("dataset_id")
    if actual_id != plan.dataset_id:
        raise ExperimentIntegrityError("dataset does not match the experiment plan")
    if plan.fold_plan.dataset_id != plan.dataset_id:
        raise ExperimentIntegrityError("fold plan does not match the experiment dataset")
    if dataset.label_name != plan.label_name:
        raise ExperimentIntegrityError("dataset label does not match the experiment plan")


def _resolve_fold(plan: ExperimentPlan, fold: PurgedFold) -> PurgedFold:
    if not isinstance(fold, PurgedFold):
        raise ExperimentError("fold must be a PurgedFold")
    for candidate in plan.fold_plan.folds:
        if candidate.fold_id == fold.fold_id:
            return candidate
    raise ExperimentIntegrityError("fold does not belong to the experiment plan")


def _key_text(key: tuple[object, pd.Timestamp]) -> str:
    return f"({key[0]!r}, {key[1].isoformat()})"


def _bounded_keys(keys: Sequence[tuple[object, pd.Timestamp]]) -> str:
    ordered = sorted(keys, key=lambda key: (str(key[0]), key[1].isoformat()))
    return ", ".join(_key_text(key) for key in ordered[:_SAMPLE_LIMIT])


def _normalize_predictions(
    predictions: pd.DataFrame,
    *,
    instrument_col: str,
    decision_col: str,
) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame):
        raise ExperimentError("predictions must be a pandas DataFrame")
    if len(set(predictions.columns)) != len(predictions.columns):
        raise ExperimentError("prediction frame has duplicate columns")
    required = [instrument_col, decision_col, "prediction"]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ExperimentError(f"prediction frame is missing columns {missing}")
    normalized = predictions.loc[:, required].copy()
    try:
        normalized[decision_col] = ensure_utc_series(
            normalized[decision_col],
            name=decision_col,
        )
    except (TemporalError, TypeError, ValueError) as exc:
        raise ExperimentError(f"prediction decision timestamps are invalid: {exc}") from exc
    return normalized


def _prediction_keys(
    predictions: pd.DataFrame,
    *,
    instrument_col: str,
    decision_col: str,
) -> list[tuple[object, pd.Timestamp]]:
    return [
        (instrument, pd.Timestamp(decision))
        for instrument, decision in zip(
            predictions[instrument_col].tolist(),
            predictions[decision_col].tolist(),
            strict=True,
        )
    ]


def _check_coverage(
    expected: Sequence[tuple[object, pd.Timestamp]],
    actual: Sequence[tuple[object, pd.Timestamp]],
) -> None:
    try:
        expected_set = set(expected)
        actual_set = set(actual)
        counts = Counter(actual)
    except TypeError as exc:
        raise ExperimentError("prediction keys must be hashable scalars") from exc
    duplicates = [key for key, count in counts.items() if count > 1]
    missing = expected_set - actual_set
    extra = actual_set - expected_set
    if not duplicates and not missing and not extra:
        return

    parts: list[str] = []
    if duplicates:
        excess = sum(count - 1 for count in counts.values() if count > 1)
        parts.append(
            f"{len(duplicates)} duplicate key(s), {excess} excess row(s); "
            f"sample: {_bounded_keys(duplicates)}"
        )
    if missing:
        parts.append(f"{len(missing)} missing key(s); sample: {_bounded_keys(tuple(missing))}")
    if extra:
        parts.append(f"{len(extra)} extra key(s); sample: {_bounded_keys(tuple(extra))}")
    raise PredictionCoverageError("prediction coverage failed: " + "; ".join(parts))


def _validate_prediction_values(predictions: pd.DataFrame) -> np.ndarray:
    values: list[float] = []
    for position, value in enumerate(predictions["prediction"].tolist()):
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if missing:
            raise ExperimentError(f"prediction row {position} is null or NaN")
        if isinstance(value, (bool, np.bool_)):
            raise ExperimentError(f"prediction row {position} is boolean")
        if not isinstance(value, (Real, np.integer, np.floating)):
            raise ExperimentError(
                f"prediction row {position} is not a real number "
                f"(got {type(value).__name__})"
            )
        try:
            numeric = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ExperimentError(
                f"prediction row {position} is not a finite real number"
            ) from exc
        if not math.isfinite(numeric):
            raise ExperimentError(f"prediction row {position} is not finite")
        values.append(numeric)
    return np.asarray(values, dtype=float)


def _labels_for_keys(
    dataset: ResearchDataset,
    keys: Sequence[tuple[object, pd.Timestamp]],
) -> np.ndarray:
    frame = dataset.frame
    instrument_col = dataset.instrument_col
    decision_col = dataset.decision_col
    required = [instrument_col, decision_col, dataset.label_name, "label_status"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ExperimentIntegrityError(f"dataset is missing label columns {missing}")
    try:
        decisions = ensure_utc_series(frame[decision_col], name=decision_col)
    except (TemporalError, TypeError, ValueError) as exc:
        raise ExperimentIntegrityError(f"dataset timestamps are invalid: {exc}") from exc
    labels = pd.to_numeric(frame[dataset.label_name], errors="coerce")
    status = frame["label_status"].astype("string").fillna("").astype(str)
    label_by_key: dict[tuple[object, pd.Timestamp], float] = {}
    for position in range(len(frame)):
        if status.iloc[position] != "ok":
            continue
        value = labels.iloc[position]
        if pd.isna(value):
            continue
        try:
            numeric = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ExperimentIntegrityError(
                f"dataset label for row {position} is not a finite number"
            ) from exc
        if not math.isfinite(numeric):
            continue
        key = (
            frame.iloc[position][instrument_col],
            pd.Timestamp(decisions.iloc[position]),
        )
        label_by_key[key] = numeric
    resolved: list[float] = []
    for key in keys:
        if key not in label_by_key:
            raise ExperimentIntegrityError(
                f"dataset has no valid label for test key {_key_text(key)}"
            )
        resolved.append(label_by_key[key])
    return np.asarray(resolved, dtype=float)


def _checked_metric(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ExperimentError(f"metric {name!r} is not finite")
    return numeric


def _calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    names: Sequence[str],
) -> dict[str, float | None]:
    if len(actual) != len(predicted) or len(actual) == 0:
        raise ExperimentError("metrics require at least one paired observation")
    result: dict[str, float | None] = {}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        error = predicted - actual
        for name in names:
            if name == "mae":
                result[name] = _checked_metric(name, float(np.mean(np.abs(error))))
            elif name == "rmse":
                result[name] = _checked_metric(name, float(np.sqrt(np.mean(np.square(error)))))
            elif name == "mean_error":
                result[name] = _checked_metric(name, float(np.mean(error)))
            elif name == "pearson":
                if len(actual) < 2:
                    result[name] = None
                    continue
                centered_actual = actual - float(np.mean(actual))
                centered_predicted = predicted - float(np.mean(predicted))
                actual_ss = float(np.dot(centered_actual, centered_actual))
                predicted_ss = float(np.dot(centered_predicted, centered_predicted))
                if actual_ss <= 0.0 or predicted_ss <= 0.0:
                    result[name] = None
                else:
                    result[name] = _checked_metric(
                        name,
                        float(
                            np.dot(centered_actual, centered_predicted)
                            / math.sqrt(actual_ss * predicted_ss)
                        ),
                    )
            elif name == "directional_accuracy":
                nonzero = actual != 0.0
                if not bool(nonzero.any()):
                    result[name] = None
                else:
                    matches = np.sign(predicted[nonzero]) == np.sign(actual[nonzero])
                    result[name] = _checked_metric(name, float(np.mean(matches)))
    return result


def _prediction_hash(
    predictions: pd.DataFrame,
    *,
    instrument_col: str,
    decision_col: str,
) -> str:
    return hash_frame(
        predictions,
        key_cols=[instrument_col, decision_col],
        columns=[instrument_col, decision_col, "prediction"],
        schema_version="quantforge.predictions.v1",
    )


def _evaluate_fold_core(
    dataset: ResearchDataset,
    plan: ExperimentPlan,
    fold: PurgedFold,
    predictions: pd.DataFrame,
    names: tuple[str, ...],
) -> tuple[FoldResult, pd.DataFrame, np.ndarray, np.ndarray]:
    instrument_col = dataset.instrument_col
    decision_col = dataset.decision_col
    normalized = _normalize_predictions(
        predictions,
        instrument_col=instrument_col,
        decision_col=decision_col,
    )
    actual_keys = _prediction_keys(
        normalized,
        instrument_col=instrument_col,
        decision_col=decision_col,
    )
    _check_coverage(fold.test_keys, actual_keys)
    predicted = _validate_prediction_values(normalized)
    actual = _labels_for_keys(dataset, fold.test_keys)
    by_key = dict(zip(actual_keys, predicted, strict=True))
    ordered = np.asarray([by_key[key] for key in fold.test_keys], dtype=float)
    metrics = _calculate_metrics(actual, ordered, names)
    predictions_hash = _prediction_hash(
        normalized,
        instrument_col=instrument_col,
        decision_col=decision_col,
    )
    result = FoldResult(
        fold_id=fold.fold_id,
        n_train=len(fold.train_indices),
        n_test=len(fold.test_keys),
        n_obs=len(actual),
        metrics=metrics,
        predictions_hash=predictions_hash,
        test_start=fold.test_start,
        test_end=fold.test_end,
    )
    return result, normalized, actual, ordered


def evaluate_fold(
    dataset: ResearchDataset,
    plan: ExperimentPlan,
    fold: PurgedFold,
    predictions: pd.DataFrame,
    *,
    metric_names: tuple[str, ...] | None = None,
) -> FoldResult:
    _prepare_dataset(dataset, plan)
    names = _metric_names(metric_names, plan.spec.metric_names)
    canonical_fold = _resolve_fold(plan, fold)
    result, _, _, _ = _evaluate_fold_core(
        dataset,
        plan,
        canonical_fold,
        predictions,
        names,
    )
    return result


def evaluate_experiment(
    dataset: ResearchDataset,
    plan: ExperimentPlan,
    predictions_by_fold: Mapping[str, pd.DataFrame],
    *,
    metric_names: tuple[str, ...] | None = None,
) -> ExperimentResult:
    _prepare_dataset(dataset, plan)
    names = _metric_names(metric_names, plan.spec.metric_names)
    if not isinstance(predictions_by_fold, Mapping):
        raise ExperimentError("predictions_by_fold must be a mapping")
    expected_ids = [fold.fold_id for fold in plan.fold_plan.folds]
    actual_ids = set(predictions_by_fold)
    missing = set(expected_ids) - actual_ids
    extra = actual_ids - set(expected_ids)
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(
                f"{len(missing)} missing fold(s); sample: "
                f"{sorted(str(value) for value in missing)[:_SAMPLE_LIMIT]}"
            )
        if extra:
            parts.append(
                f"{len(extra)} extra fold(s); sample: "
                f"{sorted(str(value) for value in extra)[:_SAMPLE_LIMIT]}"
            )
        raise ExperimentError("fold prediction coverage failed: " + "; ".join(parts))

    fold_results: list[FoldResult] = []
    normalized_frames: list[pd.DataFrame] = []
    pooled_actual: list[np.ndarray] = []
    pooled_predicted: list[np.ndarray] = []
    for fold in plan.fold_plan.folds:
        result, normalized, actual, predicted = _evaluate_fold_core(
            dataset,
            plan,
            fold,
            predictions_by_fold[fold.fold_id],
            names,
        )
        fold_results.append(result)
        normalized_frames.append(normalized)
        pooled_actual.append(actual)
        pooled_predicted.append(predicted)

    pooled_frame = pd.concat(normalized_frames, ignore_index=True)
    duplicated = pooled_frame.duplicated(
        subset=[dataset.instrument_col, dataset.decision_col]
    )
    if bool(duplicated.any()):
        raise ExperimentIntegrityError("pooled out-of-sample prediction keys overlap")
    actual = np.concatenate(pooled_actual)
    predicted = np.concatenate(pooled_predicted)
    pooled = _calculate_metrics(actual, predicted, names)
    predictions_hash = _prediction_hash(
        pooled_frame,
        instrument_col=dataset.instrument_col,
        decision_col=dataset.decision_col,
    )
    return ExperimentResult(
        experiment_id=plan.experiment_id,
        dataset_id=plan.dataset_id,
        fold_plan_hash=plan.fold_plan.fold_plan_hash,
        folds=tuple(fold_results),
        pooled=pooled,
        pooled_n_obs=len(actual),
        predictions_hash=predictions_hash,
        metric_names=names,
    )
