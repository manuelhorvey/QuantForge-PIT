"""Deterministic experiment specification and dataset-bound planning."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from ..dataset import ResearchDataset, canonical_json, hash_payload
from .errors import ExperimentError, ExperimentIntegrityError
from .folds import FoldPlan, WalkForwardSpec, plan_folds

_METRICS = frozenset(
    {"mae", "rmse", "mean_error", "pearson", "directional_accuracy"}
)
_SELECTION_METRICS = frozenset({"mae", "rmse", "directional_accuracy"})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentError(f"{name} must be a non-empty string")
    return value


def _validate_parameter(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int, np.integer, np.bool_)):
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ExperimentError(f"parameters.{path} must be finite")
        return
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_parameter(key, f"{path}.<key>")
            _validate_parameter(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_parameter(item, f"{path}[{index}]")
        return
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            _validate_parameter(value.item(), path)
            return
        if value.dtype.kind in "fc" and not bool(np.isfinite(value).all()):
            raise ExperimentError(f"parameters.{path} must contain only finite values")
        if value.dtype.kind == "O":
            for index, item in enumerate(value.flat):
                _validate_parameter(item, f"{path}[{index}]")
        return
    raise ExperimentError(f"parameters.{path} is not canonical-JSON-safe")


@dataclass(frozen=True)
class ExperimentSpec:
    hypothesis: str
    dataset_id: str
    label_name: str
    feature_names: tuple[str, ...]
    folds: WalkForwardSpec
    estimator: str
    parameters: Mapping[str, Any]
    code_version: str
    metric_names: tuple[str, ...] = ("mae", "rmse", "directional_accuracy")
    selection_metric: str = "rmse"
    schema_version: str = "quantforge.experiment.v1"
    experiment_id: str = field(init=False)

    def __post_init__(self) -> None:
        hypothesis = _text(self.hypothesis, "hypothesis")
        dataset_id = _text(self.dataset_id, "dataset_id")
        label_name = _text(self.label_name, "label_name")
        estimator = _text(self.estimator, "estimator")
        code_version = _text(self.code_version, "code_version")
        if not isinstance(self.folds, WalkForwardSpec):
            raise ExperimentError("folds must be a WalkForwardSpec")
        if isinstance(self.feature_names, str):
            raise ExperimentError("feature_names must be a sequence of names")
        feature_names = tuple(self.feature_names)
        if not feature_names:
            raise ExperimentError("feature_names must not be empty")
        if not all(isinstance(name, str) and name.strip() for name in feature_names):
            raise ExperimentError("feature_names must contain non-empty strings")
        if len(set(feature_names)) != len(feature_names):
            raise ExperimentError("feature_names must be unique")
        if isinstance(self.metric_names, str):
            raise ExperimentError("metric_names must be a sequence of names")
        metric_names = tuple(self.metric_names)
        if not metric_names:
            raise ExperimentError("metric_names must not be empty")
        if len(set(metric_names)) != len(metric_names):
            raise ExperimentError("metric_names must be unique")
        unknown = [name for name in metric_names if name not in _METRICS]
        if unknown:
            raise ExperimentError(f"unknown metric names {unknown}")
        if self.selection_metric not in metric_names:
            raise ExperimentError("selection_metric must be in metric_names")
        if self.selection_metric not in _SELECTION_METRICS:
            raise ExperimentError(
                f"selection_metric must be one of {sorted(_SELECTION_METRICS)}"
            )
        if self.schema_version != "quantforge.experiment.v1":
            raise ExperimentError("schema_version must be 'quantforge.experiment.v1'")
        if not isinstance(self.parameters, Mapping):
            raise ExperimentError("parameters must be a mapping")
        parameters = copy.deepcopy(dict(self.parameters))
        for key, value in parameters.items():
            _validate_parameter(key, "<key>")
            _validate_parameter(value, str(key))
        try:
            canonical_json(parameters)
        except (TypeError, ValueError) as exc:
            raise ExperimentError(f"parameters are not canonical-JSON-safe: {exc}") from exc

        identity = {
            "schema_version": self.schema_version,
            "hypothesis": hypothesis,
            "dataset_id": dataset_id,
            "label_name": label_name,
            "feature_names": list(feature_names),
            "fold_spec": self.folds.to_identity_dict(),
            "estimator": estimator,
            "parameters": parameters,
            "code_version": code_version,
            "metric_names": list(metric_names),
            "selection_metric": self.selection_metric,
        }
        experiment_id = hash_payload(identity)
        object.__setattr__(self, "hypothesis", hypothesis)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "label_name", label_name)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "estimator", estimator)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "code_version", code_version)
        object.__setattr__(self, "metric_names", metric_names)
        object.__setattr__(self, "experiment_id", experiment_id)


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_id: str
    dataset_id: str
    label_name: str
    feature_names: tuple[str, ...]
    spec: ExperimentSpec
    fold_plan: FoldPlan


def plan_experiment(dataset: ResearchDataset, spec: ExperimentSpec) -> ExperimentPlan:
    if not isinstance(dataset, ResearchDataset):
        raise ExperimentIntegrityError("dataset must be a ResearchDataset")
    if not isinstance(spec, ExperimentSpec):
        raise ExperimentError("spec must be an ExperimentSpec")
    try:
        dataset.verify_integrity()
    except (TypeError, ValueError) as exc:
        raise ExperimentIntegrityError(f"dataset integrity check failed: {exc}") from exc
    actual_dataset_id = dataset.manifest.get("dataset_id")
    if actual_dataset_id != spec.dataset_id:
        raise ExperimentIntegrityError("spec dataset_id does not match the dataset")
    if dataset.label_name != spec.label_name:
        raise ExperimentIntegrityError("spec label_name does not match the dataset")
    if dataset.feature_names != spec.feature_names:
        raise ExperimentIntegrityError(
            "spec feature_names do not match the dataset feature order"
        )
    try:
        fold_plan = plan_folds(dataset, spec.folds, dataset_id=spec.dataset_id)
    except ExperimentIntegrityError:
        raise
    except ExperimentError as exc:
        raise ExperimentIntegrityError(f"experiment planning failed: {exc}") from exc
    return ExperimentPlan(
        experiment_id=spec.experiment_id,
        dataset_id=spec.dataset_id,
        label_name=spec.label_name,
        feature_names=spec.feature_names,
        spec=spec,
        fold_plan=fold_plan,
    )
