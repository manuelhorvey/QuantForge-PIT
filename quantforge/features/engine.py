"""Point-in-time feature computation over declared feature specs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .._research import (
    max_availability,
    normalize_bars,
    normalize_decisions,
    resolve_event_revisions,
    validate_finite_inputs,
)
from ..data.errors import SchemaValidationError
from .spec import GLOBAL_METADATA_COLUMNS, FeatureSpec

_METADATA_SUFFIXES = (
    "__available_at",
    "__source_max_available_at",
    "__source",
    "__revision",
    "__status",
    "__reason",
)
@dataclass(frozen=True)
class FeatureResult:
    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    quality: Mapping[str, Any] = field(default_factory=dict)
    manifest: Mapping[str, Any] = field(default_factory=dict)


def _scalar_result(value: Any) -> float:
    if isinstance(value, pd.Series):
        if value.empty:
            return float("nan")
        value = value.iloc[-1]
    elif isinstance(value, (pd.Index, np.ndarray, list, tuple)):
        array = np.asarray(value).reshape(-1)
        if array.size == 0:
            return float("nan")
        value = array[-1]
    if value is pd.NA or value is pd.NaT or value is None:
        return float("nan")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError("Feature computation returned a non-numeric value") from exc
    if np.isinf(numeric):
        raise SchemaValidationError("Feature computation returned infinity")
    return numeric


def _metadata_columns(names: Sequence[str]) -> list[str]:
    columns: list[str] = []
    for name in names:
        columns.extend(f"{name}{suffix}" for suffix in _METADATA_SUFFIXES)
    return columns


def compute_features(
    decisions: pd.DataFrame,
    bars: pd.DataFrame,
    specs: Sequence[FeatureSpec],
    *,
    frequency: str | None = None,
    instrument_col: str = "symbol",
    decision_col: str = "decision_timestamp",
) -> FeatureResult:
    """Compute features using only data eligible at each decision time.

    The engine filters by event and availability time, resolves the highest
    available revision per event, isolates instruments and frequencies, and
    evaluates each custom function against a copy of the eligible history.
    """

    specs = tuple(specs)
    if not specs:
        raise SchemaValidationError("At least one feature spec is required")
    if not all(isinstance(spec, FeatureSpec) for spec in specs):
        raise SchemaValidationError("specs must contain FeatureSpec instances")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise SchemaValidationError("Feature names must be unique")
    if any(name in {instrument_col, decision_col} for name in names):
        raise SchemaValidationError("Feature names cannot duplicate key columns")

    decision_frame = normalize_decisions(
        decisions, instrument_col=instrument_col, decision_col=decision_col
    )
    bar_frame = normalize_bars(
        bars,
        instrument_col=instrument_col,
        event_col="timestamp",
        availability_col="available_at",
        frequency=frequency,
        context="compute_features",
    )
    for spec in specs:
        missing = [column for column in spec.inputs if column not in bar_frame.columns]
        if missing:
            raise SchemaValidationError(f"Missing feature input columns {missing}")

    bar_groups = {
        key: group for key, group in bar_frame.groupby(instrument_col, sort=False, dropna=False)
    }
    empty_bars = bar_frame.iloc[0:0].copy()
    output_columns = (
        [instrument_col, decision_col]
        + names
        + list(GLOBAL_METADATA_COLUMNS)
        + _metadata_columns(names)
    )
    rows: list[dict[str, Any]] = []

    for _, decision_row in decision_frame.iterrows():
        instrument = decision_row[instrument_col]
        decision_time = decision_row[decision_col]
        instrument_bars = bar_groups.get(instrument, empty_bars)
        eligible = instrument_bars.loc[
            (instrument_bars["timestamp"] <= decision_time)
            & (instrument_bars["available_at"] <= decision_time)
        ]
        resolved = resolve_event_revisions(
            eligible,
            instrument_col=instrument_col,
            event_col="timestamp",
            availability_col="available_at",
        )
        row: dict[str, Any] = {
            instrument_col: instrument,
            decision_col: decision_time,
        }
        feature_statuses: list[str] = []
        feature_reasons: list[str] = []
        feature_sources: set[str] = set()
        feature_revisions: list[int] = []
        feature_availability: list[pd.Timestamp] = []
        for spec in specs:
            history = resolved
            if spec.lag:
                history = history.iloc[:-spec.lag] if spec.lag < len(history) else history.iloc[0:0]
            if spec.window is not None:
                history = history.tail(spec.window)
            min_periods = spec.min_periods
            if min_periods is None:
                min_periods = spec.window if spec.window is not None else 1
            validate_finite_inputs(history, spec.inputs)
            status = "ok"
            reason = ""
            value = float("nan")
            if len(history) < min_periods:
                status = "insufficient_history"
                reason = f"eligible_rows={len(history)} < min_periods={min_periods}"
            else:
                value = _scalar_result(spec.compute(history.copy()))
                if np.isnan(value):
                    status = "warmup_nan"
                    reason = "feature computation returned NaN"
            source_values = (
                sorted({str(value) for value in history["source"].dropna()})
                if "source" in history.columns
                else []
            )
            row[f"{spec.name}"] = value
            row[f"{spec.name}__available_at"] = decision_time
            row[f"{spec.name}__source_max_available_at"] = max_availability(
                history, availability_col="available_at"
            )
            row[f"{spec.name}__source"] = ",".join(source_values)
            row[f"{spec.name}__revision"] = (
                int(history["revision"].max()) if not history.empty else None
            )
            row[f"{spec.name}__status"] = status
            row[f"{spec.name}__reason"] = reason
            feature_statuses.append(status)
            if reason:
                feature_reasons.append(f"{spec.name}: {reason}")
            if source_values:
                feature_sources.update(source_values)
            if not history.empty:
                feature_revisions.append(int(history["revision"].max()))
            max_available = row[f"{spec.name}__source_max_available_at"]
            if max_available is not None:
                feature_availability.append(max_available)
        row["feature_available_at"] = decision_time
        row["feature_source_max_available_at"] = max(
            feature_availability, default=None
        )
        row["feature_source"] = ",".join(sorted(feature_sources))
        row["feature_revision"] = max(feature_revisions, default=None)
        row["feature_status"] = (
            "ok" if all(status == "ok" for status in feature_statuses) else "not_ok"
        )
        row["feature_reason"] = "; ".join(feature_reasons)
        rows.append(row)

    frame = pd.DataFrame(rows, columns=output_columns)
    if not frame.empty:
        frame = frame.sort_values([instrument_col, decision_col], kind="stable").reset_index(
            drop=True
        )
    quality = {
        "rows": int(len(frame)),
        "status_counts": {
            name: dict(
                Counter(
                    frame[f"{name}__status"].tolist() if not frame.empty else []
                )
            )
            for name in names
        },
        "aggregate_status_counts": dict(
            Counter(frame["feature_status"].tolist() if not frame.empty else [])
        ),
    }
    manifest = {
        "schema_version": "quantforge.features.v1",
        "feature_names": names,
        "specs": [spec.to_dict() for spec in specs],
        "instrument_column": instrument_col,
        "decision_column": decision_col,
        "frequency": frequency,
        "rows": int(len(frame)),
        "columns": list(output_columns),
        "status_counts": quality["status_counts"],
        "aggregate_status_counts": quality["aggregate_status_counts"],
    }
    return FeatureResult(
        frame=frame,
        feature_names=tuple(names),
        quality=quality,
        manifest=manifest,
    )
