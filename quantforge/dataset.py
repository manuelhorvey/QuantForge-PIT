"""Deterministic research-dataset assembly and canonical provenance hashing."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ._research import normalize_decisions
from .data.errors import IncompleteUniverseError, SchemaValidationError
from .data.timeutils import ensure_utc_series
from .features.engine import FeatureResult
from .features.spec import GLOBAL_METADATA_COLUMNS
from .labels import LABEL_METADATA_COLUMNS, LabelResult

FEATURE_METADATA_SUFFIXES = (
    "__available_at",
    "__source_max_available_at",
    "__source",
    "__revision",
    "__status",
    "__reason",
)
_FEATURE_TIME_COLUMNS = ("feature_available_at", "feature_source_max_available_at")
_LABEL_TIME_COLUMNS = tuple(
    column
    for column in LABEL_METADATA_COLUMNS
    if column.endswith("_at") or column.endswith("_timestamp")
)
_DATASET_RESERVED_COLUMNS = {
    "dataset_status",
    "dataset_reason",
    "universe_status",
    "universe_eligible",
    "universe_available_at",
    "universe_membership_status",
}
_DATASET_STATUSES = {
    "ok",
    "missing_label",
    "missing_features",
    "label_not_ok",
}


@dataclass(frozen=True)
class ResearchDataset:
    """Validated point-in-time research dataset.

    The dataclass is frozen so the frame reference cannot be replaced, and
    construction stores a deep copy so later mutations of the caller's source
    frame cannot alter the dataset. ``frame`` is immutable by contract:
    pandas DataFrames expose mutating operations, so callers must not modify
    ``dataset.frame``. A manifest with complete root and component hashes is
    required; it makes mutation detectable. Call :func:`validate_dataset` or
    :meth:`verify_integrity` after any suspected mutation.
    """

    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    label_name: str
    quality: Mapping[str, Any] = field(default_factory=dict)
    manifest: Mapping[str, Any] = field(default_factory=dict)
    filtered_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    filtered_reasons: Mapping[str, int] = field(default_factory=dict)
    instrument_col: str = "symbol"
    decision_col: str = "decision_timestamp"

    def __post_init__(self) -> None:
        if isinstance(self.frame, pd.DataFrame):
            object.__setattr__(self, "frame", self.frame.copy(deep=True))
        validate_dataset(self)

    def verify_integrity(self) -> bool:
        """Re-run full validation and return True when the dataset is unchanged."""

        validate_dataset(self)
        return True


def _canonical(value: Any) -> Any:
    if value is None or value is pd.NaT or value is pd.NA:
        return {"t": "null"}
    if isinstance(value, (bool, np.bool_)):
        return {"t": "bool", "v": bool(value)}
    if isinstance(value, (int, np.integer)):
        return {"t": "int", "v": int(value)}
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isnan(numeric):
            return {"t": "nan"}
        if math.isinf(numeric):
            raise ValueError("Cannot hash an infinite numeric value")
        return {"t": "float", "v": numeric.hex()}
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64, date)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return {"t": "null"}
        if timestamp.tzinfo is None:
            raise ValueError("Cannot hash a timezone-naive timestamp")
        return {"t": "timestamp", "v": timestamp.tz_convert("UTC").isoformat()}
    if isinstance(value, str):
        return {"t": "str", "v": value}
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _canonical(value.item())
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return {"t": "list", "v": [_canonical(item) for item in value]}
    if isinstance(value, Mapping):
        encoded = [
            (json.dumps(_canonical(key), sort_keys=True, separators=(",", ":")), _canonical(item))
            for key, item in value.items()
        ]
        encoded.sort(key=lambda pair: pair[0])
        return {"t": "map", "v": [[key, item] for key, item in encoded]}
    if callable(value):
        raise TypeError("Cannot hash a callable value")
    return {"t": "str", "v": str(value)}


def canonical_json(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def hash_frame(
    frame: pd.DataFrame,
    *,
    key_cols: Sequence[str] = ("symbol", "decision_timestamp"),
    columns: Sequence[str] | None = None,
    schema_version: str | None = None,
) -> str:
    if not isinstance(frame, pd.DataFrame):
        raise SchemaValidationError("hash_frame requires a pandas DataFrame")
    selected = list(columns) if columns is not None else list(frame.columns)
    missing = [c for c in key_cols if c not in frame.columns]
    if missing:
        raise SchemaValidationError(f"Cannot hash without key columns {missing}")
    missing = [c for c in selected if c not in frame.columns]
    if missing:
        raise SchemaValidationError(f"Cannot hash missing columns {missing}")
    key_list = list(key_cols)
    if len(set(key_list)) != len(key_list):
        raise SchemaValidationError("Cannot hash with duplicate key columns")
    if bool(frame.duplicated(subset=key_list).any()):
        raise SchemaValidationError(
            f"Cannot hash a frame with duplicate key rows: {key_list}"
        )
    ordered = frame.sort_values(key_list, kind="stable")
    rows = [
        [_canonical(row[column]) for column in selected]
        for _, row in ordered.iterrows()
    ]
    payload = {
        "schema_version": schema_version,
        "columns": selected,
        "rows": rows,
    }
    return hash_payload(payload)


def _feature_metadata_columns(
    columns: Sequence[str], keys: Sequence[str], names: Sequence[str]
) -> list[str]:
    present = set(columns)
    ordered = [column for column in GLOBAL_METADATA_COLUMNS if column in present]
    for name in names:
        for suffix in FEATURE_METADATA_SUFFIXES:
            candidate = f"{name}{suffix}"
            if candidate in present:
                ordered.append(candidate)
    excluded = set(keys) | set(names) | set(ordered)
    extras = sorted(
        c
        for c in columns
        if c not in excluded and any(c.endswith(suffix) for suffix in FEATURE_METADATA_SUFFIXES)
    )
    return ordered + extras


def _label_metadata_columns(columns: Sequence[str], keys: Sequence[str], label_name: str) -> list[str]:
    present = set(columns)
    excluded = set(keys) | {label_name}
    ordered = [c for c in LABEL_METADATA_COLUMNS if c in present]
    extras = sorted(c for c in columns if c not in excluded and c not in set(ordered))
    return ordered + extras


def _infer_feature_names(frame: pd.DataFrame, keys: Sequence[str]) -> list[str]:
    return sorted(
        c
        for c in frame.columns
        if c not in set(keys)
        and c not in set(GLOBAL_METADATA_COLUMNS)
        and not any(c.endswith(suffix) for suffix in FEATURE_METADATA_SUFFIXES)
    )


def _infer_label_name(
    frame: pd.DataFrame, keys: Sequence[str], label_name: str | None
) -> str:
    if label_name is not None:
        if label_name not in frame.columns:
            raise SchemaValidationError(f"Missing label column {label_name!r}")
        return label_name
    if "label" in frame.columns:
        return "label"
    excluded = set(keys) | set(LABEL_METADATA_COLUMNS)
    candidates = [c for c in frame.columns if c not in excluded]
    if len(candidates) != 1:
        raise SchemaValidationError(
            "Could not infer label column; pass label_name explicitly"
        )
    return candidates[0]


def _feature_time_columns(columns: Sequence[str]) -> list[str]:
    return [
        column
        for column in columns
        if column in _FEATURE_TIME_COLUMNS
        or column.endswith("__available_at")
        or column.endswith("__source_max_available_at")
    ]


def _prepare_features(
    frame: pd.DataFrame,
    names: Sequence[str],
    *,
    decision_col: str,
) -> pd.DataFrame:
    out = frame.copy()
    decisions = out[decision_col]
    for column in _feature_time_columns(out.columns):
        out[column] = ensure_utc_series(out[column], name=column, allow_missing=True)
        present = out[column].notna()
        if bool((out.loc[present, column] > decisions.loc[present]).any()):
            raise SchemaValidationError(
                f"Feature source availability {column!r} is after its decision timestamp"
            )
    for name in names:
        numeric = pd.to_numeric(out[name], errors="coerce")
        non_numeric = numeric.isna() & out[name].notna()
        if bool(non_numeric.any()):
            raise SchemaValidationError(f"Feature {name!r} must be numeric")
        finite = numeric.notna() & np.isfinite(
            numeric.to_numpy(dtype=float, na_value=np.nan)
        )
        if bool((~finite & numeric.notna()).any()):
            raise SchemaValidationError(f"Feature {name!r} contains infinity")
        status_column = f"{name}__status"
        if status_column in out.columns:
            status = (
                out[status_column]
                .astype("string")
                .str.strip()
                .fillna("")
                .astype(str)
            )
            if bool((status.eq("ok") & ~finite).any()):
                raise SchemaValidationError(
                    f"Feature {name!r} has status 'ok' but no finite value"
                )
            if bool((status.ne("") & ~status.eq("ok") & numeric.notna()).any()):
                raise SchemaValidationError(
                    f"Feature {name!r} has a non-ok status with a finite value"
                )
        aggregate_status = "feature_status"
        if aggregate_status in out.columns:
            aggregate = (
                out[aggregate_status]
                .astype("string")
                .str.strip()
                .fillna("")
                .astype(str)
            )
            if bool((aggregate.eq("ok") & ~finite).any()):
                raise SchemaValidationError(
                    f"Feature {name!r} is aggregate-ok but has no finite value"
                )
    return out


def _prepare_labels(
    frame: pd.DataFrame,
    label_name: str,
    *,
    decision_col: str,
    require_intervals: bool = False,
) -> pd.DataFrame:
    out = frame.copy()
    if "label_status" not in out.columns:
        out["label_status"] = np.where(out[label_name].isna(), "missing_value", "ok")
    if "label_reason" not in out.columns:
        out["label_reason"] = ""
    for column in LABEL_METADATA_COLUMNS:
        if column not in out.columns:
            if column in _LABEL_TIME_COLUMNS:
                out[column] = pd.Series(
                    pd.NaT, index=out.index, dtype="datetime64[ns, UTC]"
                )
            else:
                out[column] = ""
    for column in _LABEL_TIME_COLUMNS:
        out[column] = ensure_utc_series(out[column], name=column, allow_missing=True)
    out["label_status"] = (
        out["label_status"]
        .astype("string")
        .str.strip()
        .fillna("missing_value")
        .replace("", "missing_value")
        .astype(str)
    )
    out["label_reason"] = out["label_reason"].fillna("").astype(str)

    label_numeric = pd.to_numeric(out[label_name], errors="coerce")
    non_numeric_label = label_numeric.isna() & out[label_name].notna()
    if bool(non_numeric_label.any()):
        raise SchemaValidationError("Label values must be numeric")
    if bool(np.isinf(label_numeric.to_numpy(dtype=float, na_value=np.nan)).any()):
        raise SchemaValidationError("Label values contain infinity")
    status = out["label_status"]
    ok = status.eq("ok")
    if bool((ok & label_numeric.isna()).any()):
        raise SchemaValidationError("Labels with status 'ok' require a finite value")
    if bool((~ok & label_numeric.notna()).any()):
        raise SchemaValidationError(
            "Labels with a missing/unavailable status must not carry a value"
        )

    decision = out[decision_col]
    start = out["label_start_timestamp"]
    end = out["label_end_timestamp"]
    available = out["label_available_at"]
    if bool((start.notna() & (start <= decision)).any()):
        raise SchemaValidationError(
            "label_start_timestamp must be after its decision timestamp"
        )
    if bool((end.notna() & start.isna()).any()):
        raise SchemaValidationError(
            "label_end_timestamp requires label_start_timestamp"
        )
    if bool((end < start).any()):
        raise SchemaValidationError(
            "label_start_timestamp must be <= label_end_timestamp"
        )
    if bool((available.notna() & start.isna()).any()):
        raise SchemaValidationError(
            "label_available_at requires label_start_timestamp"
        )
    if bool((available < start).any()):
        raise SchemaValidationError(
            "label_available_at must be >= label_start_timestamp"
        )
    if bool((available < end).any()):
        raise SchemaValidationError(
            "label_available_at must be >= label_end_timestamp"
        )
    for column in ("entry_available_at", "exit_available_at", "benchmark_available_at"):
        if column in out.columns:
            value = out[column]
            if bool((value.notna() & start.isna()).any()):
                raise SchemaValidationError(
                    f"{column} requires label_start_timestamp"
                )
            if bool((value < start).any()):
                raise SchemaValidationError(
                    f"{column} must be >= label_start_timestamp"
                )
    if bool((out["exit_available_at"] < end).any()):
        raise SchemaValidationError(
            "exit_available_at must be >= label_end_timestamp"
        )
    if bool((available.notna() & out["entry_available_at"].notna()).any()):
        if bool((available < out["entry_available_at"]).any()):
            raise SchemaValidationError(
                "label_available_at must be >= entry_available_at"
            )
    if bool((available.notna() & out["exit_available_at"].notna()).any()):
        if bool((available < out["exit_available_at"]).any()):
            raise SchemaValidationError(
                "label_available_at must be >= exit_available_at"
            )
    if bool((available.notna() & out["benchmark_available_at"].notna()).any()):
        if bool((available < out["benchmark_available_at"]).any()):
            raise SchemaValidationError(
                "label_available_at must be >= benchmark_available_at"
            )
    if bool((available.notna() & out["adjustment_available_at"].notna()).any()):
        if bool((available < out["adjustment_available_at"]).any()):
            raise SchemaValidationError(
                "label_available_at must be >= adjustment_available_at"
            )
    if require_intervals and bool(
        (ok & (start.isna() | end.isna() | available.isna())).any()
    ):
        raise SchemaValidationError(
            "Labels with status 'ok' require start, end, and availability timestamps"
        )
    return out


def _normalize_universe(
    universe: pd.DataFrame,
    *,
    instrument_col: str,
    decision_col: str,
) -> pd.DataFrame:
    if not isinstance(universe, pd.DataFrame):
        raise SchemaValidationError("universe must be a pandas DataFrame")
    missing = [c for c in (instrument_col, decision_col) if c not in universe.columns]
    if missing:
        raise SchemaValidationError(f"Missing universe columns {missing}")
    eligibility_column = next(
        (c for c in ("eligible", "member", "included") if c in universe.columns),
        None,
    )
    if eligibility_column is None:
        raise SchemaValidationError("universe must provide eligible, member, or included")
    work = normalize_decisions(
        universe, instrument_col=instrument_col, decision_col=decision_col
    )
    if "available_at" in work.columns:
        available = ensure_utc_series(
            work["available_at"], name="available_at", allow_missing=True
        )
        work["universe_available_at"] = available
        work = work.loc[available <= work[decision_col]].copy()
    else:
        work["universe_available_at"] = pd.Series(
            pd.NaT, index=work.index, dtype="datetime64[ns, UTC]"
        )
    status_column = next(
        (column for column in ("status", "membership_status") if column in work.columns),
        None,
    )
    if status_column is None:
        work["universe_membership_status"] = ""
    else:
        work["universe_membership_status"] = (
            work[status_column].fillna("").astype(str)
        )
    work["universe_eligible"] = work[eligibility_column].astype("boolean")
    return work[
        [
            instrument_col,
            decision_col,
            "universe_eligible",
            "universe_available_at",
            "universe_membership_status",
        ]
    ].copy()


def _manifest_hash_check(
    manifest: Mapping[str, Any], name: str, actual: str
) -> None:
    if name not in manifest:
        raise SchemaValidationError(f"manifest is missing {name}")
    expected = manifest[name]
    if not isinstance(expected, str) or not expected:
        raise SchemaValidationError(f"manifest {name} must be a non-empty string")
    if actual != expected:
        raise SchemaValidationError(f"{name} does not match the dataset frame")


def _verify_manifest_integrity(dataset: ResearchDataset, frame: pd.DataFrame) -> None:
    manifest = dataset.manifest
    if not manifest:
        return
    keys = [dataset.instrument_col, dataset.decision_col]
    names = list(dataset.feature_names)
    label_name = dataset.label_name

    expected_columns = manifest.get("columns")
    if expected_columns is not None:
        if not isinstance(expected_columns, (list, tuple)):
            raise SchemaValidationError("manifest columns must be a sequence")
        if list(expected_columns) != list(frame.columns):
            raise SchemaValidationError(
                "Dataset frame columns do not match the manifest columns"
            )
    expected_rows = manifest.get("rows")
    if expected_rows is not None:
        if not isinstance(expected_rows, int) or isinstance(expected_rows, bool):
            raise SchemaValidationError("manifest rows must be an integer")
        if expected_rows != len(frame):
            raise SchemaValidationError(
                "manifest row count does not match the dataset frame"
            )
    expected_names = manifest.get("feature_names")
    if expected_names is not None:
        if not isinstance(expected_names, (list, tuple)):
            raise SchemaValidationError("manifest feature_names must be a sequence")
        if list(expected_names) != names:
            raise SchemaValidationError(
                "manifest feature_names do not match the dataset"
            )
    expected_label = manifest.get("label_name")
    if expected_label is not None and expected_label != label_name:
        raise SchemaValidationError("manifest label_name does not match the dataset")

    feature_metadata = manifest.get("feature_metadata_columns")
    if feature_metadata is None:
        feature_metadata = _feature_metadata_columns(frame.columns, keys, names)
    else:
        if not isinstance(feature_metadata, (list, tuple)):
            raise SchemaValidationError(
                "manifest feature_metadata_columns must be a sequence"
            )
        feature_metadata = list(feature_metadata)
        missing = [column for column in feature_metadata if column not in frame.columns]
        if missing:
            raise SchemaValidationError(
                f"manifest feature_metadata_columns are missing from the frame: {missing}"
            )
    label_metadata = manifest.get("label_metadata_columns")
    if label_metadata is None:
        label_metadata = [
            column for column in LABEL_METADATA_COLUMNS if column in frame.columns
        ]
    else:
        if not isinstance(label_metadata, (list, tuple)):
            raise SchemaValidationError(
                "manifest label_metadata_columns must be a sequence"
            )
        label_metadata = list(label_metadata)
        missing = [column for column in label_metadata if column not in frame.columns]
        if missing:
            raise SchemaValidationError(
                f"manifest label_metadata_columns are missing from the frame: {missing}"
            )

    universe_columns = [
        "universe_eligible",
        "universe_available_at",
        "universe_membership_status",
    ]
    present_universe = [column for column in universe_columns if column in frame.columns]
    if present_universe and len(present_universe) != len(universe_columns):
        raise SchemaValidationError(
            f"Dataset frame has incomplete universe metadata columns: {present_universe}"
        )

    try:
        feature_content_hash = hash_frame(
            frame,
            key_cols=keys,
            columns=keys + names,
            schema_version="quantforge.features.v1",
        )
        feature_source_hash = hash_frame(
            frame,
            key_cols=keys,
            columns=keys + feature_metadata,
            schema_version="quantforge.features.v1",
        )
        label_content_hash = hash_frame(
            frame,
            key_cols=keys,
            columns=keys + [label_name],
            schema_version="quantforge.labels.v1",
        )
        label_source_hash = hash_frame(
            frame,
            key_cols=keys,
            columns=keys + label_metadata,
            schema_version="quantforge.labels.v1",
        )
        if present_universe:
            universe_source_hash = hash_frame(
                frame,
                key_cols=keys,
                columns=keys + universe_columns,
                schema_version="quantforge.universe.v1",
            )
        else:
            universe_source_hash = hash_payload({"universe": None})
        dataset_content_hash = hash_frame(
            frame,
            key_cols=keys,
            columns=list(frame.columns),
            schema_version="quantforge.dataset.v1",
        )
    except (SchemaValidationError, ValueError, TypeError) as exc:
        raise SchemaValidationError(
            f"Cannot recompute dataset integrity hashes: {exc}"
        ) from exc

    for name, actual in (
        ("feature_content_hash", feature_content_hash),
        ("feature_source_hash", feature_source_hash),
        ("label_content_hash", label_content_hash),
        ("label_source_hash", label_source_hash),
        ("universe_source_hash", universe_source_hash),
        ("dataset_content_hash", dataset_content_hash),
    ):
        _manifest_hash_check(manifest, name, actual)

    source_hashes = manifest.get("source_hashes", {})
    if not isinstance(source_hashes, Mapping):
        raise SchemaValidationError("manifest source_hashes must be a mapping")
    source_hash = hash_payload(
        {
            "feature_source": feature_source_hash,
            "label_source": label_source_hash,
            "universe_source": universe_source_hash,
            "source_hashes": dict(source_hashes),
        }
    )
    _manifest_hash_check(manifest, "source_hash", source_hash)


def validate_dataset(dataset: ResearchDataset) -> None:
    """Validate a research dataset's frame, metadata, and manifest.

    This validator is called by :class:`ResearchDataset` construction and can
    also be invoked explicitly. It never mutates the supplied dataset. The
    manifest is required and must contain the root hashes, a quality mapping
    matching ``dataset.quality``, and the component/source hashes. Content and
    source hashes are recomputed from the current frame, so a mutated frame or
    stale manifest fails validation.
    """

    if not isinstance(dataset, ResearchDataset):
        raise SchemaValidationError("validate_dataset requires a ResearchDataset")
    frame = dataset.frame
    if not isinstance(frame, pd.DataFrame):
        raise SchemaValidationError("ResearchDataset.frame must be a DataFrame")
    if not isinstance(dataset.feature_names, tuple) or not dataset.feature_names:
        raise SchemaValidationError("ResearchDataset.feature_names must be a non-empty tuple")
    if not all(isinstance(name, str) and name for name in dataset.feature_names):
        raise SchemaValidationError("Feature names must be non-empty strings")
    if not isinstance(dataset.label_name, str) or not dataset.label_name:
        raise SchemaValidationError("ResearchDataset.label_name is required")
    if len(set(dataset.feature_names)) != len(dataset.feature_names):
        raise SchemaValidationError("Feature names must be unique")
    if dataset.label_name in dataset.feature_names:
        raise SchemaValidationError("Label name cannot duplicate a feature name")
    if dataset.instrument_col == dataset.decision_col:
        raise SchemaValidationError("Instrument and decision columns must differ")
    if len(set(frame.columns)) != len(frame.columns):
        raise SchemaValidationError("Dataset frame has duplicate columns")
    required = [dataset.instrument_col, dataset.decision_col, dataset.label_name]
    required.extend(dataset.feature_names)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SchemaValidationError(f"Dataset frame is missing columns {missing}")
    # Validate keys and timestamp normalization without mutating the frame.
    normalized = normalize_decisions(
        frame,
        instrument_col=dataset.instrument_col,
        decision_col=dataset.decision_col,
    )
    _prepare_features(
        normalized,
        dataset.feature_names,
        decision_col=dataset.decision_col,
    )
    labels = _prepare_labels(
        normalized,
        dataset.label_name,
        decision_col=dataset.decision_col,
    )

    status = labels["label_status"]
    if "dataset_status" in normalized.columns:
        dataset_status = (
            normalized["dataset_status"]
            .astype("string")
            .str.strip()
            .fillna("")
            .astype(str)
        )
        unknown = ~dataset_status.isin(_DATASET_STATUSES)
        if bool(unknown.any()):
            raise SchemaValidationError(
                f"Unknown dataset_status values: {sorted(set(dataset_status[unknown]))}"
            )
        if bool((dataset_status.eq("ok") & ~status.eq("ok")).any()):
            raise SchemaValidationError("dataset_status='ok' requires label_status='ok'")
    if "universe_status" in normalized.columns:
        universe_status = (
            normalized["universe_status"]
            .astype("string")
            .str.strip()
            .fillna("")
            .astype(str)
        )
        valid_universe = {"known", "unknown_membership"}
        unknown = ~universe_status.isin(valid_universe)
        if bool(unknown.any()):
            raise SchemaValidationError(
                f"Unknown universe_status values: {sorted(set(universe_status[unknown]))}"
            )
        if "universe_eligible" in normalized.columns:
            eligible = normalized["universe_eligible"]
            known = universe_status.eq("known")
            if bool((known & eligible.isna()).any()):
                raise SchemaValidationError(
                    "Known universe membership requires non-null eligibility"
                )
            if bool((~known & eligible.notna()).any()):
                raise SchemaValidationError(
                    "Unknown universe membership must not claim an eligibility"
                )

    manifest = dataset.manifest
    if not isinstance(manifest, Mapping):
        raise SchemaValidationError("ResearchDataset.manifest must be a mapping")
    if not manifest:
        raise SchemaValidationError("ResearchDataset.manifest must not be empty")
    for name in ("manifest_hash", "dataset_id"):
        if name not in manifest:
            raise SchemaValidationError(f"manifest is missing {name}")
        value = manifest[name]
        if not isinstance(value, str) or not value:
            raise SchemaValidationError(f"manifest {name} must be a non-empty string")
    schema_version = manifest.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise SchemaValidationError("manifest schema_version must be a non-empty string")
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_hash", "dataset_id"}
    }
    if manifest["manifest_hash"] != hash_payload(payload):
        raise SchemaValidationError("manifest_hash does not match the manifest")
    expected_id = hash_payload(
        {
            "schema_version": schema_version,
            "manifest_hash": manifest["manifest_hash"],
        }
    )
    if manifest["dataset_id"] != expected_id:
        raise SchemaValidationError("dataset_id does not match manifest_hash")
    if "quality" not in manifest:
        raise SchemaValidationError("manifest is missing quality")
    manifest_quality = manifest["quality"]
    if not isinstance(manifest_quality, Mapping):
        raise SchemaValidationError("manifest quality must be a mapping")
    if not isinstance(dataset.quality, Mapping):
        raise SchemaValidationError("ResearchDataset.quality must be a mapping")
    if dict(dataset.quality) != dict(manifest_quality):
        raise SchemaValidationError(
            "ResearchDataset.quality does not match manifest quality"
        )
    _verify_manifest_integrity(dataset, frame)

    filtered = dataset.filtered_frame
    if not isinstance(filtered, pd.DataFrame):
        raise SchemaValidationError("filtered_frame must be a DataFrame")
    if not filtered.empty:
        missing = [
            column
            for column in (dataset.instrument_col, dataset.decision_col)
            if column not in filtered.columns
        ]
        if missing:
            raise SchemaValidationError(f"filtered_frame is missing key columns {missing}")
        if bool(
            filtered.duplicated(
                subset=[dataset.instrument_col, dataset.decision_col]
            ).any()
        ):
            raise SchemaValidationError("filtered_frame has duplicate key rows")


def build_dataset(
    features: FeatureResult | pd.DataFrame,
    labels: LabelResult | pd.DataFrame,
    *,
    universe: pd.DataFrame | None = None,
    require_universe: bool = False,
    filter_universe: bool = False,
    feature_names: Sequence[str] | None = None,
    label_name: str | None = None,
    source_hashes: Mapping[str, str] | None = None,
    instrument_col: str = "symbol",
    decision_col: str = "decision_timestamp",
) -> ResearchDataset:
    """Join one feature row and one label row per symbol/decision key.

    Missing sides are retained with explicit status. Universe membership is
    optional; when supplied, missing membership is reported as unknown rather
    than treated as inclusion. Filtering requires ``filter_universe=True`` and
    reports the removed rows.
    """

    if not isinstance(features, (FeatureResult, pd.DataFrame)):
        raise SchemaValidationError("features must be a FeatureResult or DataFrame")
    if not isinstance(labels, (LabelResult, pd.DataFrame)):
        raise SchemaValidationError("labels must be a LabelResult or DataFrame")
    if filter_universe and universe is None:
        raise SchemaValidationError("filter_universe=True requires a universe frame")
    keys = [instrument_col, decision_col]
    if isinstance(features, FeatureResult):
        feature_frame = features.frame.copy()
        names = list(feature_names or features.feature_names)
    else:
        feature_frame = features.copy()
        names = list(feature_names) if feature_names is not None else _infer_feature_names(
            feature_frame, keys
        )
    if not names:
        raise SchemaValidationError("At least one feature column is required")
    if len(set(names)) != len(names):
        raise SchemaValidationError("Feature names must be unique")
    reserved = (
        _DATASET_RESERVED_COLUMNS
        | set(LABEL_METADATA_COLUMNS)
        | set(GLOBAL_METADATA_COLUMNS)
    )
    conflicting_names = [name for name in names if name in reserved or name in keys]
    if conflicting_names:
        raise SchemaValidationError(
            f"Feature names collide with dataset metadata columns {conflicting_names}"
        )
    suffix_names = [
        name
        for name in names
        if any(name.endswith(suffix) for suffix in FEATURE_METADATA_SUFFIXES)
    ]
    if suffix_names:
        raise SchemaValidationError(
            f"Feature names use reserved metadata suffixes: {suffix_names}"
        )
    missing = [name for name in names if name not in feature_frame.columns]
    if missing:
        raise SchemaValidationError(f"Missing feature columns {missing}")

    if isinstance(labels, LabelResult):
        label_frame = labels.frame.copy()
        chosen_label = label_name or labels.label_name
    else:
        label_frame = labels.copy()
        chosen_label = _infer_label_name(label_frame, keys, label_name)
    if chosen_label in names:
        raise SchemaValidationError("Label name cannot duplicate a feature name")
    if chosen_label in keys:
        raise SchemaValidationError("Label name cannot duplicate a dataset key column")
    if chosen_label in reserved:
        raise SchemaValidationError(
            f"Label name {chosen_label!r} collides with a dataset metadata column"
        )

    feature_frame = normalize_decisions(
        feature_frame, instrument_col=instrument_col, decision_col=decision_col
    )
    feature_frame = _prepare_features(
        feature_frame, names, decision_col=decision_col
    )
    label_frame = normalize_decisions(
        label_frame, instrument_col=instrument_col, decision_col=decision_col
    )
    label_frame = _prepare_labels(
        label_frame,
        chosen_label,
        decision_col=decision_col,
        require_intervals=isinstance(labels, LabelResult),
    )
    feature_metadata = _feature_metadata_columns(
        feature_frame.columns, keys, names
    )
    if isinstance(features, FeatureResult):
        allowed_metadata = set(GLOBAL_METADATA_COLUMNS) | {
            f"{name}{suffix}"
            for name in names
            for suffix in FEATURE_METADATA_SUFFIXES
        }
        feature_metadata = [
            column for column in feature_metadata if column in allowed_metadata
        ]
    label_metadata = _label_metadata_columns(
        label_frame.columns, keys, chosen_label
    )
    feature_columns = keys + names + feature_metadata
    label_columns = keys + [chosen_label] + label_metadata
    collisions = (set(feature_columns) & set(label_columns)) - set(keys)
    if collisions:
        raise SchemaValidationError(
            f"Feature and label columns collide: {sorted(collisions)}"
        )

    for name in names:
        numeric = pd.to_numeric(feature_frame[name], errors="coerce")
        non_numeric = numeric.isna() & feature_frame[name].notna()
        if bool(non_numeric.any()):
            raise SchemaValidationError(f"Feature {name!r} must be numeric")
        if bool(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any()):
            raise SchemaValidationError(f"Feature {name!r} contains infinity")
    label_numeric = pd.to_numeric(label_frame[chosen_label], errors="coerce")
    non_numeric_label = label_numeric.isna() & label_frame[chosen_label].notna()
    if bool(non_numeric_label.any()):
        raise SchemaValidationError("Label values must be numeric")
    if bool(np.isinf(label_numeric.to_numpy(dtype=float, na_value=np.nan)).any()):
        raise SchemaValidationError("Label values contain infinity")

    merged = pd.merge(
        feature_frame[feature_columns],
        label_frame[label_columns],
        on=keys,
        how="outer",
        suffixes=("", "__label"),
        indicator=True,
    )
    indicator = merged.pop("_merge")
    missing_features = indicator.eq("right_only")
    missing_labels = indicator.eq("left_only")
    label_status = merged["label_status"].fillna("missing_label").astype(str)
    merged["label_status"] = label_status
    status = pd.Series("ok", index=merged.index, dtype=object)
    reason = pd.Series("", index=merged.index, dtype=object)
    status.loc[missing_features] = "missing_features"
    reason.loc[missing_features] = "no matching feature row"
    status.loc[missing_labels] = "missing_label"
    reason.loc[missing_labels] = "no matching label row"
    both = ~(missing_features | missing_labels)
    label_not_ok = both & label_status.ne("ok")
    status.loc[label_not_ok] = "label_not_ok"
    reason.loc[label_not_ok] = merged.loc[label_not_ok, "label_reason"].fillna("").astype(str)
    merged["dataset_status"] = status
    merged["dataset_reason"] = reason

    rows_before_universe = len(merged)
    filtered = pd.DataFrame(columns=merged.columns)
    filtered_reasons: dict[str, int] = {}
    universe_source_hash = hash_payload({"universe": None})
    if universe is not None:
        universe_frame = _normalize_universe(
            universe, instrument_col=instrument_col, decision_col=decision_col
        )
        merged = pd.merge(
            merged,
            universe_frame[
                keys
                + [
                    "universe_eligible",
                    "universe_available_at",
                    "universe_membership_status",
                ]
            ],
            on=keys,
            how="left",
            indicator="universe_merge",
        )
        universe_merge = merged.pop("universe_merge")
        known = universe_merge.eq("both")
        merged["universe_status"] = np.where(known, "known", "unknown_membership")
        merged["universe_eligible"] = merged["universe_eligible"].astype("boolean")
        eligibility_known = merged["universe_eligible"].notna()
        if require_universe and not bool((known & eligibility_known).all()):
            missing_count = int((~(known & eligibility_known)).sum())
            raise IncompleteUniverseError(
                f"{missing_count} dataset row(s) have no PIT-valid universe membership"
            )
        if filter_universe:
            eligible = merged["universe_eligible"].eq(True).fillna(False)
            keep = known & eligible
            filtered = merged.loc[~keep].copy()
            unknown_count = int((~known).sum())
            ineligible_count = int((known & ~eligible).sum())
            if unknown_count:
                filtered_reasons["unknown_membership"] = unknown_count
            if ineligible_count:
                filtered_reasons["not_eligible"] = ineligible_count
            merged = merged.loc[keep].copy()
    elif require_universe:
        raise IncompleteUniverseError("require_universe=True requires a universe frame")

    merged = merged.sort_values(keys, kind="stable").reset_index(drop=True)
    feature_values = [name for name in names]
    output_columns = keys + ["dataset_status", "dataset_reason"]
    if universe is not None:
        output_columns += [
            "universe_status",
            "universe_eligible",
            "universe_available_at",
            "universe_membership_status",
        ]
    output_columns += feature_values + feature_metadata + [chosen_label] + label_metadata
    output_columns = [column for column in output_columns if column in merged.columns]
    merged = merged[output_columns]

    if isinstance(features, FeatureResult):
        selected_names = set(names)
        selected_specs = [
            spec
            for spec in (features.manifest.get("specs") or [])
            if spec.get("name") in selected_names
        ]
        feature_spec_hash = hash_payload(
            {
                "schema_version": features.manifest.get("schema_version"),
                "feature_names": names,
                "specs": selected_specs,
                "instrument_column": features.manifest.get("instrument_column"),
                "decision_column": features.manifest.get("decision_column"),
                "frequency": features.manifest.get("frequency"),
            }
        )
    else:
        feature_spec_hash = hash_payload(
            {"schema_version": "quantforge.features.v1", "feature_names": names}
        )
    if isinstance(labels, LabelResult):
        label_spec_hash = hash_payload(
            {
                "schema_version": labels.manifest.get("schema_version"),
                "label_name": chosen_label,
                "spec": labels.manifest.get("spec"),
                "instrument_column": labels.manifest.get("instrument_column"),
                "decision_column": labels.manifest.get("decision_column"),
                "horizon_unit": labels.manifest.get("horizon_unit"),
                "frequency": labels.manifest.get("frequency"),
                "execution_convention": labels.manifest.get("execution_convention"),
                "backtest_compatibility": labels.manifest.get("backtest_compatibility"),
            }
        )
    else:
        label_spec_hash = hash_payload(
            {
                "schema_version": "quantforge.labels.v1",
                "label_name": chosen_label,
            }
        )
    feature_content_hash = hash_frame(
        merged,
        key_cols=keys,
        columns=keys + names,
        schema_version="quantforge.features.v1",
    )
    feature_source_hash = hash_frame(
        merged,
        key_cols=keys,
        columns=keys + feature_metadata,
        schema_version="quantforge.features.v1",
    )
    label_content_hash = hash_frame(
        merged,
        key_cols=keys,
        columns=keys + [chosen_label],
        schema_version="quantforge.labels.v1",
    )
    label_source_hash = hash_frame(
        merged,
        key_cols=keys,
        columns=keys + label_metadata,
        schema_version="quantforge.labels.v1",
    )
    dataset_content_hash = hash_frame(
        merged,
        key_cols=keys,
        columns=list(merged.columns),
        schema_version="quantforge.dataset.v1",
    )
    if universe is not None:
        universe_source_hash = hash_frame(
            merged,
            key_cols=keys,
            columns=keys
            + [
                "universe_eligible",
                "universe_available_at",
                "universe_membership_status",
            ],
            schema_version="quantforge.universe.v1",
        )
    source_hash = hash_payload(
        {
            "feature_source": feature_source_hash,
            "label_source": label_source_hash,
            "universe_source": universe_source_hash,
            "source_hashes": dict(source_hashes or {}),
        }
    )
    quality = {
        "feature_rows": int(len(feature_frame)),
        "label_rows": int(len(label_frame)),
        "rows_before_universe": int(rows_before_universe),
        "rows": int(len(merged)),
        "missing_features": int(missing_features.sum()),
        "missing_labels": int(missing_labels.sum()),
        "label_status_counts": dict(Counter(label_status.tolist())),
        "dataset_status_counts": dict(Counter(merged["dataset_status"].tolist())),
        "filtered_rows": int(len(filtered)),
        "filtered_reasons": filtered_reasons,
    }
    manifest: dict[str, Any] = {
        "schema_version": "quantforge.dataset.v1",
        "feature_schema": "quantforge.features.v1",
        "label_schema": "quantforge.labels.v1",
        "feature_names": names,
        "label_name": chosen_label,
        "feature_metadata_columns": list(feature_metadata),
        "label_metadata_columns": list(label_metadata),
        "feature_spec_hash": feature_spec_hash,
        "feature_content_hash": feature_content_hash,
        "feature_source_hash": feature_source_hash,
        "label_spec_hash": label_spec_hash,
        "label_content_hash": label_content_hash,
        "label_source_hash": label_source_hash,
        "universe_source_hash": universe_source_hash,
        "source_hash": source_hash,
        "source_hashes": dict(source_hashes or {}),
        "dataset_content_hash": dataset_content_hash,
        "rows": int(len(merged)),
        "columns": output_columns,
        "filters": {
            "require_universe": require_universe,
            "filter_universe": filter_universe,
            "removed": filtered_reasons,
        },
        "quality": quality,
        "canonicalization": {
            "encoding": "type-tagged JSON; SHA-256",
            "row_order": "stable sort by symbol, decision_timestamp",
            "column_order": "explicit fixed list",
            "nan": "distinct from null and the string 'NaN'",
            "infinity": "rejected",
        },
    }
    manifest_hash = hash_payload(manifest)
    manifest["manifest_hash"] = manifest_hash
    manifest["dataset_id"] = hash_payload(
        {"schema_version": "quantforge.dataset.v1", "manifest_hash": manifest_hash}
    )
    return ResearchDataset(
        frame=merged,
        feature_names=tuple(names),
        label_name=chosen_label,
        quality=quality,
        manifest=manifest,
        filtered_frame=(
            filtered.sort_values(keys, kind="stable").reset_index(drop=True)
            if not filtered.empty
            else filtered.reset_index(drop=True)
        ),
        filtered_reasons=dict(filtered_reasons),
        instrument_col=instrument_col,
        decision_col=decision_col,
    )
