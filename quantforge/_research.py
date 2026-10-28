"""Shared validation helpers for the Module 2 research engines."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .data.errors import ConflictingRevisionsError, SchemaValidationError, TemporalError
from .data.pipeline import normalize_temporal
from .data.timeutils import ensure_utc_series


def validate_instruments(frame: pd.DataFrame, *, instrument_col: str) -> None:
    if instrument_col not in frame.columns:
        raise SchemaValidationError(f"Missing instrument column {instrument_col!r}")
    values = frame[instrument_col]
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise SchemaValidationError("Instrument keys must be non-empty and non-null")


def normalize_decisions(
    frame: pd.DataFrame,
    *,
    instrument_col: str,
    decision_col: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise SchemaValidationError("decisions must be a pandas DataFrame")
    missing = [c for c in (instrument_col, decision_col) if c not in frame.columns]
    if missing:
        raise SchemaValidationError(f"Missing decision columns {missing}")
    out = frame.copy()
    validate_instruments(out, instrument_col=instrument_col)
    out[decision_col] = ensure_utc_series(out[decision_col], name=decision_col)
    if bool(out.duplicated(subset=[instrument_col, decision_col]).any()):
        raise SchemaValidationError("Decision keys must be unique by instrument and timestamp")
    return out.sort_values([instrument_col, decision_col], kind="stable").reset_index(drop=True)


def select_frequency(
    frame: pd.DataFrame,
    frequency: str | None,
    *,
    context: str,
) -> pd.DataFrame:
    if "frequency" not in frame.columns:
        if frequency is not None:
            raise SchemaValidationError(
                f"{context}: frequency={frequency!r} requested but no frequency column exists"
            )
        return frame
    values = frame["frequency"].astype(str)
    distinct = sorted(values.dropna().unique())
    if frequency is None:
        if len(distinct) > 1:
            raise SchemaValidationError(
                f"{context}: mixed frequencies {distinct}; pass frequency= explicitly"
            )
        return frame
    selected = frame.loc[values.eq(str(frequency))].copy()
    if selected.empty:
        raise SchemaValidationError(
            f"{context}: no rows for frequency={frequency!r}; available={distinct}"
        )
    return selected


def normalize_bars(
    frame: pd.DataFrame,
    *,
    instrument_col: str,
    event_col: str,
    availability_col: str,
    frequency: str | None,
    context: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise SchemaValidationError("bars must be a pandas DataFrame")
    missing = [c for c in (instrument_col, event_col, availability_col) if c not in frame.columns]
    if missing:
        raise SchemaValidationError(f"{context}: missing bar columns {missing}")
    work = select_frequency(frame.copy(), frequency, context=context)
    work = normalize_temporal(work)
    validate_instruments(work, instrument_col=instrument_col)
    if "revision" not in work.columns:
        work["revision"] = 0
    else:
        numeric, invalid = coerce_revision(work["revision"])
        if bool(invalid.any()):
            raise SchemaValidationError("revision must be a non-negative integer")
        work["revision"] = numeric.astype("int64")
    return work.sort_values(
        [instrument_col, event_col, "revision", availability_col], kind="stable"
    ).reset_index(drop=True)


def coerce_revision(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = numeric.isna()
    finite = numeric.notna() & np.isfinite(numeric)
    invalid |= numeric.notna() & ~finite
    invalid |= finite & ((numeric < 0) | (numeric % 1 != 0))
    return numeric, invalid


def resolve_event_revisions(
    frame: pd.DataFrame,
    *,
    instrument_col: str,
    event_col: str,
    availability_col: str,
    revision_col: str = "revision",
    value_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    if revision_col not in work.columns:
        work[revision_col] = 0
    numeric, invalid = coerce_revision(work[revision_col])
    if bool(invalid.any()):
        raise SchemaValidationError("revision must be a non-negative integer")
    work[revision_col] = numeric.astype("int64")
    if value_cols is None:
        value_cols = [
            c
            for c in work.columns
            if c
            not in {
                instrument_col,
                event_col,
                revision_col,
                availability_col,
                "source",
                "ingestion_timestamp",
            }
        ]
    else:
        missing = [c for c in value_cols if c not in work.columns]
        if missing:
            raise SchemaValidationError(f"Missing revision payload columns {missing}")
        value_cols = list(value_cols)

    kept: list[pd.Series] = []
    for key, group in work.groupby([instrument_col, event_col], sort=False, dropna=False):
        max_revision = group[revision_col].max()
        top = group[group[revision_col] == max_revision]
        if top[list(value_cols)].drop_duplicates().shape[0] > 1:
            raise ConflictingRevisionsError(
                f"Conflicting records for {instrument_col}={key[0]!r} at event={key[1]} "
                f"revision={max_revision}"
            )
        kept.append(top.sort_values([revision_col, availability_col], kind="stable").iloc[-1])
    if not kept:
        return work.iloc[0:0].copy()
    out = pd.DataFrame(kept).reset_index(drop=True)
    return out.sort_values([instrument_col, event_col], kind="stable").reset_index(drop=True)


def validate_finite_inputs(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            raise SchemaValidationError(f"Missing feature input column {column!r}")
        numeric = pd.to_numeric(frame[column], errors="coerce")
        non_numeric = numeric.isna() & frame[column].notna()
        if bool(non_numeric.any()):
            raise SchemaValidationError(f"Feature input {column!r} must be numeric")
        if bool(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any()):
            raise SchemaValidationError(f"Feature input {column!r} contains infinity")


def max_availability(frame: pd.DataFrame, *, availability_col: str) -> pd.Timestamp | None:
    if frame.empty or availability_col not in frame.columns:
        return None
    values = pd.to_datetime(frame[availability_col], utc=True, errors="coerce").dropna()
    if values.empty:
        return None
    return pd.Timestamp(values.max())


def latest_availability_map(
    frame: pd.DataFrame,
    *,
    instrument_col: str,
    event_col: str,
    availability_col: str,
) -> dict[tuple[Any, pd.Timestamp], pd.Timestamp]:
    if frame.empty:
        return {}
    work = frame[[instrument_col, event_col, availability_col]].copy()
    work[event_col] = pd.to_datetime(work[event_col], utc=True, errors="coerce")
    work[availability_col] = pd.to_datetime(work[availability_col], utc=True, errors="coerce")
    work = work.dropna(subset=[event_col, availability_col])
    result: dict[tuple[Any, pd.Timestamp], pd.Timestamp] = {}
    for (instrument, event), group in work.groupby([instrument_col, event_col], sort=False):
        result[(instrument, pd.Timestamp(event))] = pd.Timestamp(group[availability_col].max())
    return result


def require_utc(value: Any, *, name: str) -> pd.Timestamp:
    if value is None or pd.isna(value):
        raise TemporalError(f"{name} is required")
    timestamp = pd.Timestamp(value)
    if timestamp is pd.NaT or timestamp.tzinfo is None:
        raise TemporalError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")
