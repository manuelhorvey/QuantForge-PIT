"""Deterministic, point-in-time data pipeline.

Temporal semantics
------------------
* Canonical timezone: **UTC**. Naive/ambiguous timestamps are rejected.
* ``timestamp`` (a.k.a. event time) is when the underlying event occurred; for a
  bar it is the bar end/event time.
* ``publication_timestamp`` is when the source published the information.
* ``available_at`` is when the information became usable by a strategy. The PIT
  rule is inclusive: a record is usable at decision ``t`` iff
  ``available_at <= t``. A record stamped exactly at ``t`` is usable.
* ``ingestion_timestamp`` is when our system recorded the record.
* ``decision_timestamp`` is when a strategy acts.
* ``valid_from``/``valid_to`` delimit a version, as a half-open interval
  ``[valid_from, valid_to)``.

A timestamp filter alone does not remove all look-ahead bias. These helpers
enforce the mechanical invariants; vendor semantics, restatement vintages, and
lifecycle coverage remain data-quality dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from .contracts import DataContractError
from .errors import (
    ConflictingRevisionsError,
    PointInTimeError,
    SchemaValidationError,
    TemporalError,
)
from .quality import (
    QualityReport,
    RejectedRecord,
    assess_frame,
    coverage_stats,
    freshness_stats,
    missing_value_stats,
)
from .timeutils import ensure_utc, ensure_utc_series, utc_now

DEFAULT_TEMPORAL_COLUMNS = (
    "timestamp",
    "publication_timestamp",
    "available_at",
    "ingestion_timestamp",
    "decision_timestamp",
    "valid_from",
    "valid_to",
)


@dataclass(frozen=True)
class TemporalContract:
    """Column names and the inclusive availability convention."""

    event_col: str = "timestamp"
    publication_col: str | None = "publication_timestamp"
    availability_col: str = "available_at"
    ingestion_col: str | None = "ingestion_timestamp"
    decision_col: str = "decision_timestamp"
    valid_from_col: str | None = "valid_from"
    valid_to_col: str | None = "valid_to"
    instrument_col: str = "symbol"
    inclusive_availability: bool = True


DEFAULT_CONTRACT = TemporalContract()
MARKET_REQUIRED = ("timestamp", "symbol", "open", "high", "low", "close", "volume")
MARKET_NUMERIC = ("open", "high", "low", "close", "volume")


@dataclass
class NormalizationResult:
    """Accepted/rejected rows plus the report that explains every rejection."""

    accepted: pd.DataFrame
    rejected: pd.DataFrame
    report: QualityReport
    assumptions: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.report.rejected_count == 0 and not self.accepted.empty


def _raise_temporal_series(series: pd.Series, *, name: str) -> pd.Series:
    try:
        return ensure_utc_series(series, name=name)
    except TemporalError:
        raise
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TemporalError(f"{name} could not be parsed") from exc


def normalize_temporal(
    frame: pd.DataFrame,
    *,
    contract: TemporalContract = DEFAULT_CONTRACT,
    require_availability: bool = True,
) -> pd.DataFrame:
    """Strictly convert every present temporal column and enforce ordering.

    Ordering enforced when the columns are present:
    ``event <= publication <= availability <= ingestion`` and
    ``valid_from < valid_to``.
    """

    out = frame.copy()
    present = [c for c in DEFAULT_TEMPORAL_COLUMNS if c in out.columns]
    optional = {
        contract.publication_col,
        contract.ingestion_col,
        contract.valid_from_col,
        contract.valid_to_col,
    }
    for column in present:
        out[column] = ensure_utc_series(out[column], name=column, allow_missing=column in optional)
    if require_availability and contract.availability_col not in out.columns:
        raise SchemaValidationError(f"Missing required availability column {contract.availability_col!r}")

    event = contract.event_col
    available = contract.availability_col
    if event in out and available in out:
        bad = out[available] < out[event]
        if bad.any():
            raise TemporalError("available_at precedes event timestamp")
    publication = contract.publication_col
    if publication and publication in out and event in out:
        if (out[publication] < out[event]).any():
            raise TemporalError("publication_timestamp precedes event timestamp")
        if available in out and (out[available] < out[publication]).any():
            raise TemporalError("available_at precedes publication_timestamp")
    ingestion = contract.ingestion_col
    if ingestion and ingestion in out and available in out:
        if (out[ingestion] < out[available]).any():
            raise TemporalError("ingestion_timestamp precedes available_at")
    valid_from, valid_to = contract.valid_from_col, contract.valid_to_col
    if valid_from and valid_to and valid_from in out and valid_to in out:
        both = out[valid_to].notna()
        if (both & (out[valid_to] <= out[valid_from])).any():
            raise TemporalError("valid_to must be strictly after valid_from")
    return out


def assert_point_in_time(
    frame: pd.DataFrame,
    decision: Any,
    *,
    availability_col: str = "available_at",
    inclusive: bool = True,
    decision_name: str = "decision_timestamp",
) -> None:
    """Raise if any row became available after its decision time."""

    if availability_col not in frame.columns:
        raise SchemaValidationError(f"Missing {availability_col!r}")
    available = ensure_utc_series(frame[availability_col], name=availability_col)
    if isinstance(decision, pd.Series) or (
        not isinstance(decision, (str, pd.Timestamp)) and hasattr(decision, "__len__")
    ):
        decisions = ensure_utc_series(decision, name=decision_name)
        if len(decisions) != len(frame):
            raise SchemaValidationError("decision series length mismatch")
        if not decisions.index.equals(frame.index):
            decisions = pd.Series(decisions.to_numpy(), index=frame.index)
    else:
        scalar = ensure_utc(decision, name=decision_name)
        decisions = pd.Series(pd.array([scalar] * len(frame), dtype="datetime64[ns, UTC]"), index=frame.index)
    violations = available >= decisions if not inclusive else available > decisions
    if bool(violations.any()):
        count = int(violations.sum())
        raise PointInTimeError(f"Point-in-time violation: {count} record(s) available after decision")


def _assert_publication_consistency(
    frame: pd.DataFrame,
    *,
    available_col: str = "available_at",
    publication_col: str = "publication_timestamp",
) -> None:
    """Fail closed when a record claims publication after it was available."""

    if publication_col not in frame.columns:
        return
    available = _raise_temporal_series(frame[available_col], name=available_col)
    publication = _raise_temporal_series(frame[publication_col], name=publication_col)
    bad = publication.notna() & available.notna() & (publication > available)
    if bool(bad.any()):
        count = int(bad.sum())
        raise PointInTimeError(
            f"{count} record(s) have publication_timestamp after available_at"
        )


def _select_frequency(
    frame: pd.DataFrame,
    frequency: str | None,
    *,
    context: str,
) -> pd.DataFrame:
    """Require an explicit frequency when a frame contains multiple frequencies."""

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


def _aggregate(values: pd.Series, how: str, *, min_periods: int) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < min_periods:
        return float("nan")
    if how == "mean":
        return float(clean.mean())
    if how == "sum":
        return float(clean.sum())
    if how == "std":
        return float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    if how == "median":
        return float(clean.median())
    if how == "last":
        return float(clean.iloc[-1])
    if how == "count":
        return float(len(clean))
    raise ValueError(f"Unsupported aggregation {how!r}")


def pit_rolling(
    frame: pd.DataFrame,
    value_col: str,
    window: int,
    *,
    agg: str = "mean",
    contract: TemporalContract = DEFAULT_CONTRACT,
    decision_col: str | None = None,
    by: str | None = None,
    min_periods: int | None = None,
    inclusive: bool = True,
    frequency: str | None = None,
) -> pd.Series:
    """Trailing rolling feature where eligibility is decided per decision time.

    For every row, only same-instrument observations whose ``available_at`` is at
    or before that row's decision time (and whose event time is not later) can
    contribute. Duplicate event timestamps keep the highest revision available at
    that decision; availability is only a tie-breaker.
    """

    if window < 1:
        raise ValueError("window must be >= 1")
    decision_col = decision_col or contract.decision_col
    by = by or contract.instrument_col
    for column in (by, contract.event_col, contract.availability_col, decision_col, value_col):
        if column not in frame.columns:
            raise SchemaValidationError(f"Missing required column {column!r}")
    if min_periods is None:
        min_periods = window
    if min_periods < 0 or min_periods > window:
        raise ValueError("min_periods must be within [0, window]")

    work = _select_frequency(frame.copy(), frequency, context="pit_rolling")
    work[contract.event_col] = _raise_temporal_series(work[contract.event_col], name=contract.event_col)
    work[contract.availability_col] = _raise_temporal_series(
        work[contract.availability_col], name=contract.availability_col
    )
    work[decision_col] = _raise_temporal_series(work[decision_col], name=decision_col)
    work[value_col] = pd.to_numeric(work[value_col], errors="raise")
    _assert_publication_consistency(
        work,
        available_col=contract.availability_col,
        publication_col=contract.publication_col or "publication_timestamp",
    )
    if "revision" in work.columns:
        numeric_revision, invalid_revision = _coerce_revision(work["revision"])
        if bool(invalid_revision.any()):
            raise SchemaValidationError("revision must be a non-negative integer")
        work["revision"] = numeric_revision.astype("int64")
    else:
        work["revision"] = 0

    result = pd.Series(np.nan, index=work.index, dtype=float, name=f"{value_col}_{agg}_{window}")
    for index, row in work.iterrows():
        decision = row[decision_col]
        eligibility = work[by].eq(row[by]) & (work[contract.event_col] <= decision)
        eligibility &= (
            work[contract.availability_col] <= decision
            if inclusive
            else work[contract.availability_col] < decision
        )
        eligible = work.loc[eligibility].sort_values(
            [contract.event_col, "revision", contract.availability_col], kind="stable"
        )
        eligible = eligible.drop_duplicates(subset=[contract.event_col], keep="last")
        result.loc[index] = _aggregate(
            eligible[value_col].tail(window), agg, min_periods=min_periods
        )
    return result


def pit_asof_join(
    decisions: pd.DataFrame,
    records: pd.DataFrame,
    *,
    by: str = "symbol",
    decision_col: str = "decision_timestamp",
    available_col: str = "available_at",
    revision_col: str | None = "revision",
    inclusive: bool = True,
    value_cols: Sequence[str] | None = None,
    frequency: str | None = None,
) -> pd.DataFrame:
    """Backward-only as-of join keyed per instrument using availability time.

    Duplicate records at the same ``(by, available_col)`` are resolved by highest
    revision; conflicting payloads at the same revision raise instead of being
    silently chosen. Joins never cross instruments because pandas' ``by=``
    grouping is used. Mixed-frequency input requires an explicit ``frequency``.
    """

    if by not in decisions or by not in records:
        raise SchemaValidationError(f"Join key {by!r} missing from one side")
    if decision_col not in decisions:
        raise SchemaValidationError(f"Missing decision column {decision_col!r}")
    if available_col not in records:
        raise SchemaValidationError(f"Missing availability column {available_col!r}")
    if decisions[by].isna().any() or records[by].isna().any():
        raise SchemaValidationError("Join key cannot be null")

    left = decisions.copy()
    left[decision_col] = _raise_temporal_series(left[decision_col], name=decision_col)
    right = _select_frequency(records.copy(), frequency, context="pit_asof_join")
    right[available_col] = _raise_temporal_series(right[available_col], name=available_col)
    _assert_publication_consistency(right, available_col=available_col)

    revision = revision_col if revision_col and revision_col in right.columns else None
    if revision:
        numeric_revision, invalid_revision = _coerce_revision(right[revision])
        if bool(invalid_revision.any()):
            raise SchemaValidationError("revision must be a non-negative integer")
        right[revision] = numeric_revision.astype("int64")
    internal_rev = revision or "__qf_revision"
    if not revision:
        right[internal_rev] = 0

    if value_cols is None:
        ignored = {by, available_col, internal_rev, "source", "ingestion_timestamp"}
        value_cols = [c for c in right.columns if c not in ignored]

    prepared: list[pd.DataFrame] = []
    for (_, _), group in right.groupby([by, available_col], sort=False, dropna=False):
        max_revision = group[internal_rev].max()
        top = group[group[internal_rev] == max_revision]
        payload = top[list(value_cols)]
        if payload.drop_duplicates().shape[0] > 1:
            key = group.iloc[0][by]
            stamp = group.iloc[0][available_col]
            raise ConflictingRevisionsError(
                f"Conflicting records for {by}={key!r} at availability={stamp} "
                f"revision={max_revision}"
            )
        prepared.append(top.sort_values(internal_rev, kind="stable").iloc[[-1]])
    resolved = pd.concat(prepared, ignore_index=True) if prepared else right.iloc[0:0].copy()
    resolved = resolved.sort_values([available_col, by], kind="stable")

    left = left.sort_values([decision_col, by], kind="stable")
    merged = pd.merge_asof(
        left,
        resolved,
        left_on=decision_col,
        right_on=available_col,
        by=by,
        direction="backward",
        allow_exact_matches=inclusive,
        suffixes=("", "_record"),
    )
    if available_col in merged.columns:
        known = merged[available_col].notna()
        bad = (
            merged.loc[known, available_col] > merged.loc[known, decision_col]
            if inclusive
            else merged.loc[known, available_col] >= merged.loc[known, decision_col]
        )
        if bool(bad.any()):
            raise PointInTimeError("As-of join attached a record unavailable at decision time")
    merged["matched"] = merged[available_col].notna() if available_col in merged else False
    if not revision and internal_rev in merged.columns:
        merged = merged.drop(columns=[internal_rev])
    return merged.sort_values([decision_col, by], kind="stable").reset_index(drop=True)


def resample_ohlcv(
    frame: pd.DataFrame,
    rule: str,
    *,
    contract: TemporalContract = DEFAULT_CONTRACT,
    label: Literal["left", "right"] = "right",
    closed: Literal["left", "right"] = "right",
    origin: str = "start_day",
    by: str | None = None,
    drop_empty: bool = True,
    frequency: str | None = None,
) -> pd.DataFrame:
    """Resample OHLCV with explicit interval closure/labelling.

    ``label='right', closed='right'`` (the safe default) labels a bar by the end
    of its interval. The output ``available_at`` is the maximum availability of
    constituent bars and never earlier than the output timestamp, so a later
    higher-frequency bar cannot leak backwards into an earlier resampled bar.
    Mixed-frequency input requires an explicit ``frequency`` selection.
    """

    if label not in {"left", "right"} or closed not in {"left", "right"}:
        raise ValueError("label and closed must be 'left' or 'right'")
    by = by or contract.instrument_col
    required = [contract.event_col, contract.availability_col, by, "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise SchemaValidationError(f"Missing columns for resample: {missing}")

    work = _select_frequency(frame.copy(), frequency, context="resample_ohlcv")
    work = normalize_temporal(work, contract=contract)
    work = work.sort_values([contract.event_col, by], kind="stable")
    event = contract.event_col
    available = contract.availability_col

    groupers: list[Any] = [by]
    groupers.append(pd.Grouper(key=event, freq=rule, label=label, closed=closed, origin=origin))
    aggregations: dict[str, Any] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        available: "max",
    }
    for optional in ("revision", "source"):
        if optional in work.columns:
            aggregations[optional] = "max" if optional == "revision" else "last"
    try:
        out = work.groupby(groupers, sort=True, dropna=False).agg(aggregations).reset_index()
    except ValueError as exc:
        raise TemporalError(f"Invalid resample specification: {exc}") from exc

    out[available] = out[[available, event]].max(axis=1)
    if drop_empty:
        out = out[out["open"].notna()].copy()
    out["bar_frequency"] = rule
    bad_envelope = (out["high"] < out[["open", "close", "low"]].max(axis=1)) | (
        out["low"] > out[["open", "close", "high"]].min(axis=1)
    )
    if bool(bad_envelope.any()):
        raise DataContractError("Resampling produced an invalid OHLC envelope")
    out = out.sort_values([event, by], kind="stable").reset_index(drop=True)
    out.attrs["resample"] = {"rule": rule, "label": label, "closed": closed, "origin": origin}
    return out


def flag_stale(
    frame: pd.DataFrame,
    *,
    decision_col: str = "decision_timestamp",
    availability_col: str = "available_at",
    max_age: str | pd.Timedelta = "5D",
    inclusive: bool = True,
) -> pd.DataFrame:
    """Add ``data_age`` and ``is_stale`` columns without dropping observations."""

    if decision_col not in frame or availability_col not in frame:
        raise SchemaValidationError("decision and availability columns are required")
    out = frame.copy()
    out[decision_col] = _raise_temporal_series(out[decision_col], name=decision_col)
    out[availability_col] = _raise_temporal_series(out[availability_col], name=availability_col)
    ages = out[decision_col] - out[availability_col]
    if bool((ages < pd.Timedelta(0)).any()):
        raise PointInTimeError("available_at is later than decision_timestamp")
    limit = pd.Timedelta(max_age)
    if limit < pd.Timedelta(0):
        raise ValueError("max_age cannot be negative")
    out["data_age"] = ages
    out["is_stale"] = ages > limit if inclusive else ages >= limit
    return out


def resolve_revisions(
    frame: pd.DataFrame,
    *,
    key_cols: Sequence[str],
    revision_col: str = "revision",
    available_col: str = "available_at",
    value_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Keep the deterministic latest revision per key; reject same-revision conflicts."""

    if not key_cols:
        raise ValueError("key_cols cannot be empty")
    missing = [c for c in key_cols if c not in frame.columns]
    if missing:
        raise SchemaValidationError(f"Missing key columns {missing}")
    work = frame.copy()
    work[available_col] = _raise_temporal_series(work[available_col], name=available_col)
    if revision_col in work.columns:
        numeric_revision, invalid_revision = _coerce_revision(work[revision_col])
        if bool(invalid_revision.any()):
            raise SchemaValidationError("revision must be a non-negative integer")
        work[revision_col] = numeric_revision.astype("int64")
    else:
        work[revision_col] = 0
    if value_cols is None:
        value_cols = [
            c for c in work.columns if c not in {*key_cols, revision_col, available_col, "source", "ingestion_timestamp"}
        ]

    kept: list[pd.Series] = []
    for _, group in work.groupby(list(key_cols), sort=False, dropna=False):
        max_revision = group[revision_col].max()
        top = group[group[revision_col] == max_revision]
        if top[list(value_cols)].drop_duplicates().shape[0] > 1:
            raise ConflictingRevisionsError(
                f"Conflicting revisions for key {key_cols}={tuple(group.iloc[0][list(key_cols)])}"
            )
        kept.append(top.sort_values([revision_col, available_col], kind="stable").iloc[-1])
    if not kept:
        return work.iloc[0:0].drop(columns=[revision_col]).copy()
    out = pd.DataFrame(kept).reset_index(drop=True)
    return out.sort_values(list(key_cols), kind="stable").reset_index(drop=True)


def _coerce_revision(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return numeric revisions and a mask of invalid (non-integral/negative) values."""

    numeric = pd.to_numeric(series, errors="coerce")
    invalid = numeric.isna() & series.notna()
    finite = numeric.notna() & np.isfinite(numeric)
    invalid |= numeric.notna() & ~finite
    invalid |= finite & (numeric < 0)
    invalid |= finite & (numeric % 1 != 0)
    return numeric, invalid


def _parse_temporal_column(
    series: pd.Series, *, name: str, allow_missing: bool = False
) -> tuple[pd.Series, list[str | None]]:
    values: list[pd.Timestamp] = []
    errors: list[str | None] = []
    for value in series:
        if allow_missing and (value is None or pd.isna(value)):
            values.append(pd.NaT)
            errors.append(None)
            continue
        try:
            values.append(ensure_utc(value, name=name))
            errors.append(None)
        except TemporalError as exc:
            values.append(pd.NaT)
            errors.append(str(exc))
    parsed = pd.Series(pd.array(values, dtype="datetime64[ns, UTC]"), index=series.index)
    return parsed, errors


def _rejection_frame(
    original: pd.DataFrame, reasons: pd.Series, *, extra: pd.DataFrame | None = None
) -> pd.DataFrame:
    mask = reasons.notna()
    rejected = original.loc[mask].copy()
    rejected["rejection_reason"] = reasons.loc[mask].astype(str)
    if extra is not None:
        for column in extra.columns:
            rejected[column] = extra.loc[mask, column]
    return rejected


def _record_rejections(report: QualityReport, rejected: pd.DataFrame, *, limit: int = 25) -> None:
    for index, row in rejected.iterrows():
        reason = str(row["rejection_reason"])
        report.add_rejection(reason, sample_limit=limit)
        if len(report.rejected_sample) < limit:
            record = {k: v for k, v in row.items() if k != "rejection_reason"}
            report.rejected_sample.append(
                RejectedRecord(row_index=index, reason=reason, record=record)
            )


def _deduplicate_preserving_revisions(
    frame: pd.DataFrame,
    *,
    key_cols: Sequence[str],
    revision_col: str,
    available_col: str,
    payload_cols: Sequence[str],
    conflict_reason: str,
    reasons: pd.Series,
) -> tuple[pd.DataFrame, int]:
    """Deduplicate exact rows while retaining every distinct valid revision.

    Same-revision rows with different payloads are ambiguous: every row in that
    revision group is marked with ``conflict_reason`` and omitted. Other
    revisions remain usable.
    """

    work = frame.copy()
    work["_input_order"] = np.arange(len(work))
    work = work.sort_values(
        [*key_cols, revision_col, available_col], kind="stable"
    )
    data_cols = [c for c in work.columns if c != "_input_order"]
    exact = work.duplicated(subset=data_cols, keep="first")
    duplicates = int(exact.sum())
    work = work.loc[~exact].copy()

    keep_indices: list[Any] = []
    for _, group in work.groupby([*key_cols, revision_col], sort=False, dropna=False):
        payload = group[list(payload_cols)].drop_duplicates()
        if payload.shape[0] > 1:
            for idx in group.index:
                reasons.loc[idx] = conflict_reason
            continue
        keep_indices.append(
            group.sort_values(available_col, kind="stable").iloc[-1].name
        )

    kept = work.loc[keep_indices].copy() if keep_indices else work.iloc[0:0].copy()
    return kept, max(duplicates, 0)


def _resolve_market_duplicates(
    frame: pd.DataFrame,
    reasons: pd.Series,
    *,
    contract: TemporalContract,
    value_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, int, int]:
    """Return (kept, reasons, duplicate_rows, ordering_violations).

    Distinct revisions are retained. Exact duplicates are removed once;
    conflicting payloads at the same revision are rejected explicitly.
    """

    key = [contract.event_col, contract.instrument_col]
    payload = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        contract.availability_col,
        "source",
    ]
    for optional in (contract.publication_col,):
        if optional and optional in frame.columns:
            payload.append(optional)
    payload = [c for c in payload if c in frame.columns]
    if value_cols is not None:
        missing = [c for c in value_cols if c not in frame.columns]
        if missing:
            raise SchemaValidationError(f"Missing value columns for conflict checks: {missing}")
        for column in value_cols:
            if column not in payload:
                payload.append(column)

    kept, duplicates = _deduplicate_preserving_revisions(
        frame,
        key_cols=key,
        revision_col="revision",
        available_col=contract.availability_col,
        payload_cols=payload,
        conflict_reason="conflicting_duplicate",
        reasons=reasons,
    )
    kept = kept.sort_values(
        [*key, "revision", contract.availability_col], kind="stable"
    ).reset_index(drop=True)
    ordering = 0
    if len(kept) > 1:
        ordering = int(kept["_input_order"].diff().lt(0).sum())
    return kept, reasons, duplicates, ordering


def normalize_market_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    ingestion_timestamp: Any | None = None,
    frequency: str = "1d",
    dataset: str = "bars",
    schema_version: str = "1",
    contract: TemporalContract = DEFAULT_CONTRACT,
    assume_available_at: Literal["reject", "event_time"] = "reject",
    value_cols: Sequence[str] | None = None,
) -> NormalizationResult:
    """Validate/normalize canonical bars without imputing or inventing data.

    ``available_at`` is required by default. Supplying
    ``assume_available_at='event_time'`` is an explicit, reported assumption, not
    a silent default.
    """

    if assume_available_at not in {"reject", "event_time"}:
        raise SchemaValidationError(
            "assume_available_at must be 'reject' or 'event_time'"
        )
    missing = [c for c in MARKET_REQUIRED if c not in frame.columns]
    if missing:
        raise SchemaValidationError(f"Missing required market columns: {missing}")
    original = frame.copy()
    original["_input_order"] = np.arange(len(original))
    reasons = pd.Series([None] * len(original), index=original.index, dtype=object)
    assumptions: list[str] = []

    work = original.copy()
    if contract.availability_col not in work.columns:
        if assume_available_at == "reject":
            raise SchemaValidationError(
                f"{contract.availability_col!r} is required; pass assume_available_at='event_time' "
                "only when that assumption is documented"
            )
        work[contract.availability_col] = work[contract.event_col]
        assumptions.append("available_at was assumed to equal event timestamp")

    timestamp_violations = 0
    parsed_event, event_errors = _parse_temporal_column(work[contract.event_col], name=contract.event_col)
    work[contract.event_col] = parsed_event
    parsed_available, available_errors = _parse_temporal_column(
        work[contract.availability_col], name=contract.availability_col
    )
    work[contract.availability_col] = parsed_available
    for position, idx in enumerate(work.index):
        err = event_errors[position]
        err2 = available_errors[position]
        if err or err2:
            timestamp_violations += 1
            reasons.loc[idx] = "; ".join(e for e in (err, err2) if e)

    publication = contract.publication_col
    if publication and publication in work.columns:
        parsed_publication, publication_errors = _parse_temporal_column(
            work[publication], name=publication, allow_missing=True
        )
        work[publication] = parsed_publication
        for position, idx in enumerate(work.index):
            if publication_errors[position]:
                timestamp_violations += 1
                reasons.loc[idx] = "; ".join(
                    e for e in (reasons.loc[idx], publication_errors[position]) if e
                )
        reasons.loc[reasons.isna() & (work[publication] < work[contract.event_col])] = (
            "publication_before_event"
        )
        reasons.loc[reasons.isna() & (work[publication] > work[contract.availability_col])] = (
            "availability_before_publication"
        )

    ingestion = ensure_utc(ingestion_timestamp, name="ingestion_timestamp") if ingestion_timestamp is not None else utc_now()
    work[contract.ingestion_col or "ingestion_timestamp"] = ingestion
    work["source"] = source
    work["frequency"] = frequency
    work["schema_version"] = schema_version
    if "revision" not in work.columns:
        work["revision"] = 0
    else:
        numeric_revision, invalid_revision = _coerce_revision(work["revision"])
        reasons.loc[reasons.isna() & invalid_revision] = "invalid_revision"
        work["revision"] = numeric_revision.where(
            numeric_revision.notna() & ~invalid_revision, 0
        ).astype("int64")

    for column in MARKET_NUMERIC:
        numeric = pd.to_numeric(work[column], errors="coerce")
        non_numeric = numeric.isna() & work[column].notna()
        reasons.loc[reasons.isna() & non_numeric] = f"non_numeric_{column}"
        work[column] = numeric
    reasons.loc[reasons.isna() & work[list(MARKET_NUMERIC)].isna().any(axis=1)] = "null_ohlcv"
    reasons.loc[
        reasons.isna() & np.isinf(work[list(MARKET_NUMERIC)]).any(axis=1)
    ] = "non_finite_ohlcv"
    reasons.loc[reasons.isna() & (work[["open", "high", "low", "close"]] <= 0).any(axis=1)] = (
        "non_positive_price"
    )
    reasons.loc[reasons.isna() & (work["volume"] < 0)] = "negative_volume"
    bad_envelope = (work["high"] < work[["open", "close", "low"]].max(axis=1)) | (
        work["low"] > work[["open", "close", "high"]].min(axis=1)
    )
    reasons.loc[reasons.isna() & bad_envelope] = "invalid_ohlc_envelope"
    reasons.loc[reasons.isna() & (work[contract.availability_col] < work[contract.event_col])] = (
        "availability_before_event"
    )
    reasons.loc[
        reasons.isna()
        & (
            work[contract.instrument_col].isna()
            | work[contract.instrument_col].astype(str).str.strip().eq("")
        )
    ] = "missing_instrument"
    reasons.loc[
        reasons.isna() & (ingestion < work[contract.availability_col])
    ] = "ingestion_before_availability"

    cleaned = work.loc[reasons.isna()].copy()
    cleaned, reasons, duplicates, ordering = _resolve_market_duplicates(
        cleaned, reasons, contract=contract, value_cols=value_cols
    )
    accepted = cleaned.drop(columns=["_input_order"]).copy()
    accepted["revision"] = accepted["revision"].astype("int64")
    accepted = accepted.sort_values(
        [
            contract.event_col,
            contract.instrument_col,
            "revision",
            contract.availability_col,
        ],
        kind="stable",
    ).reset_index(drop=True)

    all_reasons = reasons.reindex(original.index)
    rejected_mask = all_reasons.notna()
    rejected = original.loc[rejected_mask].copy()
    rejected["rejection_reason"] = all_reasons.loc[rejected_mask].astype(str)
    rejected = rejected.drop(columns=["_input_order"], errors="ignore")

    report = assess_frame(
        accepted,
        dataset=dataset,
        source=source,
        schema_version=schema_version,
        key_cols=(contract.event_col, contract.instrument_col),
        instrument_col=contract.instrument_col,
        timestamp_col=contract.event_col,
        availability_col=contract.availability_col,
    )
    report.input_count = len(original)
    report.accepted_count = len(accepted)
    report.duplicate_count = duplicates
    report.ordering_violations = ordering
    report.timestamp_violations = timestamp_violations
    report.rejection_reasons = {}
    report.rejected_count = 0
    report.rejected_sample = []
    report.missing_values = missing_value_stats(original)
    report.coverage = coverage_stats(
        accepted, instrument_col=contract.instrument_col, timestamp_col=contract.event_col
    )
    report.freshness = freshness_stats(
        accepted,
        reference_time=ingestion,
        availability_col=contract.availability_col,
        timestamp_col=contract.event_col,
    )
    report.assumptions = assumptions
    _record_rejections(report, rejected)
    report.checks = {
        "schema": True,
        "unique_keys": report.duplicate_count == 0,
        "chronological": report.ordering_violations == 0,
        "timestamps": report.timestamp_violations == 0,
        "ohlcv": True,
    }
    return NormalizationResult(accepted=accepted, rejected=rejected, report=report, assumptions=assumptions)


def normalize_event_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    dataset: str,
    ingestion_timestamp: Any | None = None,
    schema_version: str = "1",
    contract: TemporalContract = DEFAULT_CONTRACT,
    instrument_col: str | None = None,
    value_cols: Sequence[str] | None = None,
) -> NormalizationResult:
    """Generic PIT normalization for fundamentals, alternative data, and events."""

    instrument = instrument_col or contract.instrument_col
    required = [contract.event_col, contract.availability_col, instrument]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise SchemaValidationError(f"Missing required event columns: {missing}")

    original = frame.copy()
    original["_input_order"] = np.arange(len(original))
    reasons = pd.Series([None] * len(original), index=original.index, dtype=object)
    work = original.copy()
    timestamp_violations = 0
    for column in (contract.event_col, contract.availability_col):
        parsed, errors = _parse_temporal_column(work[column], name=column)
        work[column] = parsed
        for position, idx in enumerate(work.index):
            if errors[position]:
                timestamp_violations += 1
                reasons.loc[idx] = errors[position]
    publication = contract.publication_col
    if publication and publication in work.columns:
        parsed, errors = _parse_temporal_column(
            work[publication], name=publication, allow_missing=True
        )
        work[publication] = parsed
        for position, idx in enumerate(work.index):
            if errors[position]:
                timestamp_violations += 1
                reasons.loc[idx] = "; ".join(e for e in (reasons.loc[idx], errors[position]) if e)

    reasons.loc[reasons.isna() & (work[contract.availability_col] < work[contract.event_col])] = (
        "availability_before_event"
    )
    if publication and publication in work.columns:
        reasons.loc[reasons.isna() & (work[publication] < work[contract.event_col])] = (
            "publication_before_event"
        )
        reasons.loc[
            reasons.isna() & (work[publication] > work[contract.availability_col])
        ] = "availability_before_publication"

    ingestion = ensure_utc(ingestion_timestamp, name="ingestion_timestamp") if ingestion_timestamp is not None else utc_now()
    work["ingestion_timestamp"] = ingestion
    work["source"] = source
    work["schema_version"] = schema_version
    if "revision" in work:
        numeric_revision, invalid_revision = _coerce_revision(work["revision"])
        reasons.loc[reasons.isna() & invalid_revision] = "invalid_revision"
        work["revision"] = numeric_revision.where(
            numeric_revision.notna() & ~invalid_revision, 0
        ).astype("int64")
    else:
        work["revision"] = 0
    reasons.loc[
        reasons.isna()
        & (work[instrument].isna() | work[instrument].astype(str).str.strip().eq(""))
    ] = "missing_instrument"
    reasons.loc[
        reasons.isna() & (ingestion < work[contract.availability_col])
    ] = "ingestion_before_availability"

    cleaned = work.loc[reasons.isna()].drop(columns=["_input_order"]).copy()
    if value_cols is None:
        excluded = {
            contract.event_col,
            instrument,
            "revision",
            contract.availability_col,
            "source",
            "ingestion_timestamp",
        }
        conflict_cols = [c for c in cleaned.columns if c not in excluded]
    else:
        missing = [c for c in value_cols if c not in cleaned.columns]
        if missing:
            raise SchemaValidationError(f"Missing value columns for conflict checks: {missing}")
        conflict_cols = list(value_cols)

    resolved, duplicates = _deduplicate_preserving_revisions(
        cleaned,
        key_cols=(contract.event_col, instrument),
        revision_col="revision",
        available_col=contract.availability_col,
        payload_cols=conflict_cols,
        conflict_reason="conflicting_revision",
        reasons=reasons,
    )
    accepted = (
        resolved.drop(columns=["_input_order"])
        .sort_values(
            [contract.event_col, instrument, "revision", contract.availability_col],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    all_reasons = reasons.reindex(original.index)
    rejected = original.loc[all_reasons.notna()].copy()
    rejected["rejection_reason"] = all_reasons.loc[all_reasons.notna()].astype(str)
    rejected = rejected.drop(columns=["_input_order"], errors="ignore")

    report = assess_frame(
        accepted,
        dataset=dataset,
        source=source,
        schema_version=schema_version,
        key_cols=(contract.event_col, instrument),
        instrument_col=instrument,
        timestamp_col=contract.event_col,
        availability_col=contract.availability_col,
    )
    report.input_count = len(original)
    report.accepted_count = len(accepted)
    report.duplicate_count = duplicates
    report.timestamp_violations = timestamp_violations
    report.rejection_reasons = {}
    report.rejected_count = 0
    report.rejected_sample = []
    _record_rejections(report, rejected)
    report.checks = {
        "schema": True,
        "unique_keys": report.duplicate_count == 0,
        "timestamps": report.timestamp_violations == 0,
        "point_in_time": True,
    }
    return NormalizationResult(accepted=accepted, rejected=rejected, report=report)


def pit_snapshot(
    frame: pd.DataFrame,
    decision_time: Any,
    *,
    contract: TemporalContract = DEFAULT_CONTRACT,
    instrument: str | None = None,
    resolve_conflicting_revisions: bool = True,
) -> pd.DataFrame:
    """Rows valid and available at ``decision_time`` (inclusive availability)."""

    decision = ensure_utc(decision_time, name="decision_time")
    work = normalize_temporal(frame, contract=contract)
    mask = (
        work[contract.availability_col] <= decision
        if contract.inclusive_availability
        else work[contract.availability_col] < decision
    )
    valid_from, valid_to = contract.valid_from_col, contract.valid_to_col
    if valid_from and valid_from in work.columns:
        mask &= work[valid_from] <= decision
    if valid_to and valid_to in work.columns:
        mask &= work[valid_to].isna() | (work[valid_to] > decision)
    if instrument is not None:
        mask &= work[contract.instrument_col].eq(instrument)
    selected = work.loc[mask].copy()
    if resolve_conflicting_revisions and not selected.empty:
        selected = resolve_revisions(
            selected,
            key_cols=(contract.event_col, contract.instrument_col),
            available_col=contract.availability_col,
        )
    selected[contract.decision_col] = decision
    assert_point_in_time(
        selected,
        decision,
        availability_col=contract.availability_col,
        inclusive=contract.inclusive_availability,
    )
    return selected.sort_values(
        [contract.event_col, contract.instrument_col], kind="stable"
    ).reset_index(drop=True)
