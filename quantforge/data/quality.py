"""Data-quality reporting.

Quality failures are data, not exceptions to swallow: callers receive counts,
rejection reasons, and a bounded sample of rejected rows. Storage/connector
orchestration decides whether rejected rows are fatal, but they are never
silently dropped from the report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from .timeutils import ensure_utc


@dataclass(frozen=True)
class RejectedRecord:
    """One rejected input row with a machine-readable reason."""

    row_index: Any
    reason: str
    record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"row_index": self.row_index, "reason": self.reason, "record": self.record}


@dataclass
class QualityReport:
    """Actionable ingestion / PIT quality report."""

    dataset: str
    source: str
    schema_version: str
    input_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    duplicate_count: int = 0
    ordering_violations: int = 0
    timestamp_violations: int = 0
    pit_violations: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    rejected_sample: list[RejectedRecord] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    freshness: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)

    def add_rejection(self, reason: str, *, sample_limit: int = 25) -> None:
        self.rejected_count += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
        if len(self.rejected_sample) >= sample_limit:
            return

    def record_rejection(self, rejected: RejectedRecord, *, sample_limit: int = 25) -> None:
        self.add_rejection(rejected.reason, sample_limit=sample_limit)
        if len(self.rejected_sample) < sample_limit:
            self.rejected_sample.append(rejected)

    @property
    def passed(self) -> bool:
        return (
            self.rejected_count == 0
            and self.timestamp_violations == 0
            and self.pit_violations == 0
            and all(self.checks.values())
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rejected_sample"] = [r.to_dict() for r in self.rejected_sample]
        data["passed"] = self.passed
        return data

    def summary(self) -> str:
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(self.rejection_reasons.items()))
        return (
            f"{self.dataset} from {self.source}: input={self.input_count} "
            f"accepted={self.accepted_count} rejected={self.rejected_count}"
            + (f" [{reasons}]" if reasons else "")
        )


def missing_value_stats(frame: pd.DataFrame) -> dict[str, int]:
    """Count missing values per column (0 counts omitted)."""

    counts = frame.isna().sum()
    return {str(col): int(v) for col, v in counts.items() if int(v) > 0}


def coverage_stats(
    frame: pd.DataFrame,
    *,
    instrument_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> dict[str, Any]:
    """Per-instrument observation counts and timestamp bounds."""

    if frame.empty or instrument_col not in frame or timestamp_col not in frame:
        return {"instruments": {}, "instrument_count": 0}
    out: dict[str, Any] = {}
    for instrument, group in frame.groupby(instrument_col, sort=True, dropna=False):
        ts = pd.to_datetime(group[timestamp_col], utc=True, errors="coerce")
        out[str(instrument)] = {
            "count": int(len(group)),
            "start": None if ts.dropna().empty else ts.min().isoformat(),
            "end": None if ts.dropna().empty else ts.max().isoformat(),
        }
    return {"instruments": out, "instrument_count": len(out)}


def freshness_stats(
    frame: pd.DataFrame,
    *,
    reference_time: Any | None = None,
    availability_col: str = "available_at",
    timestamp_col: str = "timestamp",
) -> dict[str, Any]:
    """Freshness relative to an explicit reference time (never wall-clock by default)."""

    if frame.empty or availability_col not in frame:
        return {}
    available = pd.to_datetime(frame[availability_col], utc=True, errors="coerce")
    ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce") if timestamp_col in frame else None
    result: dict[str, Any] = {
        "latest_available_at": None if available.dropna().empty else available.max().isoformat(),
        "earliest_available_at": None if available.dropna().empty else available.min().isoformat(),
    }
    if ts is not None and not ts.dropna().empty:
        result["latest_event_timestamp"] = ts.max().isoformat()
    if reference_time is not None:
        ref = ensure_utc(reference_time, name="reference_time")
        lag = (ref - available.max()).total_seconds() if not available.dropna().empty else None
        result["as_of"] = ref.isoformat()
        result["max_lag_seconds"] = lag
    return result


def assess_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
    source: str,
    schema_version: str,
    key_cols: tuple[str, ...] = ("timestamp", "symbol"),
    instrument_col: str = "symbol",
    timestamp_col: str = "timestamp",
    availability_col: str = "available_at",
    decision_col: str | None = None,
    reference_time: Any | None = None,
) -> QualityReport:
    """Assess an already-canonicalized frame without mutating it.

    Ordering violations count adjacent drops of ``(timestamp, instrument)`` in the
    supplied order. Duplicate counts are exact duplicates beyond the first on
    ``key_cols``. Missing values are reported, not imputed.
    """

    report = QualityReport(
        dataset=dataset,
        source=source,
        schema_version=schema_version,
        input_count=len(frame),
        accepted_count=len(frame),
    )
    report.missing_values = missing_value_stats(frame)
    report.coverage = coverage_stats(frame, instrument_col=instrument_col, timestamp_col=timestamp_col)
    report.freshness = freshness_stats(
        frame,
        reference_time=reference_time,
        availability_col=availability_col,
        timestamp_col=timestamp_col,
    )

    present_keys = [c for c in key_cols if c in frame.columns]
    if present_keys:
        report.duplicate_count = int(frame.duplicated(subset=present_keys).sum())

    if timestamp_col in frame.columns and instrument_col in frame.columns and len(frame) > 1:
        ordered = frame[[timestamp_col, instrument_col]].copy()
        ordered[timestamp_col] = pd.to_datetime(ordered[timestamp_col], utc=True, errors="coerce")
        decreasing = ordered[timestamp_col].diff().lt(pd.Timedelta(0))
        report.ordering_violations = int(decreasing.sum())

    if (
        decision_col is not None
        and decision_col in frame.columns
        and availability_col in frame.columns
    ):
        available = pd.to_datetime(frame[availability_col], utc=True, errors="coerce")
        decision = pd.to_datetime(frame[decision_col], utc=True, errors="coerce")
        report.pit_violations = int((available > decision).sum())

    report.checks = {
        "no_missing_timestamps": bool(
            timestamp_col in frame.columns
            and pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce").notna().all()
        ),
        "no_missing_availability": bool(
            availability_col in frame.columns
            and pd.to_datetime(frame[availability_col], utc=True, errors="coerce").notna().all()
        ),
        "unique_keys": report.duplicate_count == 0,
        "chronological": report.ordering_violations == 0,
        "point_in_time": report.pit_violations == 0,
    }
    return report
