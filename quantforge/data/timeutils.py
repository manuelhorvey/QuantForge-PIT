"""Canonical timezone handling.

Canonical timezone is **UTC**. Every temporal value entering the data layer must
be timezone-aware. Naive timestamps are rejected because their instant depends on
an unstated locale/DST assumption. Zone-aware local timestamps that are ambiguous
during a DST fold are also rejected: callers must attach an explicit UTC offset or
convert to UTC before ingestion.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .errors import TemporalError

CANONICAL_TIMEZONE = "UTC"
UTC = timezone.utc


def is_ambiguous_local(ts: pd.Timestamp) -> bool:
    """Return True when a zone-aware timestamp falls in a DST fold."""

    if ts is pd.NaT or ts.tzinfo is None:
        return False
    try:
        dt: datetime = ts.to_pydatetime()
    except (ValueError, OverflowError):  # pragma: no cover - pandas edge cases
        return True
    naive = dt.replace(tzinfo=None)
    try:
        fold0 = naive.replace(tzinfo=dt.tzinfo, fold=0).utcoffset()
        fold1 = naive.replace(tzinfo=dt.tzinfo, fold=1).utcoffset()
    except (ValueError, OverflowError):
        return True
    return fold0 != fold1


def ensure_utc(value: Any, *, name: str = "timestamp") -> pd.Timestamp:
    """Convert one value to a timezone-aware UTC ``Timestamp``.

    Raises:
        TemporalError: for ``None``/``NaT``, naive values, or ambiguous local
            timestamps.
    """

    if value is None:
        raise TemporalError(f"{name} is required but was None")
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:  # pytz can raise AmbiguousTimeError here
        raise TemporalError(f"{name} is not a valid timestamp: {value!r}") from exc
    if ts is pd.NaT:
        raise TemporalError(f"{name} is NaT")
    if ts.tzinfo is None:
        raise TemporalError(
            f"{name} is timezone-naive ({value!r}); attach an explicit UTC offset"
        )
    if is_ambiguous_local(ts):
        raise TemporalError(
            f"{name} is ambiguous across a DST fold ({value!r}); "
            "attach an explicit UTC offset"
        )
    return ts.tz_convert(CANONICAL_TIMEZONE)


def ensure_utc_series(
    values: Any, *, name: str = "timestamp", allow_missing: bool = False
) -> pd.Series:
    """Convert a series-like to dtype ``datetime64[ns, UTC]`` elementwise.

    When ``allow_missing`` is true, null/``NaT`` entries remain ``NaT`` (useful
    for optional publication or open-ended validity columns).
    """

    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if series.empty:
        return pd.Series(pd.array([], dtype="datetime64[ns, UTC]"), index=series.index)
    if allow_missing:
        parsed = series.map(
            lambda v: pd.NaT if v is None or pd.isna(v) else ensure_utc(v, name=name)
        )
    else:
        parsed = series.map(lambda v: ensure_utc(v, name=name))
    return pd.Series(pd.array(parsed.to_numpy(), dtype="datetime64[ns, UTC]"), index=series.index)


def ensure_utc_columns(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """Copy ``frame`` and convert every present column in ``columns`` to UTC."""

    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = ensure_utc_series(out[column], name=column)
    return out


def utc_now() -> pd.Timestamp:
    """Current time as a UTC ``Timestamp`` (injectable in tests)."""

    return pd.Timestamp.now(tz=CANONICAL_TIMEZONE)


def to_iso_utc(value: Any, *, name: str = "timestamp") -> str:
    """Canonical ISO-8601 UTC string used for storage and checksums."""

    return ensure_utc(value, name=name).isoformat()


def coerce_optional_utc(value: Any, *, name: str = "timestamp") -> pd.Timestamp | None:
    """Convert or return ``None`` for a missing optional timestamp."""

    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if not isinstance(value, (list, tuple, set, dict)):
        try:
            if bool(pd.isna(value)):
                return None
        except (TypeError, ValueError):  # pragma: no cover - array-like edge case
            pass
    return ensure_utc(value, name=name)
