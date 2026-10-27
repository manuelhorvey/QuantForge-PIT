"""Trading-calendar interface and session alignment.

``SimpleExchangeCalendar`` is a deterministic fixture calendar: callers must
declare weekdays, holidays, early closes, and the exchange timezone. It is not a
substitute for a complete exchange calendar. If ``exchange_calendars`` is
installed, ``ExchangeCalendarsAdapter`` delegates to it; otherwise the adapter
fails explicitly with installation guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Iterable, Mapping, Protocol, cast
from zoneinfo import ZoneInfo

import pandas as pd

from .errors import CalendarError, CalendarUnavailableError, SchemaValidationError
from .timeutils import ensure_utc_series

UTC = ZoneInfo("UTC")


@dataclass(frozen=True)
class Session:
    calendar_name: str
    session_date: date
    open: pd.Timestamp
    close: pd.Timestamp
    is_early_close: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "open", pd.Timestamp(self.open).tz_convert("UTC"))
        object.__setattr__(self, "close", pd.Timestamp(self.close).tz_convert("UTC"))
        if self.close <= self.open:
            raise CalendarError("session close must be after open")

    def contains(self, timestamp: Any) -> bool:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            raise CalendarError("timestamp must be timezone-aware")
        ts = ts.tz_convert("UTC")
        return self.open <= ts <= self.close


class TradingCalendar(Protocol):
    name: str
    timezone: ZoneInfo

    def is_session(self, day: date) -> bool: ...

    def session(self, day: date) -> Session: ...

    def sessions(self, start: date, end: date) -> list[Session]: ...

    def next_session(self, day: date) -> date: ...

    def previous_session(self, day: date) -> date: ...


@dataclass
class SimpleExchangeCalendar:
    """Explicit, deterministic calendar driven by declared fixtures."""

    name: str
    timezone: str | ZoneInfo = "America/New_York"
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    open_time: time = time(9, 30)
    close_time: time = time(16, 0)
    holidays: frozenset[date] = field(default_factory=frozenset)
    early_closes: Mapping[date, time] = field(default_factory=dict)
    closed_dates: frozenset[date] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        self.timezone = ZoneInfo(self.timezone) if isinstance(self.timezone, str) else self.timezone
        if not self.name:
            raise ValueError("calendar name is required")
        if not self.weekdays or any(d < 0 or d > 6 for d in self.weekdays):
            raise ValueError("weekdays must be integers in [0, 6]")
        if self.open_time >= self.close_time:
            raise ValueError("open_time must be before close_time")
        self.holidays = frozenset(self.holidays)
        self.closed_dates = frozenset(self.closed_dates)
        self.early_closes = dict(self.early_closes)

    def is_session(self, day: date) -> bool:
        return (
            day.weekday() in self.weekdays
            and day not in self.holidays
            and day not in self.closed_dates
        )

    def _local_bounds(self, day: date) -> tuple[datetime, datetime]:
        close_time = self.early_closes.get(day, self.close_time)
        timezone = cast(ZoneInfo, self.timezone)
        return (
            datetime.combine(day, self.open_time).replace(tzinfo=timezone),
            datetime.combine(day, close_time).replace(tzinfo=timezone),
        )

    def session(self, day: date) -> Session:
        if not self.is_session(day):
            raise CalendarError(f"{day} is not a session on {self.name}")
        local_open, local_close = self._local_bounds(day)
        open_ts = pd.Timestamp(local_open)
        close_ts = pd.Timestamp(local_close)
        for label, stamp in (("open", open_ts), ("close", close_ts)):
            self._reject_ambiguous_or_nonexistent(day, label, stamp)
        return Session(
            calendar_name=self.name,
            session_date=day,
            open=open_ts.tz_convert("UTC"),
            close=close_ts.tz_convert("UTC"),
            is_early_close=day in self.early_closes,
        )

    def _reject_ambiguous_or_nonexistent(self, day: date, label: str, stamp: pd.Timestamp) -> None:
        naive = stamp.replace(tzinfo=None)
        tz = stamp.tzinfo
        off0 = naive.replace(tzinfo=tz, fold=0).utcoffset()
        off1 = naive.replace(tzinfo=tz, fold=1).utcoffset()
        if off0 != off1:
            raise CalendarError(
                f"Ambiguous {label} time for {day} in {self.timezone}; declare an explicit session"
            )
        back = stamp.tz_convert(self.timezone).replace(tzinfo=None)
        if back != naive:
            raise CalendarError(
                f"Nonexistent {label} time for {day} in {self.timezone}; declare an explicit session"
            )

    def sessions(self, start: date, end: date) -> list[Session]:
        if start > end:
            raise ValueError("start must be <= end")
        result: list[Session] = []
        current = start
        while current <= end:
            if self.is_session(current):
                result.append(self.session(current))
            current += timedelta(days=1)
        return result

    def next_session(self, day: date) -> date:
        current = day + timedelta(days=1)
        for _ in range(3700):
            if self.is_session(current):
                return current
            current += timedelta(days=1)
        raise CalendarError(f"No session found after {day} within ten years on {self.name}")

    def previous_session(self, day: date) -> date:
        current = day - timedelta(days=1)
        for _ in range(3700):
            if self.is_session(current):
                return current
            current -= timedelta(days=1)
        raise CalendarError(f"No session found before {day} within ten years on {self.name}")


class ExchangeCalendarsAdapter:
    """Optional adapter for the mature ``exchange_calendars`` package."""

    def __init__(self, name: str) -> None:
        try:
            import exchange_calendars as xcals  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise CalendarUnavailableError(
                "exchange_calendars is not installed; install it to use this adapter "
                "or provide explicit calendar fixtures"
            ) from exc
        self.name = name
        try:
            self._calendar = xcals.get_calendar(name)
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise CalendarError(f"Unknown exchange_calendars calendar {name!r}: {exc}") from exc
        self.timezone = ZoneInfo(str(self._calendar.tz))

    def is_session(self, day: date) -> bool:  # pragma: no cover - optional dependency
        return bool(self._calendar.is_session(pd.Timestamp(day)))

    def session(self, day: date) -> Session:  # pragma: no cover - optional dependency
        stamp = pd.Timestamp(day)
        if not self._calendar.is_session(stamp):
            raise CalendarError(f"{day} is not a session on {self.name}")
        opens = self._calendar.session_open(stamp).tz_convert("UTC")
        closes = self._calendar.session_close(stamp).tz_convert("UTC")
        early = self._calendar.early_closes
        is_early = bool(stamp in getattr(early, "index", []))
        return Session(
            calendar_name=self.name,
            session_date=day,
            open=opens,
            close=closes,
            is_early_close=is_early,
        )

    def sessions(self, start: date, end: date) -> list[Session]:  # pragma: no cover
        return [self.session(day.date()) for day in self._calendar.sessions_in_range(
            pd.Timestamp(start), pd.Timestamp(end)
        )]

    def next_session(self, day: date) -> date:  # pragma: no cover
        return self._calendar.next_session(pd.Timestamp(day)).date()

    def previous_session(self, day: date) -> date:  # pragma: no cover
        return self._calendar.previous_session(pd.Timestamp(day)).date()


@dataclass(frozen=True)
class InstrumentSchedule:
    instrument: str
    calendar: TradingCalendar
    valid_from: date | None = None
    valid_to: date | None = None
    extra_holidays: frozenset[date] = field(default_factory=frozenset)

    def applies_on(self, day: date) -> bool:
        if self.valid_from is not None and day < self.valid_from:
            return False
        if self.valid_to is not None and day > self.valid_to:
            return False
        return True

    def is_session(self, day: date) -> bool:
        return self.applies_on(day) and day not in self.extra_holidays and self.calendar.is_session(day)

    def session(self, day: date) -> Session:
        if day in self.extra_holidays:
            raise CalendarError(f"{day} is an instrument-specific holiday for {self.instrument}")
        return self.calendar.session(day)

    def expected_sessions(self, start: date, end: date) -> list[date]:
        return [s.session_date for s in self.calendar.sessions(start, end) if self.is_session(s.session_date)]


def align_bars_to_sessions(
    bars: pd.DataFrame,
    schedules: Mapping[str, InstrumentSchedule],
    *,
    instrument_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Classify bars against explicit instrument schedules.

    Adds ``session_date``, ``session_status`` and ``is_valid_session_bar``.
    ``session_status`` distinguishes unknown schedules, out-of-range schedules,
    market closures, and timestamps outside declared session bounds.
    """

    if instrument_col not in bars or timestamp_col not in bars:
        raise SchemaValidationError("instrument and timestamp columns are required")
    out = bars.copy()
    out[timestamp_col] = ensure_utc_series(out[timestamp_col], name=timestamp_col)
    session_dates: list[date | pd.NaT] = []
    statuses: list[str] = []
    for _, row in out.iterrows():
        symbol = str(row[instrument_col])
        schedule = schedules.get(symbol)
        stamp: pd.Timestamp = row[timestamp_col]
        if schedule is None:
            session_dates.append(pd.NaT)
            statuses.append("unknown_schedule")
            continue
        day = stamp.tz_convert(schedule.calendar.timezone).date()
        if not schedule.applies_on(day):
            session_dates.append(pd.NaT)
            statuses.append("outside_schedule")
            continue
        if day in schedule.extra_holidays or not schedule.calendar.is_session(day):
            session_dates.append(pd.NaT)
            statuses.append("market_closed")
            continue
        session = schedule.session(day)
        if not session.contains(stamp):
            session_dates.append(day)
            statuses.append("outside_session")
            continue
        session_dates.append(day)
        statuses.append("ok")
    out["session_date"] = session_dates
    out["session_status"] = statuses
    out["is_valid_session_bar"] = out["session_status"].eq("ok")
    return out


def missing_session_bars(
    bars: pd.DataFrame,
    schedules: Mapping[str, InstrumentSchedule],
    start: date,
    end: date,
    *,
    instrument_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Expected sessions with no valid bar.

    Legitimate market closures are absent from ``expected`` and therefore never
    reported as missing. Bars that fall outside a session do not count as present.
    """

    aligned = align_bars_to_sessions(
        bars, schedules, instrument_col=instrument_col, timestamp_col=timestamp_col
    )
    rows: list[dict[str, Any]] = []
    for instrument, schedule in sorted(schedules.items()):
        expected = set(schedule.expected_sessions(start, end))
        actual = set(
            aligned.loc[
                aligned[instrument_col].astype(str).eq(instrument)
                & aligned["is_valid_session_bar"],
                "session_date",
            ]
        )
        for missing in sorted(expected - actual):
            rows.append({"instrument": instrument, "session_date": missing})
    if not rows:
        return pd.DataFrame(columns=["instrument", "session_date"])
    return pd.DataFrame(rows).sort_values(["instrument", "session_date"]).reset_index(drop=True)


class CalendarRegistry:
    """Small explicit registry to keep provider calendars out of the pipeline core."""

    def __init__(self, calendars: Iterable[TradingCalendar] | None = None) -> None:
        self._calendars: dict[str, TradingCalendar] = {}
        for calendar in calendars or ():
            self.register(calendar)

    def register(self, calendar: TradingCalendar) -> None:
        if calendar.name in self._calendars:
            raise CalendarError(f"Calendar {calendar.name!r} already registered")
        self._calendars[calendar.name] = calendar

    def get(self, name: str) -> TradingCalendar:
        try:
            return self._calendars[name]
        except KeyError as exc:
            raise CalendarError(f"Unknown calendar {name!r}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._calendars))
