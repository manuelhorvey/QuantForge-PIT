"""Versioned, point-in-time corporate-action adjustments.

Conventions shipped:
* ``split``: split/reverse-split price adjustment only. Price factor for a bar
  before an effective split is ``1 / ratio``; volume factor is ``ratio`` where
  ``ratio = new_shares / old_shares``.
* ``dividend``: cash-dividend adjustment only. For a bar before the effective
  date, factor is ``1 - cash_dividend / reference_close`` where the reference
  close is the raw close immediately before the effective date. Missing
  reference data raises; it is never imputed.
* ``total_return``: split and dividend adjustments combined.

Raw prices remain in the returned frame; adjusted values are added as
``adjusted_*`` columns. Actions effective exactly at a bar timestamp are *not*
applied to that bar (the bar is already post-action). PIT adjustment only uses
actions whose availability and effective timestamps are at/before the decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from .errors import (
    CorporateActionError,
    DoubleAdjustmentError,
    IncompleteCorporateActionDataError,
)
from .timeutils import ensure_utc, ensure_utc_series, coerce_optional_utc


class ActionType(str, Enum):
    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    CASH_DIVIDEND = "cash_dividend"
    SYMBOL_CHANGE = "symbol_change"
    RENAME = "rename"
    DELISTING = "delisting"
    MERGER = "merger"
    ACQUISITION = "acquisition"
    OTHER = "other"


_PRICE_ACTION_TYPES = {ActionType.SPLIT, ActionType.REVERSE_SPLIT, ActionType.CASH_DIVIDEND}


@dataclass(frozen=True)
class ActionCoverage:
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    complete: bool = False
    source: str = "unspecified"
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", coerce_optional_utc(self.start, name="coverage.start"))
        object.__setattr__(self, "end", coerce_optional_utc(self.end, name="coverage.end"))
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("coverage.start must be <= coverage.end")

    def covers(self, start: pd.Timestamp | None, end: pd.Timestamp | None) -> bool:
        if not self.complete:
            return False
        if self.start is not None and start is not None and start < self.start:
            return False
        if self.end is not None and end is not None and end > self.end:
            return False
        return True


@dataclass(frozen=True)
class CorporateAction:
    action_id: str
    instrument_id: str
    action_type: ActionType
    effective_timestamp: pd.Timestamp
    available_at: pd.Timestamp
    ratio: float | None = None
    cash_amount: float | None = None
    currency: str | None = None
    new_instrument_id: str | None = None
    new_symbol: str | None = None
    revision: int = 0
    source: str = "unspecified"
    announced_at: pd.Timestamp | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id is required")
        if not self.instrument_id:
            raise ValueError("instrument_id is required")
        object.__setattr__(self, "action_type", ActionType(self.action_type))
        object.__setattr__(
            self, "effective_timestamp", ensure_utc(self.effective_timestamp, name="effective_timestamp")
        )
        object.__setattr__(self, "available_at", ensure_utc(self.available_at, name="available_at"))
        object.__setattr__(
            self, "announced_at", coerce_optional_utc(self.announced_at, name="announced_at")
        )
        if self.announced_at is not None and self.announced_at > self.available_at:
            raise CorporateActionError("announced_at cannot be after available_at")
        if int(self.revision) < 0:
            raise ValueError("revision must be >= 0")
        object.__setattr__(self, "revision", int(self.revision))
        if self.action_type in {ActionType.SPLIT, ActionType.REVERSE_SPLIT}:
            if self.ratio is None or float(self.ratio) <= 0:
                raise CorporateActionError("split actions require a positive ratio")
            object.__setattr__(self, "ratio", float(self.ratio))
            if self.action_type is ActionType.SPLIT and float(self.ratio) <= 1:
                raise CorporateActionError("split ratio must be > 1 (new shares per old share)")
            if self.action_type is ActionType.REVERSE_SPLIT and float(self.ratio) >= 1:
                raise CorporateActionError("reverse_split ratio must be < 1")
        if self.action_type is ActionType.CASH_DIVIDEND:
            if self.cash_amount is None or float(self.cash_amount) <= 0:
                raise CorporateActionError("cash dividends require a positive cash_amount")
            object.__setattr__(self, "cash_amount", float(self.cash_amount))
            if not self.currency:
                raise CorporateActionError("cash dividends require a currency")
        if self.action_type in {ActionType.SYMBOL_CHANGE, ActionType.RENAME} and not self.new_symbol:
            raise CorporateActionError(f"{self.action_type.value} requires new_symbol")

    @property
    def is_price_action(self) -> bool:
        return self.action_type in _PRICE_ACTION_TYPES


@dataclass(frozen=True)
class AdjustmentResult:
    frame: pd.DataFrame
    convention: str
    as_of: pd.Timestamp | None
    actions_applied: tuple[str, ...] = ()
    adjusted_columns: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "convention": self.convention,
            "as_of": None if self.as_of is None else self.as_of.isoformat(),
            "actions_applied": list(self.actions_applied),
            "adjusted_columns": dict(self.adjusted_columns),
            "rows": len(self.frame),
        }


class CorporateActionLedger:
    """Append-only, revision-aware corporate-action ledger."""

    def __init__(
        self,
        actions: Sequence[CorporateAction] | None = None,
        *,
        coverage: ActionCoverage | None = None,
        coverage_by_instrument: Mapping[str, ActionCoverage] | None = None,
    ) -> None:
        self._actions: dict[str, dict[int, CorporateAction]] = {}
        self.coverage = coverage
        self.coverage_by_instrument = dict(coverage_by_instrument or {})
        for action in actions or ():
            self.add(action)

    def add(self, action: CorporateAction) -> None:
        if not isinstance(action, CorporateAction):
            raise TypeError("action must be a CorporateAction")
        revisions = self._actions.setdefault(action.action_id, {})
        if action.revision in revisions:
            if revisions[action.revision] == action:
                return
            raise CorporateActionError(
                f"Conflicting revision {action.revision} for action_id={action.action_id!r}"
            )
        if revisions and action.revision <= max(revisions):
            raise CorporateActionError(
                f"revision {action.revision} for action_id={action.action_id!r} is not monotonically increasing"
            )
        revisions[action.revision] = action

    def add_many(self, actions: Sequence[CorporateAction]) -> None:
        for action in actions:
            self.add(action)

    def __len__(self) -> int:
        return len(self._actions)

    def all_latest(self) -> list[CorporateAction]:
        latest = [max(revisions.values(), key=lambda a: a.revision) for revisions in self._actions.values()]
        return sorted(latest, key=lambda a: (a.effective_timestamp, a.instrument_id, a.action_id))

    def as_of(self, decision_time: Any, *, instrument: str | None = None) -> list[CorporateAction]:
        decision = ensure_utc(decision_time, name="decision_time")
        selected: list[CorporateAction] = []
        for revisions in self._actions.values():
            eligible = [a for a in revisions.values() if a.available_at <= decision]
            if not eligible:
                continue
            latest = max(eligible, key=lambda a: a.revision)
            if instrument is not None and latest.instrument_id != instrument:
                continue
            selected.append(latest)
        return sorted(selected, key=lambda a: (a.effective_timestamp, a.instrument_id, a.action_id))

    def get(self, action_id: str) -> CorporateAction | None:
        revisions = self._actions.get(action_id)
        if not revisions:
            return None
        return max(revisions.values(), key=lambda a: a.revision)

    def check_coverage(
        self,
        instrument: str,
        *,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
        required: bool = True,
    ) -> None:
        coverage = self.coverage_by_instrument.get(instrument, self.coverage)
        if coverage is None:
            if required:
                raise IncompleteCorporateActionDataError(
                    f"No declared corporate-action coverage for {instrument!r}"
                )
            return
        if required and not coverage.covers(start, end):
            raise IncompleteCorporateActionDataError(
                f"Corporate-action coverage for {instrument!r} is incomplete for "
                f"[{start}, {end}] (declared complete={coverage.complete})"
            )


def _detect_double_adjustment(frame: pd.DataFrame) -> None:
    markers = {"adj_factor", "adjustment_convention", "adjusted_close", "adjusted_open"}
    present = markers.intersection(frame.columns)
    if present:
        raise DoubleAdjustmentError(
            f"Frame already carries adjustment columns {sorted(present)}; adjust raw data once"
        )


def _reference_close(group: pd.DataFrame, event_col: str, close_col: str, effective: pd.Timestamp) -> float:
    prior = group[group[event_col] < effective].sort_values(event_col, kind="stable")
    if prior.empty:
        raise CorporateActionError(
            f"No raw close available before dividend effective timestamp {effective}"
        )
    value = prior[close_col].iloc[-1]
    if pd.isna(value) or float(value) <= 0:
        raise CorporateActionError(f"Invalid dividend reference close {value!r} at {effective}")
    return float(value)


def _split_actions_by_event(
    actions: Sequence[CorporateAction], *, instrument: str
) -> dict[pd.Timestamp, list[CorporateAction]]:
    grouped: dict[pd.Timestamp, list[CorporateAction]] = {}
    for action in actions:
        if action.instrument_id != instrument:
            continue
        if action.action_type in {ActionType.SPLIT, ActionType.REVERSE_SPLIT}:
            grouped.setdefault(action.effective_timestamp, []).append(action)
    for effective, group in grouped.items():
        ratios: set[float] = set()
        for action in group:
            if action.ratio is None:
                raise CorporateActionError(
                    f"Split action {action.action_id} has no ratio"
                )
            ratios.add(round(float(action.ratio), 12))
        if len(group) > 1 or len(ratios) > 1:
            raise CorporateActionError(
                f"Ambiguous split actions for {instrument!r} at {effective}: "
                "multiple action records would double-adjust"
            )
    return grouped


def adjust_prices(
    raw_bars: pd.DataFrame,
    ledger: CorporateActionLedger,
    *,
    convention: str = "split",
    decision_time: Any | None = None,
    instrument_col: str = "symbol",
    event_col: str = "timestamp",
    price_cols: Sequence[str] = ("open", "high", "low", "close"),
    close_col: str = "close",
    volume_col: str = "volume",
    instrument_resolver: Callable[[str], str] | None = None,
    require_complete: bool = True,
) -> AdjustmentResult:
    """Adjust raw bars once, optionally restricted to PIT knowledge.

    ``instrument_resolver`` maps a bar's symbol to a stable instrument id. By
    default the symbol itself is used, which is only valid if the caller has
    already normalized symbols.
    """

    if convention not in {"split", "dividend", "total_return"}:
        raise ValueError("convention must be split, dividend, or total_return")
    if convention != "split" and close_col not in raw_bars.columns:
        raise CorporateActionError("dividend conventions require a close column")
    _detect_double_adjustment(raw_bars)
    missing = [c for c in (event_col, instrument_col, close_col) if c not in raw_bars.columns]
    if missing:
        raise CorporateActionError(f"Missing columns for adjustment: {missing}")

    work = raw_bars.copy()
    work[event_col] = ensure_utc_series(work[event_col], name=event_col)
    if "available_at" in work.columns:
        work["available_at"] = ensure_utc_series(work["available_at"], name="available_at")
    resolver = instrument_resolver or (lambda symbol: symbol)

    decision = ensure_utc(decision_time, name="decision_time") if decision_time is not None else None
    actions = ledger.as_of(decision) if decision is not None else ledger.all_latest()
    if decision is not None:
        actions = [a for a in actions if a.effective_timestamp <= decision]

    price_actions = [
        a
        for a in actions
        if a.action_type in _PRICE_ACTION_TYPES
        and (
            convention == "total_return"
            or (convention == "split" and a.action_type in {ActionType.SPLIT, ActionType.REVERSE_SPLIT})
            or (convention == "dividend" and a.action_type is ActionType.CASH_DIVIDEND)
        )
    ]

    adjusted = work.copy()
    factors = pd.Series(1.0, index=adjusted.index, dtype=float)
    volume_factors = pd.Series(1.0, index=adjusted.index, dtype=float)
    applied: set[str] = set()

    for symbol, group in adjusted.groupby(instrument_col, sort=True, dropna=False):
        instrument = resolver(str(symbol))
        start = group[event_col].min() if not group.empty else None
        end = group[event_col].max() if not group.empty else None
        ledger.check_coverage(str(instrument), start=start, end=end, required=require_complete)
        instrument_actions = [a for a in price_actions if a.instrument_id == str(instrument)]
        _split_actions_by_event(instrument_actions, instrument=str(instrument))
        split_actions = [
            a for a in instrument_actions if a.action_type in {ActionType.SPLIT, ActionType.REVERSE_SPLIT}
        ]
        dividend_actions = [
            a for a in instrument_actions if a.action_type is ActionType.CASH_DIVIDEND
        ]
        for index, row in group.iterrows():
            event = row[event_col]
            factor = 1.0
            volume_factor = 1.0
            for action in split_actions:
                if action.effective_timestamp > event:
                    if action.ratio is None:
                        raise CorporateActionError(
                            f"Split action {action.action_id} has no ratio"
                        )
                    ratio = float(action.ratio)
                    factor *= 1.0 / ratio
                    volume_factor *= ratio
                    applied.add(action.action_id)
            for action in dividend_actions:
                if action.effective_timestamp > event:
                    if action.cash_amount is None:
                        raise CorporateActionError(
                            f"Dividend action {action.action_id} has no cash_amount"
                        )
                    reference = _reference_close(group, event_col, close_col, action.effective_timestamp)
                    dividend_factor = 1.0 - float(action.cash_amount) / reference
                    if dividend_factor <= 0:
                        raise CorporateActionError(
                            f"Dividend {action.action_id} implies a non-positive price factor"
                        )
                    factor *= dividend_factor
                    applied.add(action.action_id)
            factors.loc[index] = factor
            volume_factors.loc[index] = volume_factor

    adjusted["adj_factor"] = factors
    adjusted["volume_factor"] = volume_factors
    adjusted["adjustment_convention"] = convention
    mapping: dict[str, str] = {}
    for column in price_cols:
        if column in adjusted.columns:
            adjusted[f"adjusted_{column}"] = adjusted[column] * adjusted["adj_factor"]
            mapping[column] = f"adjusted_{column}"
    if volume_col in adjusted.columns:
        adjusted[f"adjusted_{volume_col}"] = adjusted[volume_col] * adjusted["volume_factor"]
        mapping[volume_col] = f"adjusted_{volume_col}"

    return AdjustmentResult(
        frame=adjusted.sort_values([event_col, instrument_col], kind="stable").reset_index(drop=True),
        convention=convention,
        as_of=decision,
        actions_applied=tuple(sorted(applied)),
        adjusted_columns=mapping,
    )


def pit_adjusted_prices(
    raw_bars: pd.DataFrame,
    ledger: CorporateActionLedger,
    decision_time: Any,
    **kwargs: Any,
) -> AdjustmentResult:
    """Adjust only data/actions that were available at ``decision_time``."""

    decision = ensure_utc(decision_time, name="decision_time")
    work = raw_bars
    contract_event = kwargs.get("event_col", "timestamp")
    if "available_at" in work.columns:
        work = work.loc[ensure_utc_series(work["available_at"], name="available_at") <= decision]
    elif contract_event in work.columns:
        work = work.loc[ensure_utc_series(work[contract_event], name=contract_event) <= decision]
    return adjust_prices(work, ledger, decision_time=decision, **kwargs)
