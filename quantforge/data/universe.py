"""Survivorship-safe, point-in-time instrument universe.

Universe snapshots are built from versioned membership, lifecycle, and symbol
mapping records whose ``available_at`` is at/before the decision. Today's
surviving tickers are never used to reconstruct history. Missing historical
coverage is surfaced as an explicit status and can cause a fail-closed error.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping

import pandas as pd

from .errors import IncompleteUniverseError, UniverseError
from .timeutils import coerce_optional_utc, ensure_utc


class InstrumentStatus(str, Enum):
    NOT_LISTED = "not_listed"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELISTED = "delisted"
    BANKRUPT = "bankrupt"
    ACQUIRED = "acquired"
    MERGED = "merged"
    UNKNOWN = "unknown"


class LifecycleEventType(str, Enum):
    LISTED = "listed"
    DELISTED = "delisted"
    BANKRUPT = "bankrupt"
    ACQUIRED = "acquired"
    MERGED = "merged"
    RENAMED = "renamed"
    SUSPENDED = "suspended"
    RESUMED = "resumed"
    OTHER = "other"


_STATUS_BY_EVENT = {
    LifecycleEventType.LISTED: InstrumentStatus.ACTIVE,
    LifecycleEventType.DELISTED: InstrumentStatus.DELISTED,
    LifecycleEventType.BANKRUPT: InstrumentStatus.BANKRUPT,
    LifecycleEventType.ACQUIRED: InstrumentStatus.ACQUIRED,
    LifecycleEventType.MERGED: InstrumentStatus.MERGED,
    LifecycleEventType.RENAMED: InstrumentStatus.ACTIVE,
    LifecycleEventType.SUSPENDED: InstrumentStatus.SUSPENDED,
    LifecycleEventType.RESUMED: InstrumentStatus.ACTIVE,
}


def _as_date(value: Any, *, name: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    try:
        return pd.Timestamp(value).date()
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} is not a valid date: {value!r}") from exc


@dataclass(frozen=True)
class Instrument:
    instrument_id: str
    symbol: str
    exchange: str
    available_at: pd.Timestamp
    currency: str = "USD"
    asset_type: str = "equity"
    listing_date: date | None = None
    delisting_date: date | None = None
    revision: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id is required")
        object.__setattr__(self, "listing_date", _as_date(self.listing_date, name="listing_date"))
        object.__setattr__(self, "delisting_date", _as_date(self.delisting_date, name="delisting_date"))
        object.__setattr__(self, "available_at", ensure_utc(self.available_at, name="available_at"))
        object.__setattr__(self, "revision", int(self.revision))
        if self.listing_date and self.delisting_date and self.listing_date > self.delisting_date:
            raise ValueError("listing_date cannot be after delisting_date")


@dataclass(frozen=True)
class LifecycleEvent:
    event_id: str
    instrument_id: str
    event_type: LifecycleEventType
    effective_timestamp: pd.Timestamp
    available_at: pd.Timestamp
    successor_instrument_id: str | None = None
    reason: str = ""
    revision: int = 0
    source: str = "unspecified"

    def __post_init__(self) -> None:
        if not self.event_id or not self.instrument_id:
            raise ValueError("event_id and instrument_id are required")
        object.__setattr__(self, "event_type", LifecycleEventType(self.event_type))
        object.__setattr__(
            self, "effective_timestamp", ensure_utc(self.effective_timestamp, name="effective_timestamp")
        )
        object.__setattr__(self, "available_at", ensure_utc(self.available_at, name="available_at"))
        object.__setattr__(self, "revision", int(self.revision))


@dataclass(frozen=True)
class SymbolMapping:
    mapping_id: str
    symbol: str
    instrument_id: str
    exchange: str
    effective_from: pd.Timestamp
    available_at: pd.Timestamp
    effective_to: pd.Timestamp | None = None
    revision: int = 0

    def __post_init__(self) -> None:
        if not self.mapping_id or not self.symbol or not self.instrument_id:
            raise ValueError("mapping_id, symbol, and instrument_id are required")
        object.__setattr__(self, "effective_from", ensure_utc(self.effective_from, name="effective_from"))
        object.__setattr__(self, "effective_to", coerce_optional_utc(self.effective_to, name="effective_to"))
        object.__setattr__(self, "available_at", ensure_utc(self.available_at, name="available_at"))
        object.__setattr__(self, "revision", int(self.revision))
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to must be after effective_from")
        if self.effective_to is not None and self.available_at > self.effective_to:
            raise ValueError("mapping became available after it expired")


@dataclass(frozen=True)
class UniverseMembership:
    membership_id: str
    universe: str
    instrument_id: str
    effective_from: pd.Timestamp
    available_at: pd.Timestamp
    effective_to: pd.Timestamp | None = None
    included: bool = True
    reason: str = ""
    revision: int = 0
    source: str = "unspecified"

    def __post_init__(self) -> None:
        if not self.membership_id or not self.universe or not self.instrument_id:
            raise ValueError("membership_id, universe, and instrument_id are required")
        object.__setattr__(self, "effective_from", ensure_utc(self.effective_from, name="effective_from"))
        object.__setattr__(self, "effective_to", coerce_optional_utc(self.effective_to, name="effective_to"))
        object.__setattr__(self, "available_at", ensure_utc(self.available_at, name="available_at"))
        object.__setattr__(self, "revision", int(self.revision))
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to must be after effective_from")

    def is_active_at(self, as_of: pd.Timestamp) -> bool:
        if as_of < self.effective_from:
            return False
        if self.effective_to is not None and as_of >= self.effective_to:
            return False
        return True


@dataclass(frozen=True)
class UniverseCoverage:
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

    def covers(self, as_of: pd.Timestamp) -> bool:
        if not self.complete:
            return False
        if self.start is not None and as_of < self.start:
            return False
        if self.end is not None and as_of > self.end:
            return False
        return True


@dataclass(frozen=True)
class UniverseMember:
    instrument_id: str
    symbol: str | None
    exchange: str | None
    as_of: pd.Timestamp
    status: InstrumentStatus
    economically_existed: bool | None
    listed: bool | None
    tradable: bool | None
    member: bool
    membership_available_at: pd.Timestamp
    data_available: bool | None = None
    reasons: tuple[str, ...] = ()

    @property
    def eligible(self) -> bool:
        return bool(
            self.member
            and self.listed is True
            and self.tradable is True
            and self.data_available is not False
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument_id": self.instrument_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "as_of": self.as_of.isoformat(),
            "status": self.status.value,
            "economically_existed": self.economically_existed,
            "listed": self.listed,
            "tradable": self.tradable,
            "member": self.member,
            "eligible": self.eligible,
            "membership_available_at": self.membership_available_at.isoformat(),
            "data_available": self.data_available,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class UniverseSnapshot:
    universe: str
    as_of: pd.Timestamp
    members: tuple[UniverseMember, ...]
    excluded: tuple[UniverseMember, ...] = ()
    coverage_status: str = "unknown"
    incomplete_reasons: tuple[str, ...] = ()
    late_membership_records: tuple[str, ...] = ()
    unknown_instruments: tuple[str, ...] = ()
    coverage: UniverseCoverage | None = None
    snapshot_id: str = ""

    @property
    def eligible(self) -> tuple[UniverseMember, ...]:
        return tuple(member for member in self.members if member.eligible)

    @property
    def complete(self) -> bool:
        return self.coverage_status == "complete"

    def assert_complete(self) -> None:
        if not self.complete:
            raise IncompleteUniverseError(
                f"Universe {self.universe!r} at {self.as_of} is {self.coverage_status}: "
                f"{', '.join(self.incomplete_reasons) or 'no reason supplied'}"
            )

    def to_frame(self) -> pd.DataFrame:
        rows = [member.to_dict() for member in self.members + self.excluded]
        if not rows:
            return pd.DataFrame(
                columns=[
                    "instrument_id",
                    "symbol",
                    "exchange",
                    "as_of",
                    "status",
                    "economically_existed",
                    "listed",
                    "tradable",
                    "member",
                    "eligible",
                    "membership_available_at",
                    "data_available",
                    "reasons",
                ]
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "universe": self.universe,
            "as_of": self.as_of.isoformat(),
            "members": [m.to_dict() for m in self.members],
            "excluded": [m.to_dict() for m in self.excluded],
            "coverage_status": self.coverage_status,
            "incomplete_reasons": list(self.incomplete_reasons),
            "late_membership_records": list(self.late_membership_records),
            "unknown_instruments": list(self.unknown_instruments),
            "snapshot_id": self.snapshot_id,
        }


class UniverseStore:
    """Versioned repository for instruments, lifecycle events, mappings, membership."""

    def __init__(self, *, coverage: UniverseCoverage | None = None) -> None:
        self._instruments: dict[str, dict[int, Instrument]] = {}
        self._events: dict[str, LifecycleEvent] = {}
        self._mappings: dict[str, dict[int, SymbolMapping]] = {}
        self._memberships: dict[str, dict[int, UniverseMembership]] = {}
        self.coverage = coverage

    # ---------------------------------------------------------------- mutations
    def add_instrument(self, instrument: Instrument) -> None:
        revisions = self._instruments.setdefault(instrument.instrument_id, {})
        self._add_revision(revisions, instrument, action_id=instrument.instrument_id)

    def add_lifecycle_event(self, event: LifecycleEvent) -> None:
        existing = self._events.get(event.event_id)
        if existing is not None:
            if existing == event:
                return
            raise UniverseError(f"Conflicting lifecycle event_id={event.event_id!r}")
        self._events[event.event_id] = event

    def add_symbol_mapping(self, mapping: SymbolMapping) -> None:
        revisions = self._mappings.setdefault(mapping.mapping_id, {})
        self._add_revision(revisions, mapping, action_id=mapping.mapping_id)

    def add_membership(self, membership: UniverseMembership) -> None:
        revisions = self._memberships.setdefault(membership.membership_id, {})
        self._add_revision(revisions, membership, action_id=membership.membership_id)

    @staticmethod
    def _add_revision(revisions: dict[int, Any], value: Any, *, action_id: str) -> None:
        if value.revision in revisions:
            if revisions[value.revision] == value:
                return
            raise UniverseError(f"Conflicting revision {value.revision} for {action_id!r}")
        if revisions and value.revision <= max(revisions):
            raise UniverseError(f"Revision {value.revision} for {action_id!r} is not increasing")
        revisions[value.revision] = value

    def declare_coverage(self, coverage: UniverseCoverage) -> None:
        self.coverage = coverage

    # ------------------------------------------------------------------ queries
    def instrument_at(self, instrument_id: str, as_of: Any) -> Instrument | None:
        decision = ensure_utc(as_of, name="as_of")
        revisions = self._instruments.get(instrument_id)
        if not revisions:
            return None
        eligible = [i for i in revisions.values() if i.available_at <= decision]
        if not eligible:
            return None
        return max(eligible, key=lambda i: i.revision)

    def resolve_symbol_mapping(
        self, symbol: str, as_of: Any, *, exchange: str | None = None
    ) -> SymbolMapping | None:
        decision = ensure_utc(as_of, name="as_of")
        candidates: list[SymbolMapping] = []
        for revisions in self._mappings.values():
            eligible = [
                m
                for m in revisions.values()
                if m.symbol == symbol
                and m.available_at <= decision
                and m.effective_from <= decision
                and (m.effective_to is None or decision < m.effective_to)
                and (exchange is None or m.exchange == exchange)
            ]
            if eligible:
                candidates.append(max(eligible, key=lambda m: m.revision))
        if not candidates:
            return None
        candidates.sort(key=lambda m: (m.effective_from, m.revision, m.mapping_id))
        winner = candidates[-1]
        tied = [
            m
            for m in candidates
            if m.effective_from == winner.effective_from
            and m.revision == winner.revision
            and m.instrument_id != winner.instrument_id
        ]
        if tied:
            raise UniverseError(f"Ambiguous symbol mapping for {symbol!r} at {decision}")
        return winner

    def resolve_symbol(self, symbol: str, as_of: Any, *, exchange: str | None = None) -> str | None:
        mapping = self.resolve_symbol_mapping(symbol, as_of, exchange=exchange)
        return None if mapping is None else mapping.instrument_id

    def symbol_for(self, instrument_id: str, as_of: Any) -> str | None:
        decision = ensure_utc(as_of, name="as_of")
        candidates: list[SymbolMapping] = []
        for revisions in self._mappings.values():
            eligible = [
                m
                for m in revisions.values()
                if m.instrument_id == instrument_id
                and m.available_at <= decision
                and m.effective_from <= decision
                and (m.effective_to is None or decision < m.effective_to)
            ]
            if eligible:
                candidates.append(max(eligible, key=lambda m: m.revision))
        if not candidates:
            return None
        candidates.sort(key=lambda m: (m.effective_from, m.revision, m.mapping_id))
        return candidates[-1].symbol

    def status_at(
        self, instrument_id: str, as_of: Any
    ) -> tuple[InstrumentStatus, LifecycleEvent | None]:
        decision = ensure_utc(as_of, name="as_of")
        instrument = self.instrument_at(instrument_id, decision)
        if instrument is None:
            return InstrumentStatus.UNKNOWN, None
        events = [
            event
            for event in self._events.values()
            if event.instrument_id == instrument_id
            and event.available_at <= decision
            and event.effective_timestamp <= decision
        ]
        if events:
            latest = max(
                events,
                key=lambda e: (e.effective_timestamp, e.revision, e.event_id),
            )
            return _STATUS_BY_EVENT.get(latest.event_type, InstrumentStatus.UNKNOWN), latest
        day = decision.date()
        if instrument.listing_date is not None and day < instrument.listing_date:
            return InstrumentStatus.NOT_LISTED, None
        if instrument.delisting_date is not None and day >= instrument.delisting_date:
            return InstrumentStatus.DELISTED, None
        if instrument.listing_date is not None or instrument.delisting_date is not None:
            return InstrumentStatus.ACTIVE, None
        return InstrumentStatus.UNKNOWN, None

    def _latest_memberships(
        self, universe: str, as_of: pd.Timestamp
    ) -> tuple[list[UniverseMembership], list[UniverseMembership]]:
        selected: list[UniverseMembership] = []
        late: list[UniverseMembership] = []
        for revisions in self._memberships.values():
            for membership in revisions.values():
                if membership.universe != universe:
                    continue
                if membership.is_active_at(as_of):
                    if membership.available_at <= as_of:
                        selected.append(membership)
                    else:
                        late.append(membership)
        latest_by_id: dict[str, UniverseMembership] = {}
        for membership in selected:
            current = latest_by_id.get(membership.membership_id)
            if current is None or (membership.revision, membership.available_at) > (
                current.revision,
                current.available_at,
            ):
                latest_by_id[membership.membership_id] = membership
        by_instrument: dict[str, list[UniverseMembership]] = {}
        for membership in latest_by_id.values():
            by_instrument.setdefault(membership.instrument_id, []).append(membership)
        resolved: list[UniverseMembership] = []
        for instrument_id, rows in by_instrument.items():
            rows.sort(key=lambda m: (m.effective_from, m.revision, m.membership_id))
            winner = rows[-1]
            tied = [
                m
                for m in rows
                if m.effective_from == winner.effective_from
                and m.revision == winner.revision
                and (m.included, m.reason) != (winner.included, winner.reason)
            ]
            if tied:
                raise UniverseError(
                    f"Conflicting overlapping membership for {instrument_id!r} at {as_of}"
                )
            resolved.append(winner)
        resolved.sort(key=lambda m: m.instrument_id)
        late = sorted(late, key=lambda m: (m.effective_from, m.membership_id))
        return resolved, late

    def _data_available(
        self, instrument_id: str, data_coverage: Mapping[str, Any] | None
    ) -> bool | None:
        if data_coverage is None:
            return None
        if instrument_id not in data_coverage:
            return None
        value = data_coverage[instrument_id]
        if isinstance(value, bool):
            return value
        if isinstance(value, Mapping):
            return bool(value.get("available", value.get("complete", False)))
        if isinstance(value, (tuple, list)) and len(value) == 2 and value[0] is not None and value[1] is not None:
            start = pd.Timestamp(value[0])
            end = pd.Timestamp(value[1])
            return start.tzinfo is not None and end.tzinfo is not None
        return bool(value)

    def universe_as_of(
        self,
        universe: str,
        as_of: Any,
        *,
        require_complete: bool = False,
        data_coverage: Mapping[str, Any] | None = None,
        include_excluded: bool = False,
    ) -> UniverseSnapshot:
        decision = ensure_utc(as_of, name="as_of")
        memberships, late = self._latest_memberships(universe, decision)
        incomplete: list[str] = []
        unknown: list[str] = []

        if self.coverage is None:
            coverage_status = "unknown"
            incomplete.append("no coverage declaration")
        elif not self.coverage.covers(decision):
            coverage_status = "out_of_coverage"
            incomplete.append("decision outside declared coverage")
        else:
            coverage_status = "complete"
        if late:
            incomplete.append("membership records became available after the decision")
            if coverage_status == "complete":
                coverage_status = "incomplete_late_membership"

        members: list[UniverseMember] = []
        excluded: list[UniverseMember] = []
        for membership in memberships:
            instrument = self.instrument_at(membership.instrument_id, decision)
            membership_available = membership.available_at
            if instrument is None:
                unknown.append(membership.instrument_id)
                if coverage_status == "complete":
                    coverage_status = "incomplete_unknown_instrument"
                incomplete.append(f"instrument metadata unavailable for {membership.instrument_id}")
                member = UniverseMember(
                    instrument_id=membership.instrument_id,
                    symbol=None,
                    exchange=None,
                    as_of=decision,
                    status=InstrumentStatus.UNKNOWN,
                    economically_existed=None,
                    listed=None,
                    tradable=None,
                    member=membership.included,
                    membership_available_at=membership_available,
                    data_available=self._data_available(membership.instrument_id, data_coverage),
                    reasons=("instrument_metadata_unavailable",),
                )
                (members if membership.included else excluded).append(member)
                continue

            status, _ = self.status_at(membership.instrument_id, decision)
            symbol = self.symbol_for(membership.instrument_id, decision)
            reasons: list[str] = []
            if symbol is None:
                symbol = instrument.symbol
                reasons.append("symbol_mapping_unavailable")
            if status is InstrumentStatus.UNKNOWN:
                reasons.append("lifecycle_status_unknown")
                if coverage_status == "complete":
                    coverage_status = "incomplete_lifecycle_status"
                incomplete.append(
                    f"lifecycle status unknown for {membership.instrument_id}"
                )
            day = decision.date()
            if status is InstrumentStatus.UNKNOWN:
                economically_existed = None
                listed = None
                tradable = None
            else:
                if instrument.listing_date is not None and day < instrument.listing_date:
                    economically_existed = False
                elif instrument.delisting_date is not None and day > instrument.delisting_date:
                    economically_existed = False
                else:
                    economically_existed = True
                if status is InstrumentStatus.NOT_LISTED:
                    economically_existed = False
                listed = status in {InstrumentStatus.ACTIVE, InstrumentStatus.SUSPENDED}
                tradable = status is InstrumentStatus.ACTIVE
            data_value = self._data_available(membership.instrument_id, data_coverage)
            if data_value is False:
                reasons.append("insufficient_data")
            member = UniverseMember(
                instrument_id=instrument.instrument_id,
                symbol=symbol,
                exchange=instrument.exchange,
                as_of=decision,
                status=status,
                economically_existed=economically_existed,
                listed=listed,
                tradable=tradable,
                member=membership.included,
                membership_available_at=membership_available,
                data_available=data_value,
                reasons=tuple(reasons),
            )
            (members if membership.included else excluded).append(member)

        members.sort(key=lambda m: m.instrument_id)
        excluded.sort(key=lambda m: m.instrument_id)
        if unknown and coverage_status == "complete":
            coverage_status = "incomplete_unknown_instrument"
        if require_complete and coverage_status != "complete":
            raise IncompleteUniverseError(
                f"Universe {universe!r} at {decision} is {coverage_status}: "
                f"{', '.join(incomplete) or 'no reason supplied'}"
            )

        payload = {
            "universe": universe,
            "as_of": decision.isoformat(),
            "members": [
                {
                    "instrument_id": m.instrument_id,
                    "symbol": m.symbol,
                    "member": m.member,
                    "status": m.status.value,
                    "data_available": m.data_available,
                }
                for m in members + excluded
            ],
        }
        snapshot_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return UniverseSnapshot(
            universe=universe,
            as_of=decision,
            members=tuple(members),
            excluded=tuple(excluded) if include_excluded else (),
            coverage_status=coverage_status,
            incomplete_reasons=tuple(incomplete),
            late_membership_records=tuple(m.membership_id for m in late),
            unknown_instruments=tuple(sorted(set(unknown))),
            coverage=self.coverage,
            snapshot_id=snapshot_id,
        )
