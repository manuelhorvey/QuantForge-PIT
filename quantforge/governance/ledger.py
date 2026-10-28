"""Hash-chained append-only trial ledger."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "quantforge.trial_ledger.v2"
_REGISTRATION_STATUSES = frozenset({"PREREGISTERED", "UNVALIDATED"})
_TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "ABANDONED", "EXCLUDED"})
_STATUSES = _REGISTRATION_STATUSES | _TERMINAL_STATUSES
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "trial_id",
        "hypothesis",
        "parameters",
        "data_hash",
        "metrics",
        "status",
        "metadata",
        "created_at",
        "previous_sha256",
        "entry_sha256",
    }
)


class LedgerIntegrityError(ValueError):
    """Raised when a ledger record or chain violates the v2 contract."""


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    hypothesis: str
    parameters: Mapping[str, Any]
    data_hash: str
    metrics: Mapping[str, Any]
    status: str
    metadata: Mapping[str, Any]
    created_at: str
    previous_sha256: str | None
    entry_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "trial_id": self.trial_id,
            "hypothesis": self.hypothesis,
            "parameters": copy.deepcopy(dict(self.parameters)),
            "data_hash": self.data_hash,
            "metrics": copy.deepcopy(dict(self.metrics)),
            "status": self.status,
            "metadata": copy.deepcopy(dict(self.metadata)),
            "created_at": self.created_at,
            "previous_sha256": self.previous_sha256,
            "entry_sha256": self.entry_sha256,
        }


class TrialLedger:
    """Append-only JSONL ledger with a verified SHA-256 hash chain."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        trial_id: str,
        hypothesis: str,
        parameters: Mapping[str, Any],
        data_hash: str,
        metrics: Mapping[str, Any],
        status: str = "UNVALIDATED",
        metadata: Mapping[str, Any] | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        trial_id = _non_empty_text(trial_id, "trial_id")
        hypothesis = _non_empty_text(hypothesis, "hypothesis")
        data_hash = _non_empty_text(data_hash, "data_hash")
        status = _status(status)
        parameters = _json_mapping(parameters, "parameters")
        metrics = _json_mapping(metrics, "metrics")
        metadata = _json_mapping({} if metadata is None else metadata, "metadata")
        _experiment_id(metadata)
        created_at_text = _created_at(created_at)

        existing = self._load()
        previous_sha256 = existing[-1].entry_sha256 if existing else None
        candidate = TrialRecord(
            trial_id=trial_id,
            hypothesis=hypothesis,
            parameters=parameters,
            data_hash=data_hash,
            metrics=metrics,
            status=status,
            metadata=metadata,
            created_at=created_at_text,
            previous_sha256=previous_sha256,
            entry_sha256="",
        )
        _validate_lifecycle((*existing, candidate))

        payload = candidate.to_dict()
        payload.pop("entry_sha256")
        entry_sha256 = _record_hash(payload)
        record = {**payload, "entry_sha256": entry_sha256}
        line = _canonical_json(record)

        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return entry_sha256

    def read(self) -> tuple[TrialRecord, ...]:
        return self._load()

    def verify(self) -> int:
        return len(self._load())

    def count(self) -> int:
        return self.verify()

    def summarize(self) -> dict[str, Any]:
        records = self._load()
        trials: dict[str, list[TrialRecord]] = {}
        for record in records:
            trials.setdefault(record.trial_id, []).append(record)

        by_status: dict[str, int] = {}
        by_experiment: dict[str, int] = {}
        unregistered_trials = 0
        for events in trials.values():
            latest = events[-1]
            by_status[latest.status] = by_status.get(latest.status, 0) + 1
            experiment_id = _experiment_id(latest.metadata)
            by_experiment[experiment_id] = by_experiment.get(experiment_id, 0) + 1
            if len(events) == 1 and latest.status in _TERMINAL_STATUSES:
                unregistered_trials += 1

        return {
            "schema_version": SCHEMA_VERSION,
            "entries": len(records),
            "trials": len(trials),
            "by_status": dict(sorted(by_status.items())),
            "by_experiment": dict(sorted(by_experiment.items())),
            "unregistered_trials": unregistered_trials,
            "head_sha256": records[-1].entry_sha256 if records else None,
        }

    def _load(self) -> tuple[TrialRecord, ...]:
        if not self.path.exists():
            return ()
        try:
            text = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise LedgerIntegrityError(f"cannot read ledger: {exc}") from exc
        if not text:
            return ()
        if not text.endswith("\n"):
            raise LedgerIntegrityError("ledger does not end with a newline")

        records: list[TrialRecord] = []
        previous_sha256: str | None = None
        for line_number, line in enumerate(text[:-1].split("\n"), start=1):
            if not line:
                raise LedgerIntegrityError(f"blank ledger line at line {line_number}")
            record = _parse_record(line, line_number)
            if record.previous_sha256 != previous_sha256:
                raise LedgerIntegrityError(
                    f"previous_sha256 mismatch at line {line_number}"
                )
            previous_sha256 = record.entry_sha256
            records.append(record)
        _validate_lifecycle(records)
        return tuple(records)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _record_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _parse_record(line: str, line_number: int) -> TrialRecord:
    try:
        raw = json.loads(
            line,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except LedgerIntegrityError as exc:
        raise LedgerIntegrityError(f"invalid ledger line {line_number}: {exc}") from exc
    except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as exc:
        raise LedgerIntegrityError(f"invalid JSON at line {line_number}: {exc}") from exc
    if not isinstance(raw, dict):
        raise LedgerIntegrityError(f"ledger line {line_number} must be an object")
    if set(raw) != _RECORD_FIELDS:
        missing = sorted(_RECORD_FIELDS - set(raw))
        extra = sorted(set(raw) - _RECORD_FIELDS)
        raise LedgerIntegrityError(
            f"ledger line {line_number} has invalid fields; missing={missing}, extra={extra}"
        )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise LedgerIntegrityError(f"unsupported schema at line {line_number}")
    try:
        canonical = _canonical_json(raw)
    except (TypeError, ValueError) as exc:
        raise LedgerIntegrityError(
            f"ledger line {line_number} contains non-canonical values: {exc}"
        ) from exc
    if canonical != line:
        raise LedgerIntegrityError(f"ledger line {line_number} is not canonical JSON")

    trial_id = _non_empty_text(raw["trial_id"], "trial_id", line_number)
    hypothesis = _non_empty_text(raw["hypothesis"], "hypothesis", line_number)
    data_hash = _non_empty_text(raw["data_hash"], "data_hash", line_number)
    status = _status(raw["status"], line_number)
    parameters = _json_mapping(raw["parameters"], "parameters", line_number)
    metrics = _json_mapping(raw["metrics"], "metrics", line_number)
    metadata = _json_mapping(raw["metadata"], "metadata", line_number)
    _experiment_id(metadata, line_number)
    created_at = _created_at(raw["created_at"], line_number)
    if created_at != raw["created_at"]:
        raise LedgerIntegrityError(f"created_at is not canonical at line {line_number}")
    previous_sha256 = _sha256(raw["previous_sha256"], "previous_sha256", line_number, allow_none=True)
    entry_sha256 = _sha256(raw["entry_sha256"], "entry_sha256", line_number)
    assert entry_sha256 is not None
    payload = {key: value for key, value in raw.items() if key != "entry_sha256"}
    expected = _record_hash(payload)
    if not hmac.compare_digest(expected, entry_sha256):
        raise LedgerIntegrityError(f"entry_sha256 mismatch at line {line_number}")

    return TrialRecord(
        trial_id=trial_id,
        hypothesis=hypothesis,
        parameters=parameters,
        data_hash=data_hash,
        metrics=metrics,
        status=status,
        metadata=metadata,
        created_at=created_at,
        previous_sha256=previous_sha256,
        entry_sha256=entry_sha256,
    )


def _validate_lifecycle(records: Sequence[TrialRecord]) -> None:
    grouped: dict[str, list[TrialRecord]] = {}
    for record in records:
        grouped.setdefault(record.trial_id, []).append(record)

    for trial_id, events in grouped.items():
        identity = _identity(events[0])
        for event in events[1:]:
            if _identity(event) != identity:
                raise LedgerIntegrityError(f"trial identity changed for {trial_id!r}")
        if len(events) > 2:
            raise LedgerIntegrityError(f"trial {trial_id!r} has more than two events")
        if len(events) == 1:
            continue

        first, second = events
        if first.status in _TERMINAL_STATUSES:
            if second.status in _TERMINAL_STATUSES:
                raise LedgerIntegrityError(f"duplicate terminal event for {trial_id!r}")
            raise LedgerIntegrityError(f"event after terminal event for {trial_id!r}")
        if second.status in _REGISTRATION_STATUSES:
            raise LedgerIntegrityError(f"duplicate registration event for {trial_id!r}")


def _identity(record: TrialRecord) -> tuple[str, str, str, bool, str | None]:
    experiment_id = _experiment_id(record.metadata)
    return (
        record.hypothesis,
        _canonical_json(record.parameters),
        record.data_hash,
        "experiment_id" in record.metadata,
        experiment_id if "experiment_id" in record.metadata else None,
    )


def _non_empty_text(value: Any, field: str, line_number: int | None = None) -> str:
    if not isinstance(value, str) or not value:
        suffix = f" at line {line_number}" if line_number is not None else ""
        raise LedgerIntegrityError(f"{field} must be a non-empty string{suffix}")
    return value


def _status(value: Any, line_number: int | None = None) -> str:
    value = _non_empty_text(value, "status", line_number)
    if value not in _STATUSES:
        suffix = f" at line {line_number}" if line_number is not None else ""
        raise LedgerIntegrityError(f"unknown status {value!r}{suffix}")
    return value


def _created_at(value: Any, line_number: int | None = None) -> str:
    suffix = f" at line {line_number}" if line_number is not None else ""
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise LedgerIntegrityError(f"created_at must be ISO 8601{suffix}") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise LedgerIntegrityError(f"created_at must be a datetime or string{suffix}")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LedgerIntegrityError(f"created_at must be timezone-aware{suffix}")
    return parsed.astimezone(timezone.utc).isoformat()


def _sha256(
    value: Any,
    field: str,
    line_number: int | None = None,
    *,
    allow_none: bool = False,
) -> str | None:
    if value is None and allow_none:
        return None
    suffix = f" at line {line_number}" if line_number is not None else ""
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise LedgerIntegrityError(f"{field} must be 64 lowercase hex characters{suffix}")
    return value


def _experiment_id(
    metadata: Mapping[str, Any], line_number: int | None = None
) -> str:
    if "experiment_id" not in metadata:
        return ""
    value = metadata["experiment_id"]
    if not isinstance(value, str) or not value:
        suffix = f" at line {line_number}" if line_number is not None else ""
        raise LedgerIntegrityError(
            f"metadata.experiment_id must be a non-empty string{suffix}"
        )
    return value


def _json_mapping(
    value: Any, field: str, line_number: int | None = None
) -> dict[str, Any]:
    suffix = f" at line {line_number}" if line_number is not None else ""
    if not isinstance(value, Mapping):
        raise LedgerIntegrityError(f"{field} must be a mapping{suffix}")
    normalized = _json_value(value, field, line_number)
    if not isinstance(normalized, dict):
        raise LedgerIntegrityError(f"{field} must be a mapping{suffix}")
    return normalized


def _json_value(value: Any, field: str, line_number: int | None = None) -> Any:
    suffix = f" at line {line_number}" if line_number is not None else ""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LedgerIntegrityError(f"{field} contains a non-finite number{suffix}")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LedgerIntegrityError(f"{field} keys must be strings{suffix}")
            result[key] = _json_value(item, field, line_number)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, field, line_number) for item in value]
    raise LedgerIntegrityError(f"{field} contains a non-JSON value{suffix}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LedgerIntegrityError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise LedgerIntegrityError(f"non-finite JSON number {value!r}")


__all__ = ["LedgerIntegrityError", "SCHEMA_VERSION", "TrialLedger", "TrialRecord"]
