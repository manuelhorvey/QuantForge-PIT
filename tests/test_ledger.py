"""Module 3 trial-ledger contract tests."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from quantforge.governance import LedgerIntegrityError, TrialLedger
from quantforge.governance.ledger import SCHEMA_VERSION


def _ledger(tmp_path: Path) -> TrialLedger:
    return TrialLedger(tmp_path / "trials.jsonl")


def _append(
    ledger: TrialLedger,
    *,
    trial_id: str = "trial-1",
    status: str = "PREREGISTERED",
    **overrides,
) -> str:
    values: dict[str, Any] = {
        "trial_id": trial_id,
        "hypothesis": "Momentum predicts returns.",
        "parameters": {"alpha": 1.0},
        "data_hash": "dataset-hash",
        "metrics": {"mae": 0.1},
        "status": status,
        "metadata": {"experiment_id": "experiment-1"},
        "created_at": "2024-01-01T00:00:00Z",
    }
    values.update(overrides)
    return ledger.append(**values)


def _canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_records(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(_canonical(record) for record in records) + "\n", encoding="utf-8")


def _event(**overrides) -> dict:
    values = {
        "trial_id": "trial-1",
        "hypothesis": "Momentum predicts returns.",
        "parameters": {"alpha": 1.0},
        "data_hash": "dataset-hash",
        "metrics": {"mae": 0.1},
        "status": "PREREGISTERED",
        "metadata": {"experiment_id": "experiment-1"},
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    values.update(overrides)
    return values


def _chain(events: list[dict]) -> list[dict]:
    records: list[dict] = []
    previous: str | None = None
    for event in events:
        payload = {"schema_version": SCHEMA_VERSION, **event, "previous_sha256": previous}
        entry = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        record = {**payload, "entry_sha256": entry}
        records.append(record)
        previous = entry
    return records


def test_append_read_roundtrip_and_entry_hash(tmp_path):
    ledger = _ledger(tmp_path)
    digest = _append(
        ledger,
        parameters={"alpha": 1, "nested": [1, True, None]},
        metrics={"mae": 0.1, "pearson": None},
        metadata={"experiment_id": "experiment-1", "note": "registered"},
    )

    records = ledger.read()

    assert len(records) == 1
    record = records[0]
    assert record.trial_id == "trial-1"
    assert record.hypothesis == "Momentum predicts returns."
    assert record.parameters == {"alpha": 1, "nested": [1, True, None]}
    assert record.data_hash == "dataset-hash"
    assert record.metrics == {"mae": 0.1, "pearson": None}
    assert record.status == "PREREGISTERED"
    assert record.metadata == {"experiment_id": "experiment-1", "note": "registered"}
    assert record.created_at == "2024-01-01T00:00:00+00:00"
    assert record.previous_sha256 is None
    assert record.entry_sha256 == digest
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert record.to_dict() == {
        "schema_version": SCHEMA_VERSION,
        "trial_id": "trial-1",
        "hypothesis": "Momentum predicts returns.",
        "parameters": {"alpha": 1, "nested": [1, True, None]},
        "data_hash": "dataset-hash",
        "metrics": {"mae": 0.1, "pearson": None},
        "status": "PREREGISTERED",
        "metadata": {"experiment_id": "experiment-1", "note": "registered"},
        "created_at": "2024-01-01T00:00:00+00:00",
        "previous_sha256": None,
        "entry_sha256": digest,
    }
    assert ledger.verify() == 1
    assert ledger.count() == 1


def test_genesis_and_multi_entry_hash_chain(tmp_path):
    ledger = _ledger(tmp_path)
    first = _append(ledger)
    second = _append(ledger, status="COMPLETED", created_at="2024-01-02T00:00:00Z")
    third = _append(ledger, trial_id="trial-2", status="FAILED")

    records = ledger.read()

    assert [record.previous_sha256 for record in records] == [None, first, second]
    assert [record.entry_sha256 for record in records] == [first, second, third]
    assert ledger.verify() == 3
    assert ledger.summarize()["head_sha256"] == third


def test_append_normalizes_datetime_and_defaults_unvalidated(tmp_path):
    ledger = _ledger(tmp_path)
    digest = ledger.append(
        trial_id="trial-1",
        hypothesis="h",
        parameters={},
        data_hash="data",
        metrics={},
        created_at=datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
    )

    record = ledger.read()[0]

    assert record.entry_sha256 == digest
    assert record.status == "UNVALIDATED"
    assert record.metadata == {}
    assert record.created_at == "2024-01-01T12:00:00+00:00"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda records: records[0].update(hypothesis="changed"),
        lambda records: records[1].update(previous_sha256="0" * 64),
        lambda records: records[0].update(entry_sha256="0" * 64),
        lambda records: records.reverse(),
        lambda records: records.pop(1),
    ],
)
def test_tampering_is_detected(tmp_path, mutation):
    ledger = _ledger(tmp_path)
    _append(ledger)
    _append(ledger, status="COMPLETED")
    _append(ledger, trial_id="trial-2", status="FAILED")
    records = _read_records(ledger.path)
    mutation(records)
    _write_records(ledger.path, records)

    with pytest.raises(LedgerIntegrityError):
        TrialLedger(ledger.path).verify()


def test_malformed_and_noncanonical_files_fail(tmp_path):
    ledger = _ledger(tmp_path)
    _append(ledger)
    record = _read_records(ledger.path)[0]
    path = ledger.path

    path.write_text("{\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        TrialLedger(path).verify()

    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        TrialLedger(path).verify()

    path.write_text("\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        TrialLedger(path).verify()

    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        TrialLedger(path).verify()

    path.write_text(_canonical(record), encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        TrialLedger(path).verify()

    missing = dict(record)
    missing.pop("metrics")
    path.write_text(_canonical(missing) + "\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        TrialLedger(path).verify()

    bad_schema = dict(record)
    bad_schema["schema_version"] = "quantforge.trial_ledger.v1"
    path.write_text(_canonical(bad_schema) + "\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        TrialLedger(path).verify()

    extra = dict(record)
    extra["extra"] = True
    path.write_text(_canonical(extra) + "\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        TrialLedger(path).verify()

    line = _canonical(record).replace('"trial_id":"trial-1"', '"trial_id":"trial-1","trial_id":"trial-1"')
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        TrialLedger(path).verify()

    path.write_text('{"metrics":{"mae":NaN}}\n', encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        TrialLedger(path).verify()

    path.write_text('{"metrics":{"mae":1e999}}\n', encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        TrialLedger(path).verify()

    path.write_text(_canonical(record) + " \n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        TrialLedger(path).verify()


@pytest.mark.parametrize(
    "overrides",
    [
        {"trial_id": ""},
        {"status": "UNKNOWN"},
        {"parameters": []},
        {"metrics": None},
        {"metadata": []},
        {"metadata": {"experiment_id": 1}},
        {"parameters": {1: "not-a-string-key"}},
        {"metrics": {"mae": float("nan")}},
        {"metrics": {"mae": float("inf")}},
        {"created_at": "not-a-date"},
        {"created_at": "2024-01-01T00:00:00"},
    ],
)
def test_append_rejects_invalid_input_without_writing(tmp_path, overrides):
    ledger = _ledger(tmp_path)

    with pytest.raises(LedgerIntegrityError):
        _append(ledger, **overrides)

    assert not ledger.path.exists()


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("PREREGISTERED", "PREREGISTERED"),
        ("UNVALIDATED", "UNVALIDATED"),
        ("UNVALIDATED", "PREREGISTERED"),
    ],
)
def test_duplicate_registration_events_fail(tmp_path, first, second):
    ledger = _ledger(tmp_path)
    _append(ledger, status=first)
    before = ledger.path.read_text(encoding="utf-8")

    with pytest.raises(LedgerIntegrityError, match="duplicate registration"):
        _append(ledger, status=second)

    assert ledger.path.read_text(encoding="utf-8") == before
    assert ledger.verify() == 1


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("COMPLETED", "COMPLETED"),
        ("COMPLETED", "FAILED"),
        ("FAILED", "COMPLETED"),
    ],
)
def test_duplicate_terminal_events_fail(tmp_path, first, second):
    ledger = _ledger(tmp_path)
    _append(ledger, status=first)
    before = ledger.path.read_text(encoding="utf-8")

    with pytest.raises(LedgerIntegrityError, match="duplicate terminal"):
        _append(ledger, status=second)

    assert ledger.path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("terminal", ["COMPLETED", "FAILED", "ABANDONED", "EXCLUDED"])
def test_event_after_terminal_fails(tmp_path, terminal):
    ledger = _ledger(tmp_path)
    _append(ledger, status=terminal)
    before = ledger.path.read_text(encoding="utf-8")

    with pytest.raises(LedgerIntegrityError, match="event after terminal"):
        _append(ledger, status="PREREGISTERED")

    assert ledger.path.read_text(encoding="utf-8") == before


def test_registration_then_terminal_then_event_fails(tmp_path):
    ledger = _ledger(tmp_path)
    _append(ledger)
    _append(ledger, status="COMPLETED")
    before = ledger.path.read_text(encoding="utf-8")

    with pytest.raises(LedgerIntegrityError, match="more than two events"):
        _append(ledger, status="FAILED")

    assert ledger.path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    "events",
    [
        [{"status": "PREREGISTERED"}, {"status": "PREREGISTERED"}],
        [{"status": "COMPLETED"}, {"status": "COMPLETED"}],
        [{"status": "COMPLETED"}, {"status": "PREREGISTERED"}],
        [
            {"status": "PREREGISTERED"},
            {"status": "COMPLETED"},
            {"status": "FAILED"},
        ],
        [
            {"status": "PREREGISTERED"},
            {"status": "COMPLETED", "hypothesis": "Different hypothesis."},
        ],
    ],
)
def test_verification_rejects_invalid_lifecycle_written_externally(tmp_path, events):
    ledger = _ledger(tmp_path)
    _write_records(ledger.path, _chain([_event(**event) for event in events]))

    with pytest.raises(LedgerIntegrityError):
        TrialLedger(ledger.path).verify()


@pytest.mark.parametrize(
    "changes",
    [
        {"hypothesis": "Different hypothesis."},
        {"parameters": {"alpha": 2.0}},
        {"parameters": {"alpha": 1}},
        {"data_hash": "other-dataset"},
        {"metadata": {"experiment_id": "experiment-2"}},
        {"metadata": {}},
    ],
)
def test_lifecycle_identity_changes_fail(tmp_path, changes):
    ledger = _ledger(tmp_path)
    _append(ledger)
    before = ledger.path.read_text(encoding="utf-8")

    with pytest.raises(LedgerIntegrityError, match="identity changed"):
        _append(ledger, status="COMPLETED", **changes)

    assert ledger.path.read_text(encoding="utf-8") == before


def test_valid_registration_then_terminal_is_accepted(tmp_path):
    ledger = _ledger(tmp_path)
    _append(ledger)
    _append(ledger, status="COMPLETED")

    records = ledger.read()
    summary = ledger.summarize()

    assert [record.status for record in records] == ["PREREGISTERED", "COMPLETED"]
    assert summary["entries"] == 2
    assert summary["trials"] == 1
    assert summary["by_status"] == {"COMPLETED": 1}
    assert summary["unregistered_trials"] == 0


def test_terminal_only_is_valid_and_reported_as_unregistered(tmp_path):
    ledger = _ledger(tmp_path)
    _append(ledger, status="COMPLETED", metadata={})

    summary = ledger.summarize()

    assert summary["entries"] == 1
    assert summary["trials"] == 1
    assert summary["by_status"] == {"COMPLETED": 1}
    assert summary["by_experiment"] == {"": 1}
    assert summary["unregistered_trials"] == 1

    with pytest.raises(LedgerIntegrityError, match="duplicate terminal"):
        _append(ledger, status="FAILED", metadata={})


def test_summary_counts_entries_trials_statuses_experiments_and_head(tmp_path):
    ledger = _ledger(tmp_path)
    _append(ledger, metadata={"experiment_id": "experiment-a"})
    _append(ledger, status="COMPLETED", metadata={"experiment_id": "experiment-a"})
    _append(
        ledger,
        trial_id="trial-2",
        status="UNVALIDATED",
        metadata={"experiment_id": "experiment-b"},
    )
    third = _append(ledger, trial_id="trial-3", status="FAILED", metadata={})
    fourth = _append(
        ledger,
        trial_id="trial-4",
        status="ABANDONED",
        metadata={"experiment_id": "experiment-a"},
    )

    assert ledger.count() == 5
    assert ledger.summarize() == {
        "schema_version": SCHEMA_VERSION,
        "entries": 5,
        "trials": 4,
        "by_status": {
            "ABANDONED": 1,
            "COMPLETED": 1,
            "FAILED": 1,
            "UNVALIDATED": 1,
        },
        "by_experiment": {"": 1, "experiment-a": 2, "experiment-b": 1},
        "unregistered_trials": 2,
        "head_sha256": fourth,
    }
    assert third != fourth


def test_empty_ledger_verifies_and_summarizes(tmp_path):
    ledger = _ledger(tmp_path)

    assert ledger.read() == ()
    assert ledger.verify() == 0
    assert ledger.count() == 0
    assert ledger.summarize() == {
        "schema_version": SCHEMA_VERSION,
        "entries": 0,
        "trials": 0,
        "by_status": {},
        "by_experiment": {},
        "unregistered_trials": 0,
        "head_sha256": None,
    }


def test_append_never_extends_corrupt_ledger(tmp_path):
    ledger = _ledger(tmp_path)
    _append(ledger)
    ledger.path.write_text(ledger.path.read_text(encoding="utf-8") + "not-json\n", encoding="utf-8")
    before = ledger.path.read_text(encoding="utf-8")

    with pytest.raises(LedgerIntegrityError):
        _append(ledger, trial_id="trial-2")

    assert ledger.path.read_text(encoding="utf-8") == before


def test_append_flushes_and_fsyncs(tmp_path, monkeypatch):
    ledger = _ledger(tmp_path)
    calls: list[int] = []
    monkeypatch.setattr("quantforge.governance.ledger.os.fsync", lambda fd: calls.append(fd))

    _append(ledger)

    assert len(calls) == 1
