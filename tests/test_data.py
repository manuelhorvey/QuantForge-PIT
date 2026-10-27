"""Deterministic Module 1 tests: PIT, corporate actions, calendars, universe, connectors."""

from __future__ import annotations

import sqlite3
from datetime import date, time
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantforge.data.calendars import (
    CalendarUnavailableError,
    ExchangeCalendarsAdapter,
    InstrumentSchedule,
    SimpleExchangeCalendar,
    align_bars_to_sessions,
    missing_session_bars,
)
from quantforge.data.connectors import (
    ConnectorConfig,
    DataRequest,
    HttpJSONConnector,
    HttpRequest,
    HttpResponse,
    IncrementalPolicy,
    IngestionStatus,
    LocalFileConnector,
    ingest,
    urllib_transport,
)
from quantforge.data.corporate_actions import (
    ActionCoverage,
    ActionType,
    CorporateAction,
    CorporateActionLedger,
    IncompleteCorporateActionDataError,
    adjust_prices,
    pit_adjusted_prices,
)
from quantforge.data.errors import (
    ConflictingRevisionsError,
    ConnectorError,
    DoubleAdjustmentError,
    NetworkDisabledError,
    PointInTimeError,
    RetryExhaustedError,
    SchemaValidationError,
    StorageConflictError,
    StorageError,
    TemporalError,
)
from quantforge.data.pipeline import (
    flag_stale,
    normalize_event_frame,
    normalize_market_frame,
    pit_asof_join,
    pit_rolling,
    pit_snapshot,
    resample_ohlcv,
)
from quantforge.data.storage import (
    ParquetStore,
    RawArchive,
    SQLiteStore,
    frame_fingerprint,
)
from quantforge.data.timeutils import ensure_utc
from quantforge.data.universe import (
    IncompleteUniverseError,
    Instrument,
    InstrumentStatus,
    LifecycleEvent,
    LifecycleEventType,
    SymbolMapping,
    UniverseCoverage,
    UniverseMembership,
    UniverseStore,
)


# --------------------------------------------------------------------------- helpers
def bars(
    timestamps: list[str],
    *,
    symbol: str = "A",
    closes: list[float] | None = None,
    available: list[str] | None = None,
    **overrides: list[Any],
) -> pd.DataFrame:
    closes = closes or [100.0 + i for i in range(len(timestamps))]
    available = available or [
        (pd.Timestamp(ts) + pd.Timedelta(seconds=1)).isoformat() for ts in timestamps
    ]
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": [symbol] * len(timestamps),
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(timestamps),
            "available_at": available,
        }
    )
    for column, values in overrides.items():
        frame[column] = values
    return frame


def normalize(frame: pd.DataFrame, *, source: str = "test", **kwargs):
    return normalize_market_frame(
        frame,
        source=source,
        ingestion_timestamp="2024-02-01T00:00:00Z",
        **kwargs,
    )


def action_ledger(*actions: CorporateAction, coverage: ActionCoverage | None = None) -> CorporateActionLedger:
    return CorporateActionLedger(
        list(actions),
        coverage=coverage
        or ActionCoverage(
            start="2000-01-01T00:00:00Z", end="2035-01-01T00:00:00Z", complete=True
        ),
    )


# ------------------------------------------------------------------------- PIT / time
def test_naive_timestamp_is_rejected_with_diagnostic():
    frame = bars(["2024-01-01T00:00:00Z"])
    frame["timestamp"] = pd.to_datetime(["2024-01-01 00:00:00"])  # naive
    result = normalize(frame)
    assert result.accepted.empty
    assert result.report.rejected_count == 1
    assert "timezone-naive" in result.rejected.iloc[0]["rejection_reason"]
    with pytest.raises(TemporalError):
        ensure_utc("2024-01-01 00:00:00")


def test_ambiguous_dst_timestamp_is_rejected():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    ambiguous = datetime(2024, 11, 3, 1, 30, tzinfo=ZoneInfo("America/New_York"))
    with pytest.raises(TemporalError):
        ensure_utc(ambiguous, name="timestamp")


def test_exact_availability_boundary_is_inclusive():
    records = pd.DataFrame(
        {"symbol": ["A"], "available_at": ["2024-01-01T00:01:00Z"], "value": [7.0]}
    )
    decisions = pd.DataFrame(
        {"symbol": ["A"], "decision_timestamp": ["2024-01-01T00:01:00Z"]}
    )
    out = pit_asof_join(decisions, records)
    assert out.loc[0, "matched"] is np.bool_(True)
    assert out.loc[0, "value"] == 7.0


def test_asof_join_excludes_future_publication():
    records = pd.DataFrame(
        {"symbol": ["A"], "available_at": ["2024-01-01T00:02:00Z"], "value": [7.0]}
    )
    decisions = pd.DataFrame(
        {"symbol": ["A"], "decision_timestamp": ["2024-01-01T00:01:00Z"]}
    )
    out = pit_asof_join(decisions, records)
    assert pd.isna(out.loc[0, "value"])
    assert not bool(out.loc[0, "matched"])


def test_asof_join_never_crosses_instruments():
    records = pd.DataFrame(
        {"symbol": ["B"], "available_at": ["2024-01-01T00:00:00Z"], "value": [7.0]}
    )
    decisions = pd.DataFrame(
        {"symbol": ["A"], "decision_timestamp": ["2024-01-01T01:00:00Z"]}
    )
    out = pit_asof_join(decisions, records)
    assert pd.isna(out.loc[0, "value"])


def test_revised_fundamentals_do_not_overwrite_earlier_snapshot():
    records = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "available_at": ["2024-01-02T00:00:00Z", "2024-01-10T00:00:00Z"],
            "revision": [0, 1],
            "value": [1.0, 2.0],
        }
    )
    decisions = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "decision_timestamp": ["2024-01-05T00:00:00Z", "2024-01-11T00:00:00Z"],
        }
    )
    out = pit_asof_join(decisions, records)
    assert out.loc[0, "value"] == 1.0
    assert out.loc[1, "value"] == 2.0


def test_conflicting_same_revision_records_fail():
    records = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "available_at": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "revision": [0, 0],
            "value": [1.0, 2.0],
        }
    )
    decisions = pd.DataFrame(
        {"symbol": ["A"], "decision_timestamp": ["2024-01-02T00:00:00Z"]}
    )
    with pytest.raises(ConflictingRevisionsError):
        pit_asof_join(decisions, records)


def test_revisions_survive_normalize_ingest_store_and_snapshot(tmp_path):
    raw = bars(
        ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
        closes=[100.0, 105.0],
        available=["2024-01-02T00:00:00Z", "2024-01-10T00:00:00Z"],
        revision=[0, 1],
    )
    connector, store, _ = local_fixture(tmp_path, raw)
    result = ingest(
        connector,
        DataRequest(instruments=("A",)),
        store=store,
        ingestion_timestamp="2024-02-01T00:00:00Z",
    )
    assert result.status is IngestionStatus.OK
    assert sorted(result.accepted["revision"].tolist()) == [0, 1]

    stored = store.read(latest=False)
    assert sorted(stored["revision"].tolist()) == [0, 1]
    early = pit_snapshot(stored, "2024-01-05T00:00:00Z")
    late = pit_snapshot(stored, "2024-01-11T00:00:00Z")
    assert early["revision"].tolist() == [0]
    assert early["close"].tolist() == [100.0]
    assert late["revision"].tolist() == [1]
    assert late["close"].tolist() == [105.0]

    conflicting = pd.concat([raw.iloc[[0]], raw.iloc[[0]].assign(close=101.0)], ignore_index=True)
    bad_connector, bad_store, _ = local_fixture(tmp_path / "conflict", conflicting)
    bad = ingest(
        bad_connector,
        DataRequest(instruments=("A",)),
        store=bad_store,
        ingestion_timestamp="2024-02-01T00:00:00Z",
    )
    assert bad.status is IngestionStatus.PARTIAL
    assert bad.report.rejected_count == 2
    assert "conflicting_duplicate" in bad.report.rejection_reasons
    assert bad_store.read(latest=False).empty


def test_rolling_feature_ignores_future_event_and_availability():
    frame = pd.DataFrame(
        {
            "symbol": ["A"] * 4,
            "timestamp": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
                "2024-01-04T00:00:00Z",
            ],
            "available_at": [
                "2024-01-01T00:00:01Z",
                "2024-01-02T00:00:01Z",
                "2024-01-07T00:00:00Z",  # known after every test decision
                "2024-01-05T00:00:00Z",
            ],
            "decision_timestamp": [
                "2024-01-02T00:00:02Z",
                "2024-01-03T00:00:02Z",
                "2024-01-04T00:00:02Z",
                "2024-01-06T00:00:00Z",
            ],
            "value": [1.0, 3.0, 100.0, 5.0],
        }
    )
    out = pit_rolling(frame, "value", 2, decision_col="decision_timestamp")
    assert out.iloc[0] == 2.0  # (1 + 3) / 2
    assert out.iloc[1] == 2.0
    assert out.iloc[2] == 2.0  # 2024-01-03 is not yet available at its decision
    assert out.iloc[3] == 4.0  # (3 + 5) / 2, late 100 excluded

    changed = frame.copy()
    changed.loc[3, "value"] = 9999.0
    second = pit_rolling(changed, "value", 2, decision_col="decision_timestamp")
    assert out.iloc[:3].tolist() == second.iloc[:3].tolist()


def test_publication_after_availability_fails_closed_in_pit_primitives():
    records = pd.DataFrame(
        {
            "symbol": ["A"],
            "publication_timestamp": ["2024-01-10T00:00:00Z"],
            "available_at": ["2024-01-01T00:00:00Z"],
            "value": [7.0],
        }
    )
    decisions = pd.DataFrame(
        {"symbol": ["A"], "decision_timestamp": ["2024-01-05T00:00:00Z"]}
    )
    with pytest.raises(PointInTimeError):
        pit_asof_join(decisions, records)

    rolling = records.assign(
        timestamp=["2024-01-01T00:00:00Z"],
        decision_timestamp=["2024-01-05T00:00:00Z"],
    )
    with pytest.raises(PointInTimeError):
        pit_rolling(rolling, "value", 1, decision_col="decision_timestamp")


def test_pit_rolling_uses_revision_order_not_availability_order():
    frame = pd.DataFrame(
        {
            "symbol": ["A"] * 3,
            "timestamp": ["2024-01-01T00:00:00Z"] * 3,
            "available_at": [
                "2024-01-10T00:00:00Z",  # revision 0 arrived late
                "2024-01-05T00:00:00Z",  # revision 1 arrived earlier
                "2024-01-01T00:00:00Z",
            ],
            "decision_timestamp": ["2024-01-11T00:00:00Z"] * 3,
            "revision": [0, 1, 0],
            "value": [1.0, 2.0, 9.0],
        }
    )
    out = pit_rolling(
        frame, "value", 1, decision_col="decision_timestamp", min_periods=1
    )
    assert out.tolist() == [2.0, 2.0, 2.0]


def test_event_conflict_rejects_only_conflicting_logical_group():
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z"] * 3,
            "symbol": ["A", "A", "B"],
            "available_at": ["2024-01-02T00:00:00Z"] * 3,
            "revision": [0, 0, 0],
            "value": [1.0, 2.0, 9.0],
        }
    )
    result = normalize_event_frame(
        frame,
        source="fixture",
        dataset="fundamentals",
        ingestion_timestamp="2024-02-01T00:00:00Z",
    )
    assert result.accepted["symbol"].tolist() == ["B"]
    assert result.accepted["value"].tolist() == [9.0]
    assert sorted(result.rejected["symbol"].tolist()) == ["A", "A"]
    assert set(result.rejected["rejection_reason"]) == {"conflicting_revision"}


def test_event_normalization_preserves_distinct_revisions():
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z"] * 2,
            "symbol": ["A", "A"],
            "available_at": ["2024-01-02T00:00:00Z", "2024-01-10T00:00:00Z"],
            "revision": [0, 1],
            "value": [1.0, 2.0],
        }
    )
    result = normalize_event_frame(
        frame,
        source="fixture",
        dataset="fundamentals",
        ingestion_timestamp="2024-02-01T00:00:00Z",
    )
    assert sorted(result.accepted["revision"].tolist()) == [0, 1]


def test_mixed_frequencies_require_explicit_selection():
    records = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "available_at": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "frequency": ["1d", "1m"],
            "close": [100.0, 101.0],
        }
    )
    decisions = pd.DataFrame(
        {"symbol": ["A"], "decision_timestamp": ["2024-01-02T00:00:00Z"]}
    )
    with pytest.raises(SchemaValidationError):
        pit_asof_join(decisions, records)
    out = pit_asof_join(decisions, records, frequency="1d")
    assert out.loc[0, "close"] == 100.0
    with pytest.raises(SchemaValidationError):
        pit_rolling(
            records.assign(timestamp=["2024-01-01T00:00:00Z"] * 2, decision_timestamp=["2024-01-02T00:00:00Z"] * 2),
            "close",
            1,
            decision_col="decision_timestamp",
        )

    frame = bars(
        ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
        closes=[100.0, 1.0],
        frequency=["1d", "1m"],
    )
    with pytest.raises(SchemaValidationError):
        resample_ohlcv(frame, "5min")
    minute = resample_ohlcv(frame, "5min", frequency="1m")
    assert minute["bar_frequency"].tolist() == ["5min"]
    assert minute["close"].tolist() == [1.0]


def test_resample_respects_closure_and_labels_by_interval_end():
    timestamps = [f"2024-01-01T00:0{i}:00Z" for i in range(0, 6)]
    frame = bars(timestamps, closes=[999.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    out = resample_ohlcv(frame, "5min", label="right", closed="right")
    assert len(out) == 2
    midnight = out[out["timestamp"] == pd.Timestamp("2024-01-01T00:00:00Z")].iloc[0]
    interval = out[out["timestamp"] == pd.Timestamp("2024-01-01T00:05:00Z")].iloc[0]
    assert midnight["close"] == 999.0
    assert interval["open"] == 1.0
    assert interval["close"] == 5.0
    assert interval["available_at"] == pd.Timestamp("2024-01-01T00:05:01Z")


def test_resampled_bar_cannot_leak_into_earlier_decision():
    timestamps = [f"2024-01-01T00:0{i}:00Z" for i in range(1, 6)]
    frame = bars(timestamps, closes=[1.0, 2.0, 3.0, 4.0, 5.0])
    resampled = resample_ohlcv(frame, "5min")
    early = pit_snapshot(resampled, "2024-01-01T00:04:59Z")
    assert early.empty
    at_close = pit_snapshot(resampled, "2024-01-01T00:05:01Z")
    assert len(at_close) == 1


def test_out_of_order_events_are_sorted_and_conflicting_duplicates_rejected():
    frame = bars(
        [
            "2024-01-03T00:00:00Z",
            "2024-01-02T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
        ],
        closes=[103.0, 102.0, 100.0, 101.0],
    )
    result = normalize(frame)
    assert list(result.accepted["timestamp"].astype(str)) == [
        "2024-01-02 00:00:00+00:00",
        "2024-01-03 00:00:00+00:00",
    ]
    assert result.report.ordering_violations > 0
    assert result.report.rejected_count == 2
    assert "conflicting_duplicate" in result.rejected.iloc[0]["rejection_reason"]


def test_missing_instrument_and_non_finite_values_are_rejected():
    frame = bars(["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"])
    frame.loc[0, "symbol"] = None
    frame.loc[1, "close"] = np.inf
    result = normalize(frame)
    assert result.accepted.empty
    reasons = set(result.rejected["rejection_reason"])
    assert any("missing_instrument" in reason for reason in reasons)
    assert any("non_finite_ohlcv" in reason for reason in reasons)


def test_ingestion_before_availability_is_rejected():
    frame = bars(["2024-01-01T00:00:00Z"], available=["2024-03-01T00:00:00Z"])
    result = normalize(frame)
    assert result.rejected.iloc[0]["rejection_reason"] == "ingestion_before_availability"


def test_exact_duplicate_ingestion_is_deduplicated():
    frame = bars(["2024-01-01T00:00:00Z"])
    result = normalize(pd.concat([frame, frame], ignore_index=True))
    assert len(result.accepted) == 1
    assert result.report.duplicate_count == 1


@pytest.mark.parametrize("revision", [1.5, -1, np.inf])
def test_invalid_revision_values_are_rejected_not_truncated(revision):
    frame = bars(["2024-01-01T00:00:00Z"], revision=[revision])
    result = normalize(frame)
    assert result.accepted.empty
    assert result.rejected.iloc[0]["rejection_reason"] == "invalid_revision"


def test_storage_rejects_invalid_revision_values(tmp_path):
    frame = normalize(bars(["2024-01-01T00:00:00Z"])).accepted.copy()
    frame["revision"] = 1.5
    with pytest.raises(SchemaValidationError):
        ParquetStore(tmp_path / "store").write(frame, source="fixture")
    with pytest.raises(SchemaValidationError):
        SQLiteStore(tmp_path / "bars.db").write(frame, source="fixture")


def test_market_value_cols_extend_same_revision_conflict_detection():
    frame = bars(
        ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
        closes=[100.0, 100.0],
        revision=[0, 0],
    )
    frame["vendor_note"] = ["a", "b"]
    plain = normalize(frame)
    assert len(plain.accepted) == 1
    checked = normalize(frame, value_cols=["vendor_note"])
    assert checked.accepted.empty
    assert set(checked.rejected["rejection_reason"]) == {"conflicting_duplicate"}


def test_invalid_assume_available_at_is_rejected():
    with pytest.raises(SchemaValidationError):
        normalize(bars(["2024-01-01T00:00:00Z"]), assume_available_at="sometimes")


def test_invalid_publication_ordering_rejected():
    frame = bars(["2024-01-01T00:00:00Z"], available=["2024-01-01T00:00:01Z"])
    frame["publication_timestamp"] = ["2024-01-01T00:00:02Z"]
    result = normalize(frame)
    assert result.report.rejected_count == 1
    assert "availability_before_publication" in result.rejected.iloc[0]["rejection_reason"]


def test_stale_observations_are_flagged_not_dropped():
    frame = bars(["2024-01-01T00:00:00Z"], available=["2024-01-01T00:00:01Z"])
    frame["decision_timestamp"] = ["2024-01-10T00:00:00Z"]
    out = flag_stale(frame, max_age="5D")
    assert bool(out.loc[0, "is_stale"])
    with pytest.raises(PointInTimeError):
        flag_stale(frame, max_age="5D", decision_col="timestamp", availability_col="available_at")


def test_pit_snapshot_honours_validity_interval_and_revisions():
    frame = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "available_at": ["2024-01-02T00:00:00Z", "2024-01-02T00:00:00Z"],
            "valid_from": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "valid_to": ["2024-01-20T00:00:00Z", "2024-01-20T00:00:00Z"],
            "revision": [0, 1],
            "value": [1.0, 2.0],
        }
    )
    selected = pit_snapshot(frame, "2024-01-10T00:00:00Z")
    assert selected.loc[0, "value"] == 2.0
    assert pit_snapshot(frame, "2024-01-21T00:00:00Z").empty


# ------------------------------------------------------------------- corporate actions
def split_ledger(ratio: float = 2.0, effective: str = "2024-01-03T00:00:00Z") -> CorporateActionLedger:
    return action_ledger(
        CorporateAction(
            "split-1",
            "A",
            ActionType.SPLIT,
            effective,
            "2024-01-01T00:00:00Z",
            ratio=ratio,
            source="fixture",
        )
    )


def test_split_adjustment_preserves_returns_and_adjusts_volume():
    frame = bars(
        ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"],
        closes=[100.0, 102.0, 51.0],
        volume=[1000.0, 1000.0, 1000.0],
    )
    result = adjust_prices(frame, split_ledger(), convention="split")
    adjusted = result.frame
    assert adjusted.loc[0, "adjusted_close"] == pytest.approx(50.0)
    assert adjusted.loc[1, "adjusted_close"] == pytest.approx(51.0)
    assert adjusted.loc[2, "adjusted_close"] == pytest.approx(51.0)
    assert adjusted.loc[0, "adjusted_volume"] == pytest.approx(2000.0)
    rets = adjusted["adjusted_close"].pct_change().dropna()
    assert rets.iloc[0] == pytest.approx(0.02)
    assert rets.iloc[1] == pytest.approx(0.0)


def test_reverse_split_adjustment():
    ledger = action_ledger(
        CorporateAction(
            "reverse-1",
            "A",
            ActionType.REVERSE_SPLIT,
            "2024-01-03T00:00:00Z",
            "2024-01-01T00:00:00Z",
            ratio=0.1,
        )
    )
    frame = bars(
        ["2024-01-01T00:00:00Z", "2024-01-03T00:00:00Z"],
        closes=[10.0, 100.0],
    )
    out = adjust_prices(frame, ledger, convention="split").frame
    assert out.loc[0, "adjusted_close"] == pytest.approx(100.0)
    assert out.loc[0, "adj_factor"] == pytest.approx(10.0)


def test_dividend_adjustment_uses_declared_reference_close():
    ledger = action_ledger(
        CorporateAction(
            "div-1",
            "A",
            ActionType.CASH_DIVIDEND,
            "2024-01-03T00:00:00Z",
            "2024-01-02T00:00:00Z",
            cash_amount=1.0,
            currency="USD",
        )
    )
    frame = bars(
        ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"],
        closes=[100.0, 99.0, 98.0],
    )
    out = adjust_prices(frame, ledger, convention="dividend").frame
    expected_factor = 1.0 - 1.0 / 99.0
    assert out.loc[0, "adj_factor"] == pytest.approx(expected_factor)
    assert out.loc[2, "adj_factor"] == pytest.approx(1.0)
    assert out.loc[0, "adjusted_close"] == pytest.approx(100.0 * expected_factor)


def test_total_return_combines_split_and_dividend():
    ledger = action_ledger(
        CorporateAction(
            "split-1", "A", ActionType.SPLIT, "2024-01-03T00:00:00Z",
            "2024-01-01T00:00:00Z", ratio=2.0,
        ),
        CorporateAction(
            "div-1", "A", ActionType.CASH_DIVIDEND, "2024-01-04T00:00:00Z",
            "2024-01-02T00:00:00Z", cash_amount=1.0, currency="USD",
        ),
    )
    frame = bars(
        ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"],
        closes=[100.0, 102.0, 51.0],
    )
    out = adjust_prices(frame, ledger, convention="total_return").frame
    expected = 0.5 * (1.0 - 1.0 / 51.0)
    assert out.loc[0, "adj_factor"] == pytest.approx(expected)
    assert out.loc[1, "adj_factor"] == pytest.approx(expected)


def test_missing_dividend_reference_fails_explicitly():
    ledger = action_ledger(
        CorporateAction(
            "div-1", "A", ActionType.CASH_DIVIDEND, "2024-01-01T00:00:00Z",
            "2023-12-01T00:00:00Z", cash_amount=1.0, currency="USD",
        )
    )
    frame = bars(
        ["2023-12-31T00:00:00Z", "2024-01-01T00:00:00Z"], closes=[100.0, 100.0]
    )
    frame.loc[0, "close"] = np.nan
    with pytest.raises(Exception, match="Invalid dividend reference close"):
        adjust_prices(frame, ledger, convention="dividend")


def test_ambiguous_split_records_fail():
    ledger = action_ledger(
        CorporateAction(
            "split-a", "A", ActionType.SPLIT, "2024-01-03T00:00:00Z",
            "2024-01-01T00:00:00Z", ratio=2.0,
        ),
        CorporateAction(
            "split-b", "A", ActionType.SPLIT, "2024-01-03T00:00:00Z",
            "2024-01-01T00:00:00Z", ratio=2.0,
        ),
    )
    frame = bars(["2024-01-01T00:00:00Z"], closes=[100.0])
    with pytest.raises(Exception, match="Ambiguous split"):
        adjust_prices(frame, ledger, convention="split")


def test_future_corporate_action_is_not_applied_to_pit_snapshot():
    ledger = split_ledger(effective="2024-01-05T00:00:00Z")
    frame = bars(["2024-01-01T00:00:00Z", "2024-01-06T00:00:00Z"], closes=[100.0, 50.0])
    early = pit_adjusted_prices(frame, ledger, "2024-01-03T00:00:00Z", convention="split")
    assert early.frame.loc[0, "adj_factor"] == pytest.approx(1.0)
    assert len(early.frame) == 1
    later = pit_adjusted_prices(frame, ledger, "2024-01-06T00:00:00Z", convention="split")
    assert later.frame.loc[0, "adj_factor"] == pytest.approx(0.5)


def test_corporate_action_revisions_are_reproducible():
    coverage = ActionCoverage(
        start="2000-01-01T00:00:00Z", end="2035-01-01T00:00:00Z", complete=True
    )
    ledger = CorporateActionLedger(
        [
            CorporateAction(
                "split-1", "A", ActionType.SPLIT, "2024-01-03T00:00:00Z",
                "2024-01-01T00:00:00Z", ratio=2.0, revision=0,
            )
        ],
        coverage=coverage,
    )
    frame = bars(["2024-01-01T00:00:00Z"], closes=[100.0])
    first = adjust_prices(
        frame, ledger, convention="split", decision_time="2024-01-05T00:00:00Z"
    )
    ledger.add(
        CorporateAction(
            "split-1", "A", ActionType.SPLIT, "2024-01-03T00:00:00Z",
            "2024-01-10T00:00:00Z", ratio=4.0, revision=1,
        )
    )
    replay = adjust_prices(
        frame, ledger, convention="split", decision_time="2024-01-05T00:00:00Z"
    )
    revised = adjust_prices(
        frame, ledger, convention="split", decision_time="2024-01-11T00:00:00Z"
    )
    assert first.frame.loc[0, "adj_factor"] == pytest.approx(0.5)
    assert replay.frame.loc[0, "adj_factor"] == pytest.approx(0.5)
    assert revised.frame.loc[0, "adj_factor"] == pytest.approx(0.25)


def test_double_adjustment_is_prevented():
    frame = bars(["2024-01-01T00:00:00Z"], closes=[100.0])
    ledger = split_ledger()
    adjusted = adjust_prices(frame, ledger, convention="split").frame
    with pytest.raises(DoubleAdjustmentError):
        adjust_prices(adjusted, ledger, convention="split")


def test_missing_action_coverage_fails_closed():
    ledger = CorporateActionLedger(
        [
            CorporateAction(
                "split-1", "A", ActionType.SPLIT, "2024-01-03T00:00:00Z",
                "2024-01-01T00:00:00Z", ratio=2.0,
            )
        ]
    )
    frame = bars(["2024-01-01T00:00:00Z"], closes=[100.0])
    with pytest.raises(IncompleteCorporateActionDataError):
        adjust_prices(frame, ledger, convention="split")
    # Bypassing the completeness gate still applies only explicitly known actions.
    partial = adjust_prices(frame, ledger, convention="split", require_complete=False)
    assert partial.frame.loc[0, "adj_factor"] == pytest.approx(0.5)


# ----------------------------------------------------------------------------- calendars
def nyse_fixture() -> SimpleExchangeCalendar:
    return SimpleExchangeCalendar(
        "XNYS-FIXTURE",
        timezone="America/New_York",
        holidays=frozenset({date(2024, 7, 4)}),
        early_closes={date(2024, 11, 29): time(13, 0)},
    )


def test_calendar_weekends_holidays_and_early_close():
    calendar = nyse_fixture()
    assert not calendar.is_session(date(2024, 7, 6))  # Saturday
    assert not calendar.is_session(date(2024, 7, 4))  # holiday
    early = calendar.session(date(2024, 11, 29))
    assert early.is_early_close
    assert early.close == pd.Timestamp("2024-11-29T18:00:00Z")


def test_calendar_dst_transitions_change_utc_session_bounds():
    calendar = nyse_fixture()
    before = calendar.session(date(2024, 3, 8))
    after = calendar.session(date(2024, 3, 11))
    assert before.open == pd.Timestamp("2024-03-08T14:30:00Z")
    assert after.open == pd.Timestamp("2024-03-11T13:30:00Z")
    assert before.close - before.open == after.close - after.open


def test_calendar_session_alignment_and_closure_vs_missing():
    calendar = nyse_fixture()
    schedule = InstrumentSchedule("A", calendar)
    frame = pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "B"],
            "timestamp": [
                "2024-07-03T15:00:00Z",
                "2024-07-04T15:00:00Z",
                "2024-07-05T22:30:00Z",
                "2024-07-03T15:00:00Z",
            ],
        }
    )
    aligned = align_bars_to_sessions(frame, {"A": schedule})
    assert aligned.loc[0, "session_status"] == "ok"
    assert aligned.loc[1, "session_status"] == "market_closed"
    assert aligned.loc[2, "session_status"] == "outside_session"
    assert aligned.loc[3, "session_status"] == "unknown_schedule"

    good = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "timestamp": ["2024-07-03T15:00:00Z", "2024-07-05T22:30:00Z"],
        }
    )
    missing = missing_session_bars(good, {"A": schedule}, date(2024, 7, 3), date(2024, 7, 5))
    assert list(missing["session_date"]) == [date(2024, 7, 5)]


def test_session_alignment_uses_exchange_local_date_for_apac():
    calendar = SimpleExchangeCalendar(
        "XNZX-FIXTURE",
        timezone="Pacific/Auckland",
        open_time=time(9),
        close_time=time(15),
    )
    schedule = InstrumentSchedule("A", calendar)
    frame = pd.DataFrame(
        {
            "symbol": ["A"],
            "timestamp": ["2024-01-01T20:00:00Z"],  # 2024-01-02 09:00 NZDT
        }
    )
    aligned = align_bars_to_sessions(frame, {"A": schedule})
    assert aligned.loc[0, "session_date"] == date(2024, 1, 2)
    assert aligned.loc[0, "session_status"] == "ok"


def test_session_alignment_uses_local_date_for_negative_utc_offset():
    calendar = nyse_fixture()
    schedule = InstrumentSchedule("A", calendar)
    frame = pd.DataFrame(
        {
            "symbol": ["A"],
            "timestamp": ["2024-07-03T01:00:00Z"],  # 2024-07-02 21:00 EDT
        }
    )
    aligned = align_bars_to_sessions(frame, {"A": schedule})
    assert aligned.loc[0, "session_date"] == date(2024, 7, 2)
    assert aligned.loc[0, "session_status"] == "outside_session"


def test_exchange_calendars_adapter_fails_explicitly_when_absent():
    try:
        import exchange_calendars  # noqa: F401
    except ImportError:
        with pytest.raises(CalendarUnavailableError):
            ExchangeCalendarsAdapter("XNYS")
    else:  # pragma: no cover - depends on optional environment
        pytest.skip("exchange_calendars is installed")


# ------------------------------------------------------------------------------- universe
def universe_fixture() -> UniverseStore:
    coverage = UniverseCoverage(
        start="2010-01-01T00:00:00Z", end="2035-01-01T00:00:00Z", complete=True
    )
    store = UniverseStore(coverage=coverage)
    store.add_instrument(
        Instrument(
            "ID-DEAD", "DEAD", "XNYS", "2010-01-01T00:00:00Z",
            listing_date=date(2010, 1, 1), delisting_date=date(2020, 6, 1),
        )
    )
    store.add_instrument(
        Instrument(
            "ID-LIVE", "LIVE", "XNYS", "2010-01-01T00:00:00Z",
            listing_date=date(2012, 1, 1),
        )
    )
    store.add_symbol_mapping(
        SymbolMapping("m-dead", "DEAD", "ID-DEAD", "XNYS",
                      "2010-01-01T00:00:00Z", "2010-01-01T00:00:00Z")
    )
    store.add_symbol_mapping(
        SymbolMapping("m-live", "LIVE", "ID-LIVE", "XNYS",
                      "2012-01-01T00:00:00Z", "2012-01-01T00:00:00Z")
    )
    store.add_membership(
        UniverseMembership("u-dead", "TEST", "ID-DEAD", "2010-01-01T00:00:00Z",
                           "2010-01-01T00:00:00Z")
    )
    store.add_membership(
        UniverseMembership("u-live", "TEST", "ID-LIVE", "2012-01-01T00:00:00Z",
                           "2012-01-01T00:00:00Z")
    )
    store.add_lifecycle_event(
        LifecycleEvent("e-dead", "ID-DEAD", LifecycleEventType.DELISTED,
                       "2020-06-01T00:00:00Z", "2020-05-01T00:00:00Z", reason="bankrupt")
    )
    return store


def test_delisted_instrument_remains_in_historical_universe():
    store = universe_fixture()
    before = store.universe_as_of("TEST", "2019-01-01T00:00:00Z")
    assert before.coverage_status == "complete"
    assert {m.instrument_id for m in before.members} == {"ID-DEAD", "ID-LIVE"}
    dead = next(m for m in before.members if m.instrument_id == "ID-DEAD")
    assert dead.tradable is True and dead.eligible
    after = store.universe_as_of("TEST", "2021-01-01T00:00:00Z")
    dead_after = next(m for m in after.members if m.instrument_id == "ID-DEAD")
    assert dead_after.status is InstrumentStatus.DELISTED
    assert dead_after.member is True
    assert dead_after.tradable is False
    assert not dead_after.eligible


@pytest.mark.parametrize(
    "event_type,status",
    [
        (LifecycleEventType.ACQUIRED, InstrumentStatus.ACQUIRED),
        (LifecycleEventType.BANKRUPT, InstrumentStatus.BANKRUPT),
        (LifecycleEventType.MERGED, InstrumentStatus.MERGED),
        (LifecycleEventType.SUSPENDED, InstrumentStatus.SUSPENDED),
    ],
)
def test_lifecycle_down_states_are_flagged(event_type, status):
    store = universe_fixture()
    store.add_lifecycle_event(
        LifecycleEvent("life", "ID-LIVE", event_type, "2019-01-01T00:00:00Z",
                       "2018-12-01T00:00:00Z")
    )
    snap = store.universe_as_of("TEST", "2019-06-01T00:00:00Z")
    live = next(m for m in snap.members if m.instrument_id == "ID-LIVE")
    assert live.status is status
    assert live.tradable is False
    assert not live.eligible


def test_symbol_rename_mapping_is_pit_aware():
    store = universe_fixture()
    store.add_symbol_mapping(
        SymbolMapping("m-renamed", "LIVE2", "ID-LIVE", "XNYS",
                      "2025-01-01T00:00:00Z", "2024-12-01T00:00:00Z")
    )
    assert store.resolve_symbol("LIVE", "2024-06-01T00:00:00Z") == "ID-LIVE"
    assert store.resolve_symbol("LIVE2", "2024-06-01T00:00:00Z") is None
    assert store.resolve_symbol("LIVE2", "2025-06-01T00:00:00Z") == "ID-LIVE"
    assert store.symbol_for("ID-LIVE", "2025-06-01T00:00:00Z") == "LIVE2"


def test_membership_available_later_cannot_leak_and_is_explicitly_incomplete():
    store = universe_fixture()
    store.add_membership(
        UniverseMembership("u-late", "TEST", "ID-LIVE", "2010-01-01T00:00:00Z",
                           "2024-01-01T00:00:00Z")
    )
    snap = store.universe_as_of("TEST", "2015-01-01T00:00:00Z")
    assert snap.coverage_status == "incomplete_late_membership"
    assert "u-late" in snap.late_membership_records
    assert any("available after the decision" in reason for reason in snap.incomplete_reasons)
    with pytest.raises(IncompleteUniverseError):
        store.universe_as_of("TEST", "2015-01-01T00:00:00Z", require_complete=True)


def test_no_coverage_declaration_is_explicitly_incomplete():
    store = universe_fixture()
    store.coverage = None
    snap = store.universe_as_of("TEST", "2019-01-01T00:00:00Z")
    assert snap.coverage_status == "unknown"
    assert not snap.complete
    with pytest.raises(IncompleteUniverseError):
        snap.assert_complete()


def test_unknown_instrument_metadata_is_incomplete():
    store = universe_fixture()
    store.add_membership(
        UniverseMembership("u-ghost", "TEST", "ID-GHOST", "2010-01-01T00:00:00Z",
                           "2010-01-01T00:00:00Z")
    )
    snap = store.universe_as_of("TEST", "2019-01-01T00:00:00Z")
    assert snap.coverage_status == "incomplete_unknown_instrument"
    assert "ID-GHOST" in snap.unknown_instruments


def test_insufficient_data_excludes_instrument_from_eligible_set():
    store = universe_fixture()
    snap = store.universe_as_of(
        "TEST", "2019-01-01T00:00:00Z", data_coverage={"ID-DEAD": False, "ID-LIVE": True}
    )
    dead = next(m for m in snap.members if m.instrument_id == "ID-DEAD")
    assert dead.data_available is False
    assert not dead.eligible
    assert "insufficient_data" in dead.reasons


def test_symbol_for_does_not_fall_back_to_instrument_symbol():
    store = UniverseStore()
    store.add_instrument(
        Instrument("ID-NO-MAP", "NOMAP", "XNYS", "2020-01-01T00:00:00Z")
    )
    assert store.symbol_for("ID-NO-MAP", "2021-01-01T00:00:00Z") is None


def test_universe_snapshot_is_reproducible():
    store = universe_fixture()
    first = store.universe_as_of("TEST", "2019-01-01T00:00:00Z")
    second = store.universe_as_of("TEST", "2019-01-01T00:00:00Z")
    assert first.snapshot_id == second.snapshot_id
    assert first.to_frame().equals(second.to_frame())


# ------------------------------------------------------------------------------ storage
def test_parquet_round_trip_preserves_schema_and_timezone(tmp_path):
    frame = normalize(bars(["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"])).accepted
    store = ParquetStore(tmp_path / "store")
    receipt = store.write(frame, source="fixture", ingestion_timestamp="2024-02-01T00:00:00Z")
    out = store.read()
    assert receipt.rows == 2
    assert str(out["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert str(out["available_at"].dtype) == "datetime64[ns, UTC]"
    assert out["close"].tolist() == frame["close"].tolist()


def test_sqlite_round_trip_preserves_schema_and_timezone(tmp_path):
    frame = normalize(bars(["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"])).accepted
    store = SQLiteStore(tmp_path / "bars.db")
    receipt = store.write(frame, source="fixture", ingestion_timestamp="2024-02-01T00:00:00Z")
    out = store.read_bars()
    assert receipt.rows == 2
    assert str(out["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert out["close"].tolist() == frame["close"].tolist()
    assert len(store.ingestion_runs()) == 1


def test_duplicate_storage_writes_are_idempotent(tmp_path):
    frame = normalize(bars(["2024-01-01T00:00:00Z"])).accepted
    for store, read in (
        (ParquetStore(tmp_path / "store"), lambda s: s.read()),
        (SQLiteStore(tmp_path / "bars.db"), lambda s: s.read_bars()),
    ):
        first = store.write(frame, source="fixture", ingestion_timestamp="2024-02-01T00:00:00Z")
        second = store.write(frame, source="fixture", ingestion_timestamp="2024-02-02T00:00:00Z")
        assert first.checksum == second.checksum
        assert second.idempotent
        assert len(read(store)) == 1


def test_sqlite_mixed_write_is_not_marked_idempotent(tmp_path):
    first = normalize(bars(["2024-01-01T00:00:00Z"])).accepted
    both = normalize(bars(["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"])).accepted
    store = SQLiteStore(tmp_path / "bars.db")
    store.write(first, source="fixture", ingestion_timestamp="2024-02-01T00:00:00Z")
    receipt = store.write(both, source="fixture", ingestion_timestamp="2024-02-02T00:00:00Z")
    assert not receipt.idempotent
    assert len(store.read_bars()) == 2


def test_same_revision_conflict_is_rejected_without_overwrite(tmp_path):
    frame = normalize(bars(["2024-01-01T00:00:00Z"], closes=[100.0])).accepted
    conflicting = frame.copy()
    conflicting["close"] = 101.0
    parquet = ParquetStore(tmp_path / "store")
    parquet.write(frame, source="fixture", ingestion_timestamp="2024-02-01T00:00:00Z")
    before = parquet.read()["close"].tolist()
    with pytest.raises(StorageConflictError):
        parquet.write(conflicting, source="fixture", ingestion_timestamp="2024-02-02T00:00:00Z")
    assert parquet.read()["close"].tolist() == before

    sqlite = SQLiteStore(tmp_path / "bars.db")
    sqlite.write(frame, source="fixture", ingestion_timestamp="2024-02-01T00:00:00Z")
    with pytest.raises(StorageConflictError):
        sqlite.write(conflicting, source="fixture", ingestion_timestamp="2024-02-02T00:00:00Z")


def test_multi_partition_parquet_conflict_preserves_inventory_and_data(tmp_path):
    store = ParquetStore(tmp_path / "store")
    b_frame = normalize(bars(["2024-01-01T00:00:00Z"], symbol="B", closes=[2.0])).accepted
    store.write(b_frame, source="fixture", ingestion_timestamp="2024-02-01T00:00:00Z")
    before = {
        str(path.relative_to(tmp_path / "store")): path.read_bytes()
        for path in (tmp_path / "store").rglob("*")
        if path.is_file()
    }

    a_frame = normalize(bars(["2024-01-02T00:00:00Z"], symbol="A", closes=[10.0])).accepted
    conflicting_b = normalize(
        bars(["2024-01-01T00:00:00Z"], symbol="B", closes=[3.0])
    ).accepted
    with pytest.raises(StorageConflictError):
        store.write(
            pd.concat([a_frame, conflicting_b], ignore_index=True),
            source="fixture",
            ingestion_timestamp="2024-02-02T00:00:00Z",
        )

    after = {
        str(path.relative_to(tmp_path / "store")): path.read_bytes()
        for path in (tmp_path / "store").rglob("*")
        if path.is_file()
    }
    assert after == before
    assert store.read(latest=False)["close"].tolist() == [2.0]


def test_parquet_commit_failure_rolls_back_after_preflight(tmp_path, monkeypatch):
    store = ParquetStore(tmp_path / "store")
    b_frame = normalize(bars(["2024-01-01T00:00:00Z"], symbol="B", closes=[2.0])).accepted
    store.write(b_frame, source="fixture", ingestion_timestamp="2024-02-01T00:00:00Z")
    before = {
        str(path.relative_to(tmp_path / "store")): path.read_bytes()
        for path in (tmp_path / "store").rglob("*")
        if path.is_file()
    }

    a_frame = normalize(bars(["2024-01-02T00:00:00Z"], symbol="A", closes=[10.0])).accepted
    b_revision = normalize(
        bars(["2024-01-01T00:00:00Z"], symbol="B", closes=[4.0], revision=[1])
    ).accepted
    original = ParquetStore._commit_staged
    calls = {"count": 0}

    def flaky_commit(self, temp, target):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated commit failure")
        return original(self, temp, target)

    monkeypatch.setattr(ParquetStore, "_commit_staged", flaky_commit)
    with pytest.raises(StorageError):
        store.write(
            pd.concat([a_frame, b_revision], ignore_index=True),
            source="fixture",
            ingestion_timestamp="2024-02-02T00:00:00Z",
        )

    after = {
        str(path.relative_to(tmp_path / "store")): path.read_bytes()
        for path in (tmp_path / "store").rglob("*")
        if path.is_file()
    }
    assert after == before
    assert store.read(latest=False)["close"].tolist() == [2.0]
    assert not list((tmp_path / "store").glob(".parquet-stage-*"))


def test_raw_archive_is_content_addressed_verified_and_redacts_secrets(tmp_path):
    archive = RawArchive(tmp_path / "raw")
    receipt = archive.write(
        b'{"payload":1}',
        source="vendor",
        received_at="2024-01-01T00:00:00Z",
        metadata={"api_key": "super-secret", "note": "ok"},
    )
    assert archive.verify(receipt)
    assert archive.read(receipt) == b'{"payload":1}'
    again = archive.write(
        b'{"payload":1}', source="vendor", received_at="2024-01-02T00:00:00Z"
    )
    assert again.sha256 == receipt.sha256
    sidecars = list((tmp_path / "raw").rglob("*.meta.json"))
    text = "\n".join(path.read_text() for path in sidecars)
    assert "super-secret" not in text
    assert "<redacted>" in text


def test_raw_archive_redacts_nested_secret_metadata(tmp_path):
    archive = RawArchive(tmp_path / "raw")
    receipt = archive.write(
        b"x",
        source="vendor",
        received_at="2024-01-01T00:00:00Z",
        metadata={"auth": {"token": "nested-secret"}, "items": [{"password": "p"}]},
    )
    text = (tmp_path / "raw").rglob("*.meta.json")
    sidecar_text = next(text).read_text()
    assert "nested-secret" not in sidecar_text
    assert "\"p\"" not in sidecar_text
    assert "<redacted>" in sidecar_text
    assert archive.verify(receipt)


def test_raw_archive_rejects_path_traversal_source(tmp_path):
    archive = RawArchive(tmp_path / "raw")
    with pytest.raises(StorageError):
        archive.write(b"x", source="..")
    with pytest.raises(StorageError):
        archive.write(b"x", source="../outside")
    assert not (tmp_path / "outside").exists()


def test_raw_archive_is_content_addressed_across_receipt_dates(tmp_path):
    archive = RawArchive(tmp_path / "raw")
    first = archive.write(b"same", source="vendor", received_at="2024-01-01T00:00:00Z")
    second = archive.write(b"same", source="vendor", received_at="2024-01-02T00:00:00Z")
    assert first.path == second.path
    assert len(list((tmp_path / "raw").rglob("*.json"))) == 2  # payload + one sidecar


def test_parquet_checksum_is_independent_of_ingestion_timestamp(tmp_path):
    frame = normalize(bars(["2024-01-01T00:00:00Z"])).accepted
    parquet_a = ParquetStore(tmp_path / "a")
    parquet_b = ParquetStore(tmp_path / "b")
    first = parquet_a.write(frame, source="fixture", ingestion_timestamp="2024-02-01T00:00:00Z")
    second = parquet_b.write(frame, source="fixture", ingestion_timestamp="2024-03-01T00:00:00Z")
    assert first.checksum == second.checksum

    sqlite_a = SQLiteStore(tmp_path / "a.db")
    sqlite_b = SQLiteStore(tmp_path / "b.db")
    first_sql = sqlite_a.write(frame, source="fixture", ingestion_timestamp="2024-02-01T00:00:00Z")
    second_sql = sqlite_b.write(frame, source="fixture", ingestion_timestamp="2024-03-01T00:00:00Z")
    assert first_sql.checksum == second_sql.checksum
    partition = first.partitions[0]
    assert parquet_a._manifest_entry(partition, {})["checksum"] == first_sql.checksum


def test_fingerprint_is_stable_across_row_order():
    frame = normalize(bars(["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"])).accepted
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    assert frame_fingerprint(frame, key_cols=("timestamp", "symbol")) == frame_fingerprint(
        shuffled, key_cols=("timestamp", "symbol")
    )


# --------------------------------------------------------------------- connectors / e2e
def local_fixture(tmp_path, frame: pd.DataFrame | None = None) -> tuple[LocalFileConnector, ParquetStore, RawArchive]:
    root = tmp_path / "incoming"
    (root / "1d").mkdir(parents=True)
    (frame if frame is not None else normalize(bars(
        ["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"]
    )).accepted).to_parquet(root / "1d" / "A.parquet", index=False)
    connector = LocalFileConnector(ConnectorConfig(source_name="local-fixture"), root)
    return connector, ParquetStore(tmp_path / "store"), RawArchive(tmp_path / "raw")


def test_local_parquet_ingestion_end_to_end(tmp_path):
    connector, store, archive = local_fixture(tmp_path)
    result = ingest(
        connector,
        DataRequest(instruments=("A",), start="2024-01-01T00:00:00Z", end="2024-01-01T00:02:00Z"),
        store=store,
        raw_archive=archive,
        ingestion_timestamp="2024-02-01T00:00:00Z",
    )
    assert result.status is IngestionStatus.OK
    assert result.report.accepted_count == 2
    assert result.storage_receipt is not None
    assert len(result.raw_receipts) == 1
    assert result.raw_receipts[0].sha256
    assert len(store.read()) == 2
    assert result.provenance.accepted_records == 2
    assert result.provenance.checksum


def test_malformed_records_rejected_with_diagnostics(tmp_path):
    frame = normalize(bars(["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"])).accepted
    frame.loc[1, "close"] = -1.0
    frame.loc[1, "low"] = -2.0
    connector, store, _ = local_fixture(tmp_path, frame)
    result = ingest(
        connector,
        DataRequest(instruments=("A",)),
        store=store,
        ingestion_timestamp="2024-02-01T00:00:00Z",
    )
    assert result.status is IngestionStatus.PARTIAL
    assert result.report.rejected_count == 1
    assert "non_positive_price" in result.report.rejection_reasons
    assert result.report.rejected_sample


def test_local_connector_rejects_path_traversal_symbol(tmp_path):
    root = tmp_path / "incoming"
    (root / "1d").mkdir(parents=True)
    connector = LocalFileConnector(ConnectorConfig(source_name="local"), root)
    with pytest.raises(SchemaValidationError):
        list(connector.fetch(DataRequest(instruments=("../outside",))))
    with pytest.raises(SchemaValidationError):
        list(connector.fetch(DataRequest(instruments=("/tmp/outside",))))


def test_local_connector_passes_invalid_timestamps_to_normalization(tmp_path):
    raw = bars(
        ["not-a-timestamp", "2024-01-01T00:00:00Z"],
        closes=[100.0, 101.0],
        available=["2024-01-02T00:00:00Z", "2024-01-02T00:00:00Z"],
    )
    connector, store, _ = local_fixture(tmp_path, raw)
    result = ingest(
        connector,
        DataRequest(instruments=("A",)),
        store=store,
        ingestion_timestamp="2024-02-01T00:00:00Z",
    )
    assert result.status is IngestionStatus.PARTIAL
    assert result.report.rejected_count == 1
    assert any("timestamp" in reason.lower() for reason in result.report.rejection_reasons)
    assert len(store.read()) == 1


def test_sqlite_connections_are_closed(tmp_path):
    store = SQLiteStore(tmp_path / "bars.db")
    with store._connect() as connection:
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_partial_local_ingestion_surfaces_missing_instrument(tmp_path):
    connector, store, _ = local_fixture(tmp_path)
    result = ingest(
        connector,
        DataRequest(instruments=("A", "MISSING")),
        store=store,
        ingestion_timestamp="2024-02-01T00:00:00Z",
    )
    assert result.status is IngestionStatus.PARTIAL
    assert any("missing instruments" in e for e in result.errors)
    assert len(store.read()) == 2
    assert len(result.accepted) == 2


def test_incremental_policy_applies_overlap_to_request(tmp_path):
    connector, store, _ = local_fixture(tmp_path)
    ingest(
        connector,
        DataRequest(instruments=("A",), start="2024-01-01T00:00:00Z"),
        store=store,
        ingestion_timestamp="2024-02-01T00:00:00Z",
    )
    result = ingest(
        connector,
        DataRequest(instruments=("A",)),
        store=store,
        incremental=IncrementalPolicy(overlap=pd.Timedelta(minutes=1)),
        ingestion_timestamp="2024-02-02T00:00:00Z",
    )
    assert result.provenance.request["start"] == "2024-01-01T00:00:00+00:00"


def test_http_connector_retries_rate_limit_and_paginates_without_network():
    calls = {"count": 0}

    def fake_transport(request: HttpRequest) -> HttpResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            return HttpResponse(429, b"", {"retry-after": "0"})
        if "page_token=next" in request.url:
            return HttpResponse(
                200,
                b'{"records":[{"timestamp":"2024-01-01T00:01:00Z","symbol":"A",'
                b'"open":1,"high":1,"low":1,"close":1,"volume":1,'
                b'"available_at":"2024-01-01T00:01:01Z"}],"next_page_token":null}',
            )
        return HttpResponse(
            200,
            b'{"records":[{"timestamp":"2024-01-01T00:00:00Z","symbol":"A",'
            b'"open":1,"high":1,"low":1,"close":1,"volume":1,'
            b'"available_at":"2024-01-01T00:00:01Z"}],"next_page_token":"next"}',
        )

    connector = HttpJSONConnector(
        ConnectorConfig(source_name="fake", max_retries=2, backoff_base_seconds=0),
        "https://vendor.invalid/records",
        transport=fake_transport,
        sleep=lambda _: None,
    )
    batches = list(connector.fetch(DataRequest(instruments=("A",))))
    assert len(batches) == 2
    assert connector.retry_attempts == 1
    assert calls["count"] == 3


def test_http_connector_exhausts_retries_on_5xx():
    calls = {"count": 0}

    def unavailable(request: HttpRequest) -> HttpResponse:
        calls["count"] += 1
        return HttpResponse(503, b"")

    connector = HttpJSONConnector(
        ConnectorConfig(source_name="fake", max_retries=2, backoff_base_seconds=0),
        "https://vendor.invalid/records",
        transport=unavailable,
        sleep=lambda _: None,
    )
    with pytest.raises(RetryExhaustedError):
        list(connector.fetch(DataRequest(instruments=("A",))))
    assert calls["count"] == 3
    assert connector.retry_attempts == 2


def test_http_connector_rejects_repeated_page_token():
    calls = {"count": 0}

    def repeated(request: HttpRequest) -> HttpResponse:
        calls["count"] += 1
        return HttpResponse(200, b'{"records":[],"next_page_token":"same"}')

    connector = HttpJSONConnector(
        ConnectorConfig(source_name="fake"),
        "https://vendor.invalid/records",
        transport=repeated,
    )
    with pytest.raises(ConnectorError, match="repeated page token"):
        list(connector.fetch(DataRequest(instruments=("A",))))
    assert calls["count"] == 2


def test_http_connector_rejects_malformed_json():
    connector = HttpJSONConnector(
        ConnectorConfig(source_name="fake"),
        "https://vendor.invalid/records",
        transport=lambda request: HttpResponse(200, b"not-json"),
    )
    with pytest.raises(ConnectorError, match="invalid JSON"):
        list(connector.fetch(DataRequest(instruments=("A",))))


def test_default_http_transport_requires_explicit_network_permission():
    with pytest.raises(NetworkDisabledError):
        urllib_transport(HttpRequest("https://vendor.invalid"), config=ConnectorConfig(source_name="x"))


def test_synthetic_end_to_end_pit_snapshot_blocks_future_data(tmp_path):
    frame = normalize(
        bars(
            ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"],
            closes=[100.0, 101.0, 999.0],
            available=[
                "2024-01-01T00:00:01Z",
                "2024-01-02T00:00:01Z",
                "2024-01-10T00:00:00Z",  # future publication
            ],
        )
    ).accepted
    connector, store, _ = local_fixture(tmp_path, frame)
    result = ingest(
        connector,
        DataRequest(instruments=("A",)),
        store=store,
        ingestion_timestamp="2024-02-01T00:00:00Z",
    )
    assert result.status is IngestionStatus.OK

    snapshot = pit_snapshot(store.read(), "2024-01-02T12:00:00Z")
    assert snapshot["close"].tolist() == [100.0, 101.0]
    assert 999.0 not in snapshot["close"].tolist()

    decision = pd.DataFrame(
        {"symbol": ["A"], "decision_timestamp": ["2024-01-02T12:00:00Z"]}
    )
    joined = pit_asof_join(decision, store.read())
    assert joined.loc[0, "close"] == 101.0


def test_deterministic_replay_after_ingestion(tmp_path):
    connector, store, _ = local_fixture(tmp_path)
    request = DataRequest(instruments=("A",))
    first = ingest(connector, request, store=store, ingestion_timestamp="2024-02-01T00:00:00Z")
    second = ingest(connector, request, store=store, ingestion_timestamp="2024-02-02T00:00:00Z")
    assert first.provenance.checksum == second.provenance.checksum
    assert second.storage_receipt is not None and second.storage_receipt.idempotent
