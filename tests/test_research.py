"""Module 2 contract tests: PIT features, labels, datasets, and time splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantforge import (
    FeatureSpec,
    LabelSpec,
    ResearchDataset,
    build_dataset,
    compute_features,
    compute_labels,
    fracdiff_spec,
    purged_time_split,
    validate_dataset,
)
from quantforge.data.corporate_actions import (
    ActionCoverage,
    ActionType,
    CorporateAction,
    CorporateActionLedger,
)
from quantforge.data.errors import (
    ConflictingRevisionsError,
    IncompleteUniverseError,
    SchemaValidationError,
)
from quantforge.dataset import hash_frame, hash_payload
from quantforge.features import momentum_spec


def frame(
    timestamps: list[str],
    *,
    symbol: str = "A",
    closes: list[float] | None = None,
    available: list[str] | None = None,
    **overrides: list[object],
) -> pd.DataFrame:
    closes = closes or [100.0 + i for i in range(len(timestamps))]
    available = available or [
        (pd.Timestamp(ts) + pd.Timedelta(seconds=1)).isoformat() for ts in timestamps
    ]
    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": [symbol] * len(timestamps),
            "open": closes,
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
            "volume": [1000.0] * len(timestamps),
            "available_at": available,
        }
    )
    for column, values in overrides.items():
        data[column] = values
    return data


def decisions(*values: str, symbol: str = "A") -> pd.DataFrame:
    return pd.DataFrame(
        {"symbol": [symbol] * len(values), "decision_timestamp": list(values)}
    )


def ledger(*actions: CorporateAction, complete: bool = True) -> CorporateActionLedger:
    return CorporateActionLedger(
        list(actions),
        coverage=ActionCoverage(
            start="2000-01-01T00:00:00Z",
            end="2035-01-01T00:00:00Z",
            complete=complete,
        ),
    )


# --------------------------------------------------------------------------- features
def test_feature_uses_only_available_history():
    bars = frame(
        ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"],
        closes=[100.0, 110.0, 999.0],
    )
    decision = decisions("2024-01-02T12:00:00Z")
    spec = FeatureSpec(
        "last_close",
        ("close",),
        lambda history: history["close"].iloc[-1],
        min_periods=1,
    )
    result = compute_features(decision, bars, [spec])
    assert result.frame.loc[0, "last_close"] == 110.0
    changed = bars.copy()
    changed.loc[2, "close"] = np.inf
    second = compute_features(decision, changed, [spec])
    assert result.frame.equals(second.frame)


def test_feature_revision_is_selected_at_decision_time():
    bars = frame(
        ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
        closes=[100.0, 200.0],
        available=["2024-01-02T00:00:00Z", "2024-01-10T00:00:00Z"],
        revision=[0, 1],
    )
    spec = FeatureSpec(
        "last_close", ("close",), lambda history: history["close"].iloc[-1], min_periods=1
    )
    early = compute_features(decisions("2024-01-05T00:00:00Z"), bars, [spec])
    late = compute_features(decisions("2024-01-11T00:00:00Z"), bars, [spec])
    assert early.frame.loc[0, "last_close"] == 100.0
    assert late.frame.loc[0, "last_close"] == 200.0


def test_feature_isolates_instruments_and_requires_frequency():
    bars = pd.concat(
        [
            frame(["2024-01-01T00:00:00Z"], symbol="A", closes=[100.0]),
            frame(["2024-01-01T00:00:00Z"], symbol="B", closes=[5.0]),
        ],
        ignore_index=True,
    )
    spec = FeatureSpec(
        "last_close", ("close",), lambda history: history["close"].iloc[-1], min_periods=1
    )
    out = compute_features(decisions("2024-01-02T00:00:00Z", symbol="A"), bars, [spec])
    assert out.frame.loc[0, "last_close"] == 100.0

    mixed = pd.concat(
        [
            frame(["2024-01-01T00:00:00Z"], closes=[100.0], frequency=["1d"]),
            frame(["2024-01-01T00:00:00Z"], closes=[5.0], frequency=["1h"]),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError):
        compute_features(decisions("2024-01-02T00:00:00Z"), mixed, [spec])
    selected = compute_features(
        decisions("2024-01-02T00:00:00Z"), mixed, [spec], frequency="1d"
    )
    assert selected.frame.loc[0, "last_close"] == 100.0


def test_feature_conflicts_fail_and_insufficient_history_is_explicit():
    conflicting = frame(
        ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
        closes=[100.0, 101.0],
        revision=[0, 0],
    )
    spec = FeatureSpec(
        "last_close", ("close",), lambda history: history["close"].iloc[-1], min_periods=2
    )
    with pytest.raises(ConflictingRevisionsError):
        compute_features(decisions("2024-01-02T00:00:00Z"), conflicting, [spec])

    one_bar = frame(["2024-01-01T00:00:00Z"], closes=[100.0])
    out = compute_features(decisions("2024-01-02T00:00:00Z"), one_bar, [spec])
    assert pd.isna(out.frame.loc[0, "last_close"])
    assert out.frame.loc[0, "last_close__status"] == "insufficient_history"


def test_feature_metadata_and_nan_are_retained():
    bars = frame(
        ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
        closes=[100.0, 110.0],
        source=["vendor", "vendor"],
        revision=[0, 1],
    )
    decision = decisions("2024-01-02T12:00:00Z")
    result = compute_features(decision, bars, [momentum_spec(1)])
    row = result.frame.iloc[0]
    assert row["momentum_1"] == pytest.approx(0.1)
    assert row["momentum_1__available_at"] == pd.Timestamp("2024-01-02T12:00:00Z")
    assert row["momentum_1__source_max_available_at"] == pd.Timestamp(
        "2024-01-02T00:00:01Z"
    )
    assert row["momentum_1__source"] == "vendor"
    assert row["momentum_1__revision"] == 1
    assert row["feature_available_at"] == pd.Timestamp("2024-01-02T12:00:00Z")
    assert row["feature_source"] == "vendor"
    assert row["feature_revision"] == 1
    assert row["feature_status"] == "ok"

    nan_spec = FeatureSpec(
        "always_nan", ("close",), lambda history: np.nan, min_periods=1
    )
    nan_out = compute_features(decision, bars, [nan_spec])
    assert nan_out.frame.loc[0, "always_nan__status"] == "warmup_nan"


def test_feature_rejects_infinite_input():
    bars = frame(["2024-01-01T00:00:00Z"], closes=[np.inf])
    spec = FeatureSpec("last", ("close",), lambda history: 1.0, min_periods=1)
    with pytest.raises(ValueError):
        compute_features(decisions("2024-01-02T00:00:00Z"), bars, [spec])


def test_fracdiff_spec_default_and_parameters_have_distinct_valid_names():
    default = fracdiff_spec()
    assert default.name == "fracdiff_0_5_threshold_1em05"
    assert default.name.isidentifier()

    quarter = fracdiff_spec(0.25)
    assert quarter.name.isidentifier()
    assert quarter.name != default.name

    threshold_variant = fracdiff_spec(0.5, threshold=1e-4)
    assert threshold_variant.name.isidentifier()
    assert threshold_variant.name != default.name

    custom = fracdiff_spec(0.5, name="my_fracdiff")
    assert custom.name == "my_fracdiff"

    with pytest.raises(ValueError):
        fracdiff_spec(-0.01)
    with pytest.raises(ValueError):
        fracdiff_spec(1.01)
    with pytest.raises(ValueError):
        fracdiff_spec("0.5")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        fracdiff_spec(0.5, threshold=0.0)
    with pytest.raises(ValueError):
        fracdiff_spec(0.5, threshold=np.inf)


def test_feature_callback_is_trusted_code_not_sandboxed():
    external_state = {"value": 999.0}
    bars = frame(["2024-01-01T00:00:00Z"], closes=[100.0])
    spec = FeatureSpec(
        "external_value",
        ("close",),
        lambda history: external_state["value"],
        min_periods=1,
    )
    out = compute_features(decisions("2024-01-02T00:00:00Z"), bars, [spec])
    assert out.frame.loc[0, "external_value"] == 999.0


# ----------------------------------------------------------------------------- labels
def test_label_uses_next_observed_bar_and_full_horizon():
    bars = frame(
        ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"],
        closes=[100.0, 110.0, 132.0],
    )
    out = compute_labels(
        decisions("2023-12-31T12:00:00Z", "2024-01-01T12:00:00Z"),
        bars,
        LabelSpec("forward_1", horizon=1),
    )
    assert out.frame.loc[0, "forward_1"] == pytest.approx(0.10)
    assert out.frame.loc[1, "forward_1"] == pytest.approx(0.20)
    assert out.frame.loc[0, "label_start_timestamp"] == pd.Timestamp(
        "2024-01-01T00:00:00Z"
    )
    assert out.frame.loc[0, "label_end_timestamp"] == pd.Timestamp(
        "2024-01-02T00:00:00Z"
    )
    assert out.frame.loc[0, "label_available_at"] == pd.Timestamp(
        "2024-01-02T00:00:01Z"
    )
    assert out.frame.loc[0, "backtest_compatibility"] == "not_claimed"


def test_label_missing_entry_exit_and_price_are_not_shortened():
    bars = frame(
        ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"],
        closes=[100.0, 110.0, 132.0],
    )
    missing_entry = compute_labels(
        decisions("2024-01-05T00:00:00Z"), bars, LabelSpec("forward_1", horizon=1)
    )
    assert missing_entry.frame.loc[0, "label_status"] == "missing_entry"

    missing_exit = compute_labels(
        decisions("2024-01-01T00:00:00Z"), bars, LabelSpec("forward_5", horizon=5)
    )
    assert missing_exit.frame.loc[0, "label_status"] == "missing_exit"

    bars.loc[1, "close"] = np.nan
    missing_price = compute_labels(
        decisions("2024-01-01T00:00:00Z"), bars, LabelSpec("forward_1", horizon=1)
    )
    assert missing_price.frame.loc[0, "label_status"] == "missing_entry_price"


def test_label_available_time_includes_late_exit_revision():
    bars = frame(
        ["2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"],
        closes=[100.0, 110.0],
        revision=[0, 0],
    )
    bars = pd.concat(
        [
            bars,
            frame(
                ["2024-01-03T00:00:00Z"],
                closes=[121.0],
                available=["2024-01-20T00:00:00Z"],
                revision=[1],
            ),
        ],
        ignore_index=True,
    )
    out = compute_labels(
        decisions("2024-01-01T00:00:00Z"), bars, LabelSpec("forward_1", horizon=1)
    )
    assert out.frame.loc[0, "forward_1"] == pytest.approx(0.21)
    assert out.frame.loc[0, "label_available_at"] == pd.Timestamp(
        "2024-01-20T00:00:00Z"
    )


def test_label_non_numeric_or_missing_price_is_explicit():
    bars = frame(
        ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"],
        closes=[100.0, 110.0, 121.0],
    )
    bars["close"] = bars["close"].astype(object)
    bars.loc[1, "close"] = "bad"
    out = compute_labels(
        decisions("2024-01-01T12:00:00Z"), bars, LabelSpec("forward_1", horizon=1)
    )
    assert out.frame.loc[0, "label_status"] == "invalid_price"

    with pytest.raises(ValueError):
        compute_labels(
            decisions("2024-01-01T12:00:00Z"),
            bars.drop(columns=["close"]),
            LabelSpec("forward_1", horizon=1),
        )


def test_label_adjusted_prices_and_provenance():
    bars = frame(
        ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"],
        closes=[100.0, 100.0, 55.0],
    )
    actions = ledger(
        CorporateAction(
            "split-1",
            "A",
            ActionType.SPLIT,
            "2024-01-03T00:00:00Z",
            "2024-01-01T00:00:00Z",
            ratio=2.0,
            source="fixture",
        )
    )
    decision = decisions("2024-01-01T12:00:00Z")
    raw = compute_labels(decision, bars, LabelSpec("raw", horizon=1))
    adjusted = compute_labels(
        decision,
        bars,
        LabelSpec("split", horizon=1, price_convention="split"),
        ledger=actions,
    )
    assert raw.frame.loc[0, "raw"] == pytest.approx(-0.45)
    assert adjusted.frame.loc[0, "split"] == pytest.approx(0.10)
    assert adjusted.frame.loc[0, "adjustment_available_at"] == pd.Timestamp(
        "2024-01-01T00:00:00Z"
    )
    assert "split-1" in adjusted.frame.loc[0, "adjustment_actions"]


def test_label_ignores_actions_after_exit():
    bars = frame(
        [
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z",
            "2024-01-03T00:00:00Z",
            "2024-01-04T00:00:00Z",
        ],
        closes=[100.0, 110.0, 121.0, 111.0],
    )
    actions = ledger(
        CorporateAction(
            "split-after-exit",
            "A",
            ActionType.SPLIT,
            "2024-01-04T00:00:00Z",
            "2024-01-01T00:00:00Z",
            ratio=2.0,
        )
    )
    out = compute_labels(
        decisions("2024-01-01T12:00:00Z"),
        bars,
        LabelSpec("split", horizon=1, price_convention="split"),
        ledger=actions,
    )
    assert out.frame.loc[0, "split"] == pytest.approx(0.10)
    assert out.frame.loc[0, "adjustment_actions"] == "[]"


def test_label_missing_adjustment_coverage_is_explicit():
    bars = frame(["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"])
    out = compute_labels(
        decisions("2024-01-01T12:00:00Z"),
        bars,
        LabelSpec("split", horizon=1, price_convention="split"),
        ledger=ledger(complete=False),
    )
    assert out.frame.loc[0, "label_status"] == "incomplete_adjustment_coverage"


def test_label_benchmark_excess_return_simple_and_log():
    bars = frame(
        ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"],
        closes=[100.0, 100.0, 110.0],
    )
    benchmark = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
            ],
            "available_at": [
                "2024-01-01T00:00:02Z",
                "2024-01-02T00:00:02Z",
                "2024-01-03T00:00:02Z",
            ],
            "bench": [10.0, 10.0, 11.0],
        }
    )
    decision = decisions("2024-01-01T12:00:00Z")
    simple = compute_labels(
        decision,
        bars,
        LabelSpec("excess", kind="excess_return", horizon=1, benchmark_col="bench"),
        benchmark=benchmark,
    )
    log = compute_labels(
        decision,
        bars,
        LabelSpec(
            "excess_log",
            kind="excess_return",
            horizon=1,
            benchmark_col="bench",
            return_type="log",
        ),
        benchmark=benchmark,
    )
    assert simple.frame.loc[0, "excess"] == pytest.approx(0.0)
    assert log.frame.loc[0, "excess_log"] == pytest.approx(0.0)
    assert simple.frame.loc[0, "benchmark_available_at"] == pd.Timestamp(
        "2024-01-03T00:00:02Z"
    )


def test_label_adjusted_benchmark_uses_matching_convention_and_provenance():
    stock = frame(
        ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"],
        closes=[100.0, 100.0, 55.0],
    )
    benchmark = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
            ],
            "symbol": ["SPY", "SPY", "SPY"],
            "available_at": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
            ],
            "bench": [100.0, 100.0, 55.0],
            "source": ["benchmark_fixture"] * 3,
        }
    )
    stock_ledger = ledger(
        CorporateAction(
            "stock-split",
            "A",
            ActionType.SPLIT,
            "2024-01-03T00:00:00Z",
            "2024-01-01T00:00:00Z",
            ratio=2.0,
        )
    )
    benchmark_ledger = ledger(
        CorporateAction(
            "benchmark-split",
            "SPY",
            ActionType.SPLIT,
            "2024-01-03T00:00:00Z",
            "2024-01-01T00:00:00Z",
            ratio=2.0,
        )
    )
    spec = LabelSpec(
        "excess_adjusted",
        kind="excess_return",
        horizon=1,
        price_convention="split",
        benchmark_col="bench",
        benchmark_price_convention="split",
    )
    out = compute_labels(
        decisions("2024-01-01T12:00:00Z"),
        stock,
        spec,
        benchmark=benchmark,
        ledger=stock_ledger,
        benchmark_ledger=benchmark_ledger,
    )
    row = out.frame.loc[0]
    assert row["excess_adjusted"] == pytest.approx(0.0)
    assert row["benchmark_price_convention"] == "split"
    assert "benchmark-split" in row["benchmark_adjustment_actions"]
    assert row["benchmark_adjustment_coverage"].startswith("complete=True")
    assert row["benchmark_source"] == "benchmark_fixture"
    assert out.manifest["benchmark_price_convention"] == "split"
    assert out.manifest["benchmark_sources"] == ["benchmark_fixture"]


def test_label_adjusted_benchmark_requires_ledger_and_complete_coverage():
    stock = frame(
        ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"],
        closes=[100.0, 100.0, 55.0],
    )
    benchmark = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
            ],
            "symbol": ["SPY"] * 3,
            "available_at": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
            ],
            "bench": [100.0, 100.0, 55.0],
        }
    )
    stock_ledger = ledger()
    spec = LabelSpec(
        "excess_adjusted",
        kind="excess_return",
        horizon=1,
        price_convention="split",
        benchmark_col="bench",
        benchmark_price_convention="split",
    )
    with pytest.raises(SchemaValidationError):
        compute_labels(
            decisions("2024-01-01T12:00:00Z"),
            stock,
            spec,
            benchmark=benchmark,
            ledger=stock_ledger,
        )
    out = compute_labels(
        decisions("2024-01-01T12:00:00Z"),
        stock,
        spec,
        benchmark=benchmark,
        ledger=stock_ledger,
        benchmark_ledger=ledger(complete=False),
    )
    assert out.frame.loc[0, "label_status"] == "incomplete_benchmark_adjustment_coverage"

    raw_spec = LabelSpec("raw_excess", kind="excess_return", benchmark_col="bench")
    with pytest.raises(SchemaValidationError):
        compute_labels(
            decisions("2024-01-01T12:00:00Z"),
            stock,
            raw_spec,
            benchmark=benchmark,
            benchmark_ledger=ledger(),
        )


def test_label_rejects_mismatched_stock_and_benchmark_conventions():
    with pytest.raises(ValueError):
        LabelSpec(
            "excess",
            kind="excess_return",
            price_convention="split",
            benchmark_col="bench",
            benchmark_price_convention="raw",
        )
    with pytest.raises(ValueError):
        LabelSpec(
            "excess",
            kind="excess_return",
            benchmark_col="bench",
            benchmark_price_convention="split",
        )


def _dividend_benchmark_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
            "symbol": ["SPY", "SPY"],
            "available_at": ["2024-01-01T00:00:02Z", "2024-01-02T00:00:02Z"],
            "bench": [100.0, 105.0],
            "source": ["benchmark_fixture", "benchmark_fixture"],
        }
    )


def test_label_dividend_benchmark_adjustment_matches_hand_calculation():
    stock = frame(
        ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
        closes=[100.0, 110.0],
    )
    benchmark = _dividend_benchmark_frame()
    stock_ledger = ledger(
        CorporateAction(
            "stock-dividend",
            "A",
            ActionType.CASH_DIVIDEND,
            "2024-01-02T00:00:00Z",
            "2024-01-01T00:00:00Z",
            cash_amount=5.0,
            currency="USD",
            source="fixture",
        )
    )
    benchmark_ledger = ledger(
        CorporateAction(
            "benchmark-dividend",
            "SPY",
            ActionType.CASH_DIVIDEND,
            "2024-01-02T00:00:00Z",
            "2024-01-01T00:00:00Z",
            cash_amount=5.0,
            currency="USD",
            source="benchmark_fixture",
        )
    )
    spec = LabelSpec(
        "excess_dividend",
        kind="excess_return",
        horizon=1,
        price_convention="dividend",
        benchmark_col="bench",
        benchmark_price_convention="dividend",
    )
    out = compute_labels(
        decisions("2023-12-31T12:00:00Z"),
        stock,
        spec,
        benchmark=benchmark,
        ledger=stock_ledger,
        benchmark_ledger=benchmark_ledger,
    )
    row = out.frame.loc[0]
    assert row["label_status"] == "ok"
    assert row["excess_dividend"] == pytest.approx(0.05263157894736841)
    assert row["price_convention"] == "dividend"
    assert row["benchmark_price_convention"] == "dividend"
    assert "stock-dividend" in row["adjustment_actions"]
    assert "benchmark-dividend" in row["benchmark_adjustment_actions"]
    assert row["adjustment_coverage"].startswith("complete=True")
    assert row["benchmark_adjustment_coverage"].startswith("complete=True")
    assert row["benchmark_source"] == "benchmark_fixture"
    assert row["label_available_at"] == pd.Timestamp("2024-01-02T00:00:02Z")

    with pytest.raises(SchemaValidationError):
        compute_labels(
            decisions("2023-12-31T12:00:00Z"),
            stock,
            spec,
            benchmark=benchmark,
            ledger=stock_ledger,
        )
    incomplete = compute_labels(
        decisions("2023-12-31T12:00:00Z"),
        stock,
        spec,
        benchmark=benchmark,
        ledger=stock_ledger,
        benchmark_ledger=ledger(complete=False),
    )
    assert (
        incomplete.frame.loc[0, "label_status"]
        == "incomplete_benchmark_adjustment_coverage"
    )


def test_label_total_return_benchmark_adjustment_matches_hand_calculation():
    stock = frame(
        ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
        closes=[100.0, 110.0],
    )
    benchmark = _dividend_benchmark_frame()
    stock_ledger = ledger(
        CorporateAction(
            "stock-split",
            "A",
            ActionType.SPLIT,
            "2024-01-02T00:00:00Z",
            "2024-01-01T00:00:00Z",
            ratio=2.0,
            source="fixture",
        ),
        CorporateAction(
            "stock-return-dividend",
            "A",
            ActionType.CASH_DIVIDEND,
            "2024-01-02T00:00:00Z",
            "2024-01-01T00:00:00Z",
            cash_amount=5.0,
            currency="USD",
            source="fixture",
        ),
    )
    benchmark_ledger = ledger(
        CorporateAction(
            "benchmark-split",
            "SPY",
            ActionType.SPLIT,
            "2024-01-02T00:00:00Z",
            "2024-01-01T00:00:00Z",
            ratio=2.0,
            source="benchmark_fixture",
        ),
        CorporateAction(
            "benchmark-return-dividend",
            "SPY",
            ActionType.CASH_DIVIDEND,
            "2024-01-02T00:00:00Z",
            "2024-01-01T00:00:00Z",
            cash_amount=5.0,
            currency="USD",
            source="benchmark_fixture",
        ),
    )
    spec = LabelSpec(
        "excess_total_return",
        kind="excess_return",
        horizon=1,
        price_convention="total_return",
        benchmark_col="bench",
        benchmark_price_convention="total_return",
    )
    out = compute_labels(
        decisions("2023-12-31T12:00:00Z"),
        stock,
        spec,
        benchmark=benchmark,
        ledger=stock_ledger,
        benchmark_ledger=benchmark_ledger,
    )
    row = out.frame.loc[0]
    assert row["label_status"] == "ok"
    assert row["excess_total_return"] == pytest.approx(0.10526315789473684)
    assert row["price_convention"] == "total_return"
    assert row["benchmark_price_convention"] == "total_return"
    assert "stock-split" in row["adjustment_actions"]
    assert "stock-return-dividend" in row["adjustment_actions"]
    assert "benchmark-split" in row["benchmark_adjustment_actions"]
    assert "benchmark-return-dividend" in row["benchmark_adjustment_actions"]
    assert row["adjustment_coverage"].startswith("complete=True")
    assert row["benchmark_adjustment_coverage"].startswith("complete=True")


# ---------------------------------------------------------------------------- datasets
def feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "decision_timestamp": ["2024-01-02T00:00:00Z"] * 2,
            "feature": [1.0, 2.0],
        }
    )


def label_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A", "C"],
            "decision_timestamp": ["2024-01-02T00:00:00Z"] * 2,
            "label": [0.1, 0.2],
        }
    )


def test_dataset_outer_join_is_one_row_per_key_with_status():
    out = build_dataset(
        feature_frame(), label_frame(), feature_names=["feature"], label_name="label"
    )
    assert len(out.frame) == 3
    assert out.frame["dataset_status"].tolist() == [
        "ok",
        "missing_label",
        "missing_features",
    ]
    assert out.frame.loc[out.frame["symbol"] == "B", "label_status"].iloc[0] == "missing_label"
    assert out.frame.loc[out.frame["symbol"] == "C", "label_status"].iloc[0] == "ok"


def test_dataset_rejects_feature_available_after_decision():
    features = feature_frame().assign(
        feature_available_at=["2024-01-02T01:00:00Z", "2024-01-02T01:00:00Z"]
    )
    with pytest.raises(ValueError):
        build_dataset(features, label_frame(), feature_names=["feature"], label_name="label")


def test_dataset_duplicate_keys_fail():
    features = pd.concat([feature_frame(), feature_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        build_dataset(features, label_frame(), feature_names=["feature"], label_name="label")


def test_dataset_rejects_feature_source_availability_after_decision():
    features = feature_frame().assign(
        feature_source_max_available_at=[
            "2024-01-02T01:00:00Z",
            "2024-01-02T01:00:00Z",
        ]
    )
    with pytest.raises(ValueError):
        build_dataset(features, label_frame(), feature_names=["feature"], label_name="label")


def test_dataset_rejects_malformed_or_invalid_label_metadata():
    features = feature_frame()
    malformed = label_frame().assign(
        label_start_timestamp=["2024-01-03T00:00:00Z"] * 2,
        label_end_timestamp=["2024-01-04T00:00:00Z"] * 2,
        label_available_at=["not-a-timestamp", "2024-01-04T00:00:00Z"],
    )
    with pytest.raises(ValueError):
        build_dataset(features, malformed, feature_names=["feature"], label_name="label")

    before_decision = label_frame().assign(
        label_start_timestamp=["2024-01-01T00:00:00Z"] * 2,
        label_end_timestamp=["2024-01-03T00:00:00Z"] * 2,
        label_available_at=["2024-01-03T00:00:00Z"] * 2,
    )
    with pytest.raises(ValueError):
        build_dataset(
            features, before_decision, feature_names=["feature"], label_name="label"
        )

    early_availability = label_frame().assign(
        label_start_timestamp=["2024-01-03T00:00:00Z"] * 2,
        label_end_timestamp=["2024-01-04T00:00:00Z"] * 2,
        label_available_at=["2024-01-03T00:00:00Z"] * 2,
    )
    with pytest.raises(ValueError):
        build_dataset(
            features, early_availability, feature_names=["feature"], label_name="label"
        )


def test_dataset_label_status_and_value_are_mutually_consistent():
    features = feature_frame()
    ok_missing = label_frame().assign(label=[np.nan, np.nan], label_status="ok")
    with pytest.raises(ValueError):
        build_dataset(features, ok_missing, feature_names=["feature"], label_name="label")

    non_ok_present = label_frame().assign(label_status="missing_exit")
    with pytest.raises(ValueError):
        build_dataset(
            features, non_ok_present, feature_names=["feature"], label_name="label"
        )

    missing = label_frame().assign(
        label=[np.nan, np.nan],
        label_status="missing_value",
        label_available_at=[pd.NaT, pd.NaT],
    )
    out = build_dataset(features, missing, feature_names=["feature"], label_name="label")
    row = out.frame.loc[out.frame["symbol"] == "A"].iloc[0]
    assert row["label_status"] == "missing_value"
    assert pd.isna(row["label_available_at"])


def test_dataset_rejects_feature_status_value_inconsistency():
    features = feature_frame().assign(feature=[np.nan, 2.0], feature__status="ok")
    with pytest.raises(ValueError):
        build_dataset(features, label_frame(), feature_names=["feature"], label_name="label")


def test_dataset_validation_runs_on_public_construction():
    out = build_dataset(
        feature_frame(), label_frame(), feature_names=["feature"], label_name="label"
    )
    validate_dataset(out)
    clone = ResearchDataset(
        frame=out.frame.copy(),
        feature_names=("feature",),
        label_name="label",
        quality=out.quality,
        manifest=dict(out.manifest),
    )
    validate_dataset(clone)

    invalid = out.frame.copy()
    invalid["label_status"] = "ok"
    invalid.loc[0, "label"] = np.nan
    with pytest.raises(ValueError):
        ResearchDataset(
            frame=invalid,
            feature_names=("feature",),
            label_name="label",
            quality=out.quality,
            manifest=dict(out.manifest),
        )


def test_dataset_universe_hash_tracks_availability_and_status():
    features = feature_frame()
    labels = label_frame()
    base = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "decision_timestamp": ["2024-01-02T00:00:00Z"] * 2,
            "available_at": ["2024-01-02T00:00:00Z"] * 2,
            "eligible": [True, True],
            "status": ["member", "member"],
        }
    )
    earlier = base.assign(
        available_at=["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"]
    )
    status_changed = base.assign(status=["member", "former"])
    reordered = base.iloc[::-1]

    def universe_hash(universe: pd.DataFrame) -> str:
        return build_dataset(
            features,
            labels,
            feature_names=["feature"],
            label_name="label",
            universe=universe,
        ).manifest["universe_source_hash"]

    assert universe_hash(base) != universe_hash(earlier)
    assert universe_hash(base) != universe_hash(status_changed)
    assert universe_hash(base) == universe_hash(reordered)


def test_hash_frame_rejects_duplicate_keys_and_is_order_stable():
    unique = feature_frame()
    assert hash_frame(unique) == hash_frame(unique.iloc[::-1])
    duplicate = pd.concat([unique.iloc[[0]], unique.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaValidationError):
        hash_frame(duplicate)


def test_dataset_universe_is_optional_and_not_assumed_inclusive():
    universe = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "decision_timestamp": ["2024-01-02T00:00:00Z"] * 2,
            "eligible": [True, False],
        }
    )
    complete = build_dataset(
        feature_frame(),
        label_frame(),
        feature_names=["feature"],
        label_name="label",
        universe=universe,
    )
    assert complete.frame.loc[complete.frame["symbol"] == "C", "universe_status"].iloc[0] == (
        "unknown_membership"
    )

    filtered = build_dataset(
        feature_frame(),
        label_frame(),
        feature_names=["feature"],
        label_name="label",
        universe=universe,
        filter_universe=True,
    )
    assert filtered.frame["symbol"].tolist() == ["A"]
    assert filtered.filtered_frame["symbol"].tolist() == ["B", "C"]
    assert filtered.filtered_reasons == filtered.quality["filtered_reasons"]
    assert filtered.quality["filtered_reasons"] == {
        "unknown_membership": 1,
        "not_eligible": 1,
    }


def test_dataset_universe_membership_must_be_available_at_decision():
    universe = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "decision_timestamp": ["2024-01-02T00:00:00Z"] * 2,
            "available_at": ["2024-01-03T00:00:00Z", "2024-01-02T00:00:00Z"],
            "eligible": [True, True],
        }
    )
    out = build_dataset(
        feature_frame(),
        label_frame(),
        feature_names=["feature"],
        label_name="label",
        universe=universe,
    )
    statuses = out.frame.set_index("symbol")["universe_status"].to_dict()
    assert statuses == {"A": "unknown_membership", "B": "known", "C": "unknown_membership"}


def test_dataset_unknown_eligibility_is_not_treated_as_included():
    universe = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "decision_timestamp": ["2024-01-02T00:00:00Z"] * 2,
            "eligible": [pd.NA, True],
        }
    )
    filtered = build_dataset(
        feature_frame(),
        label_frame(),
        feature_names=["feature"],
        label_name="label",
        universe=universe,
        filter_universe=True,
    )
    assert filtered.frame["symbol"].tolist() == ["B"]
    assert filtered.quality["filtered_reasons"] == {
        "unknown_membership": 1,
        "not_eligible": 1,
    }
    with pytest.raises(IncompleteUniverseError):
        build_dataset(
            feature_frame(),
            label_frame(),
            feature_names=["feature"],
            label_name="label",
            universe=universe,
            require_universe=True,
        )


def test_dataset_require_universe_fails_on_unknown_key():
    universe = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "decision_timestamp": ["2024-01-02T00:00:00Z"] * 2,
            "eligible": [True, False],
        }
    )
    with pytest.raises(IncompleteUniverseError):
        build_dataset(
            feature_frame(),
            label_frame(),
            feature_names=["feature"],
            label_name="label",
            universe=universe,
            require_universe=True,
        )


def test_dataset_column_order_is_stable_across_input_column_order():
    features = feature_frame()[["decision_timestamp", "feature", "symbol"]]
    labels = label_frame()[["label", "decision_timestamp", "symbol"]]
    first = build_dataset(
        feature_frame(), label_frame(), feature_names=["feature"], label_name="label"
    )
    reordered = build_dataset(
        features, labels, feature_names=["feature"], label_name="label"
    )
    assert first.frame.columns.tolist() == reordered.frame.columns.tolist()
    assert first.manifest["manifest_hash"] == reordered.manifest["manifest_hash"]


def test_dataset_hash_is_shuffle_and_row_order_stable():
    first = build_dataset(
        feature_frame(), label_frame(), feature_names=["feature"], label_name="label"
    )
    shuffled = build_dataset(
        feature_frame().iloc[::-1],
        label_frame().iloc[::-1],
        feature_names=["feature"],
        label_name="label",
    )
    assert first.manifest["dataset_content_hash"] == shuffled.manifest["dataset_content_hash"]
    assert first.manifest["manifest_hash"] == shuffled.manifest["manifest_hash"]
    assert first.frame["symbol"].tolist() == shuffled.frame["symbol"].tolist()


def test_hash_distinguishes_nan_null_and_literal_nan_string():
    assert hash_payload([np.nan]) != hash_payload([None])
    assert hash_payload([np.nan]) != hash_payload(["NaN"])


def test_hash_distinguishes_numeric_types_and_signed_zero():
    assert hash_payload([1]) != hash_payload([1.0])
    assert hash_payload([0.0]) != hash_payload([-0.0])
    assert hash_payload([pd.NaT]) == hash_payload([None])
    with pytest.raises(TypeError):
        hash_payload([lambda: None])


def test_hash_and_dataset_reject_infinity():
    with pytest.raises(ValueError):
        hash_payload([np.inf])
    features = feature_frame().assign(feature=[np.inf, 2.0])
    with pytest.raises(ValueError):
        build_dataset(features, label_frame(), feature_names=["feature"], label_name="label")


def test_dataset_manifest_exposes_separate_hashes():
    out = build_dataset(
        feature_frame(), label_frame(), feature_names=["feature"], label_name="label"
    )
    for key in (
        "feature_spec_hash",
        "feature_content_hash",
        "feature_source_hash",
        "label_spec_hash",
        "label_content_hash",
        "label_source_hash",
        "universe_source_hash",
        "source_hash",
        "dataset_content_hash",
        "manifest_hash",
        "dataset_id",
    ):
        assert out.manifest[key]


def test_dataset_validate_recomputes_hashes_from_current_frame():
    out = build_dataset(
        feature_frame(), label_frame(), feature_names=["feature"], label_name="label"
    )
    assert out.verify_integrity() is True
    out.frame.loc[out.frame["symbol"] == "A", "feature"] = 999.0
    with pytest.raises(SchemaValidationError, match="feature_content_hash"):
        validate_dataset(out)
    with pytest.raises(SchemaValidationError, match="feature_content_hash"):
        out.verify_integrity()

    status_changed = build_dataset(
        feature_frame(), label_frame(), feature_names=["feature"], label_name="label"
    )
    status_changed.frame.loc[
        status_changed.frame["symbol"] == "A", "dataset_status"
    ] = "label_not_ok"
    with pytest.raises(SchemaValidationError, match="dataset_content_hash"):
        validate_dataset(status_changed)


def test_dataset_rejects_missing_or_malformed_content_hash():
    out = build_dataset(
        feature_frame(), label_frame(), feature_names=["feature"], label_name="label"
    )
    manifest = dict(out.manifest)
    manifest.pop("dataset_content_hash")
    manifest.pop("dataset_id")
    manifest["manifest_hash"] = hash_payload(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    manifest["dataset_id"] = hash_payload(
        {
            "schema_version": manifest["schema_version"],
            "manifest_hash": manifest["manifest_hash"],
        }
    )
    with pytest.raises(SchemaValidationError, match="missing dataset_content_hash"):
        ResearchDataset(
            frame=out.frame.copy(),
            feature_names=("feature",),
            label_name="label",
            quality=out.quality,
            manifest=manifest,
        )

    manifest = dict(out.manifest)
    manifest["dataset_content_hash"] = ""
    manifest.pop("dataset_id")
    manifest["manifest_hash"] = hash_payload(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    manifest["dataset_id"] = hash_payload(
        {
            "schema_version": manifest["schema_version"],
            "manifest_hash": manifest["manifest_hash"],
        }
    )
    with pytest.raises(SchemaValidationError, match="non-empty string"):
        ResearchDataset(
            frame=out.frame.copy(),
            feature_names=("feature",),
            label_name="label",
            quality=out.quality,
            manifest=manifest,
        )


def test_dataset_input_frame_mutation_is_isolated_and_exposed_mutation_is_detected():
    features = feature_frame()
    labels = label_frame()
    out = build_dataset(features, labels, feature_names=["feature"], label_name="label")
    features.loc[0, "feature"] = 999.0
    labels.loc[0, "label"] = 999.0
    assert out.frame.loc[out.frame["symbol"] == "A", "feature"].iloc[0] == 1.0
    assert out.frame.loc[out.frame["symbol"] == "A", "label"].iloc[0] == 0.1
    assert out.verify_integrity() is True

    out.frame.loc[out.frame["symbol"] == "A", "feature"] = 777.0
    with pytest.raises(SchemaValidationError, match="feature_content_hash"):
        out.verify_integrity()


def test_dataset_direct_construction_copies_input_frame():
    built = build_dataset(
        feature_frame(), label_frame(), feature_names=["feature"], label_name="label"
    )
    source = built.frame
    out = ResearchDataset(
        frame=source,
        feature_names=("feature",),
        label_name="label",
        quality=built.quality,
        manifest=dict(built.manifest),
    )
    source.loc[0, "feature"] = 999.0
    assert out.frame.loc[0, "feature"] == 1.0


def _resign_manifest(manifest):
    resigned = dict(manifest)
    resigned["manifest_hash"] = hash_payload(
        {
            key: value
            for key, value in resigned.items()
            if key not in {"manifest_hash", "dataset_id"}
        }
    )
    resigned["dataset_id"] = hash_payload(
        {
            "schema_version": resigned["schema_version"],
            "manifest_hash": resigned["manifest_hash"],
        }
    )
    return resigned


@pytest.mark.parametrize(
    "removed",
    [
        ("dataset_id",),
        ("manifest_hash",),
        ("manifest_hash", "dataset_id"),
    ],
)
def test_dataset_rejects_missing_root_hashes(removed):
    out = build_dataset(
        feature_frame(), label_frame(), feature_names=["feature"], label_name="label"
    )
    manifest = dict(out.manifest)
    for key in removed:
        manifest.pop(key)
    with pytest.raises(SchemaValidationError, match="is missing"):
        ResearchDataset(
            frame=out.frame.copy(),
            feature_names=("feature",),
            label_name="label",
            quality=out.quality,
            manifest=manifest,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("manifest_hash", None, "non-empty string"),
        ("dataset_id", "", "non-empty string"),
        ("manifest_hash", "0" * 64, "manifest_hash does not match"),
        ("dataset_id", "0" * 64, "dataset_id does not match"),
    ],
)
def test_dataset_rejects_malformed_or_inconsistent_root_hashes(field, value, match):
    out = build_dataset(
        feature_frame(), label_frame(), feature_names=["feature"], label_name="label"
    )
    manifest = dict(out.manifest)
    manifest[field] = value
    with pytest.raises(SchemaValidationError, match=match):
        ResearchDataset(
            frame=out.frame.copy(),
            feature_names=("feature",),
            label_name="label",
            quality=out.quality,
            manifest=manifest,
        )


def test_dataset_metadata_tampering_cannot_bypass_root_hashes():
    out = build_dataset(
        feature_frame(), label_frame(), feature_names=["feature"], label_name="label"
    )
    tampered_quality = {**out.quality, "tampered": True}

    manifest = dict(out.manifest)
    manifest["quality"] = tampered_quality
    with pytest.raises(SchemaValidationError, match="manifest_hash"):
        ResearchDataset(
            frame=out.frame.copy(),
            feature_names=("feature",),
            label_name="label",
            quality=tampered_quality,
            manifest=manifest,
        )

    manifest = dict(out.manifest)
    manifest.pop("manifest_hash")
    manifest.pop("dataset_id")
    manifest["quality"] = tampered_quality
    with pytest.raises(SchemaValidationError, match="is missing"):
        ResearchDataset(
            frame=out.frame.copy(),
            feature_names=("feature",),
            label_name="label",
            quality=tampered_quality,
            manifest=manifest,
        )


@pytest.mark.parametrize(
    "manifest",
    [None, [], "", 0, (), ["manifest"], 1.5],
)
def test_dataset_rejects_non_mapping_manifest(manifest):
    out = build_dataset(
        feature_frame(), label_frame(), feature_names=["feature"], label_name="label"
    )
    with pytest.raises(SchemaValidationError, match="must be a mapping"):
        ResearchDataset(
            frame=out.frame.copy(),
            feature_names=("feature",),
            label_name="label",
            quality=out.quality,
            manifest=manifest,
        )


def test_dataset_rejects_empty_or_incomplete_manifest():
    out = build_dataset(
        feature_frame(), label_frame(), feature_names=["feature"], label_name="label"
    )
    with pytest.raises(SchemaValidationError, match="must not be empty"):
        ResearchDataset(
            frame=out.frame.copy(),
            feature_names=("feature",),
            label_name="label",
            quality=out.quality,
            manifest={},
        )

    manifest = dict(out.manifest)
    manifest.pop("quality")
    manifest = _resign_manifest(manifest)
    with pytest.raises(SchemaValidationError, match="missing quality"):
        ResearchDataset(
            frame=out.frame.copy(),
            feature_names=("feature",),
            label_name="label",
            quality=out.quality,
            manifest=manifest,
        )

    manifest = dict(out.manifest)
    manifest["quality"] = []
    manifest = _resign_manifest(manifest)
    with pytest.raises(SchemaValidationError, match="quality must be a mapping"):
        ResearchDataset(
            frame=out.frame.copy(),
            feature_names=("feature",),
            label_name="label",
            quality=out.quality,
            manifest=manifest,
        )


def test_dataset_post_construction_manifest_mutation_is_detected():
    out = build_dataset(
        feature_frame(), label_frame(), feature_names=["feature"], label_name="label"
    )
    out.manifest.pop("dataset_id")
    with pytest.raises(SchemaValidationError, match="missing dataset_id"):
        out.verify_integrity()

    mutated_type = build_dataset(
        feature_frame(), label_frame(), feature_names=["feature"], label_name="label"
    )
    object.__setattr__(mutated_type, "manifest", [])
    with pytest.raises(SchemaValidationError, match="must be a mapping"):
        mutated_type.verify_integrity()


def test_dataset_quality_must_match_manifest():
    out = build_dataset(
        feature_frame(), label_frame(), feature_names=["feature"], label_name="label"
    )
    assert out.verify_integrity() is True

    with pytest.raises(SchemaValidationError, match="quality does not match"):
        ResearchDataset(
            frame=out.frame.copy(),
            feature_names=("feature",),
            label_name="label",
            quality={**out.quality, "tampered": True},
            manifest=dict(out.manifest),
        )

    with pytest.raises(SchemaValidationError, match="must be a mapping"):
        ResearchDataset(
            frame=out.frame.copy(),
            feature_names=("feature",),
            label_name="label",
            quality=[],
            manifest=dict(out.manifest),
        )

    out.quality["tampered"] = True
    with pytest.raises(SchemaValidationError):
        out.verify_integrity()


# -------------------------------------------------------------------------- time split
def test_time_split_purges_embargo_and_reports_post_test():
    samples = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    ends = samples + pd.Timedelta(days=1)
    result = purged_time_split(
        sample_times=samples,
        label_start_times=samples,
        label_end_times=ends,
        label_available_times=ends,
        test_start="2024-01-06T00:00:00Z",
        test_end="2024-01-08T00:00:00Z",
        embargo="1D",
    )
    assert result.test.tolist() == [5, 6]
    assert result.train.tolist() == [0, 1, 2]
    assert result.excluded.tolist() == [3, 4, 7, 8, 9]
    assert result.excluded_reasons[0] == "embargo_overlap"
    assert result.excluded_reasons[1] == "label_overlaps_test"
    assert result.excluded_reasons[-1] == "post_test_not_used"


def test_time_split_boundary_equality_is_conservative():
    samples = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    ends = samples + pd.Timedelta(days=1)
    result = purged_time_split(
        sample_times=samples,
        label_start_times=samples,
        label_end_times=ends,
        label_available_times=ends,
        test_start="2024-01-04T00:00:00Z",
        test_end="2024-01-05T00:00:00Z",
    )
    assert result.test.tolist() == [3]
    assert 2 not in result.train.tolist()
    assert result.excluded_reasons[0] == "label_overlaps_test"


def test_time_split_empty_test_raises():
    samples = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    with pytest.raises(ValueError):
        purged_time_split(
            sample_times=samples,
            label_start_times=samples,
            label_end_times=samples,
            label_available_times=samples,
            test_start="2024-01-10T00:00:00Z",
            test_end="2024-01-11T00:00:00Z",
        )


def test_time_split_sorts_indices_chronologically_not_by_input_order():
    samples = pd.to_datetime(
        [
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z",
            "2024-01-02T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "2024-01-03T00:00:00Z",
            "2024-01-03T00:00:00Z",
        ],
        utc=True,
    )
    result = purged_time_split(
        sample_times=samples,
        label_start_times=samples,
        label_end_times=samples,
        label_available_times=samples,
        test_start="2024-01-03T00:00:00Z",
        test_end="2024-01-04T00:00:00Z",
    )
    assert result.train.tolist() == [0, 3, 1, 2]
    assert result.test.tolist() == [4, 5]
    assert result.excluded.tolist() == []


def test_time_split_rejects_label_start_before_sample():
    samples = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    label_start = samples - pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        purged_time_split(
            sample_times=samples,
            label_start_times=label_start,
            label_end_times=samples,
            label_available_times=samples,
            test_start="2024-01-03T00:00:00Z",
            test_end="2024-01-04T00:00:00Z",
        )


def test_time_split_rejects_invalid_label_interval():
    samples = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    with pytest.raises(ValueError):
        purged_time_split(
            sample_times=samples,
            label_start_times=samples + pd.Timedelta(days=2),
            label_end_times=samples,
            label_available_times=samples + pd.Timedelta(days=3),
            test_start="2024-01-03T00:00:00Z",
            test_end="2024-01-04T00:00:00Z",
        )
