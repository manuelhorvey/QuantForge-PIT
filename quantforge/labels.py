"""Forward-outcome labels with explicit availability and execution semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ._research import (
    latest_availability_map,
    normalize_bars,
    normalize_decisions,
    resolve_event_revisions,
)
from .data.corporate_actions import (
    ActionType,
    CorporateAction,
    CorporateActionLedger,
    pit_adjusted_prices,
)
from .data.errors import (
    CorporateActionError,
    IncompleteCorporateActionDataError,
    SchemaValidationError,
)

LABEL_EXECUTION_CONVENTION = "next_observed_bar_close_to_close"
BACKTEST_COMPATIBILITY = "not_claimed"
LABEL_METADATA_COLUMNS = (
    "label_status",
    "label_reason",
    "label_start_timestamp",
    "label_end_timestamp",
    "label_available_at",
    "entry_available_at",
    "exit_available_at",
    "benchmark_available_at",
    "adjustment_available_at",
    "price_convention",
    "return_type",
    "label_execution_convention",
    "backtest_compatibility",
    "adjustment_coverage",
    "adjustment_actions",
    "benchmark_price_convention",
    "benchmark_adjustment_coverage",
    "benchmark_adjustment_actions",
    "benchmark_source",
)


@dataclass(frozen=True)
class LabelSpec:
    name: str
    kind: str = "forward_return"
    horizon: int = 1
    price_col: str = "close"
    price_convention: str = "raw"
    return_type: str = "simple"
    benchmark_col: str | None = None
    benchmark_price_convention: str = "raw"
    version: str = "1"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("label name is required")
        if self.name in LABEL_METADATA_COLUMNS:
            raise ValueError("label name collides with label metadata columns")
        if self.kind not in {
            "forward_return",
            "excess_return",
            "simple_return",
            "log_return",
        }:
            raise ValueError("unsupported label kind")
        if (
            not isinstance(self.horizon, Integral)
            or isinstance(self.horizon, bool)
            or self.horizon < 1
        ):
            raise ValueError("horizon must be a positive integer")
        if not isinstance(self.price_col, str) or not self.price_col:
            raise ValueError("price_col must be a non-empty string")
        if self.price_convention not in {"raw", "split", "dividend", "total_return"}:
            raise ValueError("unsupported price convention")
        if self.return_type not in {"simple", "log"}:
            raise ValueError("return_type must be 'simple' or 'log'")
        if self.kind == "excess_return" and self.benchmark_col is None:
            raise ValueError("excess_return labels require benchmark_col")
        if self.benchmark_col is not None and (
            not isinstance(self.benchmark_col, str) or not self.benchmark_col
        ):
            raise ValueError("benchmark_col must be a non-empty string")
        if self.benchmark_col is not None and self.kind != "excess_return":
            raise ValueError("benchmark_col is only valid for excess_return labels")
        if self.benchmark_price_convention not in {
            "raw",
            "split",
            "dividend",
            "total_return",
        }:
            raise ValueError("unsupported benchmark price convention")
        if self.benchmark_col is None and self.benchmark_price_convention != "raw":
            raise ValueError("benchmark_price_convention requires benchmark_col")
        if (
            self.benchmark_col is not None
            and self.benchmark_price_convention != self.price_convention
        ):
            raise ValueError(
                "benchmark_price_convention must match price_convention for excess returns"
            )
        if not self.version:
            raise ValueError("version is required")

    @property
    def effective_return_type(self) -> str:
        if self.kind in {"log_return"}:
            return "log"
        if self.kind == "simple_return":
            return "simple"
        return self.return_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "horizon": self.horizon,
            "price_col": self.price_col,
            "price_convention": self.price_convention,
            "return_type": self.effective_return_type,
            "benchmark_col": self.benchmark_col,
            "benchmark_price_convention": self.benchmark_price_convention,
            "version": self.version,
            "execution_convention": LABEL_EXECUTION_CONVENTION,
            "backtest_compatibility": BACKTEST_COMPATIBILITY,
        }


@dataclass(frozen=True)
class LabelResult:
    frame: pd.DataFrame
    label_name: str
    quality: Mapping[str, Any] = field(default_factory=dict)
    manifest: Mapping[str, Any] = field(default_factory=dict)


def _split_like(action_type: ActionType) -> bool:
    return action_type in {ActionType.SPLIT, ActionType.REVERSE_SPLIT}


def _action_matches(action: CorporateAction, convention: str) -> bool:
    if convention == "split":
        return _split_like(action.action_type)
    if convention == "dividend":
        return action.action_type is ActionType.CASH_DIVIDEND
    if convention == "total_return":
        return _split_like(action.action_type) or action.action_type is ActionType.CASH_DIVIDEND
    return False


def _action_payload(action: CorporateAction) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "instrument_id": action.instrument_id,
        "action_type": action.action_type.value,
        "effective_timestamp": action.effective_timestamp.isoformat(),
        "available_at": action.available_at.isoformat(),
        "revision": action.revision,
        "source": action.source,
        "ratio": action.ratio,
        "cash_amount": action.cash_amount,
        "currency": action.currency,
    }


def _coverage_text(
    ledger: CorporateActionLedger,
    instrument: str,
) -> str:
    coverage = ledger.coverage_by_instrument.get(instrument, ledger.coverage)
    if coverage is None:
        return "unknown"
    return (
        f"complete={coverage.complete};start="
        f"{coverage.start.isoformat() if coverage.start is not None else 'none'};end="
        f"{coverage.end.isoformat() if coverage.end is not None else 'none'}"
    )


def _normalize_benchmark(
    benchmark: pd.DataFrame | None,
    *,
    spec: LabelSpec,
    frequency: str | None,
    benchmark_symbol: str | None,
    instrument_col: str,
) -> pd.DataFrame:
    if benchmark is None:
        raise SchemaValidationError("benchmark data is required for excess_return labels")
    work = benchmark.copy()
    if spec.benchmark_col not in work.columns:
        raise SchemaValidationError(f"Missing benchmark column {spec.benchmark_col!r}")
    symbol_column = instrument_col if instrument_col in work.columns else "symbol"
    if symbol_column in work.columns:
        if benchmark_symbol is not None:
            work = work.loc[
                work[symbol_column].astype(str).eq(str(benchmark_symbol))
            ].copy()
        else:
            symbols = sorted(work[symbol_column].dropna().astype(str).unique())
            if len(symbols) > 1:
                raise SchemaValidationError(
                    "benchmark frame has multiple symbols; pass benchmark_symbol"
                )
    if work.empty:
        raise SchemaValidationError("benchmark frame is empty")
    work["__qf_benchmark"] = "_benchmark"
    work = normalize_bars(
        work,
        instrument_col="__qf_benchmark",
        event_col="timestamp",
        availability_col="available_at",
        frequency=frequency,
        context="compute_labels benchmark",
    )
    numeric = pd.to_numeric(work[spec.benchmark_col], errors="coerce")
    if bool((numeric.isna() & work[spec.benchmark_col].notna()).any()):
        raise SchemaValidationError("benchmark values must be numeric")
    if bool(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any()):
        raise SchemaValidationError("benchmark values contain infinity")
    work[spec.benchmark_col] = numeric
    return work


def _price_at_event(frame: pd.DataFrame, event: pd.Timestamp, column: str) -> Any:
    selected = frame.loc[frame["timestamp"].eq(event)]
    if selected.empty:
        return None
    return selected.iloc[-1][column]


def _price_issue(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return "missing"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "non_numeric"
    if not np.isfinite(numeric):
        return "non_finite"
    if numeric <= 0:
        return "non_positive"
    return None


def _label_value(entry: float, exit_: float, return_type: str) -> float:
    if return_type == "log":
        return float(np.log(exit_ / entry))
    return float(exit_ / entry - 1.0)


def _benchmark_value(
    benchmark: pd.DataFrame,
    *,
    entry_event: pd.Timestamp,
    exit_event: pd.Timestamp,
    column: str,
    label_available: pd.Timestamp,
) -> tuple[float | None, str | None]:
    eligible = benchmark.loc[
        (benchmark["timestamp"] <= exit_event)
        & (benchmark["available_at"] <= label_available)
    ]
    if eligible.empty:
        return None, "missing_benchmark"
    resolved = resolve_event_revisions(
        eligible,
        instrument_col="__qf_benchmark",
        event_col="timestamp",
        availability_col="available_at",
    )
    entry = _price_at_event(resolved, entry_event, column)
    exit_ = _price_at_event(resolved, exit_event, column)
    if _price_issue(entry) == "missing" or _price_issue(exit_) == "missing":
        return None, "missing_benchmark"
    issue = _price_issue(entry) or _price_issue(exit_)
    if issue is not None:
        return None, "invalid_benchmark"
    return float(exit_) / float(entry) - 1.0, None


def _source_text(frame: pd.DataFrame | None) -> str:
    if frame is None or "source" not in frame.columns:
        return ""
    return ",".join(sorted({str(value) for value in frame["source"].dropna()}))


def _benchmark_instrument_id(
    benchmark: pd.DataFrame | None,
    *,
    instrument_col: str,
    benchmark_symbol: str | None,
) -> str:
    if benchmark_symbol is not None:
        return str(benchmark_symbol)
    if benchmark is not None:
        symbol_column = instrument_col if instrument_col in benchmark.columns else "symbol"
        if symbol_column in benchmark.columns:
            symbols = sorted(
                {
                    str(value)
                    for value in benchmark[symbol_column].dropna()
                }
            )
            if len(symbols) == 1:
                return symbols[0]
    return "_benchmark"


def _actions_for_interval(
    ledger: CorporateActionLedger,
    *,
    instrument: str,
    convention: str,
    entry_event: pd.Timestamp,
    exit_event: pd.Timestamp,
) -> list[CorporateAction]:
    return sorted(
        (
            action
            for action in ledger.all_latest()
            if action.instrument_id == instrument
            and _action_matches(action, convention)
            and entry_event < action.effective_timestamp <= exit_event
        ),
        key=lambda action: (
            action.effective_timestamp,
            action.action_id,
            action.revision,
        ),
    )


def _benchmark_adjusted_value(
    benchmark: pd.DataFrame,
    *,
    entry_event: pd.Timestamp,
    exit_event: pd.Timestamp,
    column: str,
    label_available: pd.Timestamp,
    ledger: CorporateActionLedger,
    convention: str,
    instrument: str,
    actions: list[CorporateAction],
) -> tuple[float | None, str | None]:
    price_window = benchmark.loc[
        (benchmark["timestamp"] >= entry_event)
        & (benchmark["timestamp"] <= exit_event)
    ]
    if price_window.empty:
        return None, "missing_benchmark"
    coverage = ledger.coverage_by_instrument.get(instrument, ledger.coverage)
    label_ledger = CorporateActionLedger(
        actions,
        coverage=coverage,
        coverage_by_instrument=({instrument: coverage} if coverage is not None else {}),
    )
    try:
        result = pit_adjusted_prices(
            price_window,
            label_ledger,
            label_available,
            convention=convention,
            instrument_col="__qf_benchmark",
            event_col="timestamp",
            price_cols=(column,),
            close_col=column,
            instrument_resolver=lambda _: instrument,
            require_complete=True,
        )
    except IncompleteCorporateActionDataError:
        return None, "incomplete_benchmark_adjustment_coverage"
    except CorporateActionError:
        return None, "benchmark_adjustment_error"
    adjusted = result.frame
    adjusted_column = result.adjusted_columns.get(column, column)
    entry = _price_at_event(adjusted, entry_event, adjusted_column)
    exit_ = _price_at_event(adjusted, exit_event, adjusted_column)
    if _price_issue(entry) == "missing" or _price_issue(exit_) == "missing":
        return None, "missing_benchmark"
    issue = _price_issue(entry) or _price_issue(exit_)
    if issue is not None:
        return None, "invalid_benchmark"
    return float(exit_) / float(entry) - 1.0, None


def compute_labels(
    decisions: pd.DataFrame,
    bars: pd.DataFrame,
    spec: LabelSpec,
    *,
    benchmark: pd.DataFrame | None = None,
    benchmark_bars: pd.DataFrame | None = None,
    benchmark_symbol: str | None = None,
    ledger: CorporateActionLedger | None = None,
    benchmark_ledger: CorporateActionLedger | None = None,
    frequency: str | None = None,
    instrument_col: str = "symbol",
    decision_col: str = "decision_timestamp",
) -> LabelResult:
    """Compute one label per decision using the frozen next-observed-bar convention.

    Entry is the first observed bar strictly after the decision; exit is
    ``horizon`` observed bars after entry. The label is computed as
    ``price[exit] / price[entry] - 1`` (or its log form). Current backtest
    execution is intentionally not claimed to be numerically equivalent.

    Adjusted excess-return labels require a matching benchmark price
    convention and a separate ``benchmark_ledger``. Raw benchmark and stock
    conventions are the default and require no benchmark ledger.
    """

    if not isinstance(spec, LabelSpec):
        raise SchemaValidationError("spec must be a LabelSpec")
    if benchmark is not None and benchmark_bars is not None:
        raise SchemaValidationError("Pass only one of benchmark or benchmark_bars")
    benchmark_frame = benchmark if benchmark is not None else benchmark_bars
    if spec.name in {instrument_col, decision_col}:
        raise SchemaValidationError("Label name cannot duplicate a key column")
    if benchmark_ledger is not None and spec.benchmark_col is None:
        raise SchemaValidationError("benchmark_ledger requires benchmark_col")
    if benchmark_ledger is not None and spec.benchmark_price_convention == "raw":
        raise SchemaValidationError(
            "benchmark_ledger is only valid for adjusted benchmark prices"
        )
    if spec.benchmark_col is not None and spec.benchmark_price_convention != "raw":
        if benchmark_ledger is None:
            raise SchemaValidationError(
                "A benchmark_ledger is required for adjusted benchmark prices"
            )
    if spec.price_convention != "raw" and ledger is None:
        raise SchemaValidationError(
            "A CorporateActionLedger is required for adjusted price labels"
        )
    decision_frame = normalize_decisions(
        decisions, instrument_col=instrument_col, decision_col=decision_col
    )
    bar_frame = normalize_bars(
        bars,
        instrument_col=instrument_col,
        event_col="timestamp",
        availability_col="available_at",
        frequency=frequency,
        context="compute_labels",
    )
    if spec.price_col not in bar_frame.columns:
        raise SchemaValidationError(f"Missing label price column {spec.price_col!r}")
    benchmark_normalized = (
        _normalize_benchmark(
            benchmark_frame,
            spec=spec,
            frequency=frequency,
            benchmark_symbol=benchmark_symbol,
            instrument_col=instrument_col,
        )
        if spec.benchmark_col is not None
        else None
    )
    benchmark_instrument = _benchmark_instrument_id(
        benchmark_frame,
        instrument_col=instrument_col,
        benchmark_symbol=benchmark_symbol,
    )
    benchmark_source_text = _source_text(benchmark_normalized)

    bar_groups = {
        key: group for key, group in bar_frame.groupby(instrument_col, sort=False, dropna=False)
    }
    empty_bars = bar_frame.iloc[0:0].copy()
    empty_benchmark = (
        benchmark_normalized.iloc[0:0].copy()
        if benchmark_normalized is not None
        else pd.DataFrame()
    )
    rows: list[dict[str, Any]] = []

    for _, decision_row in decision_frame.iterrows():
        instrument = decision_row[instrument_col]
        decision_time = decision_row[decision_col]
        instrument_bars = bar_groups.get(instrument, empty_bars)
        availability_by_event = latest_availability_map(
            instrument_bars,
            instrument_col=instrument_col,
            event_col="timestamp",
            availability_col="available_at",
        )
        event_times = sorted(
            {pd.Timestamp(value) for value in instrument_bars["timestamp"].dropna()}
        )
        entry_event: pd.Timestamp | None = None
        exit_event: pd.Timestamp | None = None
        for event in event_times:
            if event > decision_time:
                entry_event = event
                break
        if entry_event is not None:
            entry_index = event_times.index(entry_event)
            exit_index = entry_index + spec.horizon
            if exit_index < len(event_times):
                exit_event = event_times[exit_index]

        label_start = entry_event
        label_end = exit_event
        entry_available = availability_by_event.get((instrument, entry_event)) if entry_event else None
        exit_available = availability_by_event.get((instrument, exit_event)) if exit_event else None
        benchmark_available = None
        benchmark_adjustment_available = None
        benchmark_adjustment_actions: list[CorporateAction] = []
        benchmark_coverage_text = ""
        adjustment_available = None
        adjustment_actions: list[CorporateAction] = []
        coverage_text = ""
        status = "ok"
        reason = ""
        value = float("nan")
        entry_price: float | None = None
        exit_price: float | None = None
        benchmark_return: float | None = None

        if entry_event is None:
            status = "missing_entry"
            reason = "no observed bar after decision"
        elif exit_event is None:
            status = "missing_exit"
            reason = f"fewer than {spec.horizon} observed bars after entry"

        benchmark_frame_for_label = empty_benchmark
        if (
            status == "ok"
            and benchmark_normalized is not None
            and spec.benchmark_col is not None
        ):
            benchmark_frame_for_label = benchmark_normalized
            benchmark_availability = latest_availability_map(
                benchmark_frame_for_label,
                instrument_col="__qf_benchmark",
                event_col="timestamp",
                availability_col="available_at",
            )
            benchmark_available = benchmark_availability.get(
                ("_benchmark", entry_event)
            ) if entry_event else None
            exit_benchmark_available = benchmark_availability.get(
                ("_benchmark", exit_event)
            ) if exit_event else None
            benchmark_available = max(
                [t for t in (benchmark_available, exit_benchmark_available) if t is not None],
                default=None,
            )
            if benchmark_available is None:
                status = "missing_benchmark"
                reason = "benchmark has no observation at entry or exit"

        if (
            status == "ok"
            and benchmark_normalized is not None
            and spec.benchmark_price_convention != "raw"
        ):
            assert benchmark_ledger is not None
            assert entry_event is not None and exit_event is not None
            benchmark_adjustment_actions = _actions_for_interval(
                benchmark_ledger,
                instrument=benchmark_instrument,
                convention=spec.benchmark_price_convention,
                entry_event=entry_event,
                exit_event=exit_event,
            )
            benchmark_adjustment_available = max(
                [action.available_at for action in benchmark_adjustment_actions],
                default=None,
            )
            benchmark_coverage_text = _coverage_text(
                benchmark_ledger, benchmark_instrument
            )
            try:
                benchmark_ledger.check_coverage(
                    benchmark_instrument,
                    start=entry_event,
                    end=exit_event,
                    required=True,
                )
            except IncompleteCorporateActionDataError as exc:
                status = "incomplete_benchmark_adjustment_coverage"
                reason = str(exc)

        if status == "ok" and spec.price_convention != "raw":
            assert ledger is not None
            assert entry_event is not None and exit_event is not None
            actions = sorted(
                (
                    action
                    for action in ledger.all_latest()
                    if action.instrument_id == str(instrument)
                    and _action_matches(action, spec.price_convention)
                    and entry_event < action.effective_timestamp <= exit_event
                ),
                key=lambda action: (
                    action.effective_timestamp,
                    action.action_id,
                    action.revision,
                ),
            )
            adjustment_actions = actions
            adjustment_available = max(
                [action.available_at for action in actions], default=None
            )
            coverage_text = _coverage_text(ledger, str(instrument))
            try:
                ledger.check_coverage(
                    str(instrument), start=entry_event, end=exit_event, required=True
                )
            except IncompleteCorporateActionDataError as exc:
                status = "incomplete_adjustment_coverage"
                reason = str(exc)

        available_values = [entry_available, exit_available]
        if benchmark_available is not None:
            available_values.append(benchmark_available)
        if benchmark_adjustment_available is not None:
            available_values.append(benchmark_adjustment_available)
        if adjustment_available is not None:
            available_values.append(adjustment_available)
        label_available = max(
            [value for value in available_values if value is not None], default=None
        )

        if status == "ok":
            assert entry_event is not None and exit_event is not None and label_available is not None
            asof = instrument_bars.loc[
                (instrument_bars["timestamp"] <= exit_event)
                & (instrument_bars["available_at"] <= label_available)
            ]
            try:
                resolved = resolve_event_revisions(
                    asof,
                    instrument_col=instrument_col,
                    event_col="timestamp",
                    availability_col="available_at",
                )
            except Exception as exc:
                status = "adjustment_error" if spec.price_convention != "raw" else "invalid_data"
                reason = str(exc)
                resolved = asof.iloc[0:0].copy()

            if status == "ok":
                price_window = resolved.loc[
                    (resolved["timestamp"] >= entry_event)
                    & (resolved["timestamp"] <= exit_event)
                ]
                if spec.price_convention == "raw":
                    adjusted = price_window
                    price_column = spec.price_col
                else:
                    assert ledger is not None
                    coverage = ledger.coverage_by_instrument.get(
                        str(instrument), ledger.coverage
                    )
                    label_ledger = CorporateActionLedger(
                        actions,
                        coverage=coverage,
                        coverage_by_instrument=(
                            {str(instrument): coverage} if coverage is not None else {}
                        ),
                    )
                    try:
                        result = pit_adjusted_prices(
                            price_window,
                            label_ledger,
                            label_available,
                            convention=spec.price_convention,
                            instrument_col=instrument_col,
                            event_col="timestamp",
                            price_cols=(spec.price_col,),
                            close_col=spec.price_col,
                            require_complete=True,
                        )
                        adjusted = result.frame
                        price_column = result.adjusted_columns.get(
                            spec.price_col, spec.price_col
                        )
                    except IncompleteCorporateActionDataError as exc:
                        status = "incomplete_adjustment_coverage"
                        reason = str(exc)
                        adjusted = price_window
                        price_column = spec.price_col
                    except CorporateActionError as exc:
                        status = "adjustment_error"
                        reason = str(exc)
                        adjusted = price_window
                        price_column = spec.price_col

                if status == "ok":
                    entry_price = _price_at_event(adjusted, entry_event, price_column)
                    exit_price = _price_at_event(adjusted, exit_event, price_column)
                    if entry_price is None:
                        status = "missing_entry_price"
                        reason = "entry bar has no usable price"
                    elif exit_price is None:
                        status = "missing_exit_price"
                        reason = "exit bar has no usable price"
                    else:
                        entry_issue = _price_issue(entry_price)
                        exit_issue = _price_issue(exit_price)
                        if entry_issue == "missing":
                            status = "missing_entry_price"
                            reason = "entry price is null"
                        elif exit_issue == "missing":
                            status = "missing_exit_price"
                            reason = "exit price is null"
                        elif entry_issue is not None or exit_issue is not None:
                            status = "invalid_price"
                            reason = "entry or exit price is non-finite/non-positive"

                if status == "ok" and benchmark_normalized is not None:
                    if spec.benchmark_price_convention == "raw":
                        benchmark_return, benchmark_issue = _benchmark_value(
                            benchmark_normalized,
                            entry_event=entry_event,
                            exit_event=exit_event,
                            column=spec.benchmark_col or "",
                            label_available=label_available,
                        )
                    else:
                        assert benchmark_ledger is not None
                        benchmark_return, benchmark_issue = _benchmark_adjusted_value(
                            benchmark_normalized,
                            entry_event=entry_event,
                            exit_event=exit_event,
                            column=spec.benchmark_col or "",
                            label_available=label_available,
                            ledger=benchmark_ledger,
                            convention=spec.benchmark_price_convention,
                            instrument=benchmark_instrument,
                            actions=benchmark_adjustment_actions,
                        )
                    if benchmark_issue is not None:
                        status = benchmark_issue
                        reason = "benchmark entry/exit values are unavailable or invalid"

                if status == "ok":
                    assert entry_price is not None and exit_price is not None
                    raw_return = _label_value(
                        float(entry_price), float(exit_price), spec.effective_return_type
                    )
                    if benchmark_return is None:
                        value = raw_return
                    elif spec.effective_return_type == "log":
                        value = raw_return - float(np.log1p(benchmark_return))
                    else:
                        value = raw_return - benchmark_return

        if entry_price is not None and _price_issue(entry_price) is not None:
            entry_price = None
        if exit_price is not None and _price_issue(exit_price) is not None:
            exit_price = None
        action_payload = json.dumps(
            [_action_payload(action) for action in adjustment_actions],
            sort_keys=True,
            separators=(",", ":"),
        )
        benchmark_action_payload = json.dumps(
            [_action_payload(action) for action in benchmark_adjustment_actions],
            sort_keys=True,
            separators=(",", ":"),
        )
        rows.append(
            {
                instrument_col: instrument,
                decision_col: decision_time,
                spec.name: value,
                "label_status": status,
                "label_reason": reason,
                "label_start_timestamp": label_start,
                "label_end_timestamp": label_end,
                "label_available_at": label_available,
                "entry_available_at": entry_available,
                "exit_available_at": exit_available,
                "benchmark_available_at": benchmark_available,
                "adjustment_available_at": adjustment_available,
                "price_convention": spec.price_convention,
                "return_type": spec.effective_return_type,
                "label_execution_convention": LABEL_EXECUTION_CONVENTION,
                "backtest_compatibility": BACKTEST_COMPATIBILITY,
                "adjustment_coverage": coverage_text,
                "adjustment_actions": action_payload,
                "benchmark_price_convention": (
                    spec.benchmark_price_convention
                    if spec.benchmark_col is not None
                    else ""
                ),
                "benchmark_adjustment_coverage": benchmark_coverage_text,
                "benchmark_adjustment_actions": benchmark_action_payload,
                "benchmark_source": (
                    benchmark_source_text if spec.benchmark_col is not None else ""
                ),
            }
        )

    columns = [
        instrument_col,
        decision_col,
        spec.name,
        "label_status",
        "label_reason",
        "label_start_timestamp",
        "label_end_timestamp",
        "label_available_at",
        "entry_available_at",
        "exit_available_at",
        "benchmark_available_at",
        "adjustment_available_at",
        "price_convention",
        "return_type",
        "label_execution_convention",
        "backtest_compatibility",
        "adjustment_coverage",
        "adjustment_actions",
        "benchmark_price_convention",
        "benchmark_adjustment_coverage",
        "benchmark_adjustment_actions",
        "benchmark_source",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.sort_values([instrument_col, decision_col], kind="stable").reset_index(
            drop=True
        )
    status_counts = (
        {str(k): int(v) for k, v in frame["label_status"].value_counts().items()}
        if not frame.empty
        else {}
    )
    benchmark_sources = (
        sorted(
            {
                str(value)
                for value in benchmark_normalized["source"].dropna()
            }
        )
        if benchmark_normalized is not None and "source" in benchmark_normalized.columns
        else []
    )
    quality = {
        "rows": int(len(frame)),
        "status_counts": status_counts,
        "horizon": spec.horizon,
        "benchmark_sources": benchmark_sources,
    }
    manifest = {
        "schema_version": "quantforge.labels.v1",
        "label_name": spec.name,
        "spec": spec.to_dict(),
        "instrument_column": instrument_col,
        "decision_column": decision_col,
        "horizon_unit": "observed_bars",
        "frequency": frequency,
        "execution_convention": LABEL_EXECUTION_CONVENTION,
        "backtest_compatibility": BACKTEST_COMPATIBILITY,
        "benchmark_price_convention": (
            spec.benchmark_price_convention if spec.benchmark_col is not None else None
        ),
        "benchmark_sources": benchmark_sources,
        "rows": int(len(frame)),
        "columns": columns,
        "status_counts": status_counts,
    }
    return LabelResult(
        frame=frame,
        label_name=spec.name,
        quality=quality,
        manifest=manifest,
    )
