"""Module 3 Step 3 contract tests: exact-coverage OOS evaluation."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantforge.dataset import ResearchDataset, build_dataset, hash_frame
from quantforge.experiments import (
    ExperimentError,
    ExperimentIntegrityError,
    ExperimentSpec,
    PredictionCoverageError,
    PurgedFold,
    WalkForwardSpec,
    evaluate_experiment,
    evaluate_fold,
    plan_experiment,
)
from quantforge.governance import LedgerIntegrityError, TrialLedger


def day(index: int) -> pd.Timestamp:
    return pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=index)


def make_dataset(
    times: list[pd.Timestamp] | None = None,
    *,
    label_values: dict[int, float] | None = None,
) -> ResearchDataset:
    times = [day(index) for index in range(11)] if times is None else list(times)
    label_values = label_values or {}
    features = pd.DataFrame(
        {
            "symbol": ["A"] * len(times),
            "decision_timestamp": times,
            "feature": [float(index) for index in range(len(times))],
        }
    )
    labels = pd.DataFrame(
        {
            "symbol": ["A"] * len(times),
            "decision_timestamp": times,
            "label": [
                label_values.get(index, float(index)) for index in range(len(times))
            ],
            "label_status": ["ok"] * len(times),
            "label_start_timestamp": [value + pd.Timedelta(hours=1) for value in times],
            "label_end_timestamp": [value + pd.Timedelta(hours=2) for value in times],
            "label_available_at": [value + pd.Timedelta(hours=3) for value in times],
        }
    )
    return build_dataset(
        features,
        labels,
        feature_names=["feature"],
        label_name="label",
    )


def walk(
    *,
    start: int = 5,
    end: int = 7,
    step_days: int = 2,
    **overrides: Any,
) -> WalkForwardSpec:
    values: dict[str, Any] = {
        "test_start": day(start),
        "test_end": day(end),
        "step": pd.Timedelta(days=step_days),
    }
    values.update(overrides)
    return WalkForwardSpec(**values)


def spec_for(
    dataset: ResearchDataset,
    *,
    folds: WalkForwardSpec | None = None,
) -> ExperimentSpec:
    return ExperimentSpec(
        hypothesis="features predict labels",
        dataset_id=dataset.manifest["dataset_id"],
        label_name=dataset.label_name,
        feature_names=dataset.feature_names,
        folds=folds or walk(),
        estimator="ridge",
        parameters={"alpha": 1.0},
        code_version="test-v1",
    )


def predictions_for(
    fold: PurgedFold,
    values: list[float],
    *,
    extra_columns: dict[str, list[object]] | None = None,
) -> pd.DataFrame:
    data: dict[str, object] = {
        "symbol": [key[0] for key in fold.test_keys],
        "decision_timestamp": [key[1] for key in fold.test_keys],
        "prediction": values,
    }
    for name, column in (extra_columns or {}).items():
        data[name] = column
    return pd.DataFrame(data)


def test_prediction_coverage_accepts_exact_test_keys_in_any_order():
    dataset = make_dataset()
    plan = plan_experiment(dataset, spec_for(dataset))
    fold = plan.fold_plan.folds[0]
    original = predictions_for(fold, [2.0, 1.0])
    reordered = original.iloc[::-1].assign(note=["x", "y"])
    reordered = reordered.loc[:, ["note", "prediction", "decision_timestamp", "symbol"]]

    first = evaluate_fold(dataset, plan, fold, original)
    second = evaluate_fold(dataset, plan, fold, reordered)

    assert first.n_test == 2
    assert first.n_obs == 2
    assert first.test_start == fold.test_start
    assert first.test_end == fold.test_end
    assert first.metrics == second.metrics
    assert first.predictions_hash == second.predictions_hash


def test_prediction_coverage_rejects_duplicate_missing_and_extra_keys():
    dataset = make_dataset()
    plan = plan_experiment(dataset, spec_for(dataset))
    fold = plan.fold_plan.folds[0]
    valid = predictions_for(fold, [1.0, 2.0])

    duplicate = pd.concat([valid.iloc[[0]], valid], ignore_index=True)
    with pytest.raises(PredictionCoverageError, match="1 duplicate key"):
        evaluate_fold(dataset, plan, fold, duplicate)

    missing = valid.iloc[[0]]
    with pytest.raises(PredictionCoverageError, match="1 missing key"):
        evaluate_fold(dataset, plan, fold, missing)

    extra = pd.concat(
        [
            valid,
            pd.DataFrame(
                {
                    "symbol": ["A"],
                    "decision_timestamp": [day(7)],
                    "prediction": [1.0],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(PredictionCoverageError, match="1 extra key"):
        evaluate_fold(dataset, plan, fold, extra)


@pytest.mark.parametrize(
    "bad_value",
    ["1.0", None, np.nan, np.inf, np.float64(-np.inf), True, np.bool_(False)],
)
def test_prediction_values_must_be_finite_real_numbers(bad_value: object):
    dataset = make_dataset()
    plan = plan_experiment(dataset, spec_for(dataset))
    fold = plan.fold_plan.folds[0]
    predictions = predictions_for(fold, [0.0, 0.0])
    predictions["prediction"] = pd.Series([bad_value, 0.0], dtype=object)

    with pytest.raises(ExperimentError, match="prediction row 0"):
        evaluate_fold(dataset, plan, fold, predictions)


def test_metrics_match_hand_calculation_and_ignore_caller_labels():
    dataset = make_dataset()
    plan = plan_experiment(
        dataset,
        spec_for(dataset, folds=walk(max_folds=1)),
    )
    fold = plan.fold_plan.folds[0]
    metric_names = ("mae", "rmse", "mean_error", "pearson", "directional_accuracy")
    predictions = predictions_for(
        fold,
        [7.0, 3.0],
        extra_columns={"label": [999.0, -999.0]},
    )

    result = evaluate_fold(
        dataset,
        plan,
        fold,
        predictions,
        metric_names=metric_names,
    )

    assert result.n_obs == result.n_test == 2
    assert result.metrics["mae"] == pytest.approx(2.5)
    assert result.metrics["rmse"] == pytest.approx(math.sqrt(6.5))
    assert result.metrics["mean_error"] == pytest.approx(-0.5)
    assert result.metrics["pearson"] == pytest.approx(-1.0)
    assert result.metrics["directional_accuracy"] == pytest.approx(1.0)
    overall = evaluate_experiment(
        dataset,
        plan,
        {fold.fold_id: predictions},
        metric_names=metric_names,
    )
    assert json.loads(json.dumps(overall.to_dict()))["pooled_n_obs"] == 2


def test_pearson_is_none_for_insufficient_or_zero_variance_data():
    dataset = make_dataset()
    one_row_plan = plan_experiment(
        dataset,
        spec_for(dataset, folds=walk(start=5, end=6, step_days=1)),
    )
    one_row_fold = one_row_plan.fold_plan.folds[0]
    one_row = evaluate_fold(
        dataset,
        one_row_plan,
        one_row_fold,
        predictions_for(one_row_fold, [1.0]),
        metric_names=("pearson",),
    )
    assert one_row.metrics == {"pearson": None}

    plan = plan_experiment(dataset, spec_for(dataset))
    fold = plan.fold_plan.folds[0]
    zero_variance = evaluate_fold(
        dataset,
        plan,
        fold,
        predictions_for(fold, [1.0, 1.0]),
        metric_names=("pearson",),
    )
    assert zero_variance.metrics == {"pearson": None}

    constant_actual_dataset = make_dataset(label_values={5: 5.0, 6: 5.0})
    constant_actual_plan = plan_experiment(
        constant_actual_dataset,
        spec_for(constant_actual_dataset),
    )
    constant_actual_fold = constant_actual_plan.fold_plan.folds[0]
    constant_actual = evaluate_fold(
        constant_actual_dataset,
        constant_actual_plan,
        constant_actual_fold,
        predictions_for(constant_actual_fold, [1.0, 2.0]),
        metric_names=("pearson",),
    )
    assert constant_actual.metrics == {"pearson": None}


def test_directional_accuracy_ignores_zero_labels_and_zero_predictions():
    dataset = make_dataset(label_values={5: 0.0, 6: 5.0})
    plan = plan_experiment(dataset, spec_for(dataset))
    fold = plan.fold_plan.folds[0]

    result = evaluate_fold(
        dataset,
        plan,
        fold,
        predictions_for(fold, [0.0, 0.0]),
        metric_names=("directional_accuracy",),
    )

    assert result.metrics == {"directional_accuracy": 0.0}

    all_zero_dataset = make_dataset(label_values={5: 0.0, 6: 0.0})
    all_zero_plan = plan_experiment(all_zero_dataset, spec_for(all_zero_dataset))
    all_zero_fold = all_zero_plan.fold_plan.folds[0]
    undefined = evaluate_fold(
        all_zero_dataset,
        all_zero_plan,
        all_zero_fold,
        predictions_for(all_zero_fold, [1.0, 2.0]),
        metric_names=("directional_accuracy",),
    )
    assert undefined.metrics == {"directional_accuracy": None}


def test_pooled_metrics_use_combined_rows_not_average_of_folds():
    times = [day(index) for index in range(8)] + [day(9), day(10)]
    dataset = make_dataset(times=times)
    plan = plan_experiment(
        dataset,
        spec_for(dataset, folds=walk(start=5, end=7, step_days=2, max_folds=2)),
    )
    first, second = plan.fold_plan.folds
    assert len(first.test_keys) == 2
    assert len(second.test_keys) == 1
    by_fold = {
        first.fold_id: predictions_for(first, [0.0, 0.0]),
        second.fold_id: predictions_for(second, [0.0]),
    }

    result = evaluate_experiment(dataset, plan, by_fold)

    assert [fold.n_obs for fold in result.folds] == [2, 1]
    assert [fold.metrics["mae"] for fold in result.folds] == [5.5, 7.0]
    assert result.pooled["mae"] == pytest.approx(6.0)
    assert result.pooled["mae"] != pytest.approx(6.25)
    assert result.pooled_n_obs == 3
    pooled_keys = [key for fold in plan.fold_plan.folds for key in fold.test_keys]
    assert set(pooled_keys) == {("A", day(5)), ("A", day(6)), ("A", day(7))}
    assert len(pooled_keys) == len(set(pooled_keys)) == result.pooled_n_obs
    assert result.pooled_n_obs == sum(fold.n_obs for fold in result.folds)


def test_prediction_hashes_are_stable_across_row_and_extra_column_order():
    dataset = make_dataset()
    plan = plan_experiment(
        dataset,
        spec_for(dataset, folds=walk(max_folds=2)),
    )
    first_by_fold = {
        fold.fold_id: predictions_for(
            fold,
            [1.5, 2.5],
            extra_columns={"note": ["a", "b"]},
        )
        for fold in plan.fold_plan.folds
    }
    second_by_fold = {
        fold_id: frame.iloc[::-1].loc[
            :, ["note", "prediction", "symbol", "decision_timestamp"]
        ]
        for fold_id, frame in first_by_fold.items()
    }
    first_result = evaluate_experiment(dataset, plan, first_by_fold)
    second_result = evaluate_experiment(
        dataset,
        plan,
        dict(reversed(list(second_by_fold.items()))),
    )

    assert first_result.predictions_hash == second_result.predictions_hash
    expected_pooled = hash_frame(
        pd.concat(
            [first_by_fold[fold.fold_id] for fold in plan.fold_plan.folds],
            ignore_index=True,
        ),
        key_cols=[dataset.instrument_col, dataset.decision_col],
        columns=[dataset.instrument_col, dataset.decision_col, "prediction"],
        schema_version="quantforge.predictions.v1",
    )
    assert first_result.predictions_hash == expected_pooled
    for first, second in zip(first_result.folds, second_result.folds, strict=True):
        assert first.predictions_hash == second.predictions_hash
    for fold in plan.fold_plan.folds:
        expected = hash_frame(
            first_by_fold[fold.fold_id],
            key_cols=[dataset.instrument_col, dataset.decision_col],
            columns=[dataset.instrument_col, dataset.decision_col, "prediction"],
            schema_version="quantforge.predictions.v1",
        )
        assert first_result.folds[fold.ordinal].predictions_hash == expected


def test_result_serialization_uses_none_for_undefined_metrics():
    dataset = make_dataset(label_values={5: 0.0, 6: 0.0})
    plan = plan_experiment(
        dataset,
        spec_for(dataset, folds=walk(max_folds=1)),
    )
    fold = plan.fold_plan.folds[0]
    result = evaluate_experiment(
        dataset,
        plan,
        {fold.fold_id: predictions_for(fold, [0.0, 0.0])},
        metric_names=("mae", "pearson", "directional_accuracy"),
    )

    payload = result.to_dict()
    assert payload["pooled"]["pearson"] is None
    assert payload["pooled"]["directional_accuracy"] is None
    json.dumps(payload, allow_nan=False)


def test_non_finite_derived_metrics_fail_closed():
    dataset = make_dataset(label_values={5: -1e308, 6: 0.0})
    plan = plan_experiment(
        dataset,
        spec_for(dataset, folds=walk(max_folds=1)),
    )
    fold = plan.fold_plan.folds[0]

    with pytest.raises(ExperimentError, match="not finite"):
        evaluate_fold(
            dataset,
            plan,
            fold,
            predictions_for(fold, [1e308, 0.0]),
            metric_names=("mae",),
        )


def test_dataset_mutation_after_planning_is_detected_at_evaluation():
    dataset = make_dataset()
    plan = plan_experiment(dataset, spec_for(dataset))
    fold = plan.fold_plan.folds[0]
    predictions = predictions_for(fold, [1.0, 2.0])
    dataset.frame.loc[0, "feature"] = 999.0

    with pytest.raises(ExperimentIntegrityError):
        evaluate_fold(dataset, plan, fold, predictions)


def test_evaluate_experiment_requires_exact_fold_mapping_and_known_fold():
    dataset = make_dataset()
    plan = plan_experiment(
        dataset,
        spec_for(dataset, folds=walk(max_folds=1)),
    )
    fold = plan.fold_plan.folds[0]
    valid = {fold.fold_id: predictions_for(fold, [1.0, 2.0])}

    with pytest.raises(ExperimentError, match="1 missing fold"):
        evaluate_experiment(dataset, plan, {})
    with pytest.raises(ExperimentError, match="1 extra fold"):
        evaluate_experiment(dataset, plan, {**valid, "unknown": predictions_for(fold, [1.0, 2.0])})
    with pytest.raises(ExperimentIntegrityError):
        unknown_fold = PurgedFold(
            fold_id="unknown",
            ordinal=0,
            train_indices=(),
            test_indices=(),
            excluded_indices=(),
            excluded_reasons=(),
            test_start=fold.test_start,
            test_end=fold.test_end,
            embargo=fold.embargo,
            train_keys=(),
            test_keys=(),
            excluded_keys=(),
        )
        evaluate_fold(dataset, plan, unknown_fold, predictions_for(fold, [1.0, 2.0]))


def test_end_to_end_evaluation_round_trips_through_verified_ledger(tmp_path):
    dataset = make_dataset()
    plan = plan_experiment(dataset, spec_for(dataset))
    signal = {
        (row.symbol, row.decision_timestamp): float(row.feature)
        for row in dataset.frame.itertuples(index=False)
    }
    predictions = {
        fold.fold_id: predictions_for(
            fold,
            [0.5 * signal[key] + 1.0 for key in fold.test_keys],
        )
        for fold in plan.fold_plan.folds
    }
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    metadata = {
        "experiment_id": plan.experiment_id,
        "fold_plan_hash": plan.fold_plan.fold_plan_hash,
        "code_version": plan.spec.code_version,
    }
    registration_hash = ledger.append(
        trial_id=plan.experiment_id,
        hypothesis=plan.spec.hypothesis,
        parameters=plan.spec.parameters,
        data_hash=plan.dataset_id,
        metrics={},
        status="PREREGISTERED",
        metadata=metadata,
        created_at="2024-01-02T00:00:00Z",
    )

    result = evaluate_experiment(dataset, plan, predictions)
    completion_hash = ledger.append(
        trial_id=plan.experiment_id,
        hypothesis=plan.spec.hypothesis,
        parameters=plan.spec.parameters,
        data_hash=plan.dataset_id,
        metrics=result.to_dict(),
        status="COMPLETED",
        metadata={**metadata, "predictions_hash": result.predictions_hash},
        created_at="2024-01-02T00:05:00Z",
    )

    records = ledger.read()
    summary = ledger.summarize()

    assert ledger.verify() == 2
    assert [record.status for record in records] == ["PREREGISTERED", "COMPLETED"]
    assert records[0].entry_sha256 == registration_hash
    assert records[1].entry_sha256 == completion_hash
    assert records[1].metrics == result.to_dict()
    assert records[1].metadata["predictions_hash"] == result.predictions_hash
    assert summary == {
        "schema_version": "quantforge.trial_ledger.v2",
        "entries": 2,
        "trials": 1,
        "by_status": {"COMPLETED": 1},
        "by_experiment": {plan.experiment_id: 1},
        "unregistered_trials": 0,
        "head_sha256": completion_hash,
    }


def test_failed_evaluation_is_recorded_as_failed_not_completed(tmp_path):
    dataset = make_dataset()
    plan = plan_experiment(
        dataset,
        spec_for(dataset, folds=walk(max_folds=1)),
    )
    fold = plan.fold_plan.folds[0]
    predictions = {
        fold.fold_id: predictions_for(fold, [1.0, 2.0]).iloc[[0]],
    }
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    metadata = {
        "experiment_id": plan.experiment_id,
        "fold_plan_hash": plan.fold_plan.fold_plan_hash,
        "code_version": plan.spec.code_version,
    }
    ledger.append(
        trial_id=plan.experiment_id,
        hypothesis=plan.spec.hypothesis,
        parameters=plan.spec.parameters,
        data_hash=plan.dataset_id,
        metrics={},
        status="PREREGISTERED",
        metadata=metadata,
    )

    with pytest.raises(PredictionCoverageError):
        evaluate_experiment(dataset, plan, predictions)

    ledger.append(
        trial_id=plan.experiment_id,
        hypothesis=plan.spec.hypothesis,
        parameters=plan.spec.parameters,
        data_hash=plan.dataset_id,
        metrics={"error_type": "PredictionCoverageError"},
        status="FAILED",
        metadata=metadata,
    )
    before = ledger.path.read_text(encoding="utf-8")

    with pytest.raises(LedgerIntegrityError, match="more than two events|duplicate terminal"):
        ledger.append(
            trial_id=plan.experiment_id,
            hypothesis=plan.spec.hypothesis,
            parameters=plan.spec.parameters,
            data_hash=plan.dataset_id,
            metrics={},
            status="COMPLETED",
            metadata=metadata,
        )

    assert ledger.path.read_text(encoding="utf-8") == before
    assert [record.status for record in ledger.read()] == ["PREREGISTERED", "FAILED"]
    assert ledger.summarize()["by_status"] == {"FAILED": 1}
