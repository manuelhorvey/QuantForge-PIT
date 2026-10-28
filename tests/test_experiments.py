"""Module 3 Step 2 contract tests: specification identity and fold planning."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from quantforge.dataset import ResearchDataset, build_dataset
from quantforge.experiments import (
    ExperimentError,
    ExperimentIntegrityError,
    ExperimentSpec,
    WalkForwardSpec,
    plan_experiment,
    plan_folds,
)

UTC = "UTC"


def day(index: int) -> pd.Timestamp:
    return pd.Timestamp("2024-01-01", tz=UTC) + pd.Timedelta(days=index)


def make_dataset(
    times: list[pd.Timestamp] | None = None,
    *,
    missing_labels: set[int] | None = None,
    statuses: dict[int, str] | None = None,
    label_ends: dict[int, pd.Timestamp] | None = None,
    label_available: dict[int, pd.Timestamp] | None = None,
) -> ResearchDataset:
    times = [day(index) for index in range(11)] if times is None else list(times)
    missing_labels = missing_labels or set()
    statuses = statuses or {}
    label_ends = label_ends or {}
    label_available = label_available or {}
    features = pd.DataFrame(
        {
            "symbol": ["A"] * len(times),
            "decision_timestamp": times,
            "feature": [float(index) for index in range(len(times))],
            "feature2": [float(index + 10) for index in range(len(times))],
        }
    )
    label_rows: list[dict[str, object]] = []
    for index, timestamp in enumerate(times):
        if index in missing_labels:
            continue
        status = statuses.get(index, "ok")
        start = timestamp + pd.Timedelta(hours=1)
        end = label_ends.get(index, timestamp + pd.Timedelta(hours=2))
        available = label_available.get(index, end + pd.Timedelta(hours=1))
        label_rows.append(
            {
                "symbol": "A",
                "decision_timestamp": timestamp,
                "label": float(index) if status == "ok" else float("nan"),
                "label_status": status,
                "label_start_timestamp": start,
                "label_end_timestamp": end,
                "label_available_at": available,
            }
        )
    labels = pd.DataFrame(label_rows)
    return build_dataset(
        features,
        labels,
        feature_names=["feature", "feature2"],
        label_name="label",
    )


def walk(
    *,
    start: int = 5,
    end: int = 7,
    step_days: int = 2,
    **overrides: object,
) -> WalkForwardSpec:
    values: dict[str, Any] = {
        "test_start": day(start),
        "test_end": day(end),
        "step": pd.Timedelta(days=step_days),
    }
    values.update(overrides)
    return WalkForwardSpec(**values)


def experiment_spec(
    dataset: ResearchDataset,
    *,
    folds: WalkForwardSpec | None = None,
    **overrides: object,
) -> ExperimentSpec:
    values: dict[str, Any] = {
        "hypothesis": "higher feature values predict higher labels",
        "dataset_id": dataset.manifest["dataset_id"],
        "label_name": dataset.label_name,
        "feature_names": dataset.feature_names,
        "folds": folds or walk(),
        "estimator": "ridge",
        "parameters": {"alpha": 1.0},
        "code_version": "test-v1",
    }
    values.update(overrides)
    return ExperimentSpec(**values)


def test_fold_timestamps_must_be_timezone_aware_and_unambiguous():
    naive = pd.Timestamp("2024-01-03 00:00:00")
    with pytest.raises(ExperimentError):
        WalkForwardSpec(
            test_start=naive,
            test_end=pd.Timestamp("2024-01-05", tz=UTC),
            step=pd.Timedelta(days=2),
        )

    ambiguous = datetime(2024, 11, 3, 1, 30, tzinfo=ZoneInfo("America/New_York"))
    with pytest.raises(ExperimentError):
        WalkForwardSpec(
            test_start=ambiguous,
            test_end=pd.Timestamp("2024-11-04", tz=UTC),
            step=pd.Timedelta(days=1),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"test_start": day(5), "test_end": day(5), "step": pd.Timedelta(days=1)},
        {"step": pd.Timedelta(days=1)},
        {"step": pd.Timedelta(0)},
        {"step": pd.Timedelta(days=-1)},
        {"embargo": pd.Timedelta(days=-1)},
        {"max_train_span": pd.Timedelta(0)},
        {"max_folds": 0},
    ],
)
def test_walk_forward_rejects_invalid_geometry(overrides: dict[str, object]):
    values: dict[str, Any] = {
        "test_start": day(3),
        "test_end": day(5),
        "step": pd.Timedelta(days=2),
    }
    values.update(overrides)
    with pytest.raises(ExperimentError):
        WalkForwardSpec(**values)


def test_analysis_rows_exclude_missing_and_non_ok_labels_with_counts():
    dataset = make_dataset(
        missing_labels={0},
        statuses={1: "missing_exit"},
    )
    plan = plan_folds(dataset, walk(start=5, end=6, step_days=1))

    assert len(plan.analysis_indices) == 9
    assert plan.excluded_status_counts == {"label_not_ok": 1, "missing_label": 1}
    assert sum(plan.excluded_status_counts.values()) == len(dataset.frame) - len(
        plan.analysis_indices
    )
    assert day(0) not in {key[1] for key in plan.analysis_keys}
    assert day(1) not in {key[1] for key in plan.analysis_keys}
    assert day(2) in {key[1] for key in plan.analysis_keys}


def test_windows_follow_fixed_duration_geometry_without_overlap():
    plan = plan_folds(make_dataset(), walk(start=4, end=6, step_days=2, max_folds=3))
    assert [fold.test_start for fold in plan.folds] == [day(4), day(6), day(8)]
    assert [fold.test_end for fold in plan.folds] == [day(6), day(8), day(10)]
    assert plan.evaluation_start == day(4)
    assert plan.evaluation_end == day(10)
    for previous, current in zip(plan.folds, plan.folds[1:]):
        assert previous.test_end <= current.test_start


def test_fold_membership_is_disjoint_and_within_window():
    plan = plan_folds(
        make_dataset(times=[day(i) for i in range(13)]),
        walk(max_folds=2),
    )
    assert plan.evaluation_start == day(5)
    assert plan.evaluation_end == day(9)
    for fold in plan.folds:
        train = {key[1] for key in fold.train_keys}
        test = {key[1] for key in fold.test_keys}
        assert train.isdisjoint(test)
        assert all(fold.test_start <= timestamp < fold.test_end for timestamp in test)
        assert all(timestamp < fold.test_start for timestamp in train)


def test_labels_touching_test_start_and_embargo_boundary_are_purged():
    dataset = make_dataset(
        label_ends={1: day(2), 4: day(5)},
    )
    plan = plan_folds(
        dataset,
        walk(start=5, end=7, step_days=2, embargo=pd.Timedelta(days=3), max_folds=1),
    )
    fold = plan.folds[0]
    reasons = dict(zip(fold.excluded_keys, fold.excluded_reasons, strict=True))
    assert reasons[("A", day(1))] == "embargo_overlap"
    assert reasons[("A", day(4))] == "label_overlaps_test"


def test_label_availability_reaching_test_start_is_purged():
    dataset = make_dataset(label_available={4: day(5)})
    plan = plan_folds(
        dataset,
        walk(start=5, end=7, step_days=2, max_folds=1),
    )
    fold = plan.folds[0]
    reasons = dict(zip(fold.excluded_keys, fold.excluded_reasons, strict=True))
    assert dataset.frame.loc[
        dataset.frame["decision_timestamp"] == day(4), "label_available_at"
    ].iloc[0] == day(5)
    assert reasons[("A", day(4))] == "label_overlaps_test"
    assert ("A", day(4)) not in fold.train_keys


def test_post_test_rows_are_never_used_for_training():
    plan = plan_folds(make_dataset(times=[day(i) for i in range(12)]), walk())
    fold = plan.folds[0]
    assert max(key[1] for key in fold.train_keys) < fold.test_start
    reasons = dict(zip(fold.excluded_keys, fold.excluded_reasons, strict=True))
    assert reasons[("A", fold.test_end)] == "post_test_not_used"


def test_max_train_span_removes_only_older_training_rows():
    dataset = make_dataset()
    unlimited = plan_folds(dataset, walk(start=7, end=8, step_days=1, max_folds=1))
    bounded = plan_folds(
        dataset,
        walk(
            start=7,
            end=8,
            step_days=1,
            max_folds=1,
            max_train_span=pd.Timedelta(days=2),
        ),
    )
    unlimited_fold = unlimited.folds[0]
    bounded_fold = bounded.folds[0]
    assert {key[1] for key in bounded_fold.train_keys} < {
        key[1] for key in unlimited_fold.train_keys
    }
    assert all(key[1] >= day(5) for key in bounded_fold.train_keys)
    assert bounded_fold.test_keys == unlimited_fold.test_keys
    assert bounded_fold.excluded_keys == unlimited_fold.excluded_keys


def test_minimum_training_and_test_rows_fail_explicitly():
    dataset = make_dataset()
    with pytest.raises(ExperimentError, match="min_train_rows"):
        plan_folds(dataset, walk(min_train_rows=999))
    with pytest.raises(ExperimentError, match="min_test_rows"):
        plan_folds(dataset, walk(min_test_rows=999))


def test_gaps_are_allowed_and_max_folds_stops_before_data_end():
    plan = plan_folds(make_dataset(), walk(start=3, end=4, step_days=2, max_folds=2))
    assert [fold.test_start for fold in plan.folds] == [day(3), day(5)]
    assert plan.folds[0].test_end < plan.folds[1].test_start
    assert len(plan.folds) == 2
    assert plan.evaluation_end == day(6)


def test_candidate_window_with_no_test_rows_fails():
    times = [day(i) for i in range(5)] + [day(i) for i in range(7, 11)]
    dataset = make_dataset(times=times)
    with pytest.raises(ExperimentError, match="candidate fold 0"):
        plan_folds(dataset, walk(start=5, end=6, step_days=1))


def test_candidate_window_with_too_few_test_rows_fails():
    with pytest.raises(ExperimentError, match="min_test_rows"):
        plan_folds(make_dataset(), walk(start=5, end=6, step_days=1, min_test_rows=2))


def test_fixed_duration_final_window_may_extend_past_data():
    plan = plan_folds(make_dataset(), walk(start=10, end=11, step_days=1))
    assert len(plan.folds) == 1
    assert plan.folds[0].test_keys == (("A", day(10)),)
    assert plan.folds[0].test_end > day(10)


def test_reordered_dataset_preserves_canonical_identity_and_remaps_indices():
    dataset = make_dataset(times=[day(i) for i in range(12)])
    spec = experiment_spec(dataset, folds=walk(start=5, end=6, step_days=1))
    original_plan = plan_experiment(dataset, spec)
    reordered = ResearchDataset(
        frame=dataset.frame.iloc[::-1].reset_index(drop=True),
        feature_names=dataset.feature_names,
        label_name=dataset.label_name,
        quality=dataset.quality,
        manifest=dataset.manifest,
        filtered_frame=dataset.filtered_frame,
    )
    reordered_spec = experiment_spec(
        reordered, folds=walk(start=5, end=6, step_days=1)
    )
    reordered_plan = plan_experiment(reordered, reordered_spec)

    assert reordered.manifest["dataset_id"] == dataset.manifest["dataset_id"]
    assert reordered_plan.experiment_id == original_plan.experiment_id
    assert reordered_plan.fold_plan.fold_plan_hash == original_plan.fold_plan.fold_plan_hash
    assert [fold.fold_id for fold in reordered_plan.fold_plan.folds] == [
        fold.fold_id for fold in original_plan.fold_plan.folds
    ]
    assert reordered_plan.fold_plan.evaluation_start == original_plan.fold_plan.evaluation_start
    assert reordered_plan.fold_plan.evaluation_end == original_plan.fold_plan.evaluation_end
    assert {key for key in reordered_plan.fold_plan.analysis_keys} == {
        key for key in original_plan.fold_plan.analysis_keys
    }
    for original_fold, reordered_fold in zip(
        original_plan.fold_plan.folds, reordered_plan.fold_plan.folds, strict=True
    ):
        assert original_fold.train_keys == reordered_fold.train_keys
        assert original_fold.test_keys == reordered_fold.test_keys
        assert original_fold.excluded_keys == reordered_fold.excluded_keys
        assert original_fold.train_indices != reordered_fold.train_indices
        assert original_fold.test_indices != reordered_fold.test_indices


def test_experiment_id_is_stable_for_identical_inputs():
    dataset = make_dataset()
    first = experiment_spec(dataset)
    second = experiment_spec(dataset)
    assert first.experiment_id == second.experiment_id
    assert first.experiment_id == plan_experiment(dataset, first).experiment_id


def test_experiment_id_changes_with_every_identity_component():
    dataset = make_dataset()
    baseline = experiment_spec(dataset)
    variants = [
        experiment_spec(dataset, hypothesis="another hypothesis"),
        experiment_spec(dataset, dataset_id="another-dataset"),
        experiment_spec(dataset, label_name="another_label"),
        experiment_spec(dataset, feature_names=("feature2", "feature")),
        experiment_spec(dataset, estimator="linear"),
        experiment_spec(dataset, parameters={"alpha": 2.0}),
        experiment_spec(dataset, code_version="test-v2"),
        experiment_spec(dataset, metric_names=("mae", "rmse")),
        experiment_spec(dataset, selection_metric="mae"),
        experiment_spec(dataset, folds=walk(start=6, end=7, step_days=1)),
    ]
    ids = {baseline.experiment_id, *(variant.experiment_id for variant in variants)}
    assert len(ids) == len(variants) + 1


def test_parameters_are_deep_copied_and_cannot_change_experiment_id():
    dataset = make_dataset()
    parameters = {"alpha": 1.0, "nested": {"beta": [1, 2]}}
    spec = experiment_spec(dataset, parameters=parameters)
    original_id = spec.experiment_id
    parameters["alpha"] = 99.0
    parameters["nested"]["beta"].append(3)
    parameters["new"] = True
    assert spec.experiment_id == original_id
    assert spec.parameters == {"alpha": 1.0, "nested": {"beta": [1, 2]}}


def test_planning_rejects_wrong_bindings_and_mutated_dataset():
    dataset = make_dataset()
    good = experiment_spec(dataset)
    with pytest.raises(ExperimentIntegrityError):
        plan_experiment(dataset, experiment_spec(dataset, dataset_id="wrong-dataset"))
    with pytest.raises(ExperimentIntegrityError):
        plan_experiment(dataset, experiment_spec(dataset, label_name="wrong_label"))
    with pytest.raises(ExperimentIntegrityError):
        plan_experiment(
            dataset,
            experiment_spec(dataset, feature_names=("feature2", "feature")),
        )
    with pytest.raises(ExperimentIntegrityError):
        plan_folds(dataset, good.folds, dataset_id="wrong-dataset")

    dataset.frame.loc[0, "feature"] = 999.0
    with pytest.raises(ExperimentIntegrityError):
        plan_experiment(dataset, good)


def test_example_exits_zero_and_claims_no_trading_edge():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "examples" / "experiment_pipeline.py")],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    output = completed.stdout.lower()
    assert "not evidence of a trading edge" in output
    assert "profitable" not in output
