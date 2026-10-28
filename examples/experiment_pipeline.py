"""Deterministic Module 3 experiment-evaluation and ledger example.

Run from the repository root with
``PYTHONPATH=. python examples/experiment_pipeline.py``. Predictions are supplied
directly from a fixed synthetic rule; no model is fitted and no trading edge is
claimed.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from quantforge.dataset import ResearchDataset, build_dataset
from quantforge.experiments import (
    ExperimentPlan,
    ExperimentSpec,
    WalkForwardSpec,
    evaluate_experiment,
    plan_experiment,
)
from quantforge.governance import TrialLedger

DATES = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")


def make_dataset() -> ResearchDataset:
    features = pd.DataFrame(
        {
            "symbol": ["AAA"] * len(DATES),
            "decision_timestamp": DATES,
            "synthetic_signal": [
                (float(index % 5) - 2.0) / 100.0 for index in range(len(DATES))
            ],
        }
    )
    labels = pd.DataFrame(
        {
            "symbol": ["AAA"] * len(DATES),
            "decision_timestamp": DATES,
            "forward_return": [
                ((float(index % 5) - 2.0) / 100.0) * 0.5 + 0.001
                for index in range(len(DATES))
            ],
            "label_status": ["ok"] * len(DATES),
            "label_start_timestamp": [value + pd.Timedelta(hours=1) for value in DATES],
            "label_end_timestamp": [value + pd.Timedelta(hours=2) for value in DATES],
            "label_available_at": [value + pd.Timedelta(hours=3) for value in DATES],
        }
    )
    return build_dataset(
        features,
        labels,
        feature_names=["synthetic_signal"],
        label_name="forward_return",
    )


def make_predictions(
    dataset: ResearchDataset,
    plan: ExperimentPlan,
) -> dict[str, pd.DataFrame]:
    signal = {
        (row.symbol, row.decision_timestamp): float(row.synthetic_signal)
        for row in dataset.frame.itertuples(index=False)
    }
    predictions: dict[str, pd.DataFrame] = {}
    for fold in plan.fold_plan.folds:
        predictions[fold.fold_id] = pd.DataFrame(
            {
                "symbol": [key[0] for key in fold.test_keys],
                "decision_timestamp": [key[1] for key in fold.test_keys],
                "prediction": [
                    0.3 * signal[key] - 0.0004 for key in fold.test_keys
                ],
            }
        )
    return predictions


def main() -> None:
    dataset = make_dataset()
    spec = ExperimentSpec(
        hypothesis="A fixed synthetic signal is associated with the synthetic label.",
        dataset_id=dataset.manifest["dataset_id"],
        label_name=dataset.label_name,
        feature_names=dataset.feature_names,
        folds=WalkForwardSpec(
            test_start=DATES[10],
            test_end=DATES[12],
            step=pd.Timedelta(days=2),
            embargo=pd.Timedelta(days=1),
            max_folds=3,
            min_train_rows=5,
            min_test_rows=2,
        ),
        estimator="none",
        parameters={"prediction_rule": "0.3 * synthetic_signal - 0.0004"},
        code_version="module-3-example-v1",
    )
    plan = plan_experiment(dataset, spec)
    predictions = make_predictions(dataset, plan)

    with tempfile.TemporaryDirectory() as directory:
        ledger = TrialLedger(Path(directory) / "trials.jsonl")
        metadata = {
            "experiment_id": plan.experiment_id,
            "fold_plan_hash": plan.fold_plan.fold_plan_hash,
            "code_version": spec.code_version,
        }
        ledger.append(
            trial_id=plan.experiment_id,
            hypothesis=spec.hypothesis,
            parameters=spec.parameters,
            data_hash=plan.dataset_id,
            metrics={},
            status="PREREGISTERED",
            metadata=metadata,
            created_at="2024-01-02T00:00:00Z",
        )

        result = evaluate_experiment(dataset, plan, predictions)

        ledger.append(
            trial_id=plan.experiment_id,
            hypothesis=spec.hypothesis,
            parameters=spec.parameters,
            data_hash=plan.dataset_id,
            metrics=result.to_dict(),
            status="COMPLETED",
            metadata={
                **metadata,
                "predictions_hash": result.predictions_hash,
            },
            created_at="2024-01-02T00:05:00Z",
        )

        verified_entries = ledger.verify()
        summary = ledger.summarize()

    print("dataset rows:", len(dataset.frame))
    print("dataset_id:", plan.dataset_id)
    print("experiment_id:", plan.experiment_id)
    print("fold_plan_hash:", plan.fold_plan.fold_plan_hash)
    print("fold test rows:", [fold.n_obs for fold in result.folds])
    print(
        "pooled metrics:",
        json.dumps(dict(result.pooled), sort_keys=True, allow_nan=False),
    )
    print("predictions_hash:", result.predictions_hash)
    print("verified ledger entries:", verified_entries)
    print(
        "ledger summary:",
        json.dumps(summary, sort_keys=True, allow_nan=False),
    )
    print("Synthetic example only; descriptive metrics are not evidence of a trading edge.")


if __name__ == "__main__":
    main()
