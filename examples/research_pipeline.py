"""Small, deterministic Module 2 research-dataset example.

Run from the repository root with ``PYTHONPATH=. python examples/research_pipeline.py``.
Direct execution without installation or ``PYTHONPATH`` is not supported by the current
packaging layout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.dataset import build_dataset
from quantforge.features import (
    compute_features,
    momentum_spec,
    realized_volatility_spec,
)
from quantforge.labels import LabelSpec, compute_labels
from quantforge.models import purged_time_split

DATES = pd.date_range("2024-01-01", periods=14, freq="B", tz="UTC")


def make_bars() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset, symbol in enumerate(("AAA", "BBB")):
        closes = 100.0 + offset * 10.0 + np.arange(len(DATES), dtype=float)
        for timestamp, close in zip(DATES, closes, strict=True):
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "close": close,
                    "available_at": timestamp,
                    "revision": 0,
                }
            )
    return pd.DataFrame(rows)


def make_decisions() -> pd.DataFrame:
    decisions = DATES[2:10] + pd.Timedelta(hours=13)
    return pd.DataFrame(
        {
            "decision_timestamp": list(decisions) * 2,
            "symbol": ["AAA"] * len(decisions) + ["BBB"] * len(decisions),
        }
    )


def make_universe(decisions: pd.DataFrame) -> pd.DataFrame:
    universe = decisions[["symbol", "decision_timestamp"]].copy()
    universe["eligible"] = True
    universe.loc[
        (universe["symbol"] == "BBB")
        & (universe["decision_timestamp"] < pd.Timestamp("2024-01-09", tz="UTC")),
        "eligible",
    ] = False
    return universe


def main() -> None:
    bars = make_bars()
    decisions = make_decisions()

    features = compute_features(
        decisions,
        bars,
        [momentum_spec(2), realized_volatility_spec(2)],
    )
    labels = compute_labels(
        decisions,
        bars,
        LabelSpec("forward_1", horizon=1),
    )
    dataset = build_dataset(
        features,
        labels,
        universe=make_universe(decisions),
        filter_universe=True,
    )

    split = purged_time_split(
        sample_times=dataset.frame["decision_timestamp"],
        label_start_times=dataset.frame["label_start_timestamp"],
        label_end_times=dataset.frame["label_end_timestamp"],
        label_available_times=dataset.frame["label_available_at"],
        test_start=pd.Timestamp("2024-01-09", tz="UTC"),
        test_end=pd.Timestamp("2024-01-12", tz="UTC"),
        embargo="1D",
    )

    print("dataset rows:", len(dataset.frame))
    print("dataset statuses:", dataset.quality["dataset_status_counts"])
    print("filtered rows:", dataset.quality["filtered_reasons"])
    print("dataset_id:", dataset.manifest["dataset_id"])
    print("train/test/excluded:", len(split.train), len(split.test), len(split.excluded))
    print("excluded reasons:", split.excluded_reasons)
    print()
    print(
        dataset.frame[
            [
                "symbol",
                "decision_timestamp",
                "momentum_2",
                "realized_volatility_2",
                "forward_1",
                "dataset_status",
            ]
        ].head()
    )
    print()
    print("Synthetic example only; not evidence of a trading edge.")


if __name__ == "__main__":
    main()
