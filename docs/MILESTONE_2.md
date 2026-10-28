# Module 2 — Research Dataset Milestone

## Status

Implemented and validated locally. No backtest-compatibility claim is made.

Module 2 adds a declarative PIT feature engine, forward-return labels, deterministic
dataset assembly with canonical hashing, and a forward-only time split with purge and
embargo. It builds on the accepted Module 1 data layer and does not change existing
`purged_split` behavior.

## Verification

- Full suite: 165 passed.
- Module 2 tests: 69 passed.
- Targeted Ruff over new Module 2 files: clean. `quantforge/models/splits.py` retains
  three pre-existing `E701` findings in the unchanged `purged_split`.
- Repository-wide Ruff still reports pre-existing style findings (`E401`, `E701`,
  `E702`) outside the newly added files; these are not part of this remediation.
- Targeted mypy with missing imports ignored reports only pre-existing
  `quantforge/features/fractional.py` errors.

## Public surface

```python
from quantforge.features import (
    FeatureSpec, compute_features,
    momentum_spec, realized_volatility_spec,
    rolling_zscore_spec, fracdiff_spec,
)
from quantforge.labels import LabelSpec, compute_labels
from quantforge.dataset import (
    ResearchDataset, build_dataset, hash_payload, hash_frame, validate_dataset,
)
from quantforge.models import TimeSplitResult, purged_time_split
```

## Feature contract

`compute_features` evaluates each `FeatureSpec` against history that is eligible at the
decision time:

- `timestamp <= decision_timestamp`
- `available_at <= decision_timestamp`
- highest eligible revision per event wins
- conflicting equal-revision payloads fail closed
- instruments and bar frequencies are isolated
- insufficient history is explicit
- warm-up `NaN` is retained, not zero-filled
- `feature_available_at` is conservatively the decision timestamp

`fracdiff_spec` rejects `d` outside `[0, 1]` and a non-positive/non-finite threshold.
Its default name encodes both parameters using identifier-safe text, so distinct
configurations do not collide and the result is accepted by `FeatureSpec`.

Feature callbacks are arbitrary Python callables; they are trusted code, not a
sandbox. They must not read future rows, mutable global state, or network/clock state
unless that state is itself point-in-time safe. A callback that closes over external
mutable state can leak it into features.

Each feature carries per-feature metadata columns using the `__` suffix convention:
`__available_at`, `__source_max_available_at`, `__source`, `__revision`,
`__status`, and `__reason`. Aggregate `feature_*` columns summarize availability,
source, revision, status, and reasons across the row.

## Label contract

Labels use the frozen convention:

- entry is the first observed bar strictly after the decision
- exit is `horizon` observed bars after entry
- `horizon=1` is `price[exit] / price[entry] - 1`
- missing entry, exit, price, benchmark, or adjustment coverage is explicit
- the horizon is never shortened to force a value
- `label_start_timestamp` / `label_end_timestamp` are entry / exit event times
- `label_available_at` includes entry, exit, benchmark, and relevant adjustment
  availability

The label execution convention is
`next_observed_bar_close_to_close`, and
`backtest_compatibility="not_claimed"`. Current backtests execute on a different
timing convention and must not be described as numerically equivalent to these labels.

Adjusted labels require a `CorporateActionLedger`; missing coverage fails closed. The
price window used for adjustment is restricted to `[entry_event, exit_event]`, and
corporate actions outside that interval are ignored so later events cannot distort the
entry/exit ratio.

Excess-return labels require the stock and benchmark price conventions to match.
`LabelSpec.benchmark_price_convention` defaults to `raw`; an adjusted benchmark requires
its own `benchmark_ledger`, while a raw benchmark rejects a benchmark ledger. Selected
benchmark actions and availability are included in `label_available_at`, and the result
exposes `benchmark_price_convention`, `benchmark_adjustment_coverage`,
`benchmark_adjustment_actions`, and `benchmark_source`. Incomplete benchmark adjustment
coverage fails closed. Dividend and total-return benchmark adjustments have direct
hand-calculated regression coverage in addition to split-adjusted coverage.

## Dataset contract

`build_dataset` joins one feature row and one label row per `(symbol,
decision_timestamp)` key. Missing sides are retained with explicit status:

- `ok`
- `missing_label`
- `missing_features`
- `label_not_ok`

Universe membership is optional. Missing membership is `unknown_membership`, not
universal inclusion. When the universe frame carries `available_at`, membership is
usable only when `available_at <= decision_timestamp`. `require_universe=True` fails
when membership or eligibility is unavailable for a dataset row. `filter_universe=True`
removes unknown or ineligible rows; removed rows and counts are exposed through
`filtered_frame`, `filtered_reasons`, and dataset quality. When a universe is supplied,
the dataset frame also carries `universe_available_at` and
`universe_membership_status` alongside `universe_status` and `universe_eligible`. The
universe source hash includes eligibility, availability, and membership status over the
dataset rows, so a status-only change cannot silently reuse the same source identity.

Raw feature and label metadata is normalized and validated during assembly. Feature
availability/source-max timestamps cannot be after the decision; feature statuses must
agree with finite values; label intervals must be ordered and label start must be after
the decision; label availability must cover the label end; and `label_status="ok"` requires
a finite value while a non-ok status cannot carry a finite value. `validate_dataset`
is public and is also invoked by `ResearchDataset` construction; it requires a non-empty
manifest mapping with non-empty `schema_version`, `manifest_hash`, and `dataset_id`, and
quality matching `manifest["quality"]`. It checks required columns, keys, names,
statuses, manifest hashes, and filtered-frame keys. It also recomputes the feature,
label, universe, source, and dataset hashes from the current frame. A mutated frame,
stale manifest, missing hash, malformed root, or quality mismatch fails validation
instead of silently reusing the old identity.

`ResearchDataset` is frozen, so the frame reference cannot be replaced. Construction
deep-copies the input frame, isolating the dataset from later mutation of the caller's
source frame, and rejects a missing or incomplete manifest. The exposed `frame` is
immutable by contract: pandas DataFrames expose mutating operations, so callers must not
modify `dataset.frame`. Use `ResearchDataset.verify_integrity()` or `validate_dataset()`
after any suspected mutation; both recompute the hashes and fail closed when the frame or
manifest no longer matches.

Canonical hashing is type-tagged SHA-256 over canonical JSON. It distinguishes numeric
`NaN`, null, string `"NaN"`, `0.0`, `-0.0`, integer and float values; rejects
infinities and callables; rejects duplicate canonical keys; and performs a stable row
sort by `(symbol, decision_timestamp)`. The manifest exposes separate feature, label,
universe, source, content, dataset, manifest, and dataset IDs, and binds dataset
quality. The feature, label, universe, and dataset content/source hashes are computed
over the assembled dataset rows, so the dataset hash covers the full validated frame
and the component hashes can be independently recomputed from that frame.

## Time split contract

`purged_time_split` is forward-only:

- test: `test_start <= sample_time < test_end`
- training: only samples with `sample_time < test_start`
- purge: exclude when `label_end >= test_start` or `label_available >= test_start`
- embargo: exclude when either timestamp reaches `test_start - embargo`
- post-test samples: reported as `post_test_not_used`, never trained on

Boundary equality is treated conservatively. Train and test indices are returned in
chronological sample-time order with original input position as the tie-breaker. A
label start earlier than its sample timestamp is invalid; equality is accepted for
callers that use the sample timestamp as the label start marker. Existing
`purged_split` is unchanged.

## Running the example

Run the example from the repository root with the source tree on the import path:

```bash
PYTHONPATH=. python examples/research_pipeline.py
```

Direct execution without installation or `PYTHONPATH` is not supported by the current
packaging layout.

## Deferred

Persistence of research datasets, feature stores, provider-specific fundamentals or
alternative data, production corporate-action histories, and any claim that the new
label convention matches the existing backtest engine are intentionally out of scope.
