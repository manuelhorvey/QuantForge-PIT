# Module 3 — Experiment Evaluation and Integrity Milestone

## Status

Implementation complete; final acceptance decision pending owner review. Module 3
provides descriptive out-of-sample evaluation and tamper-evident trial logging. It does
**not** fit models, select strategies, establish statistical significance, or claim a
profitable trading edge.

Module 3 builds on the accepted Module 1 data layer and Module 2 research-dataset
layer. No Module 1/2 source contract was changed.

## Verification

- Full suite: 259 passed, with only the existing `dateutil` fallback warning in
  `tests/test_data.py`.
- Module 3 focused tests: 94 passed (48 ledger, 26 experiment/fold, 20 evaluation).
- Total: 165 pre-Module-3 baseline tests plus 94 Module 3 tests = 259.
- Targeted Ruff: clean for Module 3 source, the example, and Module 3 tests.
- Scoped mypy with `--ignore-missing-imports --follow-imports=skip`: clean for the
  changed source, example, and test files.
- Exact specification mypy command without `--follow-imports=skip`: **does not pass**.
  It reports two pre-existing errors in unchanged
  `quantforge/features/fractional.py:22,25`. Those errors are outside Module 3 scope and
  were not remediated.
- `compileall` and `git diff --check`: clean.
- Example: exits `0` and prints an explicit non-edge disclaimer.
- No new dependency was added to `pyproject.toml`.

## Public surface

```python
from quantforge.experiments import (
    ExperimentError,
    ExperimentIntegrityError,
    PredictionCoverageError,
    WalkForwardSpec,
    ExperimentSpec,
    PurgedFold,
    FoldPlan,
    ExperimentPlan,
    FoldResult,
    ExperimentResult,
    plan_folds,
    plan_experiment,
    evaluate_fold,
    evaluate_experiment,
)
from quantforge.governance import (
    LedgerIntegrityError,
    TrialLedger,
    TrialRecord,
)
```

`plan_folds` remains public for focused testing. The supported entry point is
`plan_experiment`, which validates dataset/spec binding before planning folds.

## End-to-end workflow

```python
dataset = build_dataset(features, labels)

spec = ExperimentSpec(
    hypothesis="...",
    dataset_id=dataset.manifest["dataset_id"],
    label_name=dataset.label_name,
    feature_names=dataset.feature_names,
    folds=WalkForwardSpec(...),
    estimator="ridge",
    parameters={"alpha": 1.0},
    code_version="<release>",
)
plan = plan_experiment(dataset, spec)

ledger.append(
    trial_id=plan.experiment_id,
    hypothesis=spec.hypothesis,
    parameters=spec.parameters,
    data_hash=plan.dataset_id,
    metrics={},
    status="PREREGISTERED",
    metadata={
        "experiment_id": plan.experiment_id,
        "fold_plan_hash": plan.fold_plan.fold_plan_hash,
        "code_version": spec.code_version,
    },
)

result = evaluate_experiment(dataset, plan, predictions_by_fold)

ledger.append(
    trial_id=plan.experiment_id,
    hypothesis=spec.hypothesis,
    parameters=spec.parameters,
    data_hash=plan.dataset_id,
    metrics=result.to_dict(),
    status="COMPLETED",
    metadata={
        "experiment_id": plan.experiment_id,
        "fold_plan_hash": plan.fold_plan.fold_plan_hash,
        "predictions_hash": result.predictions_hash,
        "code_version": spec.code_version,
    },
)
ledger.verify()
```

Predictions are supplied by the caller. The caller must fit only on
`fold.train_indices` and predict only `fold.test_indices`; Module 3 cannot prevent a
deliberate train/test leak or direct use of labels. The example therefore uses a
fixed synthetic prediction rule and fits no model.

## Dataset and experiment binding

`ExperimentSpec` binds the dataset ID, label name, ordered feature names, fold
geometry, estimator name, canonical parameters, code version, metric set, and
selection metric into a deterministic `experiment_id`. Caller parameter mappings are
deep-copied and canonicalized, so later mutation does not change the ID.

Planning revalidates the dataset and rejects:

- a dataset/spec ID mismatch;
- a label-name mismatch;
- feature-order or feature-set mismatch;
- a mutated dataset whose hashes no longer verify;
- invalid fold geometry.

## Fold contract

`WalkForwardSpec` requires timezone-aware UTC boundaries and enforces:

- `test_start < test_end`;
- positive step, at least the test-window duration;
- non-negative embargo;
- positive `min_train_rows` and `min_test_rows`;
- positive `max_train_span` and `max_folds` when supplied.

Candidate test windows have fixed duration, advance by exactly `step`, and never
overlap. Windows with too few test rows fail; gaps are allowed when `step` exceeds
the window duration; `max_folds` stops generation deterministically.

Analysis rows require `dataset_status="ok"`, an `ok` label, complete label timing,
and a finite label. Exclusion counts are reported by status. For each fold:

- training decisions must be before the test start;
- training labels are purged when their end or availability reaches the test start;
- the embargo boundary is conservative;
- post-test rows are never included in training;
- `max_train_span` removes only older training rows;
- train and test keys are disjoint.

Returned train/test indices are positional indices into the validated dataset. Fold
IDs and plan hashes are stable under equivalent dataset row reordering.

## Prediction and metric contract

Each fold requires exactly one prediction row for every test key. Row order is
irrelevant. Duplicate, missing, and extra keys fail with bounded diagnostic samples.
Predictions must be finite real numbers; booleans, nulls, `NaN`, infinities, and
non-numeric values fail. Labels come from the validated dataset, not from any
caller-supplied label column in the prediction frame.

Supported descriptive metrics:

- `mae`
- `rmse`
- `mean_error`
- `pearson`
- `directional_accuracy`

`pearson` is `None` for fewer than two observations or zero variance in either input.
`directional_accuracy` ignores zero actual labels, counts zero predictions as
non-hits, and is `None` when all actual labels are zero. Undefined metrics serialize
as `None` under `json.dumps(..., allow_nan=False)`.

Pooled metrics are calculated from all pooled OOS rows, not averaged from fold
metrics. Pooled prediction keys must not overlap. Prediction hashes use canonical
`quantforge.predictions.v1` frames and are stable under row and extra-column
reordering.

Metrics are descriptive diagnostics. They are not risk-adjusted, cost-aware,
statistically significant, or evidence of deployable alpha.

## Ledger contract

The ledger is canonical JSONL with a SHA-256 hash chain. Each trial may have one
optional registration event followed by at most one terminal event:

- registration: `PREREGISTERED`, `UNVALIDATED`;
- terminal: `COMPLETED`, `FAILED`, `ABANDONED`, `EXCLUDED`.

Terminal-only one-shot use is accepted and reported as an unregistered trial.
Hypothesis, parameters, data hash, and experiment ID cannot change between events.
Duplicate registrations, duplicate terminal events, and any event after a terminal
event fail without extending the file. A failed evaluation must be recorded as
`FAILED` (or another terminal failure status); it cannot later be recorded as
`COMPLETED`.

`verify()` checks canonical encoding, all record fields, entry hashes, previous-hash
links, and lifecycle validity. `summarize()` reports entries, unique trials, latest
status counts, experiment counts, unregistered terminal-only trials, and the head
hash.

### Integrity limits

The chain is tamper-evident, not tamper-proof. It detects modified payloads, hash
changes, reordered records, and deleted middle records. Whole-file replacement, tail
truncation, and a writer that rewrites the entire file are detectable only with an
externally published head hash or WORM/immutable storage.

The ledger is single-writer. Concurrent appends are unsupported and must be
serialized by the caller. `code_version` is supplied by the caller and hashed; it is
not code attestation. Preregistration is procedural: the ledger cannot prove the
researcher did not inspect data elsewhere.

## Example

Run from the repository root:

```bash
PYTHONPATH=. python examples/experiment_pipeline.py
```

The example builds a deterministic synthetic Module 2 dataset, creates a bound
experiment plan, generates fixed predictions from a feature-only rule, evaluates
three folds, records registration and completion in a temporary ledger, verifies the
chain, and prints the pooled descriptive metrics and ledger summary. It fits no model
and states explicitly that the metrics are not evidence of a trading edge.

## Acceptance matrix evidence

| ID | Evidence |
| --- | --- |
| M3-01 | `test_fold_timestamps_must_be_timezone_aware_and_unambiguous` |
| M3-02 | `test_walk_forward_rejects_invalid_geometry` |
| M3-03 | `test_analysis_rows_exclude_missing_and_non_ok_labels_with_counts`; non-finite branch implemented but not directly constructible through Module 2 public validation |
| M3-04 | `test_windows_follow_fixed_duration_geometry_without_overlap` |
| M3-05 | `test_windows_follow_fixed_duration_geometry_without_overlap`, `test_fold_membership_is_disjoint_and_within_window` |
| M3-06 | `test_labels_touching_test_start_and_embargo_boundary_are_purged`, `test_label_availability_reaching_test_start_is_purged` |
| M3-07 | `test_post_test_rows_are_never_used_for_training` |
| M3-08 | `test_max_train_span_removes_only_older_training_rows` |
| M3-09 | `test_minimum_training_and_test_rows_fail_explicitly` |
| M3-10 | `test_gaps_are_allowed_and_max_folds_stops_before_data_end`, `test_candidate_window_with_no_test_rows_fails`, `test_candidate_window_with_too_few_test_rows_fails`, `test_fixed_duration_final_window_may_extend_past_data` |
| M3-11 | `test_reordered_dataset_preserves_canonical_identity_and_remaps_indices` |
| M3-12 | `test_experiment_id_is_stable_for_identical_inputs` |
| M3-13 | `test_experiment_id_changes_with_every_identity_component` |
| M3-14 | `test_parameters_are_deep_copied_and_cannot_change_experiment_id` |
| M3-15 | `test_planning_rejects_wrong_bindings_and_mutated_dataset` |
| M3-16 | `test_prediction_coverage_accepts_exact_test_keys_in_any_order` |
| M3-17 | `test_prediction_coverage_rejects_duplicate_missing_and_extra_keys` |
| M3-18 | `test_prediction_values_must_be_finite_real_numbers` |
| M3-19 | `test_metrics_match_hand_calculation_and_ignore_caller_labels` |
| M3-20 | `test_pearson_is_none_for_insufficient_or_zero_variance_data` |
| M3-21 | `test_directional_accuracy_ignores_zero_labels_and_zero_predictions` |
| M3-22 | `test_pooled_metrics_use_combined_rows_not_average_of_folds` |
| M3-23 | `test_prediction_hashes_are_stable_across_row_and_extra_column_order` |
| M3-24 | `test_dataset_mutation_after_planning_is_detected_at_evaluation` |
| M3-25 | `test_append_read_roundtrip_and_entry_hash` |
| M3-26 | `test_genesis_and_multi_entry_hash_chain` |
| M3-27 | `test_tampering_is_detected` |
| M3-28 | `test_malformed_and_noncanonical_files_fail` |
| M3-29 | `test_valid_registration_then_terminal_is_accepted` |
| M3-30 | `test_duplicate_registration_events_fail` |
| M3-31 | `test_duplicate_terminal_events_fail` |
| M3-32 | `test_event_after_terminal_fails`, `test_registration_then_terminal_then_event_fails` |
| M3-33 | `test_lifecycle_identity_changes_fail` |
| M3-34 | `test_terminal_only_is_valid_and_reported_as_unregistered` |
| M3-35 | `test_summary_counts_entries_trials_statuses_experiments_and_head`, `test_empty_ledger_verifies_and_summarizes` |
| M3-36 | `test_append_never_extends_corrupt_ledger` |
| M3-37 | `test_end_to_end_evaluation_round_trips_through_verified_ledger`, `test_failed_evaluation_is_recorded_as_failed_not_completed` |
| M3-38 | Full suite passes: 259 total, including the independently run 165-test pre-Module-3 baseline |
| M3-39 | Ruff passes; scoped mypy passes with `--follow-imports=skip`; the exact specification command fails only on the two pre-existing `fractional.py` errors documented above |
| M3-40 | `test_example_exits_zero_and_claims_no_trading_edge` |

M3-03 traceability note: missing-label and non-ok-label exclusions and their counts are
directly tested. The separate non-finite-label branch is implemented as
defense-in-depth, but Module 2 dataset validation rejects infinite labels and
status/value inconsistencies, so that branch is not directly constructible through the
public dataset constructor and is not claimed as directly test-covered.

M3-39 traceability note: the exact gate command
`python -m mypy quantforge/experiments quantforge/governance/ledger.py --ignore-missing-imports`
does not pass because mypy follows imports into unchanged Module 2 code. The accepted
Module 3-specific evidence is the scoped command:
`--ignore-missing-imports --follow-imports=skip`. This distinction must remain explicit
until the pre-existing errors are separately remediated.

## Deferred

Module 3 does not implement model fitting or tuning, feature selection, CPCV/PBO,
deflated Sharpe ratio, bootstrap/Monte Carlo inference, significance testing,
backtesting, transaction-cost simulation, portfolio construction, live execution,
immutable remote storage, external head-hash publication, multi-writer coordination,
or code attestation. Those require separate specifications, data, infrastructure, and
independent review.
