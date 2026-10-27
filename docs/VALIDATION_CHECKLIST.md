# Validation and deployment checklist

## Data and PIT
- [ ] Vendor contract and field semantics documented; raw payloads checksummed and versioned.
- [ ] Every record has event time, first-available time, source, revision/vintage, timezone, and units.
- [ ] Corporate actions, delisted assets, historical membership, symbol changes, futures rolls, calendars, and session boundaries are verified.
- [ ] Late, corrected, duplicated, missing, stale, and out-of-order records tested.
- [ ] Feature availability timestamp never exceeds decision timestamp.

## Research integrity
- [ ] Hypothesis, primary metric, risk limits, feature set, model family, search space, and stopping rule preregistered.
- [ ] Full trial count includes failed/abandoned configurations and human-guided iteration.
- [ ] Train-only preprocessing; purging/embargo match each label horizon.
- [ ] Final holdout has not been used for feature or strategy design.
- [ ] Walk-forward and CPCV/PBO are computed from valid split-specific in-sample/out-of-sample outcomes.
- [ ] DSR, bootstrap/Monte Carlo, parameter stability, and regime/asset slices reported with uncertainty.

## Execution and risk
- [ ] Tick/contract size, point value, lot size, quote currency, FX conversion, fees, spread, financing, borrow, and margin are correct.
- [ ] Cost model calibrated by instrument, venue, order size, time-of-day, volatility, and liquidity.
- [ ] Order state machine handles partial fills, rejects, cancels, timeouts, retries, duplicate messages, and broker restarts.
- [ ] Reconciliation of cash, positions, orders, and fills tested under injected failures.
- [ ] Kill switch, stale-data halt, loss limits, leverage limits, allowlists, and manual emergency procedure tested.
- [ ] Paper shadow period includes diverse regimes and enough trade/event coverage; acceptance thresholds are predeclared.
- [ ] Canary deployment and rollback approved; secrets are external to code; immutable audit logs and alerts are active.

Passing unit tests only verifies local code contracts. It does not establish alpha, statistical validity, execution fidelity, or readiness for live capital.

## Module 1 verification evidence

Automated checks run locally from the repository root:

```bash
python -m pytest -q
python -m ruff check quantforge/data tests/test_data.py
python -m mypy quantforge/data tests/test_data.py --ignore-missing-imports
```

- [x] Full test suite passes (96 tests; existing baseline plus Module 1 regression coverage).
- [x] Module 1 lint passes.
- [x] Module 1 type-check passes.
- [x] PIT normalization rejects naive/ambiguous timestamps, missing symbols, non-finite/non-positive OHLCV, invalid envelopes, availability before event, and ingestion before availability.
- [x] As-of joins, rolling features, and snapshots enforce inclusive `available_at <= decision_timestamp`.
- [x] Parquet/SQLite storage round-trips, preserves UTC, retains revisions, rejects same-revision conflicts, reports idempotent replay, and stages/rolls back handled multi-partition failures.
- [x] Raw archive is SHA-256 content-addressed, verifies payloads, and redacts nested secret metadata.
- [x] Local Parquet/CSV ingestion and raw archival are exercised end to end; HTTP pagination/retries use an injected transport (no network).
- [x] Corporate-action coverage gaps, double adjustment, conflicting actions, and dividend-reference gaps fail explicitly.
- [x] Calendar fixtures cover holidays, DST ambiguity, early closes, session alignment, and missing sessions; optional adapter fails explicitly when absent.
- [x] Universe snapshots preserve delisted names, enforce PIT symbol mappings, expose late/unknown coverage, and fail closed when completeness is required.

Not established by these checks: licensed PIT data correctness, complete historical delisted-universe coverage, production exchange calendars/corporate actions, provider schema compatibility, or readiness for live capital. Those require licensed data, provider acceptance tests, and independent review.
