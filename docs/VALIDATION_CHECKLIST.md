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
