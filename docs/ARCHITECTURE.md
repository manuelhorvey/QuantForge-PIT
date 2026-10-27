# Architecture and operating controls

## Research-to-live lifecycle

1. **Acquire and version** raw vendor payloads, schema versions, ingestion timestamps, event timestamps, availability timestamps, symbol mappings, and checksums. Preserve raw inputs; corrections create new versions rather than overwriting history.
2. **Normalize** to canonical UTC timestamps and instrument identifiers. Keep exchange calendars, corporate actions, dividends, splits, futures rolls, funding/rollover, and trading status as explicit event streams. Historical equity universes must include dead/delisted names; this project does not supply that data.
3. **Enforce PIT** before feature computation. Fundamentals use publication/availability timestamps and restatement vintages, not fiscal-period end dates. Alternative data use first-observed timestamps and provider latency. `asof_join` is keyed and backward-only.
4. **Build features** with trailing windows only. Fit transformations/imputers/scalers on training windows only. Fractional differentiation is causal; its weight cutoff and order must be preregistered.
5. **Label** with explicit start/end timestamps. When labels overlap, cross-validation must purge training samples whose label intervals intersect test information intervals and apply an embargo. Never use shuffled K-fold for serial financial observations.
6. **Research and log** all hypotheses and trials before seeing holdout results. Track data/code/config hashes, trial count, selection criteria, failures, and researcher decisions. The ledger is append-only but should additionally be stored in immutable/WORM storage in production.
7. **Validate** using chronological walk-forward, purged validation, CPCV/PBO where applicable, bootstrap/Monte Carlo, parameter stability, regime slices, costs, capacity, and independent final holdout. These are complementary diagnostics, not guarantees.
8. **Simulate execution** with spread, fees, latency, partial fills, borrow/funding, market hours, price limits, impact, and instrument multipliers. OHLCV bars cannot accurately model queue position or intrabar stop/limit ordering. Use L2/tick replay when the strategy requires it.
9. **Portfolio and risk** apply limits after target generation and again before order submission. Risk limits, symbol allowlists, data freshness, broker state reconciliation, and kill-switch state must be checked independently.
10. **Deploy gradually**: offline research → out-of-sample report → paper with shadow orders → controlled canary → capped capital → monitored scale. Every transition needs a signed approval record and rollback plan.

## Safety invariants

- `available_at <= decision_timestamp` for every feature input.
- Decisions at bar close execute no earlier than the next tradable event unless the data and execution protocol prove otherwise.
- Model and scaler fit data never include validation/test observations.
- Order quantity is positive; side carries direction.
- Risk checks fail closed on missing, stale, malformed, or inconsistent inputs.
- Reconcile orders, fills, positions, cash, margin, and fees against broker records.
- No automated capital increase based only on Sharpe or a single backtest.

## Statistical notes

- DSR depends on trial count, sample length, skewness, kurtosis, and assumptions about dependence across trials. The included function is an approximate diagnostic, not a definitive proof of edge.
- PBO requires a valid matrix of in-sample-selected strategies and their out-of-sample performance across symmetric splits. The provided split geometry alone is not enough; purge/embargo label intervals and construct the performance matrix carefully.
- Walk-forward tests mimic periodic re-estimation but do not eliminate researcher degrees of freedom or structural breaks.
- Costs and impact coefficients must be estimated from actual instrument, venue, size, and time-of-day data. If calibration is absent, results are illustrative only.

## Production gaps before live use

Provider-specific connectors and data licensing; historical PIT corporate actions/fundamentals/delisted universe; exchange calendars; robust tick/LOB replay; full portfolio accounting; margin/borrow/funding; broker adapter and order reconciliation; secrets management; idempotency; alerting; durable audit log; disaster recovery; and a formal model-risk approval process. Do not connect live credentials to this scaffold.
