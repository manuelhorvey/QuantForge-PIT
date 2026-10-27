# Architecture and operating controls

## Research-to-live lifecycle

1. **Acquire and version** raw vendor payloads, schema versions, ingestion timestamps, event timestamps, availability timestamps, symbol mappings, and checksums. Module 1 provides `RawArchive` (content-addressed, redacted sidecars), local CSV/Parquet/SQLite connectors, and an injectable HTTP connector. Preserve raw inputs; corrections create new revisions rather than overwriting history.
2. **Normalize** to canonical UTC timestamps and instrument identifiers. `normalize_market_frame` rejects naive/ambiguous timestamps, missing OHLCV, non-finite values, non-positive prices, invalid envelopes, missing symbols, and unavailable rows instead of repairing them silently. Keep exchange calendars, corporate actions, dividends, splits, futures rolls, funding/rollover, and trading status as explicit event streams. `CorporateActionLedger`, fixture calendars, and `UniverseStore` implement reference versions; historical equity universes must include dead/delisted names, and this project does not supply licensed historical membership data.
3. **Enforce PIT** before feature computation. Fundamentals use publication/availability timestamps and restatement vintages, not fiscal-period end dates. Alternative data use first-observed timestamps and provider latency. Module 1 provides keyed backward-only `pit_asof_join`, per-decision `pit_rolling`, validity-aware `pit_snapshot`, and `flag_stale`; all use inclusive availability by default. `ingest` records provenance, raw checksums, normalized checksums, rejected-row diagnostics, retries, and storage receipts.
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

## Module 1 data controls

- **Storage:** partitioned Parquet with staged multi-partition replacement and rollback on handled failures; SQLite uses transactions/WAL. Atomicity is per Parquet file, so an abrupt process/power loss during the final replace loop can leave a mixed generation. Both retain revisions, reject same-revision conflicts, and report idempotent replay explicitly; logical checksums exclude `ingestion_timestamp`. `raw_archive` preserves source bytes before validation.
- **Revisions and frequency:** normalization retains distinct vintages; as-of joins and rolling features select the highest revision available at the decision time, not the latest revision in the file. Same-revision conflicting payloads fail closed. Frames with multiple frequencies require an explicit `frequency=` selection.
- **Corporate actions:** split, reverse-split, cash-dividend, and total-return conventions add `adj_factor`, `volume_factor`, and `adjusted_*` columns. Coverage is declared per ledger/instrument; missing or conflicting actions raise when completeness is required.
- **Calendars:** `SimpleExchangeCalendar` is an explicit fixture with holidays, early closes, DST ambiguity checks, and session alignment. The optional `exchange_calendars` adapter fails with installation guidance when absent.
- **Universe:** instrument metadata, lifecycle events, symbol mappings, and membership are versioned and availability-filtered. Snapshots expose incomplete/late/unknown coverage and can fail closed via `require_complete=True` or `assert_complete()`.
- **Operational limits:** no vendor credentials are stored in code or provenance; HTTP is disabled unless `allow_network=True`; tests use injected transports and local files only.

## Production gaps before live use

Provider-specific connectors and data licensing; historical PIT corporate actions/fundamentals/delisted universe sourced from a licensed provider; production exchange-calendar coverage; robust tick/LOB replay; full portfolio accounting; margin/borrow/funding; broker adapter and order reconciliation; secrets management; deployment-level idempotency; alerting; durable audit log; disaster recovery; and a formal model-risk approval process. Do not connect live credentials to this scaffold.
