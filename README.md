# QuantForge PIT — research-grade trading-system foundation

This is a deliberately conservative, modular **starting codebase**, not a claim of a complete or production-certified trading system. It implements core contracts and representative reference components for point-in-time data, feature engineering, supervised modelling, chronological validation, CPCV/PBO diagnostics, walk-forward evaluation, cost-aware event-driven simulation, risk gates, and broker interfaces. It fails closed where market metadata or execution behavior is unknown.

Module 1 (the data layer) is implemented as a provider-neutral, offline-testable component: canonical UTC normalization, PIT as-of joins/rolling features/snapshots, transactional SQLite and staged/rollback-protected Parquet storage (with documented crash limits), raw payload archival, corporate-action adjustment, fixture trading calendars, and a versioned historical universe. It does **not** include licensed vendor data or claim that a fixture calendar/corporate-action set is production-complete.

## Design principles

- **No “zero-error” promise:** financial systems cannot guarantee zero errors. The goal is explicit contracts, tests, immutable evidence, monitoring, and safe failure.
- **Point-in-time first:** records carry both event time and availability time; a record is usable only when `available_at <= decision_time`. Vendors must provide true historical publication/availability timestamps.
- **No survivorship assumptions:** universe membership is time-varying input. This repository does not ship historical delisted-security data; that must come from a licensed, survivorship-aware source.
- **No free fills:** execution simulation requires explicit spread/commission/impact assumptions; bar-only data cannot reproduce queue position or intrabar order-book dynamics.
- **No model selection on the final holdout:** log every trial, freeze the holdout, and report all attempts.
- **No live trading by default:** broker adapter is a protocol, not a real broker integration. Paper/live require separate approval, credentials, kill switch, reconciliation, and operational testing.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
python examples/demo.py
```

## Package map

- `quantforge/data/`: canonical bars, PIT validation, as-of alignment, rolling/snapshots, resampling, Parquet/SQLite storage, raw archive, connectors/ingestion, corporate actions, fixture calendars, and historical universe membership.
- `quantforge/features/`: fractional differentiation and basic factors.
- `quantforge/models/`: sklearn pipeline helper and temporal splitters.
- `quantforge/backtest/`: event-loop reference simulator and cost model.
- `quantforge/optimization/`: parameter grid, DSR diagnostic, CPCV/PBO, walk-forward.
- `quantforge/portfolio/`: constrained signal-to-target sizing.
- `quantforge/governance/`: fail-closed pre-trade risk checks and trial ledger.
- `quantforge/execution/`: broker protocol and dry-run adapter.
- `tests/`: contracts and alignment regressions.
- `docs/ARCHITECTURE.md`: lifecycle, limitations, and deployment gates.

## Canonical data contract

Bars are indexed by `(timestamp, symbol)` with `open, high, low, close, volume, available_at`. `timestamp` means bar end/event time; `available_at` means the earliest time the complete bar was actually available to the strategy. PIT checks are inclusive: `available_at <= decision_timestamp`. Naive or DST-ambiguous timestamps are rejected; canonical storage is UTC. Prices must use a documented adjustment convention. Do not silently splice adjusted and unadjusted prices.

Module 1 stores raw payloads by SHA-256 and persists canonical rows in partitioned Parquet or SQLite. Distinct revisions are retained and selected by PIT availability; same-revision conflicting payloads fail; identical replay is idempotent. Logical checksums exclude `ingestion_timestamp`, so replaying identical data at a different time does not change the fingerprint. Corporate-action adjustments operate on raw bars and add explicit factor/adjusted columns rather than overwriting raw prices. Exchange calendars are explicit fixtures by default; the optional `exchange_calendars` adapter is used only when installed. Historical universe membership, delistings, symbol mappings, and late availability are recorded as versioned PIT inputs; missing coverage is surfaced as incomplete rather than silently reconstructed.

## Module 1 public surface

```python
from quantforge.data import (
    normalize_market_frame, pit_asof_join, pit_rolling, pit_snapshot,
    resample_ohlcv, flag_stale, ParquetStore, SQLiteStore, RawArchive,
    LocalFileConnector, SQLiteConnector, HttpJSONConnector, ingest,
    CorporateActionLedger, adjust_prices, pit_adjusted_prices,
    SimpleExchangeCalendar, UniverseStore,
)
```

Operational notes:

- Use `available_at` from the source, not the bar timestamp, unless you deliberately pass `assume_available_at="event_time"` and record that assumption.
- Storage rejects missing `available_at`, silent metadata invention, invalid revisions, and same-revision conflicts.
- Parquet writes preflight every partition, stage all replacements, and roll back handled commit failures. Atomicity is per file: a process or power loss during the final replace loop can leave a mixed generation; the manifest documents each partition checksum for recovery checks.
- Frames containing multiple frequencies require an explicit `frequency=` selection; joins, rolling features, and resampling do not silently mix bar intervals.
- `ingest(..., raw_archive=...)` preserves raw pages before normalization; provenance includes request, checksums, counts, retries, and errors.
- Corporate-action coverage is explicit. Missing coverage fails when `require_complete=True`; it is never imputed.
- Universe snapshots expose `coverage_status`, late membership records, unknown instruments, and `assert_complete()`.
- The HTTP connector never uses the network unless `allow_network=True`; injected transports are for deterministic tests.

## Scope intentionally not faked

A real FIX/REST integration, tick/order-book replay, licensed historical fundamentals/alternative data, a complete survivorship-aware delisted universe, production exchange calendars, provider-specific corporate-action history, broker reconciliation, portfolio optimizer, RL environment, and live operations cannot be safely fabricated without provider-specific schemas, account constraints, licensing, and acceptance criteria. The included corporate-action ledger, fixture calendars, and universe store are reference components; they are only as complete as the data and coverage declarations supplied by the operator.

## Implemented versus interface-only

Implemented: canonical OHLCV validation; PIT normalization, as-of joins, rolling features, snapshots, resampling, and staleness flags; transactional SQLite and staged/rollback-protected Parquet storage with revisions and conflict detection; SHA-256 raw archival; local Parquet/CSV/SQLite and injectable HTTP ingestion; split/dividend/total-return adjustment with declared coverage; fixture trading calendars/session alignment; versioned instrument, membership, lifecycle, and symbol-mapping queries; causal fractional differentiation; trailing factors; a baseline sklearn regression pipeline; an unsupervised regime-clustering helper; basic purged split geometry; walk-forward windows; CPCV split geometry and a PBO diagnostic over paired IS/OOS matrices; approximate DSR; deterministic parameter enumeration; a single-asset next-bar simulator; approximate square-root impact costs; basic target weights; fail-closed risk checks; an append-only JSONL trial ledger; and a dry-run broker.

Not implemented as a production feature: licensed PIT data acquisition, a complete survivorship-aware delisted universe, provider-complete corporate actions/calendars, tick/LOB replay, multi-asset portfolio accounting, sector constraints, vendor/broker-specific production connectors, FIX/REST order state machines, margin liquidation, live monitoring, RL training/deployment, or an institutionally validated DSR/PBO pipeline. Those require domain-specific contracts, licensed data, and independent review.
