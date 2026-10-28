# QuantForge PIT — research-grade trading-system foundation

This is a deliberately conservative, modular **starting codebase**, not a claim of a complete or production-certified trading system. It implements core contracts and representative reference components for point-in-time data, research datasets, feature engineering, labeled outcomes, supervised modelling, chronological validation, CPCV/PBO diagnostics, walk-forward evaluation, cost-aware event-driven simulation, risk gates, and broker interfaces. It fails closed where market metadata or execution behavior is unknown.

Module 1 (the data layer) is implemented as a provider-neutral, offline-testable component: canonical UTC normalization, PIT as-of joins/rolling features/snapshots, transactional SQLite and staged/rollback-protected Parquet storage (with documented crash limits), raw payload archival, corporate-action adjustment, fixture trading calendars, and a versioned historical universe. It does **not** include licensed vendor data or claim that a fixture calendar/corporate-action set is production-complete.

Module 2 (the research dataset layer) adds declarative PIT feature specs, forward-outcome labels with explicit availability, deterministic dataset assembly with canonical hashes, and a forward-only purged/embargoed time split. Its label convention is intentionally **not** claimed to match the existing backtest engine.

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
PYTHONPATH=. python examples/research_pipeline.py
```

## Package map

- `quantforge/data/`: canonical bars, PIT validation, as-of alignment, rolling/snapshots, resampling, Parquet/SQLite storage, raw archive, connectors/ingestion, corporate actions, fixture calendars, and historical universe membership.
- `quantforge/features/`: feature specs, PIT feature computation, fractional differentiation, and basic factors.
- `quantforge/labels.py`: forward-return, excess-return, and adjusted-price labels.
- `quantforge/dataset.py`: feature/label dataset assembly and canonical provenance hashing.
- `quantforge/models/`: sklearn pipeline helper, legacy purged split, and forward-only time split.
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

## Module 2 public surface

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

Operational notes:

- Feature history is filtered by both `timestamp <= decision_timestamp` and `available_at <= decision_timestamp`; future rows, later revisions, and other instruments/frequencies do not leak in. Results expose aggregate `feature_*` metadata and per-feature `__` metadata. `fracdiff_spec` encodes `d` and `threshold` in an identifier-safe default name and rejects invalid values.
- Feature callbacks are trusted Python callables, not a sandbox. They must not read future rows, mutable global state, or unsafe external state.
- Labels enter at the first observed bar strictly after the decision and exit `horizon` observed bars later. Missing entry/exit/price/benchmark/adjustment data stays explicit; the horizon is never shortened.
- `label_execution_convention` is `next_observed_bar_close_to_close` and `backtest_compatibility` is `not_claimed`; do not describe existing backtests as numerically equivalent.
- Adjusted labels require a `CorporateActionLedger`; incomplete coverage fails closed instead of implying missing actions, and actions outside the entry/exit interval are ignored. Excess returns require matching stock/benchmark price conventions; an adjusted benchmark needs its own ledger, and benchmark adjustment provenance is recorded.
- Dataset hashes use type-tagged canonical JSON and SHA-256, including distinct encodings for null, numeric `NaN`, and the string `"NaN"`. Required `manifest_hash` and `dataset_id` roots bind feature, label, universe, source, and quality metadata. Duplicate canonical keys are rejected. Feature, label, universe, and dataset content/source hashes are computed over the assembled dataset rows and can be recomputed from the frame.
- `validate_dataset` requires a non-empty manifest mapping with non-empty `schema_version`, `manifest_hash`, and `dataset_id`, plus quality matching `manifest["quality"]`; it checks required columns, keys, statuses, manifest hashes, and filtered-frame keys. It recomputes content/source hashes from the current frame, so a mutated frame, stale manifest, missing hash, or malformed root fails closed. `ResearchDataset` construction invokes it. Raw feature/label timestamps and status/value consistency are validated during assembly.
- `ResearchDataset` stores a deep copy of its input frame and is frozen, but pandas DataFrames remain mutable. Construction requires a complete manifest and matching quality. Treat `dataset.frame` as immutable by contract and call `verify_integrity()` or `validate_dataset()` after any suspected mutation.
- Universe membership is optional. Unknown membership is surfaced as `unknown_membership`, not universal inclusion; membership `available_at` is checked when present, and filtering must be explicit. The universe source hash includes eligibility, availability, and membership status; the dataset frame exposes the corresponding universe provenance columns.
- `purged_time_split` keeps training strictly before the test start, purges labels that touch the test or embargo boundary, reports post-test rows as unused, and returns chronological train/test indices. Label starts before the sample timestamp are rejected.

See [`docs/MILESTONE_2.md`](docs/MILESTONE_2.md) for the full contract and limitations.

## Scope intentionally not faked

A real FIX/REST integration, tick/order-book replay, licensed historical fundamentals/alternative data, a complete survivorship-aware delisted universe, production exchange calendars, provider-specific corporate-action history, broker reconciliation, portfolio optimizer, RL environment, and live operations cannot be safely fabricated without provider-specific schemas, account constraints, licensing, and acceptance criteria. The included corporate-action ledger, fixture calendars, and universe store are reference components; they are only as complete as the data and coverage declarations supplied by the operator.

## Implemented versus interface-only

Implemented: canonical OHLCV validation; PIT normalization, as-of joins, rolling features, snapshots, resampling, and staleness flags; transactional SQLite and staged/rollback-protected Parquet storage with revisions and conflict detection; SHA-256 raw archival; local Parquet/CSV/SQLite and injectable HTTP ingestion; split/dividend/total-return adjustment with declared coverage; fixture trading calendars/session alignment; versioned instrument, membership, lifecycle, and symbol-mapping queries; declarative PIT feature specifications and computation; forward-return and excess-return labels with explicit availability; deterministic feature/label dataset assembly and canonical hashing; forward-only purged/embargoed time splitting; causal fractional differentiation; trailing factors; a baseline sklearn regression pipeline; an unsupervised regime-clustering helper; legacy basic purged split geometry; walk-forward windows; CPCV split geometry and a PBO diagnostic over paired IS/OOS matrices; approximate DSR; deterministic parameter enumeration; a single-asset next-bar simulator; approximate square-root impact costs; basic target weights; fail-closed risk checks; an append-only JSONL trial ledger; and a dry-run broker.

Not implemented as a production feature: licensed PIT data acquisition, a complete survivorship-aware delisted universe, provider-complete corporate actions/calendars, tick/LOB replay, multi-asset portfolio accounting, sector constraints, vendor/broker-specific production connectors, FIX/REST order state machines, margin liquidation, live monitoring, a feature store or dataset persistence layer, backtest/label execution equivalence, RL training/deployment, or an institutionally validated DSR/PBO pipeline. Those require domain-specific contracts, licensed data, and independent review.
