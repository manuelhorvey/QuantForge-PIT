# QuantForge PIT — research-grade trading-system foundation

This is a deliberately conservative, modular **starting codebase**, not a claim of a complete or production-certified trading system. It implements core contracts and representative reference components for point-in-time data, feature engineering, supervised modelling, chronological validation, CPCV/PBO diagnostics, walk-forward evaluation, cost-aware event-driven simulation, risk gates, and broker interfaces. It fails closed where market metadata or execution behavior is unknown.

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

- `quantforge/data/`: canonical bars, PIT validation, as-of alignment.
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

Bars are indexed by `(timestamp, symbol)` with `open, high, low, close, volume, available_at`. `timestamp` means bar end/event time; `available_at` means the earliest time the complete bar was actually available to the strategy. Prices must use a documented adjustment convention. Do not silently splice adjusted and unadjusted prices. Corporate actions, symbol changes, delistings, trading calendars, futures rolls, FX rollover, and venue-specific sessions belong in source-specific adapters.

## Scope intentionally not faked

A real FIX/REST integration, tick/order-book replay, corporate-action ledger, historical fundamentals/alternative data, delisted universe, exchange calendars, broker reconciliation, portfolio optimizer, RL environment, and live operations cannot be safely fabricated without provider-specific schemas, account constraints, and acceptance criteria. Interfaces and gates should be extended against those actual dependencies.

## Implemented versus interface-only

Implemented: canonical OHLCV validation, PIT as-of joins, causal fractional differentiation, trailing factors, a baseline sklearn regression pipeline, an unsupervised regime-clustering helper, basic purged split geometry, walk-forward windows, CPCV split geometry and a PBO diagnostic over paired IS/OOS matrices, approximate DSR, deterministic parameter enumeration, a single-asset next-bar simulator, approximate square-root impact costs, basic target weights, fail-closed risk checks, an append-only JSONL trial ledger, and a dry-run broker.

Not implemented as a production feature: RL training/deployment, complete tick/LOB event replay, multi-asset portfolio accounting, sector constraints, corporate-action processing, historical constituent reconstruction, vendor/broker connectors, FIX/REST order state machines, margin liquidation, live monitoring, or an institutionally validated DSR/PBO pipeline. Those require domain-specific contracts and independent review.
