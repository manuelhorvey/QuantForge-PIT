# Module 1 — Data Layer Milestone

## Status

Accepted with non-blocking issues.

Module 1 provides canonical PIT normalization, as-of joins, rolling features, snapshots, resampling, Parquet/SQLite storage, raw archival, local/HTTP ingestion, corporate-action adjustment, fixture calendars, and a versioned historical universe.

## Verification

- Full suite: 96 passed.
- Module 1 Ruff: clean.
- Module 1 mypy: clean.
- Repository-wide Ruff/mypy failures outside Module 1 remain pre-existing and are not addressed by this milestone.

## Parquet crash-consistency limitation

Parquet writes preflight all partitions, stage replacements, and roll back handled commit failures. Atomicity is per file, not crash-atomic across a multi-partition set: an abrupt process or power loss during the final replace loop can leave a mixed generation. The store manifest records per-partition checksums for recovery checks.

This milestone does not claim crash-atomic multi-partition transactions.

## Deferred

Licensed PIT data, production exchange calendars and corporate actions, complete survivorship-aware delisted-universe coverage, provider-specific production connectors, and live-capital readiness require external data, provider acceptance tests, and independent review.
