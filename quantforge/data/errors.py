"""Structured exception hierarchy for the data layer.

All data-layer failures derive from :class:`QuantForgeDataError` so callers can
fail closed with one catch-all while still distinguishing operational problems
(connectors, storage) from contract violations (PIT, schema, adjustments).
"""

from __future__ import annotations


class QuantForgeDataError(Exception):
    """Base class for every data-layer error."""


class DataValidationError(QuantForgeDataError, ValueError):
    """A record, frame, or configuration violates a declared contract."""


class TemporalError(DataValidationError):
    """Timestamp is naive, ambiguous, non-monotonic, or otherwise unusable."""


class PointInTimeError(DataValidationError):
    """A future observation would be exposed to an earlier decision."""


class ConflictingRevisionsError(DataValidationError):
    """Two records claim the same identity and revision but different values."""


class IngestionError(QuantForgeDataError):
    """Base class for ingestion failures."""


class ConnectorError(IngestionError):
    """A provider-independent connector failed."""


class NetworkDisabledError(ConnectorError):
    """A network connector was invoked without explicit network permission."""


class RetryExhaustedError(ConnectorError):
    """Bounded retries were exhausted."""


class ProviderUnavailableError(ConnectorError):
    """The provider is unavailable (5xx, outage, timeout)."""


class RateLimitError(ConnectorError):
    """The provider rate-limited the request."""

    retry_after: float | None = None


class PartialResponseError(ConnectorError):
    """Only part of a requested dataset was returned."""


class SchemaValidationError(DataValidationError):
    """Input does not satisfy the expected schema."""


class StorageError(QuantForgeDataError):
    """Storage failed; no silent partial state is implied."""


class StorageConflictError(StorageError):
    """Stored and incoming records conflict for the same identity/revision."""


class CorporateActionError(DataValidationError):
    """Corporate-action data is missing, conflicting, or unsupported."""


class IncompleteCorporateActionDataError(CorporateActionError):
    """Adjustment was requested without declared complete action coverage."""


class DoubleAdjustmentError(CorporateActionError):
    """An already-adjusted frame was passed to an adjustment routine."""


class CalendarError(QuantForgeDataError):
    """Trading-calendar data or a session lookup is invalid."""


class CalendarUnavailableError(CalendarError, ImportError):
    """An optional calendar dependency is not installed."""


class UniverseError(QuantForgeDataError):
    """Universe membership data is invalid or conflicting."""


class IncompleteUniverseError(UniverseError):
    """Complete historical universe coverage was required but is unavailable."""
