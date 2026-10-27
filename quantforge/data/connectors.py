"""Provider-independent ingestion connectors and orchestration.

The only fully exercised external path in this repository is **local files**
(Parquet/CSV) and **local SQLite**. The HTTP connector is real code with an
injectable transport so pagination, retries, rate limits, and provider failures
can be tested deterministically without network access. No commercial/vendor API
is claimed to be integrated.

Credentials are supplied through an external provider/callable and are never
written to logs, request provenance, stored metadata, or raw-data sidecars.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

import pandas as pd

from .errors import (
    ConnectorError,
    IngestionError,
    NetworkDisabledError,
    PartialResponseError,
    ProviderUnavailableError,
    RateLimitError,
    RetryExhaustedError,
    SchemaValidationError,
)
from .pipeline import normalize_market_frame
from .quality import QualityReport
from .storage import RawArchive, RawArchiveReceipt, StorageReceipt, frame_fingerprint
from .timeutils import ensure_utc, utc_now

_LOG = logging.getLogger("quantforge.data.connectors")


class IngestionStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True)
class ConnectorConfig:
    source_name: str
    allow_network: bool = False
    timeout_seconds: float = 10.0
    max_retries: int = 3
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0
    rate_limit_per_second: float | None = None
    page_size: int = 5000
    max_pages: int = 1000
    user_agent: str = "quantforge-pit/0.1"

    def __post_init__(self) -> None:
        if not self.source_name:
            raise ValueError("source_name is required")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.backoff_base_seconds < 0 or self.backoff_max_seconds < 0:
            raise ValueError("backoff values cannot be negative")
        if self.backoff_max_seconds < self.backoff_base_seconds:
            raise ValueError("backoff_max_seconds must be >= backoff_base_seconds")
        if self.rate_limit_per_second is not None and self.rate_limit_per_second <= 0:
            raise ValueError("rate_limit_per_second must be positive or None")
        if self.page_size <= 0 or self.max_pages <= 0:
            raise ValueError("page_size and max_pages must be positive")

    @property
    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_attempts=self.max_retries + 1,
            base_delay=self.backoff_base_seconds,
            max_delay=self.backoff_max_seconds,
        )


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0 or self.jitter < 0:
            raise ValueError("retry delays cannot be negative")

    def delay_for(self, attempt: int, *, retry_after: float | None = None) -> float:
        if retry_after is not None and retry_after >= 0:
            return min(self.max_delay, float(retry_after))
        return min(self.max_delay, self.base_delay * (2 ** max(attempt - 1, 0)))


@dataclass(frozen=True)
class DataRequest:
    """Provider-neutral request. Contains no credentials."""

    instruments: tuple[str, ...] = ()
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    frequency: str = "1d"
    fields: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    dataset: str = "bars"

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruments", tuple(str(i) for i in self.instruments))
        object.__setattr__(self, "fields", tuple(str(f) for f in self.fields))
        if self.start is not None:
            object.__setattr__(self, "start", ensure_utc(self.start, name="start"))
        if self.end is not None:
            object.__setattr__(self, "end", ensure_utc(self.end, name="end"))
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("start must be <= end")
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "instruments": list(self.instruments),
            "start": None if self.start is None else self.start.isoformat(),
            "end": None if self.end is None else self.end.isoformat(),
            "frequency": self.frequency,
            "fields": list(self.fields),
            "params": {str(k): str(v) for k, v in sorted(self.params.items())},
            "dataset": self.dataset,
        }


@dataclass(frozen=True)
class DataBatch:
    records: pd.DataFrame
    raw_payload: bytes | None = None
    raw_name: str | None = None
    page_token: str | None = None


@dataclass(frozen=True)
class SourceProvenance:
    source: str
    dataset: str
    schema_version: str
    retrieved_at: str
    request: Mapping[str, Any]
    pages: int
    retries: int
    input_records: int
    accepted_records: int
    rejected_records: int
    checksum: str | None
    raw_sha256: tuple[str, ...] = ()
    endpoint: str | None = None
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "dataset": self.dataset,
            "schema_version": self.schema_version,
            "retrieved_at": self.retrieved_at,
            "request": dict(self.request),
            "pages": self.pages,
            "retries": self.retries,
            "input_records": self.input_records,
            "accepted_records": self.accepted_records,
            "rejected_records": self.rejected_records,
            "checksum": self.checksum,
            "raw_sha256": list(self.raw_sha256),
            "endpoint": self.endpoint,
            "errors": list(self.errors),
        }


@dataclass
class IngestionResult:
    status: IngestionStatus
    accepted: pd.DataFrame
    rejected: pd.DataFrame
    report: QualityReport
    provenance: SourceProvenance
    storage_receipt: StorageReceipt | None = None
    raw_receipts: tuple[RawArchiveReceipt, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is IngestionStatus.OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "report": self.report.to_dict(),
            "provenance": self.provenance.to_dict(),
            "storage_receipt": None if self.storage_receipt is None else self.storage_receipt.to_dict(),
            "raw_receipts": [r.to_dict() for r in self.raw_receipts],
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class IncrementalPolicy:
    enabled: bool = True
    overlap: pd.Timedelta = pd.Timedelta(days=1)

    def __post_init__(self) -> None:
        if pd.Timedelta(self.overlap) < pd.Timedelta(0):
            raise ValueError("overlap cannot be negative")


class DataSource(Protocol):
    name: str
    config: ConnectorConfig

    def fetch(self, request: DataRequest) -> Iterable[DataBatch]:
        ...


class _RateLimiter:
    def __init__(
        self,
        rate_per_second: float | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = 0.0 if rate_per_second is None else 1.0 / rate_per_second
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self.interval <= 0:
            return
        if self._last is not None:
            remaining = self.interval - (self._clock() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()


class BaseConnector:
    """Shared retry, rate-limit, and structured logging behavior."""

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.name = config.source_name
        self._sleep = sleep
        self._clock = clock
        self._limiter = _RateLimiter(
            config.rate_limit_per_second, clock=clock, sleep=sleep
        )
        self.retry_attempts = 0

    def _log(self, level: int, event: str, **fields: Any) -> None:
        _LOG.log(level, event, extra={"event": event, **fields})

    def _call_with_retries(self, operation: Callable[[], Any], *, description: str) -> Any:
        policy = self.config.retry_policy
        last_error: Exception | None = None
        for attempt in range(1, policy.max_attempts + 1):
            self._limiter.wait()
            try:
                return operation()
            except (RateLimitError, ProviderUnavailableError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                if attempt >= policy.max_attempts:
                    break
                retry_after = getattr(exc, "retry_after", None)
                delay = policy.delay_for(attempt, retry_after=retry_after)
                self.retry_attempts += 1
                self._log(
                    logging.WARNING,
                    "connector.retry",
                    source=self.name,
                    description=description,
                    attempt=attempt,
                    max_attempts=policy.max_attempts,
                    delay_seconds=delay,
                    error=type(exc).__name__,
                )
                self._sleep(delay)
        raise RetryExhaustedError(
            f"{self.name}: {description} failed after {policy.max_attempts} attempt(s): {last_error}"
        ) from last_error


class LocalFileConnector(BaseConnector):
    """Functional local Parquet/CSV ingestion path (no network)."""

    def __init__(
        self,
        config: ConnectorConfig,
        root_dir: str | Path,
        *,
        file_format: str = "auto",
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(config, sleep=sleep, clock=clock)
        self.root = Path(root_dir)
        if not self.root.exists():
            raise ConnectorError(f"Local data root does not exist: {self.root}")
        if file_format not in {"auto", "parquet", "csv"}:
            raise ValueError("file_format must be auto, parquet, or csv")
        self.file_format = file_format

    @staticmethod
    def _component(value: str, *, label: str) -> str:
        text = str(value)
        if (
            not text
            or text in {".", ".."}
            or Path(text).is_absolute()
            or "/" in text
            or "\\" in text
        ):
            raise SchemaValidationError(f"Invalid local data {label}: {value!r}")
        return text

    def _paths(self, symbol: str, frequency: str) -> list[Path]:
        symbol = self._component(symbol, label="symbol")
        frequency = self._component(frequency, label="frequency")
        base = self.root / frequency
        patterns = [
            base / f"{symbol}.parquet",
            base / symbol / "*.parquet",
            base / f"{symbol}.csv",
            base / symbol / "*.csv",
            self.root / f"{symbol}.parquet",
            self.root / symbol / "*.parquet",
        ]
        found: list[Path] = []
        for pattern in patterns:
            if "*" in str(pattern):
                found.extend(sorted(pattern.parent.glob(pattern.name)))
            elif pattern.exists():
                found.append(pattern)
        unique = sorted({p for p in found if p.is_file()})
        if self.file_format != "auto":
            unique = [p for p in unique if p.suffix.lstrip(".") == self.file_format]
        root = self.root.resolve()
        for path in unique:
            try:
                path.resolve().relative_to(root)
            except ValueError as exc:
                raise ConnectorError(f"Local data path escapes root: {path}") from exc
        return unique

    def _read(self, path: Path) -> pd.DataFrame:
        try:
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            if path.suffix == ".csv":
                return pd.read_csv(path)
        except Exception as exc:
            raise ConnectorError(f"Failed to read local file {path}: {exc}") from exc
        raise ConnectorError(f"Unsupported file format: {path.suffix}")

    def fetch(self, request: DataRequest) -> Iterator[DataBatch]:
        self.retry_attempts = 0
        self._log(logging.INFO, "connector.fetch.start", source=self.name, request=request.to_public_dict())
        missing: list[str] = []
        for symbol in request.instruments or ():
            paths = self._paths(symbol, request.frequency)
            if not paths:
                missing.append(symbol)
                continue
            for path in paths:
                frame = self._read(path)
                if "timestamp" in frame.columns and frame["timestamp"].notna().any():
                    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
                    in_range = pd.Series(True, index=frame.index)
                    if request.start is not None:
                        in_range &= timestamps >= request.start
                    if request.end is not None:
                        in_range &= timestamps <= request.end
                    frame = frame.loc[timestamps.isna() | in_range].copy()
                if frame.empty:
                    continue
                yield DataBatch(
                    records=frame.reset_index(drop=True),
                    raw_payload=path.read_bytes(),
                    raw_name=str(path),
                )
        if missing:
            message = f"{self.name}: missing instruments in local store: {sorted(missing)}"
            self._log(logging.WARNING, "connector.fetch.partial", source=self.name, missing=missing)
            raise PartialResponseError(message)
        self._log(logging.INFO, "connector.fetch.complete", source=self.name)


class SQLiteConnector(BaseConnector):
    """Read bars from a local SQLite database (no network)."""

    def __init__(
        self,
        config: ConnectorConfig,
        path: str | Path,
        *,
        table: str = "bars",
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(config, sleep=sleep, clock=clock)
        self.path = Path(path)
        if not self.path.exists():
            raise ConnectorError(f"SQLite database does not exist: {self.path}")
        if not table.replace("_", "").isalnum() or not table[0].isalpha():
            raise ValueError("table must be a simple SQL identifier")
        self.table = table

    def fetch(self, request: DataRequest) -> Iterator[DataBatch]:
        import sqlite3

        self.retry_attempts = 0
        try:
            connection = sqlite3.connect(self.path)
        except sqlite3.Error as exc:
            raise ConnectorError(f"Failed to open {self.path}: {exc}") from exc
        try:
            connection.row_factory = sqlite3.Row
            clauses = ["frequency = ?"]
            params: list[Any] = [request.frequency]
            if request.instruments:
                clauses.append(
                    "symbol IN (" + ",".join("?" * len(request.instruments)) + ")"
                )
                params.extend(request.instruments)
            if request.start is not None:
                clauses.append("timestamp >= ?")
                params.append(request.start.isoformat())
            if request.end is not None:
                clauses.append("timestamp <= ?")
                params.append(request.end.isoformat())
            query = f"SELECT * FROM {self.table} WHERE " + " AND ".join(clauses)
            try:
                rows = [dict(row) for row in connection.execute(query, params)]
            except sqlite3.Error as exc:
                raise ConnectorError(f"SQLite query failed: {exc}") from exc
        finally:
            connection.close()
        found = {str(row.get("symbol")) for row in rows}
        missing = sorted(set(request.instruments) - found)
        if rows:
            yield DataBatch(records=pd.DataFrame(rows), raw_name=f"sqlite://{self.path}")
        if missing:
            raise PartialResponseError(f"{self.name}: missing instruments: {missing}")


@dataclass(frozen=True)
class HttpRequest:
    url: str
    method: str = "GET"
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 10.0
    body: bytes | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)


def urllib_transport(request: HttpRequest, *, config: ConnectorConfig) -> HttpResponse:
    """Default HTTP transport. Requires explicit network permission."""

    if not config.allow_network:
        raise NetworkDisabledError(
            f"{config.source_name}: network access disabled; pass allow_network=True explicitly"
        )
    http_request = urllib.request.Request(
        request.url,
        method=request.method,
        headers={**dict(request.headers), "User-Agent": config.user_agent},
        data=request.body,
    )
    try:
        with urllib.request.urlopen(http_request, timeout=request.timeout_seconds) as response:
            return HttpResponse(
                status=int(response.status),
                body=response.read(),
                headers={k.lower(): v for k, v in response.headers.items()},
            )
    except urllib.error.HTTPError as exc:
        body = exc.read()
        if exc.code == 429:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            error = RateLimitError(f"{config.source_name}: HTTP 429")
            if retry_after:
                try:
                    error.retry_after = float(retry_after)
                except ValueError:
                    pass
            raise error from exc
        if 500 <= exc.code < 600:
            raise ProviderUnavailableError(f"{config.source_name}: HTTP {exc.code}") from exc
        raise ConnectorError(f"{config.source_name}: HTTP {exc.code}: {body[:200]!r}") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise ProviderUnavailableError(f"{config.source_name}: transport failure: {exc}") from exc


class HttpJSONConnector(BaseConnector):
    """Paginated JSON-over-HTTP connector with injectable transport.

    ``transport`` receives an :class:`HttpRequest` and returns an
    :class:`HttpResponse`; the default transport is :func:`urllib_transport`.
    Tests inject a deterministic fake, so the suite never touches the network.
    """

    def __init__(
        self,
        config: ConnectorConfig,
        base_url: str,
        *,
        transport: Callable[[HttpRequest], HttpResponse] | None = None,
        headers_provider: Callable[[], Mapping[str, str]] | None = None,
        records_field: str = "records",
        next_token_field: str = "next_page_token",
        page_token_param: str = "page_token",
        page_size_param: str = "limit",
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(config, sleep=sleep, clock=clock)
        self.base_url = base_url
        self.transport = transport or (lambda request: urllib_transport(request, config=config))
        self.headers_provider = headers_provider
        self.records_field = records_field
        self.next_token_field = next_token_field
        self.page_token_param = page_token_param
        self.page_size_param = page_size_param

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.headers_provider is not None:
            headers.update(dict(self.headers_provider()))
        return headers

    def fetch(self, request: DataRequest) -> Iterator[DataBatch]:
        self.retry_attempts = 0
        page_token: str | None = None
        seen_tokens: set[str] = set()
        for page in range(self.config.max_pages):
            params: dict[str, Any] = dict(request.params)
            if request.instruments:
                params["symbols"] = ",".join(request.instruments)
            if request.start is not None:
                params["start"] = request.start.isoformat()
            if request.end is not None:
                params["end"] = request.end.isoformat()
            params["frequency"] = request.frequency
            if request.fields:
                params["fields"] = ",".join(request.fields)
            params[self.page_size_param] = self.config.page_size
            if page_token:
                params[self.page_token_param] = page_token
            url = self.base_url + ("&" if "?" in self.base_url else "?") + urllib.parse.urlencode(
                params, doseq=True
            )
            http_request = HttpRequest(
                url=url,
                headers=self._headers(),
                timeout_seconds=self.config.timeout_seconds,
            )

            def _fetch_once(request_spec: HttpRequest = http_request) -> HttpResponse:
                response = self.transport(request_spec)
                if response.status == 429:
                    retry_after_header = response.headers.get("retry-after")
                    error = RateLimitError(f"{self.name}: HTTP 429")
                    if retry_after_header:
                        try:
                            error.retry_after = float(retry_after_header)
                        except ValueError:
                            pass
                    raise error
                if 500 <= response.status < 600:
                    raise ProviderUnavailableError(
                        f"{self.name}: provider unavailable ({response.status})"
                    )
                return response

            response = self._call_with_retries(
                _fetch_once, description=f"fetch page {page + 1}"
            )
            if response.status != 200:
                raise ConnectorError(f"{self.name}: unexpected HTTP {response.status}")
            try:
                payload = json.loads(response.body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ConnectorError(f"{self.name}: invalid JSON payload") from exc
            if isinstance(payload, dict):
                records = payload.get(self.records_field, [])
            else:
                records = payload
            if not isinstance(records, list):
                raise SchemaValidationError(f"{self.name}: records must be a JSON list")
            frame = pd.DataFrame(records)
            yield DataBatch(
                records=frame,
                raw_payload=response.body,
                raw_name=f"page-{page}.json",
                page_token=page_token,
            )
            next_token = payload.get(self.next_token_field) if isinstance(payload, dict) else None
            if not next_token:
                self._log(
                    logging.INFO,
                    "connector.fetch.complete",
                    source=self.name,
                    pages=page + 1,
                    retries=self.retry_attempts,
                )
                return
            if next_token in seen_tokens:
                raise ConnectorError(f"{self.name}: provider repeated page token")
            seen_tokens.add(str(next_token))
            page_token = str(next_token)
        raise ConnectorError(f"{self.name}: exceeded max_pages={self.config.max_pages}")


def _frames(batches: Sequence[DataBatch]) -> pd.DataFrame:
    frames = [batch.records for batch in batches if isinstance(batch.records, pd.DataFrame) and not batch.records.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def ingest(
    source: DataSource,
    request: DataRequest,
    *,
    store: Any,
    raw_archive: RawArchive | None = None,
    frequency: str = "1d",
    schema_version: str = "1",
    incremental: IncrementalPolicy | None = None,
    ingestion_timestamp: Any | None = None,
) -> IngestionResult:
    """Fetch, validate, persist, and report one ingestion run.

    The store write is one logical commit from the pipeline's perspective;
    Parquet replacements are staged and rolled back on handled commit failures
    (atomicity is per file, not crash-atomic across partitions). Any
    connector/partial failure is surfaced in ``status``/``errors`` while
    successfully fetched rows are still validated and persisted (never silently
    discarded).
    """

    started = ensure_utc(ingestion_timestamp) if ingestion_timestamp is not None else utc_now()
    effective_request = request
    policy = incremental or IncrementalPolicy()
    if policy.enabled and hasattr(store, "max_timestamp"):
        watermark = store.max_timestamp(frequency=frequency)
        if watermark is not None:
            proposed_start = watermark - pd.Timedelta(policy.overlap)
            if request.start is None or proposed_start > request.start:
                effective_request = replace(request, start=proposed_start)

    batches: list[DataBatch] = []
    raw_receipts: list[RawArchiveReceipt] = []
    errors: list[str] = []
    partial = False
    failed = False

    _LOG.info(
        "ingestion.start",
        extra={
            "event": "ingestion.start",
            "source": source.name,
            "request": effective_request.to_public_dict(),
        },
    )
    try:
        for batch in source.fetch(effective_request):
            batches.append(batch)
            if raw_archive is not None and batch.raw_payload is not None:
                suffix = Path(batch.raw_name or "").suffix.lstrip(".") or "bin"
                raw_receipt = raw_archive.write(
                    batch.raw_payload,
                    source=source.name,
                    extension=suffix,
                    received_at=started,
                    metadata={
                        "request": effective_request.to_public_dict(),
                        "page_token": batch.page_token,
                    },
                )
                raw_receipts.append(raw_receipt)
    except PartialResponseError as exc:
        partial = True
        errors.append(str(exc))
        _LOG.warning(
            "ingestion.partial",
            extra={"event": "ingestion.partial", "source": source.name, "error": str(exc)},
        )
    except (ConnectorError, IngestionError) as exc:
        failed = True
        errors.append(str(exc))
        _LOG.error(
            "ingestion.failed",
            extra={"event": "ingestion.failed", "source": source.name, "error": str(exc)},
        )

    records = _frames(batches)
    if records.empty:
        report = QualityReport(
            dataset=store.dataset,
            source=source.name,
            schema_version=schema_version,
            input_count=0,
            accepted_count=0,
            rejected_count=0,
            checks={"records_present": False},
        )
        status = IngestionStatus.FAILED if failed or errors else IngestionStatus.EMPTY
        provenance = SourceProvenance(
            source=source.name,
            dataset=store.dataset,
            schema_version=schema_version,
            retrieved_at=started.isoformat(),
            request=effective_request.to_public_dict(),
            pages=len(batches),
            retries=getattr(source, "retry_attempts", 0),
            input_records=0,
            accepted_records=0,
            rejected_records=0,
            checksum=None,
            raw_sha256=tuple(r.sha256 for r in raw_receipts),
            errors=tuple(errors),
        )
        return IngestionResult(
            status=status,
            accepted=pd.DataFrame(),
            rejected=pd.DataFrame(),
            report=report,
            provenance=provenance,
            raw_receipts=tuple(raw_receipts),
            errors=tuple(errors),
        )

    try:
        normalized = normalize_market_frame(
            records,
            source=source.name,
            ingestion_timestamp=started,
            frequency=frequency,
            dataset=store.dataset,
            schema_version=schema_version,
        )
    except SchemaValidationError as exc:
        failed = True
        errors.append(str(exc))
        report = QualityReport(
            dataset=store.dataset,
            source=source.name,
            schema_version=schema_version,
            input_count=len(records),
            accepted_count=0,
            rejected_count=len(records),
            rejection_reasons={"schema": len(records)},
            checks={"schema": False},
        )
        provenance = SourceProvenance(
            source=source.name,
            dataset=store.dataset,
            schema_version=schema_version,
            retrieved_at=started.isoformat(),
            request=effective_request.to_public_dict(),
            pages=len(batches),
            retries=getattr(source, "retry_attempts", 0),
            input_records=len(records),
            accepted_records=0,
            rejected_records=len(records),
            checksum=None,
            raw_sha256=tuple(r.sha256 for r in raw_receipts),
            errors=tuple(errors),
        )
        return IngestionResult(
            status=IngestionStatus.FAILED,
            accepted=pd.DataFrame(),
            rejected=records,
            report=report,
            provenance=provenance,
            raw_receipts=tuple(raw_receipts),
            errors=tuple(errors),
        )

    receipt: StorageReceipt | None = None
    if not normalized.accepted.empty:
        receipt = store.write(
            normalized.accepted, source=source.name, ingestion_timestamp=started
        )
    if normalized.report.rejected_count:
        partial = True

    status = IngestionStatus.OK
    if failed:
        status = IngestionStatus.FAILED
    elif partial or normalized.report.rejected_count:
        status = IngestionStatus.PARTIAL
    elif normalized.accepted.empty:
        status = IngestionStatus.EMPTY

    checksum = None
    if not normalized.accepted.empty:
        content_columns = [
            c for c in normalized.accepted.columns if c != "ingestion_timestamp"
        ]
        checksum = frame_fingerprint(
            normalized.accepted,
            key_cols=("timestamp", "symbol"),
            columns=content_columns,
            schema_version=schema_version,
        )
    provenance = SourceProvenance(
        source=source.name,
        dataset=store.dataset,
        schema_version=schema_version,
        retrieved_at=started.isoformat(),
        request=effective_request.to_public_dict(),
        pages=len(batches),
        retries=getattr(source, "retry_attempts", 0),
        input_records=normalized.report.input_count,
        accepted_records=normalized.report.accepted_count,
        rejected_records=normalized.report.rejected_count,
        checksum=checksum,
        raw_sha256=tuple(r.sha256 for r in raw_receipts),
        errors=tuple(errors),
    )
    _LOG.info(
        "ingestion.complete",
        extra={
            "event": "ingestion.complete",
            "source": source.name,
            "status": status.value,
            "accepted": normalized.report.accepted_count,
            "rejected": normalized.report.rejected_count,
            "pages": len(batches),
            "retries": getattr(source, "retry_attempts", 0),
        },
    )
    return IngestionResult(
        status=status,
        accepted=normalized.accepted,
        rejected=normalized.rejected,
        report=normalized.report,
        provenance=provenance,
        storage_receipt=receipt,
        raw_receipts=tuple(raw_receipts),
        errors=tuple(errors),
    )
