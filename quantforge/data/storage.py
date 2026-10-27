"""Storage abstractions: transactional SQLite, staged Parquet persistence, and raw-data archive.

No HDF5/bcolz adapter is shipped: ``h5py`` is not a project dependency and the
supported paths (Parquet and SQLite) cover the module's requirements without a
legacy binary format. Raw payloads are archived immutably with SHA-256 names.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .errors import SchemaValidationError, StorageConflictError, StorageError
from .timeutils import ensure_utc, ensure_utc_series, utc_now

TEMPORAL_COLUMNS = (
    "timestamp",
    "publication_timestamp",
    "available_at",
    "ingestion_timestamp",
    "decision_timestamp",
    "valid_from",
    "valid_to",
)


def _jsonable(value: Any) -> Any:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        return repr(value)
    if isinstance(value, (int, bool, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def frame_fingerprint(
    frame: pd.DataFrame,
    *,
    key_cols: Sequence[str] = ("timestamp", "symbol"),
    columns: Sequence[str] | None = None,
    schema_version: str | None = None,
) -> str:
    """Stable SHA-256 over a deterministic, sorted, canonical representation."""

    if frame.empty:
        return hashlib.sha256(b"empty").hexdigest()
    cols = list(columns) if columns is not None else sorted(str(c) for c in frame.columns)
    missing = [c for c in key_cols if c not in frame.columns]
    if missing:
        raise SchemaValidationError(f"Cannot fingerprint without key columns {missing}")
    ordered = frame.sort_values(list(key_cols), kind="stable")
    digest = hashlib.sha256()
    if schema_version:
        digest.update(f"schema={schema_version}\n".encode())
    for _, row in ordered.iterrows():
        payload = {column: _jsonable(row[column]) for column in cols}
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _safe_partition(value: Any) -> str:
    text = str(value)
    clean = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in text) or "_"
    return "_" if clean in {".", ".."} else clean


def _validate_revision(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = numeric.isna() & series.notna()
    finite = numeric.notna() & np.isfinite(numeric)
    invalid |= numeric.notna() & ~finite
    invalid |= finite & ((numeric < 0) | (numeric % 1 != 0))
    if bool(invalid.any()):
        raise SchemaValidationError("revision must be a non-negative integer")
    return numeric.fillna(0).astype("int64")


@dataclass(frozen=True)
class StorageReceipt:
    store: str
    dataset: str
    rows: int
    partitions: tuple[str, ...]
    checksum: str
    schema_version: str
    idempotent: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "store": self.store,
            "dataset": self.dataset,
            "rows": self.rows,
            "partitions": list(self.partitions),
            "checksum": self.checksum,
            "schema_version": self.schema_version,
            "idempotent": self.idempotent,
            "details": dict(self.details),
        }


def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise SchemaValidationError(f"Missing storage columns: {missing}")


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    ingestion_timestamp: Any,
    schema_version: str,
    frequency: str,
) -> pd.DataFrame:
    _require_columns(frame, ("timestamp", "symbol"))
    work = frame.copy()
    for column in TEMPORAL_COLUMNS:
        if column in work.columns and column != "ingestion_timestamp":
            work[column] = ensure_utc_series(
                work[column],
                name=column,
                allow_missing=column in {"publication_timestamp", "valid_from", "valid_to"},
            )
    if "available_at" not in work.columns:
        raise SchemaValidationError("Storage requires available_at; do not silently invent it")
    work["available_at"] = ensure_utc_series(work["available_at"], name="available_at")
    work["ingestion_timestamp"] = ensure_utc(ingestion_timestamp, name="ingestion_timestamp")
    work["schema_version"] = schema_version
    if "frequency" not in work.columns:
        work["frequency"] = frequency
    work["frequency"] = work["frequency"].astype(str)
    if "source" not in work.columns:
        work["source"] = source
    if "revision" not in work.columns:
        work["revision"] = 0
    work["revision"] = _validate_revision(work["revision"])
    if work["symbol"].isna().any() or work["symbol"].astype(str).str.strip().eq("").any():
        raise SchemaValidationError("Storage requires a non-empty symbol")
    if bool((work["available_at"] < work["timestamp"]).any()):
        raise SchemaValidationError("available_at cannot precede timestamp")
    return work.reset_index(drop=True)


class ParquetStore:
    """Partitioned Parquet store with staged, rollback-protected partition replacement.

    Layout: ``dataset/frequency=.../instrument=.../date=YYYY-MM-DD/part-000.parquet``.
    All revisions are retained; ``read(latest=True)`` selects the highest revision
    per ``(timestamp, symbol, frequency)``. Identical re-ingestion is idempotent;
    same-revision conflicting payloads raise rather than overwrite.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        dataset: str = "bars",
        frequency: str = "1d",
        schema_version: str = "1",
        require_availability: bool = True,
    ) -> None:
        self.root = Path(root)
        self.dataset = dataset
        self.frequency = frequency
        self.schema_version = schema_version
        self.require_availability = require_availability
        self._manifest_lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def dataset_root(self) -> Path:
        return self.root / self.dataset

    def _partition_path(self, frequency: str, symbol: str, day: str) -> Path:
        return (
            self.dataset_root
            / f"frequency={_safe_partition(frequency)}"
            / f"instrument={_safe_partition(symbol)}"
            / f"date={_safe_partition(day)}"
            / "part-000.parquet"
        )

    def write(
        self,
        frame: pd.DataFrame,
        *,
        source: str,
        ingestion_timestamp: Any | None = None,
        mode: str = "upsert",
    ) -> StorageReceipt:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if mode not in {"upsert", "append"}:
            raise ValueError("mode must be 'upsert' or 'append'")
        stamp = ensure_utc(ingestion_timestamp) if ingestion_timestamp is not None else utc_now()
        work = _prepare_frame(
            frame,
            source=source,
            ingestion_timestamp=stamp,
            schema_version=self.schema_version,
            frequency=self.frequency,
        )
        work["_day"] = work["timestamp"].dt.strftime("%Y-%m-%d")
        plans: list[dict[str, Any]] = []
        checksums: list[str] = []

        for (frequency, symbol, day), incoming in work.groupby(
            ["frequency", "symbol", "_day"], sort=True, dropna=False
        ):
            path = self._partition_path(str(frequency), str(symbol), str(day))
            existing = pd.DataFrame()
            if path.exists():
                existing = pd.read_parquet(path)
                for column in TEMPORAL_COLUMNS:
                    if column in existing.columns:
                        existing[column] = ensure_utc_series(
                            existing[column],
                            name=column,
                            allow_missing=column
                            in {"publication_timestamp", "valid_from", "valid_to"},
                        )

            combined = self._reconcile(existing, incoming.drop(columns=["_day"]))
            combined = combined.sort_values(
                ["timestamp", "symbol", "frequency", "revision"], kind="stable"
            ).reset_index(drop=True)
            checksum = self._logical_fingerprint(combined)
            checksums.append(checksum)
            rel = str(path.relative_to(self.root))

            changed = True
            if path.exists():
                old_checksum = self._manifest_entry(rel, {}).get("checksum")
                existing_checksum = self._logical_fingerprint(existing)
                if old_checksum == checksum or existing_checksum == checksum:
                    changed = False

            plans.append(
                {
                    "path": path,
                    "relative": rel,
                    "combined": combined,
                    "checksum": checksum,
                    "changed": changed,
                    "manifest": {
                        "rows": int(len(combined)),
                        "checksum": checksum,
                        "schema_version": self.schema_version,
                        "source": source,
                        "frequency": str(frequency),
                        "instrument": str(symbol),
                        "date": str(day),
                        "updated_at": stamp.isoformat(),
                    },
                }
            )

        changed_plans = [plan for plan in plans if plan["changed"]]
        if changed_plans:
            self._commit_partitions(changed_plans, pa=pa, pq=pq)

        aggregate = hashlib.sha256("\n".join(sorted(checksums)).encode()).hexdigest()
        return StorageReceipt(
            store="parquet",
            dataset=self.dataset,
            rows=len(work),
            partitions=tuple(sorted(str(plan["relative"]) for plan in plans)),
            checksum=aggregate,
            schema_version=self.schema_version,
            idempotent=not bool(changed_plans),
        )

    def _logical_fingerprint(self, frame: pd.DataFrame) -> str:
        columns = [c for c in frame.columns if c != "ingestion_timestamp"]
        return frame_fingerprint(
            frame,
            key_cols=("timestamp", "symbol", "frequency", "revision"),
            columns=columns,
            schema_version=self.schema_version,
        )

    def _commit_partitions(
        self, plans: Sequence[Mapping[str, Any]], *, pa: Any, pq: Any
    ) -> None:
        """Stage all partitions, back up targets, then replace with rollback.

        Atomicity is per file. Handled failures during commit restore all
        already-replaced files; a process/power crash mid-commit can still leave
        a partially replaced multi-partition set.
        """

        staging = Path(tempfile.mkdtemp(prefix=".parquet-stage-", dir=self.root))
        staged: dict[Path, Path] = {}
        backups: dict[Path, Path] = {}
        manifest_temp: Path | None = None
        committed: list[Path] = []
        try:
            for plan in plans:
                staged[plan["path"]] = self._stage_parquet(
                    plan["combined"], plan["path"], staging, pa, pq
                )
            for plan in plans:
                target = plan["path"]
                if target.exists():
                    backups[target] = self._backup_partition(target, staging)
            manifest_temp = self._stage_manifest(
                {plan["relative"]: plan["manifest"] for plan in plans}
            )
            try:
                for target, temp in staged.items():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    self._commit_staged(temp, target)
                    committed.append(target)
                self._commit_manifest(manifest_temp)
                manifest_temp = None
            except Exception as exc:
                self._rollback_partitions(committed, backups)
                raise StorageError(
                    "Parquet multi-partition commit failed; previous partitions restored"
                ) from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if manifest_temp is not None and manifest_temp.exists():
                manifest_temp.unlink(missing_ok=True)

    def _stage_parquet(
        self,
        frame: pd.DataFrame,
        target: Path,
        staging: Path,
        pa: Any,
        pq: Any,
    ) -> Path:
        fd, name = tempfile.mkstemp(
            dir=staging, prefix=f"{target.parent.name}-{target.name}-", suffix=".stage"
        )
        os.close(fd)
        temp = Path(name)
        try:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            pq.write_table(table, temp, compression="zstd")
        except Exception as exc:
            temp.unlink(missing_ok=True)
            raise StorageError(f"Failed to stage Parquet partition {target}: {exc}") from exc
        return temp

    def _backup_partition(self, target: Path, staging: Path) -> Path:
        fd, name = tempfile.mkstemp(
            dir=staging, prefix=f"{target.parent.name}-{target.name}-", suffix=".backup"
        )
        os.close(fd)
        backup = Path(name)
        try:
            shutil.copy2(target, backup)
        except Exception as exc:
            backup.unlink(missing_ok=True)
            raise StorageError(f"Failed to back up Parquet partition {target}: {exc}") from exc
        return backup

    def _commit_staged(self, temp: Path, target: Path) -> None:
        os.replace(temp, target)

    def _rollback_partitions(
        self, committed: Sequence[Path], backups: Mapping[Path, Path]
    ) -> None:
        for target in reversed(committed):
            backup = backups.get(target)
            try:
                if backup is not None and backup.exists():
                    os.replace(backup, target)
                else:
                    target.unlink(missing_ok=True)
            except OSError:
                continue

    def _stage_manifest(self, entries: Mapping[str, Mapping[str, Any]]) -> Path:
        path = self._manifest_path()
        manifest = self._load_manifest()
        for relative, entry in entries.items():
            manifest.setdefault("partitions", {})[relative] = dict(entry)
        fd, name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}-", suffix=".stage"
        )
        os.close(fd)
        temp = Path(name)
        try:
            temp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            temp.unlink(missing_ok=True)
            raise
        return temp

    def _commit_manifest(self, temp: Path) -> None:
        with self._manifest_lock:
            os.replace(temp, self._manifest_path())

    def _reconcile(self, existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
        frames = [f for f in (existing, incoming) if not f.empty]
        if not frames:
            return incoming.iloc[0:0].copy()
        combined = pd.concat(frames, ignore_index=True, sort=False)
        data_cols = [c for c in combined.columns if c != "ingestion_timestamp"]
        exact = combined.duplicated(subset=data_cols, keep="first")
        combined = combined.loc[~exact].copy()
        key = ["timestamp", "symbol", "frequency", "revision"]
        duplicates = combined.duplicated(subset=key, keep=False)
        if bool(duplicates.any()):
            for key_values, group in combined[duplicates].groupby(key, sort=False, dropna=False):
                payload = group.drop(columns=["ingestion_timestamp"], errors="ignore")
                if payload.drop_duplicates().shape[0] > 1:
                    raise StorageConflictError(
                        f"Conflicting payload for partition key {key}={key_values}"
                    )
            combined = combined.drop_duplicates(subset=key, keep="first")
        return combined

    def _atomic_write_parquet(self, frame: pd.DataFrame, path: Path, pa: Any, pq: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=".parquet.tmp", delete=False
            ) as handle:
                temp_name = handle.name
            table = pa.Table.from_pandas(frame, preserve_index=False)
            pq.write_table(table, temp_name, compression="zstd")
            os.replace(temp_name, path)
            temp_name = None
        except Exception as exc:  # pragma: no cover - filesystem failures
            raise StorageError(f"Failed to atomically write {path}: {exc}") from exc
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

    def _manifest_path(self) -> Path:
        return self.root / f".{self.dataset}.manifest.json"

    def _load_manifest(self) -> dict[str, Any]:
        path = self._manifest_path()
        if not path.exists():
            return {"dataset": self.dataset, "schema_version": self.schema_version, "partitions": {}}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - corrupt file
            raise StorageError(f"Corrupt manifest {path}") from exc

    def _manifest_entry(self, relative: str, default: Any) -> Any:
        return self._load_manifest().get("partitions", {}).get(relative, default)

    def _update_manifest(self, relative: str, entry: Mapping[str, Any]) -> None:
        with self._manifest_lock:
            manifest = self._load_manifest()
            manifest.setdefault("partitions", {})[relative] = dict(entry)
            path = self._manifest_path()
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temp, path)

    def manifest(self) -> dict[str, Any]:
        return self._load_manifest()

    def read(
        self,
        *,
        symbol: str | None = None,
        frequency: str | None = None,
        start: Any | None = None,
        end: Any | None = None,
        latest: bool = True,
    ) -> pd.DataFrame:
        files = sorted(self.dataset_root.glob("frequency=*/instrument=*/date=*/part-*.parquet"))
        frames: list[pd.DataFrame] = []
        for path in files:
            if frequency is not None and path.parts[-4] != f"frequency={_safe_partition(frequency)}":
                continue
            if symbol is not None and path.parts[-3] != f"instrument={_safe_partition(symbol)}":
                continue
            try:
                frames.append(pd.read_parquet(path))
            except Exception as exc:  # pragma: no cover - corrupt parquet
                raise StorageError(f"Failed to read {path}: {exc}") from exc
        if not frames:
            return pd.DataFrame(columns=["timestamp", "symbol", "frequency", "revision"])
        out = pd.concat(frames, ignore_index=True, sort=False)
        for column in TEMPORAL_COLUMNS:
            if column in out.columns:
                out[column] = ensure_utc_series(
                    out[column],
                    name=column,
                    allow_missing=column in {"publication_timestamp", "valid_from", "valid_to"},
                )
        if start is not None:
            out = out[out["timestamp"] >= ensure_utc(start, name="start")]
        if end is not None:
            out = out[out["timestamp"] <= ensure_utc(end, name="end")]
        if latest and not out.empty:
            out = out.sort_values(
                ["timestamp", "symbol", "frequency", "revision"], kind="stable"
            ).drop_duplicates(subset=["timestamp", "symbol", "frequency"], keep="last")
        return out.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)

    def max_timestamp(
        self, *, symbol: str | None = None, frequency: str | None = None
    ) -> pd.Timestamp | None:
        frame = self.read(symbol=symbol, frequency=frequency, latest=True)
        if frame.empty:
            return None
        return ensure_utc(frame["timestamp"].max())


class SQLiteStore:
    """SQLite store for metadata and lightweight bar datasets.

    Every write is a single transaction. Existing rows are only replaced when the
    payload is identical (idempotent); same-revision conflicts raise.
    """

    BAR_COLUMNS = [
        "timestamp",
        "symbol",
        "frequency",
        "revision",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "available_at",
        "source",
        "ingestion_timestamp",
        "schema_version",
    ]

    def __init__(self, path: str | Path, *, dataset: str = "bars", schema_version: str = "1") -> None:
        self.path = Path(path)
        self.dataset = dataset
        self.schema_version = schema_version
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Any:
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS bars (
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    frequency TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    available_at TEXT NOT NULL,
                    source TEXT,
                    ingestion_timestamp TEXT,
                    schema_version TEXT,
                    PRIMARY KEY (timestamp, symbol, frequency, revision)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_runs (
                    run_id TEXT PRIMARY KEY,
                    dataset TEXT NOT NULL,
                    source TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    rows INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT
                )
                """
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.commit()

    def write(
        self,
        frame: pd.DataFrame,
        *,
        source: str,
        ingestion_timestamp: Any | None = None,
        mode: str = "upsert",
    ) -> StorageReceipt:
        if mode not in {"upsert", "append"}:
            raise ValueError("mode must be 'upsert' or 'append'")
        stamp = ensure_utc(ingestion_timestamp) if ingestion_timestamp is not None else utc_now()
        work = _prepare_frame(
            frame,
            source=source,
            ingestion_timestamp=stamp,
            schema_version=self.schema_version,
            frequency="1d",
        )
        _require_columns(work, ("open", "high", "low", "close", "volume"))
        rows: list[tuple[Any, ...]] = []
        for _, row in work.iterrows():
            rows.append(
                (
                    row["timestamp"].isoformat(),
                    str(row["symbol"]),
                    str(row["frequency"]),
                    int(row["revision"]),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                    row["available_at"].isoformat(),
                    str(row.get("source", source)),
                    row["ingestion_timestamp"].isoformat(),
                    str(row.get("schema_version", self.schema_version)),
                )
            )

        checksum = frame_fingerprint(
            work.drop(columns=["ingestion_timestamp"]),
            key_cols=("timestamp", "symbol", "frequency", "revision"),
            schema_version=self.schema_version,
        )
        run_id = hashlib.sha256(
            f"{source}|{stamp.isoformat()}|{checksum}".encode()
        ).hexdigest()
        idempotent = True
        inserted = 0
        matched = 0
        with self._connect() as connection:
            existing: dict[tuple[Any, ...], sqlite3.Row] = {}
            keys_by_symbol: dict[str, list[tuple[Any, ...]]] = {}
            for row in rows:
                keys_by_symbol.setdefault(row[1], []).append(row[0])
            for symbol, timestamps in keys_by_symbol.items():
                for offset in range(0, len(timestamps), 500):
                    chunk = timestamps[offset : offset + 500]
                    query = (
                        "SELECT * FROM bars WHERE symbol = ? AND timestamp IN ("
                        + ",".join("?" * len(chunk))
                        + ")"
                    )
                    for record in connection.execute(query, (symbol, *chunk)):
                        existing[
                            (
                                record["timestamp"],
                                record["symbol"],
                                record["frequency"],
                                record["revision"],
                            )
                        ] = record
            for row in rows:
                key = row[0], row[1], row[2], row[3]
                current = existing.get(key)
                if current is not None:
                    current_data = (
                        current["open"],
                        current["high"],
                        current["low"],
                        current["close"],
                        current["volume"],
                        current["available_at"],
                    )
                    new_data = (row[4], row[5], row[6], row[7], row[8], row[9])
                    if current_data != new_data:
                        raise StorageConflictError(
                            f"Conflicting SQLite payload for key {key}"
                        )
                    matched += 1
                    continue
                inserted += 1
                connection.execute(
                    f"INSERT INTO bars ({','.join(self.BAR_COLUMNS)}) VALUES ({','.join('?' * len(self.BAR_COLUMNS))})",
                    row,
                )
            connection.execute(
                "INSERT OR REPLACE INTO ingestion_runs VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    self.dataset,
                    source,
                    stamp.isoformat(),
                    utc_now().isoformat(),
                    len(rows),
                    checksum,
                    "ok",
                    json.dumps({"mode": mode}),
                ),
            )
            connection.commit()

        idempotent = bool(rows) and inserted == 0 and matched == len(rows)
        return StorageReceipt(
            store="sqlite",
            dataset=self.dataset,
            rows=len(work),
            partitions=(str(self.path),),
            checksum=checksum,
            schema_version=self.schema_version,
            idempotent=idempotent,
        )

    def read_bars(
        self,
        *,
        symbol: str | None = None,
        frequency: str | None = None,
        start: Any | None = None,
        end: Any | None = None,
        latest: bool = True,
    ) -> pd.DataFrame:
        query = "SELECT * FROM bars"
        clauses: list[str] = []
        params: list[Any] = []
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol)
        if frequency is not None:
            clauses.append("frequency = ?")
            params.append(frequency)
        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(ensure_utc(start, name="start").isoformat())
        if end is not None:
            clauses.append("timestamp <= ?")
            params.append(ensure_utc(end, name="end").isoformat())
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        if latest:
            query = (
                "SELECT * FROM (SELECT *, ROW_NUMBER() OVER "
                "(PARTITION BY timestamp, symbol, frequency ORDER BY revision DESC) rn "
                f"FROM ({query})) WHERE rn = 1"
            )
        with self._connect() as connection:
            rows = [dict(row) for row in connection.execute(query, params)]
        if not rows:
            return pd.DataFrame(columns=self.BAR_COLUMNS)
        frame = pd.DataFrame(rows).drop(columns=["rn"], errors="ignore")
        for column in ("timestamp", "available_at", "ingestion_timestamp"):
            if column in frame.columns:
                frame[column] = ensure_utc_series(frame[column], name=column)
        return frame.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)

    def max_timestamp(self, *, symbol: str | None = None, frequency: str | None = None) -> pd.Timestamp | None:
        frame = self.read_bars(symbol=symbol, frequency=frequency, latest=True)
        if frame.empty:
            return None
        return ensure_utc(frame["timestamp"].max())

    def set_metadata(self, key: str, value: Any) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (key, json.dumps(_jsonable(value), sort_keys=True)),
            )
            connection.commit()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def ingestion_runs(self) -> pd.DataFrame:
        with self._connect() as connection:
            rows = [dict(row) for row in connection.execute("SELECT * FROM ingestion_runs ORDER BY started_at")]
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class RawArchiveReceipt:
    path: str
    sha256: str
    size: int
    source: str
    received_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "source": self.source,
            "received_at": self.received_at,
        }


_SECRET_MARKERS = ("token", "secret", "password", "passwd", "apikey", "api_key", "authorization")


def _redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>"
            if any(marker in str(key).lower() for marker in _SECRET_MARKERS)
            else _redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    return _jsonable(value)


def _redact(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    return {
        str(key): "<redacted>"
        if any(marker in str(key).lower() for marker in _SECRET_MARKERS)
        else _redact_value(value)
        for key, value in metadata.items()
    }


class RawArchive:
    """Immutable, date-independent content-addressed raw payload archive."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, path: Path) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise StorageError(f"Raw archive path escapes root: {path}") from exc
        return resolved

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
            raise StorageError(f"Invalid raw archive {label}: {value!r}")
        return _safe_partition(text)

    def write(
        self,
        payload: bytes | str | Mapping[str, Any] | list[Any],
        *,
        source: str,
        extension: str = "json",
        received_at: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RawArchiveReceipt:
        stamp = ensure_utc(received_at) if received_at is not None else utc_now()
        if isinstance(payload, bytes):
            data = payload
        elif isinstance(payload, str):
            data = payload.encode("utf-8")
        else:
            data = json.dumps(payload, sort_keys=True, default=_jsonable).encode("utf-8")
        sha = hashlib.sha256(data).hexdigest()
        clean_source = self._component(source, label="source")
        clean_extension = self._component(extension, label="extension")
        path = self.root / clean_source / f"{sha}.{clean_extension}"
        path = self._safe_path(path)
        if path.exists():
            existing = hashlib.sha256(path.read_bytes()).hexdigest()
            if existing != sha:  # pragma: no cover - content-addressed collision
                raise StorageConflictError(f"Raw archive collision at {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_bytes(data)
            os.replace(temp, path)
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        if not sidecar.exists():
            sidecar.write_text(
                json.dumps(
                    {
                        "sha256": sha,
                        "size": len(data),
                        "source": source,
                        "received_at": stamp.isoformat(),
                        "metadata": _redact(metadata),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return RawArchiveReceipt(
            path=str(path), sha256=sha, size=len(data), source=source, received_at=stamp.isoformat()
        )

    def read(self, receipt: RawArchiveReceipt | str | Path) -> bytes:
        path = self._safe_path(
            Path(receipt.path if isinstance(receipt, RawArchiveReceipt) else receipt)
        )
        return path.read_bytes()

    def verify(self, receipt: RawArchiveReceipt | str | Path) -> bool:
        path = self._safe_path(
            Path(receipt.path if isinstance(receipt, RawArchiveReceipt) else receipt)
        )
        expected = receipt.sha256 if isinstance(receipt, RawArchiveReceipt) else path.stem
        return hashlib.sha256(path.read_bytes()).hexdigest() == expected
