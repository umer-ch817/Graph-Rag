"""Durable local jobs for T12a (dashboard, R11).

SQLite owns job/run state; Neo4j owns retrievable chunks/entities; JSONL
artifacts own evaluation payloads. This module owns the jobs side: datasets,
index snapshots, jobs and attempts. ``run_store.py`` owns runs, predictions,
usage events and comparisons against the same database file.

Guarantees (TRD reliability, APP-FLOW state machine):

- WAL + foreign keys + UTC timestamps + schema-version row.
- Atomic claim: one worker claims each queued job once.
- Heartbeat leases: stale running jobs become ``interrupted`` on restart.
- Cancellation is honored between items; in-flight calls cannot be undone.
- Budget pause preserves validated checkpoint entries.
- Duplicate active submissions (kind + snapshot + config) return the
  existing job instead of creating a second active job.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1
SCHEMA_LABEL = "jobs-v1"
DEFAULT_DB_PATH = "artifacts/jobs.sqlite3"
DEFAULT_LEASE_TIMEOUT_S = 30.0

JobState = Literal[
    "queued",
    "running",
    "completed",
    "failed",
    "paused_budget",
    "cancelling",
    "cancelled",
    "interrupted",
]

ACTIVE_STATES = ("queued", "running", "cancelling", "paused_budget", "interrupted")
TERMINAL_STATES = ("completed", "cancelled")
RESUMABLE_STATES = ("interrupted", "cancelled", "failed", "paused_budget")

_KIND_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class JobError(RuntimeError):
    code = "JOB_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidInputError(JobError, ValueError):
    code = "INVALID_INPUT"


class DependencyUnavailableError(JobError):
    code = "DEPENDENCY_UNAVAILABLE"


class ConfigMismatchError(JobError, ValueError):
    code = "CONFIG_MISMATCH"


class BudgetExceededError(JobError):
    code = "BUDGET_EXCEEDED"


class ProviderError(JobError):
    code = "PROVIDER_ERROR"


class ValidationFailedError(JobError, ValueError):
    code = "VALIDATION_FAILED"


class CancelledError(JobError):
    code = "CANCELLED"


class IndexNotReadyError(JobError):
    code = "INDEX_NOT_READY"


_ERROR_BY_CODE: dict[str, type[JobError]] = {
    "INVALID_INPUT": InvalidInputError,
    "DEPENDENCY_UNAVAILABLE": DependencyUnavailableError,
    "CONFIG_MISMATCH": ConfigMismatchError,
    "BUDGET_EXCEEDED": BudgetExceededError,
    "PROVIDER_ERROR": ProviderError,
    "VALIDATION_FAILED": ValidationFailedError,
    "CANCELLED": CancelledError,
    "INDEX_NOT_READY": IndexNotReadyError,
}


def error_for_code(code: str, message: str) -> JobError:
    cls = _ERROR_BY_CODE.get(code, JobError)
    try:
        return cls(message)
    except Exception:
        return JobError(f"{code}: {message}")


class JobRecord(Record):
    id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    snapshot_id: str | None = None
    config_json: dict[str, Any] = Field(default_factory=dict)
    config_hash: str = Field(min_length=1)
    idempotency_key: str | None = None
    state: JobState = "queued"
    progress_done: int = Field(ge=0, strict=True)
    progress_total: int = Field(ge=0, strict=True)
    checkpoint_json: dict[str, Any] = Field(default_factory=dict)
    cancel_requested: bool = False
    error_code: str | None = None
    error_message: str | None = None
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)


class JobAttempt(Record):
    id: int = Field(ge=1, strict=True)
    job_id: str = Field(min_length=1)
    attempt_number: int = Field(ge=1, strict=True)
    worker_id: str = Field(min_length=1)
    started_at: str = Field(min_length=1)
    heartbeat_at: str = Field(min_length=1)
    ended_at: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class JobRef(Record):
    job_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    state: JobState
    created: bool = False


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _validate_kind(kind: str) -> str:
    if not isinstance(kind, str) or not kind.strip():
        raise InvalidInputError("job kind must be a nonempty string")
    cleaned = kind.strip()
    if len(cleaned) > 64 or not _KIND_RE.match(cleaned):
        raise InvalidInputError("job kind must match ^[A-Za-z0-9][A-Za-z0-9_.-]*$ (max 64 chars)")
    return cleaned


def _validate_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not job_id.strip():
        raise InvalidInputError("job_id must be a nonempty string")
    if len(job_id) > 128:
        raise InvalidInputError("job_id exceeds 128 chars")
    return job_id.strip()


def _validate_worker_id(worker_id: str) -> str:
    if not isinstance(worker_id, str) or not worker_id.strip():
        raise InvalidInputError("worker_id must be a nonempty string")
    cleaned = worker_id.strip()
    if len(cleaned) > 128:
        raise InvalidInputError("worker_id exceeds 128 chars")
    return cleaned


def canonical_config_hash(kind: str, snapshot_id: str | None, config: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"kind": kind, "snapshot_id": snapshot_id, "config": config},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _parse_json_object(raw: str | None, *, what: str) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationFailedError(f"Stored {what} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationFailedError(f"Stored {what} must be a JSON object")
    return payload


_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  label TEXT NOT NULL,
  applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  checksum TEXT NOT NULL,
  seed INTEGER NOT NULL,
  sample_count INTEGER NOT NULL,
  manifest_path TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS index_snapshots (
  id TEXT PRIMARY KEY,
  dataset_id TEXT REFERENCES datasets(id) ON DELETE SET NULL,
  fingerprint TEXT UNIQUE NOT NULL,
  status TEXT NOT NULL,
  vector_status TEXT NOT NULL,
  graph_status TEXT NOT NULL,
  coverage_json TEXT NOT NULL DEFAULT '{}',
  artifact_path TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  snapshot_id TEXT REFERENCES index_snapshots(id) ON DELETE SET NULL,
  config_json TEXT NOT NULL DEFAULT '{}',
  config_hash TEXT NOT NULL,
  idempotency_key TEXT,
  state TEXT NOT NULL,
  progress_done INTEGER NOT NULL DEFAULT 0,
  progress_total INTEGER NOT NULL DEFAULT 0,
  checkpoint_json TEXT NOT NULL DEFAULT '{}',
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  error_code TEXT,
  error_message TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_jobs_active
  ON jobs(kind, snapshot_id, config_hash)
  WHERE state IN ('queued','running','cancelling','paused_budget','interrupted');
CREATE INDEX IF NOT EXISTS ix_jobs_state_created ON jobs(state, created_at);
CREATE TABLE IF NOT EXISTS job_attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  attempt_number INTEGER NOT NULL,
  worker_id TEXT NOT NULL,
  started_at TEXT NOT NULL,
  heartbeat_at TEXT NOT NULL,
  ended_at TEXT,
  error_code TEXT,
  error_message TEXT,
  UNIQUE(job_id, attempt_number)
);
CREATE INDEX IF NOT EXISTS ix_attempts_job ON job_attempts(job_id);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  snapshot_id TEXT REFERENCES index_snapshots(id) ON DELETE SET NULL,
  job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
  strategy TEXT NOT NULL,
  split TEXT,
  config_hash TEXT NOT NULL,
  config_json TEXT NOT NULL DEFAULT '{}',
  question_set_hash TEXT NOT NULL,
  state TEXT NOT NULL,
  artifact_path TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  question_id TEXT NOT NULL,
  answer TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL,
  citations_json TEXT NOT NULL DEFAULT '[]',
  retrieval_json TEXT,
  latency_ms REAL,
  usage_json TEXT,
  failure_code TEXT,
  PRIMARY KEY(run_id, question_id)
);
CREATE TABLE IF NOT EXISTS usage_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_attempt_id INTEGER REFERENCES job_attempts(id) ON DELETE SET NULL,
  run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
  stage TEXT NOT NULL,
  request_id TEXT NOT NULL,
  model TEXT,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  estimated_cost_usd REAL,
  currency TEXT NOT NULL DEFAULT 'USD',
  pricing_version TEXT,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS comparisons (
  id TEXT PRIMARY KEY,
  run_ids_json TEXT NOT NULL,
  compatibility_hash TEXT NOT NULL,
  metrics_json TEXT,
  intervals_json TEXT,
  report_path TEXT,
  created_at TEXT NOT NULL
);
"""


def _connect(path: str | Path) -> sqlite3.Connection:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(dest), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(path: str | Path) -> Path:
    dest = Path(path)
    conn = _connect(dest)
    try:
        conn.executescript(_SCHEMA_SQL)
        now = utc_now_iso()
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, label, applied_at) VALUES (?, ?, ?)",
            (SCHEMA_VERSION, SCHEMA_LABEL, now),
        )
    finally:
        conn.close()
    return dest


def _row_to_job(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        id=str(row["id"]),
        kind=str(row["kind"]),
        snapshot_id=str(row["snapshot_id"]) if row["snapshot_id"] is not None else None,
        config_json=_parse_json_object(row["config_json"], what="config_json"),
        config_hash=str(row["config_hash"]),
        idempotency_key=str(row["idempotency_key"]) if row["idempotency_key"] is not None else None,
        state=str(row["state"]),  # type: ignore[arg-type]
        progress_done=int(row["progress_done"]),
        progress_total=int(row["progress_total"]),
        checkpoint_json=_parse_json_object(row["checkpoint_json"], what="checkpoint_json"),
        cancel_requested=bool(int(row["cancel_requested"])),
        error_code=str(row["error_code"]) if row["error_code"] is not None else None,
        error_message=str(row["error_message"]) if row["error_message"] is not None else None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_attempt(row: sqlite3.Row) -> JobAttempt:
    return JobAttempt(
        id=int(row["id"]),
        job_id=str(row["job_id"]),
        attempt_number=int(row["attempt_number"]),
        worker_id=str(row["worker_id"]),
        started_at=str(row["started_at"]),
        heartbeat_at=str(row["heartbeat_at"]),
        ended_at=str(row["ended_at"]) if row["ended_at"] is not None else None,
        error_code=str(row["error_code"]) if row["error_code"] is not None else None,
        error_message=str(row["error_message"]) if row["error_message"] is not None else None,
    )


class JobStore:
    """Durable SQLite job queue. One instance per database file."""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        self._path = Path(path)
        init_db(self._path)

    @property
    def path(self) -> Path:
        return self._path

    def _open(self) -> sqlite3.Connection:
        return _connect(self._path)

    # -- datasets / snapshots (minimal, for T12b dashboard) -----------------

    def create_dataset(
        self,
        *,
        dataset_id: str,
        source: str,
        checksum: str,
        seed: int,
        sample_count: int,
        manifest_path: str,
    ) -> dict[str, Any]:
        if not dataset_id.strip() or not source.strip() or not checksum.strip():
            raise InvalidInputError("dataset id/source/checksum must be nonempty")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise InvalidInputError("seed must be a non-negative integer")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool):
            raise InvalidInputError("sample_count must be an integer")
        if sample_count <= 0:
            raise InvalidInputError("sample_count must be positive")
        if not manifest_path.strip():
            raise InvalidInputError("manifest_path must be nonempty")
        now = utc_now_iso()
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                "INSERT OR REPLACE INTO datasets"
                "(id, source, checksum, seed, sample_count, manifest_path, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    dataset_id.strip(),
                    source.strip(),
                    checksum.strip(),
                    seed,
                    sample_count,
                    manifest_path.strip(),
                    now,
                ),
            )
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        return {
            "id": dataset_id.strip(),
            "source": source.strip(),
            "checksum": checksum.strip(),
            "seed": seed,
            "sample_count": sample_count,
            "manifest_path": manifest_path.strip(),
            "created_at": now,
        }

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        conn = self._open()
        try:
            row = conn.execute(
                "SELECT * FROM datasets WHERE id = ?", (dataset_id.strip(),)
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def create_snapshot(
        self,
        *,
        snapshot_id: str,
        dataset_id: str | None,
        fingerprint: str,
        status: str = "pending",
        vector_status: str = "pending",
        graph_status: str = "pending",
        coverage: dict[str, Any] | None = None,
        artifact_path: str = "",
    ) -> dict[str, Any]:
        if not snapshot_id.strip() or not fingerprint.strip():
            raise InvalidInputError("snapshot_id and fingerprint must be nonempty")
        for label, value in (
            ("status", status),
            ("vector_status", vector_status),
            ("graph_status", graph_status),
        ):
            if value not in ("pending", "building", "ready", "failed", "partial"):
                raise InvalidInputError(f"{label} must be a snapshot status, got {value!r}")
        now = utc_now_iso()
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            if dataset_id is not None:
                exists = conn.execute(
                    "SELECT id FROM datasets WHERE id = ?", (dataset_id,)
                ).fetchone()
                if exists is None:
                    raise InvalidInputError(f"dataset {dataset_id!r} does not exist")
            conn.execute(
                "INSERT OR REPLACE INTO index_snapshots"
                "(id, dataset_id, fingerprint, status, vector_status, graph_status,"
                " coverage_json, artifact_path, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot_id.strip(),
                    dataset_id,
                    fingerprint.strip(),
                    status,
                    vector_status,
                    graph_status,
                    json.dumps(coverage or {}, sort_keys=True),
                    artifact_path,
                    now,
                    now,
                ),
            )
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        return {
            "id": snapshot_id.strip(),
            "dataset_id": dataset_id,
            "fingerprint": fingerprint.strip(),
            "status": status,
            "vector_status": vector_status,
            "graph_status": graph_status,
            "coverage": coverage or {},
            "artifact_path": artifact_path,
            "created_at": now,
            "updated_at": now,
        }

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        conn = self._open()
        try:
            row = conn.execute(
                "SELECT * FROM index_snapshots WHERE id = ?", (snapshot_id.strip(),)
            ).fetchone()
            if row is None:
                return None
            payload = dict(row)
            payload["coverage"] = _parse_json_object(
                payload.get("coverage_json"), what="coverage_json"
            )
            return payload
        finally:
            conn.close()

    def update_snapshot_status(
        self,
        snapshot_id: str,
        *,
        status: str | None = None,
        vector_status: str | None = None,
        graph_status: str | None = None,
        coverage: dict[str, Any] | None = None,
        artifact_path: str | None = None,
    ) -> dict[str, Any]:
        current = self.get_snapshot(snapshot_id)
        if current is None:
            raise InvalidInputError(f"snapshot {snapshot_id!r} does not exist")
        for label, value in (
            ("status", status),
            ("vector_status", vector_status),
            ("graph_status", graph_status),
        ):
            if value is not None and value not in (
                "pending",
                "building",
                "ready",
                "failed",
                "partial",
            ):
                raise InvalidInputError(f"{label} must be a snapshot status, got {value!r}")
        now = utc_now_iso()
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                "UPDATE index_snapshots SET status = COALESCE(?, status),"
                " vector_status = COALESCE(?, vector_status),"
                " graph_status = COALESCE(?, graph_status),"
                " coverage_json = COALESCE(?, coverage_json),"
                " artifact_path = COALESCE(?, artifact_path), updated_at = ?"
                " WHERE id = ?",
                (
                    status,
                    vector_status,
                    graph_status,
                    json.dumps(coverage, sort_keys=True) if coverage is not None else None,
                    artifact_path,
                    now,
                    snapshot_id.strip(),
                ),
            )
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        updated = self.get_snapshot(snapshot_id)
        assert updated is not None
        return updated

    # -- jobs ---------------------------------------------------------------

    def enqueue_job(
        self,
        kind: str,
        config: dict[str, Any] | None = None,
        *,
        snapshot_id: str | None = None,
        idempotency_key: str | None = None,
        job_id: str | None = None,
        progress_total: int | None = None,
    ) -> JobRef:
        clean_kind = _validate_kind(kind)
        cfg = dict(config or {})
        try:
            json.dumps(cfg, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise InvalidInputError(f"config must be JSON-serializable: {exc}") from exc
        snap = snapshot_id.strip() if isinstance(snapshot_id, str) and snapshot_id.strip() else None
        key = (
            idempotency_key.strip()
            if isinstance(idempotency_key, str) and idempotency_key.strip()
            else None
        )
        if key is not None and len(key) > 128:
            raise InvalidInputError("idempotency_key exceeds 128 chars")
        if snap is not None:
            if self.get_snapshot(snap) is None:
                # Snapshot FK is nullable; unknown snapshots are rejected early
                # rather than failing silently inside the worker.
                raise InvalidInputError(f"snapshot {snap!r} does not exist")
        items = cfg.get("items")
        if progress_total is None:
            progress_total = len(items) if isinstance(items, list) else 0
        if not isinstance(progress_total, int) or isinstance(progress_total, bool):
            raise InvalidInputError("progress_total must be an integer")
        if progress_total < 0:
            raise InvalidInputError("progress_total must be non-negative")
        if isinstance(items, list) and progress_total != len(items):
            raise InvalidInputError("progress_total must equal len(config['items'])")
        config_hash = canonical_config_hash(clean_kind, snap, cfg)
        dedup_hash = f"{config_hash}:{key}" if key else config_hash
        now = utc_now_iso()
        new_id = (
            job_id.strip()
            if isinstance(job_id, str) and job_id.strip()
            else f"job_{uuid.uuid4().hex[:12]}"
        )
        _validate_job_id(new_id)
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            # Idempotent active-job guard: same kind+snapshot+config returns existing.
            existing = conn.execute(
                "SELECT id, kind, state FROM jobs WHERE kind = ?"
                " AND ((snapshot_id IS NULL AND ? IS NULL) OR snapshot_id = ?)"
                " AND config_hash = ? AND state IN"
                " ('queued','running','cancelling','paused_budget','interrupted')"
                " ORDER BY created_at ASC LIMIT 1",
                (clean_kind, snap, snap, dedup_hash),
            ).fetchone()
            if existing is not None:
                conn.execute("ROLLBACK;")
                return JobRef(
                    job_id=str(existing["id"]),
                    kind=str(existing["kind"]),
                    state=str(existing["state"]),  # type: ignore[arg-type]
                    created=False,
                )
            conn.execute(
                "INSERT INTO jobs(id, kind, snapshot_id, config_json, config_hash,"
                " idempotency_key, state, progress_done, progress_total,"
                " checkpoint_json, cancel_requested, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, 'queued', 0, ?, '{}', 0, ?, ?)",
                (
                    new_id,
                    clean_kind,
                    snap,
                    json.dumps(cfg, sort_keys=True, ensure_ascii=False),
                    dedup_hash,
                    key,
                    int(progress_total),
                    now,
                    now,
                ),
            )
            conn.execute("COMMIT;")
        except sqlite3.IntegrityError as exc:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise InvalidInputError(f"job enqueue violates uniqueness: {exc}") from exc
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        return JobRef(job_id=new_id, kind=clean_kind, state="queued", created=True)

    def get_job(self, job_id: str) -> JobRecord:
        clean = _validate_job_id(job_id)
        conn = self._open()
        try:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (clean,)).fetchone()
            if row is None:
                raise InvalidInputError(f"job {clean!r} does not exist")
            return _row_to_job(row)
        finally:
            conn.close()

    def list_jobs(
        self,
        *,
        state: JobState | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> list[JobRecord]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise InvalidInputError("limit must be an integer in 1..1000")
        clauses: list[str] = []
        params: list[Any] = []
        if state is not None:
            if state not in (
                "queued",
                "running",
                "completed",
                "failed",
                "paused_budget",
                "cancelling",
                "cancelled",
                "interrupted",
            ):
                raise InvalidInputError(f"unknown job state {state!r}")
            clauses.append("state = ?")
            params.append(state)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(_validate_kind(kind))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        conn = self._open()
        try:
            rows = conn.execute(
                f"SELECT * FROM jobs {where} ORDER BY created_at ASC LIMIT ?",
                (*params, limit),
            ).fetchall()
            return [_row_to_job(row) for row in rows]
        finally:
            conn.close()

    def cancel_job(self, job_id: str) -> JobRecord:
        clean = _validate_job_id(job_id)
        now = utc_now_iso()
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (clean,)).fetchone()
            if row is None:
                raise InvalidInputError(f"job {clean!r} does not exist")
            state = str(row["state"])
            if state == "completed":
                raise InvalidInputError(f"job {clean!r} is already completed; cannot cancel")
            if state in ("cancelled", "cancelling"):
                conn.execute("ROLLBACK;")
                return self.get_job(clean)
            if state == "queued":
                conn.execute(
                    "UPDATE jobs SET state = 'cancelled', cancel_requested = 1,"
                    " error_code = 'CANCELLED',"
                    " error_message = 'Cancelled before start', updated_at = ?"
                    " WHERE id = ?",
                    (now, clean),
                )
                latest = conn.execute(
                    "SELECT attempt_number FROM job_attempts WHERE job_id = ?"
                    " ORDER BY attempt_number DESC LIMIT 1",
                    (clean,),
                ).fetchone()
                next_attempt = int(latest["attempt_number"]) + 1 if latest else 1
                conn.execute(
                    "INSERT INTO job_attempts(job_id, attempt_number, worker_id,"
                    " started_at, heartbeat_at, ended_at, error_code, error_message)"
                    " VALUES (?, ?, 'api', ?, ?, ?, 'CANCELLED', 'Cancelled before start')",
                    (clean, next_attempt, now, now, now),
                )
            else:
                # running / failed / paused_budget / interrupted: request cancellation;
                # the worker observes the flag at the next item boundary.
                next_state = "cancelling" if state == "running" else "cancelled"
                conn.execute(
                    "UPDATE jobs SET cancel_requested = 1, state = ?,"
                    " error_code = COALESCE(error_code, 'CANCELLED'),"
                    " error_message = COALESCE(error_message, 'Cancellation requested'),"
                    " updated_at = ? WHERE id = ?",
                    (next_state, now, clean),
                )
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        return self.get_job(clean)

    def resume_job(self, job_id: str) -> JobRecord:
        clean = _validate_job_id(job_id)
        now = utc_now_iso()
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (clean,)).fetchone()
            if row is None:
                raise InvalidInputError(f"job {clean!r} does not exist")
            state = str(row["state"])
            if state not in RESUMABLE_STATES:
                raise InvalidInputError(
                    f"job {clean!r} in state {state!r} cannot resume;"
                    f" resumable: {sorted(RESUMABLE_STATES)}"
                )
            conn.execute(
                "UPDATE jobs SET state = 'queued', cancel_requested = 0,"
                " error_code = NULL, error_message = NULL, updated_at = ?"
                " WHERE id = ?",
                (now, clean),
            )
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        return self.get_job(clean)

    def claim_job(self, job_id: str, worker_id: str) -> JobRecord | None:
        clean = _validate_job_id(job_id)
        worker = _validate_worker_id(worker_id)
        now = utc_now_iso()
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            cursor = conn.execute(
                "UPDATE jobs SET state = 'running', cancel_requested = 0,"
                " error_code = NULL, error_message = NULL, updated_at = ?"
                " WHERE id = ? AND state = 'queued'",
                (now, clean),
            )
            if cursor.rowcount == 0:
                conn.execute("ROLLBACK;")
                return None
            latest = conn.execute(
                "SELECT attempt_number FROM job_attempts WHERE job_id = ?"
                " ORDER BY attempt_number DESC LIMIT 1",
                (clean,),
            ).fetchone()
            next_attempt = int(latest["attempt_number"]) + 1 if latest else 1
            conn.execute(
                "INSERT INTO job_attempts(job_id, attempt_number, worker_id,"
                " started_at, heartbeat_at) VALUES (?, ?, ?, ?, ?)",
                (clean, next_attempt, worker, now, now),
            )
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        return self.get_job(clean)

    def claim_next_job(self, worker_id: str, *, kinds: list[str] | None = None) -> JobRecord | None:
        worker = _validate_worker_id(worker_id)
        clean_kinds = None
        if kinds is not None:
            clean_kinds = [_validate_kind(k) for k in kinds]
            if not clean_kinds:
                raise InvalidInputError("kinds must be nonempty when supplied")
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            if clean_kinds is None:
                row = conn.execute(
                    "SELECT id FROM jobs WHERE state = 'queued' ORDER BY created_at ASC LIMIT 1"
                ).fetchone()
            else:
                placeholders = ",".join("?" for _ in clean_kinds)
                row = conn.execute(
                    f"SELECT id FROM jobs WHERE state = 'queued'"
                    f" AND kind IN ({placeholders}) ORDER BY created_at ASC LIMIT 1",
                    tuple(clean_kinds),
                ).fetchone()
            if row is None:
                conn.execute("ROLLBACK;")
                return None
            job_id = str(row["id"])
            conn.execute("ROLLBACK;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        return self.claim_job(job_id, worker)

    def heartbeat(self, job_id: str, worker_id: str) -> None:
        clean = _validate_job_id(job_id)
        worker = _validate_worker_id(worker_id)
        now = utc_now_iso()
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            latest = conn.execute(
                "SELECT id, worker_id, ended_at FROM job_attempts WHERE job_id = ?"
                " ORDER BY attempt_number DESC LIMIT 1",
                (clean,),
            ).fetchone()
            if latest is None:
                raise InvalidInputError(f"job {clean!r} has no attempts to heartbeat")
            if str(latest["worker_id"]) != worker:
                raise InvalidInputError(
                    f"worker {worker!r} does not own job {clean!r}"
                    f" (owner {str(latest['worker_id'])!r})"
                )
            if latest["ended_at"] is not None:
                raise InvalidInputError(f"latest attempt for job {clean!r} already ended")
            conn.execute(
                "UPDATE job_attempts SET heartbeat_at = ? WHERE id = ?",
                (now, int(latest["id"])),
            )
            conn.execute("UPDATE jobs SET updated_at = ? WHERE id = ?", (now, clean))
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()

    def update_progress(
        self,
        job_id: str,
        *,
        progress_done: int,
        progress_total: int | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> JobRecord:
        clean = _validate_job_id(job_id)
        if not isinstance(progress_done, int) or isinstance(progress_done, bool):
            raise InvalidInputError("progress_done must be an integer")
        if progress_done < 0:
            raise InvalidInputError("progress_done must be non-negative")
        if progress_total is not None and (
            not isinstance(progress_total, int) or isinstance(progress_total, bool)
        ):
            raise InvalidInputError("progress_total must be an integer")
        if progress_total is not None and progress_total < 0:
            raise InvalidInputError("progress_total must be non-negative")
        if checkpoint is not None:
            try:
                json.dumps(checkpoint, sort_keys=True)
            except (TypeError, ValueError) as exc:
                raise InvalidInputError(f"checkpoint must be JSON-serializable: {exc}") from exc
        now = utc_now_iso()
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (clean,)).fetchone()
            if row is None:
                raise InvalidInputError(f"job {clean!r} does not exist")
            total = (
                int(progress_total) if progress_total is not None else int(row["progress_total"])
            )
            if progress_done > total:
                raise InvalidInputError("progress_done cannot exceed progress_total")
            conn.execute(
                "UPDATE jobs SET progress_done = ?, progress_total = ?,"
                " checkpoint_json = COALESCE(?, checkpoint_json), updated_at = ?"
                " WHERE id = ?",
                (
                    int(progress_done),
                    int(total),
                    json.dumps(checkpoint, sort_keys=True, ensure_ascii=False)
                    if checkpoint is not None
                    else None,
                    now,
                    clean,
                ),
            )
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        return self.get_job(clean)

    def _finish_running(
        self,
        job_id: str,
        *,
        state: JobState,
        error_code: str | None,
        error_message: str | None,
    ) -> JobRecord:
        clean = _validate_job_id(job_id)
        now = utc_now_iso()
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (clean,)).fetchone()
            if row is None:
                raise InvalidInputError(f"job {clean!r} does not exist")
            if str(row["state"]) not in ("running", "cancelling"):
                raise InvalidInputError(
                    f"job {clean!r} in state {str(row['state'])!r} cannot finish as {state!r}"
                )
            conn.execute(
                "UPDATE jobs SET state = ?, error_code = ?, error_message = ?,"
                " updated_at = ? WHERE id = ?",
                (state, error_code, error_message, now, clean),
            )
            latest = conn.execute(
                "SELECT id FROM job_attempts WHERE job_id = ? AND ended_at IS NULL"
                " ORDER BY attempt_number DESC LIMIT 1",
                (clean,),
            ).fetchone()
            if latest is not None:
                conn.execute(
                    "UPDATE job_attempts SET ended_at = ?, error_code = ?,"
                    " error_message = ? WHERE id = ?",
                    (now, error_code, error_message, int(latest["id"])),
                )
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        return self.get_job(clean)

    def complete_job(self, job_id: str) -> JobRecord:
        return self._finish_running(job_id, state="completed", error_code=None, error_message=None)

    def fail_job(self, job_id: str, *, error_code: str, error_message: str) -> JobRecord:
        if not error_code.strip() or not error_message.strip():
            raise InvalidInputError("fail_job needs error_code and error_message")
        return self._finish_running(
            job_id,
            state="failed",
            error_code=error_code.strip(),
            error_message=" ".join(error_message.split())[:500],
        )

    def pause_budget_job(self, job_id: str, *, error_message: str) -> JobRecord:
        if not error_message.strip():
            raise InvalidInputError("pause_budget_job needs an error_message")
        return self._finish_running(
            job_id,
            state="paused_budget",
            error_code="BUDGET_EXCEEDED",
            error_message=" ".join(error_message.split())[:500],
        )

    def cancel_finish_job(self, job_id: str, *, error_message: str = "Cancelled") -> JobRecord:
        return self._finish_running(
            job_id,
            state="cancelled",
            error_code="CANCELLED",
            error_message=" ".join(error_message.split())[:500] or "Cancelled",
        )

    def get_attempts(self, job_id: str) -> list[JobAttempt]:
        clean = _validate_job_id(job_id)
        conn = self._open()
        try:
            rows = conn.execute(
                "SELECT * FROM job_attempts WHERE job_id = ? ORDER BY attempt_number ASC",
                (clean,),
            ).fetchall()
            return [_row_to_attempt(row) for row in rows]
        finally:
            conn.close()

    def recover_stale_jobs(self, lease_timeout_s: float = DEFAULT_LEASE_TIMEOUT_S) -> list[str]:
        if (
            not isinstance(lease_timeout_s, (int, float))
            or isinstance(lease_timeout_s, bool)
            or lease_timeout_s <= 0
        ):
            raise InvalidInputError("lease_timeout_s must be positive")
        now = datetime.now(UTC)
        conn = self._open()
        recovered: list[str] = []
        try:
            conn.execute("BEGIN IMMEDIATE;")
            rows = conn.execute(
                "SELECT j.id AS job_id, a.id AS attempt_id, a.heartbeat_at AS heartbeat_at"
                " FROM jobs j JOIN job_attempts a ON a.job_id = j.id"
                " WHERE j.state IN ('running','cancelling') AND a.ended_at IS NULL"
            ).fetchall()
            for row in rows:
                try:
                    beat = datetime.fromisoformat(str(row["heartbeat_at"]))
                    if beat.tzinfo is None:
                        beat = beat.replace(tzinfo=UTC)
                except ValueError:
                    beat = datetime.fromtimestamp(0, UTC)
                age_s = (now - beat).total_seconds()
                if age_s < float(lease_timeout_s):
                    continue
                job_id = str(row["job_id"])
                stamp = utc_now_iso()
                conn.execute(
                    "UPDATE jobs SET state = 'interrupted',"
                    " error_code = 'INTERRUPTED',"
                    " error_message = 'Worker lease expired; resume to continue',"
                    " updated_at = ? WHERE id = ?",
                    (stamp, job_id),
                )
                conn.execute(
                    "UPDATE job_attempts SET ended_at = ?, error_code = 'INTERRUPTED',"
                    " error_message = 'Worker lease expired' WHERE id = ?",
                    (stamp, int(row["attempt_id"])),
                )
                recovered.append(job_id)
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        return sorted(recovered)


# -- service-contract helpers (BACKEND-SCHEMA) -------------------------------


def enqueue_index(
    store: JobStore,
    *,
    snapshot_id: str,
    stage: str,
    budget: dict[str, Any] | Any,
    idempotency_key: str | None = None,
    items: list[dict[str, Any]] | None = None,
) -> JobRef:
    """Enqueue a vector/graph index build (idempotent for active jobs)."""
    if stage not in ("vector", "graph"):
        raise InvalidInputError(f"stage must be 'vector' or 'graph', got {stage!r}")
    kind = f"index_{stage}"
    if hasattr(budget, "model_dump"):
        budget_payload = budget.model_dump(mode="json")  # type: ignore[union-attr]
    elif isinstance(budget, dict):
        budget_payload = dict(budget)
    else:
        raise InvalidInputError("budget must be a BudgetConfig or dict")
    config: dict[str, Any] = {"stage": stage, "budget": budget_payload}
    if items is not None:
        if not isinstance(items, list):
            raise InvalidInputError("items must be a list when supplied")
        config["items"] = list(items)
    return store.enqueue_job(kind, config, snapshot_id=snapshot_id, idempotency_key=idempotency_key)


def enqueue_evaluation(
    store: JobStore,
    *,
    snapshot_id: str | None,
    strategy: str,
    split: str,
    config: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
    items: list[dict[str, Any]] | None = None,
) -> JobRef:
    """Enqueue one evaluation arm (idempotent for active jobs)."""
    if not strategy.strip():
        raise InvalidInputError("strategy must be nonempty")
    if not split.strip():
        raise InvalidInputError("split must be nonempty")
    payload = dict(config or {})
    payload.update({"strategy": strategy.strip(), "split": split.strip()})
    if items is not None:
        payload["items"] = list(items)
    return store.enqueue_job(
        "evaluation", payload, snapshot_id=snapshot_id, idempotency_key=idempotency_key
    )


def get_job_status(store: JobStore, job_id: str) -> dict[str, Any]:
    """BACKEND-SCHEMA get_job: progress/error/retryability snapshot."""
    job = store.get_job(job_id)
    attempts = store.get_attempts(job_id)
    retryable = job.state in ("failed", "interrupted", "paused_budget", "cancelled")
    return {
        "job_id": job.id,
        "kind": job.kind,
        "snapshot_id": job.snapshot_id,
        "state": job.state,
        "progress_done": job.progress_done,
        "progress_total": job.progress_total,
        "cancel_requested": job.cancel_requested,
        "error_code": job.error_code,
        "error_message": job.error_message,
        "retryable": retryable,
        "attempts": [a.model_dump(mode="json") for a in attempts],
        "checkpoint": dict(job.checkpoint_json),
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


__all__ = [
    "ACTIVE_STATES",
    "DEFAULT_DB_PATH",
    "DEFAULT_LEASE_TIMEOUT_S",
    "RESUMABLE_STATES",
    "SCHEMA_LABEL",
    "SCHEMA_VERSION",
    "BudgetExceededError",
    "CancelledError",
    "ConfigMismatchError",
    "DependencyUnavailableError",
    "IndexNotReadyError",
    "InvalidInputError",
    "JobAttempt",
    "JobError",
    "JobRecord",
    "JobRef",
    "JobStore",
    "ProviderError",
    "TERMINAL_STATES",
    "ValidationFailedError",
    "canonical_config_hash",
    "enqueue_evaluation",
    "enqueue_index",
    "error_for_code",
    "get_job_status",
    "init_db",
    "utc_now_iso",
]
