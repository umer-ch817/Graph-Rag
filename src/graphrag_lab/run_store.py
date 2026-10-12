"""Run/result persistence for T12a (evaluation, R09/R12).

Shares the SQLite file owned by ``jobs.py``: runs, per-question predictions,
usage events and run comparisons. SQLite marks completion only after durable
artifacts and validations finish; this module never fabricates metrics and
never defaults unknown monetary cost to zero.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from graphrag_lab.jobs import InvalidInputError, _connect, init_db, utc_now_iso

RunState = Literal["queued", "running", "completed", "failed", "cancelled"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class RunRecord(Record):
    id: str = Field(min_length=1)
    snapshot_id: str | None = None
    job_id: str | None = None
    strategy: str = Field(min_length=1)
    split: str | None = None
    config_hash: str = Field(min_length=1)
    config_json: dict[str, Any] = Field(default_factory=dict)
    question_set_hash: str = Field(min_length=1)
    state: RunState = "queued"
    artifact_path: str | None = None
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)


class PredictionRecord(Record):
    run_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    answer: str = ""
    status: str = Field(min_length=1)
    citations_json: list[Any] = Field(default_factory=list)
    retrieval_json: dict[str, Any] | None = None
    latency_ms: float | None = None
    usage_json: dict[str, Any] | None = None
    failure_code: str | None = None


class UsageEvent(Record):
    id: int = Field(ge=0, strict=True)
    job_attempt_id: int | None = None
    run_id: str | None = None
    stage: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    model: str | None = None
    input_tokens: int = Field(ge=0, strict=True)
    output_tokens: int = Field(ge=0, strict=True)
    estimated_cost_usd: float | None = None
    currency: str = "USD"
    pricing_version: str | None = None
    status: str = Field(min_length=1)
    created_at: str = Field(min_length=1)


class ComparisonRecord(Record):
    id: str = Field(min_length=1)
    run_ids: list[str] = Field(min_length=1)
    compatibility_hash: str = Field(min_length=1)
    metrics_json: dict[str, Any] | None = None
    intervals_json: dict[str, Any] | None = None
    report_path: str | None = None
    created_at: str = Field(min_length=1)


def _validate_id(value: str, *, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidInputError(f"{what} must be a nonempty string")
    cleaned = value.strip()
    if len(cleaned) > 128:
        raise InvalidInputError(f"{what} exceeds 128 chars")
    return cleaned


def config_hash_for(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_json(raw: str | None, *, what: str, default: Any) -> Any:
    if raw is None or raw == "":
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        from graphrag_lab.jobs import ValidationFailedError

        raise ValidationFailedError(f"Stored {what} is not valid JSON: {exc}") from exc


class RunStore:
    """Persistence for runs/predictions/usage/comparisons in the jobs database."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        init_db(self._path)

    @property
    def path(self) -> Path:
        return self._path

    def _open(self) -> sqlite3.Connection:
        return _connect(self._path)

    # -- runs ---------------------------------------------------------------

    def create_run(
        self,
        *,
        strategy: str,
        question_set_hash: str,
        config: dict[str, Any] | None = None,
        snapshot_id: str | None = None,
        job_id: str | None = None,
        split: str | None = None,
        run_id: str | None = None,
    ) -> RunRecord:
        if not strategy.strip():
            raise InvalidInputError("strategy must be nonempty")
        if not question_set_hash.strip():
            raise InvalidInputError("question_set_hash must be nonempty")
        cfg = dict(config or {})
        try:
            json.dumps(cfg, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise InvalidInputError(f"config must be JSON-serializable: {exc}") from exc
        new_id = (
            run_id.strip()
            if isinstance(run_id, str) and run_id.strip()
            else f"run_{uuid.uuid4().hex[:12]}"
        )
        _validate_id(new_id, what="run_id")
        snap = snapshot_id.strip() if isinstance(snapshot_id, str) and snapshot_id.strip() else None
        job = job_id.strip() if isinstance(job_id, str) and job_id.strip() else None
        hashed = config_hash_for({"strategy": strategy.strip(), "config": cfg})
        now = utc_now_iso()
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            if snap is not None:
                exists = conn.execute(
                    "SELECT id FROM index_snapshots WHERE id = ?", (snap,)
                ).fetchone()
                if exists is None:
                    raise InvalidInputError(f"snapshot {snap!r} does not exist")
            if job is not None:
                exists = conn.execute("SELECT id FROM jobs WHERE id = ?", (job,)).fetchone()
                if exists is None:
                    raise InvalidInputError(f"job {job!r} does not exist")
            conn.execute(
                "INSERT INTO runs(id, snapshot_id, job_id, strategy, split, config_hash,"
                " config_json, question_set_hash, state, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)",
                (
                    new_id,
                    snap,
                    job,
                    strategy.strip(),
                    split.strip() if isinstance(split, str) and split.strip() else None,
                    hashed,
                    json.dumps(cfg, sort_keys=True, ensure_ascii=False),
                    question_set_hash.strip(),
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
            raise InvalidInputError(f"run violates uniqueness: {exc}") from exc
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        return self.get_run(new_id)

    def get_run(self, run_id: str) -> RunRecord:
        clean = _validate_id(run_id, what="run_id")
        conn = self._open()
        try:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (clean,)).fetchone()
            if row is None:
                raise InvalidInputError(f"run {clean!r} does not exist")
            return RunRecord(
                id=str(row["id"]),
                snapshot_id=str(row["snapshot_id"]) if row["snapshot_id"] is not None else None,
                job_id=str(row["job_id"]) if row["job_id"] is not None else None,
                strategy=str(row["strategy"]),
                split=str(row["split"]) if row["split"] is not None else None,
                config_hash=str(row["config_hash"]),
                config_json=_parse_json(row["config_json"], what="config_json", default={}),
                question_set_hash=str(row["question_set_hash"]),
                state=str(row["state"]),  # type: ignore[arg-type]
                artifact_path=str(row["artifact_path"])
                if row["artifact_path"] is not None
                else None,
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
            )
        finally:
            conn.close()

    def list_runs(self, *, snapshot_id: str | None = None, limit: int = 100) -> list[RunRecord]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise InvalidInputError("limit must be an integer in 1..1000")
        conn = self._open()
        try:
            if snapshot_id is None:
                rows = conn.execute(
                    "SELECT * FROM runs ORDER BY created_at ASC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE snapshot_id = ? ORDER BY created_at ASC LIMIT ?",
                    (snapshot_id.strip(), limit),
                ).fetchall()
            out: list[RunRecord] = []
            for row in rows:
                out.append(
                    RunRecord(
                        id=str(row["id"]),
                        snapshot_id=str(row["snapshot_id"])
                        if row["snapshot_id"] is not None
                        else None,
                        job_id=str(row["job_id"]) if row["job_id"] is not None else None,
                        strategy=str(row["strategy"]),
                        split=str(row["split"]) if row["split"] is not None else None,
                        config_hash=str(row["config_hash"]),
                        config_json=_parse_json(row["config_json"], what="config_json", default={}),
                        question_set_hash=str(row["question_set_hash"]),
                        state=str(row["state"]),  # type: ignore[arg-type]
                        artifact_path=str(row["artifact_path"])
                        if row["artifact_path"] is not None
                        else None,
                        created_at=str(row["created_at"]),
                        updated_at=str(row["updated_at"]),
                    )
                )
            return out
        finally:
            conn.close()

    def update_run_state(
        self, run_id: str, state: RunState, *, artifact_path: str | None = None
    ) -> RunRecord:
        clean = _validate_id(run_id, what="run_id")
        if state not in ("queued", "running", "completed", "failed", "cancelled"):
            raise InvalidInputError(f"unknown run state {state!r}")
        if state == "completed" and artifact_path is not None:
            if not Path(artifact_path).exists():
                raise InvalidInputError(
                    f"run {clean!r} cannot complete: artifact {artifact_path!r} is missing;"
                    " persist durable artifacts first"
                )
        now = utc_now_iso()
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            exists = conn.execute("SELECT id FROM runs WHERE id = ?", (clean,)).fetchone()
            if exists is None:
                raise InvalidInputError(f"run {clean!r} does not exist")
            conn.execute(
                "UPDATE runs SET state = ?, artifact_path = COALESCE(?, artifact_path),"
                " updated_at = ? WHERE id = ?",
                (state, artifact_path, now, clean),
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
        return self.get_run(clean)

    # -- predictions ----------------------------------------------------------

    def upsert_prediction(
        self,
        run_id: str,
        question_id: str,
        *,
        answer: str = "",
        status: str,
        citations: list[Any] | None = None,
        retrieval: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        usage: dict[str, Any] | None = None,
        failure_code: str | None = None,
    ) -> PredictionRecord:
        run = _validate_id(run_id, what="run_id")
        question = _validate_id(question_id, what="question_id")
        if status not in ("supported", "insufficient_evidence", "failed"):
            raise InvalidInputError(f"unknown prediction status {status!r}")
        if latency_ms is not None and (not isinstance(latency_ms, (int, float)) or latency_ms < 0):
            raise InvalidInputError("latency_ms must be non-negative when supplied")
        for label, payload in (
            ("citations", citations),
            ("retrieval", retrieval),
            ("usage", usage),
        ):
            if payload is not None:
                try:
                    json.dumps(payload)
                except (TypeError, ValueError) as exc:
                    raise InvalidInputError(f"{label} must be JSON-serializable: {exc}") from exc
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            exists = conn.execute("SELECT id FROM runs WHERE id = ?", (run,)).fetchone()
            if exists is None:
                raise InvalidInputError(f"run {run!r} does not exist")
            conn.execute(
                "INSERT INTO predictions(run_id, question_id, answer, status, citations_json,"
                " retrieval_json, latency_ms, usage_json, failure_code)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(run_id, question_id) DO UPDATE SET"
                " answer = excluded.answer, status = excluded.status,"
                " citations_json = excluded.citations_json,"
                " retrieval_json = excluded.retrieval_json,"
                " latency_ms = excluded.latency_ms, usage_json = excluded.usage_json,"
                " failure_code = excluded.failure_code",
                (
                    run,
                    question,
                    answer,
                    status,
                    json.dumps(citations or [], ensure_ascii=False, sort_keys=True),
                    json.dumps(retrieval, ensure_ascii=False, sort_keys=True)
                    if retrieval is not None
                    else None,
                    float(latency_ms) if latency_ms is not None else None,
                    json.dumps(usage, ensure_ascii=False, sort_keys=True)
                    if usage is not None
                    else None,
                    failure_code,
                ),
            )
            conn.execute("UPDATE runs SET updated_at = ? WHERE id = ?", (utc_now_iso(), run))
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        return self.get_prediction(run, question)

    def get_prediction(self, run_id: str, question_id: str) -> PredictionRecord:
        run = _validate_id(run_id, what="run_id")
        question = _validate_id(question_id, what="question_id")
        conn = self._open()
        try:
            row = conn.execute(
                "SELECT * FROM predictions WHERE run_id = ? AND question_id = ?",
                (run, question),
            ).fetchone()
            if row is None:
                raise InvalidInputError(f"prediction {run!r}/{question!r} does not exist")
            return PredictionRecord(
                run_id=str(row["run_id"]),
                question_id=str(row["question_id"]),
                answer=str(row["answer"]),
                status=str(row["status"]),
                citations_json=_parse_json(
                    row["citations_json"], what="citations_json", default=[]
                ),
                retrieval_json=_parse_json(
                    row["retrieval_json"], what="retrieval_json", default=None
                ),
                latency_ms=float(row["latency_ms"]) if row["latency_ms"] is not None else None,
                usage_json=_parse_json(row["usage_json"], what="usage_json", default=None),
                failure_code=str(row["failure_code"]) if row["failure_code"] is not None else None,
            )
        finally:
            conn.close()

    def list_predictions(self, run_id: str) -> list[PredictionRecord]:
        run = _validate_id(run_id, what="run_id")
        conn = self._open()
        try:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE run_id = ? ORDER BY question_id ASC", (run,)
            ).fetchall()
            return [
                PredictionRecord(
                    run_id=str(row["run_id"]),
                    question_id=str(row["question_id"]),
                    answer=str(row["answer"]),
                    status=str(row["status"]),
                    citations_json=_parse_json(
                        row["citations_json"], what="citations_json", default=[]
                    ),
                    retrieval_json=_parse_json(
                        row["retrieval_json"], what="retrieval_json", default=None
                    ),
                    latency_ms=float(row["latency_ms"]) if row["latency_ms"] is not None else None,
                    usage_json=_parse_json(row["usage_json"], what="usage_json", default=None),
                    failure_code=str(row["failure_code"])
                    if row["failure_code"] is not None
                    else None,
                )
                for row in rows
            ]
        finally:
            conn.close()

    # -- usage events ----------------------------------------------------------

    def record_usage(
        self,
        *,
        stage: str,
        request_id: str,
        status: str,
        job_attempt_id: int | None = None,
        run_id: str | None = None,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost_usd: float | None = None,
        currency: str = "USD",
        pricing_version: str | None = None,
    ) -> UsageEvent:
        if not stage.strip() or not request_id.strip() or not status.strip():
            raise InvalidInputError("stage/request_id/status must be nonempty")
        for label, value in (("input_tokens", input_tokens), ("output_tokens", output_tokens)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise InvalidInputError(f"{label} must be a non-negative integer")
        if estimated_cost_usd is not None and (
            not isinstance(estimated_cost_usd, (int, float)) or estimated_cost_usd < 0
        ):
            raise InvalidInputError("estimated_cost_usd must be non-negative when supplied")
        # Unknown cost stays None; never default to zero. Callers must pass an
        # explicit zero only when the provider reports zero.
        now = utc_now_iso()
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            cursor = conn.execute(
                "INSERT INTO usage_events(job_attempt_id, run_id, stage, request_id, model,"
                " input_tokens, output_tokens, estimated_cost_usd, currency,"
                " pricing_version, status, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_attempt_id,
                    run_id.strip() if isinstance(run_id, str) and run_id.strip() else None,
                    stage.strip(),
                    request_id.strip(),
                    model.strip() if isinstance(model, str) and model.strip() else None,
                    int(input_tokens),
                    int(output_tokens),
                    float(estimated_cost_usd) if estimated_cost_usd is not None else None,
                    currency.strip() or "USD",
                    pricing_version,
                    status.strip(),
                    now,
                ),
            )
            event_id = int(cursor.lastrowid)
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        return self.get_usage_event(event_id)

    def get_usage_event(self, event_id: int) -> UsageEvent:
        conn = self._open()
        try:
            row = conn.execute(
                "SELECT * FROM usage_events WHERE id = ?", (int(event_id),)
            ).fetchone()
            if row is None:
                raise InvalidInputError(f"usage event {event_id!r} does not exist")
            return UsageEvent(
                id=int(row["id"]),
                job_attempt_id=int(row["job_attempt_id"])
                if row["job_attempt_id"] is not None
                else None,
                run_id=str(row["run_id"]) if row["run_id"] is not None else None,
                stage=str(row["stage"]),
                request_id=str(row["request_id"]),
                model=str(row["model"]) if row["model"] is not None else None,
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                estimated_cost_usd=float(row["estimated_cost_usd"])
                if row["estimated_cost_usd"] is not None
                else None,
                currency=str(row["currency"]),
                pricing_version=str(row["pricing_version"])
                if row["pricing_version"] is not None
                else None,
                status=str(row["status"]),
                created_at=str(row["created_at"]),
            )
        finally:
            conn.close()

    def sum_usage_for_run(self, run_id: str) -> dict[str, Any]:
        run = _validate_id(run_id, what="run_id")
        conn = self._open()
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(input_tokens),0) AS inp,"
                " COALESCE(SUM(output_tokens),0) AS out,"
                " SUM(estimated_cost_usd) AS cost, COUNT(*) AS n"
                " FROM usage_events WHERE run_id = ?",
                (run,),
            ).fetchone()
            assert row is not None
            return {
                "run_id": run,
                "events": int(row["n"]),
                "input_tokens": int(row["inp"]),
                "output_tokens": int(row["out"]),
                "estimated_cost_usd": float(row["cost"]) if row["cost"] is not None else None,
            }
        finally:
            conn.close()

    # -- comparisons ------------------------------------------------------------

    def create_comparison(
        self,
        *,
        run_ids: list[str],
        compatibility_hash: str,
        metrics: dict[str, Any] | None = None,
        intervals: dict[str, Any] | None = None,
        report_path: str | None = None,
        comparison_id: str | None = None,
    ) -> ComparisonRecord:
        if not isinstance(run_ids, list) or len(run_ids) < 1:
            raise InvalidInputError("run_ids must be a nonempty list")
        cleaned = [_validate_id(v, what="run_id") for v in run_ids]
        if len(set(cleaned)) != len(cleaned):
            raise InvalidInputError("run_ids must not repeat")
        if not compatibility_hash.strip():
            raise InvalidInputError("compatibility_hash must be nonempty")
        for label, payload in (("metrics", metrics), ("intervals", intervals)):
            if payload is not None:
                try:
                    json.dumps(payload)
                except (TypeError, ValueError) as exc:
                    raise InvalidInputError(f"{label} must be JSON-serializable: {exc}") from exc
        new_id = (
            comparison_id.strip()
            if isinstance(comparison_id, str) and comparison_id.strip()
            else f"cmp_{uuid.uuid4().hex[:12]}"
        )
        _validate_id(new_id, what="comparison_id")
        now = utc_now_iso()
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            for rid in cleaned:
                exists = conn.execute("SELECT id FROM runs WHERE id = ?", (rid,)).fetchone()
                if exists is None:
                    raise InvalidInputError(f"run {rid!r} does not exist")
            conn.execute(
                "INSERT INTO comparisons(id, run_ids_json, compatibility_hash,"
                " metrics_json, intervals_json, report_path, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    new_id,
                    json.dumps(sorted(cleaned), ensure_ascii=False),
                    compatibility_hash.strip(),
                    json.dumps(metrics, sort_keys=True, ensure_ascii=False)
                    if metrics is not None
                    else None,
                    json.dumps(intervals, sort_keys=True, ensure_ascii=False)
                    if intervals is not None
                    else None,
                    report_path,
                    now,
                ),
            )
            conn.execute("COMMIT;")
        except sqlite3.IntegrityError as exc:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise InvalidInputError(f"comparison violates uniqueness: {exc}") from exc
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        return self.get_comparison(new_id)

    def get_comparison(self, comparison_id: str) -> ComparisonRecord:
        clean = _validate_id(comparison_id, what="comparison_id")
        conn = self._open()
        try:
            row = conn.execute("SELECT * FROM comparisons WHERE id = ?", (clean,)).fetchone()
            if row is None:
                raise InvalidInputError(f"comparison {clean!r} does not exist")
            return ComparisonRecord(
                id=str(row["id"]),
                run_ids=list(json.loads(str(row["run_ids_json"]))),
                compatibility_hash=str(row["compatibility_hash"]),
                metrics_json=_parse_json(row["metrics_json"], what="metrics_json", default=None),
                intervals_json=_parse_json(
                    row["intervals_json"], what="intervals_json", default=None
                ),
                report_path=str(row["report_path"]) if row["report_path"] is not None else None,
                created_at=str(row["created_at"]),
            )
        finally:
            conn.close()


__all__ = [
    "ComparisonRecord",
    "PredictionRecord",
    "RunRecord",
    "RunState",
    "RunStore",
    "UsageEvent",
    "config_hash_for",
]
