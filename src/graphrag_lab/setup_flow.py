"""Setup-flow view model for T12b (dashboard, R11).

Pure logic behind the Dataset & Index dashboard: readiness, the single
Overview next action, persisted index progress/spend, and the duplicate
submission guard. No Streamlit import here so ``tests/e2e/test_setup_flow.py``
runs without the UI dependency; ``app/`` owns presentation and calls this
module plus the durable ``jobs``/``run_store`` services.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from graphrag_lab.jobs import JobRecord, JobStore

SetupState = Literal[
    "missing",
    "blocked",
    "queued",
    "running",
    "cancelling",
    "paused_budget",
    "failed",
    "cancelled",
    "interrupted",
    "ready",
    "partial",
]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class SectionStatus(Record):
    """One guided section: Dataset, Vector index, Graph pilot & build, Ready."""

    name: str = Field(min_length=1)
    state: SetupState = "missing"
    headline: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    job_id: str | None = None
    progress_done: int = Field(default=0, ge=0, strict=True)
    progress_total: int = Field(default=0, ge=0, strict=True)
    failed_items: int = Field(default=0, ge=0, strict=True)
    attempts: int = Field(default=0, ge=0, strict=True)
    retryable: bool = False
    action_label: str = Field(min_length=1)
    creates_new_snapshot: bool = False


class NextAction(Record):
    label: str = Field(min_length=1)
    target: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class SetupSummary(Record):
    dataset: SectionStatus
    vector: SectionStatus
    graph: SectionStatus
    ready: SectionStatus
    next_action: NextAction
    worker_seen: bool = False
    worker_message: str = Field(min_length=1)
    neo4j_ok: bool = True
    neo4j_message: str = Field(min_length=1)


class SpendSummary(Record):
    budget_calls: int | None = None
    budget_tokens: int | None = None
    budget_cost_usd: float | None = None
    used_items: int = Field(ge=0, strict=True)
    failed_items: int = Field(ge=0, strict=True)
    total_items: int = Field(ge=0, strict=True)
    estimated_cost_usd: float | None = None
    cost_label: str = Field(min_length=1)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def worker_status(
    last_heartbeat_iso: str | None,
    lease_timeout_s: float = 30.0,
    *,
    now_iso: str | None = None,
) -> dict[str, Any]:
    """Classify worker liveness without touching the network."""
    if not isinstance(lease_timeout_s, (int, float)) or lease_timeout_s <= 0:
        raise ValueError("lease_timeout_s must be positive")
    now = _parse_iso(now_iso) or datetime.now(UTC)
    beat = _parse_iso(last_heartbeat_iso)
    if beat is None:
        return {
            "seen": False,
            "state": "never",
            "message": (
                "Worker has not reported yet. Jobs stay queued. "
                "Start it with: python -m graphrag_lab.cli worker"
            ),
        }
    age_s = (now - beat).total_seconds()
    if age_s < 0:
        age_s = 0.0
    if age_s <= float(lease_timeout_s):
        return {
            "seen": True,
            "state": "ready",
            "age_s": round(age_s, 1),
            "message": f"Worker heartbeat {round(age_s, 1)}s ago. Ready.",
        }
    return {
        "seen": False,
        "state": "stale",
        "age_s": round(age_s, 1),
        "message": (
            f"Worker heartbeat expired ({round(age_s, 1)}s ago). "
            "Interrupted jobs keep their checkpoints; resume after restarting "
            "the worker with: python -m graphrag_lab.cli worker"
        ),
    }


def neo4j_message(uri: str, *, ok: bool, detail: str = "") -> str:
    if ok:
        return f"Neo4j reachable at {uri}. Queued work can run."
    hint = detail.strip() or "connection refused"
    return (
        f"Neo4j unavailable at {uri} ({hint}). Queued work is kept; "
        "start Docker (`docker compose up -d --wait neo4j`) then resume. "
        "No completed work is lost."
    )


def idempotency_key_for(*parts: str) -> str:
    """Stable 12-hex submission key for one logical dashboard submission."""
    canonical = json.dumps(list(parts), ensure_ascii=False, separators=(",", ":")).encode()
    return f"ui-{hashlib.sha256(canonical).hexdigest()[:12]}"


def _job_section(
    name: str,
    jobs: list[JobRecord],
    *,
    empty_headline: str,
    empty_reason: str,
    empty_action: str,
    build_action: str,
    attempts: int = 0,
) -> SectionStatus:
    if not jobs:
        return SectionStatus(
            name=name,
            state="missing",
            headline=empty_headline,
            reason=empty_reason,
            action_label=empty_action,
        )
    # Newest first: the dashboard surfaces the latest job's state.
    ordered = sorted(jobs, key=lambda job: job.created_at, reverse=True)
    job = ordered[0]
    checkpoint = dict(job.checkpoint_json or {})
    failed = checkpoint.get("failed", {})
    n_failed = len(failed) if isinstance(failed, dict) else 0
    state_map: dict[str, SetupState] = {
        "queued": "queued",
        "running": "running",
        "cancelling": "cancelling",
        "paused_budget": "paused_budget",
        "failed": "failed",
        "cancelled": "cancelled",
        "interrupted": "interrupted",
        "completed": "ready",
    }
    state = state_map.get(job.state, "blocked")
    retryable = job.state in ("failed", "interrupted", "paused_budget", "cancelled")
    if job.state == "completed":
        return SectionStatus(
            name=name,
            state="ready",
            headline=f"{name} ready.",
            reason=f"Job {job.id} completed {job.progress_done}/{job.progress_total} items.",
            job_id=job.id,
            progress_done=job.progress_done,
            progress_total=job.progress_total,
            failed_items=0,
            attempts=attempts,
            retryable=False,
            action_label="Rebuild (creates a new snapshot)",
        )
    headlines = {
        "queued": f"{name} queued.",
        "running": f"{name} running.",
        "cancelling": f"{name} cancelling.",
        "paused_budget": f"{name} paused on budget.",
        "failed": f"{name} failed.",
        "cancelled": f"{name} cancelled.",
        "interrupted": f"{name} interrupted.",
    }
    reasons = {
        "queued": f"Job {job.id} waits for the worker. Progress is persisted.",
        "running": (
            f"Job {job.id}: {job.progress_done}/{job.progress_total} items. "
            "Cancel is a secondary action; completed calls are kept."
        ),
        "cancelling": (
            f"Job {job.id} stops at the next item boundary. Validated work is preserved."
        ),
        "paused_budget": (
            f"Job {job.id} hit its call/token cap "
            f"({job.progress_done}/{job.progress_total} items kept). "
            "Revise the budget, then resume."
        ),
        "failed": (
            f"Job {job.id} has {n_failed} failed item(s); "
            f"{job.progress_done}/{job.progress_total} processed. "
            f"Reason: {job.error_message or job.error_code or 'see job record'}. "
            "Settings are preserved; retry failed items only."
        ),
        "cancelled": (
            f"Job {job.id} was cancelled at "
            f"{job.progress_done}/{job.progress_total} items. Validated work is kept."
        ),
        "interrupted": (
            f"Job {job.id} lost its worker lease at "
            f"{job.progress_done}/{job.progress_total} items. "
            "Resume reuses validated entries in a new attempt."
        ),
    }
    actions = {
        "queued": "View progress",
        "running": "View progress",
        "cancelling": "View progress",
        "paused_budget": "Revise budget and resume",
        "failed": "Retry failed chunks",
        "cancelled": "Resume from checkpoint",
        "interrupted": "Resume from checkpoint",
    }
    key = job.state if job.state in headlines else "queued"
    return SectionStatus(
        name=name,
        state=state,
        headline=headlines[key],
        reason=reasons[key],
        job_id=job.id,
        progress_done=job.progress_done,
        progress_total=job.progress_total,
        failed_items=n_failed,
        attempts=attempts,
        retryable=retryable,
        action_label=actions.get(key, build_action),
    )


def summarize_setup(
    store: JobStore,
    *,
    worker_seen: bool = False,
    worker_message: str = "Worker status unknown.",
    neo4j_ok: bool = True,
    neo4j_uri: str = "bolt://127.0.0.1:7687",
    neo4j_detail: str = "",
) -> SetupSummary:
    """Build the four guided sections plus the single Overview next action."""
    jobs = store.list_jobs(limit=200)
    vector_jobs = [job for job in jobs if job.kind in ("index_vector", "demo")]
    graph_jobs = [job for job in jobs if job.kind in ("index_graph",)]

    def _attempts(candidate: list[JobRecord]) -> int:
        if not candidate:
            return 0
        latest = sorted(candidate, key=lambda job: job.created_at, reverse=True)[0]
        try:
            return len(store.get_attempts(latest.id))
        except Exception:
            return 0

    vector_attempts = _attempts(vector_jobs)
    graph_attempts = _attempts(graph_jobs)
    datasets = _list_datasets(store)
    if not datasets:
        dataset = SectionStatus(
            name="Dataset",
            state="missing",
            headline="No corpus yet.",
            reason="Start with a reproducible question set (default 150, seed 42).",
            action_label="Prepare dataset",
        )
    else:
        latest = datasets[0]
        dataset = SectionStatus(
            name="Dataset",
            state="ready",
            headline=f"Dataset {latest['id']} ready.",
            reason=(
                f"{latest['sample_count']} questions, seed {latest['seed']}. "
                f"Manifest: {latest['manifest_path']}."
            ),
            action_label="Prepare a new sample (creates a new snapshot)",
        )
    vector = _job_section(
        "Vector index",
        vector_jobs,
        empty_headline="Vector index not built.",
        empty_reason=(
            "Build chunk vectors first."
            if datasets
            else "Prepare a dataset first; vectors need corpus chunks."
        ),
        empty_action="Build vector index",
        build_action="Build vector index",
        attempts=vector_attempts,
    )
    if not datasets:
        vector = vector.model_copy(
            update={
                "state": "blocked",
                "reason": "Prepare a dataset first; vectors need corpus chunks.",
            }
        )
    graph = _job_section(
        "Graph pilot & build",
        graph_jobs,
        empty_headline="Graph not built.",
        empty_reason=(
            "Audit the 20-chunk pilot, then build the graph."
            if vector.state == "ready"
            else "Build the vector index first; the graph needs resolved chunks."
        ),
        empty_action="Run graph pilot",
        build_action="Build graph",
        attempts=graph_attempts,
    )
    if vector.state != "ready":
        graph = graph.model_copy(
            update={
                "state": "blocked" if not graph_jobs else graph.state,
                "reason": (
                    "Build the vector index first; the graph needs resolved chunks."
                    if not graph_jobs
                    else graph.reason
                ),
            }
        )
    vector_ready = vector.state == "ready"
    graph_ready = graph.state == "ready"
    if vector_ready and graph_ready:
        ready = SectionStatus(
            name="Ready to compare",
            state="ready",
            headline="Ready to compare.",
            reason="Vector and graph indexes are ready. Ask a question to compare arms.",
            action_label="Ask a question",
        )
    elif vector_ready:
        ready = SectionStatus(
            name="Ready to compare",
            state="ready",
            headline="Vector ready; graph unavailable.",
            reason="Vector modes are enabled. Build the graph for hybrid comparison.",
            action_label="Ask a question (vector only)",
        )
    else:
        ready = SectionStatus(
            name="Ready to compare",
            state="blocked",
            headline="Not ready to compare.",
            reason="Complete the dataset and vector steps first.",
            action_label="Continue setup",
        )
    next_action = _next_action(
        has_dataset=bool(datasets),
        vector=vector,
        graph=graph,
        ready=ready,
    )
    return SetupSummary(
        dataset=dataset,
        vector=vector,
        graph=graph,
        ready=ready,
        next_action=next_action,
        worker_seen=worker_seen,
        worker_message=worker_message,
        neo4j_ok=neo4j_ok,
        neo4j_message=neo4j_message(neo4j_uri, ok=neo4j_ok, detail=neo4j_detail),
    )


def _next_action(
    *, has_dataset: bool, vector: SectionStatus, graph: SectionStatus, ready: SectionStatus
) -> NextAction:
    if not has_dataset:
        return NextAction(
            label="Prepare dataset",
            target="dataset",
            reason="Start with a reproducible question set.",
        )
    if vector.state in ("missing", "blocked"):
        return NextAction(
            label="Build vector index",
            target="dataset",
            reason="Dataset is ready; build chunk vectors next.",
        )
    if vector.state in ("queued", "running", "cancelling"):
        return NextAction(
            label="View indexing progress",
            target="dataset",
            reason=f"Vector job {vector.job_id} is {vector.state}.",
        )
    if vector.state in ("failed", "cancelled", "interrupted", "paused_budget"):
        return NextAction(
            label=vector.action_label,
            target="dataset",
            reason=vector.reason,
        )
    if vector.state == "ready" and graph.state in ("missing", "blocked"):
        return NextAction(
            label="Run graph pilot",
            target="dataset",
            reason="Vectors are ready; audit the 20-chunk pilot before the full build.",
        )
    if graph.state in ("queued", "running", "cancelling"):
        return NextAction(
            label="View graph progress",
            target="dataset",
            reason=f"Graph job {graph.job_id} is {graph.state}.",
        )
    if graph.state in ("failed", "cancelled", "interrupted", "paused_budget"):
        return NextAction(
            label=graph.action_label,
            target="dataset",
            reason=graph.reason,
        )
    if ready.state == "ready":
        return NextAction(
            label="Ask a question",
            target="compare",
            reason=ready.reason,
        )
    return NextAction(label="Continue setup", target="dataset", reason=ready.reason)


def _list_datasets(store: JobStore) -> list[dict[str, Any]]:
    import sqlite3 as _sqlite

    conn = _sqlite.connect(str(store.path), timeout=10.0)
    conn.row_factory = _sqlite.Row
    try:
        rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def latest_dataset_id(store: JobStore) -> str | None:
    """Newest dataset id for session defaults (None when no corpus)."""
    rows = _list_datasets(store)
    return str(rows[0]["id"]) if rows else None


def progress_panel(job: JobRecord, *, attempts: int = 0) -> dict[str, Any]:
    """Single progress panel: processed/total, failures, attempts. No fake ETA."""
    checkpoint = dict(job.checkpoint_json or {})
    failed = checkpoint.get("failed", {})
    n_failed = len(failed) if isinstance(failed, dict) else 0
    total = max(0, int(job.progress_done)) if job.progress_total == 0 else job.progress_total
    return {
        "job_id": job.id,
        "kind": job.kind,
        "state": job.state,
        "processed": job.progress_done,
        "total": job.progress_total or total,
        "failed": n_failed,
        "attempts": attempts,
        "error_code": job.error_code,
        "error_message": job.error_message,
        "retryable": job.state in ("failed", "interrupted", "paused_budget", "cancelled"),
        "cancel_requested": job.cancel_requested,
    }


def spend_summary(
    job: JobRecord,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    estimated_cost_usd: float | None = None,
) -> SpendSummary:
    """Separate budget caps from actual spend; unknown money stays unknown."""
    budget = dict(job.config_json.get("budget", {}) or {})
    calls = budget.get("max_calls")
    tokens = budget.get("max_total_tokens")
    cost = budget.get("max_cost_usd")
    checkpoint = dict(job.checkpoint_json or {})
    failed = checkpoint.get("failed", {})
    n_failed = len(failed) if isinstance(failed, dict) else 0
    if estimated_cost_usd is not None:
        label = f"${float(estimated_cost_usd):.6f} (estimated)"
    elif cost is not None:
        label = "unknown (budget set, provider pricing not reported)"
    else:
        label = "unknown (unpriced local compute; no monetary cap set)"
    return SpendSummary(
        budget_calls=int(calls) if isinstance(calls, int) else None,
        budget_tokens=int(tokens) if isinstance(tokens, int) else None,
        budget_cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
        used_items=job.progress_done,
        failed_items=n_failed,
        total_items=job.progress_total,
        estimated_cost_usd=float(estimated_cost_usd) if estimated_cost_usd is not None else None,
        cost_label=label,
    )


def pilot_vs_full_estimate(pilot: dict[str, Any], total_chunks: int) -> dict[str, Any]:
    """Linear pilot extrapolation, labeled as an estimate (assumptions shown)."""
    from graphrag_lab.extraction import estimate_full_extraction

    estimate = estimate_full_extraction(pilot, total_chunks)
    estimate["assumption"] = (
        "Linear extrapolation from the pilot sample; actual cost varies with "
        "chunk length, retries and quarantined items."
    )
    return estimate


def enqueue_index_once(
    store: JobStore,
    *,
    snapshot_id: str,
    stage: str,
    budget: Any,
    idempotency_key: str,
    items: list[dict[str, Any]] | None = None,
) -> tuple[Any, bool]:
    """Enqueue unless the same active submission already exists.

    Returns (JobRef, submitted). ``submitted`` is False when Streamlit reruns
    (or double clicks) hit the same idempotency key: the existing active job
    is returned and no duplicate job is created.
    """
    from graphrag_lab.jobs import enqueue_index

    ref = enqueue_index(
        store,
        snapshot_id=snapshot_id,
        stage=stage,
        budget=budget,
        idempotency_key=idempotency_key,
        items=items,
    )
    return ref, bool(ref.created)


def config_change_creates_snapshot(current_fingerprint: str | None, new_fingerprint: str) -> bool:
    """Fingerprinted changes build a new snapshot; completed runs never mutate."""
    if not new_fingerprint.strip():
        raise ValueError("new_fingerprint must be nonempty")
    if current_fingerprint is None:
        return True
    return current_fingerprint.strip() != new_fingerprint.strip()


def recent_runs(run_store: Any, *, limit: int = 5) -> list[dict[str, Any]]:
    """Recent runs for Overview: question excerpt lives in predictions, not here."""
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
        raise ValueError("limit must be an integer in 1..50")
    runs = run_store.list_runs(limit=limit)
    return [
        {
            "run_id": run.id,
            "strategy": run.strategy,
            "split": run.split,
            "state": run.state,
            "updated_at": run.updated_at,
        }
        for run in runs
    ]


__all__ = [
    "NextAction",
    "SectionStatus",
    "SetupState",
    "SetupSummary",
    "SpendSummary",
    "config_change_creates_snapshot",
    "enqueue_index_once",
    "idempotency_key_for",
    "latest_dataset_id",
    "neo4j_message",
    "pilot_vs_full_estimate",
    "progress_panel",
    "recent_runs",
    "spend_summary",
    "summarize_setup",
    "worker_status",
]
