"""Single local worker for T12a (dashboard, R11).

Claims queued SQLite jobs, executes deterministic items with per-item
checkpoints, and honors cancellation and budget pauses between items. The
worker continues independently of Streamlit reruns; the dashboard enqueues
work and polls persisted status.

Checkpoint rule: validated (completed) items are never re-executed. Resume
reuses ``checkpoint.completed`` and retries only failed/unprocessed items in
a new job attempt. Cancellation cannot undo an in-flight call; the flag is
observed at the next item boundary.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from graphrag_lab.jobs import (
    DEFAULT_DB_PATH,
    DEFAULT_LEASE_TIMEOUT_S,
    BudgetExceededError,
    CancelledError,
    InvalidInputError,
    JobRecord,
    JobStore,
)

HandlerFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]

_WORKER_VERSION = "worker-v1"


def _validate_item(item: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise InvalidInputError(f"job item {index} must be an object")
    item_id = item.get("id", item.get("item_id"))
    if not isinstance(item_id, str) or not item_id.strip():
        raise InvalidInputError(f"job item {index} needs a nonempty string 'id'")
    if len(item_id.strip()) > 128:
        raise InvalidInputError(f"job item {index} id exceeds 128 chars")
    return dict(item)


def _load_items(job: JobRecord) -> list[dict[str, Any]]:
    raw = job.config_json.get("items", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise InvalidInputError(f"job {job.id!r} config['items'] must be a list")
    items = [_validate_item(entry, index=i) for i, entry in enumerate(raw)]
    seen: set[str] = set()
    for entry in items:
        item_id = str(entry.get("id", entry.get("item_id"))).strip()
        if item_id in seen:
            raise InvalidInputError(f"job {job.id!r} repeats item id {item_id!r}")
        seen.add(item_id)
    return items


def _checkpoint_view(job: JobRecord) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = dict(job.checkpoint_json or {})
    completed = checkpoint.get("completed", {})
    failed = checkpoint.get("failed", {})
    if not isinstance(completed, dict) or not isinstance(failed, dict):
        raise InvalidInputError(f"job {job.id!r} checkpoint must hold completed/failed objects")
    return dict(completed), dict(failed)


def _sanitize(message: str, limit: int = 500) -> str:
    return " ".join(str(message).split())[:limit] or "job item failed"


class Worker:
    """Claim and execute jobs from one SQLite file (single-writer design)."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        *,
        worker_id: str | None = None,
        lease_timeout_s: float = DEFAULT_LEASE_TIMEOUT_S,
    ) -> None:
        self._store = JobStore(db_path)
        cleaned = (
            worker_id.strip()
            if isinstance(worker_id, str) and worker_id.strip()
            else f"worker_{uuid.uuid4().hex[:8]}"
        )
        if not cleaned:
            raise InvalidInputError("worker_id must be nonempty")
        self._worker_id = cleaned
        if (
            not isinstance(lease_timeout_s, (int, float))
            or isinstance(lease_timeout_s, bool)
            or lease_timeout_s <= 0
        ):
            raise InvalidInputError("lease_timeout_s must be positive")
        self._lease_timeout_s = float(lease_timeout_s)
        self._handlers: dict[str, HandlerFn] = {}

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def store(self) -> JobStore:
        return self._store

    def register_handler(self, kind: str, fn: HandlerFn) -> None:
        if not kind.strip():
            raise InvalidInputError("handler kind must be nonempty")
        if not callable(fn):
            raise InvalidInputError("handler must be callable")
        self._handlers[kind.strip()] = fn

    def recover_stale(self) -> list[str]:
        return self._store.recover_stale_jobs(self._lease_timeout_s)

    def run_once(
        self,
        *,
        kinds: list[str] | None = None,
        max_items: int | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> str | None:
        self.recover_stale()
        claimed = self._store.claim_next_job(self._worker_id, kinds=kinds)
        if claimed is None:
            return None
        self.run_job(claimed.id, max_items=max_items, sleep_fn=sleep_fn)
        return claimed.id

    def _record_item_progress(
        self,
        job_id: str,
        completed: dict[str, Any],
        failed: dict[str, Any],
        total_items: int,
    ) -> JobRecord:
        """Persist item progress without moving it backwards.

        Handlers may report finer-grained progress than one unit per item
        (e.g. the vector indexer counts chunks inside a single control item).
        Item accounting never rewinds handler-reported counters and never
        shrinks a handler-reported total.
        """
        current = self._store.get_job(job_id)
        done = max(len(completed) + len(failed), int(current.progress_done))
        total = max(int(current.progress_total), int(total_items))
        done = min(done, total)
        return self._store.update_progress(
            job_id,
            progress_done=done,
            progress_total=total,
            checkpoint={"completed": completed, "failed": failed},
        )

    def run_job(
        self,
        job_id: str,
        *,
        max_items: int | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> JobRecord:
        if max_items is not None and (
            not isinstance(max_items, int) or isinstance(max_items, bool) or max_items < 0
        ):
            raise InvalidInputError("max_items must be a non-negative integer when supplied")
        _ = sleep_fn  # reserved for backoff-capable handlers; failures sleep locally
        job = self._store.get_job(job_id)
        if job.state == "queued":
            claimed = self._store.claim_job(job_id, self._worker_id)
            if claimed is None:
                # Another worker claimed it first; return the current record.
                return self._store.get_job(job_id)
            job = claimed
        if job.state not in ("running", "cancelling"):
            raise InvalidInputError(
                f"job {job_id!r} in state {job.state!r} cannot run; resume it to 'queued' first"
            )
        try:
            items = _load_items(job)
        except InvalidInputError as exc:
            return self._store.fail_job(job_id, error_code="INVALID_INPUT", error_message=str(exc))
        if not items:
            # Control jobs with no items (e.g. CLI probes) complete without a
            # handler. Real index/evaluation jobs must carry executable items:
            # completing them silently would fake progress.
            if job.kind in ("index_vector", "index_graph", "evaluation"):
                return self._store.fail_job(
                    job_id,
                    error_code="INVALID_INPUT",
                    error_message=(
                        f"No executable items for kind {job.kind!r}; "
                        "the index/evaluation executor is not wired yet"
                    ),
                )
            self._store.update_progress(job_id, progress_done=0, progress_total=0)
            return self._store.complete_job(job_id)
        handler = self._handlers.get(job.kind)
        if handler is None:
            return self._store.fail_job(
                job_id,
                error_code="INVALID_INPUT",
                error_message=f"No handler for kind {job.kind!r}",
            )
        completed, failed = _checkpoint_view(job)
        # Deterministic order: config order filtered to uncompleted ids.
        # Previously failed ids are retried (they are not in completed);
        # within one attempt each pending item is visited once.
        pending = [entry for entry in items if str(entry.get("id")).strip() not in completed]
        retried_in_attempt: set[str] = set()
        processed_this_call = 0
        total = len(items)
        self._record_item_progress(job_id, completed, failed, total)
        for entry in pending:
            if max_items is not None and processed_this_call >= max_items:
                # Simulated crash/stop: leave the job running with its partial
                # checkpoint so recover/resume can be exercised honestly.
                return self._store.get_job(job_id)
            fresh = self._store.get_job(job_id)
            if fresh.cancel_requested or fresh.state == "cancelling":
                self._record_item_progress(job_id, completed, failed, total)
                return self._store.cancel_finish_job(
                    job_id, error_message="Cancelled at item boundary"
                )
            item_id = str(entry.get("id")).strip()
            if item_id in retried_in_attempt:
                continue
            ctx: dict[str, Any] = {
                "job_id": job_id,
                "worker_id": self._worker_id,
                "worker_version": _WORKER_VERSION,
                "kind": job.kind,
                "attempt": len(self._store.get_attempts(job_id)),
            }
            try:
                result = handler(dict(entry), dict(ctx))
                if not isinstance(result, dict):
                    raise InvalidInputError("handler must return a dict")
                try:
                    json.dumps(result, sort_keys=True)
                except (TypeError, ValueError) as exc:
                    raise InvalidInputError(f"handler result not JSON-serializable: {exc}") from exc
                completed[item_id] = result
                failed.pop(item_id, None)
            except (BudgetExceededError, CancelledError) as exc:
                code = getattr(exc, "code", "BUDGET_EXCEEDED") or "BUDGET_EXCEEDED"
                message = _sanitize(str(exc) or code)
                self._record_item_progress(job_id, completed, failed, total)
                if code == "CANCELLED":
                    return self._store.cancel_finish_job(job_id, error_message=message)
                return self._store.pause_budget_job(job_id, error_message=message)
            except Exception as exc:  # noqa: BLE001 - item failures quarantine, job continues
                code = getattr(exc, "code", "PROVIDER_ERROR") or "PROVIDER_ERROR"
                failed[item_id] = {"error_code": str(code), "error_message": _sanitize(str(exc))}
                retried_in_attempt.add(item_id)
            processed_this_call += 1
            self._record_item_progress(job_id, completed, failed, total)
            try:
                self._store.heartbeat(job_id, self._worker_id)
            except InvalidInputError:
                # Ownership edge (e.g. recovered mid-run): stop without losing checkpoint.
                return self._store.get_job(job_id)
        final = self._store.get_job(job_id)
        if final.cancel_requested or final.state == "cancelling":
            return self._store.cancel_finish_job(
                job_id, error_message="Cancelled at completion boundary"
            )
        if failed:
            first = sorted(failed)[0]
            return self._store.fail_job(
                job_id,
                error_code=str(failed[first].get("error_code", "PROVIDER_ERROR")),
                error_message=f"{len(failed)} item(s) failed; first {first}: "
                f"{failed[first].get('error_message', 'failed')}",
            )
        return self._store.complete_job(job_id)

    def run_forever(
        self,
        *,
        kinds: list[str] | None = None,
        poll_interval_s: float = 1.0,
        max_jobs: int | None = None,
        max_items_per_job: int | None = None,
    ) -> int:
        if (
            not isinstance(poll_interval_s, (int, float))
            or isinstance(poll_interval_s, bool)
            or poll_interval_s <= 0
        ):
            raise InvalidInputError("poll_interval_s must be positive")
        if max_jobs is not None and (
            not isinstance(max_jobs, int) or isinstance(max_jobs, bool) or max_jobs < 0
        ):
            raise InvalidInputError("max_jobs must be a non-negative integer when supplied")
        done = 0
        while True:
            if max_jobs is not None and done >= max_jobs:
                return done
            job_id = self.run_once(kinds=kinds, max_items=max_items_per_job)
            if job_id is None:
                if max_jobs is not None:
                    return done
                time.sleep(float(poll_interval_s))
                continue
            done += 1


def demo_handler_factory(
    calls: list[str] | None = None,
    *,
    fail_ids: Sequence[str] | None = None,
    budget_after: int | None = None,
    budget_error: str = "Call budget exceeded",
) -> HandlerFn:
    """Deterministic test handler: records item ids, optionally fails/pauses.

    - ``fail_ids`` items raise PROVIDER_ERROR (quarantined, job continues).
    - ``budget_after`` raises BUDGET_EXCEEDED after that many successful calls
      (worker must pause preserving completed work).
    - Every successful call appends the item id to ``calls`` so tests can
      assert no duplicate completed calls across interrupt/resume.
    """

    seen: list[str] = calls if calls is not None else []
    failing = set(fail_ids or [])

    def _handler(item: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        _ = ctx
        item_id = str(item.get("id")).strip()
        if item_id in failing:
            from graphrag_lab.jobs import ProviderError

            raise ProviderError(f"demo failure for {item_id}")
        if budget_after is not None and len(seen) >= budget_after:
            raise BudgetExceededError(budget_error)
        seen.append(item_id)
        return {"item_id": item_id, "ok": True}

    return _handler


__all__ = [
    "HandlerFn",
    "Worker",
    "demo_handler_factory",
]
