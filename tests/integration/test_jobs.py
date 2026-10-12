"""Durable local jobs on SQLite (T12a, R11).

Offline: no Neo4j, embedding or LLM downloads. Each test uses an isolated
temporary database file. The interrupt/resume test proves validated items are
never re-executed after a mid-fixture stop.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from graphrag_lab.jobs import (
    BudgetExceededError,
    InvalidInputError,
    JobStore,
    enqueue_evaluation,
    enqueue_index,
    get_job_status,
)
from graphrag_lab.run_store import RunStore
from graphrag_lab.worker import Worker, demo_handler_factory


def _items(n: int, prefix: str = "item") -> list[dict[str, Any]]:
    return [{"id": f"{prefix}_{i:02d}", "n": i} for i in range(n)]


def test_state_survives_restart(tmp_path: Path):
    db = tmp_path / "jobs.sqlite3"
    first = JobStore(db)
    ref = first.enqueue_job("demo", {"items": _items(3)}, idempotency_key="restart-1")
    assert ref.created is True
    # Simulate process restart: drop the instance and reopen the same file.
    del first
    second = JobStore(db)
    job = second.get_job(ref.job_id)
    assert job.kind == "demo"
    assert job.state == "queued"
    assert job.progress_total == 3
    assert job.progress_done == 0


def test_one_worker_claims_each_job_once(tmp_path: Path):
    db = tmp_path / "jobs.sqlite3"
    store = JobStore(db)
    ref = store.enqueue_job("demo", {"items": _items(2)}, idempotency_key="claim-1")
    first = store.claim_job(ref.job_id, "worker-a")
    assert first is not None and first.state == "running"
    # A second worker racing for the same job gets nothing.
    second = store.claim_job(ref.job_id, "worker-b")
    assert second is None
    # claim_next_job also refuses the already-running job.
    store2 = JobStore(db)
    assert store2.claim_job(ref.job_id, "worker-c") is None


def test_duplicate_active_submission_returns_existing(tmp_path: Path):
    db = tmp_path / "jobs.sqlite3"
    store = JobStore(db)
    config = {"items": _items(2), "note": "same"}
    first = store.enqueue_job("demo", config, idempotency_key="dup-1")
    second = store.enqueue_job("demo", config, idempotency_key="dup-1")
    assert first.created is True
    assert second.created is False
    assert first.job_id == second.job_id
    # After completion a new job with the same key may be enqueued.
    worker = Worker(db, worker_id="w-dup")
    worker.register_handler("demo", demo_handler_factory())
    worker.run_job(first.job_id)
    assert store.get_job(first.job_id).state == "completed"
    third = store.enqueue_job("demo", config, idempotency_key="dup-1")
    assert third.created is True
    assert third.job_id != first.job_id


def test_interrupt_mid_fixture_resumes_without_duplicate_calls(tmp_path: Path):
    db = tmp_path / "jobs.sqlite3"
    store = JobStore(db)
    calls: list[str] = []
    ref = store.enqueue_job("demo", {"items": _items(6)}, idempotency_key="resume-1")
    worker = Worker(db, worker_id="w-resume", lease_timeout_s=60.0)
    worker.register_handler("demo", demo_handler_factory(calls))
    # Process two items then stop, simulating a crash (job stays running).
    worker.run_job(ref.job_id, max_items=2)
    partial = store.get_job(ref.job_id)
    assert partial.state == "running"
    assert partial.progress_done == 2
    assert calls == ["item_00", "item_01"]
    # A fresh worker instance recovers the stale lease and resumes.
    worker2 = Worker(db, worker_id="w-resume-2", lease_timeout_s=0.01)
    time.sleep(0.02)
    recovered = worker2.recover_stale()
    assert ref.job_id in recovered
    assert store.get_job(ref.job_id).state == "interrupted"
    resumed = store.resume_job(ref.job_id)
    assert resumed.state == "queued"
    worker2.register_handler("demo", demo_handler_factory(calls))
    worker2.run_job(ref.job_id)
    final = store.get_job(ref.job_id)
    assert final.state == "completed"
    assert final.progress_done == 6
    # No validated item was re-executed: exactly one call per item id.
    assert sorted(calls) == [f"item_{i:02d}" for i in range(6)]
    assert len(set(calls)) == 6


def test_cancellation_preserves_validated_work(tmp_path: Path):
    db = tmp_path / "jobs.sqlite3"
    store = JobStore(db)
    calls: list[str] = []

    def slow_handler(item: dict, ctx: dict) -> dict:
        calls.append(str(item["id"]))
        if len(calls) == 2:
            # Request cancellation from inside the run (as the dashboard would).
            store.cancel_job(ctx["job_id"])
        return {"ok": True}

    ref = store.enqueue_job("demo", {"items": _items(5)}, idempotency_key="cancel-1")
    worker = Worker(db, worker_id="w-cancel")
    worker.register_handler("demo", slow_handler)
    worker.run_job(ref.job_id)
    cancelled = store.get_job(ref.job_id)
    assert cancelled.state == "cancelled"
    assert cancelled.progress_done == 2
    assert calls == ["item_00", "item_01"]
    # Resume continues after the checkpoint without repeating validated items.
    store.resume_job(ref.job_id)
    worker2 = Worker(db, worker_id="w-cancel-2")
    worker2.register_handler("demo", demo_handler_factory(calls))
    worker2.run_job(ref.job_id)
    final = store.get_job(ref.job_id)
    assert final.state == "completed"
    assert sorted(calls) == [f"item_{i:02d}" for i in range(5)]


def test_cancel_before_start_marks_cancelled(tmp_path: Path):
    db = tmp_path / "jobs.sqlite3"
    store = JobStore(db)
    ref = store.enqueue_job("demo", {"items": _items(2)}, idempotency_key="cancel-early")
    cancelled = store.cancel_job(ref.job_id)
    assert cancelled.state == "cancelled"
    assert cancelled.cancel_requested is True
    status = get_job_status(store, ref.job_id)
    assert status["retryable"] is True
    resumed = store.resume_job(ref.job_id)
    assert resumed.state == "queued"
    worker = Worker(db, worker_id="w-early")
    worker.register_handler("demo", demo_handler_factory())
    worker.run_job(ref.job_id)
    assert store.get_job(ref.job_id).state == "completed"


def test_budget_pause_preserves_validated_work(tmp_path: Path):
    db = tmp_path / "jobs.sqlite3"
    store = JobStore(db)
    calls: list[str] = []
    ref = store.enqueue_job(
        "demo",
        {"items": _items(5), "budget": {"max_calls": 2, "max_total_tokens": 20000}},
        idempotency_key="budget-1",
    )
    worker = Worker(db, worker_id="w-budget")
    worker.register_handler("demo", demo_handler_factory(calls, budget_after=2))
    worker.run_job(ref.job_id)
    paused = store.get_job(ref.job_id)
    assert paused.state == "paused_budget"
    assert paused.error_code == "BUDGET_EXCEEDED"
    assert paused.progress_done == 2
    assert calls == ["item_00", "item_01"]
    # Revised budget: resume and finish the remaining items.
    store.resume_job(ref.job_id)
    worker2 = Worker(db, worker_id="w-budget-2")
    worker2.register_handler("demo", demo_handler_factory(calls))
    worker2.run_job(ref.job_id)
    final = store.get_job(ref.job_id)
    assert final.state == "completed"
    assert sorted(calls) == [f"item_{i:02d}" for i in range(5)]


def test_failed_items_quarantine_and_retry_only_failures(tmp_path: Path):
    db = tmp_path / "jobs.sqlite3"
    store = JobStore(db)
    calls: list[str] = []
    ref = store.enqueue_job("demo", {"items": _items(4)}, idempotency_key="fail-1")
    worker = Worker(db, worker_id="w-fail")
    worker.register_handler("demo", demo_handler_factory(calls, fail_ids=["item_01"]))
    worker.run_job(ref.job_id)
    failed = store.get_job(ref.job_id)
    assert failed.state == "failed"
    assert failed.progress_done == 4
    assert failed.checkpoint_json["failed"].keys() == {"item_01"}
    assert sorted(calls) == ["item_00", "item_02", "item_03"]
    # Resume retries the quarantined item; validated items are not repeated.
    store.resume_job(ref.job_id)
    worker2 = Worker(db, worker_id="w-fail-2")
    worker2.register_handler("demo", demo_handler_factory(calls))
    worker2.run_job(ref.job_id)
    final = store.get_job(ref.job_id)
    assert final.state == "completed"
    assert sorted(calls) == [f"item_{i:02d}" for i in range(4)]


def test_stale_lease_recovery_marks_interrupted(tmp_path: Path):
    db = tmp_path / "jobs.sqlite3"
    store = JobStore(db)
    ref = store.enqueue_job("demo", {"items": _items(2)}, idempotency_key="stale-1")
    claimed = store.claim_job(ref.job_id, "worker-gone")
    assert claimed is not None
    # Fresh lease: nothing recovered.
    assert store.recover_stale_jobs(lease_timeout_s=60.0) == []
    assert store.get_job(ref.job_id).state == "running"
    # Expired lease: running -> interrupted with checkpoint intact.
    recovered = store.recover_stale_jobs(lease_timeout_s=0.0 + 0.001)
    # heartbeat was just written, so use a tiny timeout after a short sleep.
    if ref.job_id not in recovered:
        time.sleep(0.01)
        recovered = store.recover_stale_jobs(lease_timeout_s=0.001)
    assert ref.job_id in recovered
    assert store.get_job(ref.job_id).state == "interrupted"


def test_enqueue_index_and_evaluation_helpers(tmp_path: Path):
    db = tmp_path / "jobs.sqlite3"
    store = JobStore(db)
    store.create_dataset(
        dataset_id="ds_t12a",
        source="fixture",
        checksum="abc123",
        seed=42,
        sample_count=4,
        manifest_path="artifacts/dataset/manifest.json",
    )
    store.create_snapshot(snapshot_id="snap_t12a", dataset_id="ds_t12a", fingerprint="fp_t12a")
    from graphrag_lab.config import BudgetConfig

    budget = BudgetConfig(max_calls=20, max_total_tokens=20000, max_cost_usd=None)
    index_ref = enqueue_index(
        store, snapshot_id="snap_t12a", stage="vector", budget=budget, idempotency_key="idx-1"
    )
    assert index_ref.kind == "index_vector"
    dup = enqueue_index(
        store, snapshot_id="snap_t12a", stage="vector", budget=budget, idempotency_key="idx-1"
    )
    assert dup.created is False and dup.job_id == index_ref.job_id
    with pytest.raises(InvalidInputError):
        enqueue_index(store, snapshot_id="snap_t12a", stage="bogus", budget=budget)
    eval_ref = enqueue_evaluation(
        store,
        snapshot_id="snap_t12a",
        strategy="vector",
        split="tuning",
        config={"question_ids": ["q1", "q2"]},
        items=[{"id": "q1"}, {"id": "q2"}],
        idempotency_key="eval-1",
    )
    assert eval_ref.kind == "evaluation"
    status = get_job_status(store, eval_ref.job_id)
    assert status["progress_total"] == 2
    assert status["retryable"] is False


def test_run_store_predictions_usage_and_comparison(tmp_path: Path):
    db = tmp_path / "jobs.sqlite3"
    JobStore(db)  # ensure schema
    runs = RunStore(db)
    run = runs.create_run(
        strategy="vector", question_set_hash="qh_t12a", config={"max_chunks": 8}, split="tuning"
    )
    assert run.state == "queued"
    runs.upsert_prediction(run.id, "q1", answer="Lisbon", status="supported", citations=[])
    runs.upsert_prediction(run.id, "q2", answer="", status="failed", failure_code="PROVIDER_ERROR")
    assert len(runs.list_predictions(run.id)) == 2
    assert runs.get_prediction(run.id, "q1").answer == "Lisbon"
    event = runs.record_usage(
        stage="answer",
        request_id="req_1",
        status="ok",
        run_id=run.id,
        model="fake-llm-test",
        input_tokens=10,
        output_tokens=5,
        estimated_cost_usd=None,
    )
    assert event.estimated_cost_usd is None
    totals = runs.sum_usage_for_run(run.id)
    assert totals["input_tokens"] == 10 and totals["output_tokens"] == 5
    assert totals["estimated_cost_usd"] is None
    # Completion requires the artifact to exist (durable artifacts first).
    artifact = tmp_path / "results.jsonl"
    with pytest.raises(InvalidInputError, match="durable artifacts"):
        runs.update_run_state(run.id, "completed", artifact_path=str(artifact))
    artifact.write_text("{}\n", encoding="utf-8")
    done = runs.update_run_state(run.id, "completed", artifact_path=str(artifact))
    assert done.state == "completed"
    run2 = runs.create_run(strategy="hybrid_rerank", question_set_hash="qh_t12a", split="tuning")
    cmp = runs.create_comparison(
        run_ids=[run.id, run2.id], compatibility_hash="compat_t12a", metrics={"n": 2}
    )
    assert sorted(cmp.run_ids) == sorted([run.id, run2.id])
    assert runs.get_comparison(cmp.id).compatibility_hash == "compat_t12a"


def test_budget_errors_are_distinct_from_provider_errors():
    assert BudgetExceededError("cap").code == "BUDGET_EXCEEDED"
    from graphrag_lab.jobs import ProviderError

    assert ProviderError("down").code == "PROVIDER_ERROR"


def test_handler_reported_progress_is_never_rewound(tmp_path: Path):
    """T12c: a handler may count sub-item units; item accounting preserves them."""
    db = tmp_path / "jobs.sqlite3"
    store = JobStore(db)
    ref = store.enqueue_job("demo", {"items": [{"id": "build"}]}, idempotency_key="sub-item-1")

    def chunk_counter(item: dict, ctx: dict) -> dict:
        job_store = JobStore(db)
        for done in (1, 2, 3):
            job_store.update_progress(ctx["job_id"], progress_done=done, progress_total=3)
        return {"chunks": 3}

    worker = Worker(db, worker_id="w-subitem")
    worker.register_handler("demo", chunk_counter)
    worker.run_job(ref.job_id)
    final = store.get_job(ref.job_id)
    assert final.state == "completed"
    assert (final.progress_done, final.progress_total) == (3, 3)
    # Resume bookkeeping still works after handler-managed totals.
    assert store.get_attempts(ref.job_id)[0].worker_id == "w-subitem"
