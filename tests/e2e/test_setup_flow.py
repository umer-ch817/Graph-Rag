"""Setup and index dashboard flow (T12b, R11).

Covers the dashboard contract without the Streamlit runtime: readiness and
the single Overview next action, persisted index progress/spend, duplicate
submission prevention across reruns, and every setup state from the UI-UX
state coverage (no-corpus, queued, running, failed, cancelled,
paused-budget, interrupted, ready). Pure ``setup_flow`` logic plus the
durable ``jobs`` store; ``app/`` owns presentation only.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from app.components.status import job_error_text, progress_line, spend_line
from app.components.theme import STATUS_TEXT, TOKENS, base_css, pill_text
from graphrag_lab.config import BudgetConfig
from graphrag_lab.jobs import JobStore, enqueue_index, get_job_status
from graphrag_lab.run_store import RunStore
from graphrag_lab.setup_flow import (
    config_change_creates_snapshot,
    enqueue_index_once,
    idempotency_key_for,
    latest_dataset_id,
    neo4j_message,
    pilot_vs_full_estimate,
    progress_panel,
    recent_runs,
    spend_summary,
    summarize_setup,
    worker_status,
)
from graphrag_lab.worker import Worker, demo_handler_factory


def _seed_store(db: Path) -> JobStore:
    store = JobStore(db)
    store.create_dataset(
        dataset_id="ds_e2e",
        source="fixture",
        checksum="sha_e2e",
        seed=42,
        sample_count=4,
        manifest_path="artifacts/dataset/manifest.json",
    )
    store.create_snapshot(snapshot_id="snap_e2e", dataset_id="ds_e2e", fingerprint="fp_e2e")
    return store


def _items(n: int) -> list[dict[str, Any]]:
    return [{"id": f"chunk_{i:02d}"} for i in range(n)]


def test_no_corpus_points_at_prepare(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    assert latest_dataset_id(store) is None
    summary = summarize_setup(store)
    assert summary.dataset.state == "missing"
    assert summary.vector.state == "blocked"
    assert "Prepare a dataset first" in summary.vector.reason
    assert summary.next_action.label == "Prepare dataset"
    assert summary.next_action.target == "dataset"
    assert summary.ready.state == "blocked"


def test_dataset_ready_then_vector_build_is_next(tmp_path: Path):
    store = _seed_store(tmp_path / "jobs.sqlite3")
    assert latest_dataset_id(store) == "ds_e2e"
    summary = summarize_setup(store)
    assert summary.dataset.state == "ready"
    assert "ds_e2e" in summary.dataset.headline
    assert summary.vector.state == "missing"
    assert summary.next_action.label == "Build vector index"
    assert summary.graph.state == "blocked"


def test_duplicate_submission_prevented_across_reruns(tmp_path: Path):
    store = _seed_store(tmp_path / "jobs.sqlite3")
    budget = BudgetConfig(max_calls=20, max_total_tokens=20000, max_cost_usd=None)
    key = idempotency_key_for("vector", "index_vector", "snap_e2e", "fp_e2e")
    first, submitted = enqueue_index_once(
        store,
        snapshot_id="snap_e2e",
        stage="vector",
        budget=budget,
        idempotency_key=key,
        items=_items(4),
    )
    assert submitted is True
    # A Streamlit rerun (or double click) reuses the same key: no second job.
    second, resubmitted = enqueue_index_once(
        store,
        snapshot_id="snap_e2e",
        stage="vector",
        budget=budget,
        idempotency_key=key,
        items=_items(4),
    )
    assert resubmitted is False
    assert second.job_id == first.job_id
    assert len(store.list_jobs(kind="index_vector")) == 1


def test_progress_and_spend_persist_across_restart(tmp_path: Path):
    db = tmp_path / "jobs.sqlite3"
    store = _seed_store(db)
    budget = BudgetConfig(max_calls=20, max_total_tokens=20000, max_cost_usd=None)
    ref, _ = enqueue_index_once(
        store,
        snapshot_id="snap_e2e",
        stage="vector",
        budget=budget,
        idempotency_key="k-progress",
        items=_items(4),
    )
    calls: list[str] = []
    worker = Worker(db, worker_id="w-e2e")
    worker.register_handler("index_vector", demo_handler_factory(calls))
    worker.run_job(ref.job_id, max_items=2)
    # Simulate browser close + reopen: fresh instances over the same file.
    del store, worker
    reopened = JobStore(db)
    job = reopened.get_job(ref.job_id)
    assert (job.progress_done, job.progress_total) == (2, 4)
    panel = progress_panel(job, attempts=len(reopened.get_attempts(job.id)))
    assert panel["processed"] == 2 and panel["total"] == 4 and panel["attempts"] == 1
    assert progress_line(2, 4) == "2/4 chunks processed"
    spend = spend_summary(job)
    assert spend.budget_calls == 20 and spend.budget_tokens == 20000
    assert spend.estimated_cost_usd is None
    assert "unknown" in spend.cost_label and spend.cost_label != "$0.000000 (estimated)"
    line = spend_line(
        budget_calls=spend.budget_calls,
        budget_tokens=spend.budget_tokens,
        budget_cost_usd=spend.budget_cost_usd,
        used_items=spend.used_items,
        cost_label=spend.cost_label,
    )
    assert "calls" in line and "tokens" in line and "unknown" in line


def test_state_walk_matches_status_coverage(tmp_path: Path):
    db = tmp_path / "jobs.sqlite3"
    store = _seed_store(db)
    budget = BudgetConfig(max_calls=20, max_total_tokens=20000, max_cost_usd=None)

    def fresh(key: str, n: int = 2):
        # The dashboard surfaces the newest job, so each walk step uses a
        # unique idempotency key and asserts against the latest section state.
        ref, _ = enqueue_index_once(
            store,
            snapshot_id="snap_e2e",
            stage="vector",
            budget=budget,
            idempotency_key=key,
            items=_items(n),
        )
        return ref

    def vector_state() -> str:
        return summarize_setup(store).vector.state

    ref = fresh("walk-queued")
    assert vector_state() == "queued"
    assert summarize_setup(store).next_action.label == "View indexing progress"

    store.claim_job(ref.job_id, "w-walk")
    assert vector_state() == "running"

    worker = Worker(db, worker_id="w-walk")
    worker.register_handler("index_vector", demo_handler_factory(fail_ids=["chunk_00"]))
    worker.run_job(ref.job_id)
    assert vector_state() == "failed"
    failed = store.get_job(ref.job_id)
    assert "Retry failed chunks" in summarize_setup(store).vector.action_label
    assert "preserved" in summarize_setup(store).vector.reason
    assert "Retry failed items" in job_error_text(failed)

    store.resume_job(ref.job_id)
    worker2 = Worker(db, worker_id="w-walk-2")
    worker2.register_handler("index_vector", demo_handler_factory())
    worker2.run_job(ref.job_id)
    assert vector_state() == "ready"
    assert summarize_setup(store).next_action.label == "Run graph pilot"

    ref2 = fresh("walk-cancel")
    store.cancel_job(ref2.job_id)
    assert vector_state() == "cancelled"
    store.resume_job(ref2.job_id)
    worker2.run_job(ref2.job_id)
    assert vector_state() == "ready"

    ref3 = fresh("walk-budget")
    limited = Worker(db, worker_id="w-budget")
    limited.register_handler("index_vector", demo_handler_factory([], budget_after=0))
    limited.run_job(ref3.job_id)
    paused = store.get_job(ref3.job_id)
    assert paused.state == "paused_budget"
    assert vector_state() == "paused_budget"
    assert paused.error_code == "BUDGET_EXCEEDED"

    ref4 = fresh("walk-interrupt")
    store.claim_job(ref4.job_id, "w-gone")
    time.sleep(0.01)
    store.recover_stale_jobs(lease_timeout_s=0.001)
    assert vector_state() == "interrupted"


def test_vector_ready_graph_absent_enables_vector_modes(tmp_path: Path):
    store = _seed_store(tmp_path / "jobs.sqlite3")
    budget = BudgetConfig(max_calls=20, max_total_tokens=20000, max_cost_usd=None)
    ref, _ = enqueue_index_once(
        store,
        snapshot_id="snap_e2e",
        stage="vector",
        budget=budget,
        idempotency_key="k-ready",
        items=_items(2),
    )
    worker = Worker(str(store.path), worker_id="w-ready")
    worker.register_handler("index_vector", demo_handler_factory())
    worker.run_job(ref.job_id)
    summary = summarize_setup(store)
    assert summary.vector.state == "ready"
    assert summary.ready.headline == "Vector ready; graph unavailable."
    assert "vector only" in summary.ready.action_label


def test_graph_ready_points_at_compare(tmp_path: Path):
    db = tmp_path / "jobs.sqlite3"
    store = _seed_store(db)
    budget = BudgetConfig(max_calls=20, max_total_tokens=20000, max_cost_usd=None)
    for stage, key in (("vector", "k-gv"), ("graph", "k-gg")):
        ref, _ = enqueue_index_once(
            store,
            snapshot_id="snap_e2e",
            stage=stage,
            budget=budget,
            idempotency_key=key,
            items=_items(2),
        )
        worker = Worker(db, worker_id=f"w-{key}")
        worker.register_handler(f"index_{stage}", demo_handler_factory())
        worker.run_job(ref.job_id)
    summary = summarize_setup(store)
    assert summary.ready.state == "ready"
    assert summary.next_action.label == "Ask a question"
    assert summary.next_action.target == "compare"


def test_worker_and_neo4j_errors_are_understandable():
    never = worker_status(None)
    assert never["seen"] is False
    assert "python -m graphrag_lab.cli worker" in never["message"]
    stale = worker_status("2026-01-01T00:00:00+00:00", now_iso="2026-09-29T00:01:00+00:00")
    assert stale["state"] == "stale" and "resume" in stale["message"].lower()
    fresh = worker_status("2026-09-29T00:01:00+00:00", now_iso="2026-09-29T00:01:05+00:00")
    assert fresh["state"] == "ready"
    down = neo4j_message("bolt://127.0.0.1:7687", ok=False, detail="refused")
    assert "Queued work is kept" in down and "docker compose up" in down
    assert "lost" in down
    up = neo4j_message("bolt://127.0.0.1:7687", ok=True)
    assert "reachable" in up


def test_pilot_estimate_labels_assumptions():
    pilot = {
        "pilot_size": 20,
        "n_extracted": 20,
        "calls": 20,
        "input_tokens": 7000,
        "output_tokens": 800,
    }
    estimate = pilot_vs_full_estimate(pilot, 150)
    assert estimate["total_chunks"] == 150
    assert estimate["estimated_calls"] == 150
    assert "assumption" in estimate and "varies" in estimate["assumption"]


def test_snapshot_change_flag_and_idempotency_keys():
    assert config_change_creates_snapshot(None, "fp-new") is True
    assert config_change_creates_snapshot("fp-a", "fp-a") is False
    assert config_change_creates_snapshot("fp-a", "fp-b") is True
    assert idempotency_key_for("a", "b") == idempotency_key_for("a", "b")
    assert idempotency_key_for("a", "b") != idempotency_key_for("a", "c")
    with pytest.raises(ValueError):
        config_change_creates_snapshot("fp-a", "  ")


def test_overview_recent_runs_and_single_action(tmp_path: Path):
    db = tmp_path / "jobs.sqlite3"
    JobStore(db)
    runs = RunStore(db)
    first = runs.create_run(strategy="vector", question_set_hash="qh", split="tuning")
    second = runs.create_run(strategy="hybrid_rerank", question_set_hash="qh", split="tuning")
    rows = recent_runs(runs, limit=5)
    assert [row["run_id"] for row in rows] == [first.id, second.id]
    assert all(row["state"] == "queued" for row in rows)
    with pytest.raises(ValueError):
        recent_runs(runs, limit=0)


def test_shared_tokens_match_spec_and_pills_carry_text():
    assert TOKENS["action"] == "#176348"
    assert TOKENS["background"] == "#F6F7F5"
    assert TOKENS["danger"] == "#A32932"
    assert TOKENS["focus"] == "#245FC7"
    assert STATUS_TEXT["paused_budget"] == "Paused on budget"
    assert "Paused on budget" in pill_text("paused_budget")
    assert "Ready" in pill_text("ready")
    css = base_css()
    assert TOKENS["action"] in css and "outline" in css


def test_enqueue_index_validates_stage_and_snapshot(tmp_path: Path):
    store = _seed_store(tmp_path / "jobs.sqlite3")
    budget = BudgetConfig(max_calls=5, max_total_tokens=5000, max_cost_usd=None)
    with pytest.raises(Exception, match="stage must be"):
        enqueue_index(store, snapshot_id="snap_e2e", stage="bogus", budget=budget)
    with pytest.raises(Exception, match="does not exist"):
        enqueue_index(store, snapshot_id="snap_missing", stage="vector", budget=budget)
    status = get_job_status(
        store,
        enqueue_index(
            store,
            snapshot_id="snap_e2e",
            stage="graph",
            budget=budget,
            idempotency_key="k-graph",
            items=_items(1),
        ).job_id,
    )
    assert status["progress_total"] == 1 and status["retryable"] is False
