"""Local vector index builds (T12c, C4).

Offline: fake embedding backend and stubbed Neo4j helpers, so no model
download and no database are needed. Proves batch progress persists,
cancellation stops between batches, snapshot gating is honest, and API cost
stays unknown (never zero).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from graphrag_lab.index_tasks import make_index_vector_handler, run_vector_index
from graphrag_lab.jobs import CancelledError, InvalidInputError, JobStore


class FakeEmbedBackend:
    model_name = "fake-embed-test"
    dimension = 8
    max_seq_length = 512

    def token_length(self, text: str) -> int:
        return len(text.split()) + 3

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            seed = sum(text.encode("utf-8")) % 97 + 1
            raw = [float((seed + i) % 11 + 1) for i in range(self.dimension)]
            norm = math.sqrt(sum(v * v for v in raw))
            vectors.append([v / norm for v in raw])
        return vectors


def _dataset_dir(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    paragraphs = [
        {"id": "p_one", "title": "Ada Lovelace", "sentences": ["Ada wrote the first program."]},
        {
            "id": "p_two",
            "title": "Lisbon",
            "sentences": ["Lisbon is a city.", "It is in Portugal."],
        },
    ]
    (dataset / "paragraphs.jsonl").write_text(
        "".join(json.dumps(p, sort_keys=True) + "\n" for p in paragraphs), encoding="utf-8"
    )
    (dataset / "manifest.json").write_text(
        json.dumps({"corpus_id": "hp_fake", "sample": {}}, sort_keys=True), encoding="utf-8"
    )
    return dataset


def _seed_job(tmp_path: Path) -> tuple[JobStore, str]:
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.create_snapshot(snapshot_id="snap_fake", dataset_id=None, fingerprint="fp_fake")
    ref = store.enqueue_job(
        "index_vector",
        {"items": [{"id": "build", "dataset_dir": str(tmp_path / "dataset")}]},
        snapshot_id="snap_fake",
        idempotency_key="idx-fake",
    )
    return store, ref.job_id


def _stub_neo4j(monkeypatch: pytest.MonkeyPatch, calls: dict[str, Any]) -> None:
    import graphrag_lab.neo4j_store as neo

    monkeypatch.setattr(
        neo, "verify_connectivity", lambda driver, db: calls.setdefault("verify", True)
    )
    monkeypatch.setattr(neo, "ensure_chunk_constraints", lambda driver, db: None)
    monkeypatch.setattr(neo, "ensure_vector_index", lambda driver, db, dim: calls.update(dim=dim))

    def _ensure_meta(driver: Any, db: str, **kw: Any) -> str:
        calls["meta"] = kw
        return "fp_test"

    monkeypatch.setattr(neo, "ensure_index_meta", _ensure_meta)

    def _ingest(driver: Any, db: str, chunks: Any, vectors: Any, **kw: Any) -> int:
        calls.setdefault("ingested", []).extend(
            [c["id"] if isinstance(c, dict) else c.id for c in chunks]
        )
        calls["vectors"] = [list(v) for v in vectors]
        calls["model"] = kw.get("embedding_model")
        return len(chunks)

    monkeypatch.setattr(neo, "ingest_chunks", _ingest)


def test_vector_build_ingests_with_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset = _dataset_dir(tmp_path)
    store, job_id = _seed_job(tmp_path)
    calls: dict[str, Any] = {}
    _stub_neo4j(monkeypatch, calls)
    batches: list[tuple[int, int]] = []
    summary = run_vector_index(
        dataset_dir=dataset,
        snapshot_id="snap_fake",
        store=store,
        job_id=job_id,
        embedding_model="fake-embed-test",
        batch_size=1,
        driver=object(),
        embed_backend=FakeEmbedBackend(),
        on_batch=lambda done, total: batches.append((done, total)),
    )
    assert summary["n_chunks"] == 2
    assert summary["dimension"] == 8
    assert summary["index_fingerprint"] == "fp_test"
    assert summary["estimated_api_cost_usd"] is None
    assert calls["dim"] == 8 and calls["model"] == "fake-embed-test"
    assert len(calls["ingested"]) == 2
    assert all(cid.startswith("c_") for cid in calls["ingested"])
    assert len(set(calls["ingested"])) == 2
    assert batches == [(1, 2), (2, 2)]
    job = store.get_job(job_id)
    assert (job.progress_done, job.progress_total) == (2, 2)
    snapshot = store.get_snapshot("snap_fake")
    assert snapshot is not None and snapshot["vector_status"] == "ready"
    assert snapshot["status"] == "building"  # graph not ready: never labeled fully ready
    for vector in calls["vectors"]:
        assert len(vector) == 8
        assert abs(math.sqrt(sum(v * v for v in vector)) - 1.0) < 1e-6


def test_vector_build_marks_ready_when_graph_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset = _dataset_dir(tmp_path)
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.create_snapshot(
        snapshot_id="snap_both", dataset_id=None, fingerprint="fp_both", graph_status="ready"
    )
    ref = store.enqueue_job(
        "index_vector",
        {"items": [{"id": "build", "dataset_dir": str(dataset)}]},
        snapshot_id="snap_both",
        idempotency_key="idx-both",
    )
    _stub_neo4j(monkeypatch, {})
    run_vector_index(
        dataset_dir=dataset,
        snapshot_id="snap_both",
        store=store,
        job_id=ref.job_id,
        embedding_model="fake-embed-test",
        driver=object(),
        embed_backend=FakeEmbedBackend(),
    )
    assert store.get_snapshot("snap_both")["status"] == "ready"


def test_vector_build_respects_cancellation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _dataset_dir(tmp_path)
    store, job_id = _seed_job(tmp_path)
    _stub_neo4j(monkeypatch, {})
    store.cancel_job(job_id)
    with pytest.raises(CancelledError):
        run_vector_index(
            dataset_dir=tmp_path / "dataset",
            snapshot_id="snap_fake",
            store=store,
            job_id=job_id,
            embedding_model="fake-embed-test",
            driver=object(),
            embed_backend=FakeEmbedBackend(),
        )


def test_vector_build_needs_manifest_and_snapshot(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.create_snapshot(snapshot_id="snap_empty", dataset_id=None, fingerprint="fp_empty")
    ref = store.enqueue_job(
        "index_vector",
        {"items": [{"id": "build", "dataset_dir": str(tmp_path / "absent")}]},
        snapshot_id="snap_empty",
        idempotency_key="idx-absent",
    )
    with pytest.raises(InvalidInputError, match="Prepare a dataset first"):
        run_vector_index(
            dataset_dir=tmp_path / "absent",
            snapshot_id="snap_empty",
            store=store,
            job_id=ref.job_id,
            embedding_model="fake-embed-test",
            driver=object(),
            embed_backend=FakeEmbedBackend(),
        )
    with pytest.raises(InvalidInputError, match="does not exist"):
        run_vector_index(
            dataset_dir=tmp_path,
            snapshot_id="snap_missing",
            store=store,
            job_id=ref.job_id,
            embedding_model="fake-embed-test",
            driver=object(),
            embed_backend=FakeEmbedBackend(),
        )


def test_index_handler_rejects_other_kinds(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    handler = make_index_vector_handler(store, embedding_model="fake-embed-test")
    store.enqueue_job("demo", {"items": [{"id": "x"}]}, job_id="job_demo_x")
    with pytest.raises(InvalidInputError, match="index handler got kind"):
        handler({"id": "x"}, {"job_id": "job_demo_x"})
