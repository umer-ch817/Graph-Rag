"""Local vector index builds for T12c (dashboard, C4).

Runs the corpus → chunks → embeddings → Neo4j pipeline for one
``index_vector`` job with batched progress, cancellation between batches,
and honest snapshot gating. Graph builds stay LLM-gated (extraction pilot,
budget and credentials) and are not executed here.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from graphrag_lab.chunking import CHUNKER_VERSION, ChunkConfig, chunk_dataset_directory
from graphrag_lab.embeddings import (
    SentenceTransformerBackend,
    build_embedding_text,
    validate_chunk_fits_model,
)
from graphrag_lab.jobs import CancelledError, InvalidInputError, JobStore

DEFAULT_DATASET_DIR = "artifacts/dataset"
DEFAULT_BATCH_SIZE = 32


def _read_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise InvalidInputError(
            f"No prepared dataset at {dataset_dir} (missing manifest.json). "
            "Prepare a dataset first."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InvalidInputError(f"Dataset manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or not manifest.get("corpus_id"):
        raise InvalidInputError("Dataset manifest has no corpus_id")
    return manifest


def _read_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise InvalidInputError(f"No chunks in {chunks_path}")
    return rows


def run_vector_index(
    *,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    snapshot_id: str,
    store: JobStore,
    job_id: str,
    embedding_model: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    database: str = "neo4j",
    driver: Any | None = None,
    embed_backend: Any | None = None,
    on_batch: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Embed every chunk and ingest idempotently; batch progress persists.

    Raises CancelledError when the job's cancel flag is set (checked between
    batches; an in-flight embedding batch cannot be undone). Records wall
    time as local compute; API cost stays unknown (None), never zero.
    """
    if not snapshot_id.strip():
        raise InvalidInputError("snapshot_id must be nonempty")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise InvalidInputError("batch_size must be a positive integer")
    snapshot = store.get_snapshot(snapshot_id.strip())
    if snapshot is None:
        raise InvalidInputError(f"snapshot {snapshot_id!r} does not exist")
    dataset_path = Path(dataset_dir)
    manifest = _read_manifest(dataset_path)
    corpus_id = str(manifest["corpus_id"])
    chunks_path = chunk_dataset_directory(dataset_path, corpus_id=corpus_id)
    chunks = _read_chunks(chunks_path)

    from graphrag_lab.neo4j_store import (
        create_driver,
        ensure_chunk_constraints,
        ensure_index_meta,
        ensure_vector_index,
        ingest_chunks,
        verify_connectivity,
    )

    backend = embed_backend or SentenceTransformerBackend(embedding_model)
    if backend.model_name != embedding_model:
        from graphrag_lab.neo4j_store import ConfigMismatchError

        raise ConfigMismatchError(
            f"Requested model {embedding_model!r} != loaded model {backend.model_name!r}"
        )
    for position, chunk in enumerate(chunks):
        validate_chunk_fits_model(str(chunk["title"]), str(chunk["text"]), backend)
        _ = position
    dimension = int(backend.dimension)

    from graphrag_lab.config import load_config

    owned_driver = False
    if driver is None:
        config = load_config("configs/default.yaml")
        if config.neo4j_password is None:
            raise InvalidInputError(
                "GRAPHRAG_NEO4J_PASSWORD is not set; cannot build the vector index"
            )
        driver = create_driver(config)
        database = config.neo4j.database
        owned_driver = True
    started = time.monotonic()
    store.update_snapshot_status(snapshot_id.strip(), status="building", vector_status="building")
    try:
        verify_connectivity(driver, database)
        ensure_chunk_constraints(driver, database)
        ensure_vector_index(driver, database, dimension)
        fingerprint = ensure_index_meta(
            driver,
            database,
            corpus_id=corpus_id,
            embedding_model=backend.model_name,
            dimension=dimension,
            chunker_version=CHUNKER_VERSION,
        )
        total = len(chunks)
        store.update_progress(job_id, progress_done=0, progress_total=total)
        done = 0
        for start in range(0, total, batch_size):
            if store.get_job(job_id).cancel_requested:
                raise CancelledError(f"Job {job_id} cancelled at batch {done}/{total}")
            batch = chunks[start : start + batch_size]
            texts = [build_embedding_text(str(c["title"]), str(c["text"])) for c in batch]
            vectors = backend.embed_texts(texts)
            ingest_chunks(driver, database, batch, vectors, embedding_model=backend.model_name)
            done += len(batch)
            store.update_progress(job_id, progress_done=done, progress_total=total)
            if on_batch is not None:
                on_batch(done, total)
    finally:
        if owned_driver:
            try:
                driver.close()
            except Exception:
                pass
    elapsed_s = round(time.monotonic() - started, 1)
    overall = "ready" if snapshot.get("graph_status") == "ready" else "building"
    store.update_snapshot_status(snapshot_id.strip(), status=overall, vector_status="ready")
    return {
        "snapshot_id": snapshot_id.strip(),
        "corpus_id": corpus_id,
        "n_chunks": len(chunks),
        "dimension": dimension,
        "embedding_model": backend.model_name,
        "chunker_version": CHUNKER_VERSION,
        "index_fingerprint": fingerprint,
        "elapsed_s": elapsed_s,
        "estimated_api_cost_usd": None,
        "limitation": "Wall time is measured local compute, not priced; API cost unknown.",
    }


def make_index_vector_handler(
    store: JobStore, *, embedding_model: str, database: str = "neo4j"
) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    """Worker handler: one ``{"id": "build", ...}`` item builds the index."""

    def _handler(item: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        job = store.get_job(str(ctx["job_id"]))
        if job.kind != "index_vector":
            raise InvalidInputError(f"index handler got kind {job.kind!r}")
        dataset_dir = str(item.get("dataset_dir", DEFAULT_DATASET_DIR))
        batch_size = item.get("batch_size", DEFAULT_BATCH_SIZE)
        try:
            batch_int = int(batch_size)
        except (TypeError, ValueError) as exc:
            raise InvalidInputError(f"batch_size must be an integer: {exc}") from exc
        config = dict(job.config_json.get("budget", {}) or {})
        _ = config
        if not job.snapshot_id:
            raise InvalidInputError("index_vector job has no snapshot_id")
        return run_vector_index(
            dataset_dir=dataset_dir,
            snapshot_id=job.snapshot_id,
            store=store,
            job_id=job.id,
            embedding_model=embedding_model,
            batch_size=batch_int,
            database=database,
        )

    return _handler


def chunk_config_for_snapshot() -> ChunkConfig:
    return ChunkConfig()


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DATASET_DIR",
    "chunk_config_for_snapshot",
    "make_index_vector_handler",
    "run_vector_index",
]
