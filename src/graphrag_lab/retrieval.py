"""Vector retrieval for T04 (vector-qa).

Initial retrieval returns up to 30 cosine candidates; later tickets add
reranking and hybrid arms under identical final budgets. Retrieval verifies
the active fingerprint and refuses mixed-corpus or offline-index reads.
"""

from __future__ import annotations

from typing import Any, Literal

from neo4j import Driver
from pydantic import BaseModel, ConfigDict, Field

from graphrag_lab.embeddings import EmbeddingBackend, ensure_normalized, validate_embedding
from graphrag_lab.neo4j_store import (
    ConfigMismatchError,
    read_index_meta,
    vector_search,
)

DEFAULT_NUM_CANDIDATES = 30
MAX_QUESTION_CHARS = 2000


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class RetrievalConfig(Record):
    """Tunable vector retrieval budgets (R03)."""

    num_candidates: int = Field(default=DEFAULT_NUM_CANDIDATES, ge=1, le=100, strict=True)
    max_question_chars: int = Field(default=MAX_QUESTION_CHARS, ge=1, strict=True)


class RetrievalRequest(Record):
    """One vector-qa lookup against the active index namespace."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    corpus_id: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)


class RetrievedChunk(Record):
    chunk_id: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    paragraph_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    sentence_ids: list[int]
    sentence_texts: list[str]
    score: float
    rank: int = Field(ge=1, strict=True)
    origin: Literal["vector"] = "vector"


class RetrievalResult(Record):
    question: str
    corpus_id: str
    embedding_model: str
    dimension: int = Field(ge=1, strict=True)
    index_fingerprint: str = Field(min_length=1)
    total_candidates: int = Field(ge=0, strict=True)
    chunks: list[RetrievedChunk]


def _validate_question(question: str, config: RetrievalConfig) -> str:
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must be a nonempty string")
    if len(question) > config.max_question_chars:
        raise ValueError(
            f"question length {len(question)} exceeds limit {config.max_question_chars}"
        )
    return cleaned


def retrieve_vector_candidates(
    request: RetrievalRequest,
    backend: EmbeddingBackend,
    driver: Driver,
    database: str,
    config: RetrievalConfig | None = None,
) -> RetrievalResult:
    """Embed the question and return ranked vector candidates."""
    cfg = config or RetrievalConfig()
    if request.corpus_id.strip() != request.corpus_id or not request.corpus_id.strip():
        raise ValueError("corpus_id must be a nonempty string without surrounding whitespace")
    cleaned = _validate_question(request.question, cfg)
    if backend.model_name != request.embedding_model:
        raise ConfigMismatchError(
            f"Request model {request.embedding_model!r} != loaded model {backend.model_name!r}; "
            "refusing mismatch"
        )
    query_vector = backend.embed_texts([cleaned])[0]
    return retrieve_with_vector(
        question=cleaned,
        query_vector=query_vector,
        corpus_id=request.corpus_id,
        embedding_model=request.embedding_model,
        dimension=backend.dimension,
        driver=driver,
        database=database,
        config=cfg,
    )


def retrieve_with_vector(
    *,
    question: str,
    query_vector: list[float],
    corpus_id: str,
    embedding_model: str,
    dimension: int,
    driver: Driver,
    database: str,
    config: RetrievalConfig | None = None,
) -> RetrievalResult:
    """Rank candidates for a precomputed query vector (testable without a model)."""
    cfg = config or RetrievalConfig()
    cleaned = _validate_question(question, cfg)
    if not corpus_id or not corpus_id.strip():
        raise ValueError("corpus_id must be a nonempty string")
    normalized = ensure_normalized(list(map(float, query_vector)))
    validate_embedding(normalized, dimension, what="query_embedding")
    meta = read_index_meta(driver, database)
    if meta is None:
        from graphrag_lab.neo4j_store import IndexNotReadyError

        raise IndexNotReadyError("No vector index fingerprint is registered")
    if meta.get("corpus_id") != corpus_id:
        raise ConfigMismatchError(
            f"Request corpus {corpus_id!r} != indexed corpus {meta.get('corpus_id')!r}; "
            "refusing mixed corpora"
        )
    if meta.get("embedding_model") != embedding_model:
        raise ConfigMismatchError(
            f"Request model {embedding_model!r} != indexed model {meta.get('embedding_model')!r}"
        )
    if int(meta.get("dimension", -1)) != dimension:
        raise ConfigMismatchError(
            f"Request dimension {dimension} != indexed dimension {meta.get('dimension')!r}"
        )
    rows: list[dict[str, Any]] = vector_search(driver, database, normalized, cfg.num_candidates)
    # The store enforces a single corpus, so every ANN hit belongs to it.
    # Defense in depth: drop any unexpected cross-corpus row rather than leaking it.
    rows = [row for row in rows if row.get("corpus_id") == corpus_id]
    chunks = [
        RetrievedChunk(
            chunk_id=str(row["id"]),
            corpus_id=str(row["corpus_id"]),
            paragraph_id=str(row["paragraph_id"]),
            title=str(row["title"]),
            text=str(row["text"]),
            sentence_ids=[int(v) for v in list(row["sentence_ids"] or [])],
            sentence_texts=[str(v) for v in list(row["sentence_texts"] or [])],
            score=float(row["score"]),
            rank=index + 1,
            origin="vector",
        )
        for index, row in enumerate(rows)
    ]
    fingerprint = str(meta["fingerprint"])
    return RetrievalResult(
        question=cleaned,
        corpus_id=corpus_id,
        embedding_model=embedding_model,
        dimension=dimension,
        index_fingerprint=fingerprint,
        total_candidates=len(chunks),
        chunks=chunks,
    )


# ---------------------------------------------------------------------------
# T11 reranked vector arm (B): same candidates, cross-encoder reorder.
# ---------------------------------------------------------------------------


def order_chunks_by_rerank(chunks: list[Any], ordered_ids: list[str]) -> list[Any]:
    """Reorder chunk objects into reranked order (deterministic).

    Duplicates were already removed by the reranker; any id in
    ``ordered_ids`` missing from ``chunks`` raises INVALID_INPUT instead of
    fabricating evidence.
    """
    by_id: dict[str, Any] = {}
    for item in chunks:
        cid = (
            item["chunk_id"]
            if isinstance(item, dict)
            else getattr(item, "chunk_id", getattr(item, "id", None))
        )
        if isinstance(cid, str) and cid not in by_id:
            by_id[cid] = item
    ordered: list[Any] = []
    for cid in ordered_ids:
        if cid not in by_id:
            raise ValueError(f"Reranked chunk id {cid!r} has no source chunk; refusing")
        ordered.append(by_id[cid])
    return ordered


def rerank_vector_chunks(
    question: str,
    chunks: list[Any],
    reranker: Any,
    rerank_config: Any | None = None,
) -> tuple[list[Any], Any]:
    """Deduplicate (first wins) and rerank vector candidates.

    Returns (ordered_chunks, RerankResult). Raw vector cosine scores are kept
    on the chunk objects for audit and never added to reranker logits; the
    order comes from reranker scores only with chunk-id tie-breaks. Reranker
    failure degrades to the deduped input order with ``degraded=True``.
    """
    from graphrag_lab.reranking import RerankConfig, deduplicate_chunks, rerank_candidates

    cfg = rerank_config or RerankConfig()
    if not isinstance(cfg, RerankConfig):
        cfg = RerankConfig.model_validate(cfg if isinstance(cfg, dict) else cfg.model_dump())
    deduped, _removed = deduplicate_chunks(list(chunks))
    result = rerank_candidates(question, deduped, reranker, cfg)
    ordered = order_chunks_by_rerank(deduped, result.ordered_chunk_ids)
    return ordered, result
