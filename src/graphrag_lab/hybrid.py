"""Bounded hybrid retrieval for T10 (hybrid-qa, R07).

Merges vector candidates with evidence-backed graph expansion:

1. Retrieve vector candidates (30 by default, deterministic score/id order).
2. Link up to five question entities without creating graph nodes.
3. Expand seeds through REL for at most two hops, either direction for
   recall while retaining semantic direction in displayed triples.
4. Fetch relation evidence chunks first, then mentioning chunks (lower
   pre-rerank priority).
5. Enforce frontier budgets before expansion (25 edges/node/hop, 100
   visited entities, 40 graph chunks, 2s timeout); record every cap hit.
6. Merge by chunk id with deterministic reciprocal-rank fusion (no raw
   score arithmetic); keep vector/graph/both origin, ranks, path ids and
   distance; select at most 60 candidates before reranking (T11).

No linked entity or traversal failure falls back to vector retrieval with
a visible flag; config mismatches still raise instead of hiding errors.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from graphrag_lab.embeddings import EmbeddingBackend

HYBRID_VERSION = "hybrid-v1"
RRF_K = 60

DEFAULT_MAX_HOPS = 2
DEFAULT_MAX_EDGES_PER_NODE = 25
DEFAULT_MAX_VISITED_ENTITIES = 100
DEFAULT_MAX_GRAPH_CHUNKS = 40
DEFAULT_GRAPH_TIMEOUT_S = 2.0
DEFAULT_MAX_MERGED = 60
DEFAULT_NUM_VECTOR_CANDIDATES = 30
DEFAULT_MAX_QUESTION_CHARS = 2000


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class HybridConfig(Record):
    """Bounded budgets for hybrid retrieval (R07, TRD hybrid-qa)."""

    max_hops: int = Field(default=DEFAULT_MAX_HOPS, ge=1, le=2, strict=True)
    max_edges_per_node: int = Field(default=DEFAULT_MAX_EDGES_PER_NODE, ge=1, le=200, strict=True)
    max_visited_entities: int = Field(
        default=DEFAULT_MAX_VISITED_ENTITIES, ge=1, le=1000, strict=True
    )
    max_graph_chunks: int = Field(default=DEFAULT_MAX_GRAPH_CHUNKS, ge=1, le=500, strict=True)
    graph_timeout_s: float = Field(default=DEFAULT_GRAPH_TIMEOUT_S, gt=0.0)
    max_merged: int = Field(default=DEFAULT_MAX_MERGED, ge=1, le=100, strict=True)
    num_vector_candidates: int = Field(
        default=DEFAULT_NUM_VECTOR_CANDIDATES, ge=1, le=100, strict=True
    )
    max_question_chars: int = Field(default=DEFAULT_MAX_QUESTION_CHARS, ge=1, strict=True)
    hybrid_version: str = Field(default=HYBRID_VERSION, min_length=1)


class HybridChunk(Record):
    chunk_id: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    paragraph_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    sentence_ids: list[int]
    sentence_texts: list[str]
    origin: Literal["vector", "graph", "both"]
    vector_rank: int | None = Field(default=None, ge=1)
    graph_rank: int | None = Field(default=None, ge=1)
    fusion_score: float = Field(ge=0.0)
    rank: int = Field(ge=1, strict=True)
    distance: int | None = Field(default=None, ge=0)
    path_ids: list[str] = Field(default_factory=list)
    evidence_kind: Literal["vector_only", "direct", "mention", "both_evidence_mention"] = (
        "vector_only"
    )


class HybridResult(Record):
    question: str
    corpus_id: str
    embedding_model: str
    dimension: int = Field(ge=1, strict=True)
    hybrid_version: str = Field(min_length=1)
    index_fingerprint: str | None = None
    graph_fingerprint: str | None = None
    total_vector: int = Field(ge=0, strict=True)
    total_graph: int = Field(ge=0, strict=True)
    total_merged: int = Field(ge=0, strict=True)
    chunks: list[HybridChunk]
    linked_entity_ids: list[str]
    visited_entities: list[str]
    cap_hits: list[str]
    fallback_to_vector: bool = False
    fallback_reason: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


def reciprocal_rank_fusion(
    vector_order: list[str],
    graph_order: list[str],
    k: int = RRF_K,
) -> dict[str, float]:
    """Deterministic RRF over ranks only; raw vector/graph scores never mix."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    for label, order in (("vector_order", vector_order), ("graph_order", graph_order)):
        if not isinstance(order, list):
            raise ValueError(f"{label} must be a list")
        if len(set(order)) != len(order):
            raise ValueError(f"{label} must not repeat chunk ids")
        for cid in order:
            if not isinstance(cid, str) or not cid:
                raise ValueError(f"{label} chunk ids must be nonempty strings")
    scores: dict[str, float] = {}
    for rank, cid in enumerate(vector_order, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(graph_order, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def _validate_question(question: str, max_chars: int) -> str:
    from graphrag_lab.llm import InvalidInputError

    if not isinstance(question, str):
        raise InvalidInputError("question must be a string")
    cleaned = question.strip()
    if not cleaned:
        raise InvalidInputError("question must be a nonempty string")
    if len(question) > max_chars:
        raise InvalidInputError(f"question length {len(question)} exceeds limit {max_chars}")
    return cleaned


def _vector_only_result(
    *,
    question: str,
    corpus_id: str,
    embedding_model: str,
    dimension: int,
    index_fingerprint: str | None,
    graph_fingerprint: str | None,
    vector_order: list[str],
    vector_details: dict[str, dict[str, Any]],
    linked_ids: list[str],
    visited: list[str],
    cap_hits: list[str],
    fallback_reason: str,
    diagnostics: dict[str, Any],
    hybrid_version: str,
    max_merged: int,
) -> HybridResult:
    scores = reciprocal_rank_fusion(vector_order, [], k=RRF_K)
    ordered = sorted(vector_order, key=lambda c: (-scores[c], c))[:max_merged]
    chunks = [
        HybridChunk(
            chunk_id=cid,
            corpus_id=str(vector_details[cid]["corpus_id"]),
            paragraph_id=str(vector_details[cid]["paragraph_id"]),
            title=str(vector_details[cid]["title"]),
            text=str(vector_details[cid]["text"]),
            sentence_ids=[int(v) for v in list(vector_details[cid]["sentence_ids"] or [])],
            sentence_texts=[str(v) for v in list(vector_details[cid]["sentence_texts"] or [])],
            origin="vector",
            vector_rank=index + 1,
            graph_rank=None,
            fusion_score=float(scores[cid]),
            rank=pos + 1,
            distance=None,
            path_ids=[],
            evidence_kind="vector_only",
        )
        for pos, (index, cid) in enumerate((vector_order.index(cid), cid) for cid in ordered)
    ]
    merged_hits = list(cap_hits)
    if len(vector_order) > max_merged and "max_merged_hit" not in merged_hits:
        merged_hits.append("max_merged_hit")
    return HybridResult(
        question=question,
        corpus_id=corpus_id,
        embedding_model=embedding_model,
        dimension=dimension,
        hybrid_version=hybrid_version,
        index_fingerprint=index_fingerprint,
        graph_fingerprint=graph_fingerprint,
        total_vector=len(vector_order),
        total_graph=0,
        total_merged=len(chunks),
        chunks=chunks,
        linked_entity_ids=list(linked_ids),
        visited_entities=list(visited),
        cap_hits=merged_hits,
        fallback_to_vector=True,
        fallback_reason=fallback_reason,
        diagnostics=dict(diagnostics),
    )


def hybrid_retrieve_with_vector(
    *,
    question: str,
    query_vector: list[float],
    corpus_id: str,
    embedding_model: str,
    dimension: int,
    driver: Any,
    database: str,
    vector_config: Any | None = None,
    hybrid_config: HybridConfig | None = None,
    linking_config: Any | None = None,
    clock: Any = None,
) -> HybridResult:
    """Merge vector candidates with bounded graph expansion (testable offline).

    ``query_vector`` avoids model downloads in tests; production callers use
    :func:`hybrid_retrieve`. Vector/config errors propagate; graph linking or
    traversal failures fall back to vector-only with a recorded reason.
    """
    from graphrag_lab.linking import LinkingConfig, link_question_entities
    from graphrag_lab.llm import InvalidInputError
    from graphrag_lab.neo4j_store import (
        ConfigMismatchError,
        GraphTimeoutError,
        expand_bounded_frontier,
        fetch_chunks_by_ids,
        read_graph_meta,
    )
    from graphrag_lab.retrieval import RetrievalConfig, retrieve_with_vector

    hcfg = hybrid_config or HybridConfig()
    if not isinstance(hcfg, HybridConfig):
        hcfg = HybridConfig.model_validate(hcfg if isinstance(hcfg, dict) else hcfg.model_dump())
    cleaned = _validate_question(question, hcfg.max_question_chars)
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise InvalidInputError("corpus_id must be a nonempty string")
    corpus = corpus_id.strip()
    if driver is None:
        raise InvalidInputError("driver must be supplied")
    if not isinstance(database, str) or not database.strip():
        raise InvalidInputError("database must be a nonempty string")

    vcfg = vector_config or RetrievalConfig(num_candidates=hcfg.num_vector_candidates)
    if not isinstance(vcfg, RetrievalConfig):
        vcfg = RetrievalConfig.model_validate(vcfg)
    if vcfg.num_candidates != hcfg.num_vector_candidates:
        # Caller's explicit vector budget wins for this lookup, but the
        # hybrid manifest keeps its own configured value for audit.
        pass

    vector_result = retrieve_with_vector(
        question=cleaned,
        query_vector=[float(v) for v in list(query_vector)],
        corpus_id=corpus,
        embedding_model=embedding_model,
        dimension=dimension,
        driver=driver,
        database=database,
        config=vcfg,
    )
    vector_order = [c.chunk_id for c in vector_result.chunks]
    vector_details = {
        c.chunk_id: {
            "corpus_id": c.corpus_id,
            "paragraph_id": c.paragraph_id,
            "title": c.title,
            "text": c.text,
            "sentence_ids": list(c.sentence_ids),
            "sentence_texts": list(c.sentence_texts),
        }
        for c in vector_result.chunks
    }

    lcfg = linking_config or LinkingConfig()
    if not isinstance(lcfg, LinkingConfig):
        lcfg = LinkingConfig.model_validate(lcfg)
    if lcfg.max_mentions > hcfg.max_mentions if hasattr(hcfg, "max_mentions") else False:
        pass  # hybrid has no mention cap of its own; linking enforces ≤5.

    linking = link_question_entities(cleaned, driver, database, corpus, lcfg)
    linked_ids = [e.entity_id for e in linking.linked]
    graph_meta = read_graph_meta(driver, database)
    graph_fingerprint = graph_meta.get("fingerprint") if graph_meta else None

    base_diag: dict[str, Any] = {
        "hybrid_version": hcfg.hybrid_version,
        "vector_candidates": len(vector_order),
        "linking": linking.diagnostics,
        "mentions": [m.model_dump(mode="json") for m in linking.mentions],
        "linked": [e.model_dump(mode="json") for e in linking.linked],
    }

    if not linked_ids:
        reason = (
            "graph_not_ready"
            if linking.diagnostics.get("graph_status") == "graph_not_ready"
            else "no_linked_entities"
        )
        diag = dict(base_diag)
        diag["expansion"] = {"status": "skipped", "reason": reason}
        return _vector_only_result(
            question=vector_result.question,
            corpus_id=corpus,
            embedding_model=embedding_model,
            dimension=dimension,
            index_fingerprint=vector_result.index_fingerprint,
            graph_fingerprint=graph_fingerprint,
            vector_order=vector_order,
            vector_details=vector_details,
            linked_ids=[],
            visited=[],
            cap_hits=[],
            fallback_reason=reason,
            diagnostics=diag,
            hybrid_version=hcfg.hybrid_version,
            max_merged=hcfg.max_merged,
        )

    try:
        expansion = expand_bounded_frontier(
            driver,
            database,
            linked_ids,
            corpus,
            max_hops=hcfg.max_hops,
            max_edges_per_node=hcfg.max_edges_per_node,
            max_visited_entities=hcfg.max_visited_entities,
            max_graph_chunks=hcfg.max_graph_chunks,
            timeout_s=hcfg.graph_timeout_s,
            clock=clock,
        )
    except GraphTimeoutError as exc:
        diag = dict(base_diag)
        diag["expansion"] = {"status": "timeout", "error": str(exc)[:300]}
        return _vector_only_result(
            question=vector_result.question,
            corpus_id=corpus,
            embedding_model=embedding_model,
            dimension=dimension,
            index_fingerprint=vector_result.index_fingerprint,
            graph_fingerprint=graph_fingerprint,
            vector_order=vector_order,
            vector_details=vector_details,
            linked_ids=linked_ids,
            visited=[],
            cap_hits=["graph_timeout_hit"],
            fallback_reason="graph_timeout",
            diagnostics=diag,
            hybrid_version=hcfg.hybrid_version,
            max_merged=hcfg.max_merged,
        )
    except ConfigMismatchError:
        raise
    except Exception as exc:  # noqa: BLE001 - traversal failure must fall back visibly
        diag = dict(base_diag)
        diag["expansion"] = {
            "status": "error",
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }
        return _vector_only_result(
            question=vector_result.question,
            corpus_id=corpus,
            embedding_model=embedding_model,
            dimension=dimension,
            index_fingerprint=vector_result.index_fingerprint,
            graph_fingerprint=graph_fingerprint,
            vector_order=vector_order,
            vector_details=vector_details,
            linked_ids=linked_ids,
            visited=[],
            cap_hits=["graph_error_hit"],
            fallback_reason="graph_error",
            diagnostics=diag,
            hybrid_version=hcfg.hybrid_version,
            max_merged=hcfg.max_merged,
        )

    graph_ids: list[str] = list(expansion["graph_chunk_ids"])
    if not graph_ids:
        diag = dict(base_diag)
        diag["expansion"] = {
            "status": "empty",
            "visited": expansion["visited"],
            "cap_hits": expansion["cap_hits"],
        }
        return _vector_only_result(
            question=vector_result.question,
            corpus_id=corpus,
            embedding_model=embedding_model,
            dimension=dimension,
            index_fingerprint=vector_result.index_fingerprint,
            graph_fingerprint=graph_fingerprint,
            vector_order=vector_order,
            vector_details=vector_details,
            linked_ids=linked_ids,
            visited=list(expansion["visited_entities"]),
            cap_hits=list(expansion["cap_hits"]),
            fallback_reason="graph_empty",
            diagnostics=diag,
            hybrid_version=hcfg.hybrid_version,
            max_merged=hcfg.max_merged,
        )

    shell_rows = fetch_chunks_by_ids(driver, database, graph_ids, corpus)
    shells = {row["id"]: row for row in shell_rows}
    # Evidence-first graph ranking: expansion already prioritizes direct
    # edge evidence over incidental mentions; preserve that order.
    graph_order = [cid for cid in graph_ids if cid in shells]
    missing = [cid for cid in graph_ids if cid not in shells]
    cap_hits: list[str] = list(expansion["cap_hits"])
    diag = dict(base_diag)
    diag["expansion"] = {
        "status": "ok",
        "visited": expansion["visited"],
        "traversed_relations": list(expansion["traversed_relations"]),
        "n_traversed_relations": len(expansion["traversed_relations"]),
        "evidence_chunks": list(expansion["evidence_chunk_ids"]),
        "mention_chunks": list(expansion["mention_chunk_ids"]),
        "missing_shells": missing,
        "cap_hits": list(expansion["cap_hits"]),
    }
    if missing:
        cap_hits.append("graph_missing_shells_hit")

    scores = reciprocal_rank_fusion(vector_order, graph_order, k=RRF_K)
    all_ids = sorted(set(vector_order) | set(graph_order), key=lambda c: (-scores[c], c))
    truncated = len(all_ids) > hcfg.max_merged
    if truncated:
        cap_hits.append("max_merged_hit")
        all_ids = all_ids[: hcfg.max_merged]

    vector_rank_of = {cid: i + 1 for i, cid in enumerate(vector_order)}
    graph_rank_of = {cid: i + 1 for i, cid in enumerate(graph_order)}
    chunk_distance: dict[str, Any] = dict(expansion.get("chunk_distance", {}))
    chunk_paths: dict[str, Any] = dict(expansion.get("chunk_paths", {}))
    chunk_kind: dict[str, Any] = dict(expansion.get("chunk_kind", {}))

    merged: list[HybridChunk] = []
    for pos, cid in enumerate(all_ids, start=1):
        if cid in vector_details:
            detail = vector_details[cid]
        elif cid in shells:
            row = shells[cid]
            detail = {
                "corpus_id": str(row["corpus_id"]),
                "paragraph_id": str(row["paragraph_id"]),
                "title": str(row["title"]),
                "text": str(row["text"]),
                "sentence_ids": [int(v) for v in list(row["sentence_ids"] or [])],
                "sentence_texts": [str(v) for v in list(row["sentence_texts"] or [])],
            }
        else:  # pragma: no cover - missing shells are filtered above
            continue
        in_vector = cid in vector_rank_of
        in_graph = cid in graph_rank_of
        origin: Literal["vector", "graph", "both"] = (
            "both" if (in_vector and in_graph) else ("graph" if in_graph else "vector")
        )
        kind_raw = str(chunk_kind.get(cid, "vector_only"))
        evidence_kind: Literal["vector_only", "direct", "mention", "both_evidence_mention"] = (
            kind_raw  # type: ignore[assignment]
            if kind_raw in ("direct", "mention", "both_evidence_mention")
            else "vector_only"
        )
        merged.append(
            HybridChunk(
                chunk_id=cid,
                corpus_id=str(detail["corpus_id"]),
                paragraph_id=str(detail["paragraph_id"]),
                title=str(detail["title"]),
                text=str(detail["text"]),
                sentence_ids=list(detail["sentence_ids"]),
                sentence_texts=list(detail["sentence_texts"]),
                origin=origin,
                vector_rank=vector_rank_of.get(cid),
                graph_rank=graph_rank_of.get(cid),
                fusion_score=float(scores[cid]),
                rank=pos,
                distance=int(chunk_distance[cid])
                if cid in chunk_distance and chunk_distance[cid] is not None
                else None,
                path_ids=[str(v) for v in list(chunk_paths.get(cid, []))],
                evidence_kind=evidence_kind,
            )
        )

    return HybridResult(
        question=vector_result.question,
        corpus_id=corpus,
        embedding_model=embedding_model,
        dimension=dimension,
        hybrid_version=hcfg.hybrid_version,
        index_fingerprint=vector_result.index_fingerprint,
        graph_fingerprint=graph_fingerprint,
        total_vector=len(vector_order),
        total_graph=len(graph_order),
        total_merged=len(merged),
        chunks=merged,
        linked_entity_ids=sorted(linked_ids),
        visited_entities=list(expansion["visited_entities"]),
        cap_hits=cap_hits,
        fallback_to_vector=False,
        fallback_reason=None,
        diagnostics=diag,
    )


def hybrid_retrieve(
    question: str,
    backend: EmbeddingBackend,
    driver: Any,
    database: str,
    corpus_id: str,
    hybrid_config: HybridConfig | None = None,
    vector_config: Any | None = None,
    linking_config: Any | None = None,
    clock: Any = None,
) -> HybridResult:
    """Embed the question then run bounded hybrid retrieval."""
    from graphrag_lab.neo4j_store import ConfigMismatchError
    from graphrag_lab.retrieval import RetrievalConfig

    hcfg = hybrid_config or HybridConfig()
    cleaned = _validate_question(question, hcfg.max_question_chars)
    if backend.model_name is None or not str(backend.model_name).strip():
        raise ValueError("backend model_name must be nonempty")
    vcfg = vector_config or RetrievalConfig(num_candidates=hcfg.num_vector_candidates)
    query_vector = backend.embed_texts([cleaned])[0]
    try:
        return hybrid_retrieve_with_vector(
            question=cleaned,
            query_vector=query_vector,
            corpus_id=corpus_id,
            embedding_model=backend.model_name,
            dimension=backend.dimension,
            driver=driver,
            database=database,
            vector_config=vcfg,
            hybrid_config=hcfg,
            linking_config=linking_config,
            clock=clock,
        )
    except ConfigMismatchError:
        raise


def rerank_hybrid_chunks(
    question: str,
    chunks: list[Any],
    reranker: Any,
    rerank_config: Any | None = None,
) -> tuple[list[Any], Any]:
    """Deduplicate (first wins) and rerank merged hybrid candidates.

    Mirrors :func:`graphrag_lab.retrieval.rerank_vector_chunks`: the order
    comes from reranker scores only (chunk-id tie-break); vector ranks, graph
    ranks, fusion scores and distances are retained on the chunk objects for
    audit and never summed with reranker logits. Failure degrades to the
    deduped pre-rerank order with ``degraded=True``.
    """
    from graphrag_lab.reranking import RerankConfig, deduplicate_chunks, rerank_candidates
    from graphrag_lab.retrieval import order_chunks_by_rerank

    cfg = rerank_config or RerankConfig()
    if not isinstance(cfg, RerankConfig):
        cfg = RerankConfig.model_validate(cfg if isinstance(cfg, dict) else cfg.model_dump())
    deduped, _removed = deduplicate_chunks(list(chunks))
    result = rerank_candidates(question, deduped, reranker, cfg)
    ordered = order_chunks_by_rerank(deduped, result.ordered_chunk_ids)
    return ordered, result


def traversed_relations_for_result(result: HybridResult) -> list[dict[str, Any]]:
    """Return traversed REL edges for triples (arm D), or [] on fallback."""
    expansion = (result.diagnostics or {}).get("expansion", {})
    relations = expansion.get("traversed_relations", [])
    if isinstance(relations, int):
        # Backwards compatibility with pre-T11 diagnostics that stored only a
        # count; without edge evidence no triple can be built evidence-only.
        return []
    if not isinstance(relations, list):
        return []
    return [dict(r) for r in relations if isinstance(r, dict)]


__all__ = [
    "HYBRID_VERSION",
    "HybridChunk",
    "HybridConfig",
    "HybridResult",
    "reciprocal_rank_fusion",
    "hybrid_retrieve",
    "hybrid_retrieve_with_vector",
    "rerank_hybrid_chunks",
    "traversed_relations_for_result",
]
