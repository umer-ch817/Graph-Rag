"""Evidence-backed graph snapshot for T09 (graph-index, R06).

Builds a deterministic, corpus-scoped graph snapshot from validated chunk
extractions and conservative resolution output. Every directed ``REL`` edge
and every ``MENTIONS`` link preserves chunk evidence; repeated loads union
evidence deterministically; an incomplete snapshot cannot activate without
an explicit partial flag.

Neo4j persistence lives in ``neo4j_store.py``; this module owns the pure
snapshot, coverage and activation logic so it stays testable offline.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from graphrag_lab.chunking import ChunkRecord
from graphrag_lab.corpus import sha256_hex
from graphrag_lab.extraction import (
    EXTRACTION_VERSION,
    SCHEMA_VERSION,
    ChunkExtraction,
    ExtractionFailure,
    coerce_chunk_view,
)
from graphrag_lab.llm import InvalidInputError, ValidationFailedError
from graphrag_lab.models import EntityType, RelationType
from graphrag_lab.resolution import ResolutionResult

GRAPH_SNAPSHOT_VERSION = "graph-snapshot-v1"
GRAPH_MANIFEST_VERSION = "graph-manifest-v1"
REL_ID_PREFIX = "rel_"


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class GraphConfigMismatchError(ValueError):
    """Corpus/version mismatch; mirrors neo4j_store CONFIG_MISMATCH."""

    code = "CONFIG_MISMATCH"


class GraphNotReadyError(RuntimeError):
    """Incomplete snapshot activation refused; mirrors INDEX_NOT_READY."""

    code = "INDEX_NOT_READY"


class MentionEdge(Record):
    """One Chunk -> Entity link with aggregated mention evidence."""

    chunk_id: str = Field(min_length=1)
    canonical_entity_id: str = Field(min_length=1)
    mention_ids: list[str] = Field(min_length=1)
    sentence_ids: list[int]
    surface_forms: list[str] = Field(min_length=1)


class GraphRelation(Record):
    """One directed Entity -> Entity edge with unioned evidence chunks."""

    id: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    type: RelationType
    source_entity_id: str = Field(min_length=1)
    target_entity_id: str = Field(min_length=1)
    evidence_chunk_ids: list[str] = Field(min_length=1)
    extraction_version: str = Field(min_length=1)


class RelationEvidenceQuote(Record):
    sentence_id: int = Field(ge=0, strict=True)
    quote: str = Field(min_length=1)


class RelationEvidence(Record):
    """Nested evidence for one (relation, chunk) pair (JSONL artifact)."""

    relation_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    source_entity_id: str = Field(min_length=1)
    target_entity_id: str = Field(min_length=1)
    type: RelationType
    evidence: list[RelationEvidenceQuote] = Field(min_length=1)
    extraction_version: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)


class GraphSnapshot(Record):
    """Deterministic snapshot; all lists are sorted for stable hashes."""

    corpus_id: str = Field(min_length=1)
    resolver_version: str = Field(min_length=1)
    extraction_version: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    snapshot_version: str = Field(min_length=1)
    fingerprint: str = Field(min_length=1)
    entities: list[Any] = Field(default_factory=list)
    mentions: list[MentionEdge] = Field(default_factory=list)
    relations: list[GraphRelation] = Field(default_factory=list)
    evidence: list[RelationEvidence] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


class CoverageReport(Record):
    """Honest extraction coverage for the activation gate."""

    corpus_id: str = Field(min_length=1)
    n_chunks: int = Field(ge=0, strict=True)
    n_extracted: int = Field(ge=0, strict=True)
    n_failed: int = Field(ge=0, strict=True)
    n_unprocessed: int = Field(ge=0, strict=True)
    extracted_chunk_ids: list[str]
    failed_chunk_ids: list[str]
    missing_chunk_ids: list[str]
    coverage_ratio: float = Field(ge=0.0, le=1.0)
    is_complete: bool
    is_partial: bool = False


def build_relation_id(
    corpus_id: str,
    source_entity_id: str,
    relation_type: RelationType | str,
    target_entity_id: str,
) -> str:
    """Deterministic relation id scoped to corpus + endpoints + type."""
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise InvalidInputError("corpus_id must be a nonempty string")
    if not isinstance(source_entity_id, str) or not source_entity_id.strip():
        raise InvalidInputError("source_entity_id must be a nonempty string")
    if not isinstance(target_entity_id, str) or not target_entity_id.strip():
        raise InvalidInputError("target_entity_id must be a nonempty string")
    try:
        rel_type = (
            relation_type
            if isinstance(relation_type, RelationType)
            else RelationType(relation_type)
        )
    except ValueError as exc:
        raise ValidationFailedError(f"Unknown relation type {relation_type!r}") from exc
    digest = sha256_hex([corpus_id, source_entity_id, rel_type.value, target_entity_id])
    return f"{REL_ID_PREFIX}{digest[:12]}"


def build_graph_fingerprint(
    corpus_id: str,
    resolver_version: str,
    extraction_version: str,
    schema_version: str,
) -> str:
    """Fingerprint isolating one graph snapshot namespace.

    A resolver or extraction/schema change rebuilds the snapshot; historical
    run evidence is never silently remapped.
    """
    for label, value in (
        ("corpus_id", corpus_id),
        ("resolver_version", resolver_version),
        ("extraction_version", extraction_version),
        ("schema_version", schema_version),
    ):
        if not isinstance(value, str) or not value.strip():
            raise InvalidInputError(f"{label} must be a nonempty string")
    return sha256_hex([corpus_id, resolver_version, extraction_version, schema_version])


def _validate_corpus_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidInputError("corpus_id must be a nonempty string")
    if any(c.isspace() for c in value):
        raise InvalidInputError("corpus_id must contain no whitespace")
    return value.strip()


def _coerce_chunks(chunks: Any) -> tuple[str, dict[str, dict[str, Any]]]:
    if chunks is None or isinstance(chunks, (str, bytes)):
        raise InvalidInputError("chunks must be a sequence of chunk records")
    try:
        items = list(chunks)
    except TypeError as exc:
        raise InvalidInputError("chunks must be a sequence") from exc
    if not items:
        raise InvalidInputError("chunks must be a nonempty list")
    by_id: dict[str, dict[str, Any]] = {}
    corpora: set[str] = set()
    for index, item in enumerate(items):
        if isinstance(item, ChunkRecord):
            view = coerce_chunk_view(item)
        elif isinstance(item, dict):
            view = coerce_chunk_view(item)
        else:
            view = coerce_chunk_view(item)
        chunk_id = view["chunk_id"]
        if chunk_id in by_id:
            raise InvalidInputError(f"Duplicate chunk id: {chunk_id!r}")
        corpus_id = view.get("corpus_id")
        if not isinstance(corpus_id, str) or not corpus_id.strip():
            raise InvalidInputError(f"Chunk {chunk_id!r} has no corpus id")
        corpora.add(corpus_id)
        by_id[chunk_id] = view
        _ = index
    if len(corpora) != 1:
        raise GraphConfigMismatchError(
            f"All chunks in one snapshot must share one corpus_id, got {sorted(corpora)}"
        )
    return next(iter(corpora)), by_id


def _coerce_extractions(extractions: Any) -> list[ChunkExtraction]:
    if extractions is None or isinstance(extractions, (str, bytes)):
        raise InvalidInputError("extractions must be a sequence of chunk extractions")
    try:
        items = list(extractions)
    except TypeError as exc:
        raise InvalidInputError("extractions must be a sequence") from exc
    parsed = [
        item if isinstance(item, ChunkExtraction) else ChunkExtraction.model_validate(item)
        for item in items
    ]
    seen: set[str] = set()
    for item in parsed:
        if item.chunk_id in seen:
            raise InvalidInputError(f"Duplicate extraction chunk id: {item.chunk_id!r}")
        seen.add(item.chunk_id)
    return parsed


def _coerce_resolution(resolution: Any) -> ResolutionResult:
    if isinstance(resolution, ResolutionResult):
        return resolution
    if isinstance(resolution, dict):
        return ResolutionResult.model_validate(resolution)
    # Pydantic models from resolution.py expose model_validate as well.
    try:
        return ResolutionResult.model_validate(resolution)
    except Exception as exc:
        raise InvalidInputError(f"resolution must be a ResolutionResult: {exc}") from exc


def build_graph_snapshot(
    chunks: Sequence[ChunkRecord | dict[str, Any]],
    extractions: Sequence[ChunkExtraction | dict[str, Any]],
    resolution: ResolutionResult | dict[str, Any],
    *,
    extraction_version: str = EXTRACTION_VERSION,
    schema_version: str = SCHEMA_VERSION,
) -> GraphSnapshot:
    """Group validated extractions by canonical entities into a snapshot.

    Raises GraphConfigMismatchError on corpus/version drift and
    ValidationFailedError on dangling endpoints or evidence outside the
    chunk's sentences (non-verbatim or foreign-chunk quotes).
    """
    if not isinstance(extraction_version, str) or not extraction_version.strip():
        raise InvalidInputError("extraction_version must be a nonempty string")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise InvalidInputError("schema_version must be a nonempty string")
    corpus_id, chunk_by_id = _coerce_chunks(chunks)
    corpus_id = _validate_corpus_id(corpus_id)
    parsed_extractions = _coerce_extractions(extractions)
    result = _coerce_resolution(resolution)
    if result.corpus_id != corpus_id:
        raise GraphConfigMismatchError(
            f"Resolution corpus {result.corpus_id!r} != chunk corpus {corpus_id!r}; refusing mix"
        )
    for item in parsed_extractions:
        if item.schema_version != schema_version:
            raise ValidationFailedError(
                f"Chunk {item.chunk_id!r} schema_version {item.schema_version!r} "
                f"!= expected {schema_version!r}"
            )
        if item.chunk_id not in chunk_by_id:
            raise ValidationFailedError(
                f"Extraction chunk {item.chunk_id!r} is outside the snapshot corpus"
            )

    mention_to_canonical: dict[str, str] = {}
    for assignment in result.assignments:
        if assignment.chunk_id not in chunk_by_id:
            raise ValidationFailedError(
                f"Resolution mention {assignment.mention_id!r} references "
                f"chunk {assignment.chunk_id!r} outside the snapshot corpus"
            )
        if assignment.mention_id in mention_to_canonical:
            raise InvalidInputError(f"Duplicate mention id: {assignment.mention_id!r}")
        mention_to_canonical[assignment.mention_id] = assignment.canonical_entity_id
    canonical_ids = {entity.id for entity in result.entities}
    for entity in result.entities:
        if entity.corpus_id != corpus_id:
            raise GraphConfigMismatchError(
                f"Entity {entity.id!r} corpus {entity.corpus_id!r} != snapshot corpus {corpus_id!r}"
            )

    # Every extraction entity must resolve; every relation endpoint must exist.
    for item in parsed_extractions:
        for entity in item.entities:
            mention_id = f"{item.chunk_id}#{entity.local_id}"
            if mention_id not in mention_to_canonical:
                raise ValidationFailedError(
                    f"Chunk {item.chunk_id!r} entity {entity.local_id!r} has no "
                    "resolution assignment; rebuild resolution from these extractions"
                )

    mentions_accum: dict[tuple[str, str], dict[str, Any]] = {}
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in parsed_extractions:
        view = chunk_by_id[item.chunk_id]
        allowed = set(view["sentence_ids"])
        by_sentence = dict(zip(view["sentence_ids"], view["sentence_texts"]))
        for entity in item.entities:
            canonical = mention_to_canonical[f"{item.chunk_id}#{entity.local_id}"]
            key = (item.chunk_id, canonical)
            slot = mentions_accum.setdefault(
                key, {"mention_ids": set(), "sentence_ids": set(), "surface_forms": set()}
            )
            slot["mention_ids"].add(f"{item.chunk_id}#{entity.local_id}")
            for sid in entity.sentence_ids:
                if sid not in allowed:
                    raise ValidationFailedError(
                        f"Chunk {item.chunk_id!r} entity {entity.local_id!r} cites "
                        f"sentence {sid} outside this chunk"
                    )
                slot["sentence_ids"].add(int(sid))
            slot["surface_forms"].add(entity.name)
        for relation in item.relations:
            source_mention = f"{item.chunk_id}#{relation.source}"
            target_mention = f"{item.chunk_id}#{relation.target}"
            if source_mention not in mention_to_canonical:
                raise ValidationFailedError(
                    f"Chunk {item.chunk_id!r} relation source {relation.source!r} is dangling"
                )
            if target_mention not in mention_to_canonical:
                raise ValidationFailedError(
                    f"Chunk {item.chunk_id!r} relation target {relation.target!r} is dangling"
                )
            source_canonical = mention_to_canonical[source_mention]
            target_canonical = mention_to_canonical[target_mention]
            if source_canonical not in canonical_ids or target_canonical not in canonical_ids:
                raise ValidationFailedError(
                    f"Chunk {item.chunk_id!r} relation endpoints do not resolve to entities"
                )
            spans: list[dict[str, Any]] = []
            for span in relation.evidence:
                if span.sentence_id not in allowed:
                    raise ValidationFailedError(
                        f"Chunk {item.chunk_id!r} relation evidence cites sentence "
                        f"{span.sentence_id} outside this chunk"
                    )
                sentence_text = by_sentence[span.sentence_id]
                if span.quote not in sentence_text:
                    raise ValidationFailedError(
                        f"Chunk {item.chunk_id!r} relation quote is not a verbatim "
                        f"span of sentence {span.sentence_id}"
                    )
                spans.append({"sentence_id": int(span.sentence_id), "quote": span.quote})
            spans.sort(key=lambda s: (s["sentence_id"], s["quote"]))
            key = (source_canonical, relation.type.value, target_canonical)
            slot = grouped.setdefault(key, {"chunks": {}, "type": relation.type})
            if item.chunk_id in slot["chunks"]:
                # Same relation repeated in one chunk: union spans deterministically.
                existing = slot["chunks"][item.chunk_id]
                merged = sorted({json.dumps(s, sort_keys=True) for s in [*existing, *spans]})
                slot["chunks"][item.chunk_id] = [json.loads(s) for s in merged]
            else:
                slot["chunks"][item.chunk_id] = spans

    mentions = [
        MentionEdge(
            chunk_id=chunk_id,
            canonical_entity_id=canonical,
            mention_ids=sorted(accum["mention_ids"]),
            sentence_ids=sorted(accum["sentence_ids"]),
            surface_forms=sorted(accum["surface_forms"]),
        )
        for (chunk_id, canonical), accum in mentions_accum.items()
    ]
    mentions.sort(key=lambda m: (m.chunk_id, m.canonical_entity_id))

    relations: list[GraphRelation] = []
    evidence_rows: list[RelationEvidence] = []
    for (source, type_value, target), slot in grouped.items():
        rel_type = slot["type"]
        rel_id = build_relation_id(corpus_id, source, rel_type, target)
        chunk_ids = sorted(slot["chunks"])
        relations.append(
            GraphRelation(
                id=rel_id,
                corpus_id=corpus_id,
                type=rel_type,
                source_entity_id=source,
                target_entity_id=target,
                evidence_chunk_ids=chunk_ids,
                extraction_version=extraction_version.strip(),
            )
        )
        for chunk_id in chunk_ids:
            spans = sorted(slot["chunks"][chunk_id], key=lambda s: (s["sentence_id"], s["quote"]))
            evidence_rows.append(
                RelationEvidence(
                    relation_id=rel_id,
                    chunk_id=chunk_id,
                    source_entity_id=source,
                    target_entity_id=target,
                    type=rel_type,
                    evidence=[
                        RelationEvidenceQuote(sentence_id=s["sentence_id"], quote=s["quote"])
                        for s in spans
                    ],
                    extraction_version=extraction_version.strip(),
                    schema_version=schema_version.strip(),
                )
            )
    relations.sort(key=lambda r: r.id)
    evidence_rows.sort(key=lambda e: (e.relation_id, e.chunk_id))

    fingerprint = build_graph_fingerprint(
        corpus_id, result.resolver_version, extraction_version.strip(), schema_version.strip()
    )
    total_spans = sum(len(e.evidence) for e in evidence_rows)
    stats: dict[str, Any] = {
        "n_chunks": len(chunk_by_id),
        "n_extracted_chunks": len(parsed_extractions),
        "n_entities": len(result.entities),
        "n_mention_edges": len(mentions),
        "n_relations": len(relations),
        "n_evidence_pairs": len(evidence_rows),
        "n_evidence_spans": total_spans,
        "resolver_version": result.resolver_version,
        "extraction_version": extraction_version.strip(),
        "schema_version": schema_version.strip(),
        "snapshot_version": GRAPH_SNAPSHOT_VERSION,
        "limitation": (
            "Graph edges preserve validated chunk evidence only; syntactically valid "
            "but false claims remain possible and require pilot audit before scaling."
        ),
    }
    entities_sorted = sorted(result.entities, key=lambda e: e.id)
    return GraphSnapshot(
        corpus_id=corpus_id,
        resolver_version=result.resolver_version,
        extraction_version=extraction_version.strip(),
        schema_version=schema_version.strip(),
        snapshot_version=GRAPH_SNAPSHOT_VERSION,
        fingerprint=fingerprint,
        entities=[e.model_dump(mode="json") for e in entities_sorted],
        mentions=mentions,
        relations=relations,
        evidence=evidence_rows,
        stats=stats,
    )


def validate_graph_snapshot(
    snapshot: GraphSnapshot | dict[str, Any],
    chunks: Sequence[ChunkRecord | dict[str, Any]] | None = None,
) -> GraphSnapshot:
    """Re-check endpoints, evidence references and corpus scoping."""
    record = (
        snapshot if isinstance(snapshot, GraphSnapshot) else GraphSnapshot.model_validate(snapshot)
    )
    corpus_id = _validate_corpus_id(record.corpus_id)
    expected = build_graph_fingerprint(
        record.corpus_id, record.resolver_version, record.extraction_version, record.schema_version
    )
    if record.fingerprint != expected:
        raise ValidationFailedError("Graph snapshot fingerprint does not match its versions")
    entity_ids = set()
    for entity in record.entities:
        eid = entity.get("id") if isinstance(entity, dict) else getattr(entity, "id", None)
        ecorpus = (
            entity.get("corpus_id")
            if isinstance(entity, dict)
            else getattr(entity, "corpus_id", None)
        )
        if not isinstance(eid, str) or not eid:
            raise ValidationFailedError("Graph entity has no id")
        if ecorpus != corpus_id:
            raise GraphConfigMismatchError(
                f"Entity {eid!r} corpus {ecorpus!r} != snapshot corpus {corpus_id!r}"
            )
        entity_ids.add(eid)
    chunk_ids: set[str] | None = None
    chunk_views: dict[str, dict[str, Any]] | None = None
    if chunks is not None:
        _, chunk_views = _coerce_chunks(chunks)
        if set(chunk_views) and next(iter(chunk_views.values()))["corpus_id"] != corpus_id:
            raise GraphConfigMismatchError("Snapshot corpus does not match supplied chunks")
        chunk_ids = set(chunk_views)
    for mention in record.mentions:
        if mention.canonical_entity_id not in entity_ids:
            raise ValidationFailedError(
                f"MENTIONS {mention.chunk_id!r}->{mention.canonical_entity_id!r} has no entity"
            )
        if chunk_ids is not None and mention.chunk_id not in chunk_ids:
            raise ValidationFailedError(
                f"MENTIONS chunk {mention.chunk_id!r} is outside the snapshot corpus"
            )
        if not mention.mention_ids or not mention.surface_forms:
            raise ValidationFailedError("MENTIONS link needs mention ids and surface forms")
    for relation in record.relations:
        if relation.corpus_id != corpus_id:
            raise GraphConfigMismatchError(
                f"Relation {relation.id!r} corpus mismatch; refusing mixed corpora"
            )
        if (
            relation.source_entity_id not in entity_ids
            or relation.target_entity_id not in entity_ids
        ):
            raise ValidationFailedError(f"Relation {relation.id!r} has a dangling endpoint")
        expected_id = build_relation_id(
            corpus_id, relation.source_entity_id, relation.type, relation.target_entity_id
        )
        if relation.id != expected_id:
            raise ValidationFailedError(
                f"Relation {relation.id!r} does not hash its endpoints/type; refusing"
            )
        if not relation.evidence_chunk_ids:
            raise ValidationFailedError(f"Relation {relation.id!r} needs evidence chunks")
        if sorted(relation.evidence_chunk_ids) != list(relation.evidence_chunk_ids):
            raise ValidationFailedError(f"Relation {relation.id!r} evidence must be sorted")
        if len(set(relation.evidence_chunk_ids)) != len(relation.evidence_chunk_ids):
            raise ValidationFailedError(f"Relation {relation.id!r} repeats evidence chunks")
        if chunk_ids is not None:
            for cid in relation.evidence_chunk_ids:
                if cid not in chunk_ids:
                    raise ValidationFailedError(
                        f"Relation {relation.id!r} evidence chunk {cid!r} is outside the corpus"
                    )
    evidence_keys = {(e.relation_id, e.chunk_id) for e in record.evidence}
    for relation in record.relations:
        for cid in relation.evidence_chunk_ids:
            if (relation.id, cid) not in evidence_keys:
                raise ValidationFailedError(
                    f"Relation {relation.id!r} evidence chunk {cid!r} has no evidence row"
                )
    for row in record.evidence:
        parent = next((r for r in record.relations if r.id == row.relation_id), None)
        if parent is None:
            raise ValidationFailedError(
                f"Evidence row {row.relation_id!r}/{row.chunk_id!r} has no parent relation"
            )
        if row.chunk_id not in parent.evidence_chunk_ids:
            raise ValidationFailedError("Evidence row chunk is not listed on its relation")
        if (
            row.source_entity_id != parent.source_entity_id
            or row.target_entity_id != parent.target_entity_id
        ):
            raise ValidationFailedError("Evidence row endpoints do not match its relation")
    return record


def compute_coverage(
    chunks: Sequence[ChunkRecord | dict[str, Any]],
    extractions: Sequence[ChunkExtraction | dict[str, Any]] | None = None,
    failures: Sequence[ExtractionFailure | dict[str, Any]] | None = None,
) -> CoverageReport:
    """Report extraction completeness honestly (counts, never claims)."""
    _, chunk_by_id = _coerce_chunks(chunks)
    corpus_id = next(iter(chunk_by_id.values()))["corpus_id"]
    parsed_extractions = _coerce_extractions(extractions or [])
    parsed_failures = [
        item if isinstance(item, ExtractionFailure) else ExtractionFailure.model_validate(item)
        for item in (failures or [])
    ]
    extracted_ids = sorted(item.chunk_id for item in parsed_extractions)
    failed_ids = sorted(item.chunk_id for item in parsed_failures)
    if len(set(extracted_ids)) != len(extracted_ids):
        raise InvalidInputError("Duplicate extraction chunk ids")
    if len(set(failed_ids)) != len(failed_ids):
        raise InvalidInputError("Duplicate failure chunk ids")
    overlap = set(extracted_ids).intersection(failed_ids)
    if overlap:
        raise InvalidInputError(f"Chunks both extracted and failed: {sorted(overlap)[:3]}")
    for cid in [*extracted_ids, *failed_ids]:
        if cid not in chunk_by_id:
            raise ValidationFailedError(f"Coverage chunk {cid!r} is outside the corpus")
    chunk_ids = sorted(chunk_by_id)
    missing = sorted(set(chunk_ids) - set(extracted_ids) - set(failed_ids))
    n_chunks = len(chunk_ids)
    ratio = (len(extracted_ids) / n_chunks) if n_chunks else 0.0
    return CoverageReport(
        corpus_id=corpus_id,
        n_chunks=n_chunks,
        n_extracted=len(extracted_ids),
        n_failed=len(failed_ids),
        n_unprocessed=len(missing),
        extracted_chunk_ids=extracted_ids,
        failed_chunk_ids=failed_ids,
        missing_chunk_ids=missing,
        coverage_ratio=round(ratio, 4),
        is_complete=(not failed_ids and not missing),
        is_partial=False,
    )


def activate_graph_snapshot(
    coverage: CoverageReport | dict[str, Any],
    *,
    allow_partial: bool = False,
) -> CoverageReport:
    """Gate snapshot activation; incomplete graphs need explicit partial opt-in.

    A complete snapshot activates with ``is_partial=False``. An incomplete
    snapshot raises INDEX_NOT_READY unless ``allow_partial`` is True, in which
    case it returns a copy with ``is_partial=True`` so every downstream result
    must carry the partial-index flag and coverage statistics. Partial graphs
    are never silently labeled complete.
    """
    record = (
        coverage
        if isinstance(coverage, CoverageReport)
        else CoverageReport.model_validate(coverage)
    )
    if not isinstance(allow_partial, bool):
        raise InvalidInputError("allow_partial must be a boolean")
    if record.is_complete:
        if record.is_partial:
            raise InvalidInputError("Complete coverage cannot carry a partial flag")
        return record
    if not allow_partial:
        raise GraphNotReadyError(
            f"Graph snapshot is incomplete: {record.n_extracted}/{record.n_chunks} extracted, "
            f"{record.n_failed} failed, {record.n_unprocessed} unprocessed "
            f"(missing={record.missing_chunk_ids[:3]}); refusing silent activation"
        )
    return record.model_copy(update={"is_partial": True})


def summarize_graph_snapshot(snapshot: GraphSnapshot | dict[str, Any]) -> dict[str, Any]:
    """Count-level audit summary; never a precision claim."""
    record = (
        snapshot if isinstance(snapshot, GraphSnapshot) else GraphSnapshot.model_validate(snapshot)
    )
    entity_types: dict[str, int] = {}
    for entity in record.entities:
        t = entity.get("type") if isinstance(entity, dict) else getattr(entity, "type", "?")
        key = t.value if isinstance(t, EntityType) else str(t)
        entity_types[key] = entity_types.get(key, 0) + 1
    relation_types: dict[str, int] = {}
    for relation in record.relations:
        key = relation.type.value
        relation_types[key] = relation_types.get(key, 0) + 1
    return {
        "corpus_id": record.corpus_id,
        "fingerprint": record.fingerprint,
        "resolver_version": record.resolver_version,
        "extraction_version": record.extraction_version,
        "schema_version": record.schema_version,
        "snapshot_version": record.snapshot_version,
        "n_entities": len(record.entities),
        "n_mention_edges": len(record.mentions),
        "n_relations": len(record.relations),
        "n_evidence_pairs": len(record.evidence),
        "entities_by_type": dict(sorted(entity_types.items())),
        "relations_by_type": dict(sorted(relation_types.items())),
        "stats": dict(record.stats),
    }


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )
    tmp.write_bytes(payload)
    tmp.rename(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(
        (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    )
    tmp.rename(path)


def _guarded_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    candidate = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )
    if path.is_file() and path.read_bytes() != candidate:
        raise ValueError(f"Existing artifact {path.name} differs; choose a new output directory")
    if not path.is_file():
        _write_jsonl_atomic(path, rows)


def _guarded_json(path: Path, payload: dict[str, Any]) -> None:
    candidate = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    if path.is_file() and path.read_bytes() != candidate:
        raise ValueError(f"Existing artifact {path.name} differs; choose a new output directory")
    if not path.is_file():
        _write_json_atomic(path, payload)


def write_graph_artifacts(
    output_dir: str | Path,
    snapshot: GraphSnapshot | dict[str, Any],
    coverage: CoverageReport | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist mentions, relations, relation evidence and the graph manifest.

    Writes are atomic and guarded: existing files with differing bytes are
    refused, never silently overwritten. Rows are sorted for determinism.
    """
    record = (
        snapshot if isinstance(snapshot, GraphSnapshot) else GraphSnapshot.model_validate(snapshot)
    )
    validate_graph_snapshot(record)
    coverage_record = (
        None
        if coverage is None
        else (
            coverage
            if isinstance(coverage, CoverageReport)
            else CoverageReport.model_validate(coverage)
        )
    )
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    mention_rows = [
        m.model_dump(mode="json")
        for m in sorted(record.mentions, key=lambda x: (x.chunk_id, x.canonical_entity_id))
    ]
    relation_rows = [
        r.model_dump(mode="json") for r in sorted(record.relations, key=lambda x: x.id)
    ]
    evidence_rows = [
        e.model_dump(mode="json")
        for e in sorted(record.evidence, key=lambda x: (x.relation_id, x.chunk_id))
    ]
    entity_rows = sorted(
        (e if isinstance(e, dict) else dict(e) for e in record.entities),
        key=lambda e: str(e.get("id")),
    )
    mentions_path = dest / "mentions.jsonl"
    relations_path = dest / "relations.jsonl"
    evidence_path = dest / "relation_evidence.jsonl"
    entities_path = dest / "entities.jsonl"
    manifest_path = dest / "manifest.json"
    _guarded_rows(mentions_path, mention_rows)
    _guarded_rows(relations_path, relation_rows)
    _guarded_rows(evidence_path, evidence_rows)
    _guarded_rows(entities_path, entity_rows)
    manifest: dict[str, Any] = {
        "manifest_version": GRAPH_MANIFEST_VERSION,
        "snapshot_version": record.snapshot_version,
        "corpus_id": record.corpus_id,
        "fingerprint": record.fingerprint,
        "resolver_version": record.resolver_version,
        "extraction_version": record.extraction_version,
        "schema_version": record.schema_version,
        "n_entities": len(entity_rows),
        "n_mention_edges": len(mention_rows),
        "n_relations": len(relation_rows),
        "n_evidence_pairs": len(evidence_rows),
        "stats": dict(record.stats),
        "summary": summarize_graph_snapshot(record),
    }
    if coverage_record is not None:
        manifest["coverage"] = coverage_record.model_dump(mode="json")
    manifest_with_hashes = dict(manifest)
    manifest_with_hashes["artifact_sha256"] = {
        "entities.jsonl": hashlib.sha256(entities_path.read_bytes()).hexdigest(),
        "mentions.jsonl": hashlib.sha256(mentions_path.read_bytes()).hexdigest(),
        "relations.jsonl": hashlib.sha256(relations_path.read_bytes()).hexdigest(),
        "relation_evidence.jsonl": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
    }
    _guarded_json(manifest_path, manifest_with_hashes)
    manifest_with_hashes["artifact_sha256"]["manifest.json"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    return {
        "output_dir": str(dest),
        "entities_jsonl": str(entities_path),
        "mentions_jsonl": str(mentions_path),
        "relations_jsonl": str(relations_path),
        "relation_evidence_jsonl": str(evidence_path),
        "manifest_json": str(manifest_path),
        "artifact_sha256": dict(manifest_with_hashes["artifact_sha256"]),
    }


__all__ = [
    "GRAPH_SNAPSHOT_VERSION",
    "CoverageReport",
    "GraphConfigMismatchError",
    "GraphNotReadyError",
    "GraphRelation",
    "GraphSnapshot",
    "MentionEdge",
    "RelationEvidence",
    "activate_graph_snapshot",
    "build_graph_fingerprint",
    "build_graph_snapshot",
    "build_relation_id",
    "compute_coverage",
    "summarize_graph_snapshot",
    "validate_graph_snapshot",
    "write_graph_artifacts",
]
