"""Evidence-backed graph snapshot on local Neo4j (T09, R06).

Requires a healthy local Neo4j at bolt://127.0.0.1:7687 with
GRAPHRAG_NEO4J_PASSWORD set (or present in the ignored repo `.env`).
No embedding or LLM downloads are needed: the bridge fixture is fully
offline. The suite wipes Chunk/Entity/IndexMeta/GraphMeta data in the
single-community-database workspace; it keeps index definitions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from graphrag_lab.chunking import (
    CHUNKER_VERSION,
    ChunkRecord,
    build_content_hash,
    count_chunk_tokens,
)
from graphrag_lab.config import load_config
from graphrag_lab.extraction import EXTRACTION_VERSION, SCHEMA_VERSION, ExtractionFailure
from graphrag_lab.graph_index import (
    CoverageReport,
    GraphNotReadyError,
    activate_graph_snapshot,
    build_graph_snapshot,
    build_relation_id,
    compute_coverage,
    summarize_graph_snapshot,
    validate_graph_snapshot,
    write_graph_artifacts,
)
from graphrag_lab.llm import ValidationFailedError
from graphrag_lab.neo4j_store import (
    ConfigMismatchError,
    count_graph,
    create_driver,
    delete_graph_data,
    delete_vector_data,
    ensure_chunk_constraints,
    ensure_graph_constraints,
    ensure_graph_meta,
    fetch_mentions_for_chunk,
    fetch_neighbors,
    fetch_relation,
    ingest_graph_snapshot,
    read_graph_meta,
)
from graphrag_lab.resolution import RESOLVER_VERSION, ResolverConfig, resolve_entities

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
BRIDGE_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "bridge.json"


def _load_password() -> str | None:
    password = os.environ.get("GRAPHRAG_NEO4J_PASSWORD")
    if password and password.strip():
        return password.strip()
    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GRAPHRAG_NEO4J_PASSWORD="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    os.environ["GRAPHRAG_NEO4J_PASSWORD"] = value
                    return value
    return None


def _load_bridge() -> dict:
    return json.loads(BRIDGE_FIXTURE.read_text(encoding="utf-8"))


def _build_chunks(bridge: dict | None = None) -> list[ChunkRecord]:
    bridge = bridge or _load_bridge()
    chunks: list[ChunkRecord] = []
    paragraphs = {p["id"]: p for p in bridge["paragraphs"]}
    for paragraph_id, chunk_id in bridge["chunk_ids"].items():
        paragraph = paragraphs[paragraph_id]
        text = " ".join(paragraph["sentences"])
        chunks.append(
            ChunkRecord(
                id=chunk_id,
                corpus_id=bridge["corpus_id"],
                paragraph_id=paragraph_id,
                title=paragraph["title"],
                text=text,
                sentence_ids=list(range(len(paragraph["sentences"]))),
                sentence_texts=list(paragraph["sentences"]),
                token_count=count_chunk_tokens(paragraph["title"], text),
                content_hash=build_content_hash(paragraph["title"], text),
                chunker_version=CHUNKER_VERSION,
            )
        )
    return sorted(chunks, key=lambda c: c.id)


def _build_extractions(bridge: dict | None = None) -> list[dict]:
    bridge = bridge or _load_bridge()
    return [dict(item) for item in bridge["extractions"]]


def _build_snapshot(bridge: dict | None = None):
    bridge = bridge or _load_bridge()
    chunks = _build_chunks(bridge)
    extractions = _build_extractions(bridge)
    resolution = resolve_entities(
        [{"chunk_id": e["chunk_id"], "entities": e["entities"]} for e in extractions],
        ResolverConfig(),
        corpus_id=bridge["corpus_id"],
    )
    snapshot = build_graph_snapshot(chunks, extractions, resolution)
    return chunks, extractions, resolution, snapshot


@pytest.fixture(scope="module")
def store():
    password = _load_password()
    if not password:
        pytest.skip("GRAPHRAG_NEO4J_PASSWORD is not set (and no repo .env entry)")
    config = load_config(str(DEFAULT_CONFIG))
    driver = create_driver(config)
    try:
        with driver.session(database=config.neo4j.database) as session:
            session.run("RETURN 1 AS healthy").single()
    except Exception as exc:
        driver.close()
        pytest.skip(f"local Neo4j is unavailable: {exc}")
    yield driver, config
    driver.close()


@pytest.fixture()
def clean_graph(store):
    driver, config = store
    database = config.neo4j.database
    delete_vector_data(driver, database)
    delete_graph_data(driver, database)
    ensure_chunk_constraints(driver, database)
    ensure_graph_constraints(driver, database)
    bridge = _load_bridge()
    ensure_graph_meta(
        driver,
        database,
        corpus_id=bridge["corpus_id"],
        resolver_version=RESOLVER_VERSION,
        extraction_version=EXTRACTION_VERSION,
        schema_version=SCHEMA_VERSION,
    )
    yield driver, database
    delete_vector_data(driver, database)
    delete_graph_data(driver, database)


def test_bridge_fixture_matches_expected_snapshot():
    bridge = _load_bridge()
    assert bridge["corpus_id"]
    assert len(bridge["paragraphs"]) == 2
    assert len(bridge["extractions"]) == 2
    chunks, extractions, resolution, snapshot = _build_snapshot(bridge)
    assert snapshot.corpus_id == bridge["corpus_id"]
    assert snapshot.resolver_version == RESOLVER_VERSION
    assert snapshot.extraction_version == EXTRACTION_VERSION
    assert snapshot.fingerprint
    assert len(snapshot.entities) == bridge["expected"]["n_entities"] == 3
    assert len(snapshot.mentions) == bridge["expected"]["n_mention_edges"] == 4
    assert len(snapshot.relations) == bridge["expected"]["n_relations"] == 2
    assert sorted(r.type.value for r in snapshot.relations) == sorted(
        bridge["expected"]["relation_types"]
    )
    names = set()
    for entity in snapshot.entities:
        item = dict(entity) if not isinstance(entity, dict) else entity
        names.add(item["canonical_name"])
    assert set(bridge["expected"]["bridge"]) <= names
    # The bridge person merges across both chunks.
    by_mention = {a.mention_id: a.canonical_entity_id for a in resolution.assignments}
    assert (
        by_mention[bridge["expected"]["merged_mentions"][0]]
        == by_mention[bridge["expected"]["merged_mentions"][1]]
    )
    validate_graph_snapshot(snapshot, chunks)
    summary = summarize_graph_snapshot(snapshot)
    assert summary["n_relations"] == 2
    assert summary["relations_by_type"] == {"BORN_IN": 1, "WRITTEN_BY": 1}


def test_relation_ids_are_deterministic_and_corpus_scoped():
    first = build_relation_id("hp_c", "ent_a", "WRITTEN_BY", "ent_b")
    second = build_relation_id("hp_c", "ent_a", "WRITTEN_BY", "ent_b")
    assert first == second and first.startswith("rel_")
    other_corpus = build_relation_id("hp_other", "ent_a", "WRITTEN_BY", "ent_b")
    assert other_corpus != first
    flipped = build_relation_id("hp_c", "ent_b", "WRITTEN_BY", "ent_a")
    assert flipped != first


def test_invalid_evidence_is_rejected():
    chunks, extractions, resolution, _ = _build_snapshot()
    # Non-verbatim quote.
    bad = json.loads(json.dumps(extractions))
    bad[0]["relations"][0]["evidence"][0]["quote"] = "A fabricated claim."
    with pytest.raises(ValidationFailedError):
        build_graph_snapshot(chunks, bad, resolution)
    # Sentence outside the chunk.
    bad_sentence = json.loads(json.dumps(extractions))
    bad_sentence[0]["relations"][0]["evidence"][0]["sentence_id"] = 99
    with pytest.raises(ValidationFailedError):
        build_graph_snapshot(chunks, bad_sentence, resolution)
    # Extraction chunk outside the corpus.
    foreign = json.loads(json.dumps(extractions))
    foreign.append(
        {
            "schema_version": "1",
            "chunk_id": "c_foreign",
            "entities": [],
            "relations": [],
        }
    )
    with pytest.raises(ValidationFailedError):
        build_graph_snapshot(chunks, foreign, resolution)


def test_incomplete_graph_cannot_activate_silently():
    chunks, extractions, _, _ = _build_snapshot()
    full = compute_coverage(chunks, extractions, [])
    assert full.is_complete and full.coverage_ratio == 1.0
    assert activate_graph_snapshot(full).is_partial is False

    partial_extractions = extractions[:1]
    missing = compute_coverage(chunks, partial_extractions, [])
    assert not missing.is_complete
    assert missing.n_unprocessed == 1
    assert (
        missing.missing_chunk_ids == ["c_t09_book"]
        or missing.missing_chunk_ids == ["c_t09_author"]
        or len(missing.missing_chunk_ids) == 1
    )
    with pytest.raises(GraphNotReadyError) as excinfo:
        activate_graph_snapshot(missing)
    assert excinfo.value.code == "INDEX_NOT_READY"
    allowed = activate_graph_snapshot(missing, allow_partial=True)
    assert allowed.is_partial is True
    assert allowed.n_extracted == 1

    failure = ExtractionFailure(
        chunk_id="c_t09_author",
        error_code="PROVIDER_ERROR",
        error_message="boom",
        attempts=3,
        model="fake",
        content_hash="hash",
        schema_version="1",
        prompt_version="extraction-prompt-v1",
    )
    failed = compute_coverage(chunks, partial_extractions, [failure])
    assert not failed.is_complete and failed.n_failed == 1
    with pytest.raises(GraphNotReadyError):
        activate_graph_snapshot(failed)


def test_load_creates_entities_mentions_and_directed_relations(clean_graph):
    driver, database = clean_graph
    bridge = _load_bridge()
    chunks, _, _, snapshot = _build_snapshot(bridge)
    assert count_graph(driver, database) == {"entities": 0, "relations": 0, "mentions": 0}

    result = ingest_graph_snapshot(driver, database, chunks, snapshot)
    assert result == {"n_chunks": 2, "n_entities": 3, "n_mentions": 4, "n_relations": 2}
    counts = count_graph(driver, database, bridge["corpus_id"])
    assert counts["entities"] == 3
    assert counts["relations"] == 2
    assert counts["mentions"] == 4

    meta = read_graph_meta(driver, database)
    assert meta is not None and meta["corpus_id"] == bridge["corpus_id"]
    assert meta["fingerprint"] == snapshot.fingerprint

    # Directed relations keep their semantic type and evidence chunks.
    for relation in snapshot.relations:
        stored = fetch_relation(driver, database, relation.id)
        assert stored is not None
        assert stored["source_entity_id"] == relation.source_entity_id
        assert stored["target_entity_id"] == relation.target_entity_id
        assert stored["type"] == relation.type.value
        assert sorted(stored["evidence_chunk_ids"]) == list(relation.evidence_chunk_ids)
        assert stored["corpus_id"] == bridge["corpus_id"]

    # MENTIONS links carry mention evidence per chunk.
    for chunk in chunks:
        rows = fetch_mentions_for_chunk(driver, database, chunk.id)
        assert len(rows) == 2
        for row in rows:
            assert row["mention_ids"] and row["surface_forms"]
            assert isinstance(row["sentence_ids"], list)


def test_repeated_load_creates_no_duplicates(clean_graph):
    driver, database = clean_graph
    chunks, _, _, snapshot = _build_snapshot()
    ingest_graph_snapshot(driver, database, chunks, snapshot)
    before = count_graph(driver, database)
    evidence_before = {
        relation.id: fetch_relation(driver, database, relation.id)["evidence_chunk_ids"]
        for relation in snapshot.relations
    }
    ingest_graph_snapshot(driver, database, chunks, snapshot)
    after = count_graph(driver, database)
    assert after == before == {"entities": 3, "relations": 2, "mentions": 4}
    for relation in snapshot.relations:
        stored = fetch_relation(driver, database, relation.id)
        assert sorted(stored["evidence_chunk_ids"]) == sorted(evidence_before[relation.id])


def test_evidence_union_is_deterministic(clean_graph):
    driver, database = clean_graph
    chunks, _, _, snapshot = _build_snapshot()
    ingest_graph_snapshot(driver, database, chunks, snapshot)
    target = sorted(snapshot.relations, key=lambda r: r.id)[0]
    template = next(e for e in snapshot.evidence if e.relation_id == target.id)
    quote = template.evidence[0].quote
    extra_chunk = {
        "id": "c_t09_note",
        "corpus_id": snapshot.corpus_id,
        "paragraph_id": "p_t09_note",
        "title": "Marginal note",
        "text": quote,
        "sentence_ids": [0],
        "sentence_texts": [quote],
        "content_hash": "hash-note",
        "chunker_version": CHUNKER_VERSION,
    }
    extended_chunks = [c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in chunks] + [
        extra_chunk
    ]
    extended = snapshot.model_copy(deep=True)
    updated_relations = []
    for relation in extended.relations:
        if relation.id == target.id:
            updated_relations.append(
                relation.model_copy(
                    update={
                        "evidence_chunk_ids": sorted([*relation.evidence_chunk_ids, "c_t09_note"])
                    }
                )
            )
        else:
            updated_relations.append(relation)
    extra_evidence = template.model_copy(update={"chunk_id": "c_t09_note"})
    extended = extended.model_copy(
        update={
            "relations": sorted(updated_relations, key=lambda r: r.id),
            "evidence": sorted(
                [*extended.evidence, extra_evidence], key=lambda e: (e.relation_id, e.chunk_id)
            ),
        }
    )
    ingest_graph_snapshot(driver, database, extended_chunks, extended)
    stored = fetch_relation(driver, database, target.id)
    assert sorted(stored["evidence_chunk_ids"]) == sorted(
        [*target.evidence_chunk_ids, "c_t09_note"]
    )
    # Repeating the extended load does not duplicate evidence.
    ingest_graph_snapshot(driver, database, extended_chunks, extended)
    assert count_graph(driver, database)["relations"] == 2
    assert sorted(fetch_relation(driver, database, target.id)["evidence_chunk_ids"]) == sorted(
        stored["evidence_chunk_ids"]
    )


def test_mismatched_fingerprint_fails_without_partial_writes(clean_graph):
    driver, database = clean_graph
    chunks, _, _, snapshot = _build_snapshot()
    ingest_graph_snapshot(driver, database, chunks, snapshot)
    before = count_graph(driver, database)

    foreign_corpus = "hp_t09_foreign"
    foreign_chunk_records = []
    for chunk in _build_chunks():
        payload = chunk.model_copy(update={"corpus_id": foreign_corpus})
        foreign_chunk_records.append(payload)
    foreign_extractions = _build_extractions()
    foreign_resolution = resolve_entities(
        [{"chunk_id": e["chunk_id"], "entities": e["entities"]} for e in foreign_extractions],
        ResolverConfig(),
        corpus_id=foreign_corpus,
    )
    foreign_snapshot = build_graph_snapshot(
        foreign_chunk_records, foreign_extractions, foreign_resolution
    )
    with pytest.raises(ConfigMismatchError, match="mismatch|Mixed|snapshot"):
        ingest_graph_snapshot(driver, database, foreign_chunk_records, foreign_snapshot)
    assert count_graph(driver, database) == before


def test_two_document_bridge_traversal(clean_graph):
    driver, database = clean_graph
    bridge = _load_bridge()
    chunks, _, _, snapshot = _build_snapshot(bridge)
    ingest_graph_snapshot(driver, database, chunks, snapshot)

    by_name = {}
    for entity in snapshot.entities:
        item = dict(entity) if not isinstance(entity, dict) else entity
        by_name[item["canonical_name"]] = item["id"]
    book_id = by_name["The Night Cartographer"]
    person_id = by_name["Elena Voss"]
    place_id = by_name["Lisbon"]

    neighbors = {row["neighbor"] for row in fetch_neighbors(driver, database, person_id)}
    assert book_id in neighbors or place_id in neighbors
    directions = {
        (row["direction"], row["neighbor"]) for row in fetch_neighbors(driver, database, person_id)
    }
    assert ("in", book_id) in directions
    assert ("out", place_id) in directions

    with driver.session(database=database) as session:
        record = session.run(
            "MATCH (book:Entity {id: $book})-[r1:REL]->(person:Entity {id: $person})"
            "-[r2:REL]->(place:Entity {id: $place}) "
            "RETURN r1.id AS first, r1.type AS first_type, r2.id AS second, "
            "r2.type AS second_type, r1.evidence_chunk_ids AS first_evidence, "
            "r2.evidence_chunk_ids AS second_evidence",
            {"book": book_id, "person": person_id, "place": place_id},
        ).single()
    assert record is not None
    assert record["first_type"] == "WRITTEN_BY"
    assert record["second_type"] == "BORN_IN"
    assert record["first_evidence"] == ["c_t09_book"]
    assert record["second_evidence"] == ["c_t09_author"]

    # Both documents mention the bridge entity.
    for chunk_id in ("c_t09_book", "c_t09_author"):
        rows = fetch_mentions_for_chunk(driver, database, chunk_id)
        assert person_id in {row["entity_id"] for row in rows}


def test_graph_artifacts_are_guarded_and_reconcilable(tmp_path):
    _, _, _, snapshot = _build_snapshot()
    coverage = compute_coverage(_build_chunks(), _build_extractions(), [])
    info = write_graph_artifacts(tmp_path / "snapshot", snapshot, coverage)
    for name in (
        "entities.jsonl",
        "mentions.jsonl",
        "relations.jsonl",
        "relation_evidence.jsonl",
        "manifest.json",
    ):
        assert (tmp_path / "snapshot" / name).is_file()
    assert set(info["artifact_sha256"]) == {
        "entities.jsonl",
        "mentions.jsonl",
        "relations.jsonl",
        "relation_evidence.jsonl",
        "manifest.json",
    }
    evidence_rows = [
        json.loads(line)
        for line in (tmp_path / "snapshot" / "relation_evidence.jsonl").read_text().splitlines()
    ]
    assert len(evidence_rows) == 2
    for row in evidence_rows:
        assert row["relation_id"] and row["chunk_id"] and row["evidence"]
        assert all(span["quote"] for span in row["evidence"])
    second = write_graph_artifacts(tmp_path / "snapshot", snapshot, coverage)
    assert second["artifact_sha256"] == info["artifact_sha256"]
    manifest = json.loads((tmp_path / "snapshot" / "manifest.json").read_text())
    assert manifest["corpus_id"] == snapshot.corpus_id
    assert manifest["coverage"]["is_complete"] is True
    assert isinstance(CoverageReport, type)
