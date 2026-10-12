"""Bounded hybrid retrieval on local Neo4j (T10, R07).

Requires a healthy local Neo4j at bolt://127.0.0.1:7687 with
GRAPHRAG_NEO4J_PASSWORD set (or present in the ignored repo `.env`).
No embedding or LLM downloads are needed: vectors are synthetic 384-dim
one-hot surrogates (same dimension as the pinned MiniLM index) and the
bridge/hub fixtures are fully offline. The suite wipes Chunk/Entity/
IndexMeta/GraphMeta data in the single-community-database workspace; it
keeps index definitions.
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
from graphrag_lab.extraction import EXTRACTION_VERSION, SCHEMA_VERSION
from graphrag_lab.graph_index import build_graph_snapshot
from graphrag_lab.hybrid import HybridConfig, hybrid_retrieve_with_vector, reciprocal_rank_fusion
from graphrag_lab.linking import LinkingConfig, link_question_entities
from graphrag_lab.neo4j_store import (
    ConfigMismatchError,
    count_graph,
    create_driver,
    delete_graph_data,
    delete_vector_data,
    ensure_chunk_constraints,
    ensure_graph_constraints,
    ensure_graph_meta,
    ensure_index_meta,
    ensure_vector_index,
    ingest_chunks,
    ingest_graph_snapshot,
)
from graphrag_lab.resolution import RESOLVER_VERSION, ResolverConfig, resolve_entities
from graphrag_lab.retrieval import RetrievalConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
BRIDGE_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "bridge.json"

DIM = 384
EMBEDDING_MODEL = "test-hybrid-model"
BRIDGE_CORPUS = "hp_t09_bridge_v1"
HUB_CORPUS = "hp_t10_hub_v1"


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


def _one_hot(index: int, dim: int = DIM) -> list[float]:
    vec = [0.0] * dim
    vec[index % dim] = 1.0
    return vec


def _load_bridge() -> dict:
    return json.loads(BRIDGE_FIXTURE.read_text(encoding="utf-8"))


def _bridge_chunks(bridge: dict | None = None) -> list[ChunkRecord]:
    bridge = bridge or _load_bridge()
    assert bridge["corpus_id"] == BRIDGE_CORPUS
    paragraphs = {p["id"]: p for p in bridge["paragraphs"]}
    chunks: list[ChunkRecord] = []
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


def _bridge_extractions(bridge: dict | None = None) -> list[dict]:
    bridge = bridge or _load_bridge()
    return [dict(item) for item in bridge["extractions"]]


def _bridge_snapshot(bridge: dict | None = None):
    bridge = bridge or _load_bridge()
    chunks = _bridge_chunks(bridge)
    extractions = _bridge_extractions(bridge)
    resolution = resolve_entities(
        [{"chunk_id": e["chunk_id"], "entities": e["entities"]} for e in extractions],
        ResolverConfig(),
        corpus_id=bridge["corpus_id"],
    )
    snapshot = build_graph_snapshot(chunks, extractions, resolution)
    return chunks, extractions, resolution, snapshot


def _setup_bridge(driver, database):
    bridge = _load_bridge()
    chunks, extractions, _, snapshot = _bridge_snapshot(bridge)
    ensure_chunk_constraints(driver, database)
    ensure_graph_constraints(driver, database)
    ensure_vector_index(driver, database, DIM)
    ensure_index_meta(
        driver,
        database,
        corpus_id=bridge["corpus_id"],
        embedding_model=EMBEDDING_MODEL,
        dimension=DIM,
        chunker_version=CHUNKER_VERSION,
    )
    ordered = sorted(chunks, key=lambda c: c.id)
    # c_t09_author < c_t09_book: author gets index 1, book index 0.
    vectors = [_one_hot(1), _one_hot(0)]
    ingest_chunks(driver, database, ordered, vectors, embedding_model=EMBEDDING_MODEL)
    ensure_graph_meta(
        driver,
        database,
        corpus_id=bridge["corpus_id"],
        resolver_version=RESOLVER_VERSION,
        extraction_version=EXTRACTION_VERSION,
        schema_version=SCHEMA_VERSION,
    )
    ingest_graph_snapshot(driver, database, ordered, snapshot)
    return bridge, ordered, snapshot


def _hub_chunks(n_leaves: int = 30) -> tuple[list[ChunkRecord], list[dict]]:
    chunks: list[ChunkRecord] = []
    extractions: list[dict] = []

    def _record(chunk_id: str, paragraph_id: str, title: str, text: str) -> ChunkRecord:
        return ChunkRecord(
            id=chunk_id,
            corpus_id=HUB_CORPUS,
            paragraph_id=paragraph_id,
            title=title,
            text=text,
            sentence_ids=[0],
            sentence_texts=[text],
            token_count=count_chunk_tokens(title, text),
            content_hash=build_content_hash(title, text),
            chunker_version=CHUNKER_VERSION,
        )

    hub_text = "Hubville is a city."
    chunks.append(_record("c_hub", "p_hub", "Hubville", hub_text))
    extractions.append(
        {
            "schema_version": "1",
            "chunk_id": "c_hub",
            "entities": [
                {
                    "local_id": "e1",
                    "name": "Hubville",
                    "type": "LOCATION",
                    "sentence_ids": [0],
                }
            ],
            "relations": [],
        }
    )
    for i in range(n_leaves):
        leaf = f"Leafville{i}"
        chunk_id = f"c_leaf{i:02d}"
        text = f"{leaf} is located in Hubville."
        chunks.append(_record(chunk_id, f"p_leaf{i:02d}", leaf, text))
        extractions.append(
            {
                "schema_version": "1",
                "chunk_id": chunk_id,
                "entities": [
                    {
                        "local_id": "e1",
                        "name": leaf,
                        "type": "LOCATION",
                        "sentence_ids": [0],
                    },
                    {
                        "local_id": "e2",
                        "name": "Hubville",
                        "type": "LOCATION",
                        "sentence_ids": [0],
                    },
                ],
                "relations": [
                    {
                        "source": "e1",
                        "type": "LOCATED_IN",
                        "target": "e2",
                        "evidence": [{"sentence_id": 0, "quote": text}],
                    }
                ],
            }
        )
    return sorted(chunks, key=lambda c: c.id), sorted(extractions, key=lambda e: e["chunk_id"])


def _setup_hub(driver, database, n_leaves: int = 30):
    chunks, extractions = _hub_chunks(n_leaves)
    resolution = resolve_entities(
        [{"chunk_id": e["chunk_id"], "entities": e["entities"]} for e in extractions],
        ResolverConfig(),
        corpus_id=HUB_CORPUS,
    )
    snapshot = build_graph_snapshot(chunks, extractions, resolution)
    ensure_chunk_constraints(driver, database)
    ensure_graph_constraints(driver, database)
    ensure_vector_index(driver, database, DIM)
    ensure_index_meta(
        driver,
        database,
        corpus_id=HUB_CORPUS,
        embedding_model=EMBEDDING_MODEL,
        dimension=DIM,
        chunker_version=CHUNKER_VERSION,
    )
    vectors = [_one_hot(i, DIM) for i in range(len(chunks))]
    ingest_chunks(driver, database, chunks, vectors, embedding_model=EMBEDDING_MODEL)
    ensure_graph_meta(
        driver,
        database,
        corpus_id=HUB_CORPUS,
        resolver_version=RESOLVER_VERSION,
        extraction_version=EXTRACTION_VERSION,
        schema_version=SCHEMA_VERSION,
    )
    ingest_graph_snapshot(driver, database, chunks, snapshot)
    return chunks, extractions, resolution, snapshot


class _StepClock:
    """Deterministic clock that expires immediately (timeout fixture)."""

    def __init__(self, step: float = 10.0) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


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
def clean(store):
    driver, config = store
    database = config.neo4j.database
    delete_vector_data(driver, database)
    delete_graph_data(driver, database)
    ensure_chunk_constraints(driver, database)
    ensure_graph_constraints(driver, database)
    yield driver, database
    delete_vector_data(driver, database)
    delete_graph_data(driver, database)


def test_rrf_merges_by_rank_without_raw_scores():
    scores = reciprocal_rank_fusion(["a", "b"], ["b", "c"], k=60)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)
    # Deterministic tie-break is by chunk id in the caller; equal scores stay equal.
    tied = reciprocal_rank_fusion(["x"], ["y"], k=60)
    assert tied["x"] == tied["y"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(["a", "a"], ["b"])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion("ab", ["b"])  # type: ignore[arg-type]


def test_bridge_merges_vector_and_graph(clean):
    driver, database = clean
    bridge, ordered, snapshot = _setup_bridge(driver, database)
    by_name = {}
    for entity in snapshot.entities:
        item = dict(entity) if not isinstance(entity, dict) else entity
        by_name[item["canonical_name"]] = item["id"]

    question = "Who wrote The Night Cartographer and where was Elena Voss born?"
    # Query close to the book chunk: vector alone returns only the book.
    query = [0.9] + [0.1] + [0.0] * (DIM - 2)
    hcfg = HybridConfig(num_vector_candidates=1)
    before = count_graph(driver, database)

    result = hybrid_retrieve_with_vector(
        question=question,
        query_vector=query,
        corpus_id=bridge["corpus_id"],
        embedding_model=EMBEDDING_MODEL,
        dimension=DIM,
        driver=driver,
        database=database,
        vector_config=RetrievalConfig(num_candidates=1),
        hybrid_config=hcfg,
    )
    # Query entities link without creating nodes.
    assert result.fallback_to_vector is False
    assert result.fallback_reason is None
    assert len(result.linked_entity_ids) == 2
    assert by_name["Elena Voss"] in result.linked_entity_ids
    assert count_graph(driver, database) == before

    # Vector alone sees one chunk; the graph adds the bridge evidence.
    assert result.total_vector == 1
    assert result.total_graph >= 2
    assert result.total_merged == 2
    ids = [c.chunk_id for c in result.chunks]
    assert "c_t09_book" in ids and "c_t09_author" in ids
    by_id = {c.chunk_id: c for c in result.chunks}
    assert by_id["c_t09_book"].origin == "both"
    assert by_id["c_t09_author"].origin == "graph"
    assert by_id["c_t09_author"].distance in (1, 2)
    assert by_id["c_t09_author"].path_ids
    assert by_id["c_t09_author"].evidence_kind in ("direct", "both_evidence_mention")
    # Two-hop bridge reaches Lisbon through Elena Voss.
    assert by_name["Lisbon"] in set(result.visited_entities)
    assert result.cap_hits == []
    assert result.index_fingerprint and result.graph_fingerprint
    assert result.graph_fingerprint == snapshot.fingerprint
    # Fusion uses ranks only: book (vector 1 + graph) outranks graph-only author.
    assert by_id["c_t09_book"].fusion_score > by_id["c_t09_author"].fusion_score
    assert by_id["c_t09_book"].vector_rank == 1
    assert by_id["c_t09_book"].graph_rank is not None
    assert [c.rank for c in result.chunks] == [1, 2]

    # Deterministic repeat.
    again = hybrid_retrieve_with_vector(
        question=question,
        query_vector=query,
        corpus_id=bridge["corpus_id"],
        embedding_model=EMBEDDING_MODEL,
        dimension=DIM,
        driver=driver,
        database=database,
        vector_config=RetrievalConfig(num_candidates=1),
        hybrid_config=hcfg,
    )
    assert [c.chunk_id for c in again.chunks] == ids
    assert again.linked_entity_ids == result.linked_entity_ids


def test_no_link_falls_back_to_vector(clean):
    driver, database = clean
    bridge, _, _ = _setup_bridge(driver, database)
    query = [0.9] + [0.1] + [0.0] * (DIM - 2)
    result = hybrid_retrieve_with_vector(
        question="zzzz qqqq wwww unrelated gibberish",
        query_vector=query,
        corpus_id=bridge["corpus_id"],
        embedding_model=EMBEDDING_MODEL,
        dimension=DIM,
        driver=driver,
        database=database,
        vector_config=RetrievalConfig(num_candidates=5),
        hybrid_config=HybridConfig(num_vector_candidates=5),
    )
    assert result.fallback_to_vector is True
    assert result.fallback_reason == "no_linked_entities"
    assert result.linked_entity_ids == []
    assert result.total_graph == 0
    assert result.total_vector >= 1
    assert all(c.origin == "vector" for c in result.chunks)
    assert result.diagnostics["expansion"]["status"] == "skipped"


def test_linking_creates_no_nodes(clean):
    driver, database = clean
    bridge, _, _ = _setup_bridge(driver, database)
    before = count_graph(driver, database)
    linking = link_question_entities(
        "Who wrote The Night Cartographer?",
        driver,
        database,
        bridge["corpus_id"],
        LinkingConfig(),
    )
    assert {e.entity_id for e in linking.linked}
    assert linking.diagnostics["creates_nodes"] is False
    assert count_graph(driver, database) == before


def test_high_degree_respects_frontier_budgets(clean):
    driver, database = clean
    _, _, _, snapshot = _setup_hub(driver, database, n_leaves=30)
    assert len(snapshot.relations) == 30
    hub_id = next(
        (dict(e) if not isinstance(e, dict) else e)["id"]
        for e in snapshot.entities
        if (dict(e) if not isinstance(e, dict) else e)["canonical_name"] == "Hubville"
    )
    query = _one_hot(0)
    # Hub chunk sorts first (c_hub < c_leaf..); find its vector index for a hub query.
    result = hybrid_retrieve_with_vector(
        question="Tell me about Hubville",
        query_vector=query,
        corpus_id=HUB_CORPUS,
        embedding_model=EMBEDDING_MODEL,
        dimension=DIM,
        driver=driver,
        database=database,
        vector_config=RetrievalConfig(num_candidates=30),
        hybrid_config=HybridConfig(),
    )
    assert result.fallback_to_vector is False
    assert hub_id in result.linked_entity_ids
    assert any(h.startswith("max_edges_per_node_hit:") for h in result.cap_hits)
    assert len(result.visited_entities) <= 100
    assert result.total_graph <= 40
    assert len({c.chunk_id for c in result.chunks}) == len(result.chunks)
    assert result.total_merged <= 60

    capped = hybrid_retrieve_with_vector(
        question="Tell me about Hubville",
        query_vector=query,
        corpus_id=HUB_CORPUS,
        embedding_model=EMBEDDING_MODEL,
        dimension=DIM,
        driver=driver,
        database=database,
        vector_config=RetrievalConfig(num_candidates=30),
        hybrid_config=HybridConfig(max_merged=5),
    )
    assert capped.total_merged == 5
    assert "max_merged_hit" in capped.cap_hits


def test_timeout_falls_back_with_flag(clean):
    driver, database = clean
    bridge, _, _ = _setup_bridge(driver, database)
    query = [0.9] + [0.1] + [0.0] * (DIM - 2)
    result = hybrid_retrieve_with_vector(
        question="Who wrote The Night Cartographer and where was Elena Voss born?",
        query_vector=query,
        corpus_id=bridge["corpus_id"],
        embedding_model=EMBEDDING_MODEL,
        dimension=DIM,
        driver=driver,
        database=database,
        vector_config=RetrievalConfig(num_candidates=5),
        hybrid_config=HybridConfig(num_vector_candidates=5, graph_timeout_s=2.0),
        clock=_StepClock(step=10.0),
    )
    assert result.fallback_to_vector is True
    assert result.fallback_reason == "graph_timeout"
    assert "graph_timeout_hit" in result.cap_hits
    assert result.total_graph == 0
    assert all(c.origin == "vector" for c in result.chunks)


def test_corpus_mismatch_raises_without_partial_writes(clean):
    driver, database = clean
    bridge, _, _ = _setup_bridge(driver, database)
    before = count_graph(driver, database)
    query = _one_hot(0)
    with pytest.raises(ConfigMismatchError):
        hybrid_retrieve_with_vector(
            question="Who wrote The Night Cartographer?",
            query_vector=query,
            corpus_id="hp_t10_foreign",
            embedding_model=EMBEDDING_MODEL,
            dimension=DIM,
            driver=driver,
            database=database,
            vector_config=RetrievalConfig(num_candidates=5),
            hybrid_config=HybridConfig(num_vector_candidates=5),
        )
    assert count_graph(driver, database) == before
