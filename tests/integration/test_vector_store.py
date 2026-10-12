"""Vector index and retrieval on local Neo4j (T04).

Requires a healthy local Neo4j at bolt://127.0.0.1:7687 with
GRAPHRAG_NEO4J_PASSWORD set (or present in the ignored repo `.env`), plus
the pinned embedding model. The suite wipes only Chunk/IndexMeta data in the
single-community-database workspace; it keeps the vector index definition.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from graphrag_lab.chunking import CHUNKER_VERSION, ChunkRecord
from graphrag_lab.config import load_config
from graphrag_lab.embeddings import (
    SentenceTransformerBackend,
    build_embedding_text,
    ensure_normalized,
    validate_chunk_fits_model,
    validate_embedding,
)
from graphrag_lab.neo4j_store import (
    ConfigMismatchError,
    IndexNotReadyError,
    count_chunks,
    create_driver,
    delete_vector_data,
    ensure_chunk_constraints,
    ensure_index_meta,
    ensure_vector_index,
    fetch_chunk_embedding,
    get_vector_index_dimension,
    get_vector_index_state,
    ingest_chunks,
    read_index_meta,
    require_vector_index_online,
    vector_search,
)
from graphrag_lab.retrieval import (
    RetrievalConfig,
    RetrievalRequest,
    retrieve_vector_candidates,
    retrieve_with_vector,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

TEST_CORPUS_ID = "hp_t04_vector_fixture_v1"

FIXTURE_PARAGRAPHS = [
    {
        "id": "p_ada",
        "title": "Ada Lovelace",
        "sentences": [
            "Ada Lovelace wrote the first computer program for the Analytical Engine.",
            "She corresponded with Charles Babbage about the engine.",
        ],
    },
    {
        "id": "p_eiffel",
        "title": "Eiffel Tower",
        "sentences": [
            "The Eiffel Tower is located in Paris and was completed in 1889.",
            "It was constructed as the entrance to the World's Fair.",
        ],
    },
    {
        "id": "p_photo",
        "title": "Photosynthesis",
        "sentences": [
            "Photosynthesis in plants converts sunlight into chemical energy.",
            "Chlorophyll absorbs light in the leaves.",
        ],
    },
]

FIXTURE_QUESTION = "Who wrote the first computer program?"
EXPECTED_TOP_PARAGRAPH = "p_ada"


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


def _build_chunks(corpus_id: str = TEST_CORPUS_ID) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for paragraph in FIXTURE_PARAGRAPHS:
        text = " ".join(paragraph["sentences"])
        chunks.append(
            ChunkRecord(
                id=f"c_{paragraph['id']}",
                corpus_id=corpus_id,
                paragraph_id=str(paragraph["id"]),
                title=str(paragraph["title"]),
                text=text,
                sentence_ids=list(range(len(paragraph["sentences"]))),
                sentence_texts=[str(s) for s in paragraph["sentences"]],
                token_count=len(text.split()) + len(str(paragraph["title"]).split()) + 3,
                content_hash=f"hash-{paragraph['id']}",
                chunker_version=CHUNKER_VERSION,
            )
        )
    return chunks


@pytest.fixture(scope="module")
def backend() -> SentenceTransformerBackend:
    pytest.importorskip("sentence_transformers", reason="embedding backend not installed")
    pytest.importorskip("torch", reason="torch backend not installed")
    return SentenceTransformerBackend()


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
def clean_index(store, backend):
    driver, config = store
    database = config.neo4j.database
    delete_vector_data(driver, database)
    ensure_chunk_constraints(driver, database)
    dimension = backend.dimension
    ensure_vector_index(driver, database, dimension)
    ensure_index_meta(
        driver,
        database,
        corpus_id=TEST_CORPUS_ID,
        embedding_model=backend.model_name,
        dimension=dimension,
        chunker_version=CHUNKER_VERSION,
    )
    yield driver, database, backend
    delete_vector_data(driver, database)


def _embed_chunks(backend: SentenceTransformerBackend, chunks: list[ChunkRecord]):
    texts = [build_embedding_text(chunk.title, chunk.text) for chunk in chunks]
    for chunk in chunks:
        validate_chunk_fits_model(chunk.title, chunk.text, backend)
    return backend.embed_texts(texts)


def test_embeddings_match_model_dimension(backend):
    dimension = backend.dimension
    assert isinstance(dimension, int) and dimension > 0
    # Pinned MiniLM-L6-v2 is 384; the assertion below guards the contract
    # (dimension read from the model) while documenting the expected value.
    assert dimension == 384
    vectors = backend.embed_texts(["hello world", "vector retrieval test"])
    assert len(vectors) == 2
    for vector in vectors:
        validate_embedding(vector, dimension)
        norm = math.sqrt(sum(v * v for v in vector))
        assert norm == pytest.approx(1.0, abs=1e-3)


def test_chunk_tokens_fit_pinned_model(backend):
    for paragraph in FIXTURE_PARAGRAPHS:
        length = validate_chunk_fits_model(
            str(paragraph["title"]), " ".join(paragraph["sentences"]), backend
        )
        assert 0 < length <= backend.max_seq_length


def test_ingest_and_retrieve_ranks_relevant_paragraph(clean_index):
    driver, database, backend = clean_index
    chunks = _build_chunks()
    vectors = _embed_chunks(backend, chunks)
    assert count_chunks(driver, database) == 0
    assert ingest_chunks(
        driver, database, chunks, vectors, embedding_model=backend.model_name
    ) == len(chunks)
    assert count_chunks(driver, database) == len(chunks)
    require_vector_index_online(driver, database)
    assert get_vector_index_state(driver, database) == "ONLINE"

    request = RetrievalRequest(
        question=FIXTURE_QUESTION,
        corpus_id=TEST_CORPUS_ID,
        embedding_model=backend.model_name,
    )
    result = retrieve_vector_candidates(
        request, backend, driver, database, RetrievalConfig(num_candidates=30)
    )
    assert result.total_candidates >= len(chunks)
    assert result.corpus_id == TEST_CORPUS_ID
    assert result.dimension == backend.dimension
    ranked_ids = [chunk.chunk_id for chunk in result.chunks]
    assert f"c_{EXPECTED_TOP_PARAGRAPH}" in ranked_ids
    # The lexically and semantically overlapping paragraph must rank first.
    assert ranked_ids[0] == f"c_{EXPECTED_TOP_PARAGRAPH}"
    top = result.chunks[0]
    assert top.paragraph_id == EXPECTED_TOP_PARAGRAPH
    assert top.origin == "vector"
    assert top.rank == 1


def test_repeated_ingestion_is_idempotent(clean_index):
    driver, database, backend = clean_index
    chunks = _build_chunks()
    vectors = _embed_chunks(backend, chunks)
    ingest_chunks(driver, database, chunks, vectors, embedding_model=backend.model_name)
    before = count_chunks(driver, database)
    first = fetch_chunk_embedding(driver, database, chunks[0].id)
    assert first is not None
    ingest_chunks(driver, database, chunks, vectors, embedding_model=backend.model_name)
    assert count_chunks(driver, database) == before == len(chunks)
    second = fetch_chunk_embedding(driver, database, chunks[0].id)
    assert second is not None
    assert len(second) == backend.dimension
    for a, b in zip(first, second):
        assert a == pytest.approx(b, abs=1e-9)


def test_mismatched_fingerprint_fails(clean_index):
    driver, database, backend = clean_index
    chunks = _build_chunks()
    vectors = _embed_chunks(backend, chunks)
    ingest_chunks(driver, database, chunks, vectors, embedding_model=backend.model_name)
    before = count_chunks(driver, database)

    other = _build_chunks(corpus_id="hp_different_corpus")
    other_vectors = _embed_chunks(backend, other)
    with pytest.raises(ConfigMismatchError, match="mismatch|Mixed|corpus"):
        ingest_chunks(driver, database, other, other_vectors, embedding_model=backend.model_name)
    assert count_chunks(driver, database) == before

    wrong_dim = [vector[: max(1, len(vector) - 1)] for vector in vectors]
    with pytest.raises((ConfigMismatchError, ValueError), match="dimension|mismatch"):
        ingest_chunks(driver, database, chunks, wrong_dim, embedding_model=backend.model_name)
    assert count_chunks(driver, database) == before


def test_online_check_gates_retrieval(clean_index):
    driver, database, backend = clean_index
    chunks = _build_chunks()
    vectors = _embed_chunks(backend, chunks)
    ingest_chunks(driver, database, chunks, vectors, embedding_model=backend.model_name)

    # Missing index name is never ONLINE.
    with pytest.raises(IndexNotReadyError, match="not ONLINE|does not exist"):
        vector_search(driver, database, vectors[0], 5, name="missing_vector_index")
    with pytest.raises(IndexNotReadyError, match="not ONLINE"):
        require_vector_index_online(driver, database, name="missing_vector_index")

    # Real index is ONLINE and serves deterministic score/id ordering.
    assert get_vector_index_dimension(driver, database) == backend.dimension
    rows = vector_search(driver, database, vectors[0], 3)
    assert len(rows) == 3
    scores = [float(row["score"]) for row in rows]
    assert scores == sorted(scores, reverse=True)

    # Retrieval still refuses a foreign corpus instead of post-filtering ANN hits.
    query = ensure_normalized(vectors[0])
    with pytest.raises(ConfigMismatchError, match="corpus|Mixed"):
        retrieve_with_vector(
            question=FIXTURE_QUESTION,
            query_vector=query,
            corpus_id="hp_foreign",
            embedding_model=backend.model_name,
            dimension=backend.dimension,
            driver=driver,
            database=database,
            config=RetrievalConfig(num_candidates=5),
        )
    meta = read_index_meta(driver, database)
    assert meta is not None and meta["corpus_id"] == TEST_CORPUS_ID
