"""Neo4j vector store for T04 (vector-qa).

Owns all Cypher for chunks and the cosine vector index. One active
corpus/index namespace is enforced via an `IndexMeta` fingerprint node:
ingestion and retrieval refuse mismatched corpus/model/dimension instead of
mixing corpora or post-filtering ANN results.

Retrieval is gated on the vector index being ONLINE; callers receive
INDEX_NOT_READY rather than partial results.

T09 adds the evidence-backed graph snapshot: canonical `Entity` nodes,
`MENTIONS` (Chunk -> Entity) links and directed `REL` (Entity -> Entity)
edges with unioned evidence chunk ids. Graph persistence is idempotent and
corpus-scoped; a separate `GraphMeta` fingerprint node isolates one
resolver/extraction snapshot and refuses silent remapping.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from neo4j import Driver, GraphDatabase

from graphrag_lab.chunking import ChunkRecord
from graphrag_lab.config import AppConfig
from graphrag_lab.embeddings import (
    build_index_fingerprint,
    ensure_normalized,
    validate_embedding,
)

VECTOR_INDEX_NAME = "chunk_embedding"
CHUNK_LABEL = "Chunk"
META_LABEL = "IndexMeta"
META_ID = "vector-index"

ENTITY_LABEL = "Entity"
GRAPH_META_LABEL = "GraphMeta"
GRAPH_META_ID = "graph-index"

INDEX_ONLINE_TIMEOUT_S = 60.0
INGEST_BATCH_SIZE = 200
GRAPH_INGEST_BATCH_SIZE = 200


class StoreError(RuntimeError):
    code = "STORE_ERROR"


class ConfigMismatchError(StoreError, ValueError):
    code = "CONFIG_MISMATCH"


class IndexNotReadyError(StoreError, RuntimeError):
    code = "INDEX_NOT_READY"


class DependencyUnavailableError(StoreError, RuntimeError):
    code = "DEPENDENCY_UNAVAILABLE"


class GraphTimeoutError(StoreError, TimeoutError):
    code = "GRAPH_TIMEOUT"


def create_driver(config: AppConfig) -> Driver:
    """Create a Bolt driver; password must come from process environment."""
    if config.neo4j_password is None:
        raise DependencyUnavailableError(
            "Neo4j password is not configured; set GRAPHRAG_NEO4J_PASSWORD"
        )
    password = config.neo4j_password.get_secret_value()
    if not password.strip():
        raise DependencyUnavailableError("Neo4j password must be nonempty")
    try:
        return GraphDatabase.driver(config.neo4j.uri, auth=("neo4j", password))
    except Exception as exc:
        raise DependencyUnavailableError(f"Cannot create Neo4j driver: {exc}") from exc


def verify_connectivity(driver: Driver, database: str) -> None:
    try:
        with driver.session(database=database) as session:
            session.run("RETURN 1 AS healthy").single()
    except Exception as exc:
        raise DependencyUnavailableError(f"Neo4j is unavailable: {exc}") from exc


def ensure_chunk_constraints(driver: Driver, database: str) -> None:
    with driver.session(database=database) as session:
        session.run(
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
        ).consume()
        session.run(
            "CREATE CONSTRAINT indexmeta_id_unique IF NOT EXISTS "
            "FOR (m:IndexMeta) REQUIRE m.id IS UNIQUE"
        ).consume()


def get_vector_index_state(
    driver: Driver, database: str, name: str = VECTOR_INDEX_NAME
) -> str | None:
    with driver.session(database=database) as session:
        record = session.run(
            "SHOW VECTOR INDEXES YIELD name, state WHERE name = $name RETURN state",
            {"name": name},
        ).single()
        return str(record["state"]) if record is not None else None


def get_vector_index_dimension(
    driver: Driver, database: str, name: str = VECTOR_INDEX_NAME
) -> int | None:
    with driver.session(database=database) as session:
        record = session.run(
            "SHOW VECTOR INDEXES YIELD name, options WHERE name = $name RETURN options",
            {"name": name},
        ).single()
        if record is None:
            return None
        options = record["options"] or {}
        config = options.get("indexConfig", options)
        for key in ("vector.dimensions", "vector.dimensions "):
            if key in config:
                return int(config[key])
        return None


def ensure_vector_index(
    driver: Driver,
    database: str,
    dimension: int,
    name: str = VECTOR_INDEX_NAME,
    timeout_s: float = INDEX_ONLINE_TIMEOUT_S,
) -> None:
    """Create the cosine vector index if absent; refuse dimension changes."""
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    existing_dim = get_vector_index_dimension(driver, database, name)
    if existing_dim is not None and existing_dim != dimension:
        raise ConfigMismatchError(
            f"Vector index {name!r} has dimension {existing_dim}, "
            f"requested {dimension}; refusing mismatch"
        )
    with driver.session(database=database) as session:
        session.run(
            f"CREATE VECTOR INDEX `{name}` IF NOT EXISTS "
            f"FOR (c:{CHUNK_LABEL}) ON (c.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: $dimension, "
            "`vector.similarity_function`: 'cosine'}}",
            {"dimension": dimension},
        ).consume()
    wait_for_vector_index_online(driver, database, name, timeout_s=timeout_s)


def wait_for_vector_index_online(
    driver: Driver,
    database: str,
    name: str = VECTOR_INDEX_NAME,
    timeout_s: float = INDEX_ONLINE_TIMEOUT_S,
) -> None:
    import time

    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    deadline = time.monotonic() + timeout_s
    last: str | None = None
    while True:
        last = get_vector_index_state(driver, database, name)
        if last == "ONLINE":
            return
        if last == "FAILED":
            raise IndexNotReadyError(f"Vector index {name!r} is in FAILED state")
        if time.monotonic() >= deadline:
            raise IndexNotReadyError(
                f"Vector index {name!r} is not ONLINE after {timeout_s}s "
                f"(last state: {last}); refusing retrieval"
            )
        time.sleep(0.5)


def require_vector_index_online(
    driver: Driver, database: str, name: str = VECTOR_INDEX_NAME
) -> None:
    state = get_vector_index_state(driver, database, name)
    if state != "ONLINE":
        raise IndexNotReadyError(
            f"Vector index {name!r} state is {state!r}, not ONLINE; refusing retrieval"
        )


def read_index_meta(driver: Driver, database: str, meta_id: str = META_ID) -> dict[str, Any] | None:
    with driver.session(database=database) as session:
        record = session.run(
            f"MATCH (m:{META_LABEL} {{id: $id}}) RETURN m",
            {"id": meta_id},
        ).single()
        if record is None:
            return None
        node = record["m"]
        return dict(node)


def ensure_index_meta(
    driver: Driver,
    database: str,
    *,
    corpus_id: str,
    embedding_model: str,
    dimension: int,
    chunker_version: str,
    meta_id: str = META_ID,
) -> str:
    """Create or verify the single active fingerprint; refuse mismatches."""
    if not corpus_id or not embedding_model or not chunker_version:
        raise ValueError("corpus_id, embedding_model and chunker_version must be nonempty")
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    fingerprint = build_index_fingerprint(corpus_id, embedding_model, dimension, chunker_version)
    with driver.session(database=database) as session:
        existing = session.run(
            f"MATCH (m:{META_LABEL} {{id: $id}}) RETURN m",
            {"id": meta_id},
        ).single()
        if existing is not None:
            node = dict(existing["m"])
            if node.get("fingerprint") != fingerprint:
                stored = (
                    f"stored corpus={node.get('corpus_id')!r} "
                    f"model={node.get('embedding_model')!r} "
                    f"dim={node.get('dimension')!r} "
                    f"chunker={node.get('chunker_version')!r}"
                )
                raise ConfigMismatchError(
                    "Stored vector index fingerprint does not match requested corpus/model: "
                    f"{stored}; requested corpus={corpus_id!r} model={embedding_model!r} "
                    f"dim={dimension!r} chunker={chunker_version!r}; refusing mixed corpora"
                )
            return str(node["fingerprint"])
        session.run(
            f"MERGE (m:{META_LABEL} {{id: $id}}) "
            "SET m.fingerprint = $fingerprint, m.corpus_id = $corpus_id, "
            "m.embedding_model = $embedding_model, m.dimension = $dimension, "
            "m.chunker_version = $chunker_version",
            {
                "id": meta_id,
                "fingerprint": fingerprint,
                "corpus_id": corpus_id,
                "embedding_model": embedding_model,
                "dimension": dimension,
                "chunker_version": chunker_version,
            },
        ).consume()
    return fingerprint


def _coerce_chunk(record: ChunkRecord | dict[str, Any]) -> ChunkRecord:
    if isinstance(record, ChunkRecord):
        return record
    return ChunkRecord.model_validate(record)


def ingest_chunks(
    driver: Driver,
    database: str,
    chunks: Sequence[ChunkRecord | dict[str, Any]],
    embeddings: Sequence[Sequence[float]],
    *,
    embedding_model: str,
) -> int:
    """Idempotently MERGE chunks with normalized embeddings.

    Validates fingerprint agreement before writing; mismatched corpus/model/
    dimension fail without partial writes. Repeated ingestion of identical
    chunks unions deterministically and changes no counts.
    """
    records = [_coerce_chunk(chunk) for chunk in chunks]
    vectors = [list(map(float, vec)) for vec in embeddings]
    if not records:
        raise ValueError("chunks must be a nonempty list")
    if len(records) != len(vectors):
        raise ValueError("chunks and embeddings length mismatch")
    if not embedding_model or not embedding_model.strip():
        raise ValueError("embedding_model must be a nonempty string")
    corpus_ids = {record.corpus_id for record in records}
    if len(corpus_ids) != 1:
        raise ValueError("All chunks in one ingest call must share one corpus_id")
    corpus_id = next(iter(corpus_ids))
    chunker_versions = {record.chunker_version for record in records}
    if len(chunker_versions) != 1:
        raise ValueError("All chunks in one ingest call must share one chunker_version")
    chunker_version = next(iter(chunker_versions))

    expected_dim = len(vectors[0])
    if expected_dim <= 0:
        raise ValueError("Embeddings must be nonempty")
    normalized: list[list[float]] = []
    for index, vector in enumerate(vectors):
        validate_embedding(vector, expected_dim, what=f"embedding {index}")
        normalized.append(ensure_normalized(vector))
    dimension = expected_dim

    meta = read_index_meta(driver, database)
    if meta is None:
        raise ConfigMismatchError(
            "No vector index fingerprint is registered; call ensure_index_meta first"
        )
    fingerprint = build_index_fingerprint(corpus_id, embedding_model, dimension, chunker_version)
    if meta.get("fingerprint") != fingerprint:
        raise ConfigMismatchError(
            f"Ingest fingerprint mismatch: stored corpus={meta.get('corpus_id')!r} "
            f"model={meta.get('embedding_model')!r} dim={meta.get('dimension')!r}; "
            f"requested corpus={corpus_id!r} model={embedding_model!r} dim={dimension!r}; "
            "refusing mixed corpora"
        )
    index_dim = get_vector_index_dimension(driver, database)
    if index_dim is not None and index_dim != dimension:
        raise ConfigMismatchError(
            f"Vector index dimension {index_dim} != embedding dimension {dimension}"
        )

    rows = [
        {
            "id": record.id,
            "corpus_id": record.corpus_id,
            "paragraph_id": record.paragraph_id,
            "title": record.title,
            "text": record.text,
            "sentence_ids": list(record.sentence_ids),
            "sentence_texts": list(record.sentence_texts),
            "embedding": vector,
            "embedding_model": embedding_model,
            "content_hash": record.content_hash,
            "chunker_version": record.chunker_version,
        }
        for record, vector in zip(records, normalized)
    ]
    with driver.session(database=database) as session:
        for start in range(0, len(rows), INGEST_BATCH_SIZE):
            batch = rows[start : start + INGEST_BATCH_SIZE]
            session.run(
                f"UNWIND $rows AS row MERGE (c:{CHUNK_LABEL} {{id: row.id}}) "
                "SET c.corpus_id = row.corpus_id, c.paragraph_id = row.paragraph_id, "
                "c.title = row.title, c.text = row.text, "
                "c.sentence_ids = row.sentence_ids, c.sentence_texts = row.sentence_texts, "
                "c.embedding = row.embedding, c.embedding_model = row.embedding_model, "
                "c.content_hash = row.content_hash, c.chunker_version = row.chunker_version",
                {"rows": batch},
            ).consume()
    return len(rows)


def count_chunks(driver: Driver, database: str, corpus_id: str | None = None) -> int:
    with driver.session(database=database) as session:
        if corpus_id is None:
            record = session.run(f"MATCH (c:{CHUNK_LABEL}) RETURN count(c) AS n").single()
        else:
            record = session.run(
                f"MATCH (c:{CHUNK_LABEL}) WHERE c.corpus_id = $corpus_id RETURN count(c) AS n",
                {"corpus_id": corpus_id},
            ).single()
        return int(record["n"]) if record else 0


def fetch_chunk_embedding(driver: Driver, database: str, chunk_id: str) -> list[float] | None:
    with driver.session(database=database) as session:
        record = session.run(
            f"MATCH (c:{CHUNK_LABEL} {{id: $id}}) RETURN c.embedding AS embedding",
            {"id": chunk_id},
        ).single()
        if record is None or record["embedding"] is None:
            return None
        return [float(v) for v in list(record["embedding"])]


def vector_search(
    driver: Driver,
    database: str,
    query_embedding: Sequence[float],
    top_k: int,
    name: str = VECTOR_INDEX_NAME,
) -> list[dict[str, Any]]:
    """Cosine ANN lookup gated on ONLINE; deterministic score/id ordering."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 100:
        raise ValueError("top_k must be an integer in 1..100")
    query = [float(v) for v in list(query_embedding)]
    if not query:
        raise ValueError("query_embedding must be nonempty")
    index_dim = get_vector_index_dimension(driver, database, name)
    if index_dim is None:
        raise IndexNotReadyError(f"Vector index {name!r} does not exist; refusing retrieval")
    validate_embedding(query, index_dim, what="query_embedding")
    require_vector_index_online(driver, database, name)
    with driver.session(database=database) as session:
        result = session.run(
            "CALL db.index.vector.queryNodes($name, $k, $embedding) "
            "YIELD node AS chunk, score "
            "RETURN chunk.id AS id, chunk.corpus_id AS corpus_id, "
            "chunk.paragraph_id AS paragraph_id, chunk.title AS title, "
            "chunk.text AS text, chunk.sentence_ids AS sentence_ids, "
            "chunk.sentence_texts AS sentence_texts, chunk.content_hash AS content_hash, "
            "chunk.chunker_version AS chunker_version, "
            "chunk.embedding_model AS embedding_model, score "
            "ORDER BY score DESC, chunk.id ASC",
            {"name": name, "k": top_k, "embedding": ensure_normalized(query)},
        )
        return [dict(record) for record in result]


def delete_vector_data(driver: Driver, database: str) -> None:
    """Test-only helper: remove all Chunk and IndexMeta nodes (index definition kept)."""
    with driver.session(database=database) as session:
        session.run(f"MATCH (c:{CHUNK_LABEL}) DETACH DELETE c").consume()
        session.run(f"MATCH (m:{META_LABEL}) DETACH DELETE m").consume()


# ---------------------------------------------------------------------------
# T09 graph snapshot (graph-index, R06)
# ---------------------------------------------------------------------------


def ensure_graph_constraints(driver: Driver, database: str) -> None:
    """Create Entity uniqueness and lookup indexes plus the GraphMeta guard."""
    with driver.session(database=database) as session:
        session.run(
            f"CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:{ENTITY_LABEL}) "
            "REQUIRE e.id IS UNIQUE"
        ).consume()
        session.run(
            f"CREATE CONSTRAINT graphmeta_id_unique IF NOT EXISTS "
            f"FOR (m:{GRAPH_META_LABEL}) REQUIRE m.id IS UNIQUE"
        ).consume()
        session.run(
            f"CREATE INDEX entity_normalized_name IF NOT EXISTS FOR (e:{ENTITY_LABEL}) "
            "ON (e.normalized_name)"
        ).consume()
        session.run(
            f"CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:{ENTITY_LABEL}) ON (e.type)"
        ).consume()


def read_graph_meta(
    driver: Driver, database: str, meta_id: str = GRAPH_META_ID
) -> dict[str, Any] | None:
    with driver.session(database=database) as session:
        record = session.run(
            f"MATCH (m:{GRAPH_META_LABEL} {{id: $id}}) RETURN m",
            {"id": meta_id},
        ).single()
        if record is None:
            return None
        return dict(record["m"])


def ensure_graph_meta(
    driver: Driver,
    database: str,
    *,
    corpus_id: str,
    resolver_version: str,
    extraction_version: str,
    schema_version: str,
    meta_id: str = GRAPH_META_ID,
) -> str:
    """Create or verify the single active graph fingerprint; refuse mismatches.

    A resolver or extraction/schema change rebuilds the snapshot; historical
    run evidence is never silently remapped.
    """
    from graphrag_lab.graph_index import build_graph_fingerprint

    if not corpus_id or not resolver_version or not extraction_version or not schema_version:
        raise ValueError(
            "corpus_id, resolver_version, extraction_version and schema_version required"
        )
    fingerprint = build_graph_fingerprint(
        corpus_id, resolver_version, extraction_version, schema_version
    )
    with driver.session(database=database) as session:
        existing = session.run(
            f"MATCH (m:{GRAPH_META_LABEL} {{id: $id}}) RETURN m",
            {"id": meta_id},
        ).single()
        if existing is not None:
            node = dict(existing["m"])
            if node.get("fingerprint") != fingerprint:
                stored = (
                    f"stored corpus={node.get('corpus_id')!r} "
                    f"resolver={node.get('resolver_version')!r} "
                    f"extraction={node.get('extraction_version')!r} "
                    f"schema={node.get('schema_version')!r}"
                )
                raise ConfigMismatchError(
                    "Stored graph fingerprint does not match requested snapshot: "
                    f"{stored}; requested corpus={corpus_id!r} resolver={resolver_version!r} "
                    f"extraction={extraction_version!r} schema={schema_version!r}; "
                    "refusing mixed snapshots"
                )
            return str(node["fingerprint"])
        session.run(
            f"MERGE (m:{GRAPH_META_LABEL} {{id: $id}}) "
            "SET m.fingerprint = $fingerprint, m.corpus_id = $corpus_id, "
            "m.resolver_version = $resolver_version, "
            "m.extraction_version = $extraction_version, m.schema_version = $schema_version",
            {
                "id": meta_id,
                "fingerprint": fingerprint,
                "corpus_id": corpus_id,
                "resolver_version": resolver_version,
                "extraction_version": extraction_version,
                "schema_version": schema_version,
            },
        ).consume()
    return fingerprint


def _normalize_snapshot(driver: Driver, database: str, snapshot: Any) -> Any:
    from graphrag_lab.graph_index import validate_graph_snapshot

    if isinstance(snapshot, dict):
        from graphrag_lab.graph_index import GraphSnapshot

        snapshot = GraphSnapshot.model_validate(snapshot)
    validate_graph_snapshot(snapshot)
    return snapshot


def _chunk_shell_rows(chunks: Sequence[Any]) -> tuple[str, list[dict[str, Any]]]:
    from graphrag_lab.extraction import coerce_chunk_view

    views = [coerce_chunk_view(item) for item in chunks]
    if not views:
        raise ValueError("chunks must be a nonempty list")
    corpora = {view["corpus_id"] for view in views}
    if len(corpora) != 1:
        raise ConfigMismatchError(f"All chunks must share one corpus_id, got {sorted(corpora)}")
    rows = [
        {
            "id": view["chunk_id"],
            "corpus_id": view["corpus_id"],
            "paragraph_id": view.get("paragraph_id"),
            "title": view["title"],
            "text": view["text"],
            "sentence_ids": list(view["sentence_ids"]),
            "sentence_texts": list(view["sentence_texts"]),
            "content_hash": view.get("content_hash"),
            "chunker_version": view.get("chunker_version"),
        }
        for view in views
    ]
    return next(iter(corpora)), rows


def ingest_graph_snapshot(
    driver: Driver,
    database: str,
    chunks: Sequence[Any],
    snapshot: Any,
    *,
    meta_id: str = GRAPH_META_ID,
    batch_size: int = GRAPH_INGEST_BATCH_SIZE,
) -> dict[str, int]:
    """Idempotently load entities, MENTIONS and directed REL edges.

    Chunk shells are MERGEd without touching stored embeddings, so a vector
    index built by T04 is preserved. Entities are MERGEd by canonical id.
    MENTIONS links union mention evidence; REL edges union evidence chunk ids
    deterministically, so repeated loads create no duplicates. Fingerprint
    mismatches fail without partial writes.
    """
    from graphrag_lab.graph_index import validate_graph_snapshot

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    record = _normalize_snapshot(driver, database, snapshot)
    corpus_id, shell_rows = _chunk_shell_rows(chunks)
    if record.corpus_id != corpus_id:
        raise ConfigMismatchError(
            f"Snapshot corpus {record.corpus_id!r} != chunk corpus {corpus_id!r}; refusing mix"
        )
    meta = read_graph_meta(driver, database, meta_id)
    if meta is None:
        raise ConfigMismatchError(
            "No graph fingerprint is registered; call ensure_graph_meta first"
        )
    if meta.get("fingerprint") != record.fingerprint:
        raise ConfigMismatchError(
            f"Graph fingerprint mismatch: stored corpus={meta.get('corpus_id')!r} "
            f"resolver={meta.get('resolver_version')!r}; "
            f"snapshot corpus={record.corpus_id!r} resolver={record.resolver_version!r}; "
            "refusing mixed snapshots"
        )
    if meta.get("corpus_id") != corpus_id:
        raise ConfigMismatchError(
            f"Chunk corpus {corpus_id!r} != graph corpus {meta.get('corpus_id')!r}"
        )
    validate_graph_snapshot(record, chunks)

    entity_map: dict[str, dict[str, Any]] = {}
    for entity in record.entities:
        item = dict(entity) if isinstance(entity, dict) else dict(entity)
        if item.get("corpus_id") != corpus_id:
            raise ConfigMismatchError(
                f"Entity {item.get('id')!r} corpus mismatch; refusing mixed corpora"
            )
        entity_map[str(item["id"])] = item

    with driver.session(database=database) as session:
        for start in range(0, len(shell_rows), batch_size):
            batch = shell_rows[start : start + batch_size]
            session.run(
                f"UNWIND $rows AS row MERGE (c:{CHUNK_LABEL} {{id: row.id}}) "
                "SET c.corpus_id = row.corpus_id, c.paragraph_id = row.paragraph_id, "
                "c.title = row.title, c.text = row.text, "
                "c.sentence_ids = row.sentence_ids, c.sentence_texts = row.sentence_texts, "
                "c.content_hash = row.content_hash, c.chunker_version = row.chunker_version",
                {"rows": batch},
            ).consume()

        entity_rows = [
            {
                "id": item["id"],
                "corpus_id": item["corpus_id"],
                "canonical_name": item["canonical_name"],
                "normalized_name": item["normalized_name"],
                "type": item["type"].value if hasattr(item["type"], "value") else str(item["type"]),
                "aliases": list(item.get("aliases") or []),
                "resolver_version": item.get("resolver_version", record.resolver_version),
            }
            for item in entity_map.values()
        ]
        for start in range(0, len(entity_rows), batch_size):
            batch = entity_rows[start : start + batch_size]
            if not batch:
                break
            session.run(
                f"UNWIND $rows AS row MERGE (e:{ENTITY_LABEL} {{id: row.id}}) "
                "SET e.corpus_id = row.corpus_id, e.canonical_name = row.canonical_name, "
                "e.normalized_name = row.normalized_name, e.type = row.type, "
                "e.aliases = row.aliases, e.resolver_version = row.resolver_version",
                {"rows": batch},
            ).consume()

        for mention in record.mentions:
            existing = session.run(
                f"MATCH (c:{CHUNK_LABEL} {{id: $chunk_id}})-[m:MENTIONS]->"
                f"(e:{ENTITY_LABEL} {{id: $entity_id}}) "
                "RETURN m.mention_ids AS mention_ids, m.sentence_ids AS sentence_ids, "
                "m.surface_forms AS surface_forms",
                {"chunk_id": mention.chunk_id, "entity_id": mention.canonical_entity_id},
            ).single()
            if existing is not None and existing["mention_ids"] is not None:
                mention_ids = sorted(set(list(existing["mention_ids"]) + list(mention.mention_ids)))
                sentence_ids = sorted(
                    set(list(existing["sentence_ids"] or []) + list(mention.sentence_ids))
                )
                surface_forms = sorted(
                    set(list(existing["surface_forms"] or []) + list(mention.surface_forms))
                )
            else:
                mention_ids = sorted(mention.mention_ids)
                sentence_ids = sorted(mention.sentence_ids)
                surface_forms = sorted(mention.surface_forms)
            session.run(
                f"MATCH (c:{CHUNK_LABEL} {{id: $chunk_id}}), "
                f"(e:{ENTITY_LABEL} {{id: $entity_id}}) "
                f"MERGE (c)-[m:MENTIONS]->(e) "
                "SET m.mention_ids = $mention_ids, m.sentence_ids = $sentence_ids, "
                "m.surface_forms = $surface_forms",
                {
                    "chunk_id": mention.chunk_id,
                    "entity_id": mention.canonical_entity_id,
                    "mention_ids": mention_ids,
                    "sentence_ids": sentence_ids,
                    "surface_forms": surface_forms,
                },
            ).consume()

        for relation in record.relations:
            rel_type = (
                relation.type.value if hasattr(relation.type, "value") else str(relation.type)
            )
            existing = session.run(
                f"MATCH (s:{ENTITY_LABEL})-[r:REL {{id: $id}}]->(t:{ENTITY_LABEL}) "
                "RETURN s.id AS source, t.id AS target, r.type AS type, "
                "r.evidence_chunk_ids AS evidence, r.corpus_id AS corpus",
                {"id": relation.id},
            ).single()
            if existing is not None:
                if (
                    existing["source"] != relation.source_entity_id
                    or existing["target"] != relation.target_entity_id
                ):
                    raise ConfigMismatchError(
                        f"Relation {relation.id!r} endpoints changed; rebuild the snapshot "
                        "instead of remapping history"
                    )
                if existing["type"] != rel_type:
                    raise ConfigMismatchError(
                        f"Relation {relation.id!r} type changed; refusing remap"
                    )
                if existing["corpus"] != corpus_id:
                    raise ConfigMismatchError(f"Relation {relation.id!r} corpus mismatch")
                evidence = sorted(
                    set(list(existing["evidence"] or []) + list(relation.evidence_chunk_ids))
                )
            else:
                evidence = sorted(relation.evidence_chunk_ids)
            session.run(
                f"MATCH (s:{ENTITY_LABEL} {{id: $source}}), (t:{ENTITY_LABEL} {{id: $target}}) "
                f"MERGE (s)-[r:REL {{id: $id}}]->(t) "
                "SET r.corpus_id = $corpus, r.type = $type, "
                "r.evidence_chunk_ids = $evidence, r.extraction_version = $extraction",
                {
                    "source": relation.source_entity_id,
                    "target": relation.target_entity_id,
                    "id": relation.id,
                    "corpus": corpus_id,
                    "type": rel_type,
                    "evidence": evidence,
                    "extraction": relation.extraction_version,
                },
            ).consume()

    return {
        "n_chunks": len(shell_rows),
        "n_entities": len(entity_map),
        "n_mentions": len(record.mentions),
        "n_relations": len(record.relations),
    }


def count_entities(driver: Driver, database: str, corpus_id: str | None = None) -> int:
    with driver.session(database=database) as session:
        if corpus_id is None:
            record = session.run(f"MATCH (e:{ENTITY_LABEL}) RETURN count(e) AS n").single()
        else:
            record = session.run(
                f"MATCH (e:{ENTITY_LABEL}) WHERE e.corpus_id = $corpus_id RETURN count(e) AS n",
                {"corpus_id": corpus_id},
            ).single()
        return int(record["n"]) if record else 0


def count_relations(driver: Driver, database: str, corpus_id: str | None = None) -> int:
    with driver.session(database=database) as session:
        if corpus_id is None:
            record = session.run("MATCH ()-[r:REL]->() RETURN count(r) AS n").single()
        else:
            record = session.run(
                "MATCH ()-[r:REL]->() WHERE r.corpus_id = $corpus_id RETURN count(r) AS n",
                {"corpus_id": corpus_id},
            ).single()
        return int(record["n"]) if record else 0


def count_mentions(driver: Driver, database: str) -> int:
    with driver.session(database=database) as session:
        record = session.run("MATCH ()-[m:MENTIONS]->() RETURN count(m) AS n").single()
        return int(record["n"]) if record else 0


def count_graph(driver: Driver, database: str, corpus_id: str | None = None) -> dict[str, int]:
    return {
        "entities": count_entities(driver, database, corpus_id),
        "relations": count_relations(driver, database, corpus_id),
        "mentions": count_mentions(driver, database),
    }


def fetch_entity(driver: Driver, database: str, entity_id: str) -> dict[str, Any] | None:
    with driver.session(database=database) as session:
        record = session.run(
            f"MATCH (e:{ENTITY_LABEL} {{id: $id}}) RETURN e",
            {"id": entity_id},
        ).single()
        if record is None:
            return None
        return dict(record["e"])


def fetch_relation(driver: Driver, database: str, relation_id: str) -> dict[str, Any] | None:
    with driver.session(database=database) as session:
        record = session.run(
            f"MATCH (s:{ENTITY_LABEL})-[r:REL {{id: $id}}]->(t:{ENTITY_LABEL}) "
            "RETURN s.id AS source, t.id AS target, r",
            {"id": relation_id},
        ).single()
        if record is None:
            return None
        payload = dict(record["r"])
        payload["source_entity_id"] = record["source"]
        payload["target_entity_id"] = record["target"]
        return payload


def fetch_neighbors(driver: Driver, database: str, entity_id: str) -> list[dict[str, Any]]:
    """Outgoing and incoming REL edges for one entity with evidence."""
    with driver.session(database=database) as session:
        result = session.run(
            f"MATCH (s:{ENTITY_LABEL} {{id: $id}})-[r:REL]->(t:{ENTITY_LABEL}) "
            "RETURN 'out' AS direction, t.id AS neighbor, r.id AS id, r.type AS type, "
            "r.evidence_chunk_ids AS evidence "
            "UNION ALL "
            f"MATCH (s:{ENTITY_LABEL})-[r:REL]->(t:{ENTITY_LABEL} {{id: $id}}) "
            "RETURN 'in' AS direction, s.id AS neighbor, r.id AS id, r.type AS type, "
            "r.evidence_chunk_ids AS evidence "
            "ORDER BY direction ASC, neighbor ASC",
            {"id": entity_id},
        )
        return [dict(row) for row in result]


def fetch_mentions_for_chunk(driver: Driver, database: str, chunk_id: str) -> list[dict[str, Any]]:
    with driver.session(database=database) as session:
        result = session.run(
            f"MATCH (c:{CHUNK_LABEL} {{id: $id}})-[m:MENTIONS]->(e:{ENTITY_LABEL}) "
            "RETURN e.id AS entity_id, e.canonical_name AS canonical_name, "
            "m.mention_ids AS mention_ids, m.sentence_ids AS sentence_ids, "
            "m.surface_forms AS surface_forms ORDER BY e.id ASC",
            {"id": chunk_id},
        )
        return [dict(row) for row in result]


def delete_graph_data(driver: Driver, database: str) -> None:
    """Test-only helper: remove Entity nodes (plus MENTIONS/REL) and GraphMeta."""
    with driver.session(database=database) as session:
        session.run(f"MATCH (e:{ENTITY_LABEL}) DETACH DELETE e").consume()
        session.run(f"MATCH (m:{GRAPH_META_LABEL}) DETACH DELETE m").consume()


# ---------------------------------------------------------------------------
# T10 bounded hybrid retrieval (hybrid-qa, R07)
# ---------------------------------------------------------------------------


def list_entities_for_corpus(driver: Driver, database: str, corpus_id: str) -> list[dict[str, Any]]:
    """Return corpus-scoped entities for offline linking (MATCH only).

    Ordered by entity id for deterministic linking. Never creates nodes.
    """
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise ValueError("corpus_id must be a nonempty string")
    if not isinstance(database, str) or not database.strip():
        raise ValueError("database must be a nonempty string")
    with driver.session(database=database) as session:
        result = session.run(
            f"MATCH (e:{ENTITY_LABEL}) WHERE e.corpus_id = $corpus_id "
            "RETURN e.id AS id, e.corpus_id AS corpus_id, "
            "e.canonical_name AS canonical_name, e.normalized_name AS normalized_name, "
            "e.type AS type, e.aliases AS aliases ORDER BY e.id ASC",
            {"corpus_id": corpus_id},
        )
        return [dict(row) for row in result]


def fetch_chunks_by_ids(
    driver: Driver, database: str, chunk_ids: Sequence[str], corpus_id: str | None = None
) -> list[dict[str, Any]]:
    """Fetch chunk shells by id; optional corpus scoping (MATCH only)."""
    ids = [str(v) for v in list(chunk_ids)]
    if not ids:
        return []
    if len(set(ids)) != len(ids):
        raise ValueError("chunk_ids must not repeat")
    if len(ids) > 1000:
        raise ValueError("chunk_ids batch exceeds 1000")
    with driver.session(database=database) as session:
        if corpus_id is None:
            result = session.run(
                f"UNWIND $ids AS cid MATCH (c:{CHUNK_LABEL} {{id: cid}}) "
                "RETURN c.id AS id, c.corpus_id AS corpus_id, "
                "c.paragraph_id AS paragraph_id, c.title AS title, c.text AS text, "
                "c.sentence_ids AS sentence_ids, c.sentence_texts AS sentence_texts, "
                "c.content_hash AS content_hash, c.chunker_version AS chunker_version, "
                "c.embedding_model AS embedding_model ORDER BY c.id ASC",
                {"ids": ids},
            )
        else:
            result = session.run(
                f"UNWIND $ids AS cid MATCH (c:{CHUNK_LABEL} {{id: cid}}) "
                "WHERE c.corpus_id = $corpus_id "
                "RETURN c.id AS id, c.corpus_id AS corpus_id, "
                "c.paragraph_id AS paragraph_id, c.title AS title, c.text AS text, "
                "c.sentence_ids AS sentence_ids, c.sentence_texts AS sentence_texts, "
                "c.content_hash AS content_hash, c.chunker_version AS chunker_version, "
                "c.embedding_model AS embedding_model ORDER BY c.id ASC",
                {"ids": ids, "corpus_id": corpus_id},
            )
        return [dict(row) for row in result]


def fetch_chunks_mentioning_entities(
    driver: Driver,
    database: str,
    entity_ids: Sequence[str],
    corpus_id: str,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Return (chunk_id, entity_id) pairs for MENTIONS links (MATCH only)."""
    ids = [str(v) for v in list(entity_ids)]
    if not ids:
        return []
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise ValueError("corpus_id must be a nonempty string")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 5000:
        raise ValueError("limit must be an integer in 1..5000")
    with driver.session(database=database) as session:
        result = session.run(
            f"MATCH (c:{CHUNK_LABEL})-[:MENTIONS]->(e:{ENTITY_LABEL}) "
            "WHERE e.id IN $entity_ids AND c.corpus_id = $corpus_id "
            "AND e.corpus_id = $corpus_id "
            "RETURN DISTINCT c.id AS chunk_id, e.id AS entity_id "
            "ORDER BY chunk_id ASC, entity_id ASC LIMIT $limit",
            {"entity_ids": ids, "corpus_id": corpus_id, "limit": limit},
        )
        return [dict(row) for row in result]


def fetch_bounded_neighbors(
    driver: Driver,
    database: str,
    entity_id: str,
    corpus_id: str,
    limit_per_node: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Fetch at most ``limit_per_node`` REL edges for one entity, both directions.

    Returns (edges, truncated) where ``truncated`` is True when the node has
    more eligible edges than the budget. Edges retain semantic source/target
    plus the traversal direction from the queried node. Deterministic order
    by (neighbor, type, relation id). Uses two fixed templates (out/in);
    ``max_hops`` looping in the caller whitelists hop counts to 1..2.
    """
    if not isinstance(entity_id, str) or not entity_id.strip():
        raise ValueError("entity_id must be a nonempty string")
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise ValueError("corpus_id must be a nonempty string")
    if (
        not isinstance(limit_per_node, int)
        or isinstance(limit_per_node, bool)
        or not 1 <= limit_per_node <= 200
    ):
        raise ValueError("limit_per_node must be an integer in 1..200")
    fetch_n = int(limit_per_node) + 1
    with driver.session(database=database) as session:
        outgoing = list(
            session.run(
                f"MATCH (s:{ENTITY_LABEL} {{id: $eid}})-[r:REL]->(t:{ENTITY_LABEL}) "
                "WHERE s.corpus_id = $corpus AND t.corpus_id = $corpus "
                "AND r.corpus_id = $corpus "
                "RETURN 'out' AS direction, t.id AS neighbor, r.id AS id, "
                "r.type AS type, s.id AS source, t.id AS target, "
                "r.evidence_chunk_ids AS evidence "
                "ORDER BY neighbor ASC, type ASC, id ASC LIMIT $limit",
                {"eid": entity_id, "corpus": corpus_id, "limit": fetch_n},
            )
        )
        incoming = list(
            session.run(
                f"MATCH (s:{ENTITY_LABEL})-[r:REL]->(t:{ENTITY_LABEL} {{id: $eid}}) "
                "WHERE s.corpus_id = $corpus AND t.corpus_id = $corpus "
                "AND r.corpus_id = $corpus "
                "RETURN 'in' AS direction, s.id AS neighbor, r.id AS id, "
                "r.type AS type, s.id AS source, t.id AS target, "
                "r.evidence_chunk_ids AS evidence "
                "ORDER BY neighbor ASC, type ASC, id ASC LIMIT $limit",
                {"eid": entity_id, "corpus": corpus_id, "limit": fetch_n},
            )
        )
    combined = [dict(row) for row in [*outgoing, *incoming]]
    combined.sort(
        key=lambda r: (
            str(r.get("neighbor")),
            str(r.get("type")),
            str(r.get("id")),
            str(r.get("direction")),
        )
    )
    truncated = len(combined) > limit_per_node
    return combined[:limit_per_node], truncated


def expand_bounded_frontier(
    driver: Driver,
    database: str,
    seed_ids: Sequence[str],
    corpus_id: str,
    *,
    max_hops: int = 2,
    max_edges_per_node: int = 25,
    max_visited_entities: int = 100,
    max_graph_chunks: int = 40,
    timeout_s: float = 2.0,
    clock: Any = None,
) -> dict[str, Any]:
    """Breadth-first REL expansion with budgets enforced before work.

    Traverses either direction for recall while retaining semantic
    source/target on every edge. Budgets: per-node edges, visited entities,
    graph chunks and a wall-clock timeout. Every cap hit is recorded; the
    caller surfaces them in hybrid diagnostics. Raises GraphTimeoutError on
    deadline expiry and ConfigMismatchError on corpus drift.
    """
    import time as _time

    seeds = [str(v) for v in list(seed_ids)]
    if not seeds:
        raise ValueError("seed_ids must be a nonempty list")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed_ids must not repeat")
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise ValueError("corpus_id must be a nonempty string")
    if not isinstance(max_hops, int) or isinstance(max_hops, bool) or max_hops not in (1, 2):
        raise ValueError("max_hops must be 1 or 2 (whitelisted templates only)")
    for label, value, low, high in (
        ("max_edges_per_node", max_edges_per_node, 1, 200),
        ("max_visited_entities", max_visited_entities, 1, 1000),
        ("max_graph_chunks", max_graph_chunks, 1, 500),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
            raise ValueError(f"{label} must be an integer in {low}..{high}")
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    now = clock if clock is not None else _time.monotonic
    deadline = now() + float(timeout_s)

    def _check_deadline() -> None:
        if now() >= deadline:
            raise GraphTimeoutError(
                f"Graph traversal exceeded timeout_s={float(timeout_s)}; "
                "fall back to vector retrieval"
            )

    # Verify seeds belong to this corpus before expanding.
    _check_deadline()
    seed_rows = list_entities_for_corpus(driver, database, corpus_id)
    known = {row["id"] for row in seed_rows}
    for seed in seeds:
        if seed not in known:
            raise ConfigMismatchError(
                f"Seed entity {seed!r} is not in corpus {corpus_id!r}; refusing mix"
            )

    visited: dict[str, int] = {seed: 0 for seed in seeds}
    frontier: list[str] = sorted(seeds)
    traversed: dict[str, dict[str, Any]] = {}
    evidence_best: dict[str, dict[str, Any]] = {}
    cap_hits: list[str] = []

    for depth in range(int(max_hops)):
        _check_deadline()
        current = sorted([eid for eid in frontier if visited.get(eid) == depth])
        if not current:
            break
        next_frontier: list[str] = []
        for entity_id in current:
            _check_deadline()
            edges, truncated = fetch_bounded_neighbors(
                driver, database, entity_id, corpus_id, int(max_edges_per_node)
            )
            if truncated:
                cap_hits.append(f"max_edges_per_node_hit:{entity_id}")
            for edge in edges:
                rel_id = str(edge["id"])
                if rel_id not in traversed:
                    traversed[rel_id] = {
                        "id": rel_id,
                        "source": str(edge["source"]),
                        "target": str(edge["target"]),
                        "type": str(edge["type"]),
                        "direction_from": str(edge["direction"]),
                        "queried_node": entity_id,
                        "hop": depth + 1,
                        "evidence_chunk_ids": sorted(
                            [str(v) for v in list(edge.get("evidence") or [])]
                        ),
                    }
                for chunk_id in traversed[rel_id]["evidence_chunk_ids"]:
                    dist = depth + 1
                    slot = evidence_best.get(chunk_id)
                    candidate = {"distance": dist, "path_ids": [rel_id]}
                    if slot is None or (dist, rel_id) < (slot["distance"], slot["path_ids"][0]):
                        evidence_best[chunk_id] = candidate
                neighbor = str(edge["neighbor"])
                if neighbor not in visited:
                    if len(visited) >= int(max_visited_entities):
                        if "max_visited_entities_hit" not in cap_hits:
                            cap_hits.append("max_visited_entities_hit")
                        continue
                    visited[neighbor] = depth + 1
                    next_frontier.append(neighbor)
        frontier = sorted(set([*frontier, *next_frontier]))
        _check_deadline()

    _check_deadline()
    mention_rows = fetch_chunks_mentioning_entities(
        driver, database, sorted(visited), corpus_id, limit=5000
    )
    mention_best: dict[str, dict[str, Any]] = {}
    for row in mention_rows:
        chunk_id = str(row["chunk_id"])
        entity_id = str(row["entity_id"])
        if chunk_id in evidence_best:
            continue
        dist = int(visited.get(entity_id, 99))
        slot = mention_best.get(chunk_id)
        if slot is None or (dist, entity_id) < (slot["distance"], slot["via_entity"]):
            mention_best[chunk_id] = {"distance": dist, "via_entity": entity_id}

    ordered_evidence = sorted(evidence_best, key=lambda c: (evidence_best[c]["distance"], c))
    ordered_mentions = sorted(mention_best, key=lambda c: (mention_best[c]["distance"], c))
    prioritized = [*ordered_evidence, *[c for c in ordered_mentions if c not in evidence_best]]
    truncated_chunks = False
    if len(prioritized) > int(max_graph_chunks):
        truncated_chunks = True
        cap_hits.append("max_graph_chunks_hit")
        prioritized = prioritized[: int(max_graph_chunks)]

    evidence_ids = [c for c in ordered_evidence if c in prioritized]
    mention_ids = [c for c in prioritized if c not in set(evidence_ids)]

    return {
        "seed_ids": sorted(seeds),
        "visited": dict(sorted(visited.items())),
        "visited_entities": sorted(visited),
        "traversed_relations": sorted(traversed.values(), key=lambda r: r["id"]),
        "evidence_chunk_ids": evidence_ids,
        "mention_chunk_ids": mention_ids,
        "graph_chunk_ids": list(prioritized),
        "chunk_distance": {
            cid: (
                evidence_best[cid]["distance"]
                if cid in evidence_best
                else mention_best[cid]["distance"]
            )
            for cid in prioritized
        },
        "chunk_paths": {
            cid: list(evidence_best[cid]["path_ids"])
            if cid in evidence_best
            else [f"mentions:{mention_best[cid]['via_entity']}"]
            for cid in prioritized
        },
        "chunk_kind": {
            cid: (
                "both_evidence_mention"
                if (cid in evidence_best and cid in mention_best)
                else ("direct" if cid in evidence_best else "mention")
            )
            for cid in prioritized
        },
        "cap_hits": list(cap_hits),
        "truncated_chunks": truncated_chunks,
        "max_hops": int(max_hops),
    }
