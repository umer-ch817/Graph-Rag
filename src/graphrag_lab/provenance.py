"""Provenance verification for T03 (corpus).

Every chunk must round-trip to its source paragraph title and sentence
indices. Variants that share a normalized title but differ in sentence
content keep distinct paragraph ids and therefore distinct chunks. Gold
supporting facts must resolve to stored sentences, and after chunking,
to at least one chunk covering each gold sentence.
"""

from __future__ import annotations

from typing import Any

from graphrag_lab.chunking import ChunkConfig, ChunkRecord, build_chunk_id, count_chunk_tokens


def index_paragraphs(paragraphs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map paragraph id to paragraph, rejecting duplicates."""
    by_id: dict[str, dict[str, Any]] = {}
    for paragraph in paragraphs:
        paragraph_id = paragraph.get("id")
        if not isinstance(paragraph_id, str) or not paragraph_id:
            raise ValueError("Every paragraph must have a nonempty string id")
        if paragraph_id in by_id:
            raise ValueError(f"Duplicate paragraph id: {paragraph_id}")
        by_id[paragraph_id] = paragraph
    return by_id


def verify_chunk(
    chunk: ChunkRecord | dict[str, Any],
    paragraphs_by_id: dict[str, dict[str, Any]],
    config: ChunkConfig | None = None,
) -> None:
    """Raise ValueError when a chunk does not round-trip to its paragraph."""
    record = chunk if isinstance(chunk, ChunkRecord) else ChunkRecord.model_validate(chunk)
    paragraph = paragraphs_by_id.get(record.paragraph_id)
    if paragraph is None:
        raise ValueError(f"Chunk {record.id} references unknown paragraph {record.paragraph_id}")
    title = paragraph.get("title")
    sentences = paragraph.get("sentences")
    if title != record.title:
        raise ValueError(
            f"Chunk {record.id} title {record.title!r} does not match "
            f"paragraph {record.paragraph_id} title {title!r}"
        )
    if not isinstance(sentences, list):
        raise ValueError(f"Paragraph {record.paragraph_id} has no sentence list")
    if not record.sentence_ids:
        raise ValueError(f"Chunk {record.id} must reference at least one sentence")
    if sorted(record.sentence_ids) != list(record.sentence_ids):
        raise ValueError(f"Chunk {record.id} sentence_ids must be sorted")
    if len(record.sentence_ids) != len(record.sentence_texts):
        raise ValueError(f"Chunk {record.id} sentence_ids/texts length mismatch")
    expected_texts: list[str] = []
    for position, sentence_id in enumerate(record.sentence_ids):
        if not isinstance(sentence_id, int) or sentence_id < 0 or sentence_id >= len(sentences):
            raise ValueError(f"Chunk {record.id} references out-of-range sentence {sentence_id}")
        if sentences[sentence_id] != record.sentence_texts[position]:
            raise ValueError(
                f"Chunk {record.id} sentence {sentence_id} text does not match paragraph"
            )
        expected_texts.append(sentences[sentence_id])
    # Contiguity: overlap is allowed, gaps are not.
    for left, right in zip(record.sentence_ids, record.sentence_ids[1:]):
        if right != left + 1:
            raise ValueError(f"Chunk {record.id} sentence_ids must be contiguous")
    if record.text != " ".join(expected_texts):
        raise ValueError(f"Chunk {record.id} text does not join its sentence texts")
    expected_tokens = count_chunk_tokens(record.title, record.text)
    if record.token_count != expected_tokens:
        raise ValueError(
            f"Chunk {record.id} stores token_count={record.token_count}, "
            f"recomputed {expected_tokens}"
        )
    cfg = config or ChunkConfig()
    if record.token_count > cfg.max_tokens:
        raise ValueError(
            f"Chunk {record.id} exceeds max_tokens={cfg.max_tokens} "
            f"({record.token_count}); refusing silent overflow"
        )
    start, end = record.sentence_ids[0], record.sentence_ids[-1] + 1
    expected_id = build_chunk_id(
        record.corpus_id, record.paragraph_id, start, end, record.chunker_version
    )
    if record.id != expected_id:
        raise ValueError(f"Chunk id {record.id} does not match corpus/paragraph/span/version")


def verify_chunks(
    chunks: list[ChunkRecord | dict[str, Any]],
    paragraphs: list[dict[str, Any]] | dict[str, dict[str, Any]],
    config: ChunkConfig | None = None,
) -> None:
    """Verify every chunk and reject duplicate chunk ids."""
    by_id = paragraphs if isinstance(paragraphs, dict) else index_paragraphs(paragraphs)
    seen: set[str] = set()
    for chunk in chunks:
        record = chunk if isinstance(chunk, ChunkRecord) else ChunkRecord.model_validate(chunk)
        if record.id in seen:
            raise ValueError(f"Duplicate chunk id: {record.id}")
        seen.add(record.id)
        verify_chunk(record, by_id, config)


def unresolved_supporting_facts(
    labels: list[dict[str, Any]],
    paragraphs_by_id: dict[str, dict[str, Any]],
    paragraph_id_by_title_variant: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return gold facts that do not resolve to a stored sentence."""
    unresolved: list[dict[str, Any]] = []
    for label in labels:
        question_id = label.get("question_id", "?")
        for fact in label.get("supporting_facts", []):
            title = fact.get("title")
            sentence_id = fact.get("sentence_id")
            paragraph_id = fact.get("paragraph_id")
            paragraph = paragraphs_by_id.get(paragraph_id) if paragraph_id else None
            if paragraph is None and paragraph_id_by_title_variant is not None and title:
                fallback = paragraph_id_by_title_variant.get(str(title))
                paragraph = paragraphs_by_id.get(fallback) if fallback else None
            sentences = paragraph.get("sentences") if isinstance(paragraph, dict) else None
            valid = (
                isinstance(paragraph, dict)
                and isinstance(sentences, list)
                and isinstance(sentence_id, int)
                and 0 <= sentence_id < len(sentences)
            )
            if not valid:
                unresolved.append(
                    {"question_id": question_id, "title": title, "sentence_id": sentence_id}
                )
    return unresolved


def verify_supporting_facts(
    labels: list[dict[str, Any]],
    paragraphs: list[dict[str, Any]] | dict[str, dict[str, Any]],
) -> None:
    """Raise when any gold supporting fact misses its stored sentence (C1)."""
    by_id = paragraphs if isinstance(paragraphs, dict) else index_paragraphs(paragraphs)
    missing = unresolved_supporting_facts(labels, by_id)
    if missing:
        raise ValueError(f"{len(missing)} supporting facts do not resolve: {missing[:3]}")


def find_chunks_for_sentence(
    chunks: list[ChunkRecord | dict[str, Any]],
    paragraph_id: str,
    sentence_id: int,
) -> list[ChunkRecord]:
    """Return chunks covering one (paragraph, sentence) pair, in chunk-id order."""
    covering: list[ChunkRecord] = []
    for chunk in chunks:
        record = chunk if isinstance(chunk, ChunkRecord) else ChunkRecord.model_validate(chunk)
        if record.paragraph_id == paragraph_id and sentence_id in record.sentence_ids:
            covering.append(record)
    return sorted(covering, key=lambda record: record.id)


def verify_gold_covered_by_chunks(
    labels: list[dict[str, Any]],
    chunks: list[ChunkRecord | dict[str, Any]],
) -> None:
    """Raise when any gold sentence is missing from all chunks (C1)."""
    missing: list[dict[str, Any]] = []
    for label in labels:
        question_id = label.get("question_id", "?")
        for fact in label.get("supporting_facts", []):
            paragraph_id = fact.get("paragraph_id")
            sentence_id = fact.get("sentence_id")
            if not find_chunks_for_sentence(chunks, paragraph_id, sentence_id):
                missing.append(
                    {
                        "question_id": question_id,
                        "paragraph_id": paragraph_id,
                        "sentence_id": sentence_id,
                    }
                )
    if missing:
        raise ValueError(f"{len(missing)} gold sentences have no covering chunk: {missing[:3]}")
