"""Sentence-preserving chunking for T03 (corpus).

One paragraph per chunk by default; long paragraphs split at sentence
boundaries with one sentence overlap. Token counts use a deterministic
offline estimator and every overflow is an explicit error, never silent
truncation. Chunk ids hash corpus id, paragraph id, span and chunker
version so identical preparation produces identical ids.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from graphrag_lab.corpus import sha256_hex

CHUNKER_VERSION = "sent-overlap-v1"
TOKEN_ESTIMATOR = "char-word-max-v1"
CHUNK_ID_PREFIX = "c_"

# Special-token overhead for title + text embedding input (CLS/SEP style).
SPECIAL_TOKEN_OVERHEAD = 3


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class ChunkConfig(Record):
    """Tunable chunking budgets (R02)."""

    target_tokens: int = Field(default=180, ge=1, strict=True)
    max_tokens: int = Field(default=256, ge=1, strict=True)
    overlap_sentences: int = Field(default=1, ge=0, strict=True)
    chunker_version: str = Field(default=CHUNKER_VERSION, min_length=1)
    token_estimator: str = Field(default=TOKEN_ESTIMATOR, min_length=1)

    def model_post_init(self, _context: Any) -> None:
        if self.target_tokens > self.max_tokens:
            raise ValueError("target_tokens must not exceed max_tokens")


class ChunkRecord(Record):
    """One retrievable chunk with exact sentence provenance."""

    id: str
    corpus_id: str = Field(min_length=1)
    paragraph_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    sentence_ids: list[int] = Field(min_length=1)
    sentence_texts: list[str] = Field(min_length=1)
    token_count: int = Field(ge=1, strict=True)
    content_hash: str = Field(min_length=1)
    chunker_version: str = Field(min_length=1)


class ChunkOverflowError(ValueError):
    """A single sentence plus title exceeds the configured token budget."""


def estimate_tokens(text: str) -> int:
    """Conservative offline upper bound for English WordPiece tokenization.

    Returns ``max(words, ceil(bytes/4))``. WordPiece splits words into
    subwords, so whitespace words alone underestimate; long words alone
    underestimate when counting words. Taking the maximum is conservative
    for English Wikipedia text. It is not a guarantee for CJK or other
    scripts without spaces; T04 must re-validate with the pinned model
    tokenizer and refuse mismatches. The estimator name is stored on the
    config so later tickets can detect estimator changes.
    """
    if not text:
        return 0
    words = len(text.split())
    char_estimate = math.ceil(len(text.encode("utf-8")) / 4)
    return max(words, char_estimate, 1)


def count_chunk_tokens(title: str, text: str) -> int:
    """Title plus content tokens including special-token overhead."""
    return estimate_tokens(title) + estimate_tokens(text) + SPECIAL_TOKEN_OVERHEAD


def build_chunk_id(
    corpus_id: str,
    paragraph_id: str,
    start: int,
    end: int,
    chunker_version: str = CHUNKER_VERSION,
) -> str:
    """Deterministic chunk id scoped to corpus, paragraph and span."""
    return CHUNK_ID_PREFIX + sha256_hex([corpus_id, paragraph_id, start, end, chunker_version])


def build_content_hash(title: str, text: str) -> str:
    return sha256_hex([title, text])


def _require_paragraph_fields(paragraph: dict[str, Any]) -> tuple[str, str, list[str]]:
    try:
        paragraph_id = paragraph["id"]
        title = paragraph["title"]
        sentences = paragraph["sentences"]
    except KeyError as exc:
        raise ValueError(f"Paragraph is missing required field: {exc}") from exc
    if not isinstance(paragraph_id, str) or not paragraph_id:
        raise ValueError("Paragraph id must be a nonempty string")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Paragraph title must be a nonempty string")
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("Paragraph must contain at least one sentence")
    for index, sentence in enumerate(sentences):
        if not isinstance(sentence, str):
            raise ValueError(f"Sentence {index} must be a string")
    # Upstream HotpotQA paragraphs occasionally contain empty strings. They
    # stay in place so sentence indices (gold labels, citations, chunk spans)
    # never shift; chunks simply carry the empty text at its original index.
    if not any(sentence.strip() for sentence in sentences):
        raise ValueError("Paragraph must contain at least one nonempty sentence")
    return paragraph_id, title, list(sentences)


def chunk_paragraph(
    paragraph: dict[str, Any],
    corpus_id: str,
    config: ChunkConfig | None = None,
) -> list[ChunkRecord]:
    """Split one paragraph into sentence-preserving chunks.

    Raises ChunkOverflowError when a single sentence plus title exceeds
    ``max_tokens`` instead of truncating. Overlap sentences appear in both
    adjacent chunks and are recorded in each chunk's ``sentence_ids``.
    """
    if not isinstance(corpus_id, str) or not corpus_id:
        raise ValueError("corpus_id must be a nonempty string")
    cfg = config or ChunkConfig()
    paragraph_id, title, sentences = _require_paragraph_fields(paragraph)
    total = count_chunk_tokens(title, " ".join(sentences))
    if total <= cfg.target_tokens:
        return [_make_chunk(corpus_id, paragraph_id, title, sentences, 0, len(sentences), cfg)]
    chunks: list[ChunkRecord] = []
    start = 0
    count = len(sentences)
    while start < count:
        single = count_chunk_tokens(title, sentences[start])
        if single > cfg.max_tokens:
            raise ChunkOverflowError(
                f"Sentence {start} of paragraph {paragraph_id} needs {single} tokens "
                f"(title plus one sentence), exceeding max_tokens={cfg.max_tokens}; "
                "refusing to truncate"
            )
        end = start + 1
        while end < count:
            candidate = count_chunk_tokens(title, " ".join(sentences[start : end + 1]))
            if candidate > cfg.target_tokens:
                break
            if candidate > cfg.max_tokens:
                break
            end += 1
        chunks.append(_make_chunk(corpus_id, paragraph_id, title, sentences, start, end, cfg))
        if end >= count:
            break
        # Overlap the tail of this chunk; always advance by at least one sentence.
        start = max(end - cfg.overlap_sentences, start + 1) if cfg.overlap_sentences else end
    for chunk in chunks:
        if chunk.token_count > cfg.max_tokens:
            raise ChunkOverflowError(
                f"Chunk {chunk.id} needs {chunk.token_count} tokens, "
                f"exceeding max_tokens={cfg.max_tokens}; refusing to truncate"
            )
    return chunks


def _make_chunk(
    corpus_id: str,
    paragraph_id: str,
    title: str,
    sentences: list[str],
    start: int,
    end: int,
    config: ChunkConfig,
) -> ChunkRecord:
    span_texts = sentences[start:end]
    text = " ".join(span_texts)
    if not text.strip():
        raise ValueError("Chunk text must be nonempty")
    token_count = count_chunk_tokens(title, text)
    return ChunkRecord(
        id=build_chunk_id(corpus_id, paragraph_id, start, end, config.chunker_version),
        corpus_id=corpus_id,
        paragraph_id=paragraph_id,
        title=title,
        text=text,
        sentence_ids=list(range(start, end)),
        sentence_texts=list(span_texts),
        token_count=token_count,
        content_hash=build_content_hash(title, text),
        chunker_version=config.chunker_version,
    )


def chunk_paragraphs(
    paragraphs: list[dict[str, Any]],
    corpus_id: str,
    config: ChunkConfig | None = None,
) -> list[ChunkRecord]:
    """Chunk every paragraph in deterministic (paragraph id, span) order."""
    cfg = config or ChunkConfig()
    seen: set[str] = set()
    for paragraph in paragraphs:
        paragraph_id = paragraph.get("id") if isinstance(paragraph, dict) else None
        if paragraph_id in seen:
            raise ValueError(f"Duplicate paragraph id: {paragraph_id}")
        seen.add(str(paragraph_id))
    ordered = sorted(paragraphs, key=lambda p: str(p.get("id", "")))
    chunks: list[ChunkRecord] = []
    for paragraph in ordered:
        chunks.extend(chunk_paragraph(paragraph, corpus_id, cfg))
    chunk_ids = [chunk.id for chunk in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Chunk id collision: identical corpus/paragraph/span/version")
    return chunks


def chunk_dataset_directory(
    dataset_dir: str | Path,
    output_name: str = "chunks.jsonl",
    config: ChunkConfig | None = None,
    corpus_id: str | None = None,
) -> Path:
    """Read ``paragraphs.jsonl`` (+ ``manifest.json``) and write ``chunks.jsonl``.

    Each line of ``chunks.jsonl`` is the canonical JSON record for one chunk.
    Returns the written path. Refuses to silently overwrite a differing file.
    """
    import json

    from graphrag_lab.corpus import canonical_bytes

    cfg = config or ChunkConfig()
    source_dir = Path(dataset_dir)
    paragraphs_path = source_dir / "paragraphs.jsonl"
    if not paragraphs_path.is_file():
        raise ValueError(f"Missing paragraphs artifact: {paragraphs_path}")
    paragraphs = [
        json.loads(line)
        for line in paragraphs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resolved_corpus_id = corpus_id
    manifest_path = source_dir / "manifest.json"
    if resolved_corpus_id is None:
        if not manifest_path.is_file():
            raise ValueError("corpus_id is required when manifest.json is absent")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        resolved_corpus_id = manifest.get("corpus_id")
    if not resolved_corpus_id:
        raise ValueError("corpus_id must be a nonempty string")
    chunks = chunk_paragraphs(paragraphs, resolved_corpus_id, cfg)
    payload = b"".join(canonical_bytes(chunk.model_dump()) for chunk in chunks)
    output_path = source_dir / output_name
    if output_path.is_file():
        if output_path.read_bytes() != payload:
            raise ValueError(
                "Existing chunks artifact differs; choose a new output or rebuild deterministically"
            )
        return output_path
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.rename(output_path)
    return output_path
