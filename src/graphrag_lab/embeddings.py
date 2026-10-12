"""Embedding backend for T04 (vector-qa).

Title plus chunk text is embedded once; query text is embedded separately.
All stored/query vectors are L2-normalized and validated (finite values,
expected dimension, nonzero norm). The embedding dimension is read from the
loaded model, never assumed, and validated against the Neo4j index.

The offline chunker (T03) uses a conservative estimator; ingestion must
re-validate every chunk with the pinned model tokenizer and refuse inputs
that exceed the model limit instead of truncating silently.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from graphrag_lab.corpus import sha256_hex

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

NORMALIZE_TOLERANCE = 1e-3


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class EmbeddingConfig(Record):
    """Pinned embedding settings (R03)."""

    model: str = Field(default=DEFAULT_EMBEDDING_MODEL, min_length=1)
    normalize: bool = True


def build_embedding_text(title: str, text: str) -> str:
    """Deterministic embedding input for a chunk: title plus text."""
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Chunk title must be a nonempty string")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Chunk text must be a nonempty string")
    return f"{title.strip()}\n{text.strip()}"


def build_index_fingerprint(
    corpus_id: str,
    embedding_model: str,
    dimension: int,
    chunker_version: str,
) -> str:
    """Fingerprint isolating one active corpus/index namespace."""
    if not corpus_id or not embedding_model or not chunker_version:
        raise ValueError("Fingerprint inputs must be nonempty strings")
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("Embedding dimension must be a positive integer")
    return sha256_hex([corpus_id, embedding_model, dimension, chunker_version])


def _require_finite(values: Sequence[float], *, what: str) -> None:
    for position, value in enumerate(values):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{what} position {position} is not a number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{what} position {position} is not finite")


def validate_embedding(
    vector: Sequence[float], expected_dim: int, *, what: str = "embedding"
) -> None:
    """Reject wrong-dimension, non-finite, or zero-norm vectors."""
    if not isinstance(expected_dim, int) or expected_dim <= 0:
        raise ValueError("expected_dim must be a positive integer")
    values = list(vector)
    if len(values) != expected_dim:
        raise ValueError(f"{what} dimension {len(values)} != expected {expected_dim}")
    _require_finite(values, what=what)
    norm = math.sqrt(sum(float(v) * float(v) for v in values))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError(f"{what} has non-positive norm; refusing zero vector")


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Return the L2-normalized copy of a finite nonzero vector."""
    values = [float(v) for v in vector]
    _require_finite(values, what="embedding")
    norm = math.sqrt(sum(v * v for v in values))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Cannot normalize a zero or non-finite vector")
    return [v / norm for v in values]


def is_normalized(vector: Sequence[float], *, tolerance: float = NORMALIZE_TOLERANCE) -> bool:
    values = [float(v) for v in vector]
    norm = math.sqrt(sum(v * v for v in values))
    return math.isfinite(norm) and abs(norm - 1.0) <= tolerance


def ensure_normalized(vector: Sequence[float]) -> list[float]:
    """Normalize consistently; already-normalized inputs are re-normalized deterministically."""
    return l2_normalize(vector)


class EmbeddingBackend(Protocol):
    """Minimal interface used by ingestion and retrieval."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def max_seq_length(self) -> int: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def token_length(self, text: str) -> int: ...


class SentenceTransformerBackend:
    """Lazy `sentence-transformers` backend; model reads supply dimension/limits."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a nonempty string")
        self._model_name = model_name.strip()
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for embeddings; install project dependencies"
                ) from exc
            # CPU-first local inference; explicit device avoids CUDA surprises.
            self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model

    @property
    def dimension(self) -> int:
        dim = int(self._load().get_sentence_embedding_dimension())
        if dim <= 0:
            raise ValueError(f"Model {self._model_name} reported invalid dimension {dim}")
        return dim

    @property
    def max_seq_length(self) -> int:
        limit = int(self._load().max_seq_length)
        if limit <= 0:
            raise ValueError(f"Model {self._model_name} reported invalid max_seq_length {limit}")
        return limit

    def token_length(self, text: str) -> int:
        """Token count with the pinned model tokenizer, including special tokens."""
        if not isinstance(text, str) or not text:
            raise ValueError("token_length input must be a nonempty string")
        model = self._load()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(f"Model {self._model_name} exposes no tokenizer")
        return int(len(tokenizer.encode(text, add_special_tokens=True)))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list) or not texts:
            raise ValueError("texts must be a nonempty list")
        for index, item in enumerate(texts):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"text {index} must be a nonempty string")
        model = self._load()
        expected = self.dimension
        limit = self.max_seq_length
        for index, item in enumerate(texts):
            if self.token_length(item) > limit:
                raise ValueError(
                    f"text {index} needs {self.token_length(item)} tokens, exceeding "
                    f"model {self._model_name} limit {limit}; refusing to truncate"
                )
        vectors = model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        result: list[list[float]] = []
        for index, row in enumerate(vectors):
            values = [float(v) for v in list(row)]
            validate_embedding(values, expected, what=f"embedding {index}")
            # Re-normalize deterministically; model already normalizes but floating
            # error and backend changes must not drift stored values.
            result.append(ensure_normalized(values))
        return result


def validate_chunk_fits_model(
    title: str,
    text: str,
    backend: EmbeddingBackend,
) -> int:
    """Raise when title+text exceeds the model input limit; return token length."""
    candidate = build_embedding_text(title, text)
    length = backend.token_length(candidate)
    if length > backend.max_seq_length:
        raise ValueError(
            f"Chunk ({length} model tokens) exceeds {backend.model_name} limit "
            f"{backend.max_seq_length}; refusing to truncate"
        )
    return length
