"""Cross-encoder reranking and controlled experiment arms for T11 (hybrid-qa, R08).

Arms (TRD evaluation):
  A ``vector`` — vector candidates, no reranker, no triples.
  B ``vector_rerank`` — vector candidates + same reranker, no triples.
  C ``hybrid_rerank`` — vector+graph merge + same reranker, no triples.
  D ``hybrid_rerank_triples`` — C plus at most 12 evidence-backed triples.

Fairness guarantees:
  * Duplicate chunk ids are removed before reranking (first occurrence wins).
  * Raw score scales are never added: RRF uses ranks only (T10) and the
    reranker reorders by its own scores only; vector/graph scores are kept
    separately for audit and never summed with reranker logits.
  * Final chunk cap (8) and total context budget (3,000 tokens) are identical
    across arms; triple tokens count against the same budget (see
    ``answering.pack_context_with_triples``).
  * Triples are evidence-only: a triple is included only when at least one of
    its evidence chunks survives final packing, and only surviving evidence
    is sent. Paths whose evidence was dropped are never sent.
  * Reranker input lengths are checked; overlong pairs are truncated and the
    truncation is recorded. Reranker failure falls back to the deterministic
    pre-rerank order with a degraded flag (never silently omitted).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from graphrag_lab.chunking import estimate_tokens

RERANK_VERSION = "rerank-v1"
RERANKER_MODEL_DEFAULT = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_BATCH_SIZE = 32
MAX_TRIPLES = 12

ExperimentArm = Literal["vector", "vector_rerank", "hybrid_rerank", "hybrid_rerank_triples"]
ARMS: tuple[str, ...] = ("vector", "vector_rerank", "hybrid_rerank", "hybrid_rerank_triples")

ARM_LABELS: dict[str, str] = {
    "vector": "A vector",
    "vector_rerank": "B vector+reranker",
    "hybrid_rerank": "C vector+graph+same reranker without triples",
    "hybrid_rerank_triples": "D C+triples",
}


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class RerankConfig(Record):
    """Tunable cross-encoder budgets (R07/R08)."""

    model: str = Field(default=RERANKER_MODEL_DEFAULT, min_length=1)
    max_seq_length: int = Field(default=DEFAULT_MAX_SEQ_LENGTH, ge=1, strict=True)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1, le=256, strict=True)
    rerank_version: str = Field(default=RERANK_VERSION, min_length=1)


class EvidenceTriple(Record):
    """One evidence-backed triple for the optional triples view (arm D)."""

    relation_id: str = Field(min_length=1)
    source_entity_id: str = Field(min_length=1)
    target_entity_id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    evidence_chunk_ids: list[str] = Field(min_length=1)


class RerankResult(Record):
    """Deterministic rerank outcome; raw scales kept separate, never summed."""

    question: str = Field(min_length=1)
    ordered_chunk_ids: list[str]
    rerank_scores: dict[str, float]
    pre_rank: dict[str, int]
    degraded: bool = False
    degraded_reason: str | None = None
    truncated_chunk_ids: list[str] = Field(default_factory=list)
    removed_duplicate_ids: list[str] = Field(default_factory=list)
    reranker_model: str = Field(min_length=1)
    rerank_version: str = Field(min_length=1)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class RerankerError(RuntimeError):
    code = "PROVIDER_ERROR"

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class RerankInputError(RerankerError, ValueError):
    code = "INVALID_INPUT"


def validate_arm(arm: str) -> ExperimentArm:
    """Raise INVALID_INPUT unless ``arm`` is one of A-D."""
    if arm not in ARMS:
        raise RerankInputError(
            f"Unknown experiment arm {arm!r}; expected one of {list(ARMS)} "
            f"({', '.join(ARM_LABELS[a] for a in ARMS)})"
        )
    return arm  # type: ignore[return-value]


def _chunk_id_of(item: Any) -> str:
    if isinstance(item, dict):
        cid = item.get("chunk_id", item.get("id"))
    else:
        cid = getattr(item, "chunk_id", getattr(item, "id", None))
    if not isinstance(cid, str) or not cid.strip():
        raise RerankInputError("Every candidate chunk needs a nonempty chunk id")
    return cid


def deduplicate_chunks(chunks: list[Any]) -> tuple[list[Any], list[str]]:
    """Remove duplicate chunk ids, keeping the first occurrence.

    Returns (deduped, removed_ids). Order is first-occurrence order; the
    caller sorts deterministically afterwards (RRF order or rerank order).
    """
    if not isinstance(chunks, list):
        raise RerankInputError("chunks must be a list")
    seen: set[str] = set()
    deduped: list[Any] = []
    removed: list[str] = []
    for item in chunks:
        cid = _chunk_id_of(item)
        if cid in seen:
            if cid not in removed:
                removed.append(cid)
            continue
        seen.add(cid)
        deduped.append(item)
    return deduped, sorted(removed)


def build_rerank_text(title: str, text: str) -> str:
    """Deterministic reranker passage input: title plus text."""
    if not isinstance(title, str) or not title.strip():
        raise RerankInputError("Chunk title must be a nonempty string")
    if not isinstance(text, str) or not text.strip():
        raise RerankInputError("Chunk text must be a nonempty string")
    return f"{title.strip()}\n{text.strip()}"


def estimate_pair_tokens(question: str, passage: str) -> int:
    """Conservative offline estimate for one (question, passage) pair."""
    if not isinstance(question, str) or not question.strip():
        raise RerankInputError("question must be a nonempty string")
    if not isinstance(passage, str) or not passage.strip():
        raise RerankInputError("passage must be a nonempty string")
    # Separator/special-token overhead mirrors the chunk estimator (+3).
    return estimate_tokens(question) + estimate_tokens(passage) + 3


def truncate_text_to_budget(text: str, budget_tokens: int) -> str:
    """Truncate text (word boundary) so its estimate fits ``budget_tokens``."""
    if not isinstance(text, str) or not text.strip():
        raise RerankInputError("text must be a nonempty string")
    if not isinstance(budget_tokens, int) or budget_tokens < 1:
        raise RerankInputError("budget_tokens must be a positive integer")
    if estimate_tokens(text) <= budget_tokens:
        return text
    words = text.split()
    if len(words) <= 1:
        # Single long token: hard cut by characters (4 chars ~ 1 token).
        chars = max(1, budget_tokens * 4)
        return text[:chars]
    low, high = 1, len(words)
    best = 1
    while low <= high:
        mid = (low + high) // 2
        if estimate_tokens(" ".join(words[:mid])) <= budget_tokens:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return " ".join(words[:best])


def _coerce_chunk_view(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        title = item.get("title")
        text = item.get("text")
    else:
        title = getattr(item, "title", None)
        text = getattr(item, "text", None)
    cid = _chunk_id_of(item)
    if not isinstance(title, str) or not title.strip():
        raise RerankInputError(f"Chunk {cid!r} has no title")
    if not isinstance(text, str) or not text.strip():
        raise RerankInputError(f"Chunk {cid!r} has no text")
    return {"chunk_id": cid, "title": title, "text": text, "raw": item}


def rerank_candidates(
    question: str,
    chunks: list[Any],
    backend: Any,
    config: RerankConfig | None = None,
) -> RerankResult:
    """Rank deduped candidates with the cross-encoder (deterministic).

    The reranker reorders by its own scores only; input vector/graph scores
    are never added to reranker logits. Ties break by chunk id. Overlong
    pairs are truncated and recorded. Backend/provider failure degrades to
    the deterministic pre-rerank order with ``degraded=True`` instead of
    raising; invalid inputs still raise.
    """
    cfg = config or RerankConfig()
    if not isinstance(question, str) or not question.strip():
        raise RerankInputError("question must be a nonempty string")
    cleaned = question.strip()
    if not isinstance(chunks, list):
        raise RerankInputError("chunks must be a list")
    if backend is None:
        raise RerankInputError("reranker backend must be supplied")
    model_name = str(getattr(backend, "model_name", cfg.model) or cfg.model).strip()
    if not model_name:
        raise RerankInputError("reranker model_name must be nonempty")

    deduped, removed = deduplicate_chunks(list(chunks))
    pre_rank = {_chunk_id_of(item): index + 1 for index, item in enumerate(deduped)}
    if not deduped:
        return RerankResult(
            question=cleaned,
            ordered_chunk_ids=[],
            rerank_scores={},
            pre_rank={},
            degraded=False,
            degraded_reason=None,
            truncated_chunk_ids=[],
            removed_duplicate_ids=removed,
            reranker_model=model_name,
            rerank_version=cfg.rerank_version,
            diagnostics={
                "rerank_version": cfg.rerank_version,
                "n_input": len(chunks),
                "n_deduped": 0,
                "n_removed_duplicates": len(removed),
                "truncated": [],
                "degraded": False,
            },
        )

    views = [_coerce_chunk_view(item) for item in deduped]
    question_tokens = estimate_tokens(cleaned)
    passages: list[str] = []
    truncated: list[str] = []
    for view in views:
        passage = build_rerank_text(view["title"], view["text"])
        pair = question_tokens + estimate_tokens(passage) + 3
        if pair > cfg.max_seq_length:
            budget = max(1, cfg.max_seq_length - question_tokens - 3)
            passage = truncate_text_to_budget(passage, budget)
            truncated.append(view["chunk_id"])
        passages.append(passage)

    try:
        raw_scores = backend.rank_pairs(cleaned, passages)
    except RerankInputError:
        raise
    except ValueError as exc:
        # Invalid inputs (e.g. bad question shape) propagate; only
        # provider/backend failures degrade.
        from graphrag_lab.llm import InvalidInputError as LlmInvalid

        if isinstance(exc, LlmInvalid):
            raise RerankInputError(str(exc)) from exc
        degraded = _degraded_result(
            cleaned,
            deduped,
            pre_rank,
            model_name,
            cfg,
            removed,
            truncated,
            f"{type(exc).__name__}: {str(exc)[:300]}",
        )
        return degraded
    except Exception as exc:  # noqa: BLE001 - reranker failure must degrade visibly
        return _degraded_result(
            cleaned,
            deduped,
            pre_rank,
            model_name,
            cfg,
            removed,
            truncated,
            f"{type(exc).__name__}: {str(exc)[:300]}",
        )

    if not isinstance(raw_scores, list) or len(raw_scores) != len(views):
        count = len(raw_scores) if isinstance(raw_scores, list) else type(raw_scores).__name__
        return _degraded_result(
            cleaned,
            deduped,
            pre_rank,
            model_name,
            cfg,
            removed,
            truncated,
            f"Reranker returned {count} scores for {len(views)} passages",
        )
    scores: dict[str, float] = {}
    for view, value in zip(views, raw_scores):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return _degraded_result(
                cleaned,
                deduped,
                pre_rank,
                model_name,
                cfg,
                removed,
                truncated,
                f"Non-numeric reranker score for chunk {view['chunk_id']!r}",
            )
        if not math.isfinite(number):
            return _degraded_result(
                cleaned,
                deduped,
                pre_rank,
                model_name,
                cfg,
                removed,
                truncated,
                f"Non-finite reranker score for chunk {view['chunk_id']!r}",
            )
        scores[view["chunk_id"]] = number

    ordered = sorted(scores, key=lambda c: (-scores[c], c))
    return RerankResult(
        question=cleaned,
        ordered_chunk_ids=ordered,
        rerank_scores=dict(scores),
        pre_rank=dict(pre_rank),
        degraded=False,
        degraded_reason=None,
        truncated_chunk_ids=sorted(truncated),
        removed_duplicate_ids=removed,
        reranker_model=model_name,
        rerank_version=cfg.rerank_version,
        diagnostics={
            "rerank_version": cfg.rerank_version,
            "n_input": len(chunks),
            "n_deduped": len(deduped),
            "n_removed_duplicates": len(removed),
            "truncated": sorted(truncated),
            "degraded": False,
        },
    )


def _degraded_result(
    question: str,
    deduped: list[Any],
    pre_rank: dict[str, int],
    model_name: str,
    cfg: RerankConfig,
    removed: list[str],
    truncated: list[str],
    reason: str,
) -> RerankResult:
    ordered = [_chunk_id_of(item) for item in deduped]
    return RerankResult(
        question=question,
        ordered_chunk_ids=ordered,
        rerank_scores={},
        pre_rank=dict(pre_rank),
        degraded=True,
        degraded_reason=reason,
        truncated_chunk_ids=sorted(truncated),
        removed_duplicate_ids=list(removed),
        reranker_model=model_name,
        rerank_version=cfg.rerank_version,
        diagnostics={
            "rerank_version": cfg.rerank_version,
            "n_input": len(pre_rank) + len(removed),
            "n_deduped": len(deduped),
            "n_removed_duplicates": len(removed),
            "truncated": sorted(truncated),
            "degraded": True,
            "degraded_reason": reason,
        },
    )


def collect_evidence_triples(
    traversed_relations: list[dict[str, Any]],
    packed_chunk_ids: list[str] | set[str],
    max_triples: int = MAX_TRIPLES,
) -> tuple[list[EvidenceTriple], dict[str, Any]]:
    """Filter traversed relations to evidence-only triples for arm D.

    A triple survives only when at least one evidence chunk is in the final
    packed set; only surviving evidence ids are sent. Relations are scanned
    in deterministic relation-id order and capped at ``max_triples`` (12).
    """
    if not isinstance(max_triples, int) or isinstance(max_triples, bool) or max_triples < 0:
        raise RerankInputError("max_triples must be a non-negative integer")
    if max_triples > MAX_TRIPLES:
        raise RerankInputError(f"max_triples must not exceed {MAX_TRIPLES}")
    packed = set(packed_chunk_ids) if not isinstance(packed_chunk_ids, set) else packed_chunk_ids
    for cid in packed:
        if not isinstance(cid, str) or not cid:
            raise RerankInputError("packed chunk ids must be nonempty strings")
    relations = list(traversed_relations or [])
    for index, rel in enumerate(relations):
        if not isinstance(rel, dict):
            raise RerankInputError(f"Traversed relation {index} must be a mapping")
        for key in ("id", "source", "target", "type", "evidence_chunk_ids"):
            alt = {
                "id": ("id", "relation_id"),
                "source": ("source", "source_entity_id"),
                "target": ("target", "target_entity_id"),
            }.get(key, (key,))
            if not any(k in rel and isinstance(rel[k], (str, list)) for k in alt):
                raise RerankInputError(f"Traversed relation {index} is missing {key!r}")

    def _rel_id(rel: dict[str, Any]) -> str:
        return str(rel.get("id", rel.get("relation_id")))

    ordered = sorted(relations, key=_rel_id)
    triples: list[EvidenceTriple] = []
    skipped_no_evidence = 0
    for rel in ordered:
        rel_id = _rel_id(rel)
        source = str(rel.get("source", rel.get("source_entity_id")))
        target = str(rel.get("target", rel.get("target_entity_id")))
        rel_type = rel.get("type")
        rel_type_str = rel_type.value if hasattr(rel_type, "value") else str(rel_type)
        raw_evidence = rel.get("evidence_chunk_ids", rel.get("evidence", [])) or []
        evidence = sorted({str(c) for c in list(raw_evidence) if isinstance(c, str) and c})
        surviving = sorted(set(evidence) & packed)
        if not surviving:
            skipped_no_evidence += 1
            continue
        triples.append(
            EvidenceTriple(
                relation_id=rel_id,
                source_entity_id=source,
                target_entity_id=target,
                type=rel_type_str,
                evidence_chunk_ids=surviving,
            )
        )
    total_eligible = len(triples)
    dropped_by_cap = max(0, total_eligible - max_triples)
    kept = triples[:max_triples]
    diagnostics = {
        "n_traversed": len(relations),
        "n_eligible": total_eligible,
        "n_kept": len(kept),
        "n_skipped_no_surviving_evidence": skipped_no_evidence,
        "n_dropped_by_cap": dropped_by_cap,
        "max_triples": max_triples,
    }
    return kept, diagnostics


def compatibility_hash(
    *,
    corpus_id: str,
    index_fingerprint: str | None,
    question_ids: list[str],
    embedding_model: str,
    answer_model: str,
    prompt_version: str,
    max_chunks: int,
    max_context_tokens: int,
) -> str:
    """Hash the fields that must stay fixed for a fair paired comparison."""
    if not corpus_id or not corpus_id.strip():
        raise RerankInputError("corpus_id must be nonempty")
    if not question_ids:
        raise RerankInputError("question_ids must be nonempty")
    canonical = json.dumps(
        {
            "corpus_id": corpus_id,
            "index_fingerprint": index_fingerprint,
            "question_ids": sorted(question_ids),
            "embedding_model": embedding_model,
            "answer_model": answer_model,
            "prompt_version": prompt_version,
            "max_chunks": max_chunks,
            "max_context_tokens": max_context_tokens,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def check_arm_compatibility(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Check that arm manifests share the fixed comparison fields.

    Strategy, reranker presence, graph fingerprint, triple inclusion and
    retrieval counts/tokens are allowed to differ (they are the experiment);
    corpus, index fingerprint, question set, embedding/answer models and final
    budgets must match. Returns {compatible, reason, compatibility_hash}.
    """
    if not isinstance(manifests, list) or not manifests:
        raise RerankInputError("manifests must be a nonempty list")
    fixed_keys = (
        "corpus_id",
        "index_fingerprint",
        "question_set_hash",
        "embedding_model",
        "answer_model",
        "prompt_version",
        "max_chunks",
        "max_context_tokens",
    )
    first = manifests[0]
    for key in fixed_keys:
        if key not in first:
            raise RerankInputError(f"Manifest is missing fixed field {key!r}")
    reference = {key: first[key] for key in fixed_keys}
    for index, manifest in enumerate(manifests[1:], start=1):
        for key in fixed_keys:
            if manifest.get(key) != reference[key]:
                return {
                    "compatible": False,
                    "reason": f"Arm {manifest.get('strategy', index)!r} differs on {key}: "
                    f"{manifest.get(key)!r} != {reference[key]!r}; refusing unlabeled paired delta",
                    "compatibility_hash": None,
                    "fixed": dict(reference),
                }
    payload = json.dumps(reference, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "compatible": True,
        "reason": "fixed fields match across arms",
        "compatibility_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "fixed": dict(reference),
    }


def build_arm_manifest(
    *,
    strategy: str,
    corpus_id: str,
    index_fingerprint: str | None,
    graph_fingerprint: str | None,
    embedding_model: str,
    answer_model: str,
    reranker_model: str | None,
    prompt_version: str,
    max_chunks: int,
    max_context_tokens: int,
    max_triples: int,
    include_triples: bool,
    num_candidates: int,
    num_merged: int,
    num_packed: int,
    total_tokens: int,
    triple_tokens: int,
    n_triples: int,
    degraded: bool,
    fallback: bool,
    fallback_reason: str | None = None,
    question_ids: list[str] | None = None,
    question_set_hash: str | None = None,
) -> dict[str, Any]:
    """Build one comparable arm manifest (no secrets, no raw-score sums)."""
    arm = validate_arm(strategy)
    manifest: dict[str, Any] = {
        "strategy": arm,
        "arm_label": ARM_LABELS[arm],
        "corpus_id": corpus_id,
        "index_fingerprint": index_fingerprint,
        "graph_fingerprint": graph_fingerprint,
        "embedding_model": embedding_model,
        "answer_model": answer_model,
        "reranker_model": reranker_model,
        "rerank_version": RERANK_VERSION,
        "prompt_version": prompt_version,
        "max_chunks": int(max_chunks),
        "max_context_tokens": int(max_context_tokens),
        "max_triples": int(max_triples),
        "include_triples": bool(include_triples),
        "num_candidates": int(num_candidates),
        "num_merged": int(num_merged),
        "num_packed": int(num_packed),
        "total_tokens": int(total_tokens),
        "triple_tokens": int(triple_tokens),
        "n_triples": int(n_triples),
        "degraded": bool(degraded),
        "fallback": bool(fallback),
        "fallback_reason": fallback_reason,
        "rerank_version_label": RERANK_VERSION,
    }
    if question_ids is not None:
        manifest["question_ids"] = sorted(question_ids)
    if question_set_hash is not None:
        manifest["question_set_hash"] = question_set_hash
    return manifest


class FakeRerankerBackend:
    """Deterministic offline reranker for unit tests.

    Scores by lexical overlap between question and passage
    (shared lowercased word count as float) unless ``fixed_scores`` is
    supplied (list aligned with the input order). ``fail=True`` raises a
    provider error to exercise the degraded fallback. Calls are recorded.
    """

    def __init__(
        self,
        *,
        model: str = "fake-reranker-test",
        fixed_scores: list[float] | None = None,
        fail: bool = False,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    ) -> None:
        if not model or not model.strip():
            raise RerankInputError("model must be a nonempty string")
        self._model = model.strip()
        self._fixed = list(fixed_scores) if fixed_scores is not None else None
        self._fail = bool(fail)
        self._max_seq = int(max_seq_length)
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def max_seq_length(self) -> int:
        return self._max_seq

    def rank_pairs(self, question: str, passages: list[str]) -> list[float]:
        if self._fail:
            raise RerankerError("Fake reranker injected failure", retryable=False)
        if not isinstance(question, str) or not question.strip():
            raise RerankInputError("question must be a nonempty string")
        if not isinstance(passages, list) or not passages:
            raise RerankInputError("passages must be a nonempty list")
        self.calls.append((question, list(passages)))
        if self._fixed is not None:
            if len(self._fixed) != len(passages):
                raise RerankInputError("fixed_scores length must match passages")
            return [float(v) for v in self._fixed]
        question_words = set(question.lower().split())
        scores: list[float] = []
        for passage in passages:
            words = set(str(passage).lower().split())
            scores.append(float(len(question_words & words)))
        return scores


class CrossEncoderBackend:
    """Lazy ``sentence-transformers`` CrossEncoder backend (CPU-first).

    Model download requires network access; unit tests use the fake backend.
    Long pairs are truncated by the caller (recorded); this backend also
    passes ``truncation=True`` to the tokenizer as defense in depth.
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_DEFAULT,
        *,
        max_seq_length: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = "cpu",
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise RerankInputError("model_name must be a nonempty string")
        self._model_name = model_name.strip()
        self._batch = int(batch_size)
        self._device = device
        self._model: Any | None = None
        self._max_seq_override = max_seq_length

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_seq_length(self) -> int:
        if self._max_seq_override is not None:
            return int(self._max_seq_override)
        model = self._load()
        for attr in ("max_seq_length", "max_length"):
            try:
                value = int(getattr(model, attr))
                if value > 0:
                    return value
            except (TypeError, ValueError, AttributeError):
                continue
        try:
            tokenizer = getattr(model, "tokenizer", None)
            length = int(getattr(tokenizer, "model_max_length", 0))
            if 0 < length < 100000:
                return length
        except (TypeError, ValueError, AttributeError):
            pass
        return DEFAULT_MAX_SEQ_LENGTH

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RerankerError(
                    "sentence-transformers is required for reranking; install project dependencies"
                ) from exc
            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    def rank_pairs(self, question: str, passages: list[str]) -> list[float]:
        if not isinstance(question, str) or not question.strip():
            raise RerankInputError("question must be a nonempty string")
        if not isinstance(passages, list) or not passages:
            raise RerankInputError("passages must be a nonempty list")
        model = self._load()
        pairs = [(question, str(p)) for p in passages]
        try:
            raw = model.predict(pairs, batch_size=self._batch, show_progress_bar=False)
        except Exception as exc:
            raise RerankerError(f"Cross-encoder inference failed: {type(exc).__name__}") from exc
        try:
            values = [float(v) for v in list(raw)]
        except (TypeError, ValueError) as exc:
            raise RerankerError("Cross-encoder returned non-numeric scores") from exc
        for value in values:
            if not math.isfinite(value):
                raise RerankerError("Cross-encoder returned a non-finite score")
        return values


__all__ = [
    "ARMS",
    "ARM_LABELS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_SEQ_LENGTH",
    "MAX_TRIPLES",
    "RERANK_VERSION",
    "RERANKER_MODEL_DEFAULT",
    "CrossEncoderBackend",
    "EvidenceTriple",
    "ExperimentArm",
    "FakeRerankerBackend",
    "RerankConfig",
    "RerankInputError",
    "RerankResult",
    "RerankerError",
    "build_arm_manifest",
    "build_rerank_text",
    "check_arm_compatibility",
    "collect_evidence_triples",
    "compatibility_hash",
    "deduplicate_chunks",
    "estimate_pair_tokens",
    "rerank_candidates",
    "truncate_text_to_budget",
    "validate_arm",
]
