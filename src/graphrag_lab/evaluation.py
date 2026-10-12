"""Baseline evaluation for T06 (evaluation).

Scores one frozen run over a fixed question denominator: every supplied
question contributes to the primary aggregate, including infrastructure
failures (zero answer score) and abstentions. Retrieval coverage is scored
twice -- over retrieval candidates and over the actual packed context --
with deduplication handled in ``scoring``. Successful-run-only diagnostics
are reported separately and never replace the fixed-denominator primary.

T11 adds controlled experiment arms (R08): ``vector`` (A), ``vector_rerank``
(B), ``hybrid_rerank`` (C) and ``hybrid_rerank_triples`` (D). Arms share the
same final chunk cap and total context budget; only the candidate source,
reranker presence and triple inclusion differ.

T17 adds the optional decomposition arm (R13, E): ``decomposition`` reuses
the same frozen fixed fields and final budgets, runs bounded sequential
sub-retrieval (T16, at most two sub-questions) and is compared on the tuning
set first. The tuning decision is frozen before any locked use; the delta
report is additive and never replaces the frozen A-D results.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from graphrag_lab.scoring import (
    SCORING_VERSION,
    RetrievalScores,
    normalize_answer,
    score_answer,
    score_retrieval,
)

EVALUATION_VERSION = "evaluation-v1"

PredictionStatus = Literal["supported", "insufficient_evidence", "failed"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class EvidenceView(Record):
    """Minimal evidence chunk view used for coverage scoring."""

    chunk_id: str = Field(min_length=1)
    paragraph_id: str | None = None
    title: str | None = None
    sentence_ids: list[int] = Field(default_factory=list)

    def model_post_init(self, _context: Any) -> None:
        if not self.paragraph_id and not self.title:
            raise ValueError("Evidence needs a paragraph_id or title")
        for value in self.sentence_ids:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("sentence_ids must be non-negative integers")


class QuestionPrediction(Record):
    """One baseline prediction with the evidence actually used."""

    question_id: str = Field(min_length=1)
    answer: str = ""
    status: PredictionStatus = "supported"
    citations: list[dict[str, Any]] = Field(default_factory=list)
    candidates: list[EvidenceView] = Field(default_factory=list)
    packed: list[EvidenceView] = Field(default_factory=list)
    latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0, strict=True)
    output_tokens: int | None = Field(default=None, ge=0, strict=True)
    estimated_cost_usd: float | None = None
    failure_code: str | None = None
    fallback: bool = False
    degraded: bool = False
    triples: list[dict[str, Any]] = Field(default_factory=list)
    reranked: bool = False
    strategy: str | None = None


class LabelView(Record):
    """Evaluation-only gold label for one question."""

    question_id: str = Field(min_length=1)
    answer: str
    supporting_facts: list[dict[str, Any]] = Field(min_length=1)
    question_type: str | None = None


class ScoredQuestion(Record):
    """Per-question scores over the fixed denominator."""

    question_id: str = Field(min_length=1)
    status: PredictionStatus
    failure_code: str | None = None
    exact_match: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    candidate: RetrievalScores
    packed: RetrievalScores
    latency_ms: float | None = None
    citations_resolve: bool = True
    abstained: bool = False
    fallback: bool = False
    degraded: bool = False


class RunConfig(Record):
    """Frozen settings identifying one baseline run."""

    run_id: str = Field(min_length=1)
    strategy: str = Field(default="vector", min_length=1)
    corpus_id: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    answer_model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    answer_version: str = Field(min_length=1)
    token_estimator: str = Field(min_length=1)
    num_candidates: int = Field(ge=0, strict=True)
    max_chunks: int = Field(ge=1, strict=True)
    max_context_tokens: int = Field(ge=1, strict=True)
    question_set_hash: str = Field(min_length=1)
    question_count: int = Field(ge=1, strict=True)
    split: str | None = None
    reranker_model: str | None = None
    graph_fingerprint: str | None = None
    index_fingerprint: str | None = None
    max_triples: int = Field(default=0, ge=0, le=12, strict=True)
    include_triples: bool = False

    def model_post_init(self, _context: Any) -> None:
        allowed = (
            "vector",
            "vector_rerank",
            "hybrid_rerank",
            "hybrid_rerank_triples",
            "decomposition",
        )
        if self.strategy not in allowed:
            raise ValueError(f"strategy must be one of {allowed}, got {self.strategy!r}")
        if self.strategy in ("vector", "decomposition") and self.reranker_model is not None:
            raise ValueError(f"strategy {self.strategy!r} must not set reranker_model")
        if self.strategy in ("vector_rerank", "hybrid_rerank", "hybrid_rerank_triples"):
            if not self.reranker_model or not self.reranker_model.strip():
                raise ValueError(f"strategy {self.strategy!r} requires reranker_model")
        if self.include_triples != (self.strategy == "hybrid_rerank_triples"):
            raise ValueError("include_triples must be True only for hybrid_rerank_triples")
        if self.max_triples != (12 if self.strategy == "hybrid_rerank_triples" else 0):
            raise ValueError("max_triples must be 12 for hybrid_rerank_triples, else 0")


class AggregateMetrics(Record):
    """Macro-averaged metrics over the fixed denominator (0-100 scale)."""

    n_total: int = Field(ge=1, strict=True)
    n_success: int = Field(ge=0, strict=True)
    n_failed: int = Field(ge=0, strict=True)
    n_abstained: int = Field(ge=0, strict=True)
    n_supported: int = Field(ge=0, strict=True)
    exact_match_mean: float
    f1_mean: float
    paragraph_recall_candidate: float
    all_support_candidate: float
    sentence_recall_candidate: float
    paragraph_recall_packed: float
    all_support_packed: float
    sentence_recall_packed: float
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    total_input_tokens: int = Field(ge=0, strict=True)
    total_output_tokens: int = Field(ge=0, strict=True)
    total_estimated_cost_usd: float | None = None


class SuccessOnlyMetrics(Record):
    """Diagnostics over successful predictions only (failures excluded)."""

    n_success: int = Field(ge=0, strict=True)
    exact_match_mean: float | None = None
    f1_mean: float | None = None
    paragraph_recall_packed: float | None = None
    all_support_packed: float | None = None
    sentence_recall_packed: float | None = None


class RunEvaluation(Record):
    """One evaluated run: frozen config plus both aggregates."""

    config: RunConfig
    aggregate: AggregateMetrics
    success_only: SuccessOnlyMetrics
    scoring_version: str = SCORING_VERSION
    evaluation_version: str = EVALUATION_VERSION
    created_at: str


def hash_question_set(question_ids: list[str]) -> str:
    """Stable hash of the frozen question set (sorted ids)."""
    if not question_ids:
        raise ValueError("question_ids must be nonempty")
    canonical = json.dumps(sorted(question_ids), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile; None for empty input."""
    if not values:
        return None
    if not 0 <= pct <= 100:
        raise ValueError("pct must be in 0..100")
    ordered = sorted(values)
    rank = max(1, int(pct / 100 * len(ordered) + 0.5))
    rank = min(rank, len(ordered))
    return float(ordered[rank - 1])


def unresolved_citations(
    citations: list[dict[str, Any] | Any],
    packed: list[EvidenceView],
) -> list[dict[str, Any]]:
    """Return citations that do not resolve to packed (chunk, sentence) pairs."""
    allowed: dict[str, set[int]] = {chunk.chunk_id: set(chunk.sentence_ids) for chunk in packed}
    missing: list[dict[str, Any]] = []
    for citation in citations:
        if isinstance(citation, dict):
            chunk_id = citation.get("chunk_id")
            sentence_ids = citation.get("sentence_ids", [])
        else:
            chunk_id = getattr(citation, "chunk_id", None)
            sentence_ids = list(getattr(citation, "sentence_ids", []) or [])
        sentences = allowed.get(chunk_id) if isinstance(chunk_id, str) else None
        if sentences is None:
            missing.append({"chunk_id": chunk_id, "reason": "unknown_chunk"})
            continue
        if not isinstance(sentence_ids, list) or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in sentence_ids
        ):
            missing.append({"chunk_id": chunk_id, "reason": "bad_sentence_ids"})
            continue
        unknown = [v for v in sentence_ids if v not in sentences]
        if unknown:
            missing.append(
                {"chunk_id": chunk_id, "sentence_ids": unknown, "reason": "unknown_sentence"}
            )
    return missing


def coerce_evidence_view(item: Any) -> EvidenceView:
    """Accept dicts or chunk-like objects (RetrievedChunk/PackedChunk/ChunkRecord)."""
    if isinstance(item, EvidenceView):
        return item
    if isinstance(item, dict):
        return EvidenceView(
            chunk_id=str(item.get("chunk_id", item.get("id", ""))),
            paragraph_id=item.get("paragraph_id"),
            title=item.get("title"),
            sentence_ids=[int(v) for v in list(item.get("sentence_ids") or [])],
        )
    return EvidenceView(
        chunk_id=str(getattr(item, "chunk_id", getattr(item, "id", ""))),
        paragraph_id=getattr(item, "paragraph_id", None),
        title=getattr(item, "title", None),
        sentence_ids=[int(v) for v in list(getattr(item, "sentence_ids", []) or [])],
    )


def score_one_question(
    label: LabelView | dict[str, Any],
    prediction: QuestionPrediction | dict[str, Any],
) -> ScoredQuestion:
    """Score one question; failures receive zero answer scores.

    Retrieval coverage is still computed from whatever candidates/packed
    evidence the prediction carries (empty evidence scores zero coverage).
    """
    label_view = label if isinstance(label, LabelView) else LabelView.model_validate(label)
    pred = (
        prediction
        if isinstance(prediction, QuestionPrediction)
        else QuestionPrediction.model_validate(prediction)
    )
    if label_view.question_id != pred.question_id:
        raise ValueError("Label and prediction question ids must match")
    if pred.status == "failed":
        answer_em, answer_f1 = 0.0, 0.0
    else:
        scores = score_answer(pred.answer, label_view.answer)
        answer_em, answer_f1 = scores.exact_match, scores.f1
    candidate_scores = score_retrieval(
        [dict(f) for f in label_view.supporting_facts],
        [view.model_dump() for view in pred.candidates],
    )
    packed_scores = score_retrieval(
        [dict(f) for f in label_view.supporting_facts],
        [view.model_dump() for view in pred.packed],
    )
    missing = unresolved_citations(list(pred.citations), list(pred.packed))
    return ScoredQuestion(
        question_id=pred.question_id,
        status=pred.status,
        failure_code=pred.failure_code,
        exact_match=answer_em,
        f1=answer_f1,
        candidate=candidate_scores,
        packed=packed_scores,
        latency_ms=pred.latency_ms,
        citations_resolve=not missing,
        abstained=pred.status == "insufficient_evidence",
        fallback=pred.fallback,
        degraded=pred.degraded,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_questions(scored: list[ScoredQuestion]) -> AggregateMetrics:
    """Macro-average over the fixed denominator (failures included as zero)."""
    if not scored:
        raise ValueError("Cannot aggregate an empty question list")
    n_total = len(scored)
    n_failed = sum(1 for item in scored if item.status == "failed")
    n_success = n_total - n_failed
    n_abstained = sum(1 for item in scored if item.status == "insufficient_evidence")
    n_supported = sum(1 for item in scored if item.status == "supported")
    latencies = [item.latency_ms for item in scored if item.latency_ms is not None]
    return AggregateMetrics(
        n_total=n_total,
        n_success=n_success,
        n_failed=n_failed,
        n_abstained=n_abstained,
        n_supported=n_supported,
        exact_match_mean=_mean([item.exact_match for item in scored]) * 100.0,
        f1_mean=_mean([item.f1 for item in scored]) * 100.0,
        paragraph_recall_candidate=_mean([item.candidate.paragraph_recall for item in scored])
        * 100.0,
        all_support_candidate=_mean([item.candidate.all_support for item in scored]) * 100.0,
        sentence_recall_candidate=_mean([item.candidate.sentence_recall for item in scored])
        * 100.0,
        paragraph_recall_packed=_mean([item.packed.paragraph_recall for item in scored]) * 100.0,
        all_support_packed=_mean([item.packed.all_support for item in scored]) * 100.0,
        sentence_recall_packed=_mean([item.packed.sentence_recall for item in scored]) * 100.0,
        latency_p50_ms=percentile([float(v) for v in latencies], 50),
        latency_p95_ms=percentile([float(v) for v in latencies], 95),
        total_input_tokens=0,
        total_output_tokens=0,
        total_estimated_cost_usd=None,
    )


def aggregate_with_usage(
    scored: list[ScoredQuestion],
    predictions: list[QuestionPrediction],
) -> AggregateMetrics:
    """Aggregate plus summed usage tokens/cost from the original predictions."""
    base = aggregate_questions(scored)
    by_id = {pred.question_id: pred for pred in predictions}
    total_in = sum(
        (by_id[item.question_id].input_tokens or 0) for item in scored if item.question_id in by_id
    )
    total_out = sum(
        (by_id[item.question_id].output_tokens or 0) for item in scored if item.question_id in by_id
    )
    costs = [
        by_id[item.question_id].estimated_cost_usd
        for item in scored
        if item.question_id in by_id and by_id[item.question_id].estimated_cost_usd is not None
    ]
    return base.model_copy(
        update={
            "total_input_tokens": int(total_in),
            "total_output_tokens": int(total_out),
            "total_estimated_cost_usd": float(sum(costs)) if costs else None,
        }
    )


def summarize_success_only(scored: list[ScoredQuestion]) -> SuccessOnlyMetrics:
    """Diagnostics over successful predictions only (failures excluded)."""
    success = [item for item in scored if item.status != "failed"]
    if not success:
        return SuccessOnlyMetrics(n_success=0)
    return SuccessOnlyMetrics(
        n_success=len(success),
        exact_match_mean=_mean([item.exact_match for item in success]) * 100.0,
        f1_mean=_mean([item.f1 for item in success]) * 100.0,
        paragraph_recall_packed=_mean([item.packed.paragraph_recall for item in success]) * 100.0,
        all_support_packed=_mean([item.packed.all_support for item in success]) * 100.0,
        sentence_recall_packed=_mean([item.packed.sentence_recall for item in success]) * 100.0,
    )


def evaluate_predictions(
    labels: list[LabelView | dict[str, Any]],
    predictions: list[QuestionPrediction | dict[str, Any]],
    config: RunConfig | dict[str, Any],
) -> tuple[list[ScoredQuestion], RunEvaluation]:
    """Score every question without dropping failures; validate frozen ids.

    Raises when label/prediction id sets differ so a final report can never
    quietly drop difficult or failed questions.
    """
    label_views = [
        item if isinstance(item, LabelView) else LabelView.model_validate(item) for item in labels
    ]
    pred_views = [
        item if isinstance(item, QuestionPrediction) else QuestionPrediction.model_validate(item)
        for item in predictions
    ]
    cfg = config if isinstance(config, RunConfig) else RunConfig.model_validate(config)
    label_ids = sorted(item.question_id for item in label_views)
    pred_ids = sorted(item.question_id for item in pred_views)
    if label_ids != pred_ids:
        raise ValueError(
            "Label and prediction question sets must match exactly; "
            f"{len(label_ids)} labels vs {len(pred_ids)} predictions"
        )
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("Duplicate question ids in labels")
    if cfg.question_count != len(label_views):
        raise ValueError("RunConfig question_count must match the supplied question set")
    expected_hash = hash_question_set(label_ids)
    if cfg.question_set_hash != expected_hash:
        raise ValueError("RunConfig question_set_hash does not match the supplied questions")
    by_label = {item.question_id: item for item in label_views}
    scored = [
        score_one_question(by_label[pred.question_id], pred)
        for pred in sorted(pred_views, key=lambda item: item.question_id)
    ]
    aggregate = aggregate_with_usage(scored, pred_views)
    success_only = summarize_success_only(scored)
    evaluation = RunEvaluation(
        config=cfg,
        aggregate=aggregate,
        success_only=success_only,
        created_at=datetime.now(UTC).isoformat(),
    )
    return scored, evaluation


def scored_to_result_rows(
    scored: list[ScoredQuestion],
    predictions: list[QuestionPrediction],
    labels: list[LabelView],
) -> list[dict[str, Any]]:
    """Join per-question scores with answers/evidence for ``results.jsonl``."""
    by_pred = {item.question_id: item for item in predictions}
    by_label = {item.question_id: item for item in labels}
    rows: list[dict[str, Any]] = []
    for item in sorted(scored, key=lambda entry: entry.question_id):
        pred = by_pred[item.question_id]
        label = by_label[item.question_id]
        triple_rows = [
            dict(t) if isinstance(t, dict) else t.model_dump(mode="json") for t in pred.triples
        ]
        rows.append(
            {
                "question_id": item.question_id,
                "answer": pred.answer,
                "gold_answer": label.answer,
                "status": pred.status,
                "failure_code": pred.failure_code,
                "exact_match": item.exact_match,
                "f1": item.f1,
                "candidate_ids": [chunk.chunk_id for chunk in pred.candidates],
                "packed_ids": [chunk.chunk_id for chunk in pred.packed],
                "packed_sentence_ids": [
                    {"chunk_id": chunk.chunk_id, "sentence_ids": list(chunk.sentence_ids)}
                    for chunk in pred.packed
                ],
                "citations": [dict(c) if isinstance(c, dict) else dict(c) for c in pred.citations],
                "citations_resolve": item.citations_resolve,
                "candidate_recall": item.candidate.model_dump(),
                "packed_recall": item.packed.model_dump(),
                "retrieval": {
                    "num_candidates": len(pred.candidates),
                    "num_packed": len(pred.packed),
                    "fallback": pred.fallback,
                    "degraded": pred.degraded,
                    "reranked": pred.reranked,
                    "strategy": pred.strategy,
                    "n_triples": len(triple_rows),
                },
                "triples": triple_rows,
                "flags": {
                    "abstained": item.abstained,
                    "fallback": pred.fallback,
                    "degraded": pred.degraded,
                    "reranked": pred.reranked,
                },
                "usage": {
                    "input_tokens": pred.input_tokens,
                    "output_tokens": pred.output_tokens,
                    "estimated_cost_usd": pred.estimated_cost_usd,
                },
                "latency_ms": pred.latency_ms,
            }
        )
    return rows


def write_jsonl_atomic(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write JSONL atomically (tmp file then rename), one canonical line per row."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )
    tmp.write_bytes(payload)
    tmp.rename(dest)
    return dest


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write one canonical JSON document atomically."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(
        (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    )
    tmp.rename(dest)
    return dest


# ---------------------------------------------------------------------------
# T11 controlled arms (hybrid-qa, R08)
# ---------------------------------------------------------------------------

EXPERIMENT_ARMS: tuple[str, ...] = (
    "vector",
    "vector_rerank",
    "hybrid_rerank",
    "hybrid_rerank_triples",
)

# T17 optional decomposition arm (R13, E). The core A-D tuple above stays
# frozen; decomposition is compared additively and never replaces A-D runs.
DECOMPOSITION_STRATEGY = "decomposition"
OPTIONAL_ARMS: tuple[str, ...] = (*EXPERIMENT_ARMS, DECOMPOSITION_STRATEGY)


def run_config_for_arm(
    *,
    run_id: str,
    strategy: str,
    corpus_id: str,
    embedding_model: str,
    answer_model: str,
    prompt_version: str,
    answer_version: str,
    token_estimator: str,
    num_candidates: int,
    max_chunks: int,
    max_context_tokens: int,
    question_ids: list[str],
    split: str | None = None,
    reranker_model: str | None = None,
    graph_fingerprint: str | None = None,
    index_fingerprint: str | None = None,
) -> RunConfig:
    """Build a frozen RunConfig for one of arms A-D (or E decomposition).

    Final budgets (``max_chunks``/``max_context_tokens``) are supplied by the
    caller so all arms share identical values; this helper only derives the
    arm-specific reranker/triple flags (B/C/D require a reranker model, D
    enables 12 triples; A and E decomposition set neither).
    """
    if strategy not in OPTIONAL_ARMS:
        raise ValueError(f"strategy must be one of {OPTIONAL_ARMS}, got {strategy!r}")
    include = strategy == "hybrid_rerank_triples"
    triples = 12 if include else 0
    return RunConfig(
        run_id=run_id,
        strategy=strategy,
        corpus_id=corpus_id,
        embedding_model=embedding_model,
        answer_model=answer_model,
        prompt_version=prompt_version,
        answer_version=answer_version,
        token_estimator=token_estimator,
        num_candidates=num_candidates,
        max_chunks=max_chunks,
        max_context_tokens=max_context_tokens,
        question_set_hash=hash_question_set(list(question_ids)),
        question_count=len(question_ids),
        split=split,
        reranker_model=reranker_model,
        graph_fingerprint=graph_fingerprint,
        index_fingerprint=index_fingerprint,
        max_triples=triples,
        include_triples=include,
    )


# ---------------------------------------------------------------------------
# T13 locked evaluation and single-hop controls (evaluation, R08-R10)
# ---------------------------------------------------------------------------

EVALUATION_CONFIG_VERSION = "evaluation-config-v1"
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_CONFIDENCE = 0.95
TUNING_SIZE = 30
EVALUATION_SIZE = 120
CONTROL_SIZE = 30
SINGLE_HOP_REGRESSION_TOLERANCE_PP = 2.0

COMPATIBILITY_FIELDS: tuple[str, ...] = (
    "corpus_id",
    "index_fingerprint",
    "question_set_hash",
    "embedding_model",
    "answer_model",
    "prompt_version",
    "max_chunks",
    "max_context_tokens",
)


class EvaluationConfig(Record):
    """Frozen locked-evaluation settings (R08-R10, no secrets)."""

    evaluation_version: str = Field(default=EVALUATION_VERSION, min_length=1)
    scoring_version: str = Field(default=SCORING_VERSION, min_length=1)
    bootstrap_seed: int = Field(default=BOOTSTRAP_SEED, ge=0, strict=True)
    bootstrap_replicates: int = Field(default=BOOTSTRAP_REPLICATES, ge=100, strict=True)
    confidence_level: float = Field(default=BOOTSTRAP_CONFIDENCE, gt=0.0, lt=1.0)
    tuning_size: int = Field(default=TUNING_SIZE, ge=1, strict=True)
    evaluation_size: int = Field(default=EVALUATION_SIZE, ge=1, strict=True)
    control_size: int = Field(default=CONTROL_SIZE, ge=1, strict=True)
    arms: list[str] = Field(default_factory=lambda: list(EXPERIMENT_ARMS))
    num_candidates: int = Field(default=30, ge=1, strict=True)
    max_chunks: int = Field(default=8, ge=1, strict=True)
    max_context_tokens: int = Field(default=3000, ge=1, strict=True)
    single_hop_path: str = Field(default="data/controls/single_hop.jsonl", min_length=1)
    require_human_review: bool = True
    single_hop_regression_tolerance_pp: float = Field(
        default=SINGLE_HOP_REGRESSION_TOLERANCE_PP, ge=0.0
    )
    # T17 optional decomposition freeze metadata (R13, E). Core A-D arms stay
    # frozen above; these fields record the bounded planner version and the
    # requirement that tuning selection freezes before any locked use.
    decomposition_version: str = Field(default="decomposition-v1", min_length=1)
    decomposition_prompt_version: str = Field(default="decomposition-planner-v1", min_length=1)
    decomposition_max_subquestions: int = Field(default=2, ge=1, le=2, strict=True)
    require_decomposition_freeze: bool = True

    def model_post_init(self, _context: Any) -> None:
        if sorted(self.arms) != sorted(EXPERIMENT_ARMS):
            raise ValueError(f"arms must be exactly {list(EXPERIMENT_ARMS)}, got {self.arms!r}")


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    """Load non-secret evaluation YAML (no credentials in this file)."""
    import yaml

    content = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if content is None:
        content = {}
    if not isinstance(content, dict):
        raise ValueError("Evaluation configuration root must be a mapping")
    if {"neo4j_password", "llm_api_key"}.intersection(content):
        raise ValueError("Credentials must be supplied through process environment variables")
    # max_triples_per_arm / compatibility_fields / decision_rule are
    # documentary in the YAML; the frozen model above is authoritative.
    allowed_extra = {
        "max_triples_per_arm",
        "compatibility_fields",
        "decision_rule",
        "decomposition_strategy",
        "decomposition_decision_rule",
        "decomposition_notes",
    }
    unknown = set(content) - set(EvaluationConfig.model_fields) - allowed_extra
    if unknown:
        raise ValueError(f"Unknown evaluation config fields: {sorted(unknown)}")
    known = {k: v for k, v in content.items() if k in EvaluationConfig.model_fields}
    return EvaluationConfig(**known)


class ControlSupportingFact(Record):
    """One gold supporting fact for a single-hop control."""

    title: str = Field(min_length=1)
    sentence_id: int = Field(ge=0, strict=True)
    paragraph_id: str = Field(min_length=1)


class ControlSource(Record):
    """The single source paragraph a control is authored from."""

    title: str = Field(min_length=1)
    paragraph_id: str = Field(min_length=1)
    sentences: list[str] = Field(min_length=1)


class SingleHopControl(Record):
    """One human-authored single-hop control (R10, authored, not measured).

    Controls are written from one paragraph fact without running the compared
    systems. Frozen use requires ``human_verified=True``; the shipped file
    stays ``False`` until an independent reviewer audits every answer/support
    mapping.
    """

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    supporting_facts: list[ControlSupportingFact] = Field(min_length=1)
    source: ControlSource
    split: Literal["control"] = "control"
    question_type: Literal["single_hop"] = "single_hop"
    human_verified: bool = False
    author_note: str = Field(default="", min_length=0)

    def model_post_init(self, _context: Any) -> None:
        if len(self.question.strip()) < 10 or len(self.question) > 500:
            raise ValueError("Control question must be 10..500 chars")
        if not self.question.strip().endswith("?"):
            raise ValueError("Control question must end with '?'")
        if not self.answer.strip():
            raise ValueError("Control answer must be nonempty")
        paragraph_ids = {fact.paragraph_id for fact in self.supporting_facts}
        if len(paragraph_ids) != 1:
            raise ValueError("Single-hop control must cite exactly one paragraph")
        only = next(iter(paragraph_ids))
        if only != self.source.paragraph_id:
            raise ValueError("Supporting facts must resolve to the control source paragraph")
        for fact in self.supporting_facts:
            if fact.title != self.source.title:
                raise ValueError("Supporting fact title must match the source title")
            if fact.sentence_id >= len(self.source.sentences):
                raise ValueError(
                    f"Supporting sentence {fact.sentence_id} out of range "
                    f"for {self.question_id!r} ({len(self.source.sentences)} sentences)"
                )
        for sentence in self.source.sentences:
            if not sentence.strip():
                raise ValueError(f"Control {self.question_id!r} has an empty source sentence")
        # Answer provenance without running any system: the normalized answer
        # must appear in at least one cited supporting sentence.
        cited = {fact.sentence_id for fact in self.supporting_facts}
        wanted = normalize_answer(self.answer)
        if not wanted:
            raise ValueError(f"Control {self.question_id!r} answer normalizes to empty")
        haystacks = [normalize_answer(self.source.sentences[i]) for i in sorted(cited)]
        if not any(wanted in text for text in haystacks):
            raise ValueError(
                f"Control {self.question_id!r} answer does not occur in its cited sentence"
            )


def load_single_hop_controls(path: str | Path) -> list[SingleHopControl]:
    """Read one JSONL controls file (one object per line, deterministic order)."""
    dest = Path(path)
    lines = dest.read_text(encoding="utf-8").splitlines()
    controls: list[SingleHopControl] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{dest} line {number} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{dest} line {number} must be a JSON object")
        try:
            control = SingleHopControl.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"{dest} line {number} invalid control: {exc}") from exc
        if control.question_id in seen:
            raise ValueError(f"Duplicate control question id: {control.question_id!r}")
        seen.add(control.question_id)
        controls.append(control)
    controls.sort(key=lambda item: item.question_id)
    return controls


def controls_freeze_hash(controls: list[SingleHopControl]) -> str:
    """Content hash of the frozen control set (canonical JSONL bytes)."""
    if not controls:
        raise ValueError("Cannot hash an empty control list")
    payload = b"".join(
        (
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        for item in sorted(controls, key=lambda c: c.question_id)
    )
    return hashlib.sha256(payload).hexdigest()


def validate_single_hop_controls(
    controls: list[SingleHopControl],
    *,
    expected_count: int = CONTROL_SIZE,
    require_verified: bool = True,
) -> dict[str, Any]:
    """Enforce the frozen single-hop gate (R10).

    Checks 30 unique controls, single-paragraph support, sentence bounds and
    answer-span provenance (via the model itself). ``require_verified=True``
    refuses draft files where any control is still ``human_verified=False``;
    the shipped file intentionally fails that gate until a reviewer audits it.
    """
    if not isinstance(expected_count, int) or isinstance(expected_count, bool):
        raise ValueError("expected_count must be an integer")
    if len(controls) != expected_count:
        raise ValueError(
            f"Single-hop controls must hold exactly {expected_count} rows, got {len(controls)}"
        )
    ids = [item.question_id for item in controls]
    if len(set(ids)) != len(ids):
        raise ValueError("Single-hop control ids must be unique")
    unverified = sorted(item.question_id for item in controls if not item.human_verified)
    if require_verified and unverified:
        raise ValueError(
            f"{len(unverified)} control(s) lack human review "
            f"(first: {unverified[0]!r}); refusing frozen use"
        )
    return {
        "n_controls": len(controls),
        "question_ids": sorted(ids),
        "question_set_hash": hash_question_set(sorted(ids)),
        "freeze_hash": controls_freeze_hash(controls),
        "n_verified": sum(1 for item in controls if item.human_verified),
        "unverified_ids": unverified,
        "require_verified": require_verified,
    }


def validate_split_disjoint(
    tuning_ids: list[str],
    evaluation_ids: list[str],
    control_ids: list[str],
    *,
    expected_tuning: int | None = TUNING_SIZE,
    expected_evaluation: int | None = EVALUATION_SIZE,
    expected_controls: int | None = CONTROL_SIZE,
) -> dict[str, Any]:
    """Enforce tuning/locked/control separation (no gold leakage across uses).

    Tuning must never contain a locked evaluation or control id. Locked and
    control sets must be disjoint. Size expectations default to 30/120/30 and
    may be disabled by passing None.
    """
    tuning = list(tuning_ids)
    locked = list(evaluation_ids)
    controls = list(control_ids)
    for label, values in (("tuning", tuning), ("evaluation", locked), ("control", controls)):
        if len(set(values)) != len(values):
            raise ValueError(f"{label} question ids must be unique")
        if not all(isinstance(v, str) and v.strip() for v in values):
            raise ValueError(f"{label} question ids must be nonempty strings")
    overlap_tuning_locked = sorted(set(tuning) & set(locked))
    overlap_tuning_controls = sorted(set(tuning) & set(controls))
    overlap_locked_controls = sorted(set(locked) & set(controls))
    if overlap_tuning_locked:
        raise ValueError(
            f"Tuning ids overlap locked evaluation ids: {overlap_tuning_locked[:5]}; "
            "do not tune on locked answers or evidence labels"
        )
    if overlap_tuning_controls:
        raise ValueError(
            f"Tuning ids overlap single-hop control ids: {overlap_tuning_controls[:5]}; "
            "do not tune on controls"
        )
    if overlap_locked_controls:
        raise ValueError(
            f"Locked ids overlap single-hop control ids: {overlap_locked_controls[:5]}"
        )
    for label, values, expected in (
        ("tuning", tuning, expected_tuning),
        ("evaluation", locked, expected_evaluation),
        ("control", controls, expected_controls),
    ):
        if expected is not None and len(values) != expected:
            raise ValueError(f"{label} set must hold {expected} ids, got {len(values)}")
    return {
        "tuning_size": len(tuning),
        "evaluation_size": len(locked),
        "control_size": len(controls),
        "tuning_hash": hash_question_set(sorted(tuning)) if tuning else None,
        "evaluation_hash": hash_question_set(sorted(locked)) if locked else None,
        "control_hash": hash_question_set(sorted(controls)) if controls else None,
    }


def assert_locked_question_use(requested_ids: list[str], frozen_ids: list[str]) -> str:
    """Require the locked run to use exactly the frozen ids (no drops/adds)."""
    if sorted(requested_ids) != sorted(frozen_ids):
        raise ValueError(
            "Locked evaluation must use exactly the frozen question ids; "
            f"{len(requested_ids)} requested vs {len(frozen_ids)} frozen"
        )
    return hash_question_set(sorted(frozen_ids))


def compatibility_hash_for_runs(configs: list[RunConfig]) -> str:
    """Hash the fixed comparison fields shared across compatible runs."""
    if not configs:
        raise ValueError("configs must be nonempty")
    first = configs[0]
    reference = {key: getattr(first, key) for key in COMPATIBILITY_FIELDS}
    canonical = json.dumps(reference, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def check_run_compatibility(configs: list[RunConfig]) -> dict[str, Any]:
    """Check that run configs share the fixed paired-comparison fields.

    Strategy-adjacent fields (strategy, reranker model, graph fingerprint,
    triples, split, answer version, token estimator, candidate counts) are the
    experiment and may differ. Corpus, index fingerprint, question set,
    embedding/answer models and final budgets must match; otherwise no
    unlabeled paired delta is produced.
    """
    if not isinstance(configs, list) or not configs:
        raise ValueError("configs must be a nonempty list")
    parsed = [
        item if isinstance(item, RunConfig) else RunConfig.model_validate(item) for item in configs
    ]
    first = parsed[0]
    reference = {key: getattr(first, key) for key in COMPATIBILITY_FIELDS}
    for index, cfg in enumerate(parsed[1:], start=1):
        for key in COMPATIBILITY_FIELDS:
            if getattr(cfg, key) != reference[key]:
                return {
                    "compatible": False,
                    "reason": (
                        f"Run {cfg.run_id!r} differs on {key}: "
                        f"{getattr(cfg, key)!r} != {reference[key]!r}; "
                        "refusing unlabeled paired delta"
                    ),
                    "compatibility_hash": None,
                    "fixed": dict(reference),
                }
    payload = json.dumps(reference, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "compatible": True,
        "reason": "fixed fields match across runs",
        "compatibility_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "fixed": dict(reference),
    }


def paired_bootstrap_interval(
    diffs_pp: list[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    """Bootstrap 95% interval for paired per-question deltas (percentage points).

    ``diffs_pp`` holds one paired difference per question (arm B minus arm A)
    already on the 0-100 scale. Failures contribute as zero-score questions
    through their scored metrics, so callers must pass the full fixed
    denominator without dropping difficult or failed questions. Resampling is
    deterministic via ``seed`` (default 42).
    """
    import random

    if not isinstance(diffs_pp, list) or not diffs_pp:
        raise ValueError("diffs_pp must be a nonempty list")
    for value in diffs_pp:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("diffs_pp must hold numbers")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 100:
        raise ValueError("replicates must be an integer >= 100")
    if not isinstance(confidence, float) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    n = len(diffs_pp)
    if n < 2:
        raise ValueError("Need at least 2 paired questions for a bootstrap interval")
    observed = float(sum(diffs_pp) / n)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(replicates):
        sample = [diffs_pp[i] for i in rng.choices(range(n), k=n)]
        means.append(float(sum(sample) / n))
    means.sort()
    alpha = 1.0 - confidence
    lower_pct = alpha / 2 * 100
    upper_pct = (1 - alpha / 2) * 100
    lower = percentile(means, lower_pct)
    upper = percentile(means, upper_pct)
    assert lower is not None and upper is not None
    return {
        "n": n,
        "observed_delta_pp": observed,
        "lower_pp": float(lower),
        "upper_pp": float(upper),
        "seed": seed,
        "replicates": replicates,
        "confidence": confidence,
    }


def _metric_value(item: ScoredQuestion, metric: str) -> float:
    """Extract one 0..1 metric from a scored question (failures score zero)."""
    if metric == "exact_match":
        return float(item.exact_match)
    if metric == "f1":
        return float(item.f1)
    if metric == "paragraph_recall_candidate":
        return float(item.candidate.paragraph_recall)
    if metric == "all_support_candidate":
        return float(item.candidate.all_support)
    if metric == "sentence_recall_candidate":
        return float(item.candidate.sentence_recall)
    if metric == "paragraph_recall_packed":
        return float(item.packed.paragraph_recall)
    if metric == "all_support_packed":
        return float(item.packed.all_support)
    if metric == "sentence_recall_packed":
        return float(item.packed.sentence_recall)
    raise ValueError(
        "metric must be one of exact_match, f1, paragraph/all/sentence "
        f"candidate/packed, got {metric!r}"
    )


PAIRED_METRICS: tuple[str, ...] = (
    "exact_match",
    "f1",
    "paragraph_recall_packed",
    "all_support_packed",
    "sentence_recall_packed",
)


def paired_delta_for_metric(
    scored_a: list[ScoredQuestion],
    scored_b: list[ScoredQuestion],
    metric: str,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    """Paired delta (B minus A, percentage points) with a bootstrap interval.

    Both arms must cover exactly the same question ids in the same fixed
    denominator; failures stay in the denominator as zero answer scores and
    are never dropped before resampling.
    """
    by_a = {item.question_id: item for item in scored_a}
    by_b = {item.question_id: item for item in scored_b}
    if sorted(by_a) != sorted(by_b):
        raise ValueError("Paired arms must cover exactly the same question ids")
    if len(by_a) != len(scored_a) or len(by_b) != len(scored_b):
        raise ValueError("Duplicate question ids in a paired arm")
    ordered = sorted(by_a)
    a_vals = [_metric_value(by_a[qid], metric) * 100.0 for qid in ordered]
    b_vals = [_metric_value(by_b[qid], metric) * 100.0 for qid in ordered]
    diffs = [b - a for a, b in zip(a_vals, b_vals)]
    interval = paired_bootstrap_interval(
        diffs, seed=seed, replicates=replicates, confidence=confidence
    )
    return {
        "metric": metric,
        "question_ids": ordered,
        "n": len(ordered),
        "mean_a": float(sum(a_vals) / len(a_vals)),
        "mean_b": float(sum(b_vals) / len(b_vals)),
        **interval,
    }


def group_by_type(
    scored: list[ScoredQuestion],
    labels: list[LabelView],
) -> dict[str, list[ScoredQuestion]]:
    """Group scored questions by label question type (failures stay grouped)."""
    by_label = {item.question_id: item for item in labels}
    groups: dict[str, list[ScoredQuestion]] = {}
    for item in scored:
        label = by_label.get(item.question_id)
        stratum = str(label.question_type) if label and label.question_type else "unknown"
        groups.setdefault(stratum, []).append(item)
    for members in groups.values():
        members.sort(key=lambda entry: entry.question_id)
    return groups


def aggregate_by_stratum(
    scored: list[ScoredQuestion],
    labels: list[LabelView],
) -> dict[str, Any]:
    """Aggregate overall plus per-type strata with counts (fixed denominator)."""
    groups = group_by_type(scored, labels)
    strata: dict[str, Any] = {}
    for name in sorted(groups):
        members = groups[name]
        strata[name] = {
            "n": len(members),
            "aggregate": aggregate_questions(members).model_dump(),
            "success_only": summarize_success_only(members).model_dump(),
            "question_ids": sorted(item.question_id for item in members),
        }
    return {
        "n_total": len(scored),
        "overall": aggregate_questions(scored).model_dump(),
        "strata": strata,
        "counts": {name: payload["n"] for name, payload in strata.items()},
    }


def compare_runs(
    scored_a: list[ScoredQuestion],
    evaluation_a: RunEvaluation,
    scored_b: list[ScoredQuestion],
    evaluation_b: RunEvaluation,
    labels: list[LabelView],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    """Paired comparison of two frozen runs with intervals and strata.

    Returns compatible=False with a reason (and no deltas) when the fixed
    fields differ or the paired ids differ, so incompatible runs can never
    produce an unlabeled paired delta. All attempts and failures stay in the
    denominator: failed questions score zero and are resampled like any other
    paired difference. Small samples carry an explicit caveat; shared evidence
    and small samples limit inference to the observed sets.
    """
    check = check_run_compatibility([evaluation_a.config, evaluation_b.config])
    ids_a = sorted(item.question_id for item in scored_a)
    ids_b = sorted(item.question_id for item in scored_b)
    if ids_a != ids_b:
        return {
            "compatible": False,
            "reason": (
                f"Paired arms cover different questions ({len(ids_a)} vs {len(ids_b)}); "
                "refusing unlabeled paired delta"
            ),
            "compatibility_hash": check.get("compatibility_hash"),
            "fixed": check.get("fixed"),
            "question_ids": None,
            "metrics": {},
            "strata": {},
            "failures": {},
        }
    if not check["compatible"]:
        return {
            "compatible": False,
            "reason": str(check["reason"]),
            "compatibility_hash": None,
            "fixed": check.get("fixed"),
            "question_ids": ids_a,
            "metrics": {},
            "strata": {},
            "failures": {},
        }
    metrics: dict[str, Any] = {}
    for metric in PAIRED_METRICS:
        metrics[metric] = paired_delta_for_metric(
            scored_a, scored_b, metric, seed=seed, replicates=replicates, confidence=confidence
        )
    by_label = {item.question_id: item for item in labels}
    strata_names = sorted(
        {
            str(by_label[qid].question_type)
            if qid in by_label and by_label[qid].question_type
            else "unknown"
            for qid in ids_a
        }
    )
    strata: dict[str, Any] = {}
    for name in strata_names:
        qids = sorted(
            qid
            for qid in ids_a
            if (
                str(by_label[qid].question_type)
                if qid in by_label and by_label[qid].question_type
                else "unknown"
            )
            == name
        )
        sub_a = [item for item in scored_a if item.question_id in set(qids)]
        sub_b = [item for item in scored_b if item.question_id in set(qids)]
        per_metric: dict[str, Any] = {}
        for metric in ("f1", "all_support_packed"):
            a_vals = [
                _metric_value(item, metric) * 100.0
                for item in sorted(sub_a, key=lambda e: e.question_id)
            ]
            b_vals = [
                _metric_value(item, metric) * 100.0
                for item in sorted(sub_b, key=lambda e: e.question_id)
            ]
            diffs = [b - a for a, b in zip(a_vals, b_vals)]
            observed = float(sum(diffs) / len(diffs)) if diffs else 0.0
            if len(diffs) >= 2:
                interval = paired_bootstrap_interval(
                    diffs, seed=seed, replicates=replicates, confidence=confidence
                )
            else:
                interval = {
                    "n": len(diffs),
                    "observed_delta_pp": observed,
                    "lower_pp": None,
                    "upper_pp": None,
                    "seed": seed,
                    "replicates": replicates,
                    "confidence": confidence,
                    "note": "insufficient n for an interval; observed delta only",
                }
            per_metric[metric] = {
                "n": len(diffs),
                "mean_a": float(sum(a_vals) / len(a_vals)) if a_vals else 0.0,
                "mean_b": float(sum(b_vals) / len(b_vals)) if b_vals else 0.0,
                **interval,
            }
        strata[name] = {"n": len(qids), "question_ids": qids, "metrics": per_metric}
    failures = {
        "arm_a": {
            "run_id": evaluation_a.config.run_id,
            "strategy": evaluation_a.config.strategy,
            "n_failed": evaluation_a.aggregate.n_failed,
            "n_abstained": evaluation_a.aggregate.n_abstained,
            "n_supported": evaluation_a.aggregate.n_supported,
        },
        "arm_b": {
            "run_id": evaluation_b.config.run_id,
            "strategy": evaluation_b.config.strategy,
            "n_failed": evaluation_b.aggregate.n_failed,
            "n_abstained": evaluation_b.aggregate.n_abstained,
            "n_supported": evaluation_b.aggregate.n_supported,
        },
    }
    warnings = [
        "Primary denominator is fixed: failures score zero and stay resampled.",
        "Shared evidence and small samples limit inference; "
        "intervals describe the observed sets, not population-wide significance.",
    ]
    if len(ids_a) < 30:
        warnings.append("Small sample: paired intervals are noisy below n=30.")
    return {
        "compatible": True,
        "reason": "fixed fields and paired ids match",
        "compatibility_hash": check["compatibility_hash"],
        "fixed": check["fixed"],
        "question_ids": ids_a,
        "n": len(ids_a),
        "metrics": metrics,
        "strata": strata,
        "failures": failures,
        "seed": seed,
        "replicates": replicates,
        "confidence": confidence,
        "evaluation_version": EVALUATION_VERSION,
        "scoring_version": SCORING_VERSION,
        "warnings": warnings,
    }


def check_tuning_selection(
    baseline: RunEvaluation,
    candidate: RunEvaluation,
) -> dict[str, Any]:
    """Apply the PRD tuning decision rule (freeze before locked evaluation).

    Select the complex arm only when tuning F1 improves and all-support
    coverage does not worsen. Costs are reported alongside but judged by the
    operator as acceptable; this helper never invents a cost verdict.
    """
    f1_delta = float(candidate.aggregate.f1_mean - baseline.aggregate.f1_mean)
    support_delta = float(
        candidate.aggregate.all_support_packed - baseline.aggregate.all_support_packed
    )
    select = f1_delta > 0 and support_delta >= 0
    f1_word = "improves" if f1_delta > 0 else "does not improve"
    sup_word = "no worsening" if support_delta >= 0 else "worsens"
    reasons = [
        f"Tuning F1 delta {f1_delta:+.2f}pp ({f1_word}).",
        f"All-support delta {support_delta:+.2f}pp ({sup_word}).",
        "Costs acceptable: operator judgement required (see query/index spend).",
    ]
    return {
        "select_candidate": bool(select),
        "baseline_run": baseline.config.run_id,
        "candidate_run": candidate.config.run_id,
        "f1_delta_pp": f1_delta,
        "all_support_delta_pp": support_delta,
        "reasons": reasons,
    }


def flag_single_hop_regression(
    delta_f1_pp: float,
    *,
    tolerance_pp: float = SINGLE_HOP_REGRESSION_TOLERANCE_PP,
) -> dict[str, Any]:
    """Flag a single-hop F1 drop beyond the provisional tolerance (default 2pp).

    This is a regression signal to investigate with sample-size caveats, not
    evidence of non-inferiority.
    """
    if not isinstance(delta_f1_pp, (int, float)) or isinstance(delta_f1_pp, bool):
        raise ValueError("delta_f1_pp must be a number")
    regression = float(delta_f1_pp) < -abs(float(tolerance_pp))
    return {
        "regression": bool(regression),
        "delta_f1_pp": float(delta_f1_pp),
        "tolerance_pp": float(tolerance_pp),
        "note": (
            "Single-hop regression beyond tolerance; investigate with sample-size "
            "caveats (provisional tolerance, not a non-inferiority claim)."
            if regression
            else "Single-hop change within the provisional tolerance."
        ),
    }


# ---------------------------------------------------------------------------
# T17 decomposition comparison (decomposition, optional, R13)
# ---------------------------------------------------------------------------

DECOMPOSITION_EVAL_VERSION = "decomposition-eval-v1"


class DecompositionSelection(Record):
    """Frozen tuning choice for the optional decomposition arm (R13, E).

    Records the baseline-vs-decomposition tuning comparison and freezes the
    decision before any locked use. Locked runs must reuse the frozen
    strategies, corpus/index fingerprints, models and final budgets; only the
    question set (tuning hash vs locked hash) and split may change. The delta
    report is additive and never replaces the frozen A-D results.
    """

    selection_id: str = Field(min_length=1)
    baseline_run_id: str = Field(min_length=1)
    candidate_run_id: str = Field(min_length=1)
    split: Literal["tuning"] = "tuning"
    select_candidate: bool = False
    f1_delta_pp: float
    all_support_delta_pp: float
    reasons: list[str] = Field(min_length=1)
    tuning_question_ids: list[str] = Field(min_length=1)
    tuning_question_set_hash: str = Field(min_length=1)
    baseline_strategy: str = Field(min_length=1)
    candidate_strategy: str = Field(min_length=1)
    fixed: dict[str, Any]
    max_chunks: int = Field(ge=1, strict=True)
    max_context_tokens: int = Field(ge=1, strict=True)
    decomposition_version: str = Field(min_length=1)
    decomposition_prompt_version: str = Field(min_length=1)
    frozen_at: str = Field(min_length=1)
    evaluation_version: str = Field(default=EVALUATION_VERSION, min_length=1)
    decomposition_eval_version: str = Field(default=DECOMPOSITION_EVAL_VERSION, min_length=1)


def _result_field(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def decomposition_to_prediction(
    question_id: str,
    result: Any,
    chunks_by_id: dict[str, dict[str, Any]],
    *,
    latency_ms: float | None = None,
    strategy: str = DECOMPOSITION_STRATEGY,
) -> QuestionPrediction:
    """Map one T16 DecompositionResult to a frozen QuestionPrediction (R13).

    Candidates come from ``combined_chunk_ids`` (every sub-retrieval, deduped)
    and packed evidence from ``packed_chunk_ids`` (final shared budget). Chunk
    provenance (paragraph_id or title plus sentence ids) is required so
    retrieval coverage scores real evidence; unknown chunks or missing
    provenance raise instead of scoring zero silently. Money stays unknown
    (None), never zero. Latency defaults to the result diagnostics
    ``elapsed_ms`` when the caller does not supply an observed value.
    """
    if not isinstance(question_id, str) or not question_id.strip():
        raise ValueError("question_id must be a nonempty string")
    if strategy != DECOMPOSITION_STRATEGY:
        raise ValueError(
            f"Decomposition predictions must use strategy {DECOMPOSITION_STRATEGY!r}, "
            f"got {strategy!r}"
        )
    if not isinstance(chunks_by_id, dict) or not chunks_by_id:
        raise ValueError("chunks_by_id must be a nonempty mapping")
    answer = _result_field(result, "answer", "")
    status = _result_field(result, "status", None)
    if status not in ("supported", "insufficient_evidence", "failed"):
        raise ValueError(f"Decomposition result has unknown status {status!r}")
    raw_citations = _result_field(result, "citations", []) or []
    if not isinstance(raw_citations, list):
        raise ValueError("Decomposition citations must be a list")
    citations: list[dict[str, Any]] = []
    for entry in raw_citations:
        if isinstance(entry, dict):
            citations.append(dict(entry))
        elif hasattr(entry, "model_dump"):
            citations.append(entry.model_dump(mode="json"))  # type: ignore[union-attr]
        else:
            raise ValueError("Decomposition citation rows must be mappings")
    combined_ids = list(_result_field(result, "combined_chunk_ids", []) or [])
    packed_ids = list(_result_field(result, "packed_chunk_ids", []) or [])
    if status == "failed" and (combined_ids or packed_ids):
        raise ValueError("Failed decomposition results must not carry packed evidence")

    def _view(chunk_id: Any) -> EvidenceView:
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Decomposition chunk ids must be nonempty strings")
        record = chunks_by_id.get(chunk_id)
        if record is None:
            raise ValueError(f"Decomposition chunk {chunk_id!r} is not in chunks_by_id")
        if not isinstance(record, dict):
            raise ValueError(f"Decomposition chunk {chunk_id!r} must be a mapping")
        paragraph_id = record.get("paragraph_id")
        title = record.get("title")
        if not paragraph_id and not title:
            raise ValueError(f"Decomposition chunk {chunk_id!r} lacks paragraph/title provenance")
        sentence_ids = list(record.get("sentence_ids") or [])
        for value in sentence_ids:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Decomposition chunk {chunk_id!r} has bad sentence ids")
        return EvidenceView(
            chunk_id=chunk_id,
            paragraph_id=paragraph_id,
            title=title,
            sentence_ids=sentence_ids,
        )

    candidates = [_view(cid) for cid in combined_ids]
    packed = [_view(cid) for cid in packed_ids]
    # Packed must be a subset of combined (final budget selects from sub-evidence).
    combined_set = set(combined_ids)
    for cid in packed_ids:
        if cid not in combined_set:
            raise ValueError(f"Decomposition packed chunk {cid!r} is not in combined evidence")
    total_in = _result_field(result, "total_input_tokens", 0) or 0
    total_out = _result_field(result, "total_output_tokens", 0) or 0
    if isinstance(total_in, bool) or not isinstance(total_in, int) or total_in < 0:
        raise ValueError("Decomposition total_input_tokens must be a non-negative int")
    if isinstance(total_out, bool) or not isinstance(total_out, int) or total_out < 0:
        raise ValueError("Decomposition total_output_tokens must be a non-negative int")
    diagnostics = _result_field(result, "diagnostics", {}) or {}
    observed_latency = latency_ms
    if observed_latency is None and isinstance(diagnostics, dict):
        elapsed = diagnostics.get("elapsed_ms")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            observed_latency = float(elapsed)
    if observed_latency is not None:
        if (
            not isinstance(observed_latency, (int, float))
            or isinstance(observed_latency, bool)
            or float(observed_latency) < 0
        ):
            raise ValueError("latency_ms must be a non-negative number")
        observed_latency = float(observed_latency)
    failure_code = _result_field(result, "failure_code", None)
    fallback = bool(_result_field(result, "fallback", False))
    return QuestionPrediction(
        question_id=question_id,
        answer=str(answer or ""),
        status=status,  # type: ignore[arg-type]
        citations=citations,
        candidates=candidates,
        packed=packed,
        latency_ms=observed_latency,
        input_tokens=int(total_in),
        output_tokens=int(total_out),
        estimated_cost_usd=None,
        failure_code=failure_code,
        fallback=fallback,
        degraded=False,
        triples=[],
        reranked=False,
        strategy=DECOMPOSITION_STRATEGY,
    )


def freeze_decomposition_selection(
    baseline: RunEvaluation,
    candidate: RunEvaluation,
    tuning_ids: list[str],
    *,
    selection_id: str,
    decomposition_version: str = "decomposition-v1",
    decomposition_prompt_version: str = "decomposition-planner-v1",
) -> DecompositionSelection:
    """Freeze the tuning choice for decomposition before any locked use (R13).

    Both runs must carry ``split == "tuning"`` and cover exactly ``tuning_ids``;
    the candidate must use strategy ``decomposition``. Fixed paired fields must
    match (see ``check_run_compatibility``) and final budgets must match, or no
    selection is frozen. The decision rule is the shared PRD rule
    (``check_tuning_selection``): select only on tuning F1 improvement without
    all-support worsening; cost acceptability stays an operator judgement.
    """
    if not isinstance(selection_id, str) or not selection_id.strip():
        raise ValueError("selection_id must be a nonempty string")
    if not isinstance(tuning_ids, list) or not tuning_ids:
        raise ValueError("tuning_ids must be a nonempty list")
    if len(set(tuning_ids)) != len(tuning_ids):
        raise ValueError("tuning_ids must be unique")
    for run, label in ((baseline, "baseline"), (candidate, "candidate")):
        if not isinstance(run, RunEvaluation):
            raise ValueError(f"{label} must be a RunEvaluation")
        if run.config.split != "tuning":
            raise ValueError(
                "Decomposition selection must compare tuning runs; "
                f"{label} split is {run.config.split!r}"
            )
    if candidate.config.strategy != DECOMPOSITION_STRATEGY:
        raise ValueError(
            "Decomposition selection candidate must use strategy "
            f"{DECOMPOSITION_STRATEGY!r}, got {candidate.config.strategy!r}"
        )
    if baseline.config.strategy == DECOMPOSITION_STRATEGY:
        raise ValueError("Decomposition selection baseline must not be the decomposition arm")
    expected_hash = hash_question_set(sorted(tuning_ids))
    for run, label in ((baseline, "baseline"), (candidate, "candidate")):
        if run.config.question_set_hash != expected_hash:
            raise ValueError(
                f"{label} question_set_hash does not match tuning_ids; "
                "compare on the tuning set before freezing"
            )
        if run.config.question_count != len(tuning_ids):
            raise ValueError(f"{label} question_count must match the tuning set")
        if run.aggregate.n_total != len(tuning_ids):
            raise ValueError(f"{label} aggregate n_total must match the tuning set")
    check = check_run_compatibility([baseline.config, candidate.config])
    if not check["compatible"]:
        raise ValueError(
            "Tuning arms are incompatible; refusing to freeze decomposition selection: "
            f"{check['reason']}"
        )
    if not isinstance(decomposition_version, str) or not decomposition_version.strip():
        raise ValueError("decomposition_version must be a nonempty string")
    if (
        not isinstance(decomposition_prompt_version, str)
        or not decomposition_prompt_version.strip()
    ):
        raise ValueError("decomposition_prompt_version must be a nonempty string")
    decision = check_tuning_selection(baseline, candidate)
    return DecompositionSelection(
        selection_id=selection_id.strip(),
        baseline_run_id=baseline.config.run_id,
        candidate_run_id=candidate.config.run_id,
        split="tuning",
        select_candidate=bool(decision["select_candidate"]),
        f1_delta_pp=float(decision["f1_delta_pp"]),
        all_support_delta_pp=float(decision["all_support_delta_pp"]),
        reasons=list(decision["reasons"]),
        tuning_question_ids=sorted(tuning_ids),
        tuning_question_set_hash=expected_hash,
        baseline_strategy=baseline.config.strategy,
        candidate_strategy=candidate.config.strategy,
        fixed=dict(check["fixed"]),
        max_chunks=int(baseline.config.max_chunks),
        max_context_tokens=int(baseline.config.max_context_tokens),
        decomposition_version=decomposition_version.strip(),
        decomposition_prompt_version=decomposition_prompt_version.strip(),
        frozen_at=datetime.now(UTC).isoformat(),
    )


def assert_decomposition_locked_use(
    selection: DecompositionSelection,
    locked_baseline_config: RunConfig,
    locked_candidate_config: RunConfig,
    locked_ids: list[str],
    expected_locked_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Require the locked run to reuse the frozen decomposition selection (R13).

    Locked configs must carry ``split == "evaluation"``, reuse the frozen
    strategies, corpus/index fingerprints, models and final budgets (only the
    question set hash and split may change), and cover exactly the frozen
    locked ids. Returns the locked hashes without running any system.
    """
    if not isinstance(selection, DecompositionSelection):
        raise ValueError("selection must be a DecompositionSelection")
    for cfg, label in (
        (locked_baseline_config, "locked baseline"),
        (locked_candidate_config, "locked candidate"),
    ):
        if not isinstance(cfg, RunConfig):
            raise ValueError(f"{label} must be a RunConfig")
        if cfg.split != "evaluation":
            raise ValueError(
                "Locked decomposition use must run on the evaluation split; "
                f"{label} split is {cfg.split!r}"
            )
    if locked_baseline_config.strategy != selection.baseline_strategy:
        raise ValueError(
            f"Locked baseline strategy {locked_baseline_config.strategy!r} does not match "
            f"the frozen selection {selection.baseline_strategy!r}"
        )
    if locked_candidate_config.strategy != selection.candidate_strategy:
        raise ValueError(
            f"Locked candidate strategy {locked_candidate_config.strategy!r} does not match "
            f"the frozen selection {selection.candidate_strategy!r}"
        )
    if locked_candidate_config.strategy != DECOMPOSITION_STRATEGY:
        raise ValueError(
            f"Frozen candidate must stay {DECOMPOSITION_STRATEGY!r}; "
            "refusing to substitute another arm on locked data"
        )
    check = check_run_compatibility([locked_baseline_config, locked_candidate_config])
    if not check["compatible"]:
        raise ValueError(
            f"Locked decomposition arms are incompatible; refusing locked use: {check['reason']}"
        )
    frozen_fixed = dict(selection.fixed)
    locked_fixed = dict(check["fixed"])
    for key, frozen_value in frozen_fixed.items():
        if key == "question_set_hash":
            continue
        if locked_fixed.get(key) != frozen_value:
            raise ValueError(
                f"Locked {key} {locked_fixed.get(key)!r} differs from the frozen tuning "
                f"selection {frozen_value!r}; reuse the frozen configuration on locked data"
            )
    if not isinstance(locked_ids, list) or not locked_ids:
        raise ValueError("locked_ids must be a nonempty list")
    if expected_locked_ids is not None:
        locked_hash = assert_locked_question_use(list(locked_ids), list(expected_locked_ids))
    else:
        locked_hash = hash_question_set(sorted(locked_ids))
        for cfg, label in (
            (locked_baseline_config, "locked baseline"),
            (locked_candidate_config, "locked candidate"),
        ):
            if cfg.question_set_hash != locked_hash:
                raise ValueError(f"{label} question_set_hash does not match locked_ids")
            if cfg.question_count != len(locked_ids):
                raise ValueError(f"{label} question_count must match locked_ids")
    return {
        "selection_id": selection.selection_id,
        "select_candidate": selection.select_candidate,
        "tuning_question_set_hash": selection.tuning_question_set_hash,
        "locked_question_set_hash": locked_hash,
        "locked_question_ids": sorted(locked_ids),
        "compatibility_hash": check["compatibility_hash"],
        "fixed": locked_fixed,
    }


def summarize_decomposition_incremental(
    comparison: dict[str, Any],
    baseline: RunEvaluation,
    candidate: RunEvaluation,
) -> dict[str, Any]:
    """Report the additive decomposition delta (R13): F1, recall, latency, cost.

    Quality deltas reuse the paired bootstrap intervals from ``compare_runs``;
    latency deltas are observed p50/p95 differences only (no interval);
    query-token and cost deltas are observed sums with unknown money staying
    unknown (None), never zero. The payload records the frozen run ids and
    compatibility hash so the delta never replaces the saved A-D results.
    """
    if not isinstance(comparison, dict) or not comparison.get("compatible"):
        raise ValueError(
            "Decomposition delta requires a compatible compare_runs result: "
            f"{comparison.get('reason') if isinstance(comparison, dict) else comparison!r}"
        )
    if candidate.config.strategy != DECOMPOSITION_STRATEGY:
        raise ValueError(
            f"Decomposition delta candidate must use {DECOMPOSITION_STRATEGY!r}, "
            f"got {candidate.config.strategy!r}"
        )
    metrics = comparison.get("metrics") or {}
    incremental: dict[str, Any] = {}
    for name in (
        "exact_match",
        "f1",
        "paragraph_recall_packed",
        "all_support_packed",
        "sentence_recall_packed",
    ):
        payload = metrics.get(name) or {}
        incremental[name] = {
            "mean_baseline": float(payload.get("mean_a", 0.0)),
            "mean_candidate": float(payload.get("mean_b", 0.0)),
            "delta_pp": float(payload.get("observed_delta_pp", 0.0)),
            "lower_pp": payload.get("lower_pp"),
            "upper_pp": payload.get("upper_pp"),
            "n": payload.get("n"),
        }
    agg_a = baseline.aggregate
    agg_b = candidate.aggregate
    if agg_a.n_total != agg_b.n_total:
        raise ValueError("Baseline and candidate aggregates cover different denominators")
    latency = {
        "baseline_p50_ms": agg_a.latency_p50_ms,
        "candidate_p50_ms": agg_b.latency_p50_ms,
        "delta_p50_ms": (
            float(agg_b.latency_p50_ms - agg_a.latency_p50_ms)
            if agg_a.latency_p50_ms is not None and agg_b.latency_p50_ms is not None
            else None
        ),
        "baseline_p95_ms": agg_a.latency_p95_ms,
        "candidate_p95_ms": agg_b.latency_p95_ms,
        "delta_p95_ms": (
            float(agg_b.latency_p95_ms - agg_a.latency_p95_ms)
            if agg_a.latency_p95_ms is not None and agg_b.latency_p95_ms is not None
            else None
        ),
        "note": "Observed latency only; no paired interval is computed for latency.",
    }
    cost_a = agg_a.total_estimated_cost_usd
    cost_b = agg_b.total_estimated_cost_usd
    cost = {
        "baseline_input_tokens": int(agg_a.total_input_tokens),
        "candidate_input_tokens": int(agg_b.total_input_tokens),
        "delta_input_tokens": int(agg_b.total_input_tokens - agg_a.total_input_tokens),
        "baseline_output_tokens": int(agg_a.total_output_tokens),
        "candidate_output_tokens": int(agg_b.total_output_tokens),
        "delta_output_tokens": int(agg_b.total_output_tokens - agg_a.total_output_tokens),
        "baseline_estimated_cost_usd": cost_a,
        "candidate_estimated_cost_usd": cost_b,
        "delta_estimated_cost_usd": (
            float(cost_b - cost_a) if cost_a is not None and cost_b is not None else None
        ),
        "cost_label_baseline": (
            f"${float(cost_a):.6f} (estimated)" if cost_a is not None else "unknown"
        ),
        "cost_label_candidate": (
            f"${float(cost_b):.6f} (estimated)" if cost_b is not None else "unknown"
        ),
        "note": (
            "Unknown money stays unknown, never zero. Cost acceptability is an "
            "operator judgement; this summary never invents a cost verdict."
        ),
    }
    return {
        "decomposition_eval_version": DECOMPOSITION_EVAL_VERSION,
        "evaluation_version": comparison.get("evaluation_version", EVALUATION_VERSION),
        "baseline_run": baseline.config.run_id,
        "baseline_strategy": baseline.config.strategy,
        "candidate_run": candidate.config.run_id,
        "candidate_strategy": candidate.config.strategy,
        "n": comparison.get("n"),
        "question_ids": list(comparison.get("question_ids") or []),
        "compatibility_hash": comparison.get("compatibility_hash"),
        "incremental": incremental,
        "latency": latency,
        "cost": cost,
        "failures": comparison.get("failures"),
        "warnings": list(comparison.get("warnings") or [])
        + [
            "Additive report only: this delta never replaces the frozen A-D results.",
            "Decomposition adds planner + sub-answer calls; extra tokens/latency are "
            "part of the result under matched final budgets.",
        ],
    }


def decomposition_report_limitations() -> list[str]:
    """Standard limitations block for decomposition delta exports (R13)."""
    return [
        "Decomposition (E) is optional and additive; frozen A-D results are unchanged.",
        "Selection freezes on the tuning set before any locked use; locked runs must "
        "reuse the frozen corpus/index, models and final budgets.",
        "Planner plus sub-answer calls add query tokens/latency under matched final "
        "budgets (top 8 chunks / 3000 tokens); extra spend is part of the result.",
        "Paired deltas describe the observed sets; shared evidence and small samples "
        "limit inference to population-wide significance.",
        "Citation validity means the citation resolves to packed evidence, not entailment.",
        "Costs cover LLM usage when reported; local compute time is measured, not priced.",
    ]
