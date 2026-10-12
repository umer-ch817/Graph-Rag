"""Baseline scoring for T06 (evaluation).

Answer scoring follows the official HotpotQA evaluator semantics
(``hotpot_evaluate_v1.py``): lowercase, punctuation removal, article
removal, whitespace fix, then exact match plus token F1 with explicit
yes/no/noanswer handling. Retrieval coverage is scored from the actual
packed evidence chunks -- not from titles alone -- with deduplication of
repeated and overlapping evidence.
"""

from __future__ import annotations

import collections
import re
import string
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCORING_VERSION = "hotpot-scoring-v1"


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class RetrievalScores(Record):
    """Coverage of one question's gold facts by one evidence list."""

    paragraph_recall: float = Field(ge=0.0, le=1.0)
    all_support: float = Field(ge=0.0, le=1.0)
    sentence_recall: float = Field(ge=0.0, le=1.0)
    n_gold_paragraphs: int = Field(ge=0, strict=True)
    n_hit_paragraphs: int = Field(ge=0, strict=True)
    n_gold_sentences: int = Field(ge=0, strict=True)
    n_hit_sentences: int = Field(ge=0, strict=True)
    k: int = Field(ge=0, strict=True)


class AnswerScores(Record):
    """Official HotpotQA answer scores for one question (0..1 scale)."""

    exact_match: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)


def normalize_answer(text: str) -> str:
    """Official HotpotQA answer normalization.

    Lowercases, removes punctuation, removes articles (a/an/the) and fixes
    whitespace. Mirrors ``normalize_answer`` in the official evaluator so
    fixture scores reconcile exactly.
    """
    if not isinstance(text, str):
        raise ValueError("Answer must be a string")

    def _lower(value: str) -> str:
        return value.lower()

    def _remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def _remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def _white_space_fix(value: str) -> str:
        return " ".join(value.split())

    return _white_space_fix(_remove_articles(_remove_punc(_lower(text))))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """True when normalized answers match exactly."""
    if not isinstance(prediction, str) or not isinstance(ground_truth, str):
        raise ValueError("Prediction and ground truth must be strings")
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    """Official HotpotQA token F1 in 0..1.

    Yes/no/noanswer answers score 0 unless both sides normalize to the same
    special token; otherwise standard token-overlap F1 applies.
    """
    if not isinstance(prediction, str) or not isinstance(ground_truth, str):
        raise ValueError("Prediction and ground truth must be strings")
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    if normalized_prediction in ("yes", "no", "noanswer") and (
        normalized_prediction != normalized_ground_truth
    ):
        return 0.0
    if normalized_ground_truth in ("yes", "no", "noanswer") and (
        normalized_ground_truth != normalized_prediction
    ):
        return 0.0
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def score_answer(prediction: str, ground_truth: str) -> AnswerScores:
    """Score one answer pair; returns EM as 0.0/1.0 and F1 in 0..1."""
    return AnswerScores(
        exact_match=1.0 if exact_match_score(prediction, ground_truth) else 0.0,
        f1=float(f1_score(prediction, ground_truth)),
    )


def _gold_keys(fact: dict[str, Any]) -> tuple[str | None, str | None, int | None]:
    title = fact.get("title")
    paragraph_id = fact.get("paragraph_id")
    sentence_id = fact.get("sentence_id")
    title_key = str(title) if isinstance(title, str) and title else None
    paragraph_key = str(paragraph_id) if isinstance(paragraph_id, str) and paragraph_id else None
    sentence_key: int | None = None
    if isinstance(sentence_id, int) and not isinstance(sentence_id, bool):
        sentence_key = sentence_id
    elif isinstance(sentence_id, str) and sentence_id.isdigit():
        sentence_key = int(sentence_id)
    return paragraph_key, title_key, sentence_key


def _evidence_keys(item: Any) -> tuple[str | None, str | None, list[int]]:
    if isinstance(item, dict):
        raw = item
        paragraph_id = raw.get("paragraph_id")
        title = raw.get("title")
        sentence_ids = raw.get("sentence_ids")
        chunk_id = raw.get("chunk_id", raw.get("id"))
    else:
        paragraph_id = getattr(item, "paragraph_id", None)
        title = getattr(item, "title", None)
        sentence_ids = getattr(item, "sentence_ids", None)
        chunk_id = getattr(item, "chunk_id", getattr(item, "id", None))
    _ = chunk_id  # chunk identity is irrelevant; coverage dedupes by (paragraph, sentence)
    paragraph_key = str(paragraph_id) if isinstance(paragraph_id, str) and paragraph_id else None
    title_key = str(title) if isinstance(title, str) and title else None
    sentences: list[int] = []
    if isinstance(sentence_ids, list):
        for value in sentence_ids:
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                sentences.append(value)
            else:
                raise ValueError("Evidence sentence_ids must be non-negative integers")
    else:
        raise ValueError("Evidence chunk must carry a sentence_ids list")
    return paragraph_key, title_key, sentences


def score_retrieval(
    gold_facts: list[dict[str, Any]],
    evidence: list[Any],
) -> RetrievalScores:
    """Score paragraph recall, all-support coverage and sentence recall.

    Gold facts use ``paragraph_id`` when present (preserving Town-style title
    variants) with a ``title`` fallback. Evidence is the actual packed or
    candidate chunk list; repeated chunks and overlapping sentences are
    deduplicated through sets, never double-counted. ``k`` records the number
    of evidence chunks supplied.
    """
    if not isinstance(gold_facts, list) or not gold_facts:
        raise ValueError("gold_facts must be a nonempty list")
    if not isinstance(evidence, list):
        raise ValueError("evidence must be a list")

    gold_paragraphs: set[str] = set()
    gold_sentences: set[tuple[str, int]] = set()
    gold_by_paragraph_id: dict[str, str] = {}
    for fact in gold_facts:
        if not isinstance(fact, dict):
            raise ValueError("Each gold fact must be a mapping")
        paragraph_key, title_key, sentence_key = _gold_keys(fact)
        if sentence_key is None or sentence_key < 0:
            raise ValueError("Each gold fact needs a non-negative sentence_id")
        if paragraph_key is not None:
            gold_paragraphs.add(f"pid:{paragraph_key}")
            gold_sentences.add((f"pid:{paragraph_key}", sentence_key))
            if title_key is not None:
                gold_by_paragraph_id[f"pid:{paragraph_key}"] = title_key
        elif title_key is not None:
            gold_paragraphs.add(f"title:{title_key}")
            gold_sentences.add((f"title:{title_key}", sentence_key))
        else:
            raise ValueError("Each gold fact needs a paragraph_id or title")

    retrieved_paragraphs: set[str] = set()
    retrieved_sentences: set[tuple[str, int]] = set()
    retrieved_titles: set[str] = set()
    retrieved_pids: set[str] = set()
    for item in evidence:
        paragraph_key, title_key, sentences = _evidence_keys(item)
        if paragraph_key is not None:
            retrieved_pids.add(paragraph_key)
            retrieved_paragraphs.add(f"pid:{paragraph_key}")
            for sid in sentences:
                retrieved_sentences.add((f"pid:{paragraph_key}", sid))
        if title_key is not None:
            retrieved_titles.add(title_key)
            # Title-level pairs support corpora where paragraph ids are absent.
            # When paragraph ids exist on both sides, pid pairs are authoritative
            # and title pairs only supplement title-only gold facts.
            for sid in sentences:
                retrieved_sentences.add((f"title:{title_key}", sid))
        if paragraph_key is None and title_key is None:
            raise ValueError("Each evidence chunk needs a paragraph_id or title")
        if paragraph_key is None and title_key is not None:
            retrieved_paragraphs.add(f"title:{title_key}")

    # A gold paragraph whose paragraph_id is present hits directly. Title-only
    # gold paragraphs hit when the title appears in any evidence chunk.
    hit_paragraphs = 0
    for key in gold_paragraphs:
        if key in retrieved_paragraphs:
            hit_paragraphs += 1
        elif key.startswith("title:") and key[6:] in retrieved_titles:
            hit_paragraphs += 1

    hit_sentences = len(gold_sentences & retrieved_sentences)

    n_gold_paragraphs = len(gold_paragraphs)
    n_gold_sentences = len(gold_sentences)
    paragraph_recall = hit_paragraphs / n_gold_paragraphs if n_gold_paragraphs else 0.0
    sentence_recall = hit_sentences / n_gold_sentences if n_gold_sentences else 0.0
    all_support = 1.0 if hit_paragraphs == n_gold_paragraphs and n_gold_paragraphs else 0.0
    return RetrievalScores(
        paragraph_recall=paragraph_recall,
        all_support=all_support,
        sentence_recall=sentence_recall,
        n_gold_paragraphs=n_gold_paragraphs,
        n_hit_paragraphs=hit_paragraphs,
        n_gold_sentences=n_gold_sentences,
        n_hit_sentences=hit_sentences,
        k=len(evidence),
    )
