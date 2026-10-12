"""Baseline scoring and artifact export (T06, checkpoint C2)."""

import json
from pathlib import Path

import pytest

from graphrag_lab.answering import AnswerConfig, answer_question, pack_context
from graphrag_lab.evaluation import (
    LabelView,
    QuestionPrediction,
    RunConfig,
    aggregate_questions,
    evaluate_predictions,
    hash_question_set,
    score_one_question,
    summarize_success_only,
)
from graphrag_lab.llm import FakeLlmBackend
from graphrag_lab.reports import export_baseline, render_markdown_summary
from graphrag_lab.scoring import (
    exact_match_score,
    f1_score,
    normalize_answer,
    score_answer,
    score_retrieval,
)


def ada_packed():
    return [
        {
            "chunk_id": "c_ada",
            "paragraph_id": "p_ada",
            "title": "Ada Lovelace",
            "text": "Ada Lovelace wrote the first algorithm.",
            "sentence_ids": [0, 1],
            "sentence_texts": [
                "Ada Lovelace wrote the first algorithm.",
                "She worked with Charles Babbage.",
            ],
        }
    ]


def ada_label():
    return {
        "question_id": "q_ada",
        "answer": "Ada Lovelace",
        "supporting_facts": [{"title": "Ada Lovelace", "sentence_id": 0, "paragraph_id": "p_ada"}],
    }


def test_normalization_matches_official_semantics():
    assert normalize_answer("The Eiffel Tower") == "eiffel tower"
    assert normalize_answer("  Ada   Lovelace! ") == "ada lovelace"
    assert normalize_answer("An Apple, a Banana & THE cherry.") == "apple banana cherry"
    assert normalize_answer("YES") == "yes"
    assert normalize_answer("NoAnswer") == "noanswer"


def test_official_em_and_f1_fixtures():
    assert exact_match_score("Ada Lovelace", "ada lovelace!")
    assert exact_match_score("The Eiffel Tower", "eiffel tower")
    assert not exact_match_score("Paris", "London")
    assert score_answer("Ada Lovelace", "Ada Lovelace").f1 == pytest.approx(1.0)
    assert score_answer("Paris", "London").f1 == pytest.approx(0.0)
    # Partial overlap: "capital of France" (3 tokens) vs "France capital"
    # (2 tokens) shares 2 tokens -> P=1.0, R=2/3, F1=0.8.
    partial = f1_score("capital of France", "France capital")
    assert partial == pytest.approx(0.8)


def test_yes_no_noanswer_handling_matches_official():
    assert f1_score("yes", "yes") == pytest.approx(1.0)
    assert f1_score("yes", "no") == pytest.approx(0.0)
    assert f1_score("no", "yes") == pytest.approx(0.0)
    assert f1_score("yes", "Ada Lovelace") == pytest.approx(0.0)
    assert f1_score("Ada Lovelace", "yes") == pytest.approx(0.0)
    assert f1_score("noanswer", "noanswer") == pytest.approx(1.0)
    assert exact_match_score("YES", "yes")
    assert not exact_match_score("yes", "no")


def test_retrieval_scores_actual_packed_sentences_not_titles():
    gold = [
        {"title": "Ada Lovelace", "sentence_id": 0, "paragraph_id": "p_ada"},
        {"title": "Eiffel Tower", "sentence_id": 1, "paragraph_id": "p_eiffel"},
    ]
    evidence = ada_packed()
    scores = score_retrieval(gold, evidence)
    assert scores.n_gold_paragraphs == 2
    assert scores.n_hit_paragraphs == 1
    assert scores.paragraph_recall == pytest.approx(0.5)
    assert scores.all_support == pytest.approx(0.0)
    assert scores.n_gold_sentences == 2
    assert scores.n_hit_sentences == 1
    assert scores.sentence_recall == pytest.approx(0.5)
    assert scores.k == 1


def test_all_support_requires_every_gold_paragraph():
    gold = ada_label()["supporting_facts"]
    full = score_retrieval(gold, ada_packed())
    assert full.all_support == pytest.approx(1.0)
    assert full.paragraph_recall == pytest.approx(1.0)
    assert full.sentence_recall == pytest.approx(1.0)
    empty = score_retrieval(gold, [])
    assert empty.all_support == pytest.approx(0.0)
    assert empty.paragraph_recall == pytest.approx(0.0)
    assert empty.sentence_recall == pytest.approx(0.0)


def test_title_variants_do_not_cross_match():
    gold = [{"title": "Town", "sentence_id": 0, "paragraph_id": "p_town_a"}]
    wrong_variant = [
        {
            "chunk_id": "c_town_b",
            "paragraph_id": "p_town_b",
            "title": "Town",
            "sentence_ids": [0],
        }
    ]
    scores = score_retrieval(gold, wrong_variant)
    assert scores.paragraph_recall == pytest.approx(0.0)
    assert scores.all_support == pytest.approx(0.0)
    assert scores.sentence_recall == pytest.approx(0.0)


def test_repeated_and_overlapping_evidence_dedupes():
    gold = ada_label()["supporting_facts"]
    duplicated = ada_packed() + ada_packed()
    scores = score_retrieval(gold, duplicated)
    assert scores.n_hit_paragraphs == 1
    assert scores.n_hit_sentences == 1
    assert scores.paragraph_recall == pytest.approx(1.0)
    # Overlapping chunks covering sentence 0 twice still count once.
    overlapping = [
        {"chunk_id": "c1", "paragraph_id": "p_ada", "title": "Ada", "sentence_ids": [0]},
        {"chunk_id": "c2", "paragraph_id": "p_ada", "title": "Ada", "sentence_ids": [0, 1]},
    ]
    overlapped = score_retrieval(gold, overlapping)
    assert overlapped.n_hit_sentences == 1
    assert overlapped.k == 2


def test_candidate_and_packed_stages_score_separately():
    gold = ada_label()["supporting_facts"]
    missing = [
        {"chunk_id": "c_other", "paragraph_id": "p_other", "title": "Other", "sentence_ids": [0]}
    ]
    candidate = score_retrieval(gold, ada_packed())
    packed = score_retrieval(gold, missing)
    assert candidate.all_support == pytest.approx(1.0)
    assert packed.all_support == pytest.approx(0.0)


def three_question_fixture():
    labels = [
        LabelView(
            question_id="q_success",
            answer="Ada Lovelace",
            supporting_facts=[{"title": "Ada Lovelace", "sentence_id": 0, "paragraph_id": "p_ada"}],
            question_type="bridge",
        ),
        LabelView(
            question_id="q_miss",
            answer="Paris",
            supporting_facts=[
                {"title": "Eiffel Tower", "sentence_id": 0, "paragraph_id": "p_eiffel"}
            ],
            question_type="comparison",
        ),
        LabelView(
            question_id="q_failed",
            answer="Paris",
            supporting_facts=[
                {"title": "Eiffel Tower", "sentence_id": 0, "paragraph_id": "p_eiffel"}
            ],
            question_type="bridge",
        ),
    ]
    predictions = [
        QuestionPrediction(
            question_id="q_success",
            answer="Ada Lovelace",
            status="supported",
            citations=[{"chunk_id": "c_ada", "sentence_ids": [0]}],
            candidates=[
                {
                    "chunk_id": "c_ada",
                    "paragraph_id": "p_ada",
                    "title": "Ada Lovelace",
                    "sentence_ids": [0, 1],
                }
            ],
            packed=[
                {
                    "chunk_id": "c_ada",
                    "paragraph_id": "p_ada",
                    "title": "Ada Lovelace",
                    "sentence_ids": [0, 1],
                }
            ],
            latency_ms=120.0,
            input_tokens=200,
            output_tokens=10,
        ),
        QuestionPrediction(
            question_id="q_miss",
            answer="London",
            status="supported",
            citations=[{"chunk_id": "c_other", "sentence_ids": [0]}],
            candidates=[
                {
                    "chunk_id": "c_other",
                    "paragraph_id": "p_other",
                    "title": "Other",
                    "sentence_ids": [0],
                }
            ],
            packed=[
                {
                    "chunk_id": "c_other",
                    "paragraph_id": "p_other",
                    "title": "Other",
                    "sentence_ids": [0],
                }
            ],
            latency_ms=140.0,
            input_tokens=200,
            output_tokens=10,
        ),
        QuestionPrediction(
            question_id="q_failed",
            answer="",
            status="failed",
            citations=[],
            candidates=[],
            packed=[],
            latency_ms=None,
            failure_code="PROVIDER_ERROR",
        ),
    ]
    return labels, predictions


def run_config_for(ids: list[str]) -> RunConfig:
    return RunConfig(
        run_id="baseline-fixture-v1",
        strategy="vector",
        corpus_id="hp_fixture_t06_v1",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        answer_model="fake-llm-test",
        prompt_version="answer-prompt-v1",
        answer_version="answer-v1",
        token_estimator="char-word-max-v1",
        num_candidates=30,
        max_chunks=8,
        max_context_tokens=3000,
        question_set_hash=hash_question_set(ids),
        question_count=len(ids),
        split="tuning",
    )


def test_fixed_denominator_includes_failed_questions():
    labels, predictions = three_question_fixture()
    scored = [score_one_question(label, pred) for label, pred in zip(labels, predictions)]
    by_id = {item.question_id: item for item in scored}
    assert by_id["q_success"].exact_match == pytest.approx(1.0)
    assert by_id["q_miss"].exact_match == pytest.approx(0.0)
    assert by_id["q_failed"].exact_match == pytest.approx(0.0)
    assert by_id["q_failed"].f1 == pytest.approx(0.0)
    assert by_id["q_failed"].status == "failed"
    assert by_id["q_success"].citations_resolve is True
    aggregate = aggregate_questions(scored)
    assert aggregate.n_total == 3
    assert aggregate.n_failed == 1
    assert aggregate.n_success == 2
    assert aggregate.n_abstained == 0
    # Fixed denominator: (1 + 0 + 0) / 3.
    assert aggregate.exact_match_mean == pytest.approx(100.0 / 3)
    assert aggregate.f1_mean == pytest.approx(100.0 / 3)
    # Success-only diagnostics exclude the failure: (1 + 0) / 2.
    success_only = summarize_success_only(scored)
    assert success_only.n_success == 2
    assert success_only.exact_match_mean == pytest.approx(50.0)


def test_evaluate_validates_frozen_question_set():
    labels, predictions = three_question_fixture()
    config = run_config_for([label.question_id for label in labels])
    scored, evaluation = evaluate_predictions(labels, predictions, config)
    assert len(scored) == 3
    assert evaluation.aggregate.n_total == 3
    assert evaluation.config.question_set_hash == hash_question_set(
        ["q_failed", "q_miss", "q_success"]
    )
    with pytest.raises(ValueError, match="must match exactly"):
        evaluate_predictions(labels, predictions[:2], config)
    wrong = config.model_copy(update={"question_count": 99})
    with pytest.raises(ValueError, match="question_count"):
        evaluate_predictions(labels, predictions, wrong)


def test_vector_baseline_runs_end_to_end_with_resolving_citations():
    question = "Who wrote the first algorithm?"
    evidence = [
        {
            "chunk_id": "c_ada",
            "paragraph_id": "p_ada",
            "title": "Ada Lovelace",
            "text": "Ada Lovelace wrote the first algorithm. She worked with Charles Babbage.",
            "sentence_ids": [0, 1],
            "sentence_texts": [
                "Ada Lovelace wrote the first algorithm.",
                "She worked with Charles Babbage.",
            ],
            "rank": 1,
        }
    ]
    backend = FakeLlmBackend(
        [
            {
                "answer": "Ada Lovelace",
                "status": "supported",
                "citations": [{"chunk_id": "c_ada", "sentence_ids": [0]}],
            }
        ]
    )
    result = answer_question(question, evidence, backend, AnswerConfig())
    assert result.status == "supported"
    packed = pack_context(question, evidence, AnswerConfig())
    label = LabelView(
        question_id="q_ada",
        answer="Ada Lovelace",
        supporting_facts=[{"title": "Ada Lovelace", "sentence_id": 0, "paragraph_id": "p_ada"}],
    )
    prediction = QuestionPrediction(
        question_id="q_ada",
        answer=result.answer,
        status=result.status,
        citations=[c.model_dump() for c in result.citations],
        candidates=[
            {
                "chunk_id": c.chunk_id,
                "paragraph_id": c.paragraph_id,
                "title": c.title,
                "sentence_ids": list(c.sentence_ids),
            }
            for c in result.packed.chunks
        ],
        packed=[
            {
                "chunk_id": c.chunk_id,
                "paragraph_id": c.paragraph_id,
                "title": c.title,
                "sentence_ids": list(c.sentence_ids),
            }
            for c in packed.chunks
        ],
        latency_ms=50.0,
    )
    scored = score_one_question(label, prediction)
    assert scored.exact_match == pytest.approx(1.0)
    assert scored.packed.all_support == pytest.approx(1.0)
    assert scored.citations_resolve is True


def test_export_fixture_baseline_and_reconcile(tmp_path: Path):
    labels, predictions = three_question_fixture()
    config = run_config_for([label.question_id for label in labels])
    scored, evaluation = evaluate_predictions(labels, predictions, config)
    out = tmp_path / "baseline"
    info = export_baseline(
        out,
        evaluation,
        scored,
        predictions,
        labels,
        limitations=["Fixture export for T06 verification."],
    )
    for name in ("results.jsonl", "aggregate.json", "manifest.json", "summary.md"):
        assert (out / name).is_file()
    assert set(info["artifact_sha256"]) == {
        "results.jsonl",
        "aggregate.json",
        "manifest.json",
        "summary.md",
    }
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    by_id = {row["question_id"]: row for row in rows}
    # Manual reconciliation: one success, one miss, one failed request.
    assert by_id["q_success"]["exact_match"] == pytest.approx(1.0)
    assert by_id["q_success"]["packed_recall"]["all_support"] == pytest.approx(1.0)
    assert by_id["q_miss"]["exact_match"] == pytest.approx(0.0)
    assert by_id["q_miss"]["packed_recall"]["all_support"] == pytest.approx(0.0)
    assert by_id["q_failed"]["status"] == "failed"
    assert by_id["q_failed"]["failure_code"] == "PROVIDER_ERROR"
    assert by_id["q_failed"]["exact_match"] == pytest.approx(0.0)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"]["corpus_id"] == "hp_fixture_t06_v1"
    assert manifest["config"]["strategy"] == "vector"
    assert manifest["question_set_hash"] == config.question_set_hash
    assert manifest["artifact_sha256"]["results.jsonl"] == info["artifact_sha256"]["results.jsonl"]
    aggregate = json.loads((out / "aggregate.json").read_text())
    assert aggregate["aggregate"]["n_total"] == 3
    assert aggregate["aggregate"]["n_failed"] == 1
    summary = (out / "summary.md").read_text()
    assert "baseline-fixture-v1" in summary
    assert "Failed / abstained / supported" in summary
    # Re-export is idempotent; differing content is refused, never overwritten.
    second = export_baseline(
        out,
        evaluation,
        scored,
        predictions,
        labels,
        limitations=["Fixture export for T06 verification."],
    )
    assert second["artifact_sha256"] == info["artifact_sha256"]
    _ = render_markdown_summary(evaluation, scored)
