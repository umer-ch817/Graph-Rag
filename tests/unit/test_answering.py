"""Grounded vector answers with validated citations (T05)."""

import json
from pathlib import Path

import pytest

from graphrag_lab.answering import (
    ABSTAIN_ANSWER,
    AnswerConfig,
    answer_question,
    build_answer_messages,
    pack_context,
)
from graphrag_lab.config import BudgetConfig
from graphrag_lab.llm import (
    BudgetExceededError,
    BudgetTracker,
    FakeLlmBackend,
    InvalidInputError,
    LlmRequest,
    ValidationFailedError,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "llm_outputs.json"
QUESTION = "Who wrote the first algorithm?"


def fixture_cases():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def evidence():
    return [
        {
            "chunk_id": "c_ada",
            "paragraph_id": "p_ada",
            "title": "Ada Lovelace",
            "text": "Ada Lovelace wrote the first algorithm. "
            "She worked with Charles Babbage on the Analytical Engine.",
            "sentence_ids": [0, 1],
            "sentence_texts": [
                "Ada Lovelace wrote the first algorithm.",
                "She worked with Charles Babbage on the Analytical Engine.",
            ],
            "rank": 1,
        },
        {
            "chunk_id": "c_eiffel",
            "paragraph_id": "p_eiffel",
            "title": "Eiffel Tower",
            "text": "The Eiffel Tower is in Paris. It was completed in 1889.",
            "sentence_ids": [0, 1],
            "sentence_texts": [
                "The Eiffel Tower is in Paris.",
                "It was completed in 1889.",
            ],
            "rank": 2,
        },
    ]


def test_fixture_file_matches_answer_schema():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1"
    assert payload["prompt_version"] == "answer-prompt-v1"
    cases = payload["cases"]
    assert {"supported_ada", "insufficient", "invented_chunk"} <= set(cases)
    assert cases["supported_ada"]["status"] == "supported"
    assert cases["supported_ada"]["citations"][0]["chunk_id"] == "c_ada"
    assert cases["insufficient"] == {
        "answer": ABSTAIN_ANSWER,
        "status": "insufficient_evidence",
        "citations": [],
    }


def test_supported_answer_returns_status_and_valid_citations():
    cases = fixture_cases()
    backend = FakeLlmBackend([cases["supported_ada"]])
    result = answer_question(QUESTION, evidence(), backend, AnswerConfig())
    assert result.status == "supported"
    assert result.answer == "Ada Lovelace"
    assert [(c.chunk_id, list(c.sentence_ids)) for c in result.citations] == [("c_ada", [0])]
    assert result.usage is not None and result.usage.total_tokens > 0
    assert result.abstained_without_llm is False
    assert result.model == "fake-llm-test"
    assert result.packed.total_tokens <= AnswerConfig().max_context_tokens
    assert len(backend.calls) == 1
    prompt_text = backend.calls[0].messages[1].content
    assert "c_ada" in prompt_text and QUESTION in prompt_text


def test_empty_context_abstains_without_llm_call():
    backend = FakeLlmBackend([])
    result = answer_question(QUESTION, [], backend, AnswerConfig())
    assert result.status == "insufficient_evidence"
    assert result.answer == ABSTAIN_ANSWER
    assert result.citations == []
    assert result.usage is None
    assert result.abstained_without_llm is True
    assert backend.calls == []


def test_all_dropped_by_token_budget_abstains_without_call():
    packed = pack_context(QUESTION, evidence(), AnswerConfig())
    tight = AnswerConfig(max_context_tokens=packed.instruction_tokens + packed.question_tokens)
    backend = FakeLlmBackend([fixture_cases()["supported_ada"]])
    result = answer_question(QUESTION, evidence(), backend, tight)
    assert result.packed.chunks == []
    assert result.status == "insufficient_evidence"
    assert backend.calls == []


def test_invented_chunk_id_is_rejected():
    backend = FakeLlmBackend([fixture_cases()["invented_chunk"]])
    with pytest.raises(ValidationFailedError, match="unknown chunk|invented"):
        answer_question(QUESTION, evidence(), backend, AnswerConfig())


def test_out_of_range_sentence_is_rejected():
    backend = FakeLlmBackend([fixture_cases()["out_of_range_sentence"]])
    with pytest.raises(ValidationFailedError, match="outside supplied context"):
        answer_question(QUESTION, evidence(), backend, AnswerConfig())


def test_supported_requires_citation_and_insufficient_forbids_them():
    with pytest.raises(ValidationFailedError, match="at least one citation"):
        answer_question(
            QUESTION, evidence(), FakeLlmBackend([fixture_cases()["supported_no_citations"]])
        )
    with pytest.raises(ValidationFailedError, match="empty citations"):
        answer_question(
            QUESTION, evidence(), FakeLlmBackend([fixture_cases()["insufficient_with_citations"]])
        )


def test_malformed_and_empty_answers_rejected():
    cases = fixture_cases()
    with pytest.raises(ValidationFailedError, match="not valid JSON"):
        answer_question(QUESTION, evidence(), FakeLlmBackend([cases["malformed"]]))
    with pytest.raises(ValidationFailedError, match="nonempty"):
        answer_question(QUESTION, evidence(), FakeLlmBackend([cases["empty_answer"]]))
    tight = AnswerConfig(max_answer_chars=10)
    with pytest.raises(ValidationFailedError, match="exceeds limit"):
        answer_question(QUESTION, evidence(), FakeLlmBackend([cases["long_answer"]]), tight)


def test_pack_enforces_chunk_and_token_budgets():
    one_chunk = pack_context(QUESTION, evidence(), AnswerConfig(max_chunks=1))
    assert [c.chunk_id for c in one_chunk.chunks] == ["c_ada"]
    assert one_chunk.dropped_chunks == 1
    full = pack_context(QUESTION, evidence(), AnswerConfig())
    assert full.total_tokens <= AnswerConfig().max_context_tokens
    # Tight budget keeps the first-ranked chunk and drops the rest.
    second_block = full.chunks[1].token_count
    tight = AnswerConfig(max_context_tokens=full.total_tokens - second_block + 1)
    packed = pack_context(QUESTION, evidence(), tight)
    assert [c.chunk_id for c in packed.chunks] == ["c_ada"]
    assert packed.dropped_chunks == 1
    messages = build_answer_messages(QUESTION, packed)
    assert "c_ada" in messages[1].content
    assert "c_eiffel" not in messages[1].content


def test_pack_is_deterministic_and_rejects_bad_inputs():
    first = pack_context(QUESTION, evidence(), AnswerConfig())
    second = pack_context(QUESTION, list(reversed(evidence())), AnswerConfig())
    # Input order is rank order; reversed input packs in the given order.
    assert [c.chunk_id for c in first.chunks] != [c.chunk_id for c in second.chunks]
    assert first.total_tokens == pack_context(QUESTION, evidence(), AnswerConfig()).total_tokens
    duplicated = evidence() + [evidence()[0]]
    with pytest.raises(InvalidInputError, match="Duplicate evidence chunk"):
        pack_context(QUESTION, duplicated, AnswerConfig())
    with pytest.raises(InvalidInputError, match="nonempty"):
        pack_context("   ", evidence(), AnswerConfig())
    with pytest.raises(InvalidInputError, match="exceeds limit"):
        pack_context(QUESTION, evidence(), AnswerConfig(max_question_chars=5))


def test_error_codes_match_backend_schema():
    backend = FakeLlmBackend([fixture_cases()["invented_chunk"]])
    try:
        answer_question(QUESTION, evidence(), backend, AnswerConfig())
    except ValidationFailedError as exc:
        assert exc.code == "VALIDATION_FAILED"
    else:
        raise AssertionError("invented citation must raise")
    try:
        pack_context("   ", evidence(), AnswerConfig())
    except InvalidInputError as exc:
        assert exc.code == "INVALID_INPUT"
    else:
        raise AssertionError("empty question must raise")


def test_budget_tracker_blocks_second_call():
    cases = fixture_cases()
    tracker = BudgetTracker(BudgetConfig(max_calls=1, max_total_tokens=100000))
    backend = FakeLlmBackend([cases["supported_ada"], cases["supported_ada"]])
    first = answer_question(QUESTION, evidence(), backend, AnswerConfig(), budget=tracker)
    assert first.status == "supported"
    assert tracker.used_calls == 1
    with pytest.raises(BudgetExceededError, match="Call budget exceeded"):
        answer_question(QUESTION, evidence(), backend, AnswerConfig(), budget=tracker)


def test_budget_tracker_blocks_token_overrun():
    tracker = BudgetTracker(BudgetConfig(max_calls=10, max_total_tokens=10))
    with pytest.raises(BudgetExceededError, match="Token budget exceeded"):
        tracker.reserve(5, 6)


def test_fake_backend_rejects_model_mismatch_and_empty_queue():
    backend = FakeLlmBackend([], model="model-a")
    request = LlmRequest(
        model="model-b",
        messages=[{"role": "user", "content": "hello"}],
        max_output_tokens=16,
    )
    with pytest.raises(InvalidInputError, match="!= backend model"):
        backend.complete(request)
    matched = FakeLlmBackend([], model="model-a")
    with pytest.raises(Exception, match="no queued outputs"):
        matched.complete(
            LlmRequest(
                model="model-a",
                messages=[{"role": "user", "content": "hello"}],
                max_output_tokens=16,
            )
        )
