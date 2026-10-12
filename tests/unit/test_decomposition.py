"""Bounded decomposition (T16, R13, optional)."""

from __future__ import annotations

import pytest

from graphrag_lab.answering import AnswerConfig, AnswerResult, answer_question, pack_context
from graphrag_lab.config import BudgetConfig
from graphrag_lab.decomposition import (
    MAX_TOTAL_LLM_CALLS,
    DecompositionConfig,
    decompose_and_answer,
    parse_planner_payload,
    plan_subquestions,
    resolve_dependent_text,
)
from graphrag_lab.llm import BudgetExceededError, BudgetTracker, FakeLlmBackend, InvalidInputError
from graphrag_lab.models import (
    DECOMPOSITION_VERSION,
    MAX_DECOMPOSITION_SUBQUESTIONS,
)

QUESTION = "Who wrote The Night Cartographer and where was she born?"
SUB_ONE = "Who wrote The Night Cartographer?"
SUB_TWO = "Where was Elena Voss born?"


def evidence_chunks():
    return [
        {
            "chunk_id": "c_book",
            "paragraph_id": "p_book",
            "title": "The Night Cartographer",
            "text": "The Night Cartographer was written by Elena Voss.",
            "sentence_ids": [0],
            "sentence_texts": ["The Night Cartographer was written by Elena Voss."],
            "rank": 1,
        },
        {
            "chunk_id": "c_author",
            "paragraph_id": "p_author",
            "title": "Elena Voss",
            "text": "Elena Voss was born in Lisbon.",
            "sentence_ids": [0],
            "sentence_texts": ["Elena Voss was born in Lisbon."],
            "rank": 2,
        },
    ]


def planner_payload(subs):
    return {"subquestions": subs}


def make_wiring(answer_texts, tracker=None):
    """Build offline retrieve/answer wiring over fixed evidence chunks."""
    chunks = evidence_chunks()
    seen_questions: list[str] = []

    def retrieve_fn(text: str):
        seen_questions.append(text)
        lowered = text.lower()
        if "night cartographer" in lowered and "where was" in lowered:
            return list(chunks)
        if "who wrote" in lowered:
            return [c for c in chunks if c["chunk_id"] == "c_book"]
        if "where was elena" in lowered or "where was she" in lowered:
            return [c for c in chunks if c["chunk_id"] == "c_author"]
        if text.strip() == QUESTION.strip():
            return list(chunks)
        return list(chunks)

    answers = list(answer_texts)

    def answer_fn(text: str, evidence):
        if not evidence:
            backend = FakeLlmBackend([], model="fake-answer")
            return answer_question(text, [], backend, AnswerConfig(), budget=tracker)
        payload = answers.pop(0) if answers else answer_texts[-1]
        backend = FakeLlmBackend([payload], model="fake-answer")
        return answer_question(text, evidence, backend, AnswerConfig(), budget=tracker)

    return retrieve_fn, answer_fn, seen_questions


def test_planner_caps_at_two_and_validates():
    cfg = DecompositionConfig()
    assert MAX_DECOMPOSITION_SUBQUESTIONS == 2
    assert cfg.max_subquestions == 2
    good = planner_payload(
        [
            {"text": SUB_ONE, "depends_on_first": False},
            {"text": SUB_TWO, "depends_on_first": True},
        ]
    )
    import json

    subs = parse_planner_payload(json.dumps(good), QUESTION, cfg)
    assert [s.text for s in subs] == [SUB_ONE, SUB_TWO]
    assert subs[1].depends_on_first is True
    # Three sub-questions are rejected, never silently truncated.
    too_many = planner_payload(
        [
            {"text": "First standalone question here?", "depends_on_first": False},
            {"text": "Second standalone question here?", "depends_on_first": False},
            {"text": "Third standalone question here?", "depends_on_first": False},
        ]
    )
    with pytest.raises(Exception, match="1..2 sub-questions"):
        parse_planner_payload(json.dumps(too_many), QUESTION, cfg)
    with pytest.raises(Exception, match="not valid JSON"):
        parse_planner_payload("not-json{{{", QUESTION, cfg)
    with pytest.raises(Exception, match="must end with"):
        parse_planner_payload(
            json.dumps(planner_payload([{"text": "No question mark", "depends_on_first": False}])),
            QUESTION,
            cfg,
        )
    first_depends = planner_payload([{"text": SUB_ONE, "depends_on_first": True}])
    with pytest.raises(Exception, match="must not depend"):
        parse_planner_payload(json.dumps(first_depends), QUESTION, cfg)
    with pytest.raises(Exception, match="nonempty"):
        plan_subquestions("   ", FakeLlmBackend([], model="m"), cfg)


def test_successful_bridge_combines_evidence_under_shared_budget():

    planner = FakeLlmBackend(
        [
            planner_payload(
                [
                    {"text": SUB_ONE, "depends_on_first": False},
                    {"text": SUB_TWO, "depends_on_first": True},
                ]
            )
        ],
        model="fake-planner",
    )
    tracker = BudgetTracker(BudgetConfig(max_calls=10, max_total_tokens=50000))
    retrieve_fn, answer_fn, seen = make_wiring(
        [
            {
                "answer": "Elena Voss",
                "status": "supported",
                "citations": [{"chunk_id": "c_book", "sentence_ids": [0]}],
            },
            {
                "answer": "Lisbon",
                "status": "supported",
                "citations": [{"chunk_id": "c_author", "sentence_ids": [0]}],
            },
            {
                "answer": "Lisbon",
                "status": "supported",
                "citations": [{"chunk_id": "c_author", "sentence_ids": [0]}],
            },
        ],
        tracker,
    )
    result = decompose_and_answer(
        QUESTION, planner, retrieve_fn, answer_fn, DecompositionConfig(), tracker
    )
    assert result.status == "supported"
    assert result.answer == "Lisbon"
    assert result.fallback is False
    assert result.stopped_early is False
    assert len(result.subquestions) == 2
    assert len(result.traces) == 2
    assert all(t.status == "supported" for t in result.traces)
    # Combined evidence deduplicates and respects the shared final budget.
    assert sorted(result.combined_chunk_ids) == ["c_author", "c_book"]
    assert len(result.packed_chunk_ids) <= AnswerConfig().max_chunks
    assert result.packed_total_tokens <= AnswerConfig().max_context_tokens
    assert result.packed_total_tokens > 0
    # Calls, context, intermediate evidence and cost are bounded and logged.
    assert result.total_llm_calls == 4  # planner + 2 subs + final
    assert result.total_llm_calls <= MAX_TOTAL_LLM_CALLS
    assert result.total_tokens == result.total_input_tokens + result.total_output_tokens
    assert result.total_tokens > 0
    assert result.estimated_cost_usd is None
    assert result.decomposition_version == DECOMPOSITION_VERSION
    assert result.diagnostics["n_subquestions"] == 2
    assert result.diagnostics["combined_unique_chunks"] == 2
    assert result.diagnostics["answer_budgets"] == {"max_chunks": 8, "max_context_tokens": 3000}
    assert len(seen) == 2  # one retrieval per sub-question; final reuses combined evidence
    assert tracker.used_calls == 4


def test_unsupported_first_answer_stops_chain():
    planner = FakeLlmBackend(
        [
            {
                "subquestions": [
                    {"text": SUB_ONE, "depends_on_first": False},
                    {"text": SUB_TWO, "depends_on_first": True},
                ]
            }
        ],
        model="fake-planner",
    )
    tracker = BudgetTracker(BudgetConfig(max_calls=10, max_total_tokens=50000))
    calls: list[str] = []

    def retrieve_fn(text: str):
        calls.append(text)
        return [c for c in evidence_chunks() if c["chunk_id"] == "c_book"]

    def answer_fn(text: str, evidence):
        if len(calls) == 1:
            backend = FakeLlmBackend(
                [
                    {
                        "answer": "Insufficient evidence.",
                        "status": "insufficient_evidence",
                        "citations": [],
                    }
                ],
                model="fake-answer",
            )
            return answer_question(text, evidence, backend, AnswerConfig(), budget=tracker)
        raise AssertionError("second sub-question must not run after unsupported first answer")

    result = decompose_and_answer(
        QUESTION, planner, retrieve_fn, answer_fn, DecompositionConfig(), tracker
    )
    assert result.status == "insufficient_evidence"
    assert result.stopped_early is True
    assert result.stop_reason == "first_sub_answer_unsupported"
    assert len(result.traces) == 1
    assert result.traces[0].status == "insufficient_evidence"
    assert len(calls) == 1
    # Unsupported intermediate is never used as fact: no dependent resolution ran.
    assert result.citations == []
    assert result.total_llm_calls == 2  # planner + first sub only
    assert tracker.used_calls == 2


def test_planner_failure_falls_back_visibly():
    planner = FakeLlmBackend(["not-json{{{ malformed"], model="fake-planner")
    tracker = BudgetTracker(BudgetConfig(max_calls=10, max_total_tokens=50000))
    retrieve_fn, answer_fn, seen = make_wiring(
        [
            {
                "answer": "Lisbon",
                "status": "supported",
                "citations": [{"chunk_id": "c_author", "sentence_ids": [0]}],
            }
        ],
        tracker,
    )
    result = decompose_and_answer(
        QUESTION, planner, retrieve_fn, answer_fn, DecompositionConfig(), tracker
    )
    assert result.fallback is True
    assert result.fallback_reason and result.fallback_reason.startswith("planner_failed")
    assert result.subquestions == []
    assert result.status == "supported"
    assert seen == [QUESTION]


def test_dependent_placeholder_resolves_and_budgets_enforced():
    from graphrag_lab.decomposition import SubQuestion

    sub = SubQuestion(text="Where was [FIRST_ANSWER] born?", depends_on_first=True)
    resolved, flag = resolve_dependent_text(sub, "Elena Voss")
    assert resolved == "Where was Elena Voss born?"
    assert flag is True
    plain = SubQuestion(text=SUB_TWO, depends_on_first=True)
    passthrough, used = resolve_dependent_text(plain, "Elena Voss")
    assert passthrough == SUB_TWO
    assert used is False
    with pytest.raises(InvalidInputError, match="nonempty"):
        resolve_dependent_text(plain, "   ")
    # Shared call cap propagates instead of silently queuing extra calls.
    tight = BudgetTracker(BudgetConfig(max_calls=1, max_total_tokens=50000))
    planner = FakeLlmBackend(
        [
            {
                "subquestions": [
                    {"text": SUB_ONE, "depends_on_first": False},
                    {"text": SUB_TWO, "depends_on_first": True},
                ]
            }
        ],
        model="fake-planner",
    )
    retrieve_fn, answer_fn, _ = make_wiring(
        [
            {
                "answer": "Elena Voss",
                "status": "supported",
                "citations": [{"chunk_id": "c_book", "sentence_ids": [0]}],
            }
        ],
        tight,
    )
    with pytest.raises(BudgetExceededError):
        decompose_and_answer(
            QUESTION, planner, retrieve_fn, answer_fn, DecompositionConfig(), tight
        )


def test_empty_combined_evidence_abstains_without_final_call():
    planner = FakeLlmBackend(
        [{"subquestions": [{"text": SUB_ONE, "depends_on_first": False}]}],
        model="fake-planner",
    )
    tracker = BudgetTracker(BudgetConfig(max_calls=10, max_total_tokens=50000))

    def retrieve_fn(text: str):
        _ = text
        return []

    answered: list[str] = []

    def answer_fn(text: str, evidence):
        answered.append(text)
        backend = FakeLlmBackend([], model="fake-answer")
        return answer_question(text, evidence, backend, AnswerConfig(), budget=tracker)

    result = decompose_and_answer(
        QUESTION, planner, retrieve_fn, answer_fn, DecompositionConfig(), tracker
    )
    # Single unsupported sub stops the chain; the final answer is the abstention.
    assert result.stopped_early is True
    assert result.status == "insufficient_evidence"
    assert answered == [SUB_ONE]
    assert tracker.used_calls == 1  # planner only; abstentions consume no LLM calls


def test_error_codes_and_versions():
    cfg = DecompositionConfig()
    assert cfg.decomposition_version == DECOMPOSITION_VERSION
    assert cfg.prompt_version
    assert cfg.max_question_chars == 2000
    packed = pack_context(QUESTION, evidence_chunks(), AnswerConfig())
    assert packed.total_tokens <= 3000
    result_cls = AnswerResult
    assert result_cls is not None
