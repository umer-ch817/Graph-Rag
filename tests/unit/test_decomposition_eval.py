"""Decomposition comparison (T17, R13, optional, E).

Tuning-first comparison of baseline vs bounded decomposition with a frozen
selection before any locked use. Deltas are additive (F1, recall, latency,
cost) and never replace the frozen A-D results. Offline only: FakeLlmBackend
and synthetic chunks, no Neo4j/network/paid calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphrag_lab.answering import AnswerConfig
from graphrag_lab.config import BudgetConfig
from graphrag_lab.decomposition import DecompositionConfig, decompose_and_answer
from graphrag_lab.evaluation import (
    DECOMPOSITION_EVAL_VERSION,
    DECOMPOSITION_STRATEGY,
    EvaluationConfig,
    LabelView,
    QuestionPrediction,
    assert_decomposition_locked_use,
    assert_locked_question_use,
    check_run_compatibility,
    compare_runs,
    decomposition_report_limitations,
    decomposition_to_prediction,
    evaluate_predictions,
    freeze_decomposition_selection,
    hash_question_set,
    load_evaluation_config,
    run_config_for_arm,
    summarize_decomposition_incremental,
)
from graphrag_lab.llm import BudgetTracker, FakeLlmBackend
from graphrag_lab.reports import build_regression_rows, export_comparison

EVAL_CONFIG_PATH = Path("configs/evaluation.yaml")


def _arm_config(run_id: str, strategy: str, ids: list[str], **overrides):
    base = {
        "run_id": run_id,
        "strategy": strategy,
        "corpus_id": "hp_t17",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "answer_model": "fake-llm-test",
        "prompt_version": "answer-prompt-v1",
        "answer_version": "answer-v1",
        "token_estimator": "char-word-max-v1",
        "num_candidates": 30,
        "max_chunks": 8,
        "max_context_tokens": 3000,
        "question_ids": ids,
        "split": "tuning",
        "reranker_model": None if strategy in ("vector", "decomposition") else "fake-reranker-test",
        "index_fingerprint": "fp-index-t17",
    }
    base.update(overrides)
    return run_config_for_arm(**base)


def _labels(ids: list[str], answers: list[str], types: list[str]) -> list[LabelView]:
    return [
        LabelView(
            question_id=qid,
            answer=ans,
            supporting_facts=[
                {"title": f"Title {qid}", "sentence_id": 0, "paragraph_id": f"p_{qid}"}
            ],
            question_type=typ,
        )
        for qid, ans, typ in zip(ids, answers, types)
    ]


def _chunks_by_id(ids: list[str]) -> dict[str, dict]:
    return {
        f"c_{qid}": {
            "chunk_id": f"c_{qid}",
            "paragraph_id": f"p_{qid}",
            "title": f"Title {qid}",
            "sentence_ids": [0],
        }
        for qid in ids
    }


def _prediction(
    qid: str,
    answer: str,
    status: str = "supported",
    latency: float | None = 20.0,
    strategy: str = "vector",
):
    evidence = (
        [
            {
                "chunk_id": f"c_{qid}",
                "paragraph_id": f"p_{qid}",
                "title": f"Title {qid}",
                "sentence_ids": [0],
            }
        ]
        if status != "failed"
        else []
    )
    return QuestionPrediction(
        question_id=qid,
        answer=answer,
        status=status,  # type: ignore[arg-type]
        citations=[{"chunk_id": f"c_{qid}", "sentence_ids": [0]}] if status == "supported" else [],
        candidates=evidence,
        packed=evidence,
        latency_ms=None if status == "failed" else latency,
        input_tokens=0 if status == "failed" else 100,
        output_tokens=0 if status == "failed" else 5,
        failure_code="PROVIDER_ERROR" if status == "failed" else None,
        strategy=strategy,
    )


def _decomposition_result(
    answer: str,
    status: str,
    chunk_ids: list[str],
    packed_ids: list[str] | None = None,
    **overrides,
):
    packed = packed_ids if packed_ids is not None else list(chunk_ids)
    payload = {
        "question": "tuning question?",
        "answer": answer,
        "status": status,
        "citations": (
            [{"chunk_id": cid, "sentence_ids": [0]} for cid in packed]
            if status == "supported"
            else []
        ),
        "packed_chunk_ids": packed,
        "packed_total_tokens": 120,
        "packed_dropped_chunks": 0,
        "combined_chunk_ids": list(chunk_ids),
        "fallback": False,
        "total_input_tokens": 140,
        "total_output_tokens": 8,
        "failure_code": "PROVIDER_ERROR" if status == "failed" else None,
        "diagnostics": {"elapsed_ms": 33.0},
    }
    payload.update(overrides)
    return payload


def test_decomposition_strategy_allowed_with_matched_budgets():
    assert DECOMPOSITION_STRATEGY == "decomposition"
    ids = ["t1", "t2"]
    base = _arm_config("base", "vector", ids)
    deco = _arm_config("deco", "decomposition", ids)
    assert deco.strategy == "decomposition"
    assert deco.reranker_model is None
    assert deco.max_triples == 0 and deco.include_triples is False
    assert (base.max_chunks, base.max_context_tokens) == (
        deco.max_chunks,
        deco.max_context_tokens,
    )
    check = check_run_compatibility([base, deco])
    assert check["compatible"] is True
    # Decomposition must not smuggle a reranker or triples.
    with pytest.raises(ValueError, match="must not set reranker"):
        _arm_config("bad", "decomposition", ids, reranker_model="fake-reranker-test")
    with pytest.raises(ValueError, match="must be one of"):
        _arm_config("bad", "hybrid_magic", ids)


def test_evaluation_config_keeps_core_arms_and_records_decomposition_freeze():
    cfg = load_evaluation_config(EVAL_CONFIG_PATH)
    assert isinstance(cfg, EvaluationConfig)
    assert sorted(cfg.arms) == sorted(
        ["vector", "vector_rerank", "hybrid_rerank", "hybrid_rerank_triples"]
    )
    assert cfg.decomposition_version == "decomposition-v1"
    assert cfg.decomposition_prompt_version == "decomposition-planner-v1"
    assert cfg.decomposition_max_subquestions == 2
    assert cfg.require_decomposition_freeze is True


def test_decomposition_to_prediction_maps_evidence_and_cost_unknown():
    ids = ["t1"]
    chunks = _chunks_by_id(ids)
    result = _decomposition_result("Paris", "supported", ["c_t1"])
    pred = decomposition_to_prediction("t1", result, chunks)
    assert pred.strategy == "decomposition"
    assert pred.status == "supported"
    assert pred.answer == "Paris"
    assert [c.chunk_id for c in pred.candidates] == ["c_t1"]
    assert [c.chunk_id for c in pred.packed] == ["c_t1"]
    assert pred.input_tokens == 140 and pred.output_tokens == 8
    assert pred.estimated_cost_usd is None
    assert pred.latency_ms == pytest.approx(33.0)
    assert pred.fallback is False and pred.reranked is False
    # Explicit latency overrides diagnostics.
    override = decomposition_to_prediction("t1", result, chunks, latency_ms=12.5)
    assert override.latency_ms == pytest.approx(12.5)
    # Unknown chunks and missing provenance raise instead of scoring silently.
    with pytest.raises(ValueError, match="not in chunks_by_id"):
        decomposition_to_prediction("t1", result, _chunks_by_id(["other"]))
    bad_chunks = {"c_t1": {"chunk_id": "c_t1", "sentence_ids": [0]}}
    with pytest.raises(ValueError, match="provenance"):
        decomposition_to_prediction("t1", result, bad_chunks)
    with pytest.raises(ValueError, match="not in combined"):
        both = _chunks_by_id(["t1"])
        both["c_other"] = {
            "chunk_id": "c_other",
            "paragraph_id": "p_other",
            "title": "Title other",
            "sentence_ids": [0],
        }
        decomposition_to_prediction(
            "t1", _decomposition_result("Paris", "supported", ["c_t1"], ["c_other"]), both
        )
    with pytest.raises(ValueError, match="must use strategy"):
        decomposition_to_prediction("t1", result, chunks, strategy="vector")
    with pytest.raises(ValueError, match="unknown status"):
        decomposition_to_prediction("t1", {"status": "magic"}, chunks)


def test_t16_chain_maps_to_scorable_prediction():
    """End-to-end T16 wiring: planner + sub-answers map to a prediction."""
    question = "Who wrote The Night Cartographer and where was she born?"
    chunks = [
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
    planner = FakeLlmBackend(
        [
            {
                "subquestions": [
                    {"text": "Who wrote The Night Cartographer?", "depends_on_first": False},
                    {"text": "Where was Elena Voss born?", "depends_on_first": True},
                ]
            }
        ],
        model="fake-planner",
    )
    tracker = BudgetTracker(BudgetConfig(max_calls=10, max_total_tokens=50000))

    def retrieve_fn(text: str):
        lowered = text.lower()
        if "who wrote" in lowered:
            return [c for c in chunks if c["chunk_id"] == "c_book"]
        if "where was elena" in lowered:
            return [c for c in chunks if c["chunk_id"] == "c_author"]
        return list(chunks)

    script = [
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
    ]

    def answer_fn(text: str, evidence):
        from graphrag_lab.answering import answer_question

        payload = script.pop(0)
        backend = FakeLlmBackend([payload], model="fake-answer")
        return answer_question(text, evidence, backend, AnswerConfig(), budget=tracker)

    result = decompose_and_answer(
        question, planner, retrieve_fn, answer_fn, DecompositionConfig(), tracker
    )
    assert result.status == "supported"
    by_id = {
        "c_book": {
            "chunk_id": "c_book",
            "paragraph_id": "p_book",
            "title": "The Night Cartographer",
            "sentence_ids": [0],
        },
        "c_author": {
            "chunk_id": "c_author",
            "paragraph_id": "p_author",
            "title": "Elena Voss",
            "sentence_ids": [0],
        },
    }
    pred = decomposition_to_prediction("q_bridge", result, by_id)
    assert pred.strategy == "decomposition"
    assert sorted(c.chunk_id for c in pred.candidates) == ["c_author", "c_book"]
    assert pred.input_tokens == result.total_input_tokens
    assert pred.estimated_cost_usd is None


def _tuning_pair():
    ids = ["t1", "t2", "t3", "t4"]
    types = ["bridge", "bridge", "comparison", "comparison"]
    labels = _labels(ids, ["Paris", "Paris", "Paris", "Paris"], types)
    chunks = _chunks_by_id(ids)
    preds_base = [
        _prediction("t1", "Paris", latency=20.0),
        _prediction("t2", "London", latency=22.0),
        _prediction("t3", "Paris", latency=18.0),
        _prediction("t4", "", status="failed", latency=None),
    ]
    preds_deco = [
        decomposition_to_prediction(
            "t1", _decomposition_result("Paris", "supported", ["c_t1"]), chunks, latency_ms=40.0
        ),
        decomposition_to_prediction(
            "t2", _decomposition_result("Paris", "supported", ["c_t2"]), chunks, latency_ms=42.0
        ),
        decomposition_to_prediction(
            "t3", _decomposition_result("Paris", "supported", ["c_t3"]), chunks, latency_ms=38.0
        ),
        decomposition_to_prediction(
            "t4",
            _decomposition_result("", "failed", [], [], failure_code="PROVIDER_ERROR"),
            chunks,
            latency_ms=None,
        ),
    ]
    return ids, labels, chunks, preds_base, preds_deco


def test_tuning_comparison_is_compatible_and_deterministic():
    ids, labels, _, preds_base, preds_deco = _tuning_pair()
    cfg_a = _arm_config("t17-base", "vector", ids)
    cfg_b = _arm_config("t17-deco", "decomposition", ids)
    scored_a, eval_a = evaluate_predictions(labels, preds_base, cfg_a)
    scored_b, eval_b = evaluate_predictions(labels, preds_deco, cfg_b)
    assert eval_a.aggregate.n_total == 4 and eval_b.aggregate.n_total == 4
    # Failures stay in the fixed denominator in both arms.
    assert eval_a.aggregate.n_failed == 1 and eval_b.aggregate.n_failed == 1
    first = compare_runs(scored_a, eval_a, scored_b, eval_b, labels, replicates=200)
    second = compare_runs(scored_a, eval_a, scored_b, eval_b, labels, replicates=200)
    assert first == second
    assert first["compatible"] is True
    assert first["n"] == 4
    assert set(first["metrics"]) == {
        "exact_match",
        "f1",
        "paragraph_recall_packed",
        "all_support_packed",
        "sentence_recall_packed",
    }
    # Decomposition fixes one miss: tuning F1 improves on this fixture.
    assert first["metrics"]["f1"]["observed_delta_pp"] == pytest.approx(25.0)
    rows = build_regression_rows(scored_a, scored_b, preds_base, preds_deco, labels)
    assert len(rows) == 4
    # Matched final budgets are enforced by compatibility, not by convention.
    bad = _arm_config("t17-bad", "decomposition", ids, max_context_tokens=1000)
    _, bad_eval = evaluate_predictions(labels, preds_deco, bad)
    refused = compare_runs(scored_a, eval_a, scored_b, bad_eval, labels, replicates=200)
    assert refused["compatible"] is False
    assert "refusing unlabeled paired delta" in refused["reason"]


def test_freeze_requires_tuning_and_exact_ids():
    ids, labels, _, preds_base, preds_deco = _tuning_pair()
    cfg_a = _arm_config("t17-base", "vector", ids)
    cfg_b = _arm_config("t17-deco", "decomposition", ids)
    scored_a, eval_a = evaluate_predictions(labels, preds_base, cfg_a)
    scored_b, eval_b = evaluate_predictions(labels, preds_deco, cfg_b)
    selection = freeze_decomposition_selection(eval_a, eval_b, ids, selection_id="sel-t17-01")
    assert selection.split == "tuning"
    assert selection.select_candidate is True
    assert selection.f1_delta_pp == pytest.approx(25.0)
    assert selection.tuning_question_set_hash == hash_question_set(sorted(ids))
    assert selection.candidate_strategy == "decomposition"
    assert selection.decomposition_version == "decomposition-v1"
    assert selection.frozen_at
    # Non-tuning splits cannot freeze a selection.
    eval_locked = evaluate_predictions(
        labels, preds_base, _arm_config("base-locked", "vector", ids, split="evaluation")
    )[1]
    with pytest.raises(ValueError, match="must compare tuning runs"):
        freeze_decomposition_selection(eval_locked, eval_b, ids, selection_id="sel-x")
    # Candidate must be the decomposition arm.
    with pytest.raises(ValueError, match="must use strategy 'decomposition'"):
        freeze_decomposition_selection(eval_a, eval_a, ids, selection_id="sel-x")
    # Tuning ids must match the evaluated question set exactly.
    with pytest.raises(ValueError, match="does not match tuning_ids"):
        freeze_decomposition_selection(eval_a, eval_b, ids[:3], selection_id="sel-x")
    # Incompatible budgets cannot freeze.
    bad_eval = evaluate_predictions(
        labels, preds_deco, _arm_config("bad", "decomposition", ids, max_context_tokens=999)
    )[1]
    with pytest.raises(ValueError, match="incompatible"):
        freeze_decomposition_selection(eval_a, bad_eval, ids, selection_id="sel-x")
    _ = scored_a, scored_b


def test_locked_use_reuses_frozen_configuration_and_exact_ids():
    tuning_ids, labels, _, preds_base, preds_deco = _tuning_pair()
    cfg_a = _arm_config("t17-base", "vector", tuning_ids)
    cfg_b = _arm_config("t17-deco", "decomposition", tuning_ids)
    _, eval_a = evaluate_predictions(labels, preds_base, cfg_a)
    _, eval_b = evaluate_predictions(labels, preds_deco, cfg_b)
    selection = freeze_decomposition_selection(
        eval_a, eval_b, tuning_ids, selection_id="sel-t17-lock"
    )
    locked_ids = ["l1", "l2", "l3"]
    locked_base = _arm_config("t17-base-locked", "vector", locked_ids, split="evaluation")
    locked_cand = _arm_config("t17-deco-locked", "decomposition", locked_ids, split="evaluation")
    ok = assert_decomposition_locked_use(
        selection, locked_base, locked_cand, locked_ids, expected_locked_ids=locked_ids
    )
    assert ok["locked_question_set_hash"] == hash_question_set(sorted(locked_ids))
    assert ok["selection_id"] == "sel-t17-lock"
    # Locked split must be evaluation, not tuning.
    with pytest.raises(ValueError, match="must run on the evaluation split"):
        assert_decomposition_locked_use(selection, cfg_a, cfg_b, tuning_ids)
    # Frozen strategies cannot be substituted on locked data.
    swapped = _arm_config("swap", "hybrid_rerank", locked_ids, split="evaluation")
    with pytest.raises(ValueError, match="does not match the frozen selection"):
        assert_decomposition_locked_use(selection, locked_base, swapped, locked_ids)
    # Frozen budgets/models cannot drift on locked data (both locked arms share
    # the drift so they stay compatible with each other but differ from frozen).
    drifted_base = _arm_config(
        "drift-base", "vector", locked_ids, split="evaluation", max_context_tokens=1500
    )
    drifted = _arm_config(
        "drift", "decomposition", locked_ids, split="evaluation", max_context_tokens=1500
    )
    with pytest.raises(ValueError, match="differs from the frozen"):
        assert_decomposition_locked_use(selection, drifted_base, drifted, locked_ids)
    # Locked ids must match exactly.
    with pytest.raises(ValueError, match="exactly the frozen question ids"):
        assert_decomposition_locked_use(
            selection, locked_base, locked_cand, locked_ids[:2], locked_ids
        )
    assert assert_locked_question_use(locked_ids, list(locked_ids))


def test_incremental_summary_reports_f1_recall_latency_cost_without_replacing():
    ids, labels, _, preds_base, preds_deco = _tuning_pair()
    cfg_a = _arm_config("t17-base", "vector", ids)
    cfg_b = _arm_config("t17-deco", "decomposition", ids)
    scored_a, eval_a = evaluate_predictions(labels, preds_base, cfg_a)
    scored_b, eval_b = evaluate_predictions(labels, preds_deco, cfg_b)
    comparison = compare_runs(scored_a, eval_a, scored_b, eval_b, labels, replicates=200)
    summary = summarize_decomposition_incremental(comparison, eval_a, eval_b)
    assert summary["decomposition_eval_version"] == DECOMPOSITION_EVAL_VERSION
    assert summary["candidate_strategy"] == "decomposition"
    assert summary["n"] == 4
    assert summary["incremental"]["f1"]["delta_pp"] == pytest.approx(25.0)
    for name in ("paragraph_recall_packed", "all_support_packed", "sentence_recall_packed"):
        assert name in summary["incremental"]
        assert summary["incremental"][name]["delta_pp"] == pytest.approx(0.0)
    # Latency is observed only (no interval), cost deltas are observed sums.
    assert summary["latency"]["delta_p50_ms"] is not None
    assert summary["cost"]["delta_input_tokens"] == (
        eval_b.aggregate.total_input_tokens - eval_a.aggregate.total_input_tokens
    )
    # Unknown money stays unknown, never zero.
    assert summary["cost"]["baseline_estimated_cost_usd"] is None
    assert summary["cost"]["candidate_estimated_cost_usd"] is None
    assert summary["cost"]["delta_estimated_cost_usd"] is None
    assert summary["cost"]["cost_label_baseline"] == "unknown"
    assert any("never replaces" in w for w in summary["warnings"])
    assert any("part of the result" in w for w in summary["warnings"])
    # Incompatible comparisons never yield an incremental delta.
    bad_eval = evaluate_predictions(
        labels, preds_deco, _arm_config("bad", "decomposition", ids, corpus_id="hp_other")
    )[1]
    refused = compare_runs(scored_a, eval_a, scored_b, bad_eval, labels, replicates=200)
    with pytest.raises(ValueError, match="compatible compare_runs"):
        summarize_decomposition_incremental(refused, eval_a, bad_eval)
    with pytest.raises(ValueError, match="must use 'decomposition'"):
        summarize_decomposition_incremental(comparison, eval_a, eval_a)
    # Inputs are not mutated: old results stay frozen.
    assert eval_a.config.strategy == "vector"
    assert eval_b.config.strategy == "decomposition"


def test_decomposition_delta_exports_reconcile_and_stay_guarded(tmp_path):
    ids, labels, _, preds_base, preds_deco = _tuning_pair()
    cfg_a = _arm_config("t17-base", "vector", ids)
    cfg_b = _arm_config("t17-deco", "decomposition", ids)
    scored_a, eval_a = evaluate_predictions(labels, preds_base, cfg_a)
    scored_b, eval_b = evaluate_predictions(labels, preds_deco, cfg_b)
    comparison = compare_runs(scored_a, eval_a, scored_b, eval_b, labels, replicates=200)
    summary = summarize_decomposition_incremental(comparison, eval_a, eval_b)
    assert summary["n"] == 4
    out = tmp_path / "t17-delta"
    info = export_comparison(
        out,
        comparison,
        eval_a,
        eval_b,
        scored_a,
        scored_b,
        preds_base,
        preds_deco,
        labels,
        limitations=decomposition_report_limitations(),
        indexing_costs={
            "baseline": {"estimated_cost_usd": None, "note": "unpriced local compute"},
            "candidate": {"estimated_cost_usd": None, "note": "unpriced local compute"},
        },
    )
    assert info["n"] == 4
    payload = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert payload["run_b"]["config"]["strategy"] == "decomposition"
    assert any("additive" in line.lower() for line in payload["limitations"])
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "unknown" in report
    assert "Decomposition" in report or "decomposition" in report or "additive" in report
    assert "$0.000000" not in report
    # Guarded re-export: identical bytes are idempotent, differing bytes refuse.
    second = export_comparison(
        out,
        comparison,
        eval_a,
        eval_b,
        scored_a,
        scored_b,
        preds_base,
        preds_deco,
        labels,
        limitations=decomposition_report_limitations(),
        indexing_costs={
            "baseline": {"estimated_cost_usd": None, "note": "unpriced local compute"},
            "candidate": {"estimated_cost_usd": None, "note": "unpriced local compute"},
        },
    )
    assert second["artifact_sha256"] == info["artifact_sha256"]
    with pytest.raises(ValueError, match="differs; choose a new output"):
        export_comparison(
            out,
            comparison,
            eval_a,
            eval_b,
            scored_a,
            scored_b,
            preds_base,
            preds_deco,
            labels,
            limitations=["Different limitations block."],
            indexing_costs={
                "baseline": {"estimated_cost_usd": None, "note": "unpriced local compute"},
                "candidate": {"estimated_cost_usd": None, "note": "unpriced local compute"},
            },
        )


def test_decomposition_limitations_and_config_documentary_keys():
    limits = decomposition_report_limitations()
    assert any("never replaces" in line or "unchanged" in line for line in limits)
    assert any("tuning" in line.lower() for line in limits)
    assert any("budget" in line.lower() for line in limits)
    import yaml

    raw = yaml.safe_load(EVAL_CONFIG_PATH.read_text(encoding="utf-8"))
    assert raw["decomposition_strategy"] == "decomposition"
    assert raw["decomposition_max_subquestions"] == 2
    cfg = load_evaluation_config(EVAL_CONFIG_PATH)
    assert cfg.decomposition_version == raw["decomposition_version"]
    # Unknown fields still refuse; documentary decomposition keys are accepted.
    raw["unknown_field_xyz"] = 1
    tmp = Path("test_tmp_eval_deco_unknown.yaml")
    try:
        tmp.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown evaluation config fields"):
            load_evaluation_config(tmp)
    finally:
        if tmp.exists():
            tmp.unlink()
