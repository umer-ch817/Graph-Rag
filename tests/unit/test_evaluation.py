"""Locked evaluation and single-hop controls (T13, R08-R10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphrag_lab.evaluation import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONTROL_SIZE,
    EVALUATION_SIZE,
    TUNING_SIZE,
    EvaluationConfig,
    LabelView,
    QuestionPrediction,
    SingleHopControl,
    aggregate_by_stratum,
    assert_locked_question_use,
    check_run_compatibility,
    check_tuning_selection,
    compare_runs,
    controls_freeze_hash,
    evaluate_predictions,
    flag_single_hop_regression,
    group_by_type,
    hash_question_set,
    load_evaluation_config,
    load_single_hop_controls,
    paired_bootstrap_interval,
    paired_delta_for_metric,
    run_config_for_arm,
    validate_single_hop_controls,
    validate_split_disjoint,
)

CONTROLS_PATH = Path("data/controls/single_hop.jsonl")
EVAL_CONFIG_PATH = Path("configs/evaluation.yaml")


def test_evaluation_config_loads_frozen_defaults():
    cfg = load_evaluation_config(EVAL_CONFIG_PATH)
    assert isinstance(cfg, EvaluationConfig)
    assert cfg.bootstrap_seed == BOOTSTRAP_SEED == 42
    assert cfg.bootstrap_replicates == BOOTSTRAP_REPLICATES == 2000
    assert cfg.confidence_level == pytest.approx(0.95)
    assert cfg.tuning_size == TUNING_SIZE == 30
    assert cfg.evaluation_size == EVALUATION_SIZE == 120
    assert cfg.control_size == CONTROL_SIZE == 30
    assert sorted(cfg.arms) == sorted(
        ["vector", "vector_rerank", "hybrid_rerank", "hybrid_rerank_triples"]
    )
    assert cfg.num_candidates == 30
    assert cfg.max_chunks == 8
    assert cfg.max_context_tokens == 3000
    assert cfg.require_human_review is True
    assert cfg.single_hop_regression_tolerance_pp == pytest.approx(2.0)
    _check_unknown_field()


def _check_unknown_field():
    import yaml

    payload = yaml.safe_load(EVAL_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["unknown_field_xyz"] = 1
    tmp = Path("test_tmp_eval_unknown.yaml")
    try:
        import yaml as _yaml

        tmp.write_text(_yaml.safe_dump(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown evaluation config fields"):
            load_evaluation_config(tmp)
    finally:
        if tmp.exists():
            tmp.unlink()
    return True


def _read_controls_raw():
    return [
        json.loads(line)
        for line in CONTROLS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_single_hop_controls_frozen_shape_human_verified():
    controls = load_single_hop_controls(CONTROLS_PATH)
    assert len(controls) == 30
    ids = [c.question_id for c in controls]
    assert len(set(ids)) == 30
    assert ids == sorted(ids)
    for control in controls:
        assert control.split == "control"
        assert control.question_type == "single_hop"
        assert control.question.endswith("?")
        # Single-paragraph provenance without running any system.
        assert len({f.paragraph_id for f in control.supporting_facts}) == 1
        assert control.source.paragraph_id == control.supporting_facts[0].paragraph_id
        assert control.source.title == control.supporting_facts[0].title
    # Human-reviewed file (T13, 2026-09-29): all rows verified, frozen use open.
    ok = validate_single_hop_controls(controls, require_verified=False)
    assert ok["n_controls"] == 30
    assert ok["n_verified"] == 30
    assert ok["unverified_ids"] == []
    assert ok["question_set_hash"] == hash_question_set(ids)
    assert ok["freeze_hash"] == controls_freeze_hash(controls)
    reviewed = validate_single_hop_controls(controls, require_verified=True)
    assert reviewed["n_verified"] == 30
    assert reviewed["freeze_hash"] == ok["freeze_hash"]


def test_single_hop_validation_rejects_bad_provenance():
    base = load_single_hop_controls(CONTROLS_PATH)[0].model_dump(mode="json")

    def _control(**overrides):
        payload = dict(base)
        payload.update(overrides)
        return SingleHopControl.model_validate(payload)

    good = _control()
    assert good.question_id == "sh_01"
    # Multi-paragraph support is not single-hop.
    multi = dict(base)
    multi["supporting_facts"] = [
        {"title": multi["source"]["title"], "sentence_id": 0, "paragraph_id": "p_other"},
        {
            "title": multi["source"]["title"],
            "sentence_id": 0,
            "paragraph_id": multi["source"]["paragraph_id"],
        },
    ]
    with pytest.raises(ValueError, match="exactly one paragraph"):
        SingleHopControl.model_validate(multi)
    # Out-of-range sentence.
    oob = dict(base)
    oob["supporting_facts"] = [
        {
            "title": base["source"]["title"],
            "sentence_id": 99,
            "paragraph_id": base["source"]["paragraph_id"],
        }
    ]
    with pytest.raises(ValueError, match="out of range"):
        SingleHopControl.model_validate(oob)
    # Answer must occur in the cited sentence.
    bad_answer = dict(base)
    bad_answer["answer"] = "Answer That Appears Nowhere At All Zzz"
    with pytest.raises(ValueError, match="does not occur"):
        SingleHopControl.model_validate(bad_answer)
    # Duplicate ids rejected on load.
    dup_lines = CONTROLS_PATH.read_text(encoding="utf-8").splitlines()[:2]
    dup_lines[1] = dup_lines[0]
    tmp = Path("test_tmp_controls_dup.jsonl")
    try:
        tmp.write_text("\n".join(dup_lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Duplicate control"):
            load_single_hop_controls(tmp)
    finally:
        if tmp.exists():
            tmp.unlink()
    # Wrong count rejected by the freeze gate.
    with pytest.raises(ValueError, match="exactly 30 rows"):
        validate_single_hop_controls(
            load_single_hop_controls(CONTROLS_PATH)[:29], require_verified=False
        )


def test_split_disjoint_and_locked_use():
    tuning = [f"t_{i:03d}" for i in range(30)]
    locked = [f"q_{i:03d}" for i in range(120)]
    controls = [f"sh_{i:02d}" for i in range(1, 31)]
    ok = validate_split_disjoint(tuning, locked, controls)
    assert ok["tuning_size"] == 30
    assert ok["evaluation_size"] == 120
    assert ok["control_size"] == 30
    with pytest.raises(ValueError, match="do not tune on locked"):
        validate_split_disjoint([*tuning[:29], locked[0]], locked, controls)
    with pytest.raises(ValueError, match="do not tune on controls"):
        validate_split_disjoint([*tuning[:29], controls[0]], locked, controls)
    with pytest.raises(ValueError, match="overlap single-hop"):
        validate_split_disjoint(tuning, [*locked[:119], controls[0]], controls)
    assert assert_locked_question_use(locked, list(locked)) == hash_question_set(sorted(locked))
    with pytest.raises(ValueError, match="exactly the frozen question ids"):
        assert_locked_question_use(locked[:119], locked)


def _arm_config(run_id, strategy, ids, **overrides):
    base = {
        "run_id": run_id,
        "strategy": strategy,
        "corpus_id": "hp_t13",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "answer_model": "fake-llm-test",
        "prompt_version": "answer-prompt-v1",
        "answer_version": "answer-v1",
        "token_estimator": "char-word-max-v1",
        "num_candidates": 30,
        "max_chunks": 8,
        "max_context_tokens": 3000,
        "question_ids": ids,
        "split": "evaluation",
        "reranker_model": None if strategy == "vector" else "fake-reranker-test",
        "index_fingerprint": "fp-index-t13",
    }
    base.update(overrides)
    return run_config_for_arm(**base)


def _labels_predictions(ids, answers, types):
    labels = [
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
    return labels


def _prediction(qid, answer, status="supported", failure=None, latency=100.0):
    return QuestionPrediction(
        question_id=qid,
        answer=answer,
        status=status,
        citations=(
            [{"chunk_id": f"c_{qid}", "sentence_ids": [0]}] if status == "supported" else []
        ),
        candidates=(
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
        ),
        packed=(
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
        ),
        latency_ms=None if status == "failed" else latency,
        input_tokens=0 if status == "failed" else 100,
        output_tokens=0 if status == "failed" else 5,
        failure_code=failure,
        strategy="vector",
    )


def test_compatibility_guards_paired_deltas():
    ids = ["q1", "q2", "q3"]
    a = _arm_config("run-a", "vector", ids)
    b = _arm_config("run-b", "hybrid_rerank", ids)
    check = check_run_compatibility([a, b])
    assert check["compatible"] is True
    assert check["compatibility_hash"]
    other = _arm_config("run-c", "hybrid_rerank", ids, corpus_id="hp_other")
    refused = check_run_compatibility([a, other])
    assert refused["compatible"] is False
    assert "refusing unlabeled paired delta" in refused["reason"]
    budget = _arm_config("run-d", "hybrid_rerank", ids, max_context_tokens=1000)
    refused_budget = check_run_compatibility([a, budget])
    assert refused_budget["compatible"] is False


def test_paired_bootstrap_deterministic_and_preserves_failures():
    ids = ["q_success", "q_miss", "q_failed"]
    labels = _labels_predictions(
        ids, ["Ada Lovelace", "Paris", "Paris"], ["bridge", "comparison", "bridge"]
    )
    preds_a = [
        _prediction("q_success", "Ada Lovelace"),
        _prediction("q_miss", "London"),
        _prediction("q_failed", "", status="failed", failure="PROVIDER_ERROR"),
    ]
    preds_b = [
        _prediction("q_success", "Ada Lovelace"),
        _prediction("q_miss", "Paris"),
        _prediction("q_failed", "", status="failed", failure="PROVIDER_ERROR"),
    ]
    cfg_a = _arm_config("run-a", "vector", ids)
    cfg_b = _arm_config("run-b", "hybrid_rerank", ids)
    scored_a, eval_a = evaluate_predictions(labels, preds_a, cfg_a)
    scored_b, eval_b = evaluate_predictions(labels, preds_b, cfg_b)
    # Fixed denominator keeps the failure: n=3, failure scores zero in both arms.
    assert eval_a.aggregate.n_total == 3
    assert eval_a.aggregate.n_failed == 1
    first = paired_delta_for_metric(scored_a, scored_b, "f1", replicates=200)
    second = paired_delta_for_metric(scored_a, scored_b, "f1", replicates=200)
    assert first == second
    assert first["n"] == 3
    # Arm B fixes one miss: F1 33.33 -> 66.67, delta +33.33pp.
    assert first["mean_a"] == pytest.approx(100.0 / 3)
    assert first["mean_b"] == pytest.approx(200.0 / 3)
    assert first["observed_delta_pp"] == pytest.approx(100.0 / 3)
    assert first["lower_pp"] <= first["observed_delta_pp"] <= first["upper_pp"]
    with pytest.raises(ValueError, match="same question ids"):
        paired_delta_for_metric(scored_a, scored_b[:2], "f1")
    with pytest.raises(ValueError, match="at least 2 paired"):
        paired_bootstrap_interval([1.0], replicates=200)
    with pytest.raises(ValueError, match="replicates"):
        paired_bootstrap_interval([1.0, 2.0], replicates=10)


def test_stratified_results_preserve_counts():
    ids = ["qb1", "qb2", "qc1", "qc2"]
    types = ["bridge", "bridge", "comparison", "comparison"]
    labels = _labels_predictions(ids, ["A", "B", "C", "D"], types)
    preds = [
        _prediction("qb1", "A"),
        _prediction("qb2", "wrong"),
        _prediction("qc1", "C"),
        _prediction("qc2", "", status="failed", failure="PROVIDER_ERROR"),
    ]
    cfg = _arm_config("run-s", "vector", ids)
    scored, _ = evaluate_predictions(labels, preds, cfg)
    groups = group_by_type(scored, labels)
    assert set(groups) == {"bridge", "comparison"}
    assert len(groups["bridge"]) == 2
    assert len(groups["comparison"]) == 2
    by_stratum = aggregate_by_stratum(scored, labels)
    assert by_stratum["n_total"] == 4
    assert by_stratum["counts"] == {"bridge": 2, "comparison": 2}
    assert by_stratum["overall"]["n_total"] == 4
    assert by_stratum["overall"]["n_failed"] == 1


def test_compare_runs_with_intervals_and_strata():
    ids = ["qb1", "qb2", "qc1", "qc2"]
    types = ["bridge", "bridge", "comparison", "comparison"]
    labels = _labels_predictions(ids, ["Ada", "Paris", "Lisbon", "Tokyo"], types)
    preds_a = [
        _prediction("qb1", "Ada"),
        _prediction("qb2", "wrong"),
        _prediction("qc1", "Lisbon"),
        _prediction("qc2", "", status="failed", failure="PROVIDER_ERROR"),
    ]
    preds_b = [
        _prediction("qb1", "Ada"),
        _prediction("qb2", "Paris"),
        _prediction("qc1", "Lisbon"),
        _prediction("qc2", "", status="failed", failure="PROVIDER_ERROR"),
    ]
    scored_a, eval_a = evaluate_predictions(labels, preds_a, _arm_config("run-a", "vector", ids))
    scored_b, eval_b = evaluate_predictions(
        labels, preds_b, _arm_config("run-b", "hybrid_rerank", ids)
    )
    compared = compare_runs(scored_a, eval_a, scored_b, eval_b, labels, replicates=200)
    assert compared["compatible"] is True
    assert compared["n"] == 4
    assert set(compared["metrics"]) == {
        "exact_match",
        "f1",
        "paragraph_recall_packed",
        "all_support_packed",
        "sentence_recall_packed",
    }
    f1 = compared["metrics"]["f1"]
    assert f1["lower_pp"] <= f1["observed_delta_pp"] <= f1["upper_pp"]
    assert set(compared["strata"]) == {"bridge", "comparison"}
    assert compared["strata"]["bridge"]["n"] == 2
    assert compared["failures"]["arm_a"]["n_failed"] == 1
    assert compared["failures"]["arm_b"]["n_failed"] == 1
    assert any("fixed" in w.lower() or "failures score zero" in w for w in compared["warnings"])
    # Incompatible runs produce no unlabeled deltas.
    bad_eval = evaluate_predictions(
        labels, preds_b, _arm_config("run-x", "hybrid_rerank", ids, corpus_id="hp_other")
    )[1]
    refused = compare_runs(scored_a, eval_a, scored_b, bad_eval, labels, replicates=200)
    assert refused["compatible"] is False
    assert refused["metrics"] == {}
    assert "refusing unlabeled paired delta" in refused["reason"]
    # Mismatched paired ids also refuse.
    short_b = [p for p in scored_b if p.question_id != "qc2"]
    assert len(short_b) == 3
    mismatched = compare_runs(scored_a, eval_a, short_b, eval_b, labels, replicates=200)
    assert mismatched["compatible"] is False


def test_tuning_selection_and_single_hop_flag():
    ids = ["q1", "q2"]
    labels = _labels_predictions(ids, ["A", "B"], ["bridge", "comparison"])
    base_preds = [_prediction("q1", "A"), _prediction("q2", "wrong")]
    better_preds = [_prediction("q1", "A"), _prediction("q2", "B")]
    worse_support = [_prediction("q1", "wrong"), _prediction("q2", "wrong")]
    _, base = evaluate_predictions(labels, base_preds, _arm_config("base", "vector", ids))
    _, better = evaluate_predictions(
        labels, better_preds, _arm_config("cand", "hybrid_rerank", ids)
    )
    decision = check_tuning_selection(base, better)
    assert decision["select_candidate"] is True
    assert decision["f1_delta_pp"] == pytest.approx(50.0)
    _, worse = evaluate_predictions(
        labels, worse_support, _arm_config("worse", "hybrid_rerank", ids)
    )
    reject = check_tuning_selection(base, worse)
    assert reject["select_candidate"] is False
    assert flag_single_hop_regression(-3.0)["regression"] is True
    assert flag_single_hop_regression(-1.9)["regression"] is False
    assert flag_single_hop_regression(0.0)["regression"] is False
