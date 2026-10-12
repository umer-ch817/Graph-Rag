"""Evaluation dashboard and report view (T14, R09/R12).

Covers the dashboard contract without the Streamlit runtime, Neo4j or an
LLM: compatible-run headlines with uncertainty and counts, refusal of
unlabeled paired deltas for incompatible runs, multi-hop/control
separation, regression filters, separated indexing/query costs, per-question
drill-down rows and CSV/JSON/Markdown exports that reconcile with the fixed
denominator including failures. ``app/pages/evaluations.py`` owns
presentation and live RunStore wiring.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.components.metrics import (
    CONFIGURE_LABEL,
    EMPTY_TITLE,
    FILTERS,
    compatibility_banner,
    cost_panels,
    empty_state,
    failure_rows,
    filter_regression_rows,
    format_delta,
    format_interval,
    format_latency,
    format_score,
    headline_cards,
    split_rows_by_tab,
    table_rows,
)
from graphrag_lab.evaluation import (
    LabelView,
    QuestionPrediction,
    compare_runs,
    evaluate_predictions,
    run_config_for_arm,
)
from graphrag_lab.reports import (
    REGRESSION_CSV_COLUMNS,
    build_comparison_payload,
    build_regression_rows,
    classify_regression_labels,
    export_comparison,
    format_cost_usd,
    render_comparison_markdown,
)


def _arm_config(run_id: str, strategy: str, ids: list[str], **overrides):
    base = {
        "run_id": run_id,
        "strategy": strategy,
        "corpus_id": "hp_t14",
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
        "index_fingerprint": "fp-index-t14",
    }
    base.update(overrides)
    return run_config_for_arm(**base)


def _fixture():
    ids = ["q_bridge_1", "q_bridge_2", "q_comp_1", "q_single_1"]
    types = ["bridge", "bridge", "comparison", "single_hop"]
    answers = ["Ada Lovelace", "Paris", "Lisbon", "Tokyo"]
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

    def _pred(qid: str, answer: str, status: str = "supported", latency: float | None = 100.0):
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
            citations=(
                [{"chunk_id": f"c_{qid}", "sentence_ids": [0]}] if status == "supported" else []
            ),
            candidates=evidence,
            packed=evidence,
            latency_ms=None if status == "failed" else latency,
            input_tokens=0 if status == "failed" else 100,
            output_tokens=0 if status == "failed" else 5,
            failure_code="PROVIDER_ERROR" if status == "failed" else None,
            strategy="vector",
        )

    preds_a = [
        _pred("q_bridge_1", "Ada Lovelace"),
        _pred("q_bridge_2", "London"),
        _pred("q_comp_1", "", status="failed"),
        _pred("q_single_1", "", status="insufficient_evidence"),
    ]
    preds_b = [
        _pred("q_bridge_1", "Ada Lovelace"),
        _pred("q_bridge_2", "Paris"),
        _pred("q_comp_1", "", status="failed"),
        _pred("q_single_1", "Tokyo"),
    ]
    return ids, labels, preds_a, preds_b


def _evaluated():
    ids, labels, preds_a, preds_b = _fixture()
    cfg_a = _arm_config("run-a", "vector", ids)
    cfg_b = _arm_config("run-b", "hybrid_rerank", ids)
    scored_a, eval_a = evaluate_predictions(labels, preds_a, cfg_a)
    scored_b, eval_b = evaluate_predictions(labels, preds_b, cfg_b)
    comparison = compare_runs(scored_a, eval_a, scored_b, eval_b, labels, replicates=200)
    assert comparison["compatible"] is True
    rows = build_regression_rows(scored_a, scored_b, preds_a, preds_b, labels)
    return {
        "ids": ids,
        "labels": labels,
        "preds_a": preds_a,
        "preds_b": preds_b,
        "scored_a": scored_a,
        "scored_b": scored_b,
        "eval_a": eval_a,
        "eval_b": eval_b,
        "comparison": comparison,
        "rows": rows,
    }


def test_headlines_lead_with_f1_support_latency_and_counts():
    fix = _evaluated()
    cards = headline_cards(fix["comparison"], fix["eval_a"], fix["eval_b"])
    assert [card["metric"] for card in cards] == ["Answer F1", "All-support coverage", "Latency"]
    f1, sup, lat = cards
    # Both absolutes are shown with counts and split beside the headline.
    assert f1["baseline"] == pytest.approx(fix["eval_a"].aggregate.f1_mean)
    assert f1["candidate"] == pytest.approx(fix["eval_b"].aggregate.f1_mean)
    assert f1["n"] == 4 and f1["split"] == "evaluation"
    assert sup["baseline"] == pytest.approx(fix["eval_a"].aggregate.all_support_packed)
    assert sup["n"] == 4
    # Quality deltas carry uncertainty; latency is observed only.
    assert f1["delta_pp"] is not None
    assert f1["lower_pp"] <= f1["delta_pp"] <= f1["upper_pp"]
    assert lat["baseline_p50_ms"] == fix["eval_a"].aggregate.latency_p50_ms
    assert lat["delta_pp"] is None
    assert format_score(None) == "Not measured"
    assert "pp" in format_delta(f1["delta_pp"], compatible=True)
    assert "CI" in format_interval(f1["lower_pp"], f1["upper_pp"], compatible=True)
    assert format_latency(None) == "Not measured"


def test_incompatible_runs_refuse_unlabeled_deltas():
    fix = _evaluated()
    other_cfg = _arm_config("run-x", "hybrid_rerank", fix["ids"], corpus_id="hp_other")
    _, other_eval = evaluate_predictions(fix["labels"], fix["preds_b"], other_cfg)
    refused = compare_runs(
        fix["scored_a"], fix["eval_a"], fix["scored_b"], other_eval, fix["labels"], replicates=200
    )
    assert refused["compatible"] is False
    assert refused["metrics"] == {}
    assert "refusing unlabeled paired delta" in refused["reason"]
    banner = compatibility_banner(refused)
    assert "Incompatible" in banner and "No unlabeled paired delta" in banner
    assert "Compatible" in compatibility_banner(fix["comparison"])
    # Exports refuse as well: no silent unlabeled delta on disk.
    with pytest.raises(ValueError, match="Incompatible runs"):
        export_comparison(
            Path("tmp-should-not-exist"),
            refused,
            fix["eval_a"],
            other_eval,
            fix["scored_a"],
            fix["scored_b"],
            fix["preds_a"],
            fix["preds_b"],
            fix["labels"],
        )
    # Headlines for incompatible pairs show absolutes but no delta.
    cards = headline_cards(refused, fix["eval_a"], other_eval)
    assert cards[0]["compatible"] is False
    assert cards[0]["delta_pp"] is None
    assert "incompatible" in format_delta(None, compatible=False).lower()


def test_regression_labels_and_filters():
    assert classify_regression_labels(5.0, "supported", "supported") == ["Improved"]
    assert classify_regression_labels(-5.0, "supported", "supported") == ["Worsened"]
    assert classify_regression_labels(0.0, "supported", "supported") == ["Unchanged"]
    # Failed/Abstained are additive with the F1 direction.
    assert classify_regression_labels(0.0, "failed", "supported") == ["Unchanged", "Failed"]
    assert classify_regression_labels(100.0, "insufficient_evidence", "supported") == [
        "Improved",
        "Abstained",
    ]
    assert set(FILTERS) == {"Improved", "Worsened", "Unchanged", "Failed", "Abstained"}

    fix = _evaluated()
    rows = fix["rows"]
    by_id = {row["question_id"]: row for row in rows}
    # q_bridge_2: London -> Paris fixes F1: Improved.
    assert by_id["q_bridge_2"]["labels"] == ["Improved"]
    # q_bridge_1: identical answers: Unchanged.
    assert by_id["q_bridge_1"]["labels"] == ["Unchanged"]
    # q_comp_1: failed in both arms: Unchanged + Failed, zero scores preserved.
    assert by_id["q_comp_1"]["labels"] == ["Unchanged", "Failed"]
    assert by_id["q_comp_1"]["f1_a"] == pytest.approx(0.0)
    # q_single_1: abstain -> supported with correct answer: Improved + Abstained.
    assert by_id["q_single_1"]["labels"] == ["Improved", "Abstained"]

    assert [row["question_id"] for row in filter_regression_rows(rows, ["Improved"])] == [
        "q_bridge_2",
        "q_single_1",
    ]
    assert [row["question_id"] for row in filter_regression_rows(rows, ["Failed"])] == ["q_comp_1"]
    assert [row["question_id"] for row in filter_regression_rows(rows, ["Abstained"])] == [
        "q_single_1"
    ]
    assert len(filter_regression_rows(rows, [])) == 4
    assert len(filter_regression_rows(rows, None)) == 4
    # Multiple active filters union; a doubly-labeled row matches either.
    assert len(filter_regression_rows(rows, ["Failed", "Abstained"])) == 2
    with pytest.raises(ValueError, match="Unknown regression filters"):
        filter_regression_rows(rows, ["Better"])


def test_multihop_control_tabs_and_tables():
    fix = _evaluated()
    split = split_rows_by_tab(fix["rows"])
    assert [row["question_id"] for row in split["control"]] == ["q_single_1"]
    assert sorted(row["question_id"] for row in split["multi_hop"]) == [
        "q_bridge_1",
        "q_bridge_2",
        "q_comp_1",
    ]
    table = table_rows(fix["rows"])
    assert list(table[0].keys())[:5] == [
        "Question",
        "Baseline score",
        "Candidate score",
        "Difference",
        "Status",
    ]
    assert table[0]["Question"] == "q_bridge_1"
    failed = failure_rows(fix["rows"])
    assert sorted(row["question_id"] for row in failed) == ["q_comp_1", "q_single_1"]
    # Pure supported rows stay out of the failure-analysis table.
    assert "q_bridge_1" not in {row["question_id"] for row in failed}


def test_costs_separate_indexing_from_query_and_label_unknown():
    assert format_cost_usd(None) == "unknown"
    assert format_cost_usd(0.001) == "$0.001000 (estimated)"
    panels = cost_panels(
        {
            "baseline": {"input_tokens": 200, "output_tokens": 10, "estimated_cost_usd": None},
            "candidate": {"input_tokens": 200, "output_tokens": 10, "estimated_cost_usd": 0.002},
        },
        {"baseline": {"estimated_cost_usd": None, "note": "unpriced local compute"}},
    )
    assert any("unknown" in line for line in panels["query"])
    assert not any(line.strip().endswith("$0.000000 (estimated)") for line in panels["query"])
    assert any("unpriced local compute" in line for line in panels["indexing"])
    # Query and indexing panels are distinct objects; money never defaults to zero.
    assert panels["query"] != panels["indexing"]


def test_empty_state_shows_no_scores():
    state = empty_state()
    assert state["title"] == EMPTY_TITLE == "No evaluation results yet"
    assert state["action"] == CONFIGURE_LABEL == "Configure evaluation"
    assert "placeholder" in state["hint"].lower() or "No placeholder" in state["hint"]


def test_exports_reconcile_with_per_question_rows_including_failures(tmp_path: Path):
    fix = _evaluated()
    out = tmp_path / "comparison"
    info = export_comparison(
        out,
        fix["comparison"],
        fix["eval_a"],
        fix["eval_b"],
        fix["scored_a"],
        fix["scored_b"],
        fix["preds_a"],
        fix["preds_b"],
        fix["labels"],
        limitations=["Fixture comparison for T14 verification."],
        indexing_costs={
            "baseline": {"estimated_cost_usd": None, "note": "unpriced local compute"},
            "candidate": {"estimated_cost_usd": 0.004, "note": "pilot extrapolation, estimated"},
        },
    )
    assert (out / "comparison.json").is_file()
    assert (out / "regression.csv").is_file()
    assert (out / "report.md").is_file()
    assert info["n"] == 4
    assert info["compatibility_hash"] == fix["comparison"]["compatibility_hash"]

    payload = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert payload["compatibility_hash"] == fix["comparison"]["compatibility_hash"]
    assert payload["n"] == 4
    # Indexing and query costs are separated; unknown money stays unknown.
    assert set(payload["query_costs"]) == {"baseline", "candidate"}
    assert set(payload["indexing_costs"]) == {"baseline", "candidate"}
    assert payload["query_costs"]["baseline"]["estimated_cost_usd"] is None
    assert payload["query_costs"]["baseline"]["label"] == "unknown"
    assert payload["indexing_costs"]["candidate"]["estimated_cost_usd"] == pytest.approx(0.004)

    with open(out / "regression.csv", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert list(reader.fieldnames or []) == list(REGRESSION_CSV_COLUMNS)
        csv_rows = list(reader)
    assert len(csv_rows) == 4
    by_id = {row["question_id"]: row for row in csv_rows}
    # Failures stay in the denominator with zero scores, never dropped.
    assert by_id["q_comp_1"]["labels"] == "Unchanged;Failed"
    assert float(by_id["q_comp_1"]["f1_a"]) == pytest.approx(0.0)
    assert float(by_id["q_comp_1"]["f1_b"]) == pytest.approx(0.0)
    # Aggregates reconcile exactly with the per-question rows.
    mean_f1_a = sum(float(row["f1_a"]) for row in csv_rows) / len(csv_rows)
    mean_f1_b = sum(float(row["f1_b"]) for row in csv_rows) / len(csv_rows)
    mean_sup_a = sum(float(row["all_support_a"]) for row in csv_rows) / len(csv_rows)
    assert mean_f1_a == pytest.approx(payload["run_a"]["aggregate"]["f1_mean"])
    assert mean_f1_b == pytest.approx(payload["run_b"]["aggregate"]["f1_mean"])
    assert mean_sup_a == pytest.approx(payload["run_a"]["aggregate"]["all_support_packed"])
    # Deltas reconcile per row.
    for row in csv_rows:
        assert float(row["delta_f1_pp"]) == pytest.approx(float(row["f1_b"]) - float(row["f1_a"]))

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "run-a vs run-b" in report
    assert "Answer F1" in report and "All-support coverage" in report and "Latency" in report
    assert "95% CI" in report
    assert "Query cost" in report and "Indexing cost" in report
    assert "unknown" in report
    assert "Fixture comparison for T14 verification." in report

    # Re-export is idempotent; differing content is refused, never overwritten.
    second = export_comparison(
        out,
        fix["comparison"],
        fix["eval_a"],
        fix["eval_b"],
        fix["scored_a"],
        fix["scored_b"],
        fix["preds_a"],
        fix["preds_b"],
        fix["labels"],
        limitations=["Fixture comparison for T14 verification."],
        indexing_costs={
            "baseline": {"estimated_cost_usd": None, "note": "unpriced local compute"},
            "candidate": {"estimated_cost_usd": 0.004, "note": "pilot extrapolation, estimated"},
        },
    )
    assert second["artifact_sha256"] == info["artifact_sha256"]
    _ = build_comparison_payload(fix["comparison"], fix["eval_a"], fix["eval_b"], fix["rows"])
    _ = render_comparison_markdown(fix["comparison"], fix["eval_a"], fix["eval_b"], fix["rows"])


def test_paired_ids_must_match_exactly():
    fix = _evaluated()
    short_b = [item for item in fix["scored_b"] if item.question_id != "q_single_1"]
    with pytest.raises(ValueError, match="same question ids"):
        build_regression_rows(
            fix["scored_a"], short_b, fix["preds_a"], fix["preds_b"][:3], fix["labels"][:3]
        )


def test_evaluations_page_module_loads_without_side_effects():
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_page", Path("app/pages/evaluations.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
    assert callable(module._build_view)
