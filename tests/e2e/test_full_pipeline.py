"""Full offline pipeline handoff (T15, R01-R12, checkpoint C5).

Runs the fixture-scale user flow without Neo4j, model downloads, network or
paid LLM calls: prepare -> chunk -> provenance -> grounded answers (Fake LLM)
-> fixed-denominator scoring -> paired comparison -> guarded exports ->
durable jobs -> dashboard view-models -> control freeze gates -> config.

Every metric is computed from actual fixture evidence; failures stay in the
denominator; unknown money stays unknown; no secrets enter artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.components.metrics import (
    compatibility_banner,
    cost_panels,
    empty_state,
    filter_regression_rows,
    headline_cards,
)
from graphrag_lab.answering import AnswerConfig, answer_question, pack_context
from graphrag_lab.chunking import ChunkConfig, chunk_paragraphs
from graphrag_lab.compare_flow import (
    assert_matched_budgets,
    build_sources,
    fallback_text,
    resolve_citation,
    tuning_examples,
    validate_arm_pair,
)
from graphrag_lab.config import load_config
from graphrag_lab.corpus import prepare_dataset
from graphrag_lab.evaluation import (
    LabelView,
    QuestionPrediction,
    assert_locked_question_use,
    compare_runs,
    controls_freeze_hash,
    evaluate_predictions,
    hash_question_set,
    load_evaluation_config,
    load_single_hop_controls,
    run_config_for_arm,
    validate_single_hop_controls,
    validate_split_disjoint,
)
from graphrag_lab.jobs import JobStore, enqueue_index
from graphrag_lab.llm import FakeLlmBackend
from graphrag_lab.models import HotpotQuestion, SampleConfig
from graphrag_lab.provenance import (
    verify_chunks,
    verify_gold_covered_by_chunks,
    verify_supporting_facts,
)
from graphrag_lab.reports import (
    build_regression_rows,
    export_baseline,
    export_comparison,
    format_cost_usd,
)
from graphrag_lab.reranking import FakeRerankerBackend, deduplicate_chunks, rerank_candidates
from graphrag_lab.setup_flow import (
    enqueue_index_once,
    spend_summary,
    summarize_setup,
)
from graphrag_lab.worker import Worker, demo_handler_factory

ANSWERS = [
    "Ada Lovelace",
    "Paris",
    "Lisbon",
    "Tokyo",
    "Berlin",
    "Madrid",
    "Rome",
    "Oslo",
]


def _synthetic_source(count: int = 8) -> list[dict]:
    rows: list[dict] = []
    for i in range(count):
        rows.append(
            {
                "_id": f"q_t15_{i:02d}",
                "question": f"What connects Person {i} to Town version {i % 2}?",
                "answer": ANSWERS[i % len(ANSWERS)],
                "type": "comparison" if i % 4 == 0 else "bridge",
                "level": "hard",
                "context": [
                    [f"Person {i}", [f"Person {i} directed a film.", f"Person {i} met Town folk."]],
                    ["Town", [f"Town fact version {i % 2}."]],
                    ["Distractor", ["Unrelated but retained."]],
                ],
                "supporting_facts": [[f"Person {i}", 0], ["Town", 0]],
            }
        )
    return rows


def _write_source(path: Path, count: int = 8) -> Path:
    path.write_text(json.dumps(_synthetic_source(count)), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _fixture_dataset(tmp_path: Path):
    source = _write_source(tmp_path / "source.json", 8)
    config = SampleConfig(sample_size=6, tuning_size=2, seed=42)
    first = prepare_dataset(source, tmp_path / "ds_one", config)
    second = prepare_dataset(source, tmp_path / "ds_two", config)
    assert first == second
    for name in ("questions.jsonl", "labels.jsonl", "contexts.jsonl", "paragraphs.jsonl"):
        assert (tmp_path / "ds_one" / name).read_bytes() == (
            tmp_path / "ds_two" / name
        ).read_bytes()
    return source, config, first, tmp_path / "ds_one"


def _chunks_for_dataset(dataset_dir: Path, corpus_id: str):
    paragraphs = _read_jsonl(dataset_dir / "paragraphs.jsonl")
    chunks = chunk_paragraphs(paragraphs, corpus_id, ChunkConfig())
    verify_chunks(chunks, paragraphs)
    labels = _read_jsonl(dataset_dir / "labels.jsonl")
    verify_supporting_facts(labels, paragraphs)
    verify_gold_covered_by_chunks(labels, chunks)
    return paragraphs, labels, chunks


def _evidence_views(chunks, paragraph_ids: set[str], title_hint: str | None = None):
    views = []
    for chunk in chunks:
        if chunk.paragraph_id in paragraph_ids:
            views.append(
                {
                    "chunk_id": chunk.id,
                    "paragraph_id": chunk.paragraph_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "sentence_ids": list(chunk.sentence_ids),
                    "sentence_texts": list(chunk.sentence_texts),
                    "rank": 1,
                }
            )
    views.sort(key=lambda v: v["chunk_id"])
    _ = title_hint
    return views


def test_fixture_prepare_is_reproducible_and_labels_separated(tmp_path: Path):
    _, _, manifest, dataset_dir = _fixture_dataset(tmp_path)
    questions = _read_jsonl(dataset_dir / "questions.jsonl")
    labels = _read_jsonl(dataset_dir / "labels.jsonl")
    paragraphs = _read_jsonl(dataset_dir / "paragraphs.jsonl")
    contexts = _read_jsonl(dataset_dir / "contexts.jsonl")
    assert len(questions) == 6
    assert len(labels) == 6
    assert all("answer" not in q and "supporting_facts" not in q for q in questions)
    # Town variants survive deduplication; shared distractor collapses.
    assert len([p for p in paragraphs if p["title"] == "Town"]) == 2
    assert len([p for p in paragraphs if p["title"] == "Distractor"]) == 1
    assert len(contexts) == 6 * 3
    by_id = {p["id"]: p for p in paragraphs}
    for label in labels:
        for fact in label["supporting_facts"]:
            paragraph = by_id[fact["paragraph_id"]]
            assert paragraph["sentences"][fact["sentence_id"]]
    assert (
        manifest.source_sha256
        == hashlib.sha256((tmp_path / "source.json").read_bytes()).hexdigest()
    )
    # Tuning/evaluation split is disjoint with preserved ids.
    assert set(manifest.split_ids["tuning"]) | set(manifest.split_ids["evaluation"]) == set(
        manifest.selected_question_ids
    )
    assert not set(manifest.split_ids["tuning"]) & set(manifest.split_ids["evaluation"])


def test_chunks_preserve_provenance_and_cover_gold(tmp_path: Path):
    _, _, manifest, dataset_dir = _fixture_dataset(tmp_path)
    paragraphs, labels, chunks = _chunks_for_dataset(dataset_dir, manifest.corpus_id)
    assert chunks
    # Deterministic ids: second chunking pass is identical.
    again = chunk_paragraphs(paragraphs, manifest.corpus_id, ChunkConfig())
    assert [c.id for c in again] == [c.id for c in chunks]
    for chunk in chunks:
        assert chunk.token_count <= ChunkConfig().max_tokens
        assert chunk.id.startswith("c_")
    assert len({c.id for c in chunks}) == len(chunks)
    _ = labels


def test_grounded_answers_use_fixed_denominator_with_failures(tmp_path: Path):
    _, _, manifest, dataset_dir = _fixture_dataset(tmp_path)
    _, labels_raw, chunks = _chunks_for_dataset(dataset_dir, manifest.corpus_id)
    eval_ids = sorted(manifest.split_ids["evaluation"])
    assert len(eval_ids) == 4
    by_label = {row["question_id"]: row for row in labels_raw}
    gold_paras = {
        qid: {fact["paragraph_id"] for fact in by_label[qid]["supporting_facts"]}
        for qid in eval_ids
    }
    all_para_ids = {c.paragraph_id for c in chunks}
    cfg = AnswerConfig()

    # q0: success — gold evidence, correct Fake LLM answer with resolving citation.
    q0 = eval_ids[0]
    views0 = _evidence_views(chunks, gold_paras[q0])
    assert views0
    backend0 = FakeLlmBackend(
        [
            {
                "answer": by_label[q0]["answer"],
                "status": "supported",
                "citations": [{"chunk_id": views0[0]["chunk_id"], "sentence_ids": [0]}],
            }
        ]
    )
    result0 = answer_question(f"Question about {q0}?", views0, backend0, cfg)
    assert result0.status == "supported"
    assert backend0.remaining == 0

    # q1: miss — non-gold evidence, wrong answer (still supported, scores zero).
    q1 = eval_ids[1]
    miss_paras = all_para_ids - gold_paras[q1]
    views1 = _evidence_views(chunks, miss_paras)[:2]
    assert views1
    backend1 = FakeLlmBackend(
        [
            {
                "answer": "Wrong Answer XYZ",
                "status": "supported",
                "citations": [{"chunk_id": views1[0]["chunk_id"], "sentence_ids": [0]}],
            }
        ]
    )
    result1 = answer_question(f"Question about {q1}?", views1, backend1, cfg)
    assert result1.status == "supported"

    # q3: abstention — empty evidence abstains without any LLM call.
    q3 = eval_ids[3]
    backend_empty = FakeLlmBackend(
        [{"answer": "unused", "status": "supported", "citations": []}],
    )
    result_empty = answer_question(f"Question about {q3}?", [], backend_empty, cfg)
    assert result_empty.status == "insufficient_evidence"
    assert result_empty.abstained_without_llm is True
    assert backend_empty.remaining == 1  # no call consumed

    labels = [
        LabelView(
            question_id=qid,
            answer=by_label[qid]["answer"],
            supporting_facts=by_label[qid]["supporting_facts"],
            question_type="bridge",
        )
        for qid in eval_ids
    ]

    def _pred(qid: str, result, views, status: str, failure: str | None = None):
        if status == "failed":
            return QuestionPrediction(
                question_id=qid,
                answer="",
                status="failed",
                citations=[],
                candidates=[],
                packed=[],
                latency_ms=None,
                failure_code=failure or "PROVIDER_ERROR",
                strategy="vector",
            )
        return QuestionPrediction(
            question_id=qid,
            answer=result.answer,
            status=result.status,
            citations=[c.model_dump() for c in result.citations],
            candidates=[
                {
                    "chunk_id": v["chunk_id"],
                    "paragraph_id": v["paragraph_id"],
                    "title": v["title"],
                    "sentence_ids": list(v["sentence_ids"]),
                }
                for v in views
            ],
            packed=[
                {
                    "chunk_id": c.chunk_id,
                    "paragraph_id": c.paragraph_id,
                    "title": c.title,
                    "sentence_ids": list(c.sentence_ids),
                }
                for c in result.packed.chunks
            ],
            latency_ms=10.0,
            input_tokens=result.packed.total_tokens,
            output_tokens=5,
            strategy="vector",
        )

    q2 = eval_ids[2]
    predictions = [
        _pred(q0, result0, views0, "supported"),
        _pred(q1, result1, views1, "supported"),
        _pred(q2, None, [], "failed"),
        QuestionPrediction(
            question_id=q3,
            answer=result_empty.answer,
            status="insufficient_evidence",
            citations=[],
            candidates=[],
            packed=[],
            latency_ms=5.0,
            input_tokens=0,
            output_tokens=0,
            strategy="vector",
        ),
    ]
    run_cfg = run_config_for_arm(
        run_id="t15-vector",
        strategy="vector",
        corpus_id="hp_t15",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        answer_model="fake-llm-test",
        prompt_version="answer-prompt-v1",
        answer_version="answer-v1",
        token_estimator="char-word-max-v1",
        num_candidates=30,
        max_chunks=8,
        max_context_tokens=3000,
        question_ids=eval_ids,
        split="evaluation",
    )
    from graphrag_lab.evaluation import evaluate_predictions as _eval

    scored, evaluation = _eval(labels, predictions, run_cfg)
    assert evaluation.aggregate.n_total == 4
    assert evaluation.aggregate.n_failed == 1
    assert evaluation.aggregate.n_abstained == 1
    assert evaluation.aggregate.n_supported == 2
    # Fixed denominator: failed + abstained + miss keep zero/low scores inside.
    assert evaluation.aggregate.f1_mean < 100.0
    assert evaluation.success_only.n_success == 3


def test_paired_arms_share_budgets_and_refuse_incompatible(tmp_path: Path):
    _, _, manifest, dataset_dir = _fixture_dataset(tmp_path)
    _, labels_raw, chunks = _chunks_for_dataset(dataset_dir, manifest.corpus_id)
    eval_ids = sorted(manifest.split_ids["evaluation"])
    labels = [
        LabelView(
            question_id=qid,
            answer="Ada Lovelace" if i == 0 else "Paris",
            supporting_facts=[
                {"title": "Person 0", "sentence_id": 0, "paragraph_id": "p_gold_a"}
                if i == 0
                else {"title": "Town", "sentence_id": 0, "paragraph_id": "p_gold_b"}
            ],
            question_type="bridge" if i < 2 else "comparison",
        )
        for i, qid in enumerate(eval_ids)
    ]
    gold_a = [
        {"chunk_id": "c_a", "paragraph_id": "p_gold_a", "title": "Person 0", "sentence_ids": [0]}
    ]
    gold_b = [{"chunk_id": "c_b", "paragraph_id": "p_gold_b", "title": "Town", "sentence_ids": [0]}]

    def _make(qid: str, answer: str, status: str, packed, failure=None):
        return QuestionPrediction(
            question_id=qid,
            answer=answer,
            status=status,  # type: ignore[arg-type]
            citations=[],
            candidates=packed,
            packed=packed if status != "failed" else [],
            latency_ms=None if status == "failed" else 20.0,
            input_tokens=0 if status == "failed" else 50,
            output_tokens=0 if status == "failed" else 5,
            failure_code=failure,
            strategy="vector",
        )

    preds_a: list[QuestionPrediction] = []
    preds_b: list[QuestionPrediction] = []
    for i, qid in enumerate(eval_ids):
        packed = gold_a if i == 0 else gold_b
        if i == 1:
            # Baseline misses q1; candidate fixes it: one Improved row.
            preds_a.append(_make(qid, "London", "supported", packed))
            preds_b.append(_make(qid, "Paris", "supported", packed))
        elif i == 2:
            preds_a.append(_make(qid, "", "failed", [], "PROVIDER_ERROR"))
            preds_b.append(_make(qid, "", "failed", [], "PROVIDER_ERROR"))
        else:
            ans = "Ada Lovelace" if i == 0 else "Paris"
            preds_a.append(_make(qid, ans, "supported", packed))
            preds_b.append(_make(qid, ans, "supported", packed))

    cfg_a = run_config_for_arm(
        run_id="t15-a",
        strategy="vector",
        corpus_id="hp_t15",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        answer_model="fake-llm-test",
        prompt_version="answer-prompt-v1",
        answer_version="answer-v1",
        token_estimator="char-word-max-v1",
        num_candidates=30,
        max_chunks=8,
        max_context_tokens=3000,
        question_ids=eval_ids,
        split="evaluation",
    )
    cfg_b = run_config_for_arm(
        run_id="t15-b",
        strategy="hybrid_rerank",
        corpus_id="hp_t15",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        answer_model="fake-llm-test",
        prompt_version="answer-prompt-v1",
        answer_version="answer-v1",
        token_estimator="char-word-max-v1",
        num_candidates=30,
        max_chunks=8,
        max_context_tokens=3000,
        question_ids=eval_ids,
        split="evaluation",
        reranker_model="fake-reranker-test",
    )
    scored_a, eval_a = evaluate_predictions(labels, preds_a, cfg_a)
    scored_b, eval_b = evaluate_predictions(labels, preds_b, cfg_b)
    comparison = compare_runs(scored_a, eval_a, scored_b, eval_b, labels, replicates=200)
    assert comparison["compatible"] is True
    assert comparison["metrics"]["f1"]["n"] == len(eval_ids)
    assert "failures" in comparison and "strata" in comparison
    rows = build_regression_rows(scored_a, scored_b, preds_a, preds_b, labels)
    assert len(rows) == len(eval_ids)
    improved = filter_regression_rows(rows, ["Improved"])
    assert len(improved) >= 1
    # Matched-budget guard for ablations.
    assert assert_matched_budgets(
        [
            {"max_chunks": 8, "max_context_tokens": 3000},
            {"max_chunks": 8, "max_context_tokens": 3000},
        ]
    ) == {"max_chunks": 8, "max_context_tokens": 3000}
    with pytest.raises(ValueError, match="share final budgets"):
        assert_matched_budgets(
            [
                {"max_chunks": 8, "max_context_tokens": 3000},
                {"max_chunks": 4, "max_context_tokens": 3000},
            ]
        )
    # Incompatible runs refuse unlabeled deltas and guarded exports refuse.
    cfg_other = cfg_b.model_copy(update={"corpus_id": "hp_other"})
    _, eval_other = evaluate_predictions(labels, preds_b, cfg_other)
    refused = compare_runs(scored_a, eval_a, scored_b, eval_other, labels, replicates=200)
    assert refused["compatible"] is False
    with pytest.raises(ValueError, match="Incompatible runs"):
        export_comparison(
            tmp_path / "refused",
            refused,
            eval_a,
            eval_other,
            scored_a,
            scored_b,
            preds_a,
            preds_b,
            labels,
        )
    _ = chunks


def test_exports_reconcile_and_keep_unknown_costs(tmp_path: Path):
    _, _, manifest, dataset_dir = _fixture_dataset(tmp_path)
    _, labels_raw, chunks = _chunks_for_dataset(dataset_dir, manifest.corpus_id)
    eval_ids = sorted(manifest.split_ids["evaluation"])[:3]
    labels = [
        LabelView(
            question_id=qid,
            answer=by["answer"],
            supporting_facts=by["supporting_facts"],
            question_type="bridge",
        )
        for qid, by in [
            (qid, next(r for r in labels_raw if r["question_id"] == qid)) for qid in eval_ids
        ]
    ]
    by_chunk = {c.paragraph_id: c for c in chunks}
    predictions: list[QuestionPrediction] = []
    for qid in eval_ids:
        gold = next(r for r in labels_raw if r["question_id"] == qid)["supporting_facts"]
        first_para = gold[0]["paragraph_id"]
        chunk = by_chunk[first_para]
        predictions.append(
            QuestionPrediction(
                question_id=qid,
                answer=next(lb.answer for lb in labels if lb.question_id == qid),
                status="supported",
                citations=[{"chunk_id": chunk.id, "sentence_ids": [0]}],
                candidates=[
                    {
                        "chunk_id": chunk.id,
                        "paragraph_id": chunk.paragraph_id,
                        "title": chunk.title,
                        "sentence_ids": [0],
                    }
                ],
                packed=[
                    {
                        "chunk_id": chunk.id,
                        "paragraph_id": chunk.paragraph_id,
                        "title": chunk.title,
                        "sentence_ids": [0],
                    }
                ],
                latency_ms=15.0,
                input_tokens=80,
                output_tokens=4,
                strategy="vector",
            )
        )
    run_cfg = run_config_for_arm(
        run_id="t15-export",
        strategy="vector",
        corpus_id="hp_t15",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        answer_model="fake-llm-test",
        prompt_version="answer-prompt-v1",
        answer_version="answer-v1",
        token_estimator="char-word-max-v1",
        num_candidates=30,
        max_chunks=8,
        max_context_tokens=3000,
        question_ids=eval_ids,
        split="tuning",
    )
    scored, evaluation = evaluate_predictions(labels, predictions, run_cfg)
    out = tmp_path / "baseline"
    info = export_baseline(
        out, evaluation, scored, predictions, labels, limitations=["T15 fixture export."]
    )
    for name in ("results.jsonl", "aggregate.json", "manifest.json", "summary.md"):
        assert (out / name).is_file()
    assert set(info["artifact_sha256"]) == {
        "results.jsonl",
        "aggregate.json",
        "manifest.json",
        "summary.md",
    }
    payload = json.loads((out / "aggregate.json").read_text(encoding="utf-8"))
    assert payload["aggregate"]["n_total"] == len(eval_ids)
    # Unknown money stays unknown, never zero.
    assert format_cost_usd(None) == "unknown"
    assert format_cost_usd(evaluation.aggregate.total_estimated_cost_usd) == "unknown"
    assert evaluation.aggregate.total_estimated_cost_usd is None
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "unknown" in text
    assert "$0.000000" not in text
    # No secrets in exports.
    blob = "".join(
        (out / name).read_text(encoding="utf-8") for name in ("aggregate.json", "manifest.json")
    )
    assert "GRAPHRAG" not in blob and "password" not in blob.lower()
    # Comparison export reconciles with per-question rows.
    cfg_b = run_config_for_arm(
        run_id="t15-export-b",
        strategy="hybrid_rerank",
        corpus_id="hp_t15",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        answer_model="fake-llm-test",
        prompt_version="answer-prompt-v1",
        answer_version="answer-v1",
        token_estimator="char-word-max-v1",
        num_candidates=30,
        max_chunks=8,
        max_context_tokens=3000,
        question_ids=eval_ids,
        split="tuning",
        reranker_model="fake-reranker-test",
    )
    scored_b, eval_b = evaluate_predictions(labels, predictions, cfg_b)
    comparison = compare_runs(scored, evaluation, scored_b, eval_b, labels, replicates=200)
    cmp_out = tmp_path / "comparison"
    cmp_info = export_comparison(
        cmp_out,
        comparison,
        evaluation,
        eval_b,
        scored,
        scored_b,
        predictions,
        predictions,
        labels,
        limitations=["T15 fixture comparison."],
        indexing_costs={
            "baseline": {"estimated_cost_usd": None, "note": "unpriced local compute"},
            "candidate": {"estimated_cost_usd": None, "note": "unpriced local compute"},
        },
    )
    assert cmp_info["n"] == len(eval_ids)
    assert (cmp_out / "comparison.json").is_file()
    assert (cmp_out / "regression.csv").is_file()
    assert (cmp_out / "report.md").is_file()
    report = (cmp_out / "report.md").read_text(encoding="utf-8")
    assert "unknown" in report
    assert "Limitations" in report


def test_durable_jobs_and_setup_states_persist(tmp_path: Path):
    db = tmp_path / "jobs.sqlite3"
    store = JobStore(db)
    assert summarize_setup(store).dataset.state == "missing"
    store.create_dataset(
        dataset_id="ds_t15",
        source="fixture",
        checksum="sha_t15",
        seed=42,
        sample_count=6,
        manifest_path="artifacts/dataset/manifest.json",
    )
    store.create_snapshot(snapshot_id="snap_t15", dataset_id="ds_t15", fingerprint="fp_t15")
    from graphrag_lab.config import BudgetConfig

    budget = BudgetConfig(max_calls=20, max_total_tokens=20000, max_cost_usd=None)
    ref, submitted = enqueue_index_once(
        store,
        snapshot_id="snap_t15",
        stage="vector",
        budget=budget,
        idempotency_key="t15-key",
        items=[{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}],
    )
    assert submitted is True
    dup, resubmitted = enqueue_index_once(
        store,
        snapshot_id="snap_t15",
        stage="vector",
        budget=budget,
        idempotency_key="t15-key",
        items=[{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}],
    )
    assert resubmitted is False
    assert dup.job_id == ref.job_id
    assert summarize_setup(store).vector.state == "queued"

    calls: list[str] = []
    worker = Worker(db, worker_id="w-t15")
    worker.register_handler("index_vector", demo_handler_factory(calls))
    worker.run_job(ref.job_id, max_items=2)
    del worker
    reopened = JobStore(db)
    assert reopened.get_job(ref.job_id).progress_done == 2
    # Same worker id resumes after a restart so the heartbeat lease still owns
    # the open attempt; a different worker id would need stale recovery first.
    worker2 = Worker(db, worker_id="w-t15")
    worker2.register_handler("index_vector", demo_handler_factory(calls))
    worker2.run_job(ref.job_id)
    assert reopened.get_job(ref.job_id).state == "completed"
    assert summarize_setup(JobStore(db)).vector.state == "ready"
    assert summarize_setup(JobStore(db)).ready.headline == "Vector ready; graph unavailable."
    spend = spend_summary(reopened.get_job(ref.job_id))
    assert spend.estimated_cost_usd is None
    assert "unknown" in spend.cost_label
    assert spend.cost_label != "$0.000000 (estimated)"
    with pytest.raises(Exception, match="stage must be"):
        enqueue_index(store, snapshot_id="snap_t15", stage="bogus", budget=budget)


def test_compare_sources_resolve_exactly_and_dashboard_reads_saved(tmp_path: Path):
    chunks = [
        {
            "chunk_id": "c_book",
            "paragraph_id": "p_book",
            "title": "The Night Cartographer",
            "text": "The Night Cartographer was written by Elena Voss.",
            "sentence_ids": [0],
            "sentence_texts": ["The Night Cartographer was written by Elena Voss."],
            "origin": "both",
            "rank": 1,
        },
        {
            "chunk_id": "c_author",
            "paragraph_id": "p_author",
            "title": "Elena Voss",
            "text": "Elena Voss was born in Lisbon.",
            "sentence_ids": [0],
            "sentence_texts": ["Elena Voss was born in Lisbon."],
            "origin": "graph",
            "rank": 2,
        },
    ]
    sources = build_sources(chunks, ["c_book"], [{"chunk_id": "c_book", "sentence_ids": [0]}])
    assert sources[0].chunk_id == "c_book"
    assert sources[0].cited is True
    resolved = resolve_citation({"chunk_id": "c_book", "sentence_ids": [0]}, sources)
    assert resolved["source"].chunk_id == "c_book"
    with pytest.raises(LookupError, match="unknown chunk"):
        resolve_citation({"chunk_id": "c_missing", "sentence_ids": [0]}, sources)
    with pytest.raises(LookupError, match="outside the retrieved context"):
        resolve_citation({"chunk_id": "c_book", "sentence_ids": [9]}, sources)
    assert validate_arm_pair("vector", "hybrid_rerank_triples") == (
        "vector",
        "hybrid_rerank_triples",
    )
    with pytest.raises(ValueError, match="must differ"):
        validate_arm_pair("vector", "vector")
    banner = fallback_text(fallback=True, reason="no_linked_entities", cap_hits=["max_merged_hit"])
    assert "No linked entities; vector evidence shown" in banner
    assert fallback_text(fallback=False, reason=None, cap_hits=[]) == ""

    # Tuning picker never exposes locked rows.
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "questions.jsonl").write_text(
        '{"id": "q_t1", "question": "Tuning one?", "type": "bridge", "split": "tuning"}\n'
        '{"id": "q_e1", "question": "Locked one?", "type": "bridge", "split": "evaluation"}\n',
        encoding="utf-8",
    )
    assert [row["id"] for row in tuning_examples(dataset)] == ["q_t1"]

    # Reranker dedupes and never sums raw scales.
    duped = chunks + [chunks[0]]
    deduped, removed = deduplicate_chunks(duped)
    assert removed == ["c_book"]
    reranked = rerank_candidates(
        "Who wrote The Night Cartographer?", chunks, FakeRerankerBackend(fixed_scores=[0.2, 0.9])
    )
    assert reranked.ordered_chunk_ids[0] == "c_author"
    assert reranked.degraded is False

    # Evaluation headlines read saved aggregates; empty state shows no scores.
    ids = ["q_a", "q_b", "q_c"]
    labels = [
        LabelView(
            question_id=qid,
            answer="Paris",
            supporting_facts=[{"title": "Town", "sentence_id": 0, "paragraph_id": "p_town"}],
            question_type="bridge",
        )
        for qid in ids
    ]
    preds = [
        QuestionPrediction(
            question_id=qid,
            answer="Paris",
            status="supported",
            citations=[],
            candidates=[
                {"chunk_id": "c_x", "paragraph_id": "p_town", "title": "Town", "sentence_ids": [0]}
            ],
            packed=[
                {"chunk_id": "c_x", "paragraph_id": "p_town", "title": "Town", "sentence_ids": [0]}
            ],
            latency_ms=10.0,
            input_tokens=10,
            output_tokens=2,
            strategy="vector",
        )
        for qid in ids
    ]
    cfg_a = run_config_for_arm(
        run_id="t15-eval-a",
        strategy="vector",
        corpus_id="hp_t15",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        answer_model="fake-llm-test",
        prompt_version="answer-prompt-v1",
        answer_version="answer-v1",
        token_estimator="char-word-max-v1",
        num_candidates=30,
        max_chunks=8,
        max_context_tokens=3000,
        question_ids=ids,
        split="evaluation",
    )
    cfg_b = run_config_for_arm(
        run_id="t15-eval-b",
        strategy="hybrid_rerank",
        corpus_id="hp_t15",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        answer_model="fake-llm-test",
        prompt_version="answer-prompt-v1",
        answer_version="answer-v1",
        token_estimator="char-word-max-v1",
        num_candidates=30,
        max_chunks=8,
        max_context_tokens=3000,
        question_ids=ids,
        split="evaluation",
        reranker_model="fake-reranker-test",
    )
    scored_a, eval_a = evaluate_predictions(labels, preds, cfg_a)
    scored_b, eval_b = evaluate_predictions(labels, preds, cfg_b)
    comparison = compare_runs(scored_a, eval_a, scored_b, eval_b, labels, replicates=200)
    cards = headline_cards(comparison, eval_a, eval_b)
    assert [c["metric"] for c in cards] == ["Answer F1", "All-support coverage", "Latency"]
    assert "Compatible" in compatibility_banner(comparison)
    panels = cost_panels(
        {"baseline": {"input_tokens": 10, "output_tokens": 2, "estimated_cost_usd": None}},
        {"baseline": {"estimated_cost_usd": None, "note": "unpriced local compute"}},
    )
    assert any("unknown" in line for line in panels["query"])
    assert empty_state()["title"] == "No evaluation results yet"


def test_single_hop_freeze_gate_and_split_discipline(tmp_path: Path):
    controls_path = tmp_path / "single_hop.jsonl"
    rows = []
    for i in range(3):
        rows.append(
            {
                "question_id": f"q_c{i:02d}",
                "question": f"Who directed Film {i}?",
                "answer": f"Person {i}",
                "supporting_facts": [
                    {"title": f"Film {i}", "sentence_id": 0, "paragraph_id": f"p_film_{i}"}
                ],
                "source": {
                    "title": f"Film {i}",
                    "paragraph_id": f"p_film_{i}",
                    "sentences": [f"Film {i} was directed by Person {i}.", "It premiered locally."],
                },
                "split": "control",
                "question_type": "single_hop",
                "human_verified": False,
            }
        )
    controls_path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8"
    )
    controls = load_single_hop_controls(controls_path)
    assert len(controls) == 3
    draft = validate_single_hop_controls(controls, expected_count=3, require_verified=False)
    assert draft["n_verified"] == 0
    assert draft["question_set_hash"] == hash_question_set(sorted(c.question_id for c in controls))
    assert controls_freeze_hash(controls)
    with pytest.raises(ValueError, match="lack human review"):
        validate_single_hop_controls(controls, expected_count=3, require_verified=True)
    # Split discipline: tuning never overlaps locked or controls.
    ok = validate_split_disjoint(
        ["q_t01", "q_t02"],
        ["q_e01", "q_e02"],
        ["q_c00", "q_c01"],
        expected_tuning=2,
        expected_evaluation=2,
        expected_controls=2,
    )
    assert ok["tuning_size"] == 2
    with pytest.raises(ValueError, match="overlap locked"):
        validate_split_disjoint(
            ["q_e01"],
            ["q_e01"],
            ["q_c09"],
            expected_tuning=1,
            expected_evaluation=1,
            expected_controls=1,
        )
    with pytest.raises(ValueError, match="do not tune on controls"):
        validate_split_disjoint(
            ["q_c00"],
            ["q_e01"],
            ["q_c00"],
            expected_tuning=1,
            expected_evaluation=1,
            expected_controls=1,
        )
    assert assert_locked_question_use(["q_e01", "q_e02"], ["q_e02", "q_e01"]) == hash_question_set(
        ["q_e01", "q_e02"]
    )
    with pytest.raises(ValueError, match="exactly the frozen question ids"):
        assert_locked_question_use(["q_e01"], ["q_e01", "q_e02"])
    # Frozen evaluation config loads without secrets.
    eval_cfg = load_evaluation_config("configs/evaluation.yaml")
    assert eval_cfg.bootstrap_seed == 42
    assert eval_cfg.bootstrap_replicates == 2000
    assert eval_cfg.tuning_size == 30
    assert eval_cfg.evaluation_size == 120
    assert eval_cfg.control_size == 30


def test_configuration_guards_and_handoff_docs_exist():
    config = load_config("configs/default.yaml")
    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.budget.max_calls is not None or config.budget.max_total_tokens is not None
    dumped = config.model_dump()
    assert "neo4j_password" not in json.dumps(dumped) or "secret" not in json.dumps(dumped).lower()
    rows = [HotpotQuestion.model_validate(r) for r in _synthetic_source(4)]
    assert len(rows) == 4
    # Packing never truncates silently: over-budget chunks are dropped and counted.
    packed = pack_context(
        "What connects Person 0 to Town?",
        [
            {
                "chunk_id": "c_big",
                "paragraph_id": "p_big",
                "title": "Big",
                "text": "word " * 2000,
                "sentence_ids": [0],
                "sentence_texts": ["word " * 2000],
                "rank": 1,
            }
        ],
        AnswerConfig(max_chunks=8, max_context_tokens=300),
    )
    assert packed.dropped_chunks >= 1
    # Handoff docs are part of the deliverable.
    for name in ("docs/DEMO.md", "docs/RESULTS.md", "README.md"):
        assert Path(name).is_file(), f"missing handoff doc {name}"
    demo = Path("docs/DEMO.md").read_text(encoding="utf-8")
    assert "docker compose" in demo
    assert "worker" in demo.lower()
    assert "streamlit" in demo.lower()
    results = Path("docs/RESULTS.md").read_text(encoding="utf-8")
    assert "unknown" in results.lower()
    assert "limitation" in results.lower()
