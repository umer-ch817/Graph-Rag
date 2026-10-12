"""Reranking and controlled experiment arms (T11, R08)."""

from __future__ import annotations

import pytest

from graphrag_lab.answering import (
    AnswerConfig,
    answer_question,
    answer_question_with_triples,
    build_answer_messages_with_triples,
    count_triple_tokens,
    pack_context,
    pack_context_with_triples,
)
from graphrag_lab.evaluation import (
    LabelView,
    QuestionPrediction,
    evaluate_predictions,
    hash_question_set,
    run_config_for_arm,
    scored_to_result_rows,
)
from graphrag_lab.hybrid import rerank_hybrid_chunks, traversed_relations_for_result
from graphrag_lab.reranking import (
    ARMS,
    FakeRerankerBackend,
    RerankConfig,
    build_arm_manifest,
    check_arm_compatibility,
    collect_evidence_triples,
    compatibility_hash,
    deduplicate_chunks,
    rerank_candidates,
    validate_arm,
)
from graphrag_lab.retrieval import order_chunks_by_rerank, rerank_vector_chunks

QUESTION = "Who wrote The Night Cartographer and where was Elena Voss born?"


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
            "score": 0.9,
        },
        {
            "chunk_id": "c_author",
            "paragraph_id": "p_author",
            "title": "Elena Voss",
            "text": "Elena Voss was born in Lisbon. She writes historical novels.",
            "sentence_ids": [0, 1],
            "sentence_texts": [
                "Elena Voss was born in Lisbon.",
                "She writes historical novels.",
            ],
            "rank": 2,
            "score": 0.8,
        },
        {
            "chunk_id": "c_other",
            "paragraph_id": "p_other",
            "title": "Eiffel Tower",
            "text": "The Eiffel Tower is in Paris. It was completed in 1889.",
            "sentence_ids": [0, 1],
            "sentence_texts": [
                "The Eiffel Tower is in Paris.",
                "It was completed in 1889.",
            ],
            "rank": 3,
            "score": 0.1,
        },
    ]


def traversed():
    return [
        {
            "id": "rel_book_author",
            "source": "ent_book",
            "target": "ent_author",
            "type": "WRITTEN_BY",
            "evidence_chunk_ids": ["c_book"],
        },
        {
            "id": "rel_author_lisbon",
            "source": "ent_author",
            "target": "ent_lisbon",
            "type": "BORN_IN",
            "evidence_chunk_ids": ["c_author"],
        },
        {
            "id": "rel_bogus",
            "source": "ent_x",
            "target": "ent_y",
            "type": "LOCATED_IN",
            "evidence_chunk_ids": ["c_missing"],
        },
    ]


def test_arms_selectable_with_shared_budgets():
    assert list(ARMS) == ["vector", "vector_rerank", "hybrid_rerank", "hybrid_rerank_triples"]
    for arm in ARMS:
        assert validate_arm(arm) == arm
    with pytest.raises(Exception, match="Unknown experiment arm"):
        validate_arm("hybrid_magic")
    ids = ["q1", "q2"]
    configs = [
        run_config_for_arm(
            run_id=f"run-{arm}",
            strategy=arm,
            corpus_id="hp_t11",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            answer_model="fake-llm-test",
            prompt_version="answer-prompt-v1",
            answer_version="answer-v1",
            token_estimator="char-word-max-v1",
            num_candidates=30,
            max_chunks=8,
            max_context_tokens=3000,
            question_ids=ids,
            split="tuning",
            reranker_model=None if arm == "vector" else "fake-reranker-test",
        )
        for arm in ARMS
    ]
    # Identical final budgets across arms; only strategy/reranker/triples differ.
    for cfg in configs:
        assert cfg.max_chunks == 8
        assert cfg.max_context_tokens == 3000
    assert configs[0].reranker_model is None
    assert all(c.reranker_model == "fake-reranker-test" for c in configs[1:])
    assert [c.max_triples for c in configs] == [0, 0, 0, 12]
    assert [c.include_triples for c in configs] == [False, False, False, True]


def test_duplicate_chunks_removed_before_rerank():
    chunks = evidence_chunks() + [evidence_chunks()[0]]
    deduped, removed = deduplicate_chunks(chunks)
    assert [c["chunk_id"] for c in deduped] == ["c_book", "c_author", "c_other"]
    assert removed == ["c_book"]
    backend = FakeRerankerBackend(fixed_scores=[0.1, 0.2, 0.3])
    result = rerank_candidates(QUESTION, chunks, backend, RerankConfig())
    assert result.removed_duplicate_ids == ["c_book"]
    assert sorted(result.ordered_chunk_ids) == ["c_author", "c_book", "c_other"]
    # Packing the reranked order never sees duplicates.
    ordered = order_chunks_by_rerank(deduped, result.ordered_chunk_ids)
    packed = pack_context(QUESTION, ordered, AnswerConfig())
    assert len({c.chunk_id for c in packed.chunks}) == len(packed.chunks)


def test_raw_score_scales_not_added():
    # Vector order prefers c_book (0.9) over c_other (0.1); reranker reverses it.
    chunks = evidence_chunks()
    backend = FakeRerankerBackend(fixed_scores=[0.0, 0.0, 9.9])
    result = rerank_candidates(QUESTION, chunks, backend, RerankConfig())
    assert result.ordered_chunk_ids[0] == "c_other"
    assert result.degraded is False
    # Scores are kept separately; the caller never sums vector+reranker.
    assert set(result.rerank_scores) == {"c_book", "c_author", "c_other"}
    assert result.pre_rank == {"c_book": 1, "c_author": 2, "c_other": 3}
    # Tie-break is deterministic by chunk id.
    tied = rerank_candidates(
        QUESTION, chunks[:2], FakeRerankerBackend(fixed_scores=[1.0, 1.0]), RerankConfig()
    )
    assert tied.ordered_chunk_ids == ["c_author", "c_book"]


def test_rerank_helpers_share_dedupe_and_order():
    chunks = evidence_chunks()
    backend = FakeRerankerBackend(fixed_scores=[0.3, 0.2, 0.1])
    ordered_v, res_v = rerank_vector_chunks(QUESTION, chunks, backend, RerankConfig())
    assert [c["chunk_id"] for c in ordered_v] == ["c_book", "c_author", "c_other"]
    assert res_v.degraded is False
    hybrid_views = [
        {"chunk_id": c["chunk_id"], "title": c["title"], "text": c["text"], "rank": i + 1}
        for i, c in enumerate(chunks)
    ]
    ordered_h, res_h = rerank_hybrid_chunks(
        QUESTION, hybrid_views, FakeRerankerBackend(fixed_scores=[0.1, 0.5, 0.4]), RerankConfig()
    )
    assert [c["chunk_id"] for c in ordered_h] == ["c_author", "c_other", "c_book"]
    assert res_h.rerank_scores["c_author"] == pytest.approx(0.5)


def test_identical_final_budgets_across_arms():
    cfg = AnswerConfig()
    chunks = evidence_chunks()
    packed_plain = pack_context(QUESTION, chunks, cfg)
    packed_triples = pack_context_with_triples(QUESTION, chunks, traversed(), cfg, 12)
    for packed in (packed_plain, packed_triples):
        assert len(packed.chunks) <= 8
        assert packed.total_tokens <= 3000
    # Same chunk packing with and without triples (triples only use leftover).
    assert [c.chunk_id for c in packed_plain.chunks] == [c.chunk_id for c in packed_triples.chunks]
    assert packed_triples.triple_tokens > 0
    assert packed_triples.total_tokens == packed_plain.total_tokens + packed_triples.triple_tokens


def test_evidence_only_triples_enforced():
    cfg = AnswerConfig()
    packed = pack_context_with_triples(QUESTION, evidence_chunks(), traversed(), cfg, 12)
    ids = {t.relation_id for t in packed.triples}
    assert ids == {"rel_author_lisbon", "rel_book_author"}
    # Every triple's evidence survives packing; the bogus path is dropped.
    packed_ids = {c.chunk_id for c in packed.chunks}
    for triple in packed.triples:
        assert set(triple.evidence_chunk_ids) <= packed_ids
    assert len(packed.triples) <= 12
    # Triple tokens count against the same budget.
    assert packed.total_tokens <= cfg.max_context_tokens
    assert packed.triple_tokens == sum(t.token_count for t in packed.triples)
    # Messages include triples but reject dropped evidence.
    messages = build_answer_messages_with_triples(QUESTION, packed)
    assert "rel_book_author" in messages[1].content
    bad = packed.model_copy(
        update={
            "triples": [
                *packed.triples,
            ]
        }
    )
    # Manually inject a triple with dropped evidence: must be refused.
    from graphrag_lab.answering import PackedTriple

    with pytest.raises(Exception, match="dropped evidence"):
        build_answer_messages_with_triples(
            QUESTION,
            packed.model_copy(
                update={
                    "triples": [
                        PackedTriple(
                            relation_id="rel_bad",
                            source_entity_id="a",
                            target_entity_id="b",
                            type="LOCATED_IN",
                            evidence_chunk_ids=["c_missing"],
                            token_count=10,
                        )
                    ]
                }
            ),
        )
    _ = bad


def test_triple_cap_and_token_accounting():
    relations = [
        {
            "id": f"rel_{i:02d}",
            "source": f"s{i}",
            "target": f"t{i}",
            "type": "LOCATED_IN",
            "evidence_chunk_ids": ["c_book"],
        }
        for i in range(15)
    ]
    kept, diag = collect_evidence_triples(relations, ["c_book"], max_triples=12)
    assert len(kept) == 12
    assert diag["n_dropped_by_cap"] == 3
    assert [t.relation_id for t in kept] == [f"rel_{i:02d}" for i in range(12)]
    for triple in kept:
        assert count_triple_tokens(triple.model_dump(mode="json")) >= 1


def test_reranker_failure_degrades_with_flag():
    chunks = evidence_chunks()
    failing = FakeRerankerBackend(fail=True)
    result = rerank_candidates(QUESTION, chunks, failing, RerankConfig())
    assert result.degraded is True
    assert result.degraded_reason
    assert result.ordered_chunk_ids == ["c_book", "c_author", "c_other"]
    ordered, res = rerank_vector_chunks(QUESTION, chunks, failing, RerankConfig())
    assert res.degraded is True
    assert [c["chunk_id"] for c in ordered] == ["c_book", "c_author", "c_other"]


def test_reranker_truncation_recorded():
    long_text = "word " * 500
    chunks = [
        {
            "chunk_id": "c_long",
            "title": "Long",
            "text": long_text,
            "rank": 1,
        }
    ]
    backend = FakeRerankerBackend(fixed_scores=[1.0])
    result = rerank_candidates(QUESTION, chunks, backend, RerankConfig(max_seq_length=20))
    assert result.truncated_chunk_ids == ["c_long"]
    assert result.diagnostics["truncated"] == ["c_long"]


def test_compatibility_hash_and_manifests():
    ids = ["q1", "q2"]
    qhash = hash_question_set(ids)
    manifests = [
        build_arm_manifest(
            strategy=arm,
            corpus_id="hp_t11",
            index_fingerprint="fp-index",
            graph_fingerprint=None if arm in ("vector", "vector_rerank") else "fp-graph",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            answer_model="fake-llm-test",
            reranker_model=None if arm == "vector" else "fake-reranker-test",
            prompt_version="answer-prompt-v1",
            max_chunks=8,
            max_context_tokens=3000,
            max_triples=12 if arm == "hybrid_rerank_triples" else 0,
            include_triples=arm == "hybrid_rerank_triples",
            num_candidates=30,
            num_merged=30 if arm.startswith("hybrid") else 30,
            num_packed=3,
            total_tokens=1500,
            triple_tokens=120 if arm == "hybrid_rerank_triples" else 0,
            n_triples=2 if arm == "hybrid_rerank_triples" else 0,
            degraded=False,
            fallback=False,
            question_ids=ids,
            question_set_hash=qhash,
        )
        for arm in ARMS
    ]
    check = check_arm_compatibility(manifests)
    assert check["compatible"] is True
    assert check["compatibility_hash"]
    h1 = compatibility_hash(
        corpus_id="hp_t11",
        index_fingerprint="fp-index",
        question_ids=ids,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        answer_model="fake-llm-test",
        prompt_version="answer-prompt-v1",
        max_chunks=8,
        max_context_tokens=3000,
    )
    h2 = compatibility_hash(
        corpus_id="hp_t11",
        index_fingerprint="fp-index",
        question_ids=list(reversed(ids)),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        answer_model="fake-llm-test",
        prompt_version="answer-prompt-v1",
        max_chunks=8,
        max_context_tokens=3000,
    )
    assert h1 == h2
    bad = dict(manifests[1])
    bad["corpus_id"] = "hp_other"
    refused = check_arm_compatibility([manifests[0], bad])
    assert refused["compatible"] is False
    assert "refusing unlabeled paired delta" in refused["reason"]


def test_all_arms_run_on_fixture_questions_with_comparable_usage():
    from graphrag_lab.llm import FakeLlmBackend

    cfg = AnswerConfig()
    reranker = FakeRerankerBackend()
    questions = [
        ("q_book", "Who wrote The Night Cartographer?"),
        ("q_author", "Where was Elena Voss born?"),
    ]
    labels = [
        LabelView(
            question_id="q_book",
            answer="Elena Voss",
            supporting_facts=[
                {"title": "The Night Cartographer", "sentence_id": 0, "paragraph_id": "p_book"}
            ],
        ),
        LabelView(
            question_id="q_author",
            answer="Lisbon",
            supporting_facts=[
                {"title": "Elena Voss", "sentence_id": 0, "paragraph_id": "p_author"}
            ],
        ),
    ]
    ids = [q for q, _ in questions]
    manifests = []
    for arm in ARMS:
        predictions = []
        for qid, text in questions:
            chunks = evidence_chunks()
            degraded = False
            fallback = False
            triples: list[dict] = []
            if arm == "vector":
                ordered = chunks
            elif arm == "vector_rerank":
                ordered, res = rerank_vector_chunks(text, chunks, reranker, RerankConfig())
                degraded = res.degraded
            else:
                ordered, res = rerank_hybrid_chunks(text, chunks, reranker, RerankConfig())
                degraded = res.degraded
            if arm == "hybrid_rerank_triples":
                packed = pack_context_with_triples(text, ordered, traversed(), cfg, 12)
                triples = [t.model_dump(mode="json") for t in packed.triples]
                backend = FakeLlmBackend(
                    [
                        {
                            "answer": "Elena Voss" if qid == "q_book" else "Lisbon",
                            "status": "supported",
                            "citations": [
                                {"chunk_id": packed.chunks[0].chunk_id, "sentence_ids": [0]}
                            ],
                        }
                    ]
                )
                result = answer_question_with_triples(
                    text, ordered, traversed(), backend, cfg, None, 12
                )
            else:
                packed = pack_context(text, ordered, cfg)
                backend = FakeLlmBackend(
                    [
                        {
                            "answer": "Elena Voss" if qid == "q_book" else "Lisbon",
                            "status": "supported",
                            "citations": [
                                {"chunk_id": packed.chunks[0].chunk_id, "sentence_ids": [0]}
                            ],
                        }
                    ]
                )
                result = answer_question(text, ordered, backend, cfg)
            assert len(result.packed.chunks) <= 8
            assert result.packed.total_tokens <= 3000
            predictions.append(
                QuestionPrediction(
                    question_id=qid,
                    answer=result.answer,
                    status=result.status,
                    citations=[c.model_dump() for c in result.citations],
                    candidates=[
                        {
                            "chunk_id": c["chunk_id"],
                            "paragraph_id": c["paragraph_id"],
                            "title": c["title"],
                            "sentence_ids": list(c["sentence_ids"]),
                        }
                        for c in chunks
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
                    triples=triples,
                    reranked=arm != "vector",
                    strategy=arm,
                    fallback=fallback,
                    degraded=degraded,
                    input_tokens=result.packed.total_tokens,
                    output_tokens=10,
                )
            )
        run_cfg = run_config_for_arm(
            run_id=f"fixture-{arm}",
            strategy=arm,
            corpus_id="hp_t11",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            answer_model="fake-llm-test",
            prompt_version="answer-prompt-v1",
            answer_version="answer-v1",
            token_estimator="char-word-max-v1",
            num_candidates=30,
            max_chunks=8,
            max_context_tokens=3000,
            question_ids=ids,
            split="tuning",
            reranker_model=None if arm == "vector" else "fake-reranker-test",
        )
        scored, evaluation = evaluate_predictions(labels, predictions, run_cfg)
        assert len(scored) == 2
        rows = scored_to_result_rows(scored, predictions, labels)
        assert len(rows) == 2
        total_tokens = sum(p.input_tokens or 0 for p in predictions)
        manifests.append(
            build_arm_manifest(
                strategy=arm,
                corpus_id="hp_t11",
                index_fingerprint="fp-index",
                graph_fingerprint=None if arm in ("vector", "vector_rerank") else "fp-graph",
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                answer_model="fake-llm-test",
                reranker_model=None if arm == "vector" else "fake-reranker-test",
                prompt_version="answer-prompt-v1",
                max_chunks=8,
                max_context_tokens=3000,
                max_triples=12 if arm == "hybrid_rerank_triples" else 0,
                include_triples=arm == "hybrid_rerank_triples",
                num_candidates=30,
                num_merged=len(chunks),
                num_packed=len(predictions[0].packed),
                total_tokens=total_tokens,
                triple_tokens=sum(len(p.triples) for p in predictions),
                n_triples=sum(len(p.triples) for p in predictions),
                degraded=any(p.degraded for p in predictions),
                fallback=False,
                question_ids=ids,
                question_set_hash=hash_question_set(ids),
            )
        )
        # Only D carries triples; all arms share budgets and question set.
        if arm == "hybrid_rerank_triples":
            assert all(len(r["triples"]) > 0 for r in rows)
        else:
            assert all(r["triples"] == [] for r in rows)
    assert check_arm_compatibility(manifests)["compatible"] is True
    # Context usage is comparable: same packed cap and token ceiling.
    for manifest in manifests:
        assert manifest["max_chunks"] == 8
        assert manifest["max_context_tokens"] == 3000
        assert manifest["total_tokens"] >= 0


def test_traversed_helper_handles_legacy_count():
    from graphrag_lab.hybrid import HybridResult

    result = HybridResult(
        question=QUESTION,
        corpus_id="hp_t11",
        embedding_model="m",
        dimension=3,
        hybrid_version="hybrid-v1",
        total_vector=1,
        total_graph=0,
        total_merged=1,
        chunks=[
            {
                "chunk_id": "c_book",
                "corpus_id": "hp_t11",
                "paragraph_id": "p_book",
                "title": "The Night Cartographer",
                "text": "The Night Cartographer was written by Elena Voss.",
                "sentence_ids": [0],
                "sentence_texts": ["The Night Cartographer was written by Elena Voss."],
                "origin": "vector",
                "fusion_score": 0.1,
                "rank": 1,
            }
        ],
        linked_entity_ids=[],
        visited_entities=[],
        cap_hits=[],
        diagnostics={"expansion": {"status": "ok", "traversed_relations": 2}},
    )
    assert traversed_relations_for_result(result) == []
