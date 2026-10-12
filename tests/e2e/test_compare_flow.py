"""Answer comparison and evidence flow (T12c, R07/R08).

Covers the dashboard contract without the Streamlit runtime, Neo4j or an
LLM: strategy labels on side-by-side panels, exact citation resolution,
keyboard-operable relation rows, visible fallback flags, persisted
question/run state, tuning-only demo examples, matched-budget ablations and
local deep links. ``app/pages/`` owns presentation and live service wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.components.sources import (
    reader_blocks,
    reader_blocks_for_citation,
    relation_table_rows,
    source_row_label,
)
from graphrag_lab.compare_flow import (
    ARM_LABELS,
    DEFAULT_PAIR,
    ArmPanel,
    assert_matched_budgets,
    build_sources,
    describe_arm,
    encode_deep_link,
    fallback_text,
    load_evidence,
    load_panels,
    parse_deep_link,
    persist_compare,
    question_id_for,
    relation_rows,
    resolve_citation,
    tuning_examples,
    validate_arm_pair,
)
from graphrag_lab.run_store import RunStore


def _chunks() -> list[dict[str, Any]]:
    return [
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
            "text": "Elena Voss was born in Lisbon. She writes novels.",
            "sentence_ids": [0, 1],
            "sentence_texts": ["Elena Voss was born in Lisbon.", "She writes novels."],
            "origin": "graph",
            "rank": 2,
        },
        {
            "chunk_id": "c_other",
            "paragraph_id": "p_other",
            "title": "Eiffel Tower",
            "text": "The Eiffel Tower is in Paris.",
            "sentence_ids": [0],
            "sentence_texts": ["The Eiffel Tower is in Paris."],
            "origin": "vector",
            "rank": 3,
        },
    ]


def _traversed() -> list[dict[str, Any]]:
    return [
        {
            "id": "rel_book_author",
            "source": "The Night Cartographer",
            "target": "Elena Voss",
            "type": "WRITTEN_BY",
            "evidence_chunk_ids": ["c_book"],
        },
        {
            "id": "rel_author_lisbon",
            "source": "Elena Voss",
            "target": "Lisbon",
            "type": "BORN_IN",
            "evidence_chunk_ids": ["c_gone"],
        },
    ]


def test_panels_retain_strategy_labels():
    left, right = validate_arm_pair("vector", "hybrid_rerank_triples")
    assert ARM_LABELS[left] == "Vector"
    assert ARM_LABELS[right] == "Hybrid + rerank + triples"
    assert DEFAULT_PAIR == ("vector", "hybrid_rerank_triples")
    assert "rerank" in describe_arm("vector_rerank").lower()
    with pytest.raises(ValueError, match="must differ"):
        validate_arm_pair("vector", "vector")
    with pytest.raises(ValueError, match="Unknown experiment arm"):
        validate_arm_pair("vector", "hybrid_magic")
    assert validate_arm_pair("vector", None) == ("vector", None)
    assert question_id_for(" Where was she born? ") == question_id_for("Where was she born?")
    with pytest.raises(ValueError):
        question_id_for("   ")


def test_sources_carry_origin_and_context_flags():
    sources = build_sources(
        _chunks(),
        ["c_book", "c_author"],
        [{"chunk_id": "c_book", "sentence_ids": [0]}],
    )
    by_id = {source.chunk_id: source for source in sources}
    assert by_id["c_book"].origin == "Both"
    assert by_id["c_author"].origin == "Graph"
    assert by_id["c_other"].origin == "Vector"
    assert by_id["c_book"].in_final_context is True
    assert by_id["c_other"].in_final_context is False
    assert by_id["c_book"].cited is True
    assert by_id["c_author"].cited is False
    assert by_id["c_author"].sentence_range == "s0–s1"
    # Packed order leads; the rest follow by chunk id.
    assert [source.chunk_id for source in sources][:2] == ["c_book", "c_author"]
    label = source_row_label(by_id["c_book"])
    assert "The Night Cartographer" in label and "s0" in label and "Both" in label
    with pytest.raises(ValueError, match="Duplicate retrieved chunk"):
        build_sources(_chunks() + [_chunks()[0]], ["c_book"], [])


def test_citations_open_exact_source_sentences():
    sources = build_sources(
        _chunks(),
        ["c_book", "c_author"],
        [{"chunk_id": "c_author", "sentence_ids": [0]}],
    )
    resolved = resolve_citation({"chunk_id": "c_author", "sentence_ids": [0]}, sources)
    assert resolved["source"].chunk_id == "c_author"
    blocks = reader_blocks_for_citation(resolved)
    assert blocks[0] == {
        "sentence_id": 0,
        "text": "Elena Voss was born in Lisbon.",
        "marker": "[cited]",
    }
    assert blocks[1]["marker"] == "[retrieved]"
    # Surrounding text stays readable; the cue is textual, not color-only.
    assert all("text" in block and "marker" in block for block in blocks)
    with pytest.raises(LookupError, match="unknown chunk"):
        resolve_citation({"chunk_id": "c_missing", "sentence_ids": [0]}, sources)
    with pytest.raises(LookupError, match="outside the retrieved context"):
        resolve_citation({"chunk_id": "c_book", "sentence_ids": [7]}, sources)


def test_relation_table_flags_missing_evidence():
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in _chunks()}
    rows = relation_rows(_traversed(), chunks_by_id)
    assert [row.relation_id for row in rows] == ["rel_author_lisbon", "rel_book_author"]
    good = next(row for row in rows if row.relation_id == "rel_book_author")
    assert good.evidence_available is True
    assert good.evidence_title == "The Night Cartographer"
    bad = next(row for row in rows if row.relation_id == "rel_author_lisbon")
    assert bad.evidence_available is False
    table = relation_table_rows(rows)
    assert table[0]["evidence"] == "c_gone (evidence not in context)"
    assert table[0]["selectable"] is False
    assert table[1]["selectable"] is True
    assert table[1]["source"] == "The Night Cartographer"
    assert table[1]["relation"] == "WRITTEN_BY"
    assert table[1]["target"] == "Elena Voss"


def test_relation_table_option_text():
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in _chunks()}
    rows = relation_table_rows(relation_rows(_traversed(), chunks_by_id))
    assert rows[1]["source"] == "The Night Cartographer"
    assert rows[1]["relation"] == "WRITTEN_BY"
    assert rows[1]["target"] == "Elena Voss"


def test_fallback_and_cap_flags_are_visible():
    banner = fallback_text(fallback=True, reason="no_linked_entities", cap_hits=["max_merged_hit"])
    assert "No linked entities; vector evidence shown" in banner
    assert "no_linked_entities" in banner
    assert "max_merged_hit" in banner
    assert fallback_text(fallback=False, reason=None, cap_hits=[]) == ""
    assert "max_graph_chunks_hit" in fallback_text(
        fallback=False, reason=None, cap_hits=["max_graph_chunks_hit"]
    )


def test_reader_blocks_label_cited_vs_retrieved():
    sources = build_sources(_chunks(), ["c_book"], [{"chunk_id": "c_book", "sentence_ids": [0]}])
    blocks = reader_blocks(sources[0], [0])
    assert blocks == [
        {
            "sentence_id": 0,
            "text": "The Night Cartographer was written by Elena Voss.",
            "marker": "[cited]",
        }
    ]
    plain = reader_blocks(sources[0])
    assert plain[0]["marker"] == "[retrieved]"


def test_persisted_runs_round_trip_without_rerun(tmp_path: Path):
    runs = RunStore(tmp_path / "jobs.sqlite3")
    panels = [
        {
            "strategy": "vector",
            "answer": "Elena Voss",
            "status": "supported",
            "citations": [{"chunk_id": "c_book", "sentence_ids": [0]}],
            "candidates": [{"chunk_id": "c_book"}, {"chunk_id": "c_other"}],
            "packed": [{"chunk_id": "c_book"}],
            "latency_ms": 12.5,
            "retrieval": {
                "num_candidates": 2,
                "num_packed": 1,
                "fallback": False,
                "chunks": _chunks()[:2],
                "packed_ids": ["c_book"],
                "traversed": [],
                "cap_hits": [],
            },
            "usage": {"input_tokens": 100, "output_tokens": 5},
        },
        {
            "strategy": "hybrid_rerank_triples",
            "answer": "",
            "status": "failed",
            "citations": [],
            "retrieval": {"num_candidates": 0, "num_packed": 0, "failure_message": "boom"},
            "failure_code": "PROVIDER_ERROR",
        },
    ]
    run_ids = persist_compare(
        runs,
        question="Who wrote The Night Cartographer?",
        question_id="q_demo",
        snapshot_id=None,
        panels=panels,
        config={"max_chunks": 8, "max_context_tokens": 3000},
    )
    assert set(run_ids) == {"vector", "hybrid_rerank_triples"}
    loaded = load_panels(runs, [run_ids["vector"], run_ids["hybrid_rerank_triples"]])
    assert [panel.strategy for panel in loaded] == ["vector", "hybrid_rerank_triples"]
    assert [panel.strategy_label for panel in loaded] == ["Vector", "Hybrid + rerank + triples"]
    assert loaded[0].status == "supported" and loaded[0].answer == "Elena Voss"
    assert loaded[0].n_candidates == 2 and loaded[0].n_packed == 1
    assert loaded[1].status == "failed" and loaded[1].failure_code == "PROVIDER_ERROR"
    # Full evidence review reads the same saved records; nothing reruns.
    evidence = load_evidence(runs, run_ids["vector"])
    assert evidence["strategy"] == "vector"
    assert [chunk["chunk_id"] for chunk in evidence["chunks"]][:2] == ["c_book", "c_author"]
    assert evidence["packed_ids"] == ["c_book"]
    assert evidence["citations"] == [{"chunk_id": "c_book", "sentence_ids": [0]}]
    with pytest.raises(ValueError, match="cannot be opened"):
        load_evidence(runs, "run_missing")
    # Panels carry no verdict: identical/different answers are never labeled.
    assert not any(hasattr(panel, "verdict") for panel in loaded)
    assert isinstance(loaded[0], ArmPanel)
    with pytest.raises(ValueError, match="cannot be opened"):
        load_panels(runs, ["run_missing"])
    with pytest.raises(ValueError, match="nonempty"):
        load_panels(runs, [])


def test_tuning_picker_excludes_locked_examples(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    tuning_one = '{"id": "q_t1", "question": "Tuning one?", "type": "bridge", "split": "tuning"}\n'
    tuning_two = (
        '{"id": "q_t2", "question": "Tuning two?", "type": "comparison", "split": "tuning"}\n'
    )
    locked_one = (
        '{"id": "q_e1", "question": "Locked one?", "type": "bridge", "split": "evaluation"}\n'
    )
    (dataset / "questions.jsonl").write_text(
        "".join([tuning_one, tuning_two, locked_one]),
        encoding="utf-8",
    )
    rows = tuning_examples(dataset)
    assert [row["id"] for row in rows] == ["q_t1", "q_t2"]
    assert all("answer" not in row for row in rows)
    with pytest.raises(ValueError, match="Prepare a dataset first"):
        tuning_examples(tmp_path / "absent")


def test_matched_budget_guard_for_ablations():
    ok = assert_matched_budgets(
        [
            {"max_chunks": 8, "max_context_tokens": 3000},
            {"max_chunks": 8, "max_context_tokens": 3000},
        ]
    )
    assert ok == {"max_chunks": 8, "max_context_tokens": 3000}
    with pytest.raises(ValueError, match="share final budgets"):
        assert_matched_budgets(
            [
                {"max_chunks": 8, "max_context_tokens": 3000},
                {"max_chunks": 4, "max_context_tokens": 3000},
            ]
        )


def test_deep_links_carry_run_and_source():
    link = encode_deep_link(["run_a", "run_b"], "c_book")
    assert link == "run=run_a&run=run_b&src=c_book"
    parsed = parse_deep_link(link)
    assert parsed == {"run_ids": ["run_a", "run_b"], "source_id": "c_book"}
    assert parse_deep_link("run=run_a") == {"run_ids": ["run_a"], "source_id": None}
    with pytest.raises(ValueError, match="no run ids"):
        parse_deep_link("src=c_book")
