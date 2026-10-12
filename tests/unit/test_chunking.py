"""Sentence-preserving chunking and provenance (T03, checkpoint C1)."""

import json
from pathlib import Path

import pytest

from graphrag_lab.chunking import (
    ChunkConfig,
    ChunkOverflowError,
    build_chunk_id,
    chunk_dataset_directory,
    chunk_paragraph,
    chunk_paragraphs,
    count_chunk_tokens,
    estimate_tokens,
)
from graphrag_lab.corpus import build_artifacts, prepare_dataset
from graphrag_lab.models import HotpotQuestion, SampleConfig
from graphrag_lab.provenance import (
    find_chunks_for_sentence,
    index_paragraphs,
    verify_chunk,
    verify_chunks,
    verify_gold_covered_by_chunks,
    verify_supporting_facts,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "corpus.json"


@pytest.fixture
def fixture_corpus():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return payload["corpus_id"], payload["paragraphs"]


@pytest.fixture
def by_id(fixture_corpus):
    _, paragraphs = fixture_corpus
    return index_paragraphs(paragraphs)


def paragraph_by_id(paragraphs, paragraph_id):
    for paragraph in paragraphs:
        if paragraph["id"] == paragraph_id:
            return paragraph
    raise AssertionError(f"missing paragraph {paragraph_id}")


def test_fixture_exists_and_has_variant_cases(fixture_corpus):
    corpus_id, paragraphs = fixture_corpus
    assert corpus_id
    ids = {p["id"] for p in paragraphs}
    assert {"p_ada", "p_long", "p_town_a", "p_town_b", "p_dense"} <= ids
    towns = [p for p in paragraphs if p["title"] == "Town"]
    assert len(towns) == 2
    assert towns[0]["sentences"] != towns[1]["sentences"]


def test_short_paragraph_is_single_chunk_and_round_trips(fixture_corpus, by_id):
    corpus_id, paragraphs = fixture_corpus
    paragraph = paragraph_by_id(paragraphs, "p_ada")
    chunks = chunk_paragraph(paragraph, corpus_id, ChunkConfig())
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.paragraph_id == "p_ada"
    assert chunk.sentence_ids == [0, 1]
    assert chunk.sentence_texts == paragraph["sentences"]
    assert chunk.text == " ".join(paragraph["sentences"])
    assert chunk.token_count == count_chunk_tokens(chunk.title, chunk.text)
    verify_chunk(chunk, by_id)


def test_long_paragraph_splits_with_one_sentence_overlap(fixture_corpus, by_id):
    corpus_id, paragraphs = fixture_corpus
    paragraph = paragraph_by_id(paragraphs, "p_long")
    # One sentence is ~23 tokens; two sentences are 40-41 tokens and three
    # are 57 tokens, so target 45 yields two-sentence windows with overlap.
    config = ChunkConfig(target_tokens=45, max_tokens=70, overlap_sentences=1)
    chunks = chunk_paragraph(paragraph, corpus_id, config)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert chunk.token_count <= config.max_tokens
        verify_chunk(chunk, by_id, config)
    # Adjacent chunks share exactly the boundary sentence (one-sentence overlap).
    for first, second in zip(chunks, chunks[1:]):
        assert first.sentence_ids[-1] == second.sentence_ids[0]
        assert set(first.sentence_ids) & set(second.sentence_ids) == {second.sentence_ids[0]}
    # Union covers every sentence with no gaps.
    covered: set[int] = set()
    for chunk in chunks:
        covered.update(chunk.sentence_ids)
    assert covered == set(range(len(paragraph["sentences"])))


def test_zero_overlap_still_covers_without_loop(fixture_corpus, by_id):
    corpus_id, paragraphs = fixture_corpus
    paragraph = paragraph_by_id(paragraphs, "p_long")
    config = ChunkConfig(target_tokens=45, max_tokens=70, overlap_sentences=0)
    chunks = chunk_paragraph(paragraph, corpus_id, config)
    assert len(chunks) >= 2
    covered: set[int] = set()
    for chunk in chunks:
        covered.update(chunk.sentence_ids)
        verify_chunk(chunk, by_id, config)
    assert covered == set(range(len(paragraph["sentences"])))


def test_long_single_sentence_is_rejected_not_truncated(fixture_corpus):
    corpus_id, paragraphs = fixture_corpus
    paragraph = paragraph_by_id(paragraphs, "p_dense")
    # Default budget already overflows for the 200-word sentence.
    with pytest.raises(ChunkOverflowError, match="exceeding max_tokens"):
        chunk_paragraph(paragraph, corpus_id, ChunkConfig())
    tiny = ChunkConfig(target_tokens=10, max_tokens=20, overlap_sentences=1)
    with pytest.raises(ChunkOverflowError, match="refusing to truncate"):
        chunk_paragraph(paragraph, corpus_id, tiny)


def test_single_sentence_over_target_but_under_max_is_kept():
    paragraph = {
        "id": "p_mid",
        "title": "T",
        "sentences": [" ".join(["word"] * 30) + ".", "Short sentence."],
    }
    # First sentence alone needs ~30+ words; allow it to exceed target but fit max.
    config = ChunkConfig(target_tokens=20, max_tokens=60, overlap_sentences=1)
    chunks = chunk_paragraph(paragraph, "hp_x", config)
    assert chunks[0].sentence_ids == [0]
    assert chunks[0].token_count <= config.max_tokens
    verify_chunks(chunks, [paragraph], config)


def test_duplicate_titles_keep_distinct_chunks(fixture_corpus, by_id):
    corpus_id, paragraphs = fixture_corpus
    first = paragraph_by_id(paragraphs, "p_town_a")
    second = paragraph_by_id(paragraphs, "p_town_b")
    assert first["title"] == second["title"] == "Town"
    assert first["sentences"] != second["sentences"]
    chunks = chunk_paragraphs([first, second], corpus_id, ChunkConfig())
    assert len(chunks) == 2
    assert chunks[0].paragraph_id != chunks[1].paragraph_id
    assert chunks[0].id != chunks[1].id
    assert chunks[0].sentence_texts != chunks[1].sentence_texts
    verify_chunks(chunks, by_id)


def test_variant_paragraphs_survive_corpus_dedup_and_chunking():
    rows = [
        HotpotQuestion.model_validate(
            {
                "_id": "q1",
                "question": "What connects the towns?",
                "answer": "yes",
                "type": "bridge",
                "level": "hard",
                "context": [
                    ["Town", ["Town fact version one."]],
                    ["Other", ["Unrelated."]],
                ],
                "supporting_facts": [["Town", 0]],
            }
        ),
        HotpotQuestion.model_validate(
            {
                "_id": "q2",
                "question": "What else connects the towns?",
                "answer": "no",
                "type": "bridge",
                "level": "hard",
                "context": [
                    ["Town", ["Town fact version two carries different detail."]],
                    ["Other", ["Unrelated."]],
                ],
                "supporting_facts": [["Town", 0]],
            }
        ),
    ]
    payloads = build_artifacts([(rows[0], "tuning"), (rows[1], "evaluation")])
    paragraphs = [json.loads(line) for line in payloads["paragraphs.jsonl"].splitlines()]
    towns = [p for p in paragraphs if p["title"] == "Town"]
    assert len(towns) == 2
    assert towns[0]["id"] != towns[1]["id"]
    chunks = chunk_paragraphs(paragraphs, "hp_variant", ChunkConfig())
    town_chunks = [c for c in chunks if c.title == "Town"]
    assert len(town_chunks) == 2
    verify_chunks(chunks, paragraphs)


def test_chunk_ids_are_deterministic_and_corpus_scoped(fixture_corpus):
    corpus_id, paragraphs = fixture_corpus
    subset = [p for p in paragraphs if p["id"] in {"p_ada", "p_town_a", "p_town_b"}]
    first = chunk_paragraphs(subset, corpus_id, ChunkConfig())
    second = chunk_paragraphs(list(reversed(subset)), corpus_id, ChunkConfig())
    assert [c.id for c in first] == [c.id for c in second]
    other = chunk_paragraphs(subset, "hp_different", ChunkConfig())
    assert {c.id for c in first}.isdisjoint({c.id for c in other})
    expected = build_chunk_id(corpus_id, "p_ada", 0, 2, "sent-overlap-v1")
    assert first[[c.paragraph_id for c in first].index("p_ada")].id == expected


def test_token_counts_are_stored_and_bounded(fixture_corpus, by_id):
    corpus_id, paragraphs = fixture_corpus
    subset = [p for p in paragraphs if p["id"] != "p_dense"]
    config = ChunkConfig(target_tokens=25, max_tokens=50, overlap_sentences=1)
    chunks = chunk_paragraphs(subset, corpus_id, config)
    for chunk in chunks:
        assert chunk.token_count == count_chunk_tokens(chunk.title, chunk.text)
        assert chunk.token_count <= config.max_tokens
        assert chunk.content_hash
    verify_chunks(chunks, by_id, config)


def test_estimate_tokens_is_conservative_and_deterministic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") >= 2
    long_word = "antidisestablishmentarianism"
    assert estimate_tokens(long_word) >= len(long_word) / 4
    assert estimate_tokens("a b c") == estimate_tokens("a b c")


def test_invalid_config_and_inputs_rejected():
    with pytest.raises(Exception, match="target_tokens"):
        ChunkConfig(target_tokens=300, max_tokens=100)
    with pytest.raises(ValueError, match="nonempty"):
        chunk_paragraph({"id": "p", "title": "T", "sentences": ["s."]}, "", ChunkConfig())
    with pytest.raises(ValueError, match="at least one sentence"):
        chunk_paragraph({"id": "p", "title": "T", "sentences": []}, "hp", ChunkConfig())
    with pytest.raises(ValueError, match="at least one nonempty sentence"):
        chunk_paragraph({"id": "p", "title": "T", "sentences": ["", "   "]}, "hp", ChunkConfig())
    with pytest.raises(ValueError, match="Duplicate paragraph id"):
        chunk_paragraphs(
            [
                {"id": "p", "title": "A", "sentences": ["one."]},
                {"id": "p", "title": "A", "sentences": ["two."]},
            ],
            "hp",
            ChunkConfig(),
        )


def test_upstream_empty_sentence_keeps_stable_indices():
    """Upstream empty strings stay in place; indices never shift (R02)."""
    paragraph = {
        "id": "p_empty",
        "title": "Town",
        "sentences": ["Town fact one.", "", "Town fact three."],
    }
    (chunk,) = chunk_paragraph(paragraph, "hp_empty", ChunkConfig())
    assert chunk.sentence_ids == [0, 1, 2]
    assert chunk.sentence_texts == ["Town fact one.", "", "Town fact three."]
    verify_chunk(chunk, index_paragraphs([paragraph]))


def test_provenance_catches_mismatched_sentence_text(fixture_corpus, by_id):
    corpus_id, paragraphs = fixture_corpus
    paragraph = paragraph_by_id(paragraphs, "p_ada")
    (chunk,) = chunk_paragraph(paragraph, corpus_id, ChunkConfig())
    tampered = chunk.model_copy(
        update={"sentence_texts": ["altered sentence.", chunk.sentence_texts[1]]}
    )
    with pytest.raises(ValueError, match="does not match paragraph"):
        verify_chunk(tampered, by_id)
    tampered_id = chunk.model_copy(update={"id": "c_wrong"})
    with pytest.raises(ValueError, match="does not match corpus"):
        verify_chunk(tampered_id, by_id)


def test_gold_facts_resolve_to_sentences_and_chunks(tmp_path):
    source = tmp_path / "source.json"
    rows = [
        {
            "_id": f"q-{i:03}",
            "question": f"What connects person {i}?",
            "answer": f"answer-{i}",
            "type": "bridge" if i % 2 else "comparison",
            "level": "hard",
            "context": [
                [f"Person {i}", [f"Person {i} directed a film.", "A second sentence."]],
                ["Town", [f"Town fact version {i % 2}."]],
            ],
            "supporting_facts": [[f"Person {i}", 1], ["Town", 0]],
        }
        for i in range(8)
    ]
    source.write_text(json.dumps(rows), encoding="utf-8")
    out = tmp_path / "dataset"
    manifest = prepare_dataset(source, out, SampleConfig(sample_size=8, tuning_size=2, seed=42))
    paragraphs = [json.loads(line) for line in (out / "paragraphs.jsonl").read_text().splitlines()]
    labels = [json.loads(line) for line in (out / "labels.jsonl").read_text().splitlines()]
    verify_supporting_facts(labels, paragraphs)
    chunks = chunk_paragraphs(paragraphs, manifest.corpus_id, ChunkConfig())
    verify_chunks(chunks, paragraphs)
    verify_gold_covered_by_chunks(labels, chunks)
    label = labels[0]
    fact = label["supporting_facts"][0]
    covering = find_chunks_for_sentence(chunks, fact["paragraph_id"], fact["sentence_id"])
    assert covering
    assert all(fact["sentence_id"] in c.sentence_ids for c in covering)


def test_chunk_dataset_directory_round_trip(tmp_path):
    source = tmp_path / "source.json"
    rows = [
        {
            "_id": f"q-{i:03}",
            "question": f"Question {i}?",
            "answer": f"answer-{i}",
            "type": "bridge",
            "level": "hard",
            "context": [[[f"Title {i}", [f"Sentence one for {i}.", f"Sentence two for {i}."]]][0]],
            "supporting_facts": [[f"Title {i}", 0]],
        }
        for i in range(4)
    ]
    source.write_text(json.dumps(rows), encoding="utf-8")
    out = tmp_path / "dataset"
    manifest = prepare_dataset(source, out, SampleConfig(sample_size=4, tuning_size=1, seed=7))
    first = chunk_dataset_directory(out, config=ChunkConfig())
    assert first.name == "chunks.jsonl"
    assert first.is_file()
    second = chunk_dataset_directory(out, config=ChunkConfig())
    assert first.read_bytes() == second.read_bytes()
    stored = [json.loads(line) for line in first.read_text().splitlines()]
    paragraphs = [json.loads(line) for line in (out / "paragraphs.jsonl").read_text().splitlines()]
    assert len(stored) == len(paragraphs)
    verify_chunks(stored, paragraphs)
    assert {c["corpus_id"] for c in stored} == {manifest.corpus_id}
    first.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        chunk_dataset_directory(out, config=ChunkConfig())
