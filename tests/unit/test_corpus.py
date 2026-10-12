"""Reproducibility and leakage boundaries for dataset preparation."""

import hashlib
import json
from collections import Counter

import pytest
from pydantic import ValidationError

from graphrag_lab.corpus import prepare_dataset, select_questions
from graphrag_lab.models import HotpotQuestion, SampleConfig


def examples(count=20):
    return [
        {
            "_id": f"question-{i:03}",
            "question": f"What connects person {i} to the town?",
            "answer": f"answer-{i}",
            "type": "bridge" if i % 4 else "comparison",
            "level": "hard",
            "context": [
                [f"Person {i}", [f"Person {i} directed a film.", "A second sentence."]],
                ["Town", [f"Town fact version {i % 2}."]],
                ["Distractor", ["Unrelated but retained."]],
            ],
            "supporting_facts": [[f"Person {i}", 1], ["Town", 0]],
        }
        for i in range(count)
    ]


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps(examples()), encoding="utf-8")
    return path


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_selection_is_order_independent_stratified_and_disjoint():
    rows = [HotpotQuestion.model_validate(row) for row in examples()]
    config = SampleConfig(sample_size=12, tuning_size=4, seed=42)
    first = select_questions(rows, config)
    assert first == select_questions(list(reversed(rows)), config)
    assert len(first) == 12
    assert Counter(row.type for row, split in first) == {"bridge": 9, "comparison": 3}
    tuning = {row.id for row, split in first if split == "tuning"}
    evaluation = {row.id for row, split in first if split == "evaluation"}
    assert len(tuning) == 4 and len(evaluation) == 8
    assert not tuning & evaluation
    assert Counter(row.type for row, split in first if split == "tuning") == {
        "bridge": 3, "comparison": 1,
    }


def test_seed_changes_selection():
    rows = [HotpotQuestion.model_validate(row) for row in examples()]
    first = select_questions(rows, SampleConfig(sample_size=12, tuning_size=4, seed=42))
    second = select_questions(rows, SampleConfig(sample_size=12, tuning_size=4, seed=43))
    assert {row.id for row, _ in first} != {row.id for row, _ in second}


def test_sample_cannot_exceed_source():
    rows = [HotpotQuestion.model_validate(row) for row in examples(3)]
    with pytest.raises(ValueError, match="source"):
        select_questions(rows, SampleConfig(sample_size=5, tuning_size=1))


@pytest.mark.parametrize("settings", [
    {"sample_size": 0}, {"sample_size": True}, {"sample_size": 10, "tuning_size": 10},
    {"sample_size": 10, "tuning_size": 0}, {"seed": -1},
])
def test_invalid_sample_config_rejected(settings):
    with pytest.raises(ValidationError):
        SampleConfig(**settings)


def test_artifacts_are_byte_reproducible_and_hash_verified(source, tmp_path):
    config = SampleConfig(sample_size=12, tuning_size=4)
    one, two = tmp_path / "one", tmp_path / "two"
    first = prepare_dataset(source, one, config)
    second = prepare_dataset(source, two, config)
    assert first == second
    for path in one.iterdir():
        assert path.read_bytes() == (two / path.name).read_bytes()
    assert first.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    for name, digest in first.artifact_sha256.items():
        assert hashlib.sha256((one / name).read_bytes()).hexdigest() == digest


def test_paragraph_variants_distractors_and_labels_are_preserved(source, tmp_path):
    out = tmp_path / "prepared"
    prepare_dataset(source, out, SampleConfig(sample_size=20, tuning_size=4))
    questions = read_lines(out / "questions.jsonl")
    labels = read_lines(out / "labels.jsonl")
    paragraphs = read_lines(out / "paragraphs.jsonl")
    contexts = read_lines(out / "contexts.jsonl")
    assert all("answer" not in q and "supporting_facts" not in q for q in questions)
    assert all("question_ids" not in p and "supporting_facts" not in p for p in paragraphs)
    assert len([p for p in paragraphs if p["title"] == "Town"]) == 2
    assert len([p for p in paragraphs if p["title"] == "Distractor"]) == 1
    assert len(contexts) == 60
    by_id = {p["id"]: p for p in paragraphs}
    assert {q["id"] for q in questions} == {label["question_id"] for label in labels}
    for label in labels:
        for fact in label["supporting_facts"]:
            paragraph = by_id[fact["paragraph_id"]]
            assert paragraph["title"] == fact["title"]
            assert paragraph["sentences"][fact["sentence_id"]]


def test_duplicate_question_ids_fail_before_output(source, tmp_path):
    source.write_text(json.dumps(examples() + examples(1)), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate question"):
        prepare_dataset(source, tmp_path / "out", SampleConfig(sample_size=12, tuning_size=4))
    assert not (tmp_path / "out").exists()


def test_invalid_supporting_fact_rejected():
    row = examples(1)[0]
    row["supporting_facts"] = [["Town", 99]]
    with pytest.raises(ValidationError, match="supporting"):
        HotpotQuestion.model_validate(row)


def test_upstream_invalid_rows_are_quarantined_and_recorded(tmp_path):
    """One upstream-invalid row (bad supporting index, as in HotpotQA dev).

    prepare quarantines it with id+reason in the manifest and samples from
    valid rows only; the raw source bytes are untouched and reruns agree.
    """
    rows = examples(8)
    rows.append(
        {
            "_id": "question-bad",
            "question": "Which team is fictional?",
            "answer": "Nowhere FC",
            "type": "bridge",
            "level": "hard",
            "context": [["Town", ["Only one sentence."]]],
            "supporting_facts": [["Town", 902]],
        }
    )
    path = tmp_path / "source.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    config = SampleConfig(sample_size=6, tuning_size=2, seed=42)
    out_one, out_two = tmp_path / "one", tmp_path / "two"
    first = prepare_dataset(path, out_one, config)
    assert first.source_question_count == 9
    assert first.quarantined_question_ids == ["question-bad"]
    assert "question-bad" in first.quarantined_reasons
    assert "question-bad" not in first.selected_question_ids
    assert len(first.selected_question_ids) == 6
    # Raw source untouched; rerun is byte-identical including the manifest.
    second = prepare_dataset(path, out_two, config)
    assert first == second
    for name in ("questions.jsonl", "labels.jsonl", "manifest.json"):
        assert (out_one / name).read_bytes() == (out_two / name).read_bytes()
    # Valid-only sources record an empty quarantine (existing behavior kept).
    clean_path = tmp_path / "clean-source.json"
    clean_path.write_text(json.dumps(examples()), encoding="utf-8")
    clean = prepare_dataset(clean_path, tmp_path / "clean", config)
    assert clean.quarantined_question_ids == []
    assert clean.quarantined_reasons == {}


def test_existing_output_is_verified_not_overwritten(source, tmp_path):
    out = tmp_path / "prepared"
    config = SampleConfig(sample_size=12, tuning_size=4)
    manifest = prepare_dataset(source, out, config)
    assert prepare_dataset(source, out, config) == manifest
    (out / "questions.jsonl").write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="Existing output"):
        prepare_dataset(source, out, config)
    assert (out / "questions.jsonl").read_text() == "corrupt"


def test_changed_source_checksum_changes_identity(source, tmp_path):
    config = SampleConfig(sample_size=12, tuning_size=4)
    first = prepare_dataset(source, tmp_path / "one", config)
    source.write_bytes(source.read_bytes() + b"\n")
    second = prepare_dataset(source, tmp_path / "two", config)
    assert first.corpus_id != second.corpus_id
