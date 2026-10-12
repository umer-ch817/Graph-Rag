"""Validated extraction and pilot audit (T07, R04)."""

import json
from pathlib import Path

import pytest

from graphrag_lab.chunking import build_content_hash
from graphrag_lab.config import BudgetConfig
from graphrag_lab.extraction import (
    COMPATIBILITY,
    EXTRACTION_VERSION,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    ChunkExtraction,
    ExtractionCache,
    ExtractionConfig,
    QuarantinedChunkError,
    build_extraction_messages,
    estimate_full_extraction,
    extract_chunks,
    extract_one_chunk,
    extraction_cache_key,
    load_extraction_config,
    parse_extraction_payload,
    render_pilot_markdown,
    select_pilot_chunks,
    summarize_pilot,
    validate_chunk_extraction,
    write_extraction_artifacts,
)
from graphrag_lab.llm import (
    BudgetExceededError,
    BudgetTracker,
    FakeLlmBackend,
    InvalidInputError,
    ProviderError,
    ValidationFailedError,
)
from graphrag_lab.models import EntityType, RelationType

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "extraction.yaml"


def ada_chunk(chunk_id: str = "c_ada") -> dict:
    title = "Ada Lovelace"
    texts = [
        "Ada Lovelace wrote the first algorithm.",
        "She worked with Charles Babbage on the Analytical Engine.",
    ]
    return {
        "chunk_id": chunk_id,
        "paragraph_id": "p_ada",
        "title": title,
        "text": " ".join(texts),
        "sentence_ids": [0, 1],
        "sentence_texts": texts,
        "content_hash": build_content_hash(title, " ".join(texts)),
        "corpus_id": "hp_test",
    }


def valid_payload(chunk_id: str = "c_ada") -> dict:
    return {
        "schema_version": "1",
        "chunk_id": chunk_id,
        "entities": [
            {"local_id": "e1", "name": "Ada Lovelace", "type": "PERSON", "sentence_ids": [0]},
            {"local_id": "e2", "name": "first algorithm", "type": "WORK", "sentence_ids": [0]},
        ],
        "relations": [
            {
                "source": "e2",
                "type": "WRITTEN_BY",
                "target": "e1",
                "evidence": [
                    {"sentence_id": 0, "quote": "Ada Lovelace wrote the first algorithm."}
                ],
            }
        ],
    }


def test_valid_extraction_parses_and_validates():
    chunk = ada_chunk()
    parsed = parse_extraction_payload(json.dumps(valid_payload()), chunk, ExtractionConfig())
    assert isinstance(parsed, ChunkExtraction)
    assert parsed.chunk_id == "c_ada"
    assert [e.local_id for e in parsed.entities] == ["e1", "e2"]
    assert parsed.relations[0].type == RelationType.WRITTEN_BY
    # Empty lists are valid extractions.
    empty = parse_extraction_payload(
        json.dumps({"schema_version": "1", "chunk_id": "c_ada", "entities": [], "relations": []}),
        chunk,
        ExtractionConfig(),
    )
    assert empty.entities == [] and empty.relations == []


def test_unknown_types_are_rejected():
    chunk = ada_chunk()
    bad_entity = valid_payload()
    bad_entity["entities"][0]["type"] = "ALIEN"
    with pytest.raises(ValidationFailedError, match="unknown entity type"):
        parse_extraction_payload(json.dumps(bad_entity), chunk, ExtractionConfig())
    bad_relation = valid_payload()
    bad_relation["relations"][0]["type"] = "LIKES"
    with pytest.raises(ValidationFailedError, match="unknown relation type"):
        parse_extraction_payload(json.dumps(bad_relation), chunk, ExtractionConfig())


def test_extra_fields_forbidden_endpoints_and_uniqueness():
    chunk = ada_chunk()
    extra = valid_payload()
    extra["entities"][0]["nickname"] = "x"
    with pytest.raises(ValidationFailedError, match="schema invalid"):
        parse_extraction_payload(json.dumps(extra), chunk, ExtractionConfig())
    dangling = valid_payload()
    dangling["relations"][0]["source"] = "e9"
    with pytest.raises(ValidationFailedError, match="dangling"):
        parse_extraction_payload(json.dumps(dangling), chunk, ExtractionConfig())
    duplicate = valid_payload()
    duplicate["entities"][1]["local_id"] = "e1"
    with pytest.raises(ValidationFailedError, match="repeats local_id"):
        parse_extraction_payload(json.dumps(duplicate), chunk, ExtractionConfig())


def test_sentence_ownership_and_verbatim_span_checks():
    chunk = ada_chunk()
    outside_entity = valid_payload()
    outside_entity["entities"][0]["sentence_ids"] = [7]
    with pytest.raises(ValidationFailedError, match="outside this chunk"):
        parse_extraction_payload(json.dumps(outside_entity), chunk, ExtractionConfig())
    outside_evidence = valid_payload()
    outside_evidence["relations"][0]["evidence"][0]["sentence_id"] = 5
    with pytest.raises(ValidationFailedError, match="outside this chunk"):
        parse_extraction_payload(json.dumps(outside_evidence), chunk, ExtractionConfig())
    invented_quote = valid_payload()
    invented_quote["relations"][0]["evidence"][0]["quote"] = "Ada invented the internet."
    with pytest.raises(ValidationFailedError, match="verbatim"):
        parse_extraction_payload(json.dumps(invented_quote), chunk, ExtractionConfig())
    # Wrong-sentence quote (verbatim elsewhere, not in cited sentence) also fails.
    miscited = valid_payload()
    miscited["relations"][0]["evidence"][0] = {
        "sentence_id": 1,
        "quote": "Ada Lovelace wrote the first algorithm.",
    }
    with pytest.raises(ValidationFailedError, match="verbatim"):
        parse_extraction_payload(json.dumps(miscited), chunk, ExtractionConfig())


def test_caps_and_direction_compatibility():
    chunk = ada_chunk()
    cfg = ExtractionConfig(max_entities=1)
    with pytest.raises(ValidationFailedError, match="max_entities"):
        parse_extraction_payload(json.dumps(valid_payload()), chunk, cfg)
    cfg_rel = ExtractionConfig(max_relations=0 if False else 1)
    assert cfg_rel.max_relations == 1
    # Direction mismatch: WRITTEN_BY requires WORK -> PERSON.
    flipped = valid_payload()
    flipped["relations"][0] = {
        "source": "e1",
        "type": "WRITTEN_BY",
        "target": "e2",
        "evidence": [{"sentence_id": 0, "quote": "Ada Lovelace wrote the first algorithm."}],
    }
    with pytest.raises(ValidationFailedError, match="rejects direction"):
        parse_extraction_payload(json.dumps(flipped), chunk, ExtractionConfig())
    # Every relation type has at least one documented direction.
    assert set(COMPATIBILITY) == set(RelationType)
    for rel, pairs in COMPATIBILITY.items():
        assert pairs, f"{rel} needs a compatibility entry"


def test_malformed_json_and_chunk_mismatch_rejected():
    chunk = ada_chunk()
    with pytest.raises(ValidationFailedError, match="not valid JSON"):
        parse_extraction_payload("not json", chunk, ExtractionConfig())
    wrong_chunk = valid_payload(chunk_id="c_other")
    with pytest.raises(ValidationFailedError, match="chunk_id"):
        parse_extraction_payload(json.dumps(wrong_chunk), chunk, ExtractionConfig())
    wrong_schema = valid_payload()
    wrong_schema["schema_version"] = "2"
    with pytest.raises(ValidationFailedError, match="schema_version"):
        parse_extraction_payload(json.dumps(wrong_schema), chunk, ExtractionConfig())


def test_cache_hit_avoids_second_call_and_budget():
    chunk = ada_chunk()
    backend = FakeLlmBackend([valid_payload()])
    tracker = BudgetTracker(BudgetConfig(max_calls=10, max_total_tokens=100000))
    cache = ExtractionCache()
    first = extract_one_chunk(chunk, backend, ExtractionConfig(), tracker, cache)
    assert first.chunk_id == "c_ada"
    assert tracker.used_calls == 1
    assert len(backend.calls) == 1
    # Queue is now empty; a cache hit must not call the backend.
    second = extract_one_chunk(chunk, backend, ExtractionConfig(), tracker, cache)
    assert second == first
    assert tracker.used_calls == 1
    assert len(backend.calls) == 1
    # Cache key covers model revision and prompt/schema versions.
    assert extraction_cache_key(chunk, "m", ExtractionConfig()) != extraction_cache_key(
        chunk, "other", ExtractionConfig()
    )


def test_retry_then_success_counts_every_attempt():
    chunk = ada_chunk()
    delays: list[float] = []
    backend = FakeLlmBackend(
        ["not json", {"schema_version": "1", "chunk_id": "c_wrong"}, valid_payload()]
    )
    tracker = BudgetTracker(BudgetConfig(max_calls=10, max_total_tokens=1000000))
    result = extract_one_chunk(
        chunk,
        backend,
        ExtractionConfig(),
        tracker,
        cache=ExtractionCache(),
        sleep_fn=delays.append,
    )
    assert result.chunk_id == "c_ada"
    assert len(backend.calls) == 3
    assert tracker.used_calls == 3
    assert len(delays) == 2


def test_exhausted_retries_quarantine_and_batch_collects_failure():
    chunk = ada_chunk()
    delays: list[float] = []
    backend = FakeLlmBackend(["bad1", "bad2", "bad3", "bad4"])
    tracker = BudgetTracker(BudgetConfig(max_calls=10, max_total_tokens=1000000))
    with pytest.raises(QuarantinedChunkError) as excinfo:
        extract_one_chunk(
            chunk, backend, ExtractionConfig(), tracker, ExtractionCache(), delays.append
        )
    assert excinfo.value.attempts == 3
    assert excinfo.value.last_code == "VALIDATION_FAILED"
    assert tracker.used_calls == 3
    assert len(delays) == 2
    # Batch collects the quarantine instead of aborting the whole run.
    backend2 = FakeLlmBackend([valid_payload(), "bad1", "bad2", "bad3"])
    good = ada_chunk("c_ada")
    bad = ada_chunk("c_bad")
    extractions, failures = extract_chunks(
        [good, bad],
        backend2,
        ExtractionConfig(),
        BudgetTracker(BudgetConfig(max_calls=10, max_total_tokens=1000000)),
        ExtractionCache(),
        lambda _: None,
    )
    assert [e.chunk_id for e in extractions] == ["c_ada"]
    assert len(failures) == 1
    assert failures[0].chunk_id == "c_bad"
    assert failures[0].attempts == 3


def test_transient_provider_error_retries_with_backoff():
    class FlakyBackend:
        model_name = "fake-llm-test"

        @property
        def model_revision(self) -> str:
            return self.model_name

        def __init__(self) -> None:
            self.calls: list = []
            self._failed = False
            self._inner = FakeLlmBackend([valid_payload()])

        def complete(self, request):  # type: ignore[no-untyped-def]
            self.calls.append(request)
            if not self._failed:
                self._failed = True
                raise ProviderError("busy", retryable=True)
            return self._inner.complete(request)

    delays: list[float] = []
    backend = FlakyBackend()
    tracker = BudgetTracker(BudgetConfig(max_calls=10, max_total_tokens=1000000))
    result = extract_one_chunk(
        ada_chunk(), backend, ExtractionConfig(), tracker, ExtractionCache(), delays.append
    )
    assert result.chunk_id == "c_ada"
    assert len(delays) == 1
    assert tracker.used_calls == 1  # failed attempt had no usage to reconcile


def test_non_retryable_provider_error_quarantines_without_retry():
    backend = FakeLlmBackend([])  # empty queue -> non-retryable PROVIDER_ERROR
    delays: list[float] = []
    with pytest.raises(QuarantinedChunkError) as excinfo:
        extract_one_chunk(
            ada_chunk(),
            backend,
            ExtractionConfig(),
            BudgetTracker(BudgetConfig(max_calls=10, max_total_tokens=1000000)),
            ExtractionCache(),
            delays.append,
        )
    assert excinfo.value.attempts == 1
    assert delays == []


def test_budget_reservation_blocks_retry_overrun():
    chunk = ada_chunk()
    backend = FakeLlmBackend(["bad1", "bad2", "bad3"])
    tracker = BudgetTracker(BudgetConfig(max_calls=1, max_total_tokens=1000000))
    with pytest.raises((BudgetExceededError, QuarantinedChunkError)):
        extract_one_chunk(
            chunk, backend, ExtractionConfig(), tracker, ExtractionCache(), lambda _: None
        )
    assert tracker.used_calls <= 1


def test_pilot_selection_is_deterministic_and_summarized(tmp_path: Path):
    chunks = [ada_chunk(f"c_{i:03d}") for i in range(25)]
    first = select_pilot_chunks(chunks, size=20, seed=42)
    second = select_pilot_chunks(list(reversed(chunks)), size=20, seed=42)
    assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in second]
    assert len(first) == 20
    ordered_ids = sorted(c["chunk_id"] for c in first)
    backend = FakeLlmBackend([valid_payload(chunk_id) for chunk_id in ordered_ids])
    tracker = BudgetTracker(BudgetConfig(max_calls=100, max_total_tokens=10000000))
    extractions, failures = extract_chunks(
        first, backend, ExtractionConfig(), tracker, ExtractionCache(), lambda _: None
    )
    report = summarize_pilot(
        extractions,
        failures,
        pilot_chunk_ids=[c["chunk_id"] for c in first],
        model="fake-llm-test",
        config=ExtractionConfig(),
        input_tokens=tracker.used_tokens,
        output_tokens=0,
        calls=tracker.used_calls,
    )
    assert report["pilot_size"] == 20
    assert report["n_extracted"] == 20 and report["n_failed"] == 0
    assert report["total_relations"] == 20
    assert report["relation_type_counts"] == {"WRITTEN_BY": 20}
    markdown = render_pilot_markdown(report)
    assert "Audit checklist" in markdown and "verbatim" in markdown
    estimate = estimate_full_extraction(
        {"pilot_size": 20, "calls": 20, "input_tokens": 2000, "output_tokens": 200},
        total_chunks=150,
    )
    assert estimate["estimated_calls"] == 150
    # Small corpora pilot to available chunks.
    assert len(select_pilot_chunks(chunks[:5], size=20, seed=42)) == 5


def test_artifact_write_is_atomic_guarded_and_config_loads(tmp_path: Path):
    chunk = ada_chunk()
    extraction = parse_extraction_payload(json.dumps(valid_payload()), chunk, ExtractionConfig())
    info = write_extraction_artifacts(
        tmp_path / "pilot",
        [extraction],
        [],
        config=ExtractionConfig(),
        model="fake-llm-test",
        corpus_id="hp_test",
    )
    for name in ("extractions.jsonl", "failures.jsonl", "manifest.json"):
        assert (tmp_path / "pilot" / name).is_file()
    assert set(info["artifact_sha256"]) == {
        "extractions.jsonl",
        "failures.jsonl",
        "manifest.json",
    }
    rows = [
        json.loads(line)
        for line in (tmp_path / "pilot" / "extractions.jsonl").read_text().splitlines()
    ]
    assert rows[0]["chunk_id"] == "c_ada"
    assert rows[0]["prompt_version"] == PROMPT_VERSION
    # Idempotent re-export; differing content is refused.
    second = write_extraction_artifacts(
        tmp_path / "pilot",
        [extraction],
        [],
        config=ExtractionConfig(),
        model="fake-llm-test",
        corpus_id="hp_test",
    )
    assert second["artifact_sha256"] == info["artifact_sha256"]
    other = valid_payload()
    other["entities"][0]["name"] = "Ada Augusta Lovelace"
    other_extraction = parse_extraction_payload(json.dumps(other), chunk, ExtractionConfig())
    with pytest.raises(ValueError, match="differs"):
        write_extraction_artifacts(
            tmp_path / "pilot",
            [other_extraction],
            [],
            config=ExtractionConfig(),
            model="fake-llm-test",
            corpus_id="hp_test",
        )
    cfg = load_extraction_config(CONFIG_PATH)
    assert cfg.schema_version == SCHEMA_VERSION == "1"
    assert cfg.prompt_version == PROMPT_VERSION
    assert cfg.extraction_version == EXTRACTION_VERSION
    assert cfg.max_entities == 40 and cfg.max_relations == 60
    assert cfg.temperature == 0.0 and cfg.max_retries == 2


def test_error_codes_match_backend_schema():
    chunk = ada_chunk()
    try:
        coerce_bad = {
            "chunk_id": "",
            "title": "t",
            "text": "x",
            "sentence_ids": [0],
            "sentence_texts": ["x"],
        }
        parse_extraction_payload(json.dumps(valid_payload()), coerce_bad, ExtractionConfig())
    except InvalidInputError as exc:
        assert exc.code == "INVALID_INPUT"
    else:
        raise AssertionError("empty chunk id must raise")
    try:
        bad = valid_payload()
        bad["relations"][0]["evidence"][0]["quote"] = "invented"
        parse_extraction_payload(json.dumps(bad), chunk, ExtractionConfig())
    except ValidationFailedError as exc:
        assert exc.code == "VALIDATION_FAILED"
    else:
        raise AssertionError("invented quote must raise")
    tracker = BudgetTracker(BudgetConfig(max_calls=1, max_total_tokens=10))
    try:
        tracker.reserve(5, 6)
    except BudgetExceededError as exc:
        assert exc.code == "BUDGET_EXCEEDED"
    else:
        raise AssertionError("token overrun must raise")
    assert build_extraction_messages(chunk, ExtractionConfig())[0].role == "system"


def test_validate_accepts_model_instances_and_rejects_bad_local_ids():
    chunk = ada_chunk()
    extraction = ChunkExtraction.model_validate(valid_payload())
    assert validate_chunk_extraction(extraction, chunk, ExtractionConfig()).chunk_id == "c_ada"
    bad = valid_payload()
    bad["entities"][0]["local_id"] = "1bad"
    with pytest.raises(ValidationFailedError, match="local_id"):
        parse_extraction_payload(json.dumps(bad), chunk, ExtractionConfig())
    assert EntityType.PERSON.value == "PERSON"
    assert RelationType.WRITTEN_BY.value == "WRITTEN_BY"
