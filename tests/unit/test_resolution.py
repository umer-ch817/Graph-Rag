"""Conservative entity resolution (T08, R05)."""

import json
from pathlib import Path

import pytest

from graphrag_lab.llm import InvalidInputError, ValidationFailedError
from graphrag_lab.models import EntityType
from graphrag_lab.resolution import (
    COMMON_FIRST_NAMES,
    LEXICAL_THRESHOLD,
    RESOLVER_VERSION,
    CanonicalEntity,
    ResolverConfig,
    is_acronym,
    lexical_similarity,
    load_resolution_config,
    normalize_name,
    resolve_entities,
    summarize_resolution,
    write_resolution_artifacts,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "entity_cases.json"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "resolution.yaml"


def extraction(chunk_id, rows):
    return {
        "chunk_id": chunk_id,
        "entities": [
            {
                "local_id": local_id,
                "name": name,
                "type": entity_type,
                "sentence_ids": sentence_ids,
            }
            for local_id, name, entity_type, sentence_ids in rows
        ],
    }


def entity_of(result, mention_id):
    for assignment in result.assignments:
        if assignment.mention_id == mention_id:
            return assignment.canonical_entity_id
    raise AssertionError(f"missing assignment {mention_id}")


def test_normalization_folds_unicode_case_and_punctuation():
    assert normalize_name("  Ada   Lovelace  ") == "ada lovelace"
    assert normalize_name("J. Smith") == "j smith"
    assert normalize_name("Ada's Engine") == "ada engine"
    assert normalize_name("José García") == "jose garcia"
    assert normalize_name("IBM") == "ibm"
    assert normalize_name("International-Business_Machines!") == "international business machines"
    with pytest.raises(InvalidInputError):
        normalize_name("   ")
    with pytest.raises(InvalidInputError):
        normalize_name("...!!!")


def test_lexical_similarity_and_acronym_helpers():
    assert lexical_similarity("ada lovelace", "ada lovelace") == 1.0
    assert lexical_similarity("ada lovelace", "zzz qqq") < 0.5
    assert lexical_similarity("john smith", "jon smith") > lexical_similarity(
        "john smith", "ada lovelace"
    )
    assert is_acronym("ibm", "international business machines")
    assert is_acronym("i b m", "international business machines")
    assert not is_acronym("ibm", "ada lovelace")
    assert not is_acronym("js", "john smith", min_tokens=3)


def test_distinctive_exact_match_merges_with_aliases_preserved():
    extractions = [
        extraction("c_1", [("e1", "Ada Lovelace", "PERSON", [0])]),
        extraction("c_2", [("e1", "Ada Lovelace", "PERSON", [1])]),
    ]
    result = resolve_entities(extractions, ResolverConfig(), corpus_id="hp_t08")
    assert len(result.entities) == 1
    entity = result.entities[0]
    assert entity.canonical_name == "Ada Lovelace"
    assert entity.normalized_name == "ada lovelace"
    assert entity.type == EntityType.PERSON
    assert entity.member_mention_ids == ["c_1#e1", "c_2#e1"]
    assert entity.aliases == ["Ada Lovelace"]
    assert all(a.decision == "merged" or "create-entity" in a.reason for a in result.assignments)
    assert result.corpus_id == "hp_t08"
    assert result.resolver_version == RESOLVER_VERSION


def test_corroborated_middle_alias_merges():
    extractions = [
        extraction("c_1", [("e1", "Ada Lovelace", "PERSON", [0])]),
        extraction("c_2", [("e1", "Ada Augusta Lovelace", "PERSON", [0])]),
    ]
    result = resolve_entities(extractions, ResolverConfig(), corpus_id="hp_t08")
    assert len(result.entities) == 1
    assert result.entities[0].canonical_name == "Ada Augusta Lovelace"
    assert sorted(result.entities[0].aliases) == ["Ada Augusta Lovelace", "Ada Lovelace"]


def test_acronym_alias_merges_same_type():
    extractions = [
        extraction("c_1", [("e1", "International Business Machines", "ORGANIZATION", [0])]),
        extraction("c_2", [("e1", "IBM", "ORGANIZATION", [0])]),
    ]
    result = resolve_entities(extractions, ResolverConfig(), corpus_id="hp_t08")
    assert len(result.entities) == 1
    assert "merge-acronym" in ";".join(a.reason for a in result.assignments)


def test_type_mismatch_never_merges():
    extractions = [
        extraction("c_p", [("e1", "Jordan", "PERSON", [0])]),
        extraction("c_l", [("e1", "Jordan", "LOCATION", [0])]),
    ]
    result = resolve_entities(extractions, ResolverConfig(), corpus_id="hp_t08")
    assert len(result.entities) == 2
    assert entity_of(result, "c_p#e1") != entity_of(result, "c_l#e1")
    assert any("type-mismatch" in a.reason for a in result.assignments)


def test_initials_ambiguity_stays_separate():
    extractions = [
        extraction("c_john", [("e1", "John Smith", "PERSON", [0])]),
        extraction("c_initial", [("e1", "J. Smith", "PERSON", [0])]),
        extraction("c_jack", [("e1", "Jack Smith", "PERSON", [0])]),
    ]
    result = resolve_entities(extractions, ResolverConfig(), corpus_id="hp_t08")
    assert entity_of(result, "c_john#e1") != entity_of(result, "c_initial#e1")
    assert entity_of(result, "c_initial#e1") != entity_of(result, "c_jack#e1")
    reasons = " ".join(a.reason for a in result.assignments)
    assert "initials" in reasons
    # Even without the conflicting Jack, the initial alone stays separate.
    pair = resolve_entities(extractions[:2], ResolverConfig(), corpus_id="hp_t08")
    assert entity_of(pair, "c_john#e1") != entity_of(pair, "c_initial#e1")


def test_same_name_distinct_people_stay_separate():
    assert "john" in COMMON_FIRST_NAMES
    extractions = [
        extraction("c_a", [("e1", "John Smith", "PERSON", [0])]),
        extraction("c_b", [("e1", "John Smith", "PERSON", [0])]),
    ]
    result = resolve_entities(extractions, ResolverConfig(), corpus_id="hp_t08")
    assert len(result.entities) == 2
    assert entity_of(result, "c_a#e1") != entity_of(result, "c_b#e1")
    assert any("common-name" in a.reason for a in result.assignments)


def test_single_token_person_stays_separate_location_merges():
    persons = [
        extraction("c_a", [("e1", "John", "PERSON", [0])]),
        extraction("c_b", [("e1", "John", "PERSON", [0])]),
    ]
    result = resolve_entities(persons, ResolverConfig(), corpus_id="hp_t08")
    assert len(result.entities) == 2
    locations = [
        extraction("c_a", [("e1", "Paris", "LOCATION", [0])]),
        extraction("c_b", [("e1", "Paris", "LOCATION", [0])]),
    ]
    merged = resolve_entities(locations, ResolverConfig(), corpus_id="hp_t08")
    assert len(merged.entities) == 1


def test_transitive_bridge_does_not_collapse_conflicting_middles():
    extractions = [
        extraction("c_a", [("e1", "John A. Smith", "PERSON", [0])]),
        extraction("c_bridge", [("e1", "John Smith", "PERSON", [0])]),
        extraction("c_b", [("e1", "John B. Smith", "PERSON", [0])]),
    ]
    result = resolve_entities(extractions, ResolverConfig(), corpus_id="hp_t08")
    assert entity_of(result, "c_a#e1") != entity_of(result, "c_b#e1")
    # The bridge joins at most one side; all three never share one entity.
    ids = {entity_of(result, m) for m in ("c_a#e1", "c_bridge#e1", "c_b#e1")}
    assert len(ids) == 2
    assert any(
        "conflicting-middle" in a.reason or "transitive" in a.reason for a in result.assignments
    )


def test_every_mention_auditable_with_scores_and_reasons():
    extractions = [
        extraction(
            "c_1",
            [
                ("e1", "Ada Lovelace", "PERSON", [0]),
                ("e2", "Analytical Engine", "WORK", [1]),
            ],
        ),
        extraction("c_2", [("e1", "Ada Augusta Lovelace", "PERSON", [0])]),
    ]
    result = resolve_entities(extractions, ResolverConfig(), corpus_id="hp_t08")
    assert len(result.assignments) == 3
    assert sorted(a.mention_id for a in result.assignments) == [
        "c_1#e1",
        "c_1#e2",
        "c_2#e1",
    ]
    for assignment in result.assignments:
        assert assignment.name and assignment.normalized_name == normalize_name(assignment.name)
        assert assignment.canonical_entity_id
        assert assignment.decision in {"merged", "separate"}
        assert assignment.reason
        assert isinstance(assignment.candidate_scores, dict) and assignment.candidate_scores
        assert assignment.resolver_version == RESOLVER_VERSION
    # Original surface forms survive on the merged entity.
    ada = next(e for e in result.entities if e.type == EntityType.PERSON)
    assert sorted(ada.aliases) == ["Ada Augusta Lovelace", "Ada Lovelace"]
    assert sorted(ada.member_mention_ids) == ["c_1#e1", "c_2#e1"]


def test_deterministic_independent_of_input_order_and_corpus_scoped():
    extractions = [
        extraction("c_b", [("e1", "John B. Smith", "PERSON", [0])]),
        extraction("c_a", [("e1", "John A. Smith", "PERSON", [0])]),
        extraction("c_bridge", [("e1", "John Smith", "PERSON", [0])]),
    ]
    first = resolve_entities(extractions, ResolverConfig(), corpus_id="hp_t08")
    second = resolve_entities(list(reversed(extractions)), ResolverConfig(), corpus_id="hp_t08")
    assert [e.id for e in first.entities] == [e.id for e in second.entities]
    assert [a.canonical_entity_id for a in first.assignments] == [
        a.canonical_entity_id for a in second.assignments
    ]
    other = resolve_entities(extractions, ResolverConfig(), corpus_id="hp_other")
    assert {e.id for e in other.entities}.isdisjoint({e.id for e in first.entities})
    assert len(other.entities) == len(first.entities)


def test_fixture_cases_match_expected_pairings():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["resolver_version"] == RESOLVER_VERSION
    assert payload["corpus_id"]
    names = {case["name"] for case in payload["cases"]}
    assert {
        "initials_ambiguity",
        "same_name_distinct_people",
        "transitive_merge_collision",
    } <= names
    for case in payload["cases"]:
        result = resolve_entities(
            case["extractions"], ResolverConfig(), corpus_id="hp_fixture_t08_v1"
        )
        for left, right in case.get("expect_same_entity", []):
            assert entity_of(result, left) == entity_of(result, right), case["name"]
        for left, right in case.get("expect_separate", []):
            assert entity_of(result, left) != entity_of(result, right), case["name"]


def test_invalid_inputs_rejected_with_codes():
    with pytest.raises(InvalidInputError) as excinfo:
        resolve_entities([], ResolverConfig(), corpus_id="   ")
    assert excinfo.value.code == "INVALID_INPUT"
    with pytest.raises(InvalidInputError):
        resolve_entities(
            [
                extraction("c_1", [("e1", "Ada Lovelace", "PERSON", [0])]),
                extraction("c_1", [("e1", "Ada Lovelace", "PERSON", [0])]),
            ],
            ResolverConfig(),
            corpus_id="hp_t08",
        )
    bad_type = extraction("c_1", [("e1", "Ada Lovelace", "ALIEN", [0])])
    with pytest.raises(ValidationFailedError) as excinfo:
        resolve_entities([bad_type], ResolverConfig(), corpus_id="hp_t08")
    assert excinfo.value.code == "VALIDATION_FAILED"
    assert isinstance(CanonicalEntity, type)


def test_artifact_write_is_atomic_guarded_and_config_loads(tmp_path):
    extractions = [
        extraction("c_1", [("e1", "Ada Lovelace", "PERSON", [0])]),
        extraction("c_2", [("e1", "Ada Augusta Lovelace", "PERSON", [0])]),
    ]
    result = resolve_entities(extractions, ResolverConfig(), corpus_id="hp_t08")
    info = write_resolution_artifacts(tmp_path / "snapshot", result, config=ResolverConfig())
    for name in ("entities.jsonl", "resolution.jsonl", "manifest.json"):
        assert (tmp_path / "snapshot" / name).is_file()
    assert set(info["artifact_sha256"]) == {
        "entities.jsonl",
        "resolution.jsonl",
        "manifest.json",
    }
    rows = [
        json.loads(line)
        for line in (tmp_path / "snapshot" / "resolution.jsonl").read_text().splitlines()
    ]
    assert {row["mention_id"] for row in rows} == {"c_1#e1", "c_2#e1"}
    assert all(row["resolver_version"] == RESOLVER_VERSION for row in rows)
    assert all(row["candidate_scores"] for row in rows)
    second = write_resolution_artifacts(tmp_path / "snapshot", result, config=ResolverConfig())
    assert second["artifact_sha256"] == info["artifact_sha256"]
    other = resolve_entities(
        [extraction("c_1", [("e1", "Ada Lovelace", "PERSON", [0])])],
        ResolverConfig(),
        corpus_id="hp_t08",
    )
    with pytest.raises(ValueError, match="differs"):
        write_resolution_artifacts(tmp_path / "snapshot", other, config=ResolverConfig())
    cfg = load_resolution_config(CONFIG_PATH)
    assert cfg.resolver_version == RESOLVER_VERSION
    assert cfg.lexical_threshold == LEXICAL_THRESHOLD == pytest.approx(0.88)
    assert cfg.common_name_guard is True
    summary = summarize_resolution(result)
    assert summary["n_entities"] == 1 and summary["n_mentions"] == 2
