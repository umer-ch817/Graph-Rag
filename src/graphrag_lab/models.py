"""Dataset boundary contracts for T02 (HotpotQA sample, no chunking).

T07 adds the fixed extraction vocabulary (R04). The enums live here so
chunking, extraction, resolution and hybrid retrieval share one definition;
``extraction.py`` owns validation against chunk evidence.

T16 adds the optional decomposition constants (R13). The planner cap lives
here so decomposition, evaluation and future dashboard code share one
definition; ``decomposition.py`` owns planning and sequential retrieval.
"""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

HOTPOT_URL = "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

Text = Annotated[str, Field(strict=True, min_length=1)]
SentenceIndex = Annotated[int, Field(strict=True, ge=0)]
QuestionType = Literal["bridge", "comparison"]
Split = Literal["tuning", "evaluation"]


class EntityType(StrEnum):
    """Fixed entity vocabulary (BACKEND-SCHEMA extraction contract)."""

    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    WORK = "WORK"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"


class RelationType(StrEnum):
    """Fixed relation vocabulary; arbitrary names are rejected."""

    DIRECTED_BY = "DIRECTED_BY"
    WRITTEN_BY = "WRITTEN_BY"
    BORN_IN = "BORN_IN"
    DIED_IN = "DIED_IN"
    LOCATED_IN = "LOCATED_IN"
    PART_OF = "PART_OF"
    MEMBER_OF = "MEMBER_OF"
    FOUNDED_BY = "FOUNDED_BY"
    SPOUSE_OF = "SPOUSE_OF"
    PARENT_OF = "PARENT_OF"
    CREATED_BY = "CREATED_BY"
    PARTICIPATED_IN = "PARTICIPATED_IN"


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class SampleConfig(Record):
    """Seeded stratified sampling parameters (R01)."""

    sample_size: int = Field(default=150, ge=2, le=200, strict=True)
    tuning_size: int = Field(default=30, ge=1, strict=True)
    seed: int = Field(default=42, ge=0, strict=True)

    @model_validator(mode="after")
    def _splits_must_be_nonempty(self) -> Self:
        if self.tuning_size >= self.sample_size:
            raise ValueError("tuning_size must be smaller than sample_size")
        return self


class HotpotQuestion(Record):
    """One validated HotpotQA dev-distractor entry."""

    id: Text = Field(alias="_id")
    question: Text
    answer: Text
    type: QuestionType
    level: Literal["easy", "medium", "hard"]
    context: list[tuple[Text, list[StrictStr]]] = Field(min_length=1)
    supporting_facts: list[tuple[Text, SentenceIndex]] = Field(min_length=1)

    @model_validator(mode="after")
    def _support_must_resolve(self) -> Self:
        seen: dict[str, list[str]] = {}
        for title, sentences in self.context:
            if not sentences:
                raise ValueError("Context paragraph must contain sentences")
            if title in seen and seen[title] != list(sentences):
                raise ValueError("Ambiguous context title within one question")
            seen[title] = list(sentences)
        for title, index in self.supporting_facts:
            if title not in seen or index >= len(seen[title]):
                raise ValueError("Unresolvable supporting fact title or sentence index")
        return self


class DatasetManifest(Record):
    """Reproducibility record linking source, selection and artifact hashes."""

    schema_version: str = "1"
    preprocessing_version: str = "hotpot-paragraphs-v1"
    sampling_version: str = "stratified-sha256-v1"
    corpus_id: str
    source_url: str
    source_sha256: str
    source_bytes: int
    source_question_count: int
    source_type_counts: dict[str, int]
    sample: SampleConfig
    selected_question_ids: list[str]
    split_ids: dict[str, list[str]]
    split_type_counts: dict[str, dict[str, int]]
    paragraph_count: int
    context_count: int
    artifact_sha256: dict[str, str]
    # Upstream-invalid rows quarantined at load (R01 honesty): ids from the
    # raw source that failed HotpotQuestion validation plus short reasons.
    # The sample draws from valid rows only; the raw source bytes (and their
    # hash) are unchanged so the exclusion stays auditable. Empty by default.
    quarantined_question_ids: list[str] = []
    quarantined_reasons: dict[str, str] = {}
    license: str = "CC-BY-SA-4.0"


# ---------------------------------------------------------------------------
# T16 bounded decomposition (R13, optional)
# ---------------------------------------------------------------------------

DECOMPOSITION_VERSION = "decomposition-v1"
DECOMPOSITION_PLANNER_VERSION = "decomposition-planner-v1"
MAX_DECOMPOSITION_SUBQUESTIONS = 2

DecompositionStrategy = Literal["decomposition"]
