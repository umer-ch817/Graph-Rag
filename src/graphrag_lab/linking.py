"""Query entity linking for T10 (hybrid-qa, R07).

Links question spans to canonical graph entities without creating nodes.
Exact alias matches win first, then acronym aliases, then conservative
lexical near-matches for non-PERSON types. PERSON fuzzy matches stay
separate by design (mirrors T08): initials, single-token and common names
never auto-link on similarity alone.

Only ``MATCH`` queries run; linking never MERGEs or CREATEs graph nodes.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from graphrag_lab.models import EntityType
from graphrag_lab.resolution import (
    is_acronym,
    lexical_similarity,
    normalize_name,
)

LINKING_VERSION = "linking-v1"
MAX_QUESTION_MENTIONS = 5
MAX_QUESTION_CHARS = 2000
LEXICAL_THRESHOLD = 0.88

_QUOTED_RE = re.compile(r'"([^"]{1,200})"|\'([^\']{1,200})\'')
_CAP_SEQ_RE = re.compile(r"\b([A-Z][A-Za-z0-9\-']*(?:\s+[A-Z][A-Za-z0-9\-']*)*)\b")
_STOP_MENTIONS = frozenset(
    {
        "who",
        "what",
        "where",
        "when",
        "why",
        "how",
        "which",
        "whom",
        "whose",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
    }
)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class LinkingConfig(Record):
    """Tunable linking budgets (R07). At most five seeds per question."""

    max_mentions: int = Field(default=MAX_QUESTION_MENTIONS, ge=1, le=5, strict=True)
    lexical_threshold: float = Field(default=LEXICAL_THRESHOLD, ge=0.0, le=1.0)
    acronym_min_tokens: int = Field(default=2, ge=2, le=10, strict=True)
    linking_version: str = Field(default=LINKING_VERSION, min_length=1)


class QuestionMention(Record):
    text: str = Field(min_length=1)
    normalized: str = Field(min_length=1)
    start: int = Field(ge=0, strict=True)
    kind: Literal["quoted", "capitalized", "exact_substring"]


class LinkedEntity(Record):
    mention_text: str = Field(min_length=1)
    normalized_mention: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    entity_type: EntityType
    match_kind: Literal["exact_alias", "acronym_alias", "lexical"]
    score: float = Field(ge=0.0, le=1.0)


class LinkingResult(Record):
    question: str
    corpus_id: str
    mentions: list[QuestionMention]
    linked: list[LinkedEntity]
    diagnostics: dict[str, Any]


class LinkingValidationError(ValueError):
    code = "INVALID_INPUT"


def _validate_question(question: str, max_chars: int = MAX_QUESTION_CHARS) -> str:
    from graphrag_lab.llm import InvalidInputError

    if not isinstance(question, str):
        raise InvalidInputError("question must be a string")
    cleaned = question.strip()
    if not cleaned:
        raise InvalidInputError("question must be a nonempty string")
    if len(question) > max_chars:
        raise InvalidInputError(f"question length {len(question)} exceeds limit {max_chars}")
    return cleaned


def _validate_corpus_id(corpus_id: str) -> str:
    from graphrag_lab.llm import InvalidInputError

    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise InvalidInputError("corpus_id must be a nonempty string")
    if any(c.isspace() for c in corpus_id):
        raise InvalidInputError("corpus_id must contain no whitespace")
    return corpus_id.strip()


def extract_question_mentions(
    question: str,
    max_mentions: int = MAX_QUESTION_MENTIONS,
) -> list[QuestionMention]:
    """Extract up to ``max_mentions`` spans (quoted, then capitalized).

    Deterministic order: quoted spans first by appearance, then capitalized
    sequences by (start, -length, text). Interrogatives and single
    characters are skipped. Never invents entity types here; typing comes
    from the linked graph entity.
    """
    from graphrag_lab.llm import InvalidInputError

    if not isinstance(max_mentions, int) or isinstance(max_mentions, bool):
        raise InvalidInputError("max_mentions must be an integer")
    if not 1 <= max_mentions <= MAX_QUESTION_MENTIONS:
        raise InvalidInputError(f"max_mentions must be in 1..{MAX_QUESTION_MENTIONS}")
    cleaned = _validate_question(question)
    candidates: list[QuestionMention] = []
    seen_norm: set[str] = set()

    def _add(text: str, start: int, kind: Literal["quoted", "capitalized"]) -> None:
        stripped = text.strip().strip(",.;:!?()[]")
        if not stripped or len(stripped) < 2:
            return
        try:
            norm = normalize_name(stripped)
        except ValueError:
            return
        if norm in _STOP_MENTIONS or norm in seen_norm:
            return
        # Single-character norms are initials, never standalone mentions.
        if len(norm) < 2:
            return
        seen_norm.add(norm)
        candidates.append(
            QuestionMention(text=stripped, normalized=norm, start=int(start), kind=kind)
        )

    for match in _QUOTED_RE.finditer(cleaned):
        span = match.group(1) if match.group(1) is not None else match.group(2)
        if span:
            _add(span, match.start(), "quoted")
    for match in _CAP_SEQ_RE.finditer(cleaned):
        _add(match.group(1), match.start(), "capitalized")

    candidates.sort(key=lambda m: (m.start, -len(m.text), m.text))
    return candidates[:max_mentions]


def _token_bounded_contains(haystack_norm: str, needle_norm: str) -> bool:
    padded = f" {haystack_norm} "
    return f" {needle_norm} " in padded


def _question_ngrams(tokens: list[str], size: int) -> list[str]:
    if size <= 0 or size > len(tokens):
        return []
    return [" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]


def _parse_entity_type(value: Any) -> EntityType:
    from graphrag_lab.llm import ValidationFailedError

    if isinstance(value, EntityType):
        return value
    if isinstance(value, str):
        try:
            return EntityType(value)
        except ValueError as exc:
            raise ValidationFailedError(f"Unknown entity type {value!r}") from exc
    raise ValidationFailedError(f"Entity type must be a string, got {type(value).__name__}")


def _coerce_entities(rows: list[dict[str, Any]], corpus_id: str) -> list[dict[str, Any]]:
    from graphrag_lab.llm import ValidationFailedError

    coerced: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValidationFailedError(f"Entity row {index} must be a mapping")
        entity_id = row.get("id")
        canonical = row.get("canonical_name")
        normalized = row.get("normalized_name")
        raw_type = row.get("type")
        aliases = row.get("aliases", [])
        if not isinstance(entity_id, str) or not entity_id:
            raise ValidationFailedError(f"Entity row {index} has no id")
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValidationFailedError(f"Entity {entity_id!r} has no canonical name")
        if not isinstance(normalized, str) or not normalized.strip():
            raise ValidationFailedError(f"Entity {entity_id!r} has no normalized name")
        entity_type = _parse_entity_type(raw_type)
        alias_list = [str(a) for a in (aliases or []) if isinstance(a, str) and a.strip()]
        if not alias_list:
            alias_list = [canonical]
        alias_norms: list[str] = []
        for alias in alias_list:
            try:
                alias_norms.append(normalize_name(alias))
            except ValueError:
                continue
        try:
            canonical_norm = normalize_name(canonical)
        except ValueError:
            canonical_norm = normalized.strip()
        if normalized.strip() not in alias_norms and canonical_norm not in alias_norms:
            alias_norms.append(normalized.strip())
        alias_norms = sorted(set(alias_norms))
        coerced.append(
            {
                "id": entity_id,
                "canonical_name": canonical,
                "normalized_name": normalized.strip(),
                "type": entity_type,
                "alias_norms": alias_norms,
            }
        )
    # Deterministic scan order regardless of store ordering.
    coerced.sort(key=lambda e: e["id"])
    _ = corpus_id
    return coerced


def link_mentions_to_entities(
    question: str,
    entities: list[dict[str, Any]],
    config: LinkingConfig | None = None,
) -> tuple[list[QuestionMention], list[LinkedEntity], dict[str, Any]]:
    """Pure linking core: exact first, then acronym, then lexical (non-PERSON).

    Returns (mentions, linked, diagnostics). No I/O; testable offline.
    Caller supplies corpus-scoped entities already filtered by corpus.
    """
    cfg = config or LinkingConfig()
    cleaned = _validate_question(question)
    try:
        question_norm = normalize_name(cleaned)
    except ValueError:
        question_norm = cleaned.casefold()
    question_tokens = question_norm.split(" ")
    base_mentions = extract_question_mentions(cleaned, cfg.max_mentions)

    exact_hits: list[dict[str, Any]] = []
    for entity in entities:
        for alias_norm in entity["alias_norms"]:
            if _token_bounded_contains(question_norm, alias_norm):
                exact_hits.append(
                    {
                        "entity": entity,
                        "alias_norm": alias_norm,
                        "score": 1.0,
                        "match_kind": "exact_alias",
                    }
                )
    # Longest alias first so "the night cartographer" beats "cartographer".
    exact_hits.sort(key=lambda h: (-len(h["alias_norm"]), h["entity"]["id"], h["alias_norm"]))

    linked: list[LinkedEntity] = []
    used_entities: set[str] = set()
    alias_to_entities: dict[str, set[str]] = {}
    for hit in exact_hits:
        alias_to_entities.setdefault(hit["alias_norm"], set()).add(hit["entity"]["id"])
    ambiguous = sorted(alias for alias, ids in alias_to_entities.items() if len(ids) > 1)

    for hit in exact_hits:
        if len(linked) >= cfg.max_mentions:
            break
        entity = hit["entity"]
        if entity["id"] in used_entities:
            continue
        mention_text = next(
            (m.text for m in base_mentions if m.normalized == hit["alias_norm"]),
            entity["canonical_name"],
        )
        linked.append(
            LinkedEntity(
                mention_text=mention_text,
                normalized_mention=hit["alias_norm"],
                entity_id=entity["id"],
                canonical_name=entity["canonical_name"],
                entity_type=entity["type"],
                match_kind="exact_alias",
                score=1.0,
            )
        )
        used_entities.add(entity["id"])

    n_exact = len(linked)
    # Acronym pass for entities without an exact hit.
    if len(linked) < cfg.max_mentions:
        acronym_hits: list[dict[str, Any]] = []
        for entity in entities:
            if entity["id"] in used_entities:
                continue
            for alias_norm in entity["alias_norms"]:
                if " " in alias_norm:
                    continue  # long-form acronym target handled below
                if len(alias_norm) < 2 or not alias_norm.isalpha():
                    continue
                window = len(alias_norm)
                found = False
                for size in (window, window + 1):
                    for ngram in _question_ngrams(question_tokens, size):
                        if is_acronym(alias_norm, ngram, min_tokens=cfg.acronym_min_tokens):
                            found = True
                            break
                    if found:
                        break
                # Also handle short alias appearing as initials of a longer
                # question phrase is covered above; the reverse (question has
                # the short form, entity has the long form) is an exact hit
                # already, so only the short-alias case needs this pass.
                if found:
                    acronym_hits.append(
                        {
                            "entity": entity,
                            "alias_norm": alias_norm,
                            "score": 0.95,
                            "match_kind": "acronym_alias",
                        }
                    )
                    break
        acronym_hits.sort(key=lambda h: (h["entity"]["id"], h["alias_norm"]))
        for hit in acronym_hits:
            if len(linked) >= cfg.max_mentions:
                break
            entity = hit["entity"]
            if entity["id"] in used_entities:
                continue
            linked.append(
                LinkedEntity(
                    mention_text=entity["canonical_name"],
                    normalized_mention=hit["alias_norm"],
                    entity_id=entity["id"],
                    canonical_name=entity["canonical_name"],
                    entity_type=entity["type"],
                    match_kind="acronym_alias",
                    score=0.95,
                )
            )
            used_entities.add(entity["id"])
    n_acronym = len(linked) - n_exact

    # Lexical pass: non-PERSON only, threshold plus token overlap.
    n_lexical = 0
    if len(linked) < cfg.max_mentions:
        lexical_hits: list[dict[str, Any]] = []
        for entity in entities:
            if entity["id"] in used_entities:
                continue
            if entity["type"] == EntityType.PERSON:
                continue  # conservative: PERSON needs exact/acronym
            best_score = 0.0
            best_alias = ""
            for alias_norm in entity["alias_norms"]:
                alias_tokens = alias_norm.split(" ")
                for size in {
                    max(1, len(alias_tokens) - 1),
                    len(alias_tokens),
                    len(alias_tokens) + 1,
                }:
                    for ngram in _question_ngrams(question_tokens, size):
                        score = lexical_similarity(alias_norm, ngram)
                        if score > best_score:
                            ngram_tokens = set(ngram.split(" "))
                            alias_set = set(alias_tokens)
                            if alias_set.intersection(ngram_tokens):
                                best_score = score
                                best_alias = alias_norm
            if best_score >= cfg.lexical_threshold:
                lexical_hits.append(
                    {
                        "entity": entity,
                        "alias_norm": best_alias,
                        "score": round(float(best_score), 4),
                        "match_kind": "lexical",
                    }
                )
        lexical_hits.sort(key=lambda h: (-h["score"], h["entity"]["id"], h["alias_norm"]))
        for hit in lexical_hits:
            if len(linked) >= cfg.max_mentions:
                break
            entity = hit["entity"]
            if entity["id"] in used_entities:
                continue
            linked.append(
                LinkedEntity(
                    mention_text=entity["canonical_name"],
                    normalized_mention=hit["alias_norm"],
                    entity_id=entity["id"],
                    canonical_name=entity["canonical_name"],
                    entity_type=entity["type"],
                    match_kind="lexical",
                    score=float(hit["score"]),
                )
            )
            used_entities.add(entity["id"])
            n_lexical += 1

    linked.sort(key=lambda e: e.entity_id)
    # Rank priority for diagnostics: exact, acronym, lexical already capped
    # deterministically above; final order by entity id keeps output stable.
    mentions: list[QuestionMention] = list(base_mentions)
    # Surface exact-substring mentions not covered by the regex extractor so
    # the audit shows every seed's question evidence.
    known_norms = {m.normalized for m in mentions}
    for item in linked:
        if item.normalized_mention not in known_norms:
            try:
                start = question_norm.index(item.normalized_mention)
            except ValueError:
                start = 0
            mentions.append(
                QuestionMention(
                    text=item.mention_text,
                    normalized=item.normalized_mention,
                    start=int(start),
                    kind="exact_substring",
                )
            )
            known_norms.add(item.normalized_mention)
    mentions.sort(key=lambda m: (m.start, -len(m.text), m.text))
    mentions = mentions[: cfg.max_mentions]

    diagnostics: dict[str, Any] = {
        "linking_version": cfg.linking_version,
        "n_mentions": len(mentions),
        "n_entities_scanned": len(entities),
        "n_linked": len(linked),
        "n_exact": n_exact,
        "n_acronym": n_acronym,
        "n_lexical": n_lexical,
        "ambiguous_aliases": ambiguous,
        "creates_nodes": False,
    }
    return mentions, linked, diagnostics


def link_question_entities(
    question: str,
    driver: Any,
    database: str,
    corpus_id: str,
    config: LinkingConfig | None = None,
) -> LinkingResult:
    """Link a question against the corpus graph; never creates nodes.

    Raises CONFIG_MISMATCH on corpus drift; returns an empty link set (not
    an exception) when the graph snapshot is absent so the caller can fall
    back to vector retrieval with a visible flag.
    """
    from graphrag_lab.llm import InvalidInputError

    cfg = config or LinkingConfig()
    cleaned = _validate_question(question)
    corpus = _validate_corpus_id(corpus_id)
    if driver is None:
        raise InvalidInputError("driver must be supplied")
    if not isinstance(database, str) or not database.strip():
        raise InvalidInputError("database must be a nonempty string")

    # Late imports avoid a hard cycle: neo4j_store never imports linking.
    from graphrag_lab.neo4j_store import (
        ConfigMismatchError,
        list_entities_for_corpus,
        read_graph_meta,
    )

    meta = read_graph_meta(driver, database)
    if meta is None:
        mentions = extract_question_mentions(cleaned, cfg.max_mentions)
        return LinkingResult(
            question=cleaned,
            corpus_id=corpus,
            mentions=mentions,
            linked=[],
            diagnostics={
                "linking_version": cfg.linking_version,
                "n_mentions": len(mentions),
                "n_entities_scanned": 0,
                "n_linked": 0,
                "n_exact": 0,
                "n_acronym": 0,
                "n_lexical": 0,
                "ambiguous_aliases": [],
                "creates_nodes": False,
                "graph_status": "graph_not_ready",
                "graph_fingerprint": None,
            },
        )
    if meta.get("corpus_id") != corpus:
        raise ConfigMismatchError(
            f"Linking corpus {corpus!r} != graph corpus {meta.get('corpus_id')!r}; "
            "refusing mixed corpora"
        )
    rows = list_entities_for_corpus(driver, database, corpus)
    entities = _coerce_entities(rows, corpus)
    mentions, linked, diagnostics = link_mentions_to_entities(cleaned, entities, cfg)
    diagnostics = dict(diagnostics)
    diagnostics["graph_status"] = "ready"
    diagnostics["graph_fingerprint"] = meta.get("fingerprint")
    diagnostics["graph_corpus_id"] = meta.get("corpus_id")
    return LinkingResult(
        question=cleaned,
        corpus_id=corpus,
        mentions=mentions,
        linked=linked,
        diagnostics=diagnostics,
    )


__all__ = [
    "LINKING_VERSION",
    "LinkingConfig",
    "LinkingResult",
    "LinkedEntity",
    "QuestionMention",
    "extract_question_mentions",
    "link_mentions_to_entities",
    "link_question_entities",
]
