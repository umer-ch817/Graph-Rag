"""Deterministic HotpotQA sampling and paragraph artifacts.

Scope is T02 only: select 150 ids (30 tuning / 120 evaluation), retain every
context paragraph including distractors, separate gold labels from retriever
input, and persist a reproducibility manifest. No chunking or retrieval here.

Upstream-invalid rows (e.g. a supporting-fact sentence index outside its
paragraph, as shipped in the HotpotQA dev release) are quarantined with ids
and reasons recorded in the manifest; the sample draws from valid rows only.
The raw source bytes are never modified, so the exclusion stays auditable.
"""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from graphrag_lab.models import HOTPOT_URL, DatasetManifest, HotpotQuestion, SampleConfig, Split

ARTIFACT_NAMES = (
    "questions.jsonl",
    "labels.jsonl",
    "contexts.jsonl",
    "paragraphs.jsonl",
)


def canonical_bytes(value: object) -> bytes:
    """Canonical JSON encoding used for every hash in this module."""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (text + "\n").encode("utf-8")


def sha256_hex(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def allocate_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder split with deterministic type-name tie breaks."""
    available = sum(counts.values())
    if available == 0 or total > available:
        raise ValueError("Requested sample exceeds available source questions")
    quotas = {kind: counts[kind] * total // available for kind in counts}
    remainder = total - sum(quotas.values())
    by_remainder = sorted(counts, key=lambda k: (-((counts[k] * total) % available), k))
    for kind in by_remainder[:remainder]:
        quotas[kind] += 1
    return quotas


def _rank(seed: int, purpose: str, question_id: str) -> str:
    return sha256_hex([seed, purpose, question_id])


def select_questions(
    rows: list[HotpotQuestion],
    config: SampleConfig,
) -> list[tuple[HotpotQuestion, Split]]:
    """Seeded stratified selection, independent of input order.

    Ranking uses SHA-256 of (seed, purpose, question id) so results do not
    depend on source order or on any platform RNG version.
    """
    if len({row.id for row in rows}) != len(rows):
        raise ValueError("Duplicate question ids in source")
    counts = Counter(row.type for row in rows)
    want = allocate_quotas(dict(counts), config.sample_size)
    want_tuning = allocate_quotas(want, config.tuning_size)
    selected: list[tuple[HotpotQuestion, Split]] = []
    for kind in sorted(counts):
        pool = sorted(
            (row for row in rows if row.type == kind),
            key=lambda row: (_rank(config.seed, "sample", row.id), row.id),
        )[: want[kind]]
        pool.sort(key=lambda row: (_rank(config.seed, "split", row.id), row.id))
        for index, row in enumerate(pool):
            split: Split = "tuning" if index < want_tuning[kind] else "evaluation"
            selected.append((row, split))
    selected.sort(key=lambda pair: pair[0].id)
    return selected


def normalize_title(title: str) -> str:
    return " ".join(unicodedata.normalize("NFC", title).split())


def build_artifacts(selected: list[tuple[HotpotQuestion, Split]]) -> dict[str, bytes]:
    """Pool every context paragraph; resolve each gold fact to its variant."""
    questions: list[dict] = []
    labels: list[dict] = []
    contexts: list[dict] = []
    paragraphs: dict[str, dict] = {}
    for row, split in selected:
        questions.append({"id": row.id, "question": row.question, "type": row.type, "split": split})
        paragraph_for_title: dict[str, str] = {}
        for position, (title, sentences) in enumerate(row.context):
            paragraph_id = "p_" + sha256_hex([normalize_title(title), list(sentences)])
            entry = paragraphs.get(paragraph_id)
            if entry is None:
                entry = {
                    "id": paragraph_id,
                    "title": title,
                    "title_aliases": [],
                    "sentences": list(sentences),
                }
                paragraphs[paragraph_id] = entry
            aliases = entry["title_aliases"]
            assert isinstance(aliases, list)
            if title not in aliases:
                aliases.append(title)
            paragraph_for_title[title] = paragraph_id
            contexts.append(
                {
                    "question_id": row.id,
                    "position": position,
                    "title": title,
                    "paragraph_id": paragraph_id,
                }
            )
        labels.append(
            {
                "question_id": row.id,
                "answer": row.answer,
                "supporting_facts": [
                    {
                        "title": title,
                        "sentence_id": index,
                        "paragraph_id": paragraph_for_title[title],
                    }
                    for title, index in row.supporting_facts
                ],
            }
        )
    for entry in paragraphs.values():
        aliases = entry["title_aliases"]
        assert isinstance(aliases, list)
        aliases.sort()
        entry["title"] = aliases[0]
    ordered = [paragraphs[key] for key in sorted(paragraphs)]
    grouped = {
        "questions.jsonl": questions,
        "labels.jsonl": labels,
        "contexts.jsonl": contexts,
        "paragraphs.jsonl": ordered,
    }
    return {
        name: b"".join(canonical_bytes(record) for record in rows) for name, rows in grouped.items()
    }


def ensure_source(path: str | Path, url: str = HOTPOT_URL) -> Path:
    """Return a cached source file, downloading it once when missing."""
    dest = Path(path)
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".download")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as handle:  # noqa: S310
        handle.write(response.read())
    tmp.rename(dest)
    return dest


def _quarantine_reason(exc: Exception) -> str:
    """One-line sanitized reason for a quarantined source row (no row text)."""
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    return text[:200]


def prepare_dataset(
    source: str | Path,
    output: str | Path,
    config: SampleConfig,
    *,
    source_url: str = HOTPOT_URL,
) -> DatasetManifest:
    """Write (or verify) one immutable artifact directory."""
    raw = Path(source).read_bytes()
    decoded = json.loads(raw)
    if not isinstance(decoded, list):
        raise ValueError("Source must be a JSON list of HotpotQA questions")
    rows: list[HotpotQuestion] = []
    quarantined: dict[str, str] = {}
    for position, item in enumerate(decoded):
        try:
            rows.append(HotpotQuestion.model_validate(item))
        except Exception as exc:  # noqa: BLE001 - record and quarantine, never silently drop
            raw_id = item.get("_id") if isinstance(item, dict) else None
            key = str(raw_id) if raw_id else f"<row {position}>"
            quarantined[key] = _quarantine_reason(exc)
    selected = select_questions(rows, config)
    source_hash = hashlib.sha256(raw).hexdigest()
    payloads = build_artifacts(selected)
    split_ids = {
        split: [row.id for row, assignment in selected if assignment == split]
        for split in ("tuning", "evaluation")
    }
    manifest = DatasetManifest(
        corpus_id="hp_" + sha256_hex([source_hash, config.model_dump(), split_ids, "hotpot-v1"]),
        source_url=source_url,
        source_sha256=source_hash,
        source_bytes=len(raw),
        source_question_count=len(decoded),
        source_type_counts=dict(Counter(row.type for row in rows)),
        sample=config,
        selected_question_ids=[row.id for row, _ in selected],
        split_ids=split_ids,
        split_type_counts={
            split: dict(Counter(row.type for row, a in selected if a == split))
            for split in split_ids
        },
        paragraph_count=payloads["paragraphs.jsonl"].count(b"\n"),
        context_count=payloads["contexts.jsonl"].count(b"\n"),
        artifact_sha256={name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
        quarantined_question_ids=sorted(quarantined),
        quarantined_reasons=dict(sorted(quarantined.items())),
    )
    payloads["manifest.json"] = canonical_bytes(manifest.model_dump())
    destination = Path(output)
    if destination.exists():
        if not destination.is_dir() or any(
            not (destination / name).is_file() or (destination / name).read_bytes() != data
            for name, data in payloads.items()
        ):
            raise ValueError(
                "Existing output differs or is incomplete; choose a new output directory"
            )
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=str(destination.parent)) as staging_parent:
        staged = Path(staging_parent) / "dataset"
        staged.mkdir()
        for name, data in payloads.items():
            target = staged / name
            with open(target, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        staged.rename(destination)
    return manifest
