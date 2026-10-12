"""Compare and evidence view model for T12c (dashboard, R07/R08).

Pure logic behind Ask & Compare and Evidence: arm metadata, tuning-only
demo examples (locked evaluation labels are never loaded here), evidence
source rows with origin labels, exact citation resolution, the accessible
relation table, persisted per-arm runs, and local deep links. No Streamlit,
Neo4j or LLM imports here so ``tests/e2e/test_compare_flow.py`` runs fully
offline; ``app/pages/compare.py`` and ``app/pages/evidence.py`` own
presentation and live service wiring.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ARMS: tuple[str, ...] = ("vector", "vector_rerank", "hybrid_rerank", "hybrid_rerank_triples")

ARM_LABELS: dict[str, str] = {
    "vector": "Vector",
    "vector_rerank": "Vector + rerank",
    "hybrid_rerank": "Hybrid + rerank",
    "hybrid_rerank_triples": "Hybrid + rerank + triples",
}

ARM_BLURBS: dict[str, str] = {
    "vector": "Top-30 cosine candidates, no reranker. Baseline.",
    "vector_rerank": "Vector candidates reordered by the cross-encoder. Isolates reranking.",
    "hybrid_rerank": "Vector plus bounded graph expansion, same reranker. Isolates graph recall.",
    "hybrid_rerank_triples": "Hybrid reranked plus up to 12 evidence-only triples. Full package.",
}

DEFAULT_PAIR: tuple[str, str] = ("vector", "hybrid_rerank_triples")

Origin = Literal["Vector", "Graph", "Both"]
PanelStatus = Literal["supported", "insufficient_evidence", "failed", "pending", "running"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class EvidenceSource(Record):
    chunk_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    paragraph_id: str = Field(min_length=1)
    sentence_ids: list[int] = Field(default_factory=list)
    sentence_texts: list[str] = Field(default_factory=list)
    sentence_range: str = Field(min_length=1)
    origin: Origin = "Vector"
    in_final_context: bool = False
    cited: bool = False
    rank: int | None = None


class RelationRow(Record):
    relation_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    relation_type: str = Field(min_length=1)
    target_name: str = Field(min_length=1)
    evidence_chunk_id: str = Field(min_length=1)
    evidence_title: str = Field(min_length=1)
    evidence_available: bool = True


class ArmPanel(Record):
    strategy: str = Field(min_length=1)
    strategy_label: str = Field(min_length=1)
    status: PanelStatus = "pending"
    answer: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float | None = None
    n_candidates: int = Field(default=0, ge=0, strict=True)
    n_packed: int = Field(default=0, ge=0, strict=True)
    fallback: bool = False
    fallback_reason: str | None = None
    degraded: bool = False
    cap_hits: list[str] = Field(default_factory=list)
    run_id: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None


def arm_label(strategy: str) -> str:
    if strategy not in ARM_LABELS:
        raise ValueError(f"Unknown experiment arm {strategy!r}; expected one of {list(ARM_LABELS)}")
    return ARM_LABELS[strategy]


def validate_arm_pair(left: str, right: str | None) -> tuple[str, str | None]:
    arm_label(left)
    if right is not None:
        arm_label(right)
        if right == left:
            raise ValueError("Comparison arms must differ; pick two distinct strategies")
    return left, right


def question_id_for(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    digest = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()[:12]
    return f"q_{digest}"


def tuning_examples(dataset_dir: str | Path, *, limit: int = 30) -> list[dict[str, Any]]:
    """Demo picker rows: tuning questions only, never answers or labels."""
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
        raise ValueError("limit must be an integer in 1..200")
    path = Path(dataset_dir) / "questions.jsonl"
    if not path.is_file():
        raise ValueError(
            f"No prepared dataset at {dataset_dir} (missing questions.jsonl). "
            "Prepare a dataset first."
        )
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"questions.jsonl is not valid JSONL: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError("questions.jsonl rows must be objects")
        if record.get("split") != "tuning":
            continue
        if not record.get("id") or not record.get("question"):
            raise ValueError("questions.jsonl tuning rows need id and question")
        rows.append(
            {
                "id": str(record["id"]),
                "question": str(record["question"]),
                "type": str(record.get("type", "")),
            }
        )
        if len(rows) >= limit:
            break
    return sorted(rows, key=lambda row: row["id"])


def _sentence_range(sentence_ids: list[int]) -> str:
    if not sentence_ids:
        return "s?"
    ordered = sorted(set(sentence_ids))
    if len(ordered) == 1:
        return f"s{ordered[0]}"
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"s{ordered[0]}–s{ordered[-1]}"
    return ", ".join(f"s{i}" for i in ordered)


def build_sources(
    chunks: list[dict[str, Any]],
    packed_ids: list[str],
    citations: list[dict[str, Any]],
) -> list[EvidenceSource]:
    """One row per retrieved chunk: origin, final-context flag, cited flag."""
    if len(set(packed_ids)) != len(packed_ids):
        raise ValueError("packed_ids must not repeat")
    packed = set(packed_ids)
    cited_ids: set[str] = set()
    for citation in citations:
        cid = citation.get("chunk_id") if isinstance(citation, dict) else None
        if isinstance(cid, str) and cid:
            cited_ids.add(cid)
    seen: set[str] = set()
    rows: list[EvidenceSource] = []
    for position, chunk in enumerate(chunks):
        cid = str(chunk.get("chunk_id", chunk.get("id", "")))
        if not cid:
            raise ValueError(f"Retrieved chunk {position} has no chunk id")
        if cid in seen:
            raise ValueError(f"Duplicate retrieved chunk id {cid!r}")
        seen.add(cid)
        sentence_ids = [int(v) for v in list(chunk.get("sentence_ids", []) or [])]
        sentence_texts = [str(v) for v in list(chunk.get("sentence_texts", []) or [])]
        if sentence_texts and len(sentence_texts) != len(sentence_ids):
            raise ValueError(f"Chunk {cid!r} sentence ids/texts mismatch")
        raw_origin = str(chunk.get("origin", "vector")).lower()
        origin: Origin = (
            "Both" if raw_origin == "both" else "Graph" if raw_origin == "graph" else "Vector"
        )
        rank = chunk.get("rank", position + 1)
        try:
            rank_value: int | None = int(rank) if rank is not None else None
        except (TypeError, ValueError):
            rank_value = None
        rows.append(
            EvidenceSource(
                chunk_id=cid,
                title=str(chunk.get("title", cid)),
                paragraph_id=str(chunk.get("paragraph_id", "")),
                sentence_ids=sentence_ids,
                sentence_texts=sentence_texts,
                sentence_range=_sentence_range(sentence_ids),
                origin=origin,
                in_final_context=cid in packed,
                cited=cid in cited_ids,
                rank=rank_value,
            )
        )
    order = {cid: index for index, cid in enumerate(packed_ids)}
    rows.sort(
        key=lambda row: (
            0 if row.chunk_id in order else 1,
            order.get(row.chunk_id, 0),
            row.chunk_id,
        )
    )
    return rows


def resolve_citation(citation: dict[str, Any], sources: list[EvidenceSource]) -> dict[str, Any]:
    """Resolve one citation to its source plus exact sentences (with context).

    Raises LookupError with an understandable reason when the citation points
    outside the retrieved evidence instead of opening the wrong text.
    """
    cid = citation.get("chunk_id") if isinstance(citation, dict) else None
    sentence_ids = citation.get("sentence_ids", []) if isinstance(citation, dict) else []
    by_id = {source.chunk_id: source for source in sources}
    if not isinstance(cid, str) or cid not in by_id:
        raise LookupError(
            f"Citation references unknown chunk {cid!r}; it is not in retrieved evidence"
        )
    source = by_id[cid]
    if not isinstance(sentence_ids, list) or not sentence_ids:
        raise LookupError(f"Citation for {cid!r} lists no sentences")
    wanted = [int(v) for v in sentence_ids]
    available = set(source.sentence_ids)
    unknown = [v for v in wanted if v not in available]
    if unknown:
        raise LookupError(
            f"Citation for {cid!r} references sentences {unknown} outside the retrieved context "
            f"({source.sentence_range})"
        )
    sentences = [
        {"sentence_id": sid, "text": text, "cited": sid in set(wanted)}
        for sid, text in zip(
            source.sentence_ids, source.sentence_texts or ["" for _ in source.sentence_ids]
        )
    ]
    return {"source": source, "sentences": sentences}


def relation_rows(
    traversed: list[dict[str, Any]], chunks_by_id: dict[str, dict[str, Any]]
) -> list[RelationRow]:
    """Accessible relation table; missing evidence is flagged, never dropped."""
    rows: list[RelationRow] = []
    for position, relation in enumerate(traversed):
        rid = str(relation.get("id", relation.get("relation_id", "")))
        if not rid:
            raise ValueError(f"Traversed relation {position} has no id")
        evidence = list(
            relation.get("evidence_chunk_ids", relation.get("evidence", []) or []) or []
        )
        primary = str(evidence[0]) if evidence else ""
        chunk = chunks_by_id.get(primary, {})
        rows.append(
            RelationRow(
                relation_id=rid,
                source_name=str(relation.get("source", relation.get("source_entity_id", "?"))),
                relation_type=str(relation.get("type", "?")),
                target_name=str(relation.get("target", relation.get("target_entity_id", "?"))),
                evidence_chunk_id=primary or "(none)",
                evidence_title=str(chunk.get("title", "(evidence not in context)"))
                if primary
                else "(none)",
                evidence_available=bool(primary) and primary in chunks_by_id,
            )
        )
    rows.sort(key=lambda row: row.relation_id)
    return rows


def fallback_text(*, fallback: bool, reason: str | None, cap_hits: list[str]) -> str:
    """Visible graph-failure wording plus explicit truncation summaries."""
    parts: list[str] = []
    if fallback:
        parts.append(
            f"No linked entities; vector evidence shown ({reason or 'no reason recorded'})"
        )
    elif reason:
        parts.append(str(reason))
    if cap_hits:
        parts.append("Traversal caps hit: " + ", ".join(sorted(set(cap_hits))))
    return " ".join(parts)


def describe_arm(strategy: str) -> str:
    return f"{arm_label(strategy)} — {ARM_BLURBS[strategy]}"


def assert_matched_budgets(configs: list[dict[str, Any]]) -> dict[str, Any]:
    """C4 ablation guard: all arms share final chunk and token budgets."""
    if not configs:
        raise ValueError("configs must be nonempty")
    chunks = {int(cfg.get("max_chunks", -1)) for cfg in configs}
    tokens = {int(cfg.get("max_context_tokens", -1)) for cfg in configs}
    if len(chunks) != 1 or len(tokens) != 1:
        raise ValueError(
            "Ablation arms must share final budgets; "
            f"got max_chunks={sorted(chunks)} max_context_tokens={sorted(tokens)}"
        )
    return {"max_chunks": next(iter(chunks)), "max_context_tokens": next(iter(tokens))}


def persist_compare(
    run_store: Any,
    *,
    question: str,
    question_id: str,
    snapshot_id: str | None,
    panels: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Persist one run per arm plus its prediction. Never overwrites history."""
    from graphrag_lab.evaluation import hash_question_set

    if not question.strip():
        raise ValueError("question must be nonempty")
    if not panels:
        raise ValueError("panels must be nonempty")
    base = dict(config or {})
    run_ids: dict[str, str] = {}
    for panel in panels:
        strategy = str(panel.get("strategy", ""))
        arm_label(strategy)
        status = str(panel.get("status", "failed"))
        if status not in ("supported", "insufficient_evidence", "failed"):
            raise ValueError(f"Panel {strategy!r} has unknown status {status!r}")
        run = run_store.create_run(
            strategy=strategy,
            question_set_hash=hash_question_set([question_id]),
            config={**base, "strategy": strategy, "question": question},
            snapshot_id=snapshot_id,
            split="interactive",
        )
        retrieval = dict(panel.get("retrieval", {}) or {})
        retrieval.setdefault("strategy", strategy)
        run_store.upsert_prediction(
            run.id,
            question_id,
            answer=str(panel.get("answer", "")),
            status=status,  # type: ignore[arg-type]
            citations=list(panel.get("citations", []) or []),
            retrieval=retrieval,
            latency_ms=panel.get("latency_ms"),
            usage=panel.get("usage"),
            failure_code=panel.get("failure_code"),
        )
        run_store.update_run_state(run.id, "completed" if status != "failed" else "failed")
        run_ids[strategy] = run.id
    return run_ids


def load_panels(run_store: Any, run_ids: list[str]) -> list[ArmPanel]:
    """Read saved runs for display. Never reruns a question implicitly."""
    if not run_ids:
        raise ValueError("run_ids must be nonempty")
    panels: list[ArmPanel] = []
    for run_id in run_ids:
        try:
            run = run_store.get_run(run_id)
        except Exception as exc:
            raise ValueError(f"Saved run {run_id!r} cannot be opened: {exc}") from exc
        predictions = run_store.list_predictions(run_id)
        if not predictions:
            raise ValueError(f"Saved run {run_id!r} has no predictions")
        pred = predictions[0]
        retrieval = dict(pred.retrieval_json or {})
        panels.append(
            ArmPanel(
                strategy=run.strategy,
                strategy_label=arm_label(run.strategy),
                status=pred.status,  # type: ignore[arg-type]
                answer=pred.answer,
                citations=list(pred.citations_json or []),
                latency_ms=pred.latency_ms,
                n_candidates=int(
                    retrieval.get("num_candidates", retrieval.get("n_candidates", 0)) or 0
                ),
                n_packed=int(retrieval.get("num_packed", retrieval.get("n_packed", 0)) or 0),
                fallback=bool(retrieval.get("fallback", False)),
                fallback_reason=retrieval.get("fallback_reason"),
                degraded=bool(retrieval.get("degraded", False)),
                cap_hits=list(retrieval.get("cap_hits", []) or []),
                run_id=run.id,
                failure_code=pred.failure_code,
                failure_message=retrieval.get("failure_message"),
            )
        )
    return panels


def load_evidence(run_store: Any, run_id: str) -> dict[str, Any]:
    """Full saved evidence for one run: chunks, packed ids, triples path data.

    Powers the Evidence page from durable records only; opening a saved
    result never triggers a new retrieval or LLM request.
    """
    try:
        run = run_store.get_run(run_id)
    except Exception as exc:
        raise ValueError(f"Saved run {run_id!r} cannot be opened: {exc}") from exc
    predictions = run_store.list_predictions(run_id)
    if not predictions:
        raise ValueError(f"Saved run {run_id!r} has no predictions")
    pred = predictions[0]
    retrieval = dict(pred.retrieval_json or {})
    return {
        "run_id": run.id,
        "strategy": run.strategy,
        "question_id": pred.question_id,
        "answer": pred.answer,
        "status": pred.status,
        "citations": list(pred.citations_json or []),
        "chunks": list(retrieval.get("chunks", []) or []),
        "packed_ids": list(retrieval.get("packed_ids", []) or []),
        "traversed": list(retrieval.get("traversed", []) or []),
        "fallback": bool(retrieval.get("fallback", False)),
        "fallback_reason": retrieval.get("fallback_reason"),
        "degraded": bool(retrieval.get("degraded", False)),
        "cap_hits": list(retrieval.get("cap_hits", []) or []),
        "latency_ms": pred.latency_ms,
        "failure_code": pred.failure_code,
        "failure_message": retrieval.get("failure_message"),
    }


def encode_deep_link(run_ids: list[str], source_id: str | None = None) -> str:
    parts = [f"run={rid}" for rid in run_ids]
    if source_id:
        parts.append(f"src={source_id}")
    return "&".join(parts)


def parse_deep_link(query: str) -> dict[str, Any]:
    """Parse a local run/source deep link (never a shareable public URL)."""
    run_ids: list[str] = []
    source_id: str | None = None
    for chunk in str(query or "").replace("?", "").split("&"):
        if "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        if key.strip() == "run" and value.strip():
            run_ids.append(value.strip())
        elif key.strip() == "src" and value.strip():
            source_id = value.strip()
    if not run_ids:
        raise ValueError("Deep link has no run ids")
    return {"run_ids": run_ids, "source_id": source_id}


__all__ = [
    "ARM_BLURBS",
    "ARM_LABELS",
    "ARMS",
    "DEFAULT_PAIR",
    "ArmPanel",
    "EvidenceSource",
    "Origin",
    "PanelStatus",
    "RelationRow",
    "arm_label",
    "assert_matched_budgets",
    "build_sources",
    "describe_arm",
    "encode_deep_link",
    "fallback_text",
    "load_evidence",
    "load_panels",
    "parse_deep_link",
    "persist_compare",
    "question_id_for",
    "relation_rows",
    "resolve_citation",
    "tuning_examples",
    "validate_arm_pair",
]
