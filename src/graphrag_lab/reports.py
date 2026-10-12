"""Artifact export for T06 (evaluation).

Writes immutable baseline artifacts -- per-question ``results.jsonl``,
``aggregate.json``, ``manifest.json`` and a Markdown ``summary.md`` -- with
atomic renames and content hashes. Exports record the frozen configuration
and explicit limitations; unknown costs stay null and are labeled unknown,
never zero. No secrets are persisted.

T14 adds paired comparison exports (R09/R12): ``comparison.json``,
``regression.csv`` and ``report.md`` joining two frozen runs. Paired deltas
are produced only for compatible runs (see ``evaluation.compare_runs``);
incompatible runs are refused with a reason and never yield an unlabeled
delta. Per-question rows preserve the fixed denominator including failures.
Indexing cost and query cost are stored and rendered separately; unknown
money stays unknown, never zero.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from graphrag_lab.evaluation import (
    LabelView,
    QuestionPrediction,
    RunEvaluation,
    ScoredQuestion,
    scored_to_result_rows,
    write_json_atomic,
    write_jsonl_atomic,
)

REPORT_VERSION = "report-v1"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_aggregate_payload(
    evaluation: RunEvaluation,
    *,
    corpus_id: str | None = None,
    limitations: list[str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Machine-readable aggregate joining config, metrics and limitations."""
    payload = evaluation.model_dump()
    payload["report_version"] = REPORT_VERSION
    payload["corpus_id"] = corpus_id or evaluation.config.corpus_id
    payload["limitations"] = list(limitations or [])
    if notes is not None:
        payload["notes"] = notes
    return payload


def render_markdown_summary(
    evaluation: RunEvaluation,
    scored: list[ScoredQuestion] | None = None,
    *,
    limitations: list[str] | None = None,
    type_counts: dict[str, int] | None = None,
) -> str:
    """Render the human-readable baseline summary (no secrets, no fabrication)."""
    cfg = evaluation.config
    agg = evaluation.aggregate
    only = evaluation.success_only
    limits = limitations or [
        "Vector baseline only; no graph traversal or reranking in this run.",
        "Citation validity means the citation resolves to packed evidence, not entailment.",
        "Costs cover LLM usage when reported; local compute time is measured, not priced.",
    ]
    lines = [
        f"# Baseline report — {cfg.run_id}",
        "",
        f"Strategy: `{cfg.strategy}` · corpus `{cfg.corpus_id}` · "
        f"{agg.n_total} questions ({cfg.split or 'split unspecified'})",
        "",
        "## Configuration",
        "",
        f"- Embedding model: `{cfg.embedding_model}`",
        f"- Answer model: `{cfg.answer_model}`",
        f"- Prompt `{cfg.prompt_version}` · answers `{cfg.answer_version}`",
        f"- Estimator `{cfg.token_estimator}` · scoring `{evaluation.scoring_version}`",
        f"- Candidates: {cfg.num_candidates} · packed: top {cfg.max_chunks} / "
        f"{cfg.max_context_tokens} tokens",
        f"- Question set hash: `{cfg.question_set_hash}`",
        "",
        "## Primary metrics (fixed denominator, 0–100 scale)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Questions (fixed denominator) | {agg.n_total} |",
        f"| Answer EM | {agg.exact_match_mean:.2f} |",
        f"| Answer F1 | {agg.f1_mean:.2f} |",
        f"| Paragraph recall (candidates) | {agg.paragraph_recall_candidate:.2f} |",
        f"| All-support coverage (candidates) | {agg.all_support_candidate:.2f} |",
        f"| Sentence recall (candidates) | {agg.sentence_recall_candidate:.2f} |",
        f"| Paragraph recall (packed) | {agg.paragraph_recall_packed:.2f} |",
        f"| All-support coverage (packed) | {agg.all_support_packed:.2f} |",
        f"| Sentence recall (packed) | {agg.sentence_recall_packed:.2f} |",
        "| Latency p50 (ms) | "
        + (f"{agg.latency_p50_ms:.1f}" if agg.latency_p50_ms is not None else "unknown")
        + " |",
        "| Latency p95 (ms) | "
        + (f"{agg.latency_p95_ms:.1f}" if agg.latency_p95_ms is not None else "unknown")
        + " |",
        f"| Failed / abstained / supported | {agg.n_failed} / {agg.n_abstained} / "
        f"{agg.n_supported} |",
        f"| Query tokens in/out | {agg.total_input_tokens} / {agg.total_output_tokens} |",
        "| Estimated query cost (USD) | "
        + (
            f"{agg.total_estimated_cost_usd:.6f}"
            if agg.total_estimated_cost_usd is not None
            else "unknown"
        )
        + " |",
        "",
        "## Successful-run-only diagnostics (failures excluded)",
        "",
        f"- n_success: {only.n_success}",
        "- EM: " + (f"{only.exact_match_mean:.2f}" if only.exact_match_mean is not None else "n/a"),
        " · F1: " + (f"{only.f1_mean:.2f}" if only.f1_mean is not None else "n/a"),
        " · packed all-support: "
        + (f"{only.all_support_packed:.2f}" if only.all_support_packed is not None else "n/a"),
        "",
    ]
    if type_counts:
        lines.extend(
            [
                "## Question mix",
                "",
                "| Type | Count |",
                "|---|---|",
            ]
        )
        for name in sorted(type_counts):
            lines.append(f"| {name} | {type_counts[name]} |")
        lines.append("")
    if scored is not None:
        lines.extend(
            [
                "## Per-question results",
                "",
                "| question_id | status | EM | F1 | packed all-support | packed ids |",
                "|---|---|---|---|---|---|",
            ]
        )
        # The table carries scores only so the summary never drifts from
        # results.jsonl (packed ids live in results.jsonl per question).
        for item in sorted(scored, key=lambda entry: entry.question_id):
            lines.append(
                f"| {item.question_id} | {item.status} | {item.exact_match:.0f} | "
                f"{item.f1:.2f} | {item.packed.all_support:.0f} | k={item.packed.k} |"
            )
        lines.append("")
    lines.extend(["## Limitations", ""])
    for item in limits:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"Report version: `{REPORT_VERSION}` · created `{evaluation.created_at}`.")
    lines.append("")
    return "\n".join(lines)


def export_baseline(
    output_dir: str | Path,
    evaluation: RunEvaluation,
    scored: list[ScoredQuestion],
    predictions: list[QuestionPrediction],
    labels: list[LabelView],
    *,
    limitations: list[str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Export ``results.jsonl``, ``aggregate.json``, ``manifest.json`` and summary.

    Returns artifact paths plus SHA-256 hashes. Writes are atomic; existing
    files with differing bytes are refused, never silently overwritten.
    """
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    rows = scored_to_result_rows(scored, predictions, labels)
    type_counts: dict[str, int] = {}
    for label in labels:
        if label.question_type:
            type_counts[label.question_type] = type_counts.get(label.question_type, 0) + 1

    results_path = dest / "results.jsonl"
    aggregate_path = dest / "aggregate.json"
    manifest_path = dest / "manifest.json"
    summary_path = dest / "summary.md"

    aggregate_payload = build_aggregate_payload(evaluation, limitations=limitations, notes=notes)
    manifest_payload = {
        "manifest_version": "baseline-manifest-v1",
        "report_version": REPORT_VERSION,
        "run_id": evaluation.config.run_id,
        "strategy": evaluation.config.strategy,
        "corpus_id": evaluation.config.corpus_id,
        "question_ids": sorted(item.question_id for item in scored),
        "question_set_hash": evaluation.config.question_set_hash,
        "config": evaluation.config.model_dump(),
        "scoring_version": evaluation.scoring_version,
        "evaluation_version": evaluation.evaluation_version,
        "created_at": evaluation.created_at,
        "question_type_counts": type_counts,
        "limitations": list(limitations or []),
    }
    summary_text = render_markdown_summary(
        evaluation, scored, limitations=limitations, type_counts=type_counts or None
    )

    _write_guarded_jsonl(results_path, rows)
    _write_guarded_json(aggregate_path, aggregate_payload)
    _write_guarded_text(summary_path, summary_text)

    artifacts = {
        name: sha256_file(dest / name) for name in ("results.jsonl", "aggregate.json", "summary.md")
    }
    manifest_with_hashes = dict(manifest_payload)
    manifest_with_hashes["artifact_sha256"] = dict(artifacts)
    _write_guarded_json(manifest_path, manifest_with_hashes)
    artifacts["manifest.json"] = sha256_file(manifest_path)
    return {
        "output_dir": str(dest),
        "results_jsonl": str(results_path),
        "aggregate_json": str(aggregate_path),
        "manifest_json": str(manifest_path),
        "summary_md": str(summary_path),
        "artifact_sha256": artifacts,
    }


def _read_bytes_if_exists(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def _write_guarded_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    existing = _read_bytes_if_exists(path)
    candidate = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )
    if existing is not None:
        if existing != candidate:
            raise ValueError(
                f"Existing artifact {path.name} differs; choose a new output directory"
            )
        return
    write_jsonl_atomic(path, rows)


def _write_guarded_json(path: Path, payload: dict[str, Any]) -> None:
    existing = _read_bytes_if_exists(path)
    candidate = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    if existing is not None:
        if existing != candidate:
            raise ValueError(
                f"Existing artifact {path.name} differs; choose a new output directory"
            )
        return
    write_json_atomic(path, payload)


def _write_guarded_text(path: Path, text: str) -> None:
    existing = _read_bytes_if_exists(path)
    candidate = text.encode("utf-8")
    if existing is not None:
        if existing != candidate:
            raise ValueError(
                f"Existing artifact {path.name} differs; choose a new output directory"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(candidate)
    tmp.rename(path)


# ---------------------------------------------------------------------------
# T14 paired comparison exports (dashboard, R09/R12)
# ---------------------------------------------------------------------------

COMPARISON_REPORT_VERSION = "comparison-report-v1"

REGRESSION_FILTERS: tuple[str, ...] = (
    "Improved",
    "Worsened",
    "Unchanged",
    "Failed",
    "Abstained",
)

REGRESSION_CSV_COLUMNS: tuple[str, ...] = (
    "question_id",
    "question_type",
    "status_a",
    "status_b",
    "answer_a",
    "answer_b",
    "gold_answer",
    "em_a",
    "em_b",
    "f1_a",
    "f1_b",
    "delta_f1_pp",
    "delta_em_pp",
    "all_support_a",
    "all_support_b",
    "delta_support_pp",
    "labels",
    "latency_a_ms",
    "latency_b_ms",
    "failure_code_a",
    "failure_code_b",
)


def classify_regression_labels(
    delta_f1_pp: float,
    status_a: str,
    status_b: str,
) -> list[str]:
    """Classify one paired row for the regression filters (R09).

    Direction comes from answer F1 only: positive delta is Improved,
    negative is Worsened, zero is Unchanged. Failed/Abstained are additive:
    a row where either arm failed carries Failed (and may also carry a
    direction label); likewise for insufficient_evidence and Abstained.
    Filters affect views, never the saved denominator.
    """
    if not isinstance(delta_f1_pp, (int, float)) or isinstance(delta_f1_pp, bool):
        raise ValueError("delta_f1_pp must be a number")
    if status_a not in ("supported", "insufficient_evidence", "failed"):
        raise ValueError(f"unknown status_a {status_a!r}")
    if status_b not in ("supported", "insufficient_evidence", "failed"):
        raise ValueError(f"unknown status_b {status_b!r}")
    labels: list[str] = []
    delta = float(delta_f1_pp)
    if delta > 1e-9:
        labels.append("Improved")
    elif delta < -1e-9:
        labels.append("Worsened")
    else:
        labels.append("Unchanged")
    if status_a == "failed" or status_b == "failed":
        labels.append("Failed")
    if status_a == "insufficient_evidence" or status_b == "insufficient_evidence":
        labels.append("Abstained")
    return labels


def _question_type_for(question_id: str, labels: list[LabelView] | dict[str, LabelView]) -> str:
    if isinstance(labels, dict):
        label = labels.get(question_id)
    else:
        label = next((item for item in labels if item.question_id == question_id), None)
    if label is None:
        return "unknown"
    kind = getattr(label, "question_type", None)
    return str(kind) if kind else "unknown"


def build_regression_rows(
    scored_a: list[ScoredQuestion],
    scored_b: list[ScoredQuestion],
    predictions_a: list[QuestionPrediction],
    predictions_b: list[QuestionPrediction],
    labels: list[LabelView],
) -> list[dict[str, Any]]:
    """Join two frozen arms into per-question regression rows.

    Both arms must cover exactly the same question ids in the fixed
    denominator; failures score zero and stay in the rows (never dropped).
    Direction labels derive from F1; Failed/Abstained are additive. Rows are
    sorted by question_id for deterministic exports.
    """
    by_scored_a = {item.question_id: item for item in scored_a}
    by_scored_b = {item.question_id: item for item in scored_b}
    ids_a = sorted(by_scored_a)
    ids_b = sorted(by_scored_b)
    if ids_a != ids_b:
        raise ValueError(
            f"Paired arms must cover exactly the same question ids; {len(ids_a)} vs {len(ids_b)}"
        )
    if len(by_scored_a) != len(scored_a) or len(by_scored_b) != len(scored_b):
        raise ValueError("Duplicate question ids in a paired arm")
    by_pred_a = {item.question_id: item for item in predictions_a}
    by_pred_b = {item.question_id: item for item in predictions_b}
    if sorted(by_pred_a) != ids_a or sorted(by_pred_b) != ids_b:
        raise ValueError("Predictions must cover exactly the scored question ids")
    by_label: dict[str, LabelView] = {item.question_id: item for item in labels}
    if sorted(by_label) != ids_a:
        raise ValueError("Labels must cover exactly the paired question ids")
    rows: list[dict[str, Any]] = []
    for qid in ids_a:
        item_a = by_scored_a[qid]
        item_b = by_scored_b[qid]
        pred_a = by_pred_a[qid]
        pred_b = by_pred_b[qid]
        label = by_label[qid]
        f1_a = float(item_a.f1) * 100.0
        f1_b = float(item_b.f1) * 100.0
        em_a = float(item_a.exact_match) * 100.0
        em_b = float(item_b.exact_match) * 100.0
        sup_a = float(item_a.packed.all_support) * 100.0
        sup_b = float(item_b.packed.all_support) * 100.0
        delta_f1 = f1_b - f1_a
        delta_em = em_b - em_a
        delta_sup = sup_b - sup_a
        row_labels = classify_regression_labels(delta_f1, item_a.status, item_b.status)
        rows.append(
            {
                "question_id": qid,
                "question_type": str(label.question_type or "unknown"),
                "status_a": item_a.status,
                "status_b": item_b.status,
                "answer_a": pred_a.answer,
                "answer_b": pred_b.answer,
                "gold_answer": label.answer,
                "em_a": round(em_a, 4),
                "em_b": round(em_b, 4),
                "f1_a": round(f1_a, 4),
                "f1_b": round(f1_b, 4),
                "delta_f1_pp": round(delta_f1, 4),
                "delta_em_pp": round(delta_em, 4),
                "all_support_a": round(sup_a, 4),
                "all_support_b": round(sup_b, 4),
                "delta_support_pp": round(delta_sup, 4),
                "labels": list(row_labels),
                "latency_a_ms": pred_a.latency_ms,
                "latency_b_ms": pred_b.latency_ms,
                "failure_code_a": item_a.failure_code,
                "failure_code_b": item_b.failure_code,
            }
        )
    rows.sort(key=lambda row: str(row["question_id"]))
    return rows


def format_cost_usd(value: float | None) -> str:
    """Honest money label: estimated with units, or unknown (never zero)."""
    if value is None:
        return "unknown"
    return f"${float(value):.6f} (estimated)"


def build_comparison_payload(
    comparison: dict[str, Any],
    evaluation_a: RunEvaluation,
    evaluation_b: RunEvaluation,
    rows: list[dict[str, Any]],
    *,
    limitations: list[str] | None = None,
    indexing_costs: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Machine-readable paired report joining aggregates and per-question rows.

    Query costs come from the frozen run aggregates (tokens + estimated
    money, unknown stays null). Indexing costs are one-time build spend
    supplied by the caller and stored separately; when absent both arms are
    recorded as unknown with an explicit note. Raises when the row set does
    not match the comparison denominator so exports can never drift from
    the paired question set.
    """
    if not isinstance(comparison, dict) or not comparison.get("compatible"):
        raise ValueError("Comparison payload requires a compatible compare_runs result")
    expected = list(comparison.get("question_ids") or [])
    got = sorted(str(row["question_id"]) for row in rows)
    if sorted(expected) != got:
        raise ValueError(
            "Regression rows must cover exactly the compared question ids; "
            f"{len(got)} rows vs {len(expected)} compared"
        )
    if evaluation_a.aggregate.n_total != len(rows) or evaluation_b.aggregate.n_total != len(rows):
        raise ValueError("Run aggregates must match the regression row count (fixed denominator)")
    indexing = dict(indexing_costs or {})
    for arm in ("baseline", "candidate"):
        if arm not in indexing:
            indexing[arm] = {"estimated_cost_usd": None, "note": "unpriced local compute"}
    payload = {
        "report_version": COMPARISON_REPORT_VERSION,
        "evaluation_version": evaluation_a.evaluation_version,
        "scoring_version": evaluation_a.scoring_version,
        "run_a": {
            "run_id": evaluation_a.config.run_id,
            "strategy": evaluation_a.config.strategy,
            "split": evaluation_a.config.split,
            "config": evaluation_a.config.model_dump(),
            "aggregate": evaluation_a.aggregate.model_dump(),
            "success_only": evaluation_a.success_only.model_dump(),
        },
        "run_b": {
            "run_id": evaluation_b.config.run_id,
            "strategy": evaluation_b.config.strategy,
            "split": evaluation_b.config.split,
            "config": evaluation_b.config.model_dump(),
            "aggregate": evaluation_b.aggregate.model_dump(),
            "success_only": evaluation_b.success_only.model_dump(),
        },
        "compatibility_hash": comparison.get("compatibility_hash"),
        "fixed": dict(comparison.get("fixed") or {}),
        "n": comparison.get("n"),
        "metrics": comparison.get("metrics"),
        "strata": comparison.get("strata"),
        "failures": comparison.get("failures"),
        "warnings": list(comparison.get("warnings") or []),
        "query_costs": {
            "baseline": {
                "input_tokens": evaluation_a.aggregate.total_input_tokens,
                "output_tokens": evaluation_a.aggregate.total_output_tokens,
                "estimated_cost_usd": evaluation_a.aggregate.total_estimated_cost_usd,
                "label": format_cost_usd(evaluation_a.aggregate.total_estimated_cost_usd),
            },
            "candidate": {
                "input_tokens": evaluation_b.aggregate.total_input_tokens,
                "output_tokens": evaluation_b.aggregate.total_output_tokens,
                "estimated_cost_usd": evaluation_b.aggregate.total_estimated_cost_usd,
                "label": format_cost_usd(evaluation_b.aggregate.total_estimated_cost_usd),
            },
        },
        "indexing_costs": indexing,
        "question_ids": sorted(expected),
        "limitations": list(
            limitations
            or [
                "Paired deltas describe the observed sets; shared evidence and small "
                "samples limit inference to population-wide significance.",
                "Citation validity means the citation resolves to packed evidence, not entailment.",
                "Costs cover LLM usage when reported; local compute time is measured, not priced.",
            ]
        ),
    }
    if notes is not None:
        payload["notes"] = notes
    return payload


def render_comparison_markdown(
    comparison: dict[str, Any],
    evaluation_a: RunEvaluation,
    evaluation_b: RunEvaluation,
    rows: list[dict[str, Any]],
    *,
    limitations: list[str] | None = None,
    indexing_costs: dict[str, Any] | None = None,
) -> str:
    """Human-readable paired report: headlines first, costs separated."""
    if not comparison.get("compatible"):
        raise ValueError("Cannot render deltas for incompatible runs")
    agg_a = evaluation_a.aggregate
    agg_b = evaluation_b.aggregate
    metrics = comparison.get("metrics") or {}
    f1 = metrics.get("f1") or {}
    sup = metrics.get("all_support_packed") or {}

    def _interval(payload: dict[str, Any]) -> str:
        if payload.get("lower_pp") is None or payload.get("upper_pp") is None:
            return "interval unavailable (insufficient n)"
        return (
            f"{float(payload['observed_delta_pp']):+.2f}pp, "
            f"95% CI [{float(payload['lower_pp']):+.2f}, "
            f"{float(payload['upper_pp']):+.2f}] "
            f"(n={payload.get('n')}, seed {payload.get('seed')}, "
            f"{payload.get('replicates')} replicates)"
        )

    split = evaluation_a.config.split or evaluation_b.config.split or "split unspecified"
    indexing = dict(indexing_costs or {})
    lines = [
        f"# Comparison report — {evaluation_a.config.run_id} vs {evaluation_b.config.run_id}",
        "",
        f"Baseline `{evaluation_a.config.strategy}` vs candidate "
        f"`{evaluation_b.config.strategy}` · {agg_a.n_total} questions ({split})",
        "",
        "## Compatibility",
        "",
        f"- Compatible: yes (`{comparison.get('compatibility_hash')}`)",
        f"- Corpus `{evaluation_a.config.corpus_id}` · embedding "
        f"`{evaluation_a.config.embedding_model}` · answer `{evaluation_a.config.answer_model}`",
        f"- Final budgets: top {evaluation_a.config.max_chunks} / "
        f"{evaluation_a.config.max_context_tokens} tokens (matched)",
        f"- Question set hash: `{evaluation_a.config.question_set_hash}`",
        "",
        "## Headline comparisons (fixed denominator, 0–100 scale)",
        "",
        "| Metric | Baseline | Candidate | Paired delta |",
        "|---|---|---|---|",
        f"| Answer F1 | {agg_a.f1_mean:.2f} | {agg_b.f1_mean:.2f} | {_interval(f1)} |",
        f"| All-support coverage (packed) | {agg_a.all_support_packed:.2f} | "
        f"{agg_b.all_support_packed:.2f} | {_interval(sup)} |",
        "| Latency p50 (ms) | "
        + (f"{agg_a.latency_p50_ms:.1f}" if agg_a.latency_p50_ms is not None else "not measured")
        + " | "
        + (f"{agg_b.latency_p50_ms:.1f}" if agg_b.latency_p50_ms is not None else "not measured")
        + " | observed only (no paired interval) |",
        "| Latency p95 (ms) | "
        + (f"{agg_a.latency_p95_ms:.1f}" if agg_a.latency_p95_ms is not None else "not measured")
        + " | "
        + (f"{agg_b.latency_p95_ms:.1f}" if agg_b.latency_p95_ms is not None else "not measured")
        + " | observed only (no paired interval) |",
        f"| Failed / abstained / supported (baseline) | {agg_a.n_failed} / "
        f"{agg_a.n_abstained} / {agg_a.n_supported} | — | n={agg_a.n_total} |",
        f"| Failed / abstained / supported (candidate) | {agg_b.n_failed} / "
        f"{agg_b.n_abstained} / {agg_b.n_supported} | — | n={agg_b.n_total} |",
        "",
        "Counts and split sit beside the headline; uncertainty sits beside the "
        "quality deltas. No celebratory coloring is applied to inconclusive results.",
        "",
        "## Strata (fixed denominator per stratum)",
        "",
        "| Stratum | n | F1 baseline → candidate | All-support baseline → candidate |",
        "|---|---|---|---|",
    ]
    strata = comparison.get("strata") or {}
    for name in sorted(strata):
        entry = strata[name]
        entry_metrics = entry.get("metrics") or {}
        f1_s = entry_metrics.get("f1") or {}
        sup_s = entry_metrics.get("all_support_packed") or {}
        lines.append(
            f"| {name} | {entry.get('n')} | "
            f"{float(f1_s.get('mean_a', 0.0)):.2f} → "
            f"{float(f1_s.get('mean_b', 0.0)):.2f} "
            f"({float(f1_s.get('observed_delta_pp', 0.0)):+.2f}pp) | "
            f"{float(sup_s.get('mean_a', 0.0)):.2f} → "
            f"{float(sup_s.get('mean_b', 0.0)):.2f} "
            f"({float(sup_s.get('observed_delta_pp', 0.0)):+.2f}pp) |"
        )
    lines.extend(
        [
            "",
            "## Per-question regression (F1 direction; Failed/Abstained additive)",
            "",
            "| question_id | baseline F1 | candidate F1 | delta (pp) | labels |",
            "|---|---|---|---|---|",
        ]
    )
    for row in sorted(rows, key=lambda entry: str(entry["question_id"])):
        labels_text = ";".join(row["labels"])
        lines.append(
            f"| {row['question_id']} | {float(row['f1_a']):.2f} | "
            f"{float(row['f1_b']):.2f} | {float(row['delta_f1_pp']):+.2f} | {labels_text} |"
        )
    lines.extend(
        [
            "",
            "## Query cost (per evaluation run, separated from indexing)",
            "",
            "| Arm | Input tokens | Output tokens | Estimated cost (USD) |",
            "|---|---|---|---|",
            f"| Baseline `{evaluation_a.config.strategy}` | "
            f"{agg_a.total_input_tokens} | {agg_a.total_output_tokens} | "
            f"{format_cost_usd(agg_a.total_estimated_cost_usd)} |",
            f"| Candidate `{evaluation_b.config.strategy}` | "
            f"{agg_b.total_input_tokens} | {agg_b.total_output_tokens} | "
            f"{format_cost_usd(agg_b.total_estimated_cost_usd)} |",
            "",
            "## Indexing cost (one-time build, separated from queries)",
            "",
        ]
    )
    if indexing:
        lines.extend(["| Scope | Detail |", "|---|---|"])
        for scope in sorted(indexing):
            detail = indexing[scope]
            if isinstance(detail, dict):
                cost = format_cost_usd(detail.get("estimated_cost_usd"))
                note = str(detail.get("note", ""))
                lines.append(f"| {scope} | {cost} {note} |".rstrip())
            else:
                lines.append(f"| {scope} | {detail} |")
    else:
        lines.append("One-time indexing cost: unknown (unpriced local compute).")
    lines.extend(["", "## Limitations", ""])
    for item in limitations or [
        "Paired deltas describe the observed sets; shared evidence and small "
        "samples limit inference to population-wide significance.",
        "Citation validity means the citation resolves to packed evidence, not entailment.",
        "Costs cover LLM usage when reported; local compute time is measured, not priced.",
    ]:
        lines.append(f"- {item}")
    lines.append("")
    for warning in comparison.get("warnings") or []:
        lines.append(f"- {warning}")
    lines.append("")
    lines.append(
        f"Report version: `{COMPARISON_REPORT_VERSION}` · "
        f"compatibility `{comparison.get('compatibility_hash')}`."
    )
    lines.append("")
    return "\n".join(lines)


def export_comparison(
    output_dir: str | Path,
    comparison: dict[str, Any],
    evaluation_a: RunEvaluation,
    evaluation_b: RunEvaluation,
    scored_a: list[ScoredQuestion],
    scored_b: list[ScoredQuestion],
    predictions_a: list[QuestionPrediction],
    predictions_b: list[QuestionPrediction],
    labels: list[LabelView],
    *,
    limitations: list[str] | None = None,
    indexing_costs: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Export ``comparison.json``, ``regression.csv`` and ``report.md``.

    Returns artifact paths plus SHA-256 hashes. Writes are atomic; existing
    files with differing bytes are refused, never silently overwritten.
    Aggregates reconcile with the per-question rows by construction: rows are
    built from the same scored lists and the payload validates the fixed
    denominator including failures.
    """
    if not isinstance(comparison, dict) or not comparison.get("compatible"):
        raise ValueError(
            f"Incompatible runs cannot produce an unlabeled paired delta: "
            f"{comparison.get('reason') if isinstance(comparison, dict) else comparison!r}"
        )
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    rows = build_regression_rows(scored_a, scored_b, predictions_a, predictions_b, labels)
    payload = build_comparison_payload(
        comparison,
        evaluation_a,
        evaluation_b,
        rows,
        limitations=limitations,
        indexing_costs=indexing_costs,
        notes=notes,
    )
    markdown = render_comparison_markdown(
        comparison,
        evaluation_a,
        evaluation_b,
        rows,
        limitations=limitations,
        indexing_costs=indexing_costs,
    )
    comparison_path = dest / "comparison.json"
    regression_path = dest / "regression.csv"
    report_path = dest / "report.md"
    _write_guarded_json(comparison_path, payload)
    _write_guarded_regression_csv(regression_path, rows)
    _write_guarded_text(report_path, markdown)
    artifacts = {
        name: sha256_file(dest / name)
        for name in ("comparison.json", "regression.csv", "report.md")
    }
    return {
        "output_dir": str(dest),
        "comparison_json": str(comparison_path),
        "regression_csv": str(regression_path),
        "report_md": str(report_path),
        "artifact_sha256": artifacts,
        "n": len(rows),
        "compatibility_hash": comparison.get("compatibility_hash"),
    }


def _write_guarded_regression_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv
    import io

    existing = _read_bytes_if_exists(path)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(REGRESSION_CSV_COLUMNS), extrasaction="ignore")
    writer.writeheader()
    for row in sorted(rows, key=lambda entry: str(entry["question_id"])):
        record = dict(row)
        record["labels"] = ";".join(row.get("labels") or [])
        for key in ("latency_a_ms", "latency_b_ms", "failure_code_a", "failure_code_b"):
            if record.get(key) is None:
                record[key] = ""
        writer.writerow({column: record.get(column, "") for column in REGRESSION_CSV_COLUMNS})
    candidate = buffer.getvalue().encode("utf-8")
    if existing is not None:
        if existing != candidate:
            raise ValueError(
                f"Existing artifact {path.name} differs; choose a new output directory"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(candidate)
    tmp.rename(path)
