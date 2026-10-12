"""Evaluations: answer the research question first.

Run selection plus compatibility status leads; three headline cards (Answer
F1, All-support coverage, latency) show absolutes with counts/split and
deltas with uncertainty only for compatible runs. Multi-hop and single-hop
controls use separate tabs; the full metrics table follows column priority
Question → Baseline → Candidate → Difference → Status with Improved /
Worsened / Unchanged / Failed / Abstained filters. Selecting a row opens
paired saved answers and evidence without rerunning anything. Indexing and
query costs are separated; unknown money stays unknown. Export offers CSV,
JSON and Markdown beside run selection. With no saved results the page
shows an empty state and no placeholder charts.

T17 (optional, R13): decomposition runs (strategy ``decomposition``) are
compared additively with a visible banner — tuning selection freezes before
any locked use, final budgets stay matched, extra planner/sub-answer
tokens and latency are part of the result, and A-D results are never
replaced.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.metrics import (
    CONFIGURE_LABEL,
    FILTERS,
    compatibility_banner,
    cost_panels,
    empty_state,
    failure_rows,
    filter_regression_rows,
    headline_cards,
    render_cost_panels,
    render_headlines,
    split_rows_by_tab,
    table_rows,
)
from app.components.status import render_footer, render_page_chrome
from graphrag_lab.compare_flow import encode_deep_link
from graphrag_lab.jobs import DEFAULT_DB_PATH
from graphrag_lab.run_store import RunStore

DEFAULT_DATASET_DIR = "artifacts/dataset"
CONTROLS_PATH = "data/controls/single_hop.jsonl"
DECOMPOSITION_STRATEGY = "decomposition"

DECOMPOSITION_BANNER = (
    "Decomposition (E) is optional and additive: this pair compares a baseline "
    "with bounded sequential sub-retrieval (at most 2 sub-questions, shared top-8 / "
    "3000-token budget). Selection freezes on the tuning set before any locked use; "
    "extra planner/sub-answer tokens and latency are part of the result; frozen A-D "
    "results are unchanged. Unknown costs stay unknown."
)


def _involves_decomposition(view: dict[str, Any]) -> bool:
    """True when either compared run uses the optional decomposition arm."""
    for key in ("run_a", "run_b"):
        run = view.get(key)
        if run is not None and getattr(run, "strategy", None) == DECOMPOSITION_STRATEGY:
            return True
    for key in ("evaluation_a", "evaluation_b"):
        evaluation = view.get(key)
        if evaluation is not None:
            try:
                if evaluation.config.strategy == DECOMPOSITION_STRATEGY:
                    return True
            except AttributeError:
                continue
    return False


def _decomposition_limitations() -> list[str]:
    """Limitations block for decomposition delta exports (additive, R13)."""
    try:
        from graphrag_lab.evaluation import decomposition_report_limitations

        return decomposition_report_limitations()
    except ImportError:
        return [DECOMPOSITION_BANNER]


def _db_path() -> str:
    import streamlit as st

    return str(st.session_state.get("db_path", DEFAULT_DB_PATH))


def _run_label(run: Any) -> str:
    return f"{run.id} · {run.strategy} · {run.split or 'split?'} · {run.state}"


def _load_labels() -> tuple[list[dict[str, Any]] | None, str | None]:
    """Frozen labels for scoring; None with a remedy when artifacts are absent."""
    dataset_labels = Path(DEFAULT_DATASET_DIR) / "labels.jsonl"
    if not dataset_labels.is_file():
        return None, (
            "Labels unavailable: no frozen dataset at "
            f"{DEFAULT_DATASET_DIR}/labels.jsonl. Scores show Not measured; "
            "saved answers below were not rerun."
        )
    try:
        rows = [
            json.loads(line)
            for line in dataset_labels.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError) as exc:
        return None, f"Labels unreadable ({exc}). Scores show Not measured."
    controls = Path(CONTROLS_PATH)
    if controls.is_file():
        try:
            from graphrag_lab.evaluation import SingleHopControl

            for line in controls.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                control = SingleHopControl.model_validate(json.loads(line))
                rows.append(
                    {
                        "question_id": control.question_id,
                        "answer": control.answer,
                        "supporting_facts": [
                            {
                                "title": fact.title,
                                "sentence_id": fact.sentence_id,
                                "paragraph_id": fact.paragraph_id,
                            }
                            for fact in control.supporting_facts
                        ],
                        "question_type": "single_hop",
                    }
                )
        except (OSError, ValueError):
            pass
    return rows, None


def _prediction_to_question_pred(record: Any) -> dict[str, Any] | None:
    """Map a saved RunStore prediction to a QuestionPrediction payload.

    Returns None when the saved retrieval cannot be scored (missing
    paragraph/title provenance); callers then show Not measured instead of
    inventing coverage.
    """
    retrieval = dict(record.retrieval_json or {})
    chunks = retrieval.get("chunks") or []
    packed_ids = retrieval.get("packed_ids") or []
    packed_set = set(packed_ids)
    candidates: list[dict[str, Any]] = []
    packed: list[dict[str, Any]] = []
    try:
        for chunk in chunks:
            if not isinstance(chunk, dict):
                return None
            cid = str(chunk.get("chunk_id", chunk.get("id", "")))
            if not cid:
                return None
            view = {
                "chunk_id": cid,
                "paragraph_id": chunk.get("paragraph_id"),
                "title": chunk.get("title"),
                "sentence_ids": list(chunk.get("sentence_ids") or []),
            }
            if not view["paragraph_id"] and not view["title"]:
                return None
            candidates.append(view)
            if cid in packed_set:
                packed.append(view)
    except (TypeError, ValueError):
        return None
    usage = dict(record.usage_json or {}) if record.usage_json else {}
    citations = list(record.citations_json or [])
    return {
        "question_id": record.question_id,
        "answer": record.answer or "",
        "status": record.status,
        "citations": citations,
        "candidates": candidates,
        "packed": packed,
        "latency_ms": record.latency_ms,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "estimated_cost_usd": usage.get("estimated_cost_usd"),
        "failure_code": record.failure_code,
        "strategy": retrieval.get("strategy"),
    }


def _build_view(store: RunStore, baseline_id: str, candidate_id: str) -> dict[str, Any]:
    """Load two saved runs and score them without any retrieval or LLM call.

    Returns a dict with either scored comparison data (evaluations,
    comparison, rows) or a ``notice`` explaining why scores show Not
    measured. Saved answers are always returned for drill-down; nothing is
    ever rerun here.
    """
    try:
        run_a = store.get_run(baseline_id)
        run_b = store.get_run(candidate_id)
    except Exception as exc:
        return {"notice": f"Saved runs cannot be opened: {exc}", "answers": []}
    preds_a = store.list_predictions(baseline_id)
    preds_b = store.list_predictions(candidate_id)
    answers = [
        {
            "run_id": baseline_id,
            "strategy": run_a.strategy,
            "predictions": [
                {
                    "question_id": p.question_id,
                    "answer": p.answer,
                    "status": p.status,
                    "citations": list(p.citations_json or []),
                    "latency_ms": p.latency_ms,
                    "failure_code": p.failure_code,
                }
                for p in preds_a
            ],
        },
        {
            "run_id": candidate_id,
            "strategy": run_b.strategy,
            "predictions": [
                {
                    "question_id": p.question_id,
                    "answer": p.answer,
                    "status": p.status,
                    "citations": list(p.citations_json or []),
                    "latency_ms": p.latency_ms,
                    "failure_code": p.failure_code,
                }
                for p in preds_b
            ],
        },
    ]
    if not preds_a or not preds_b:
        return {
            "notice": "One selected run has no saved predictions; scores show Not measured.",
            "answers": answers,
            "run_a": run_a,
            "run_b": run_b,
        }
    label_rows, label_notice = _load_labels()
    if label_rows is None:
        return {"notice": label_notice, "answers": answers, "run_a": run_a, "run_b": run_b}
    from graphrag_lab.evaluation import (
        LabelView,
        QuestionPrediction,
        RunConfig,
        compare_runs,
        evaluate_predictions,
    )
    from graphrag_lab.reports import build_regression_rows

    ids_a = sorted(p.question_id for p in preds_a)
    ids_b = sorted(p.question_id for p in preds_b)
    if ids_a != ids_b:
        return {
            "notice": (
                "Selected runs cover different questions; refusing unlabeled paired delta. "
                "Scores show Not measured for the pair."
            ),
            "answers": answers,
            "run_a": run_a,
            "run_b": run_b,
            "comparison": {
                "compatible": False,
                "reason": (
                    f"Paired arms cover different questions ({len(ids_a)} vs {len(ids_b)}); "
                    "refusing unlabeled paired delta"
                ),
            },
        }
    by_label = {
        row["question_id"]: row for row in label_rows if row.get("question_id") in set(ids_a)
    }
    if sorted(by_label) != ids_a:
        return {
            "notice": (
                "Frozen labels do not cover the selected questions; scores show Not measured. "
                "Saved answers below were not rerun."
            ),
            "answers": answers,
            "run_a": run_a,
            "run_b": run_b,
        }
    try:
        cfg_a = RunConfig.model_validate(dict(run_a.config_json))
        cfg_b = RunConfig.model_validate(dict(run_b.config_json))
    except Exception as exc:
        return {
            "notice": (
                f"Selected runs lack frozen evaluation configs ({exc}); "
                "refusing unlabeled paired delta. Saved answers below were not rerun."
            ),
            "answers": answers,
            "run_a": run_a,
            "run_b": run_b,
            "comparison": {
                "compatible": False,
                "reason": f"Frozen configs unavailable: {exc}",
            },
        }
    try:
        labels = [LabelView.model_validate(by_label[qid]) for qid in ids_a]
        qpreds_a = [
            QuestionPrediction.model_validate(_prediction_to_question_pred(p)) for p in preds_a
        ]
        qpreds_b = [
            QuestionPrediction.model_validate(_prediction_to_question_pred(p)) for p in preds_b
        ]
    except Exception as exc:
        return {
            "notice": f"Saved evidence cannot be scored ({exc}); scores show Not measured.",
            "answers": answers,
            "run_a": run_a,
            "run_b": run_b,
        }
    if any(item is None for item in qpreds_a) or any(item is None for item in qpreds_b):
        return {
            "notice": (
                "Saved retrieval lacks paragraph/title provenance; "
                "scores show Not measured. Saved answers below were not rerun."
            ),
            "answers": answers,
            "run_a": run_a,
            "run_b": run_b,
        }
    try:
        scored_a, eval_a = evaluate_predictions(labels, qpreds_a, cfg_a)
        scored_b, eval_b = evaluate_predictions(labels, qpreds_b, cfg_b)
        comparison = compare_runs(scored_a, eval_a, scored_b, eval_b, labels)
        rows = build_regression_rows(scored_a, scored_b, qpreds_a, qpreds_b, labels)
    except Exception as exc:
        return {"notice": f"Evaluation failed: {exc}", "answers": answers}
    return {
        "notice": None if comparison.get("compatible") else str(comparison.get("reason")),
        "answers": answers,
        "run_a": run_a,
        "run_b": run_b,
        "evaluation_a": eval_a,
        "evaluation_b": eval_b,
        "comparison": comparison,
        "rows": rows,
        "scored_a": scored_a,
        "scored_b": scored_b,
        "predictions_a": qpreds_a,
        "predictions_b": qpreds_b,
        "labels": labels,
    }


def _regression_csv(rows: list[dict[str, Any]]) -> str:
    from graphrag_lab.reports import REGRESSION_CSV_COLUMNS

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
    return buffer.getvalue()


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="GraphRAG Lab — Evaluations", layout="wide")
    render_page_chrome(
        title="Evaluations",
        subtitle="Compare frozen runs with paired deltas, uncertainty and honest costs.",
    )
    store = RunStore(_db_path())
    runs = store.list_runs(limit=200)
    if len(runs) < 2:
        state = empty_state()
        st.info(f"{state['title']}: {state['hint']}")
        if st.button(state["action"], key="eval-configure"):
            st.switch_page("pages/dataset.py")
        st.caption(CONFIGURE_LABEL + " lives in Dataset & Index; no scores are invented.")
        render_footer(
            corpus="see Overview", index="see Dataset & Index", models="see Overview footer"
        )
        return

    options = [_run_label(run) for run in runs]
    ids = [run.id for run in runs]
    with st.form("eval-select"):
        baseline_choice = st.selectbox("Baseline run", options, index=0)
        candidate_choice = st.selectbox(
            "Candidate run", options, index=1 if len(options) > 1 else 0
        )
        load = st.form_submit_button("Load comparison")
    baseline_id = ids[options.index(baseline_choice)]
    candidate_id = ids[options.index(candidate_choice)]
    if baseline_id == candidate_id:
        st.error("Pick two distinct runs; a run cannot be compared with itself.")
        return
    view = _build_view(store, baseline_id, candidate_id) if load or True else {}
    comparison = view.get("comparison")
    if comparison is not None:
        st.write(compatibility_banner(comparison))
    if _involves_decomposition(view):
        st.info(DECOMPOSITION_BANNER)
    if view.get("notice"):
        st.warning(str(view["notice"]))
    if "rows" not in view or "evaluation_a" not in view:
        st.write("Scores: Not measured for this pair. Saved answers are shown without deltas.")
        for bundle in view.get("answers", []):
            with st.expander(f"Saved answers — {bundle['strategy']} (`{bundle['run_id']}`)"):
                for pred in bundle["predictions"][:20]:
                    st.write(
                        f"`{pred['question_id']}` [{pred['status']}] "
                        f"{pred['answer'] or '(no answer)'}"
                    )
                st.caption("Opening a saved result never reruns the question.")
                st.code(f"deep link: ?{encode_deep_link([bundle['run_id']])}", language="text")
        # Query/index spend stays separated even when scores are missing.
        query_costs: dict[str, Any] = {}
        for bundle in view.get("answers", []):
            summary = store.sum_usage_for_run(bundle["run_id"])
            scope = (
                "candidate"
                if bundle["strategy"] == view.get("run_b", bundle).strategy
                else "baseline"
            )
            query_costs[scope] = {
                "input_tokens": summary["input_tokens"],
                "output_tokens": summary["output_tokens"],
                "estimated_cost_usd": summary["estimated_cost_usd"],
            }
        render_cost_panels(cost_panels(query_costs, {}))
        render_footer(
            corpus="see Overview", index="see Dataset & Index", models="see Overview footer"
        )
        return

    eval_a = view["evaluation_a"]
    eval_b = view["evaluation_b"]
    rows = view["rows"]
    st.subheader("Headlines")
    render_headlines(headline_cards(comparison, eval_a, eval_b))

    with st.expander("Export (beside run selection)"):
        st.caption("CSV, JSON and Markdown exports of the current paired view.")
        st.download_button(
            "Download regression CSV",
            data=_regression_csv(rows),
            file_name="regression.csv",
            mime="text/csv",
            key="eval-dl-csv",
        )
        from graphrag_lab.reports import build_comparison_payload, render_comparison_markdown

        export_limitations = _decomposition_limitations() if _involves_decomposition(view) else None
        payload = build_comparison_payload(
            comparison, eval_a, eval_b, rows, limitations=export_limitations
        )
        st.download_button(
            "Download comparison JSON",
            data=json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            file_name="comparison.json",
            mime="application/json",
            key="eval-dl-json",
        )
        st.download_button(
            "Download report Markdown",
            data=render_comparison_markdown(
                comparison, eval_a, eval_b, rows, limitations=export_limitations
            ),
            file_name="report.md",
            mime="text/markdown",
            key="eval-dl-md",
        )
        if export_limitations is not None:
            st.caption(
                "Decomposition export carries the additive R13 limitations block; "
                "frozen A-D results are unchanged."
            )

    tabs = st.tabs(["Multi-hop", "Single-hop controls"])
    split = split_rows_by_tab(rows)
    for position, key in enumerate(("multi_hop", "control")):
        with tabs[position]:
            subset = split[key]
            if not subset:
                st.write("No questions in this tab for the selected runs.")
                continue
            picked = st.multiselect(
                "Filter results",
                list(FILTERS),
                default=[],
                key=f"eval-filter-{key}",
            )
            visible = filter_regression_rows(subset, picked)
            st.write(f"{len(visible)}/{len(subset)} questions shown (denominator unchanged).")
            st.dataframe(table_rows(visible), use_container_width=True)
            failed = failure_rows(visible)
            with st.expander(f"Failure analysis ({len(failed)} rows)"):
                if not failed:
                    st.write("No failed or abstained questions in this view.")
                else:
                    st.dataframe(table_rows(failed), use_container_width=True)
            qids = [str(row["question_id"]) for row in visible]
            if qids:
                chosen = st.selectbox(
                    "Open a paired result (saved answers, never reruns)",
                    qids,
                    key=f"eval-drill-{key}",
                )
                row = next(item for item in visible if str(item["question_id"]) == chosen)
                st.write(f"Baseline answer: {row.get('answer_a') or '(no answer)'}")
                st.write(f"Candidate answer: {row.get('answer_b') or '(no answer)'}")
                st.write(f"Gold answer (evaluation review only): {row.get('gold_answer')}")
                st.caption(
                    f"Baseline F1 {float(row.get('f1_a', 0.0)):.2f} · candidate F1 "
                    f"{float(row.get('f1_b', 0.0)):.2f} · "
                    f"delta {float(row.get('delta_f1_pp', 0.0)):+.2f}pp · "
                    f"labels {';'.join(row.get('labels') or [])}"
                )
                st.code(
                    f"deep link: ?{encode_deep_link([baseline_id, candidate_id])}",
                    language="text",
                )

    st.subheader("Costs")
    query_costs = {
        "baseline": {
            "input_tokens": eval_a.aggregate.total_input_tokens,
            "output_tokens": eval_a.aggregate.total_output_tokens,
            "estimated_cost_usd": eval_a.aggregate.total_estimated_cost_usd,
        },
        "candidate": {
            "input_tokens": eval_b.aggregate.total_input_tokens,
            "output_tokens": eval_b.aggregate.total_output_tokens,
            "estimated_cost_usd": eval_b.aggregate.total_estimated_cost_usd,
        },
    }
    render_cost_panels(cost_panels(query_costs, {}))
    render_footer(corpus="see Overview", index="see Dataset & Index", models="see Overview footer")


if __name__ == "__main__":
    main()
