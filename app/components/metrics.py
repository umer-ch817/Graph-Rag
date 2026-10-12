"""Evaluation headline, regression and cost presentation (UI-UX.md revision 2).

Pure helpers build the exact strings and table rows the e2e suite asserts.
``render_*`` wrappers import Streamlit lazily so importing this module never
requires the UI dependency. Headlines lead with compatible-run F1,
all-support coverage and latency; deltas appear only for compatible runs
with uncertainty beside quality deltas. Filters affect views, never the
saved denominator. Indexing cost and query cost are always separated;
unknown money stays unknown, never zero. Opening a saved result never
reruns a question: all helpers read saved aggregates and rows only.
"""

from __future__ import annotations

from typing import Any

from graphrag_lab.reports import REGRESSION_FILTERS, format_cost_usd

EMPTY_TITLE = "No evaluation results yet"
EMPTY_HINT = (
    "Run a frozen evaluation first, then return here to compare arms. "
    "No placeholder charts are shown before measured results exist."
)
CONFIGURE_LABEL = "Configure evaluation"

FILTERS: tuple[str, ...] = REGRESSION_FILTERS

_TAB_CONTROL = "single_hop"


def empty_state() -> dict[str, str]:
    """Empty evaluations state: no fake scores, one configuration action."""
    return {"title": EMPTY_TITLE, "hint": EMPTY_HINT, "action": CONFIGURE_LABEL}


def compatibility_banner(comparison: dict[str, Any]) -> str:
    """One-line compatibility status for run selection."""
    if not isinstance(comparison, dict):
        return "Compatibility unknown: no comparison computed."
    if comparison.get("compatible"):
        return (
            f"Compatible runs (`{comparison.get('compatibility_hash')}`): "
            "paired deltas are labeled with 95% intervals."
        )
    reason = str(comparison.get("reason") or "runs differ on fixed fields")
    return f"Incompatible runs: {reason}. No unlabeled paired delta is shown."


def _metric_interval(comparison: dict[str, Any], metric: str) -> dict[str, Any] | None:
    metrics = comparison.get("metrics") if isinstance(comparison, dict) else None
    if not isinstance(metrics, dict):
        return None
    payload = metrics.get(metric)
    return dict(payload) if isinstance(payload, dict) else None


def headline_cards(
    comparison: dict[str, Any],
    evaluation_a: Any,
    evaluation_b: Any,
) -> list[dict[str, Any]]:
    """Three headline cards: F1, all-support coverage, latency.

    Each card carries both absolute values plus counts and split beside the
    headline. Paired deltas and intervals appear only when ``comparison``
    is compatible; otherwise deltas stay None with the refusal reason.
    Latency shows observed p50/p95 absolutes without a paired interval.
    """
    agg_a = evaluation_a.aggregate
    agg_b = evaluation_b.aggregate
    cfg_a = evaluation_a.config
    compatible = bool(isinstance(comparison, dict) and comparison.get("compatible"))
    reason = str(comparison.get("reason", "")) if isinstance(comparison, dict) else ""
    split = cfg_a.split or getattr(evaluation_b.config, "split", None) or "split unspecified"
    n = int(agg_a.n_total)
    cards: list[dict[str, Any]] = []

    f1_interval = _metric_interval(comparison, "f1") if compatible else None
    cards.append(
        {
            "metric": "Answer F1",
            "baseline": float(agg_a.f1_mean),
            "candidate": float(agg_b.f1_mean),
            "delta_pp": float(f1_interval["observed_delta_pp"]) if f1_interval else None,
            "lower_pp": f1_interval.get("lower_pp") if f1_interval else None,
            "upper_pp": f1_interval.get("upper_pp") if f1_interval else None,
            "n": n,
            "split": split,
            "compatible": compatible,
            "reason": reason,
        }
    )

    sup_interval = _metric_interval(comparison, "all_support_packed") if compatible else None
    cards.append(
        {
            "metric": "All-support coverage",
            "baseline": float(agg_a.all_support_packed),
            "candidate": float(agg_b.all_support_packed),
            "delta_pp": float(sup_interval["observed_delta_pp"]) if sup_interval else None,
            "lower_pp": sup_interval.get("lower_pp") if sup_interval else None,
            "upper_pp": sup_interval.get("upper_pp") if sup_interval else None,
            "n": n,
            "split": split,
            "compatible": compatible,
            "reason": reason,
        }
    )

    cards.append(
        {
            "metric": "Latency",
            "baseline_p50_ms": agg_a.latency_p50_ms,
            "candidate_p50_ms": agg_b.latency_p50_ms,
            "baseline_p95_ms": agg_a.latency_p95_ms,
            "candidate_p95_ms": agg_b.latency_p95_ms,
            "delta_pp": None,
            "n": n,
            "split": split,
            "compatible": compatible,
            "reason": reason,
        }
    )
    return cards


def format_score(value: float | None) -> str:
    """0–100 score with units; unmeasured stays explicit."""
    if value is None:
        return "Not measured"
    return f"{float(value):.2f}"


def format_delta(delta_pp: float | None, *, compatible: bool) -> str:
    """Paired delta in percentage points; incompatible shows the refusal."""
    if not compatible or delta_pp is None:
        return "No paired delta (incompatible runs)"
    sign = "+" if float(delta_pp) >= 0 else ""
    return f"{sign}{float(delta_pp):.2f}pp"


def format_interval(lower_pp: float | None, upper_pp: float | None, *, compatible: bool) -> str:
    """Uncertainty beside quality deltas; never celebratory or invented."""
    if not compatible or lower_pp is None or upper_pp is None:
        return "interval unavailable"
    return f"95% CI [{float(lower_pp):+.2f}, {float(upper_pp):+.2f}]"


def format_latency(value_ms: float | None) -> str:
    if value_ms is None:
        return "Not measured"
    return f"{float(value_ms):.1f} ms"


def filter_regression_rows(
    rows: list[dict[str, Any]], active: list[str] | set[str] | tuple[str, ...] | None
) -> list[dict[str, Any]]:
    """Filter the regression view; the saved denominator is unchanged.

    Keeps a row when any active filter appears in its labels (rows may carry
    multiple labels, e.g. Worsened + Failed). Empty/None actives return all
    rows in stable question order.
    """
    if not active:
        return sorted(list(rows), key=lambda row: str(row.get("question_id", "")))
    wanted = [str(item) for item in active]
    unknown = [item for item in wanted if item not in FILTERS]
    if unknown:
        raise ValueError(f"Unknown regression filters: {unknown}; expected {list(FILTERS)}")
    kept = [row for row in rows if any(label in set(row.get("labels") or []) for label in wanted)]
    return sorted(kept, key=lambda row: str(row.get("question_id", "")))


def split_rows_by_tab(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Multi-hop vs single-hop control tabs; controls never mix silently."""
    multi: list[dict[str, Any]] = []
    control: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("question_type", "")) == _TAB_CONTROL:
            control.append(row)
        else:
            multi.append(row)
    key = lambda row: str(row.get("question_id", ""))  # noqa: E731
    return {"multi_hop": sorted(multi, key=key), "control": sorted(control, key=key)}


def table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Full metrics table in column priority: Question → Baseline → Candidate.

    Baseline/candidate scores are packed-context answer F1 (0–100); the
    difference is candidate minus baseline in percentage points; status
    joins both arm states so failures stay visible.
    """
    ordered = sorted(rows, key=lambda row: str(row.get("question_id", "")))
    return [
        {
            "Question": str(row.get("question_id", "")),
            "Baseline score": float(row.get("f1_a", 0.0)),
            "Candidate score": float(row.get("f1_b", 0.0)),
            "Difference": float(row.get("delta_f1_pp", 0.0)),
            "Status": f"{row.get('status_a', '?')} → {row.get('status_b', '?')}",
            "Labels": ";".join(row.get("labels") or []),
        }
        for row in ordered
    ]


def failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Failure-analysis subset: any arm failed or abstained (fixed rows kept)."""
    ordered = sorted(rows, key=lambda row: str(row.get("question_id", "")))
    return [
        row
        for row in ordered
        if row.get("status_a") in ("failed", "insufficient_evidence")
        or row.get("status_b") in ("failed", "insufficient_evidence")
    ]


def cost_panels(
    query_costs: dict[str, Any] | None,
    indexing_costs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Separate query spend from one-time indexing spend (unknown stays unknown)."""
    query = dict(query_costs or {})
    indexing = dict(indexing_costs or {})

    def _query_line(scope: str) -> str:
        entry = query.get(scope) if isinstance(query.get(scope), dict) else {}
        tokens_in = entry.get("input_tokens", "?")
        tokens_out = entry.get("output_tokens", "?")
        money = format_cost_usd(entry.get("estimated_cost_usd"))
        return f"{scope}: {tokens_in} in / {tokens_out} out tokens · {money}"

    def _index_line(scope: str) -> str:
        entry = indexing.get(scope) if isinstance(indexing.get(scope), dict) else None
        if entry is None:
            return f"{scope}: unknown (unpriced local compute)"
        money = format_cost_usd(entry.get("estimated_cost_usd"))
        note = str(entry.get("note", "unpriced local compute"))
        return f"{scope}: {money} · {note}"

    return {
        "query": [_query_line("baseline"), _query_line("candidate")],
        "indexing": [_index_line("baseline"), _index_line("candidate")],
    }


def render_headlines(cards: list[dict[str, Any]]) -> None:
    import streamlit as st

    for card in cards:
        st.subheader(card["metric"])
        if card["metric"] == "Latency":
            st.write(
                f"Baseline p50 {format_latency(card.get('baseline_p50_ms'))} · "
                f"candidate p50 {format_latency(card.get('candidate_p50_ms'))}"
            )
            st.write(
                f"Baseline p95 {format_latency(card.get('baseline_p95_ms'))} · "
                f"candidate p95 {format_latency(card.get('candidate_p95_ms'))}"
            )
            st.caption(f"n={card['n']} ({card['split']}) · observed only, no paired interval")
            continue
        st.write(
            f"Baseline {format_score(card.get('baseline'))} · "
            f"candidate {format_score(card.get('candidate'))} · "
            f"{format_delta(card.get('delta_pp'), compatible=bool(card.get('compatible')))}"
        )
        if card.get("compatible"):
            st.caption(
                f"{format_interval(card.get('lower_pp'), card.get('upper_pp'), compatible=True)} "
                f"· n={card['n']} ({card['split']})"
            )
        else:
            st.caption(f"n={card['n']} ({card['split']}) · {card.get('reason')}")


def render_cost_panels(panels: dict[str, Any]) -> None:
    import streamlit as st

    st.subheader("Query cost (per evaluation run)")
    for line in panels.get("query", []):
        st.write(line)
    st.subheader("Indexing cost (one-time build)")
    for line in panels.get("indexing", []):
        st.write(line)
    st.caption("Estimated money carries units; unpriced local compute stays unknown, never $0.")


__all__ = [
    "CONFIGURE_LABEL",
    "EMPTY_HINT",
    "EMPTY_TITLE",
    "FILTERS",
    "compatibility_banner",
    "cost_panels",
    "empty_state",
    "failure_rows",
    "filter_regression_rows",
    "format_delta",
    "format_interval",
    "format_latency",
    "format_score",
    "headline_cards",
    "render_cost_panels",
    "render_headlines",
    "split_rows_by_tab",
    "table_rows",
]
