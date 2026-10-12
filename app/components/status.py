"""Shared status and progress presentation (UI-UX.md revision 2).

Pure helpers (no Streamlit import at module top) build the exact strings the
e2e suite asserts. ``render_*`` wrappers import Streamlit lazily so importing
this module never requires the UI dependency.
"""

from __future__ import annotations

from typing import Any

from app.components.theme import TOKENS, base_css, pill_text

STATE_TONE: dict[str, str] = {
    "ready": TOKENS["action"],
    "failed": TOKENS["danger"],
    "paused_budget": TOKENS["warning"],
    "cancelled": TOKENS["text_secondary"],
    "interrupted": TOKENS["warning"],
    "running": TOKENS["vector"],
    "queued": TOKENS["text_secondary"],
    "missing": TOKENS["text_secondary"],
    "blocked": TOKENS["text_secondary"],
    "cancelling": TOKENS["warning"],
    "partial": TOKENS["warning"],
}


def progress_line(processed: int, total: int, *, failed: int = 0) -> str:
    """Honest counters; never a fabricated percentage or ETA."""
    if total <= 0:
        return f"{processed} items processed (total unknown)"
    line = f"{processed}/{total} chunks processed"
    if failed:
        line += f" · {failed} failed"
    return line


def spend_line(
    *,
    budget_calls: int | None,
    budget_tokens: int | None,
    budget_cost_usd: float | None,
    used_items: int,
    cost_label: str,
) -> str:
    """Budget caps beside actual spend; money and tokens carry their units."""
    caps: list[str] = []
    if budget_calls is not None:
        caps.append(f"{budget_calls} calls")
    if budget_tokens is not None:
        caps.append(f"{budget_tokens} tokens")
    if budget_cost_usd is not None:
        caps.append(f"${budget_cost_usd:.2f}")
    cap_text = ", ".join(caps) if caps else "no caps set"
    return f"Budget: {cap_text} · used items: {used_items} · cost: {cost_label}"


def job_error_text(job: Any) -> str:
    """Understandable failure line: what failed, what is saved, next action."""
    state = getattr(job, "state", "?")
    code = getattr(job, "error_code", None)
    message = getattr(job, "error_message", None)
    done = getattr(job, "progress_done", 0)
    total = getattr(job, "progress_total", 0)
    detail = message or code or "see the job record"
    next_step = {
        "failed": "Retry failed items; settings are preserved.",
        "cancelled": "Resume from the checkpoint; validated work is kept.",
        "interrupted": "Restart the worker, then resume; checkpoints survive.",
        "paused_budget": "Revise the budget, then resume.",
    }.get(state, "Inspect the job record and retry.")
    return (
        f"Job {getattr(job, 'id', '?')} {state}: {detail} ({done}/{total} items kept). {next_step}"
    )


def render_page_chrome(*, title: str, subtitle: str) -> None:
    """Shared page header plus the token CSS. Call once per page."""
    import streamlit as st

    st.markdown(base_css(), unsafe_allow_html=True)
    st.title(title)
    st.write(subtitle)


def render_status_pill(state: str) -> None:
    import streamlit as st

    tone = STATE_TONE.get(state, TOKENS["text_secondary"])
    st.markdown(
        f"<span class='lab-pill' style='color:{tone};border-color:{tone}'>"
        f"{pill_text(state)}</span>",
        unsafe_allow_html=True,
    )


def render_progress_panel(panel: dict[str, Any], spend: str) -> None:
    import streamlit as st

    st.markdown(
        "<div class='lab-panel'>"
        f"<strong>{panel['kind']}</strong> "
        f"<span class='lab-pill'>{pill_text(str(panel['state']))}</span><br/>"
        f"{progress_line(panel['processed'], panel['total'], failed=panel['failed'])}"
        f" · attempts: {panel['attempts']}<br/>"
        f"<span class='lab-meta'>{spend}</span>"
        + (
            f"<br/><span class='lab-error'>{panel['error_message']}</span>"
            if panel.get("error_message")
            else ""
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def render_footer(*, corpus: str, index: str, models: str) -> None:
    import streamlit as st

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div class='lab-meta'>Corpus: {corpus}<br/>Index: {index}<br/>{models}</div>",
        unsafe_allow_html=True,
    )


__all__ = [
    "STATE_TONE",
    "job_error_text",
    "progress_line",
    "render_footer",
    "render_page_chrome",
    "render_progress_panel",
    "render_status_pill",
    "spend_line",
]
