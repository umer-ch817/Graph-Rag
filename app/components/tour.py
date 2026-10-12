"""Guided demo tour: a small overlay that walks first-run users page by page.

Pure step data plus helpers (no Streamlit import at module top) so unit
tests can assert the contract without the UI dependency. ``render_tour``
imports Streamlit lazily and is called once per page, right after
``render_page_chrome``. Text is deliberately very simple English so a
non-technical user can follow along.
"""

from __future__ import annotations

TOUR_STEPS: tuple[dict[str, str], ...] = (
    {
        "id": "overview",
        "page": "app/app.py",
        "menu": "app (Overview)",
        "title": "Step 1: Home (Overview)",
        "body": (
            "This is the home page. 'Next step' tells you what to do next, "
            "and the button below takes you to the right page. Press it to "
            "take the first step."
        ),
    },
    {
        "id": "dataset",
        "page": "pages/dataset.py",
        "menu": "dataset",
        "title": "Step 2: Prepare the question set",
        "body": (
            "Under number 1 'Dataset', press 'Prepare dataset'. This builds "
            "the question set. Later steps stay blocked until this is ready."
        ),
    },
    {
        "id": "vector",
        "page": "pages/dataset.py",
        "menu": "dataset",
        "title": "Step 3: Build the vector index",
        "body": (
            "Under number 2 'Vector index', press 'Build vector index', then "
            "run the worker. This builds the text search index."
        ),
    },
    {
        "id": "graph",
        "page": "pages/dataset.py",
        "menu": "dataset",
        "title": "Step 4: Build the graph",
        "body": (
            "Under number 3 'Graph pilot & build', press 'Enqueue graph "
            "build'. This extracts names and relations into a graph so "
            "answers come with evidence."
        ),
    },
    {
        "id": "compare",
        "page": "pages/compare.py",
        "menu": "compare",
        "title": "Step 5: Ask a question, see the comparison",
        "body": (
            "On the 'compare' page, type your question and run it. Both "
            "methods — vector and graph — show their answers side by side."
        ),
    },
    {
        "id": "evidence",
        "page": "pages/evidence.py",
        "menu": "evidence",
        "title": "Step 6: Check the evidence",
        "body": (
            "On the 'evidence' page, see which sentence each answer came "
            "from. A cited line means the answer is backed by that evidence."
        ),
    },
    {
        "id": "evaluations",
        "page": "pages/evaluations.py",
        "menu": "evaluations",
        "title": "Step 7: See the score — who won?",
        "body": (
            "On the 'evaluations' page, the numbers show which method "
            "performed better. That is the full demo!"
        ),
    },
)


def tour_step_count() -> int:
    """Number of guided steps."""
    return len(TOUR_STEPS)


def get_tour_step(index: int) -> dict[str, str]:
    """Return the step at ``index``, clamped to the valid range."""
    if index < 0:
        index = 0
    if index >= len(TOUR_STEPS):
        index = len(TOUR_STEPS) - 1
    return TOUR_STEPS[index]


def clamp_step(index: int) -> int:
    """Keep a step index inside the valid range."""
    return max(0, min(int(index), len(TOUR_STEPS) - 1))


def render_tour(current_page: str) -> None:
    """Sidebar launcher plus floating step card. Call once per page."""
    import streamlit as st

    if "tour_open" not in st.session_state:
        st.session_state["tour_open"] = False
    if "tour_step" not in st.session_state:
        st.session_state["tour_step"] = 0

    with st.sidebar:
        st.markdown("---")
        if st.session_state["tour_open"]:
            if st.button("⏹ Close demo", key="tour-close-side"):
                st.session_state["tour_open"] = False
                st.rerun()
        else:
            if st.button("🧭 Guided demo: walk me through", key="tour-open-side"):
                st.session_state["tour_open"] = True
                st.session_state["tour_step"] = 0
                st.rerun()

    if not st.session_state["tour_open"]:
        return

    index = clamp_step(st.session_state.get("tour_step", 0))
    step = get_tour_step(index)
    total = tour_step_count()
    on_target = step["page"] == current_page
    status = (
        "✅ You are in the right place — look above."
        if on_target
        else f"👉 Click '{step['menu']}' in the left menu, or press the button below."
    )
    st.markdown(
        "<style>"
        ".tour-card { position: fixed; right: 18px; bottom: 18px; width: 330px; "
        "z-index: 9999; background: #FFFFFF; color: #18251F; "
        "border: 2px solid #176348; border-radius: 12px; padding: 14px 16px; "
        "box-shadow: 0 4px 18px rgba(0,0,0,0.25); font-size: 14px; }"
        ".tour-card h4 { margin: 0 0 6px 0; color: #176348; }"
        ".tour-step { font-size: 12px; color: #4C5E54; }"
        "</style>"
        f"<div class='tour-card'><div class='tour-step'>Demo step {index + 1}/{total}</div>"
        f"<h4>{step['title']}</h4><div>{step['body']}</div>"
        f"<div style='margin-top:8px'>{status}</div></div>",
        unsafe_allow_html=True,
    )
    left, mid, right = st.columns(3)
    with left:
        if st.button("← Back", key="tour-prev", disabled=index <= 0):
            st.session_state["tour_step"] = index - 1
            st.rerun()
    with mid:
        if st.button("Next →", key="tour-next", disabled=index >= total - 1):
            st.session_state["tour_step"] = index + 1
            st.rerun()
    with right:
        if not on_target:
            if st.button("Go there ➜", key="tour-goto"):
                st.switch_page(step["page"])


__all__ = ["TOUR_STEPS", "clamp_step", "get_tour_step", "render_tour", "tour_step_count"]
