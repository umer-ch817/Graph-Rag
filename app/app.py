"""GraphRAG Lab Overview: one state-dependent next action plus recent runs.

First-run users land here until vectors are ready; afterwards the page points
at Ask & Compare. All readiness logic lives in ``graphrag_lab.setup_flow``;
this module only presents it with the shared theme tokens.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.components.status import render_footer, render_page_chrome, render_status_pill
from graphrag_lab.config import load_config
from graphrag_lab.jobs import DEFAULT_DB_PATH, JobStore
from graphrag_lab.run_store import RunStore
from graphrag_lab.setup_flow import neo4j_message, recent_runs, summarize_setup, worker_status

DEFAULT_CONFIG = "configs/default.yaml"


def _db_path() -> str:
    import streamlit as st

    return str(st.session_state.get("db_path", DEFAULT_DB_PATH))


def _latest_heartbeat(store: JobStore) -> str | None:
    conn = sqlite3.connect(str(store.path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT MAX(heartbeat_at) AS beat FROM job_attempts").fetchone()
        return str(row["beat"]) if row is not None and row["beat"] else None
    finally:
        conn.close()


def _neo4j_ok() -> tuple[bool, str]:
    import streamlit as st

    cached = st.session_state.get("neo4j_check")
    if isinstance(cached, dict) and cached.get("ok") in (True, False):
        return bool(cached["ok"]), str(cached.get("detail", ""))
    try:
        config = load_config(st.session_state.get("config_path", DEFAULT_CONFIG))
    except Exception as exc:
        return False, f"configuration invalid: {exc}"
    if config.neo4j_password is None:
        return False, "GRAPHRAG_NEO4J_PASSWORD is not set"
    try:
        from graphrag_lab.neo4j_store import create_driver, verify_connectivity

        driver = create_driver(config)
        try:
            verify_connectivity(driver, config.neo4j.database)
        finally:
            driver.close()
    except Exception as exc:
        detail = " ".join(str(exc).split())[:200]
        return False, detail
    return True, ""


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="GraphRAG Lab — Overview", layout="wide")
    store = JobStore(_db_path())
    runs = RunStore(_db_path())
    try:
        config = load_config(st.session_state.get("config_path", DEFAULT_CONFIG))
        models_line = (
            f"embed `{config.embedding_model}` · rerank `{config.reranker_model}` · "
            f"llm `{config.llm_model or 'unset'}`"
        )
        neo4j_uri = config.neo4j.uri
    except Exception:
        models_line = "configuration invalid; see Dataset & Index"
        neo4j_uri = "bolt://127.0.0.1:7687"
    ok, detail = _neo4j_ok()
    beat = _latest_heartbeat(store)
    worker = worker_status(beat)
    summary = summarize_setup(
        store,
        worker_seen=bool(worker["seen"]),
        worker_message=str(worker["message"]),
        neo4j_ok=ok,
        neo4j_uri=neo4j_uri,
        neo4j_detail=detail,
    )
    first_run = summary.dataset.state == "missing"
    render_page_chrome(
        title="GraphRAG Lab",
        subtitle=(
            "Start with a reproducible question set."
            if first_run
            else "Your vector index is ready."
            if summary.vector.state == "ready"
            else ("Continue setup to reach the comparison workspace.")
        ),
    )
    st.subheader("Next step")
    action = summary.next_action
    st.write(f"{action.reason}")
    if st.button(action.label, type="primary", key="overview-next"):
        if action.target == "dataset":
            st.switch_page("pages/dataset.py")
        elif action.target == "compare":
            st.switch_page("pages/compare.py")
        else:
            st.session_state["pending_target"] = action.target
            st.info("That destination is not implemented yet; setup state is saved.")
    st.subheader("Progress")
    for step in (summary.dataset, summary.vector, summary.graph, summary.ready):
        with st.container():
            left, right = st.columns([3, 1])
            with left:
                st.write(f"**{step.name}** — {step.headline}")
                st.caption(step.reason)
            with right:
                render_status_pill(step.state)
    st.subheader("Worker and database")
    st.write(worker["message"])
    st.write(neo4j_message(neo4j_uri, ok=ok, detail=detail))
    st.subheader("Recent runs")
    rows = recent_runs(runs, limit=5)
    if not rows:
        st.write("No evaluation results yet.")
    else:
        st.dataframe(rows, use_container_width=True)
    with st.expander("Dataset statistics"):
        st.json(summary.dataset.model_dump(mode="json"))
    render_footer(
        corpus=summary.dataset.headline,
        index=summary.ready.headline,
        models=models_line,
    )


if __name__ == "__main__":
    main()
