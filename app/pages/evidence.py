"""Evidence: follow a claim to its source.

Full-page review of saved runs: source list plus reader, the accessible
relation table, and retrieval details (origins, fallback, caps). Switching
arms retains the selected source when it exists in both. Returning restores
the prior question and run context. Opening a saved result never reruns it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.sources import (
    reader_blocks,
    reader_blocks_for_citation,
    render_reader,
    render_relation_table,
    render_source_list,
)
from app.components.status import render_footer, render_page_chrome
from app.components.tour import render_tour
from graphrag_lab.compare_flow import (
    arm_label,
    build_sources,
    fallback_text,
    load_evidence,
    relation_rows,
    resolve_citation,
)
from graphrag_lab.jobs import DEFAULT_DB_PATH
from graphrag_lab.run_store import RunStore

DEFAULT_CONFIG = "configs/default.yaml"


def _db_path() -> str:
    import streamlit as st

    return str(st.session_state.get("db_path", DEFAULT_DB_PATH))


def _context() -> dict[str, Any] | None:
    import streamlit as st

    params = st.query_params
    query_runs = params.get_all("run")
    query_src = params.get("src")
    if query_runs:
        return {
            "run_ids": query_runs,
            "question": "(saved result)",
            "source_id": query_src,
            "return_to": None,
        }
    stored = st.session_state.get("evidence_context")
    if isinstance(stored, dict) and stored.get("run_ids"):
        return stored
    # Fall back to the latest compare session so refresh keeps working.
    result = st.session_state.get("compare_result")
    if isinstance(result, dict) and result.get("run_ids"):
        return {
            "run_ids": list(result["run_ids"].values()),
            "question": result.get("question", ""),
            "source_id": None,
            "return_to": "compare",
        }
    return None


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="GraphRAG Lab — Evidence", layout="wide")
    render_page_chrome(title="Evidence", subtitle="Follow a claim to its source sentences.")
    render_tour("pages/evidence.py")
    context = _context()
    if context is None:
        st.info("No answer selected yet. Run a comparison first.")
        if st.button("Go to Ask & Compare", key="evidence-to-compare"):
            st.switch_page("pages/compare.py")
        return

    store = RunStore(_db_path())
    bundles: dict[str, dict[str, Any]] = {}
    for run_id in context["run_ids"]:
        try:
            bundles[run_id] = load_evidence(store, run_id)
        except ValueError as exc:
            st.error(str(exc))
            return
    arms = sorted({bundle["strategy"] for bundle in bundles.values()})
    st.caption(f"Question: {context.get('question', '')}")
    arm = st.radio(
        "Answer under review",
        arms,
        format_func=arm_label,
        key="evidence-arm",
        horizontal=True,
    )
    bundle = next(item for item in bundles.values() if item["strategy"] == arm)
    st.subheader(arm_label(arm))
    if bundle["status"] == "failed":
        st.error(
            f"This answer failed ({bundle.get('failure_code') or 'PROVIDER_ERROR'}). "
            "The request is retained; retry from Ask & Compare."
        )
    elif bundle["status"] == "insufficient_evidence":
        st.warning("This answer abstained: insufficient evidence.")
    elif bundle["answer"]:
        st.write(bundle["answer"])

    banner = fallback_text(
        fallback=bundle["fallback"],
        reason=bundle["fallback_reason"],
        cap_hits=bundle["cap_hits"],
    )
    if banner:
        st.info(banner)
    if bundle["degraded"]:
        st.info("Reranking degraded for this answer: showing pre-rerank order.")

    sources = build_sources(bundle["chunks"], bundle["packed_ids"], bundle["citations"])
    st.subheader("Sources")
    left, right = st.columns([1, 2])
    requested = context.get("source_id")
    if requested and requested not in {source.chunk_id for source in sources}:
        st.caption(f"Source {requested} is not in this arm; showing all sources.")
        requested = None
    selected = st.session_state.get("evidence_source") or requested
    if selected not in {source.chunk_id for source in sources}:
        selected = sources[0].chunk_id if sources else None
    with left:
        picked = render_source_list(sources, selected_id=selected, key_prefix="evidence")
        if picked:
            st.session_state["evidence_source"] = picked
            selected = picked
    with right:
        current = next((s for s in sources if s.chunk_id == selected), None)
        if current is None:
            st.write("Select a source to read it.")
        else:
            cited = next(
                (
                    c.get("sentence_ids", [])
                    for c in bundle["citations"]
                    if c.get("chunk_id") == current.chunk_id
                ),
                [],
            )
            try:
                resolved = resolve_citation(
                    {
                        "chunk_id": current.chunk_id,
                        "sentence_ids": cited or current.sentence_ids[:1],
                    },
                    sources,
                )
                render_reader(resolved["source"], reader_blocks_for_citation(resolved))
            except LookupError:
                render_reader(current, reader_blocks(current, [int(v) for v in cited]))

    st.subheader("Connections")
    st.caption("Retrieved connections (evidence, not model reasoning).")
    chunks_by_id = {chunk.get("chunk_id", chunk.get("id", "")): chunk for chunk in bundle["chunks"]}
    rows = relation_rows(bundle["traversed"], chunks_by_id)
    picked_evidence = render_relation_table(rows, key_prefix="evidence")
    if picked_evidence and picked_evidence in chunks_by_id:
        st.session_state["evidence_source"] = picked_evidence
        st.rerun()

    with st.expander("Retrieval details"):
        st.write(f"Sources retrieved: {len(bundle['chunks'])}")
        st.write(f"Sources in final context: {len(bundle['packed_ids'])}")
        st.write(f"Latency: {bundle.get('latency_ms') or 'unknown'} ms")
        if bundle["cap_hits"]:
            st.write("Traversal caps hit: " + ", ".join(sorted(set(bundle["cap_hits"]))))
        st.write(f"Origin labels: {[s.origin for s in sources][:10]}")

    if context.get("return_to") == "compare" or st.session_state.get("compare_result"):
        if st.button("Back to results", key="evidence-back"):
            st.switch_page("pages/compare.py")
    render_footer(corpus="see Overview", index="see Dataset & Index", models="see Overview footer")


if __name__ == "__main__":
    main()
