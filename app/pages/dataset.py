"""Dataset & Index: guided preparation (Dataset → Vector → Graph → Ready).

Each section has one main action with prerequisites inline. Indexing shows
one persisted progress panel (processed/total, attempts, failures, budget);
Cancel is secondary. After a failure settings are preserved and the button
becomes "Retry failed chunks". Submissions reuse a session idempotency key so
Streamlit reruns and double clicks never create duplicate active jobs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.status import (
    job_error_text,
    render_footer,
    render_page_chrome,
    render_progress_panel,
    render_status_pill,
    spend_line,
)
from graphrag_lab.config import load_config
from graphrag_lab.jobs import DEFAULT_DB_PATH, JobStore, get_job_status
from graphrag_lab.setup_flow import (
    config_change_creates_snapshot,
    enqueue_index_once,
    idempotency_key_for,
    latest_dataset_id,
    progress_panel,
    spend_summary,
    summarize_setup,
    worker_status,
)

DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_ARTIFACTS = "artifacts/dataset"


def _db_path() -> str:
    import streamlit as st

    return str(st.session_state.get("db_path", DEFAULT_DB_PATH))


def _submit_key(slot: str, *parts: str) -> str:
    """One stable key per logical submission; reruns reuse it (no duplicates)."""
    import streamlit as st

    key = idempotency_key_for(slot, *parts)
    prior = st.session_state.get(f"submit-{slot}")
    if isinstance(prior, str) and prior:
        return prior
    st.session_state[f"submit-{slot}"] = key
    return key


def _section_head(number: int, title: str, state: str, reason: str) -> None:
    import streamlit as st

    st.subheader(f"{number}. {title}")
    render_status_pill(state)
    st.caption(reason)


def _prepare_dataset(sample_size: int, tuning_size: int, seed: int, output: str) -> dict[str, Any]:
    from graphrag_lab.corpus import ensure_source, prepare_dataset
    from graphrag_lab.models import HOTPOT_URL, SampleConfig

    source = Path("data/raw/hotpot_dev_distractor_v1.json")
    if not source.is_file():
        ensure_source(source, HOTPOT_URL)
    manifest = prepare_dataset(
        source,
        Path(output),
        SampleConfig(sample_size=sample_size, tuning_size=tuning_size, seed=seed),
    )
    return manifest.model_dump(mode="json")


def section_dataset(store: JobStore) -> None:
    import streamlit as st

    summary = summarize_setup(store)
    _section_head(1, "Dataset", summary.dataset.state, summary.dataset.reason)
    st.write("Defaults are visible; advanced sampling and model settings are below.")
    with st.form("dataset-form"):
        sample_size = st.number_input("Sample size", min_value=100, max_value=200, value=150)
        seed = st.number_input("Seed", min_value=0, value=42)
        tuning_size = st.number_input("Tuning split", min_value=1, value=30)
        submitted = st.form_submit_button(
            "Prepare dataset", disabled=bool(st.session_state.get("preparing"))
        )
    with st.expander("Advanced sampling and model settings"):
        st.write("Embedding and reranker models come from `configs/default.yaml`.")
        try:
            config = load_config(st.session_state.get("config_path", DEFAULT_CONFIG))
            st.code(
                f"embedding_model: {config.embedding_model}\n"
                f"reranker_model: {config.reranker_model}\n"
                f"budget: calls={config.budget.max_calls} "
                f"tokens={config.budget.max_total_tokens} "
                f"cost={config.budget.max_cost_usd}",
                language="yaml",
            )
        except Exception as exc:
            st.error(f"Configuration invalid: {exc}")
        st.caption("Changing a fingerprinted setting creates a new snapshot.")
    if submitted:
        if int(tuning_size) >= int(sample_size):
            st.error("Tuning split must be smaller than the sample size. Nothing was prepared.")
            return
        st.session_state["preparing"] = True
        try:
            with st.spinner("Preparing seeded sample…"):
                manifest = _prepare_dataset(
                    int(sample_size), int(tuning_size), int(seed), DEFAULT_ARTIFACTS
                )
            store.create_dataset(
                dataset_id=str(manifest["corpus_id"]),
                source=str(manifest["source_url"]),
                checksum=str(manifest["source_sha256"]),
                seed=int(seed),
                sample_count=int(sample_size),
                manifest_path=f"{DEFAULT_ARTIFACTS}/manifest.json",
            )
            st.session_state["active_dataset"] = str(manifest["corpus_id"])
            st.success(f"Dataset {manifest['corpus_id']} ready: {sample_size} questions.")
            st.json(
                {
                    "corpus_id": manifest["corpus_id"],
                    "splits": manifest["split_type_counts"],
                    "artifacts": manifest["artifact_sha256"],
                }
            )
        except Exception as exc:
            st.error(
                "Preparation failed: "
                + " ".join(str(exc).split())[:400]
                + " Nothing was overwritten; fix the cause and retry."
            )
        finally:
            st.session_state["preparing"] = False


def _ensure_snapshot(store: JobStore, snapshot_id: str, dataset_id: str | None) -> dict[str, Any]:
    existing = store.get_snapshot(snapshot_id)
    if existing is not None:
        return existing
    import hashlib
    import json as _json

    fingerprint = hashlib.sha256(
        _json.dumps({"snapshot": snapshot_id, "dataset": dataset_id}, sort_keys=True).encode()
    ).hexdigest()[:16]
    return store.create_snapshot(
        snapshot_id=snapshot_id, dataset_id=dataset_id, fingerprint=f"local-{fingerprint}"
    )


def section_vector(store: JobStore) -> None:
    import streamlit as st

    summary = summarize_setup(store)
    vector = summary.vector
    _section_head(2, "Vector index", vector.state, vector.reason)
    dataset_id = st.session_state.get("active_dataset")
    snapshot_id = st.text_input(
        "Snapshot id", value=str(st.session_state.get("snapshot_id", "snap_local"))
    )
    st.session_state["snapshot_id"] = snapshot_id
    if summary.dataset.state != "ready":
        st.warning("Prepare a dataset first; vectors need corpus chunks.")
        return
    known = store.get_snapshot(snapshot_id)
    st.caption(
        "Creates a new snapshot."
        if config_change_creates_snapshot(
            known["fingerprint"] if known else None, f"pending-{snapshot_id}"
        )
        and known is None
        else f"Snapshot fingerprint: `{known['fingerprint']}`"
        if known
        else ""
    )
    with st.form("vector-form"):
        build = st.form_submit_button("Build vector index")
    if build:
        config = load_config(st.session_state.get("config_path", DEFAULT_CONFIG))
        snap = _ensure_snapshot(store, snapshot_id, dataset_id)
        key = _submit_key("vector", "index_vector", snap["id"], snap["fingerprint"])
        ref, submitted = enqueue_index_once(
            store,
            snapshot_id=snap["id"],
            stage="vector",
            budget=config.budget,
            idempotency_key=key,
            items=[{"id": "build", "dataset_dir": DEFAULT_ARTIFACTS}],
        )
        if not submitted:
            st.info(f"Already submitted: job {ref.job_id} is {ref.state}. No duplicate created.")
        else:
            st.success(f"Enqueued vector job {ref.job_id}. Start the worker to run it.")
    _job_panel(store, kind="index_vector", slot="vector")


def section_graph(store: JobStore) -> None:
    import streamlit as st

    summary = summarize_setup(store)
    graph = summary.graph
    _section_head(3, "Graph pilot & build", graph.state, graph.reason)
    if summary.vector.state != "ready":
        st.warning("Build the vector index first; the graph needs resolved chunks.")
    st.write(
        "Pilot rule: audit 20 chunks before any full extraction run. "
        "The pilot summary separates the estimated full-build cost from actual spend."
    )
    snapshot_id = str(st.session_state.get("snapshot_id", "snap_local"))
    with st.form("graph-form"):
        build = st.form_submit_button(
            "Enqueue graph build", disabled=summary.vector.state != "ready"
        )
    if build:
        config = load_config(st.session_state.get("config_path", DEFAULT_CONFIG))
        snap = _ensure_snapshot(store, snapshot_id, st.session_state.get("active_dataset"))
        key = _submit_key("graph", "index_graph", snap["id"], snap["fingerprint"])
        ref, submitted = enqueue_index_once(
            store,
            snapshot_id=snap["id"],
            stage="graph",
            budget=config.budget,
            idempotency_key=key,
            items=[{"id": "build", "dataset_dir": DEFAULT_ARTIFACTS}],
        )
        if not submitted:
            st.info(f"Already submitted: job {ref.job_id} is {ref.state}. No duplicate created.")
        else:
            st.success(f"Enqueued graph job {ref.job_id}. Audit the pilot before scaling.")
    _job_panel(store, kind="index_graph", slot="graph")


def _job_panel(store: JobStore, *, kind: str, slot: str) -> None:
    import streamlit as st

    jobs = [job for job in store.list_jobs(limit=50) if job.kind == kind]
    if not jobs:
        st.caption("No jobs yet for this stage.")
        return
    job = sorted(jobs, key=lambda item: item.created_at, reverse=True)[0]
    attempts = store.get_attempts(job.id)
    panel = progress_panel(job, attempts=len(attempts))
    spend = spend_summary(job)
    render_progress_panel(
        panel,
        spend_line(
            budget_calls=spend.budget_calls,
            budget_tokens=spend.budget_tokens,
            budget_cost_usd=spend.budget_cost_usd,
            used_items=spend.used_items,
            cost_label=spend.cost_label,
        ),
    )
    if job.state in ("failed", "cancelled", "interrupted", "paused_budget"):
        st.caption(job_error_text(job))
    left, right = st.columns(2)
    with left:
        if st.button("Refresh status", key=f"{slot}-refresh"):
            st.rerun()
    with right:
        if job.state in ("queued", "running"):
            if st.button("Cancel", key=f"{slot}-cancel"):
                store.cancel_job(job.id)
                st.rerun()
        elif job.state in ("failed", "cancelled", "interrupted", "paused_budget"):
            label = "Retry failed chunks" if job.state == "failed" else "Resume"
            if st.button(label, key=f"{slot}-resume"):
                store.resume_job(job.id)
                st.rerun()
    with st.expander("Details"):
        st.code(f"job_id: {job.id}\nkind: {job.kind}\nsnapshot: {job.snapshot_id or '-'}")
        st.json(get_job_status(store, job.id)["checkpoint"])


def section_ready(store: JobStore) -> None:
    import streamlit as st

    summary = summarize_setup(store)
    ready = summary.ready
    _section_head(4, "Ready to compare", ready.state, ready.reason)
    if ready.state == "ready":
        if st.button("Ask a question", type="primary", key="ready-ask"):
            st.info("Ask & Compare arrives with T12c; setup state is saved.")
    else:
        st.caption("Complete the dataset and vector steps above first.")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="GraphRAG Lab — Dataset & Index", layout="wide")
    render_page_chrome(
        title="Dataset & Index",
        subtitle="Guided preparation: dataset, vectors, graph pilot, then compare.",
    )
    store = JobStore(_db_path())
    if not st.session_state.get("active_dataset"):
        st.session_state["active_dataset"] = latest_dataset_id(store)
    beat = None
    try:
        import sqlite3

        conn = sqlite3.connect(str(store.path), timeout=10.0)
        try:
            row = conn.execute("SELECT MAX(heartbeat_at) AS beat FROM job_attempts").fetchone()
            beat = row[0] if row else None
        finally:
            conn.close()
    except Exception:
        beat = None
    worker = worker_status(str(beat) if beat else None)
    if not worker["seen"]:
        st.warning(worker["message"])
    section_dataset(store)
    section_vector(store)
    section_graph(store)
    section_ready(store)
    render_footer(corpus="see Overview", index="see Overview", models="models in Overview footer")


if __name__ == "__main__":
    main()
