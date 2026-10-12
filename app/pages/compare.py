"""Ask & Compare: run two retrieval arms side by side, inspect citations.

Question and settings persist after completion. Example selection edits the
draft only and never starts a request. Each arm renders as it finishes with
its own stage indicator; citations open the inline inspector without
discarding answers. Full review lives on the Evidence page with the same
run/source state. Gold answers are never shown here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.sources import reader_blocks_for_citation, render_reader
from app.components.status import render_footer, render_page_chrome
from app.components.tour import render_tour
from graphrag_lab.compare_flow import (
    ARM_LABELS,
    DEFAULT_PAIR,
    arm_label,
    assert_matched_budgets,
    build_sources,
    describe_arm,
    encode_deep_link,
    fallback_text,
    load_panels,
    persist_compare,
    question_id_for,
    resolve_citation,
    tuning_examples,
    validate_arm_pair,
)
from graphrag_lab.config import load_config
from graphrag_lab.jobs import DEFAULT_DB_PATH
from graphrag_lab.run_store import RunStore

DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_DATASET_DIR = "artifacts/dataset"
ANSWER_BUDGETS = {"max_chunks": 8, "max_context_tokens": 3000}


def _db_path() -> str:
    import streamlit as st

    return str(st.session_state.get("db_path", DEFAULT_DB_PATH))


def _session_defaults() -> None:
    import streamlit as st

    st.session_state.setdefault("compare_draft", "")
    st.session_state.setdefault("compare_left", DEFAULT_PAIR[0])
    st.session_state.setdefault("compare_right", DEFAULT_PAIR[1])
    st.session_state.setdefault("compare_result", None)
    st.session_state.setdefault("compare_inspector", None)


def _corpus_id() -> str | None:
    import json as _json

    manifest = Path(DEFAULT_DATASET_DIR) / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        return str(_json.loads(manifest.read_text(encoding="utf-8")).get("corpus_id"))
    except (OSError, ValueError):
        return None


def _get_backend(model_name: str) -> Any:
    import streamlit as st

    cached = st.session_state.get("embed_backend")
    if cached is not None and getattr(cached, "model_name", "") == model_name:
        return cached
    from graphrag_lab.embeddings import SentenceTransformerBackend

    backend = SentenceTransformerBackend(model_name)
    _ = backend.dimension  # load now so model errors surface before retrieval
    st.session_state["embed_backend"] = backend
    return backend


def _get_reranker(model_name: str) -> Any:
    import streamlit as st

    cached = st.session_state.get("rerank_backend")
    if cached is not None and getattr(cached, "model_name", "") == model_name:
        return cached
    from graphrag_lab.reranking import CrossEncoderBackend

    reranker = CrossEncoderBackend(model_name)
    st.session_state["rerank_backend"] = reranker
    return reranker


def _run_arm(
    strategy: str,
    question: str,
    corpus_id: str,
    config: Any,
    driver: Any,
    database: str,
    stage: Any,
) -> dict[str, Any]:
    """Execute one arm: retrieve → rerank → pack → answer. Raises on failure."""
    from graphrag_lab.answering import (
        AnswerConfig,
        answer_question,
        answer_question_with_triples,
        pack_context,
        pack_context_with_triples,
    )
    from graphrag_lab.embeddings import build_embedding_text  # noqa: F401
    from graphrag_lab.hybrid import (
        HybridConfig,
        hybrid_retrieve_with_vector,
        rerank_hybrid_chunks,
        traversed_relations_for_result,
    )
    from graphrag_lab.llm import backend_from_config
    from graphrag_lab.reranking import RerankConfig
    from graphrag_lab.retrieval import RetrievalConfig, rerank_vector_chunks, retrieve_with_vector

    _ = build_embedding_text
    answer_config = AnswerConfig(
        max_chunks=ANSWER_BUDGETS["max_chunks"],
        max_context_tokens=ANSWER_BUDGETS["max_context_tokens"],
    )
    backend = _get_backend(config.embedding_model)
    query_vector = backend.embed_texts([question.strip()])[0]
    dimension = int(backend.dimension)
    started = time.perf_counter()
    chunks: list[dict[str, Any]] = []
    traversed: list[dict[str, Any]] = []
    fallback = False
    fallback_reason: str | None = None
    degraded = False
    cap_hits: list[str] = []
    reranked = False
    num_candidates = 0

    if strategy in ("vector", "vector_rerank"):
        stage("Retrieving evidence (vector)")
        result = retrieve_with_vector(
            question=question,
            query_vector=query_vector,
            corpus_id=corpus_id,
            embedding_model=config.embedding_model,
            dimension=dimension,
            driver=driver,
            database=database,
            config=RetrievalConfig(num_candidates=30),
        )
        chunks = [{**c.model_dump(), "origin": "vector"} for c in result.chunks]
        num_candidates = len(chunks)
        ordered = chunks
        if strategy == "vector_rerank":
            stage("Reranking")
            try:
                reranker = _get_reranker(config.reranker_model)
                ordered, rerank_result = rerank_vector_chunks(
                    question, chunks, reranker, RerankConfig()
                )
                degraded = bool(rerank_result.degraded)
                if degraded:
                    cap_hits.append(
                        f"rerank_degraded:{rerank_result.degraded_reason or 'reranker'}"
                    )
            except Exception:
                degraded = True
                cap_hits.append("rerank_degraded:backend_unavailable")
                ordered = chunks
            reranked = True
        stage("Packing context")
        packed = pack_context(question, ordered, answer_config)
        packed_ids = [c.chunk_id for c in packed.chunks]
        evidence_chunks = ordered
        stage("Writing answer")
        llm = backend_from_config(config)
        answer = answer_question(question, ordered, llm, answer_config)
    else:
        stage("Retrieving evidence (vector + graph)")
        hybrid_result = hybrid_retrieve_with_vector(
            question=question,
            query_vector=query_vector,
            corpus_id=corpus_id,
            embedding_model=config.embedding_model,
            dimension=dimension,
            driver=driver,
            database=database,
            hybrid_config=HybridConfig(),
        )
        chunks = [c.model_dump() for c in hybrid_result.chunks]
        num_candidates = hybrid_result.total_merged
        fallback = bool(hybrid_result.fallback_to_vector)
        fallback_reason = hybrid_result.fallback_reason
        cap_hits = list(hybrid_result.cap_hits)
        traversed = traversed_relations_for_result(hybrid_result)
        stage("Reranking")
        try:
            reranker = _get_reranker(config.reranker_model)
            ordered, rerank_result = rerank_hybrid_chunks(
                question, chunks, reranker, RerankConfig()
            )
            degraded = bool(rerank_result.degraded)
        except Exception:
            degraded = True
            cap_hits.append("rerank_degraded:backend_unavailable")
            ordered = chunks
        reranked = True
        stage("Packing context")
        llm = backend_from_config(config)
        if strategy == "hybrid_rerank_triples":
            packed = pack_context_with_triples(question, ordered, traversed, answer_config, 12)
            packed_ids = [c.chunk_id for c in packed.chunks]
            evidence_chunks = ordered
            stage("Writing answer")
            answer = answer_question_with_triples(
                question, ordered, traversed, llm, answer_config, None, 12
            )
        else:
            packed = pack_context(question, ordered, answer_config)
            packed_ids = [c.chunk_id for c in packed.chunks]
            evidence_chunks = ordered
            stage("Writing answer")
            answer = answer_question(question, ordered, llm, answer_config)

    latency_ms = (time.perf_counter() - started) * 1000.0
    usage = answer.usage.model_dump(mode="json") if answer.usage is not None else None
    return {
        "strategy": strategy,
        "answer": answer.answer,
        "status": answer.status,
        "citations": [c.model_dump(mode="json") for c in answer.citations],
        "latency_ms": round(latency_ms, 1),
        "retrieval": {
            "strategy": strategy,
            "num_candidates": num_candidates,
            "num_packed": len(packed_ids),
            "fallback": fallback,
            "fallback_reason": fallback_reason,
            "degraded": degraded,
            "cap_hits": cap_hits,
            "reranked": reranked,
            "chunks": evidence_chunks,
            "packed_ids": packed_ids,
            "traversed": traversed,
        },
        "usage": usage,
    }


def _failed_panel(strategy: str, exc: Exception) -> dict[str, Any]:
    code = getattr(exc, "code", "PROVIDER_ERROR") or "PROVIDER_ERROR"
    message = " ".join(str(exc).split())[:400] or code
    return {
        "strategy": strategy,
        "answer": "",
        "status": "failed",
        "citations": [],
        "latency_ms": None,
        "retrieval": {
            "strategy": strategy,
            "num_candidates": 0,
            "num_packed": 0,
            "fallback": False,
            "failure_message": message,
        },
        "failure_code": code,
    }


def _render_panel(panel: dict[str, Any], evidence: dict[str, Any], *, key: str) -> None:
    import streamlit as st

    label = arm_label(panel["strategy"])
    st.subheader(label)
    status = panel["status"]
    if status == "failed":
        st.error(
            f"Failed ({panel.get('failure_code', 'PROVIDER_ERROR')}): "
            f"{panel.get('retrieval', {}).get('failure_message', 'see Details')}. "
            "The request is retained; retry explicitly."
        )
    elif status == "insufficient_evidence":
        st.warning("Insufficient evidence to answer based on the provided context.")
    else:
        st.write(panel["answer"])
    citations = panel.get("citations", [])
    for number, citation in enumerate(citations, start=1):
        try:
            sources = build_sources(
                evidence["chunks"], evidence["packed_ids"], panel.get("citations", [])
            )
            resolved = resolve_citation(citation, sources)
            title = resolved["source"].title
            sids = ",".join(f"s{sid}" for sid in citation.get("sentence_ids", []))
            if st.button(f"Open source {number}: {title}, {sids}", key=f"{key}-cite-{number}"):
                st.session_state["compare_inspector"] = {
                    "arm": panel["strategy"],
                    "chunk_id": citation.get("chunk_id"),
                }
        except LookupError as exc:
            st.caption(f"Source {number} cannot be opened: {exc}")
    meta = panel.get("retrieval", {})
    st.caption(
        f"Duration {panel.get('latency_ms') or 'unknown'} ms · "
        f"{meta.get('num_packed', 0)} sources in context"
    )
    with st.expander("Details"):
        banner = fallback_text(
            fallback=bool(meta.get("fallback", False)),
            reason=meta.get("fallback_reason"),
            cap_hits=list(meta.get("cap_hits", []) or []),
        )
        if banner:
            st.write(banner)
        if meta.get("degraded"):
            st.write("Reranking degraded: showing pre-rerank order.")
        st.code(f"strategy: {panel['strategy']}\nrun: {panel.get('run_id', 'unsaved')}")
        if panel.get("run_id"):
            st.code(f"deep link: ?{encode_deep_link([panel['run_id']])}", language="text")


def _render_inspector() -> None:
    import streamlit as st

    selection = st.session_state.get("compare_inspector")
    result = st.session_state.get("compare_result")
    if not selection or not result:
        return
    arm = selection.get("arm")
    chunk_id = selection.get("chunk_id")
    evidence = result["evidence"].get(arm)
    if not evidence:
        return
    panels = {panel["strategy"]: panel for panel in result["panels"]}
    panel = panels.get(arm, {})
    citation = next((c for c in panel.get("citations", []) if c.get("chunk_id") == chunk_id), None)
    if citation is None:
        st.warning(f"Selected source {chunk_id!r} is not cited by this answer.")
        return
    try:
        sources = build_sources(
            evidence["chunks"], evidence["packed_ids"], panel.get("citations", [])
        )
        resolved = resolve_citation(citation, sources)
    except (LookupError, ValueError) as exc:
        st.warning(f"Selected source cannot be opened: {exc}")
        return
    st.subheader("Evidence inspector")
    render_reader(resolved["source"], reader_blocks_for_citation(resolved))
    if st.button("Open full evidence review", key="inspector-full"):
        st.session_state["evidence_context"] = {
            "run_ids": list(result["run_ids"].values()),
            "question": result["question"],
            "source_id": chunk_id,
            "return_to": "compare",
        }
        st.switch_page("pages/evidence.py")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="GraphRAG Lab — Ask & Compare", layout="wide")
    render_page_chrome(title="Ask & Compare", subtitle="See what each retrieval method found.")
    render_tour("pages/compare.py")
    _session_defaults()

    corpus_id = _corpus_id()
    if corpus_id is None:
        st.info("No corpus yet. Prepare a dataset first, then ask a question.")
        if st.button("Go to Dataset & Index", key="compare-to-dataset"):
            st.switch_page("pages/dataset.py")
        return

    saved = st.query_params.get_all("run")
    if saved and st.session_state.get("compare_result") is None:
        try:
            store = RunStore(_db_path())
            panels = [panel.model_dump() for panel in load_panels(store, saved)]
            st.info("Opened saved runs. This view never reruns a question.")
            for panel in panels:
                st.subheader(arm_label(panel["strategy"]))
                st.write(panel.get("answer") or f"({panel.get('status')})")
        except ValueError as exc:
            st.error(str(exc))
        return

    with st.form("question-form"):
        question = st.text_area(
            "Question", value=st.session_state.get("compare_draft", ""), height=80
        )
        left = st.selectbox(
            "Left arm",
            list(ARM_LABELS),
            index=list(ARM_LABELS).index(st.session_state.get("compare_left", DEFAULT_PAIR[0])),
            format_func=arm_label,
        )
        right = st.selectbox(
            "Right arm",
            list(ARM_LABELS),
            index=list(ARM_LABELS).index(st.session_state.get("compare_right", DEFAULT_PAIR[1])),
            format_func=arm_label,
        )
        with st.expander("Arm details"):
            for strategy in ARM_LABELS:
                st.write(describe_arm(strategy))
        run = st.form_submit_button("Run comparison")
    st.session_state["compare_draft"] = question

    st.write("Examples (tuning questions only; locked evaluation stays out)")
    try:
        examples = tuning_examples(DEFAULT_DATASET_DIR)
    except ValueError as exc:
        st.caption(str(exc))
        examples = []
    cols = st.columns(min(4, max(1, len(examples[:4])))) if examples else []
    for col, example in zip(cols, examples[:4]):
        with col:
            if st.button(
                f"{example['type']}: {example['question'][:40]}…", key=f"ex-{example['id']}"
            ):
                st.session_state["compare_draft"] = example["question"]
                st.rerun()

    if run:
        try:
            validate_arm_pair(left, right)
            assert_matched_budgets([dict(ANSWER_BUDGETS), dict(ANSWER_BUDGETS)])
        except ValueError as exc:
            st.error(str(exc))
            return
        if not question.strip():
            st.error("Enter a question first. Nothing was run.")
            return
        st.session_state["compare_left"] = left
        st.session_state["compare_right"] = right
        try:
            config = load_config(st.session_state.get("config_path", DEFAULT_CONFIG))
        except Exception as exc:
            st.error(f"Configuration invalid: {exc}")
            return
        try:
            from graphrag_lab.neo4j_store import create_driver

            if config.neo4j_password is None:
                raise ValueError("GRAPHRAG_NEO4J_PASSWORD is not set; cannot retrieve evidence")
            driver = create_driver(config)
        except Exception as exc:
            st.error(
                "Retrieval unavailable: "
                + " ".join(str(exc).split())[:300]
                + " Fix the cause and retry; nothing was charged."
            )
            return
        try:
            config.require_llm_ready()
        except ValueError as exc:
            st.error(
                f"Answering unavailable: {exc}. "
                "Set the LLM model and API key to run comparisons; retrieval state is unchanged."
            )
            return
        panels: list[dict[str, Any]] = []
        evidence: dict[str, Any] = {}
        placeholders = st.columns(2)
        try:
            for position, strategy in enumerate((left, right)):
                with placeholders[position]:
                    with st.status(f"{arm_label(strategy)}: starting…", expanded=False) as status:

                        def _stage(
                            text: str, _status: Any = status, _label: str = arm_label(strategy)
                        ) -> None:
                            _status.update(label=f"{_label}: {text}")

                        try:
                            panel = _run_arm(
                                strategy,
                                question,
                                corpus_id,
                                config,
                                driver,
                                config.neo4j.database,
                                _stage,
                            )
                        except Exception as exc:  # noqa: BLE001 - per-arm failure persists
                            panel = _failed_panel(strategy, exc)
                        panels.append(panel)
                        evidence[strategy] = {
                            "chunks": panel["retrieval"].get("chunks", []),
                            "packed_ids": panel["retrieval"].get("packed_ids", []),
                            "traversed": panel["retrieval"].get("traversed", []),
                        }
                        status.update(label=f"{arm_label(strategy)}: done", state="complete")
        finally:
            try:
                driver.close()
            except Exception:
                pass
        qid = question_id_for(question)
        try:
            run_ids = persist_compare(
                RunStore(_db_path()),
                question=question.strip(),
                question_id=qid,
                snapshot_id=st.session_state.get("snapshot_id"),
                panels=panels,
                config={**ANSWER_BUDGETS, "question_set": "interactive"},
            )
        except Exception as exc:
            st.error(f"Persistence failed, answers are still shown: {exc}")
            run_ids = {}
        for panel in panels:
            panel["run_id"] = run_ids.get(panel["strategy"])
        st.session_state["compare_result"] = {
            "question": question.strip(),
            "question_id": qid,
            "arms": [left, right],
            "panels": panels,
            "evidence": evidence,
            "run_ids": run_ids,
        }
        st.session_state["compare_inspector"] = None

    result = st.session_state.get("compare_result")
    if result:
        st.subheader("Answers")
        st.caption("Identical answers are not a tie; different answers are not an improvement.")
        columns = st.columns(2)
        for position, panel in enumerate(result["panels"]):
            with columns[position]:
                _render_panel(panel, result["evidence"][panel["strategy"]], key=f"arm-{position}")
        _render_inspector()
    render_footer(corpus=corpus_id, index="see Dataset & Index", models="see Overview footer")


if __name__ == "__main__":
    main()
