"""Shared evidence presentation (UI-UX.md revision 2).

Pure helpers build source-list rows, reader blocks with cited-sentence cues,
and relation-table rows that the e2e suite asserts. ``render_*`` wrappers
import Streamlit lazily so importing this module never requires the UI
dependency. Text cues accompany every highlight; strategy and status are
never communicated by color alone.
"""

from __future__ import annotations

from typing import Any

from app.components.theme import TOKENS
from graphrag_lab.compare_flow import EvidenceSource, RelationRow


def source_row_label(source: EvidenceSource) -> str:
    """Descriptive accessible name, e.g. 'Open source 2: Title, s0–s1'."""
    rank = f"source {source.rank}" if source.rank is not None else "source"
    context = "in final context" if source.in_final_context else "retrieved only"
    return f"Open {rank}: {source.title}, {source.sentence_range} [{source.origin}, {context}]"


def reader_blocks(
    source: EvidenceSource, cited_ids: list[int] | None = None
) -> list[dict[str, Any]]:
    """Reader sentences with cited/retrieved labels (truth never implied)."""
    wanted = set(cited_ids or [])
    return [
        {
            "sentence_id": sid,
            "text": text,
            "marker": "[cited]" if sid in wanted else "[retrieved]",
        }
        for sid, text in zip(source.sentence_ids, source.sentence_texts)
    ]


def reader_blocks_for_citation(
    resolved: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reader sentences for one resolved citation: cited vs retrieved context."""
    return [
        {
            "sentence_id": row["sentence_id"],
            "text": row["text"],
            "marker": "[cited]" if row["cited"] else "[retrieved]",
        }
        for row in resolved.get("sentences", [])
    ]


def relation_table_rows(rows: list[RelationRow]) -> list[dict[str, Any]]:
    """Plain-data relation table: entity → relation → entity → evidence."""
    return [
        {
            "source": row.source_name,
            "relation": row.relation_type,
            "target": row.target_name,
            "evidence": (
                f"{row.evidence_title} [{row.evidence_chunk_id}]"
                if row.evidence_available
                else f"{row.evidence_chunk_id} (evidence not in context)"
            ),
            "evidence_chunk_id": row.evidence_chunk_id,
            "selectable": row.evidence_available,
        }
        for row in rows
    ]


def render_source_list(
    sources: list[EvidenceSource], *, selected_id: str | None, key_prefix: str
) -> str | None:
    """One button per source with a descriptive name. Returns the picked id."""
    import streamlit as st

    picked: str | None = None
    for source in sources:
        label = source_row_label(source)
        kind = "primary" if source.chunk_id == selected_id else "secondary"
        if st.button(label, key=f"{key_prefix}-src-{source.chunk_id}", type=kind):  # type: ignore[arg-type]
            picked = source.chunk_id
    return picked


def render_reader(source: EvidenceSource, blocks: list[dict[str, Any]]) -> None:
    import streamlit as st

    st.markdown(f"**{source.title}**")
    st.caption(
        f"`{source.chunk_id}` · paragraph `{source.paragraph_id}` · "
        f"{source.sentence_range} · {source.origin} · "
        f"{'in final context' if source.in_final_context else 'retrieved only'}"
    )
    for block in blocks:
        marker = block["marker"]
        tone = TOKENS["action"] if marker == "[cited]" else TOKENS["text_secondary"]
        st.markdown(
            f"<span style='color:{tone}'>{marker}</span> "
            f"<span>s{block['sentence_id']}:</span> {block['text']}",
            unsafe_allow_html=True,
        )
    st.caption("Cited means the answer references this sentence, not that it is verified true.")


def render_relation_table(rows: list[RelationRow], *, key_prefix: str) -> str | None:
    """Keyboard-operable relation table (radio). Returns picked evidence chunk."""
    import streamlit as st

    table = relation_table_rows(rows)
    if not table:
        st.write("No retrieved connections for this answer.")
        return None
    options = [
        f"{row['source']} —[{row['relation']}]→ {row['target']} · {row['evidence']}"
        for row in table
    ]
    choice = st.radio(
        "Retrieved connections (evidence, not model reasoning)",
        options,
        key=f"{key_prefix}-relations",
    )
    index = options.index(choice)
    row = table[index]
    return str(row["evidence_chunk_id"]) if row["selectable"] else None


__all__ = [
    "reader_blocks",
    "reader_blocks_for_citation",
    "relation_table_rows",
    "render_reader",
    "render_relation_table",
    "render_source_list",
    "source_row_label",
]
