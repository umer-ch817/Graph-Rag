"""Shared visual tokens for the GraphRAG Lab dashboard (UI-UX.md revision 2).

Pure data and CSS builders only: no Streamlit import here, so unit and e2e
tests can assert the contract without the UI dependency. Every page imports
these tokens instead of defining per-page styling.
"""

from __future__ import annotations

TOKENS: dict[str, str] = {
    "background": "#F6F7F5",
    "surface": "#FFFFFF",
    "surface_muted": "#EDF1EE",
    "text_primary": "#18251F",
    "text_secondary": "#4C5E54",
    "border": "#D6DED8",
    "control_border": "#728579",
    "action": "#176348",
    "action_hover": "#104B36",
    "focus": "#245FC7",
    "vector": "#295D91",
    "graph": "#176348",
    "warning": "#865500",
    "danger": "#A32932",
}

NAV_ORDER: tuple[str, ...] = (
    "Overview",
    "Dataset & Index",
    "Ask & Compare",
    "Evidence",
    "Evaluations",
)

STATUS_TEXT: dict[str, str] = {
    "missing": "Not started",
    "blocked": "Blocked",
    "queued": "Queued",
    "running": "Running",
    "cancelling": "Cancelling",
    "paused_budget": "Paused on budget",
    "failed": "Failed",
    "cancelled": "Cancelled",
    "interrupted": "Interrupted",
    "ready": "Ready",
    "partial": "Partial",
}

STATUS_ICON: dict[str, str] = {
    "missing": "○",
    "blocked": "⊘",
    "queued": "◷",
    "running": "▶",
    "cancelling": "◐",
    "paused_budget": "⏸",
    "failed": "✕",
    "cancelled": "■",
    "interrupted": "⚠",
    "ready": "●",
    "partial": "◑",
}


def pill_text(state: str) -> str:
    """Text-plus-icon label; status is never communicated by color alone."""
    label = STATUS_TEXT.get(state, state)
    icon = STATUS_ICON.get(state, "•")
    return f"{icon} {label}"


def base_css() -> str:
    """Minimal layout CSS using the shared tokens (borders, not shadows)."""
    return f"""
    <style>
    :root {{ color-scheme: light; }}
    .stApp {{ background-color: {TOKENS["background"]}; }}
    .lab-panel {{
      background-color: {TOKENS["surface"]};
      border: 1px solid {TOKENS["border"]};
      border-radius: 8px;
      padding: 16px 20px;
      margin-bottom: 16px;
    }}
    .lab-meta {{
      background-color: {TOKENS["surface_muted"]};
      border: 1px solid {TOKENS["border"]};
      border-radius: 6px;
      padding: 8px 12px;
      font-size: 14px;
      color: {TOKENS["text_secondary"]};
    }}
    .lab-pill {{
      display: inline-block;
      border: 1px solid {TOKENS["border"]};
      border-radius: 999px;
      padding: 2px 12px;
      font-size: 14px;
    }}
    .lab-heading {{ color: {TOKENS["text_primary"]}; }}
    .lab-error {{ color: {TOKENS["danger"]}; }}
    .lab-warning {{ color: {TOKENS["warning"]}; }}
    .lab-action {{ color: {TOKENS["action"]}; }}
    a:focus, button:focus, input:focus, textarea:focus, select:focus {{
      outline: 2px solid {TOKENS["focus"]} !important;
      outline-offset: 2px;
    }}
    </style>
    """.strip()


__all__ = ["NAV_ORDER", "STATUS_ICON", "STATUS_TEXT", "TOKENS", "base_css", "pill_text"]
