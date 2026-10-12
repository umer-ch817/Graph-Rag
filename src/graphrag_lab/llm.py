"""Structured-output LLM adapter for T05 (vector-qa).

Temperature zero, JSON mode, bounded retries (initial attempt plus at most
two retries), worst-case budget reservation and a deterministic fake backend
for offline unit tests. The HTTP backend targets OpenAI-compatible
chat-completions endpoints over stdlib ``urllib`` so T05 adds no new
third-party dependency. Provider keys stay in process environment and never
appear in errors or logs.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from graphrag_lab.chunking import TOKEN_ESTIMATOR, estimate_tokens
from graphrag_lab.config import AppConfig, BudgetConfig

LLM_SCHEMA_VERSION = "answer-v1"
PROMPT_VERSION = "answer-prompt-v1"
DEFAULT_MAX_OUTPUT_TOKENS = 256
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_API_BASE = "https://api.openai.com/v1"
MAX_RETRIES = 2


def backoff_delay_s(attempt: int) -> float:
    """Exponential backoff between retries (0.5s, 1s, 2s, ...).

    Shared by the HTTP adapter and extraction so transient-failure policy
    stays in one place. ``attempt`` is the zero-based retry index.
    """
    if not isinstance(attempt, int) or attempt < 0:
        raise ValueError("attempt must be a non-negative integer")
    return 0.5 * (2**attempt)


def is_retryable_error(exc: BaseException) -> bool:
    """True when an LLM failure is marked transient and worth retrying."""
    return bool(getattr(exc, "retryable", False))


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class LlmError(RuntimeError):
    """Base adapter error. ``code`` matches BACKEND-SCHEMA error names."""

    code = "PROVIDER_ERROR"

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class InvalidInputError(LlmError, ValueError):
    code = "INVALID_INPUT"


class DependencyUnavailableError(LlmError):
    code = "DEPENDENCY_UNAVAILABLE"


class BudgetExceededError(LlmError):
    code = "BUDGET_EXCEEDED"


class ProviderError(LlmError):
    code = "PROVIDER_ERROR"


class ValidationFailedError(LlmError, ValueError):
    code = "VALIDATION_FAILED"


class LlmUsage(Record):
    """Token usage for one call. Cost stays None while pricing is unknown."""

    model: str = Field(min_length=1)
    input_tokens: int = Field(ge=0, strict=True)
    output_tokens: int = Field(ge=0, strict=True)
    total_tokens: int = Field(ge=0, strict=True)
    estimated_cost_usd: float | None = None
    pricing_version: str | None = None


class ChatMessage(Record):
    role: Literal["system", "user", "assistant"] = "user"
    content: str = Field(min_length=1, max_length=20000)


class LlmRequest(Record):
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, ge=1, le=4096)
    json_mode: bool = True


class LlmResponse(Record):
    model: str = Field(min_length=1)
    text: str = Field(min_length=1)
    usage: LlmUsage


class LlmBackend(Protocol):
    """Minimal interface used by answering and (later) extraction."""

    @property
    def model_name(self) -> str: ...

    def complete(self, request: LlmRequest) -> LlmResponse: ...


def estimate_text_tokens(text: str) -> int:
    """Conservative offline token estimate (``char-word-max-v1``)."""
    return estimate_tokens(text)


def estimator_name() -> str:
    return TOKEN_ESTIMATOR


class BudgetTracker:
    """Enforce call/token caps; reserve worst case before each call.

    Monetary enforcement requires explicit pricing: when
    ``max_cost_usd`` is set but no per-token price is supplied, reservation
    raises DEPENDENCY_UNAVAILABLE instead of labeling unknown cost as zero.
    T05 runs without pricing, so cost stays None and call/token caps apply.
    """

    def __init__(
        self,
        budget: BudgetConfig,
        *,
        price_per_token_usd: float | None = None,
        pricing_version: str | None = None,
    ) -> None:
        self._budget = budget
        self._price = price_per_token_usd
        self._pricing_version = pricing_version
        self.used_calls = 0
        self.used_tokens = 0
        self.used_cost_usd = 0.0

    @property
    def budget(self) -> BudgetConfig:
        return self._budget

    def reserve(self, input_tokens: int, max_output_tokens: int) -> None:
        if not isinstance(input_tokens, int) or input_tokens < 0:
            raise InvalidInputError("input_tokens must be a non-negative integer")
        if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
            raise InvalidInputError("max_output_tokens must be a positive integer")
        worst_tokens = input_tokens + max_output_tokens
        if self._budget.max_calls is not None and self.used_calls + 1 > self._budget.max_calls:
            raise BudgetExceededError(
                f"Call budget exceeded: {self.used_calls + 1} > {self._budget.max_calls}"
            )
        if (
            self._budget.max_total_tokens is not None
            and self.used_tokens + worst_tokens > self._budget.max_total_tokens
        ):
            raise BudgetExceededError(
                f"Token budget exceeded: {self.used_tokens + worst_tokens} "
                f"> {self._budget.max_total_tokens}"
            )
        if self._budget.max_cost_usd is not None:
            if self._price is None:
                raise DependencyUnavailableError(
                    "Pricing metadata is required to enforce max_cost_usd; "
                    "configure a call/token cap instead"
                )
            worst_cost = self.used_cost_usd + worst_tokens * self._price
            if worst_cost > self._budget.max_cost_usd:
                raise BudgetExceededError(
                    f"Cost budget exceeded: {worst_cost:.6f} > {self._budget.max_cost_usd:.6f}"
                )

    def reconcile(self, usage: LlmUsage) -> None:
        self.used_calls += 1
        self.used_tokens += usage.total_tokens
        if usage.estimated_cost_usd is not None:
            self.used_cost_usd += usage.estimated_cost_usd


class FakeLlmBackend:
    """Deterministic offline backend replaying queued payloads.

    Each queued item is either a dict (serialized to JSON) or a raw string
    (returned verbatim to exercise malformed-JSON rejection). Calls are
    recorded; an empty queue raises PROVIDER_ERROR instead of fabricating.
    """

    def __init__(
        self, outputs: Sequence[dict[str, Any] | str], *, model: str = "fake-llm-test"
    ) -> None:
        if not model or not model.strip():
            raise InvalidInputError("model must be a nonempty string")
        self._model = model.strip()
        self._outputs: list[dict[str, Any] | str] = list(outputs)
        self.calls: list[LlmRequest] = []

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def model_revision(self) -> str:
        """Revision label used in extraction cache keys (T07)."""
        return self._model

    @property
    def remaining(self) -> int:
        return len(self._outputs)

    def complete(self, request: LlmRequest) -> LlmResponse:
        if request.model != self._model:
            raise InvalidInputError(
                f"Request model {request.model!r} != backend model {self._model!r}"
            )
        if not self._outputs:
            raise ProviderError("Fake backend has no queued outputs left", retryable=False)
        self.calls.append(request)
        payload = self._outputs.pop(0)
        text = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True)
        if not text.strip():
            raise ProviderError("Fake backend produced empty text", retryable=False)
        input_tokens = sum(estimate_text_tokens(m.content) for m in request.messages)
        output_tokens = estimate_text_tokens(text)
        return LlmResponse(
            model=self._model,
            text=text,
            usage=LlmUsage(
                model=self._model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )


def _sanitized_provider_message(status: int | None, body: str) -> str:
    snippet = body.strip().replace("\n", " ")[:300]
    if status is None:
        return f"LLM request failed ({snippet})" if snippet else "LLM request failed"
    return (
        f"LLM request failed with status {status} ({snippet})"
        if snippet
        else (f"LLM request failed with status {status}")
    )


class HttpChatBackend:
    """OpenAI-compatible chat-completions backend over stdlib urllib.

    Sends temperature-zero JSON-mode requests, parses ``choices[0].message``
    content and usage, and retries transient failures (429/5xx/timeouts) at
    most twice after the initial attempt. API keys never enter messages.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        api_base: str = DEFAULT_API_BASE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        if not model or not model.strip() or any(c.isspace() for c in model):
            raise InvalidInputError("model must be nonempty with no whitespace")
        if not api_key or not api_key.strip():
            raise DependencyUnavailableError("LLM API key must be nonempty")
        if not api_base or not api_base.strip():
            raise InvalidInputError("api_base must be a nonempty string")
        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise InvalidInputError("timeout_s must be positive")
        if not isinstance(max_retries, int) or not 0 <= max_retries <= MAX_RETRIES:
            raise InvalidInputError(f"max_retries must be an integer in 0..{MAX_RETRIES}")
        self._model = model.strip()
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._timeout_s = float(timeout_s)
        self._max_retries = max_retries

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def model_revision(self) -> str:
        """Revision label used in extraction cache keys (T07)."""
        return self._model

    def complete(self, request: LlmRequest) -> LlmResponse:
        if request.model != self._model:
            raise InvalidInputError(
                f"Request model {request.model!r} != backend model {self._model!r}"
            )
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload).encode("utf-8")
        last_error: LlmError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._post_once(body, request)
            except LlmError as exc:
                last_error = exc
                retryable = is_retryable_error(exc)
                if not retryable or attempt >= self._max_retries:
                    raise
                time.sleep(backoff_delay_s(attempt))
        raise last_error or ProviderError("LLM request failed", retryable=False)

    def _post_once(self, body: bytes, request: LlmRequest) -> LlmResponse:
        url = f"{self._api_base}/chat/completions"
        http_request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                # Key is attached to the header only; never echoed in errors.
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self._timeout_s) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            raise ProviderError(
                _sanitized_provider_message(exc.code, detail), retryable=retryable
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(
                f"LLM request transport failed: {type(exc).__name__}", retryable=True
            ) from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError("LLM returned non-JSON HTTP payload", retryable=False) from exc
        return _parse_chat_payload(data, self._model)


def _parse_chat_payload(data: dict[str, Any], model: str) -> LlmResponse:
    try:
        choices = data.get("choices")
        content = choices[0]["message"]["content"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ProviderError(
            "LLM payload has no choices[0].message.content", retryable=False
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise ProviderError("LLM returned empty content", retryable=False)
    raw_usage = data.get("usage") or {}
    try:
        input_tokens = int(raw_usage.get("prompt_tokens", estimate_text_tokens(content)))
        output_tokens = int(raw_usage.get("completion_tokens", estimate_text_tokens(content)))
        total_tokens = int(raw_usage.get("total_tokens", input_tokens + output_tokens))
    except (TypeError, ValueError) as exc:
        raise ProviderError("LLM usage block is not numeric", retryable=False) from exc
    return LlmResponse(
        model=model,
        text=content,
        usage=LlmUsage(
            model=model,
            input_tokens=max(0, input_tokens),
            output_tokens=max(0, output_tokens),
            total_tokens=max(0, total_tokens),
        ),
    )


def backend_from_config(
    config: AppConfig,
    *,
    api_base: str = DEFAULT_API_BASE,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_retries: int = MAX_RETRIES,
) -> HttpChatBackend:
    """Build the HTTP backend from validated app config (requires LLM creds)."""
    try:
        config.require_llm_ready()
    except ValueError as exc:
        raise DependencyUnavailableError(str(exc)) from exc
    assert config.llm_model is not None and config.llm_api_key is not None
    return HttpChatBackend(
        model=config.llm_model,
        api_key=config.llm_api_key.get_secret_value(),
        api_base=api_base,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
