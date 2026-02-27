"""
Native SDK interceptors for OpenAI and Anthropic clients.

Usage::

    from director_ai import guard
    client = guard(OpenAI(), facts={"refund": "within 30 days"})
    resp = client.chat.completions.create(...)  # auto-scored
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from contextvars import ContextVar

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.exceptions import HallucinationError
from director_ai.core.types import CoherenceScore

_log = logging.getLogger("DirectorAI.guard")
_score_var: ContextVar[CoherenceScore | None] = ContextVar(
    "director_ai_score", default=None
)

STREAM_CHECK_INTERVAL = 8


def get_score() -> CoherenceScore | None:
    """Retrieve the last score stored by ``on_fail="metadata"``."""
    return _score_var.get()


def guard(
    client,
    *,
    facts: dict[str, str] | None = None,
    store: GroundTruthStore | None = None,
    threshold: float = 0.6,
    use_nli: bool | None = None,
    on_fail: str = "raise",
):
    """Wrap an OpenAI/Anthropic SDK client with coherence scoring.

    Returns the same *client* object with patched sub-objects.
    """
    if on_fail not in ("raise", "log", "metadata"):
        raise ValueError(
            f"on_fail must be 'raise', 'log', or 'metadata', got {on_fail!r}"
        )

    gts = store or GroundTruthStore()
    if facts:
        for k, v in facts.items():
            gts.add(k, v)
    scorer = CoherenceScorer(
        threshold=threshold, ground_truth_store=gts, use_nli=use_nli
    )

    mod = type(client).__module__ or ""
    if mod.startswith("openai"):
        client.chat.completions = _OpenAICompletionsProxy(
            client.chat.completions, scorer, on_fail
        )
    elif mod.startswith("anthropic"):
        client.messages = _AnthropicMessagesProxy(client.messages, scorer, on_fail)
    else:
        raise TypeError(
            f"Unsupported client type: {type(client).__qualname__} "
            f"(module={mod!r}). Expected openai or anthropic SDK."
        )
    return client


def _extract_prompt(messages: list[dict]) -> str:
    """Pull the user prompt from a messages array."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return str(block.get("text", ""))
            return str(content)
    return " ".join(str(m.get("content", "")) for m in messages)


def _handle_failure(on_fail, query, response_text, score):
    if on_fail == "raise":
        raise HallucinationError(query, response_text, score)
    elif on_fail == "log":
        _log.warning(
            "Hallucination detected (coherence=%.3f): %.100s",
            score.score,
            response_text,
        )
    elif on_fail == "metadata":
        _score_var.set(score)


def _score_and_gate(scorer, on_fail, query, response_text):
    approved, cs = scorer.review(query, response_text)
    if on_fail == "metadata":
        _score_var.set(cs)
    if not approved:
        _handle_failure(on_fail, query, response_text, cs)
    return cs


# ── OpenAI proxy ────────────────────────────────────────────────────


class _OpenAICompletionsProxy:
    """Drop-in for ``client.chat.completions``."""

    def __init__(self, original, scorer, on_fail):
        self._original = original
        self._scorer = scorer
        self._on_fail = on_fail

    def create(self, **kwargs):
        prompt = _extract_prompt(kwargs.get("messages", []))
        streaming = kwargs.get("stream", False)

        if inspect.iscoroutinefunction(self._original.create):
            return self._acreate(prompt, streaming, kwargs)

        response = self._original.create(**kwargs)

        if streaming:
            return _GuardedOpenAIStream(response, self._scorer, self._on_fail, prompt)

        text = _openai_response_text(response)
        _score_and_gate(self._scorer, self._on_fail, prompt, text)
        return response

    async def _acreate(self, prompt, streaming, kwargs):
        response = await self._original.create(**kwargs)
        if streaming:
            return _GuardedOpenAIStream(response, self._scorer, self._on_fail, prompt)
        text = _openai_response_text(response)
        _score_and_gate(self._scorer, self._on_fail, prompt, text)
        return response

    def __getattr__(self, name):
        return getattr(self._original, name)


def _openai_response_text(response) -> str:
    with contextlib.suppress(IndexError, AttributeError):
        return response.choices[0].message.content or ""
    return ""


def _extract_stream_delta(chunk) -> str | None:
    with contextlib.suppress(IndexError, AttributeError):
        delta = chunk.choices[0].delta.content
        return str(delta) if delta is not None else None
    return None


class _GuardedOpenAIStream:
    """Wraps an OpenAI stream with periodic coherence checks."""

    def __init__(self, stream, scorer, on_fail, prompt):
        self._stream = stream
        self._scorer = scorer
        self._on_fail = on_fail
        self._prompt = prompt
        self._buffer = []
        self._token_count = 0

    def __iter__(self):
        for chunk in self._stream:
            delta = _extract_stream_delta(chunk)
            if delta:
                self._buffer.append(delta)
                self._token_count += 1
                if self._token_count % STREAM_CHECK_INTERVAL == 0:
                    self._periodic_check()
            yield chunk
        self._final_check()

    def __aiter__(self):
        return self._aiter_impl()

    async def _aiter_impl(self):
        async for chunk in self._stream:
            delta = _extract_stream_delta(chunk)
            if delta:
                self._buffer.append(delta)
                self._token_count += 1
                if self._token_count % STREAM_CHECK_INTERVAL == 0:
                    self._periodic_check()
            yield chunk
        self._final_check()

    def _periodic_check(self):
        text = "".join(self._buffer)
        approved, cs = self._scorer.review(self._prompt, text)
        if not approved:
            _handle_failure(self._on_fail, self._prompt, text, cs)

    def _final_check(self):
        text = "".join(self._buffer)
        if text:
            _score_and_gate(self._scorer, self._on_fail, self._prompt, text)


# ── Anthropic proxy ─────────────────────────────────────────────────


class _AnthropicMessagesProxy:
    """Drop-in for ``client.messages``."""

    def __init__(self, original, scorer, on_fail):
        self._original = original
        self._scorer = scorer
        self._on_fail = on_fail

    def create(self, **kwargs):
        prompt = _extract_prompt(kwargs.get("messages", []))
        streaming = kwargs.get("stream", False)

        if inspect.iscoroutinefunction(self._original.create):
            return self._acreate(prompt, streaming, kwargs)

        response = self._original.create(**kwargs)

        if streaming:
            return _GuardedAnthropicStream(
                response, self._scorer, self._on_fail, prompt
            )

        text = _anthropic_response_text(response)
        _score_and_gate(self._scorer, self._on_fail, prompt, text)
        return response

    async def _acreate(self, prompt, streaming, kwargs):
        response = await self._original.create(**kwargs)
        if streaming:
            return _GuardedAnthropicStream(
                response, self._scorer, self._on_fail, prompt
            )
        text = _anthropic_response_text(response)
        _score_and_gate(self._scorer, self._on_fail, prompt, text)
        return response

    def __getattr__(self, name):
        return getattr(self._original, name)


def _anthropic_response_text(response) -> str:
    with contextlib.suppress(IndexError, AttributeError):
        return response.content[0].text or ""
    return ""


def _extract_anthropic_event_text(event) -> str | None:
    text = getattr(event, "text", None)
    if text:
        return str(text)
    delta = getattr(event, "delta", None)
    if isinstance(delta, dict):
        val = delta.get("text")
        return str(val) if val is not None else None
    return None


class _GuardedAnthropicStream:
    """Wraps an Anthropic stream with periodic coherence checks."""

    def __init__(self, stream, scorer, on_fail, prompt):
        self._stream = stream
        self._scorer = scorer
        self._on_fail = on_fail
        self._prompt = prompt
        self._buffer = []
        self._token_count = 0

    def __iter__(self):
        for event in self._stream:
            text = _extract_anthropic_event_text(event)
            if text:
                self._buffer.append(text)
                self._token_count += 1
                if self._token_count % STREAM_CHECK_INTERVAL == 0:
                    self._periodic_check()
            yield event
        self._final_check()

    def __aiter__(self):
        return self._aiter_impl()

    async def _aiter_impl(self):
        async for event in self._stream:
            text = _extract_anthropic_event_text(event)
            if text:
                self._buffer.append(text)
                self._token_count += 1
                if self._token_count % STREAM_CHECK_INTERVAL == 0:
                    self._periodic_check()
            yield event
        self._final_check()

    def _periodic_check(self):
        text = "".join(self._buffer)
        approved, cs = self._scorer.review(self._prompt, text)
        if not approved:
            _handle_failure(self._on_fail, self._prompt, text, cs)

    def _final_check(self):
        text = "".join(self._buffer)
        if text:
            _score_and_gate(self._scorer, self._on_fail, self._prompt, text)
