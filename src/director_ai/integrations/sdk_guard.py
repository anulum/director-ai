# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Native SDK interceptors for OpenAI and Anthropic clients.

Usage::

    from director_ai import guard
    client = guard(OpenAI(), facts={"refund": "within 30 days"})
    resp = client.chat.completions.create(...)  # auto-scored
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from contextvars import ContextVar
from typing import Any, cast

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.exceptions import HallucinationError, InjectionDetectedError
from director_ai.core.types import CoherenceScore

_log = logging.getLogger("DirectorAI.guard")
_score_var: ContextVar[CoherenceScore | None] = ContextVar(
    "director_ai_score",
    default=None,
)

STREAM_CHECK_INTERVAL = 8


def get_score() -> CoherenceScore | None:
    """Retrieve the last score stored by ``on_fail="metadata"``."""
    return _score_var.get()


def score(
    prompt: str,
    response: str,
    *,
    facts: dict[str, str] | None = None,
    store: GroundTruthStore | None = None,
    threshold: float = 0.3,
    use_nli: bool | None = None,
    profile: str | None = None,
    injection_detection: bool = False,
    injection_threshold: float = 0.7,
) -> CoherenceScore:
    """Score a single prompt/response pair for hallucination.

    Returns a ``CoherenceScore`` without requiring an SDK client.
    When *injection_detection* is enabled, ``CoherenceScore.injection_risk``
    is populated with the intent-grounded injection risk score.
    """
    if profile is not None:
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig.from_profile(profile)
        gts = store or GroundTruthStore()
        if facts:
            for k, v in facts.items():
                gts.add(k, v)
        scorer = cfg.build_scorer(store=gts)
    else:
        gts = store or GroundTruthStore()
        if facts:
            for k, v in facts.items():
                gts.add(k, v)
        scorer = CoherenceScorer(
            threshold=threshold,
            ground_truth_store=gts,
            use_nli=use_nli,
        )
    if injection_detection:
        scorer.enable_injection_detection(injection_threshold=injection_threshold)
    _approved, cs = scorer.review(prompt, response)
    return cast(CoherenceScore, cs)


def guard(
    client: Any,
    *,
    facts: dict[str, str] | None = None,
    store: GroundTruthStore | None = None,
    threshold: float = 0.3,
    use_nli: bool | None = None,
    on_fail: str = "raise",
    injection_detection: bool = False,
    injection_threshold: float = 0.7,
) -> Any:
    """Wrap an LLM SDK client with coherence scoring.

    Supports five SDK shapes:

    - **OpenAI-compatible** (``client.chat.completions.create``):
      OpenAI, vLLM, Groq, LiteLLM, Ollama, Together.
    - **Anthropic** (``client.messages.create``).
    - **AWS Bedrock** (``client.converse`` / ``client.converse_stream``).
    - **Google Gemini** (``client.generate_content``).
    - **Cohere** (``client.chat`` without ``client.completions``).

    When *injection_detection* is enabled, each response is additionally
    checked for prompt injection via intent-grounded NLI divergence.
    The *injection_threshold* controls sensitivity (0.0–1.0).

    Returns the guarded client. For OpenAI/Anthropic the original object
    is mutated in place. For Bedrock, Gemini, and Cohere a new proxy is
    returned — **always use the return value**: ``client = guard(client, ...)``.
    """
    if on_fail not in ("raise", "log", "metadata"):
        raise ValueError(
            f"on_fail must be 'raise', 'log', or 'metadata', got {on_fail!r}",
        )

    gts = store or GroundTruthStore()
    if facts:
        for k, v in facts.items():
            gts.add(k, v)
    scorer = CoherenceScorer(
        threshold=threshold,
        ground_truth_store=gts,
        use_nli=use_nli,
    )
    inj_threshold = injection_threshold if injection_detection else None
    if injection_detection:
        scorer.enable_injection_detection(injection_threshold=injection_threshold)

    if _has_openai_shape(client):
        client.chat.completions = _OpenAICompletionsProxy(
            client.chat.completions,
            scorer,
            on_fail,
            injection_threshold=inj_threshold,
        )
    elif _has_anthropic_shape(client):
        client.messages = _AnthropicMessagesProxy(
            client.messages,
            scorer,
            on_fail,
            injection_threshold=inj_threshold,
        )
    elif _has_bedrock_shape(client):
        client = _BedrockProxy(
            client,
            scorer,
            on_fail,
            injection_threshold=inj_threshold,
        )
    elif _has_gemini_shape(client):
        client = _GeminiProxy(
            client,
            scorer,
            on_fail,
            injection_threshold=inj_threshold,
        )
    elif _has_cohere_shape(client):
        client = _CohereProxy(
            client,
            scorer,
            on_fail,
            injection_threshold=inj_threshold,
        )
    else:
        raise TypeError(
            f"Unsupported client type: {type(client).__qualname__}. "
            "Expected OpenAI, Anthropic, Bedrock, Gemini, or Cohere shape.",
        )
    return client


def _has_openai_shape(client) -> bool:
    """True if client exposes ``client.chat.completions.create`` callable."""
    chat = getattr(client, "chat", None)
    if chat is None:
        return False
    completions = getattr(chat, "completions", None)
    if completions is None:
        return False
    return callable(getattr(completions, "create", None))


def _has_anthropic_shape(client) -> bool:
    """True if client exposes ``client.messages.create`` without ``client.chat``."""
    if getattr(client, "chat", None) is not None:
        return False
    messages = getattr(client, "messages", None)
    if messages is None:
        return False
    return callable(getattr(messages, "create", None))


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
    if on_fail == "log":
        _log.warning(
            "Hallucination detected (coherence=%.3f): %.100s",
            score.score,
            response_text,
        )
    elif on_fail == "metadata":  # pragma: no branch
        _score_var.set(score)


def _handle_injection_failure(on_fail, query, response_text, score):
    """Handle a detected injection — mirrors _handle_failure semantics."""
    if on_fail == "raise":
        raise InjectionDetectedError(query, response_text, score)
    if on_fail == "log":
        risk = getattr(score, "injection_risk", None) or 0.0
        _log.warning(
            "Injection detected (risk=%.3f): %.100s",
            risk,
            response_text,
        )
    elif on_fail == "metadata":  # pragma: no branch
        _score_var.set(score)


def _check_injection(on_fail, query, response_text, cs, injection_threshold):
    """Check injection risk on a scored response and handle failure."""
    if injection_threshold is None:
        return
    risk = cs.injection_risk
    if risk is not None and risk >= injection_threshold:
        _handle_injection_failure(on_fail, query, response_text, cs)


def _score_and_gate(scorer, on_fail, query, response_text, *, injection_threshold=None):
    result = scorer.review(query, response_text)
    if asyncio.iscoroutine(result):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                approved, cs = pool.submit(asyncio.run, result).result()
        else:
            approved, cs = asyncio.run(result)
    else:
        approved, cs = result
    if on_fail == "metadata":
        _score_var.set(cs)
    if not approved:
        _handle_failure(on_fail, query, response_text, cs)
    _check_injection(on_fail, query, response_text, cs, injection_threshold)
    return cs


async def _ascore_and_gate(
    scorer, on_fail, query, response_text, *, injection_threshold=None
):
    result = scorer.review(query, response_text)
    if asyncio.iscoroutine(result):
        approved, cs = await result
    else:
        approved, cs = result
    if on_fail == "metadata":
        _score_var.set(cs)
    if not approved:
        _handle_failure(on_fail, query, response_text, cs)
    _check_injection(on_fail, query, response_text, cs, injection_threshold)
    return cs


# â”€â”€ OpenAI proxy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class _OpenAICompletionsProxy:
    """Drop-in for ``client.chat.completions``.

    Wraps either a sync or async OpenAI client. The public
    ``create`` attribute is bound to the right dispatcher at
    init time so callers see a natural method surface without
    re-assigning a method on an existing class definition.
    """

    def __init__(self, original, scorer, on_fail, *, injection_threshold=None):
        self._original = original
        self._scorer = scorer
        self._on_fail = on_fail
        self._injection_threshold = injection_threshold
        self.create: Any = (
            self._acreate_entry
            if inspect.iscoroutinefunction(original.create)
            else self._sync_create
        )

    def _sync_create(self, **kwargs):
        prompt = _extract_prompt(kwargs.get("messages", []))
        streaming = kwargs.get("stream", False)
        response = self._original.create(**kwargs)

        if streaming:
            return _GuardedOpenAIStream(
                response,
                self._scorer,
                self._on_fail,
                prompt,
                injection_threshold=self._injection_threshold,
            )

        text = _openai_response_text(response)
        _score_and_gate(
            self._scorer,
            self._on_fail,
            prompt,
            text,
            injection_threshold=self._injection_threshold,
        )
        return response

    async def _acreate_entry(self, **kwargs):
        prompt = _extract_prompt(kwargs.get("messages", []))
        streaming = kwargs.get("stream", False)
        return await self._acreate(prompt, streaming, kwargs)

    async def _acreate(self, prompt, streaming, kwargs):
        response = await self._original.create(**kwargs)
        if streaming:
            return _GuardedOpenAIStream(
                response,
                self._scorer,
                self._on_fail,
                prompt,
                injection_threshold=self._injection_threshold,
            )
        text = _openai_response_text(response)
        await _ascore_and_gate(
            self._scorer,
            self._on_fail,
            prompt,
            text,
            injection_threshold=self._injection_threshold,
        )
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

    def __init__(self, stream, scorer, on_fail, prompt, *, injection_threshold=None):
        self._stream = stream
        self._scorer = scorer
        self._on_fail = on_fail
        self._prompt = prompt
        self._buffer = []
        self._token_count = 0
        self._injection_threshold = injection_threshold

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
                    await self._aperiodic_check()
            yield chunk
        await self._afinal_check()

    async def _aperiodic_check(self):
        text = "".join(self._buffer)
        await _ascore_and_gate(self._scorer, self._on_fail, self._prompt, text)

    async def _afinal_check(self):
        text = "".join(self._buffer)
        if text:
            await _ascore_and_gate(
                self._scorer,
                self._on_fail,
                self._prompt,
                text,
                injection_threshold=self._injection_threshold,
            )

    def _periodic_check(self):
        text = "".join(self._buffer)
        approved, cs = self._scorer.review(self._prompt, text)
        if not approved:
            _handle_failure(self._on_fail, self._prompt, text, cs)

    def _final_check(self):
        text = "".join(self._buffer)
        if text:
            _score_and_gate(
                self._scorer,
                self._on_fail,
                self._prompt,
                text,
                injection_threshold=self._injection_threshold,
            )


# â”€â”€ Anthropic proxy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class _AnthropicMessagesProxy:
    """Drop-in for ``client.messages``.

    Same sync / async dispatch pattern as
    :class:`_OpenAICompletionsProxy`.
    """

    def __init__(self, original, scorer, on_fail, *, injection_threshold=None):
        self._original = original
        self._scorer = scorer
        self._on_fail = on_fail
        self._injection_threshold = injection_threshold
        self.create: Any = (
            self._acreate_entry
            if inspect.iscoroutinefunction(original.create)
            else self._sync_create
        )

    def _sync_create(self, **kwargs):
        prompt = _extract_prompt(kwargs.get("messages", []))
        streaming = kwargs.get("stream", False)
        response = self._original.create(**kwargs)

        if streaming:
            return _GuardedAnthropicStream(
                response,
                self._scorer,
                self._on_fail,
                prompt,
                injection_threshold=self._injection_threshold,
            )

        text = _anthropic_response_text(response)
        _score_and_gate(
            self._scorer,
            self._on_fail,
            prompt,
            text,
            injection_threshold=self._injection_threshold,
        )
        return response

    async def _acreate_entry(self, **kwargs):
        prompt = _extract_prompt(kwargs.get("messages", []))
        streaming = kwargs.get("stream", False)
        return await self._acreate(prompt, streaming, kwargs)

    async def _acreate(self, prompt, streaming, kwargs):
        response = await self._original.create(**kwargs)
        if streaming:
            return _GuardedAnthropicStream(
                response,
                self._scorer,
                self._on_fail,
                prompt,
                injection_threshold=self._injection_threshold,
            )
        text = _anthropic_response_text(response)
        await _ascore_and_gate(
            self._scorer,
            self._on_fail,
            prompt,
            text,
            injection_threshold=self._injection_threshold,
        )
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

    def __init__(self, stream, scorer, on_fail, prompt, *, injection_threshold=None):
        self._stream = stream
        self._scorer = scorer
        self._on_fail = on_fail
        self._prompt = prompt
        self._buffer = []
        self._token_count = 0
        self._injection_threshold = injection_threshold

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
                    await self._aperiodic_check()
            yield event
        await self._afinal_check()

    async def _aperiodic_check(self):
        text = "".join(self._buffer)
        await _ascore_and_gate(self._scorer, self._on_fail, self._prompt, text)

    async def _afinal_check(self):
        text = "".join(self._buffer)
        if text:
            await _ascore_and_gate(
                self._scorer,
                self._on_fail,
                self._prompt,
                text,
                injection_threshold=self._injection_threshold,
            )

    def _periodic_check(self):
        text = "".join(self._buffer)
        approved, cs = self._scorer.review(self._prompt, text)
        if not approved:
            _handle_failure(self._on_fail, self._prompt, text, cs)

    def _final_check(self):
        text = "".join(self._buffer)
        if text:
            _score_and_gate(
                self._scorer,
                self._on_fail,
                self._prompt,
                text,
                injection_threshold=self._injection_threshold,
            )


# â”€â”€ Bedrock proxy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _has_bedrock_shape(client) -> bool:
    """True if client exposes ``converse()`` and ``invoke_model()`` (boto3 Bedrock)."""
    return callable(getattr(client, "converse", None)) and callable(
        getattr(client, "invoke_model", None),
    )


def _bedrock_response_text(response: dict) -> str:
    """Extract text from Bedrock Converse API response."""
    with contextlib.suppress(KeyError, IndexError, TypeError):
        return response["output"]["message"]["content"][0]["text"] or ""
    return ""


def _extract_bedrock_prompt(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        return str(block["text"])
            if isinstance(content, str):
                return content
    return ""


def _extract_bedrock_stream_delta(event: dict) -> str | None:
    with contextlib.suppress(KeyError, TypeError):
        val = event["contentBlockDelta"]["delta"]["text"]
        return str(val) if val is not None else None
    return None


class _BedrockProxy:
    """Wraps a boto3 Bedrock Runtime client with coherence scoring."""

    def __init__(self, client, scorer, on_fail, *, injection_threshold=None):
        self._client = client
        self._scorer = scorer
        self._on_fail = on_fail
        self._injection_threshold = injection_threshold

    def converse(self, **kwargs):
        prompt = _extract_bedrock_prompt(kwargs.get("messages", []))
        response = self._client.converse(**kwargs)
        text = _bedrock_response_text(response)
        _score_and_gate(
            self._scorer,
            self._on_fail,
            prompt,
            text,
            injection_threshold=self._injection_threshold,
        )
        return response

    def converse_stream(self, **kwargs):
        prompt = _extract_bedrock_prompt(kwargs.get("messages", []))
        response = self._client.converse_stream(**kwargs)
        return _GuardedBedrockStream(
            response,
            self._scorer,
            self._on_fail,
            prompt,
            injection_threshold=self._injection_threshold,
        )

    def __getattr__(self, name):
        return getattr(self._client, name)


class _GuardedBedrockStream:
    """Wraps Bedrock converse_stream with periodic coherence checks."""

    def __init__(self, response, scorer, on_fail, prompt, *, injection_threshold=None):
        self._response = response
        self._scorer = scorer
        self._on_fail = on_fail
        self._prompt = prompt
        self._buffer: list[str] = []
        self._token_count = 0
        self._injection_threshold = injection_threshold

    def __iter__(self):
        stream = self._response.get("stream", self._response)
        for event in stream:
            delta = _extract_bedrock_stream_delta(event)
            if delta:
                self._buffer.append(delta)
                self._token_count += 1
                if self._token_count % STREAM_CHECK_INTERVAL == 0:
                    self._periodic_check()
            yield event
        self._final_check()

    def __aiter__(self):
        return self._aiter_impl()

    async def _aiter_impl(self):
        stream = self._response.get("stream", self._response)
        async for event in stream:
            delta = _extract_bedrock_stream_delta(event)
            if delta:
                self._buffer.append(delta)
                self._token_count += 1
                if self._token_count % STREAM_CHECK_INTERVAL == 0:
                    await _ascore_and_gate(
                        self._scorer,
                        self._on_fail,
                        self._prompt,
                        "".join(self._buffer),
                    )
            yield event
        text = "".join(self._buffer)
        if text:
            await _ascore_and_gate(
                self._scorer,
                self._on_fail,
                self._prompt,
                text,
                injection_threshold=self._injection_threshold,
            )

    def _periodic_check(self):
        text = "".join(self._buffer)
        approved, cs = self._scorer.review(self._prompt, text)
        if not approved:
            _handle_failure(self._on_fail, self._prompt, text, cs)

    def _final_check(self):
        text = "".join(self._buffer)
        if text:
            _score_and_gate(
                self._scorer,
                self._on_fail,
                self._prompt,
                text,
                injection_threshold=self._injection_threshold,
            )


# â”€â”€ Gemini proxy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _has_gemini_shape(client) -> bool:
    """True if client exposes ``generate_content()`` (google-generativeai)."""
    return callable(getattr(client, "generate_content", None))


def _extract_gemini_prompt(args: tuple, kwargs: dict) -> str:
    contents = args[0] if args else kwargs.get("contents", "")
    if isinstance(contents, str):
        return contents
    if isinstance(contents, list):
        for item in reversed(contents):
            if isinstance(item, str):
                return item
            if isinstance(item, dict):
                parts = item.get("parts", [])
                for p in parts:
                    if isinstance(p, str):
                        return p
                    if isinstance(p, dict) and "text" in p:
                        return str(p["text"])
    return str(contents)


class _GeminiProxy:
    """Wraps a google.generativeai GenerativeModel with coherence scoring."""

    def __init__(self, client, scorer, on_fail, *, injection_threshold=None):
        self._client = client
        self._scorer = scorer
        self._on_fail = on_fail
        self._injection_threshold = injection_threshold

    def generate_content(self, *args, **kwargs):
        prompt = _extract_gemini_prompt(args, kwargs)
        streaming = kwargs.get("stream", False)
        response = self._client.generate_content(*args, **kwargs)
        if streaming:
            return _GuardedGeminiStream(
                response,
                self._scorer,
                self._on_fail,
                prompt,
                injection_threshold=self._injection_threshold,
            )
        text = getattr(response, "text", "") or ""
        _score_and_gate(
            self._scorer,
            self._on_fail,
            prompt,
            text,
            injection_threshold=self._injection_threshold,
        )
        return response

    def __getattr__(self, name):
        return getattr(self._client, name)


class _GuardedGeminiStream:
    """Wraps a Gemini streaming response with periodic coherence checks."""

    def __init__(self, stream, scorer, on_fail, prompt, *, injection_threshold=None):
        self._stream = stream
        self._scorer = scorer
        self._on_fail = on_fail
        self._prompt = prompt
        self._buffer: list[str] = []
        self._token_count = 0
        self._injection_threshold = injection_threshold

    def __iter__(self):
        for chunk in self._stream:
            text = getattr(chunk, "text", None)
            if text:
                self._buffer.append(text)
                self._token_count += 1
                if self._token_count % STREAM_CHECK_INTERVAL == 0:
                    self._periodic_check()
            yield chunk
        self._final_check()

    def __aiter__(self):
        return self._aiter_impl()

    async def _aiter_impl(self):
        async for chunk in self._stream:
            text = getattr(chunk, "text", None)
            if text:
                self._buffer.append(text)
                self._token_count += 1
                if self._token_count % STREAM_CHECK_INTERVAL == 0:
                    await _ascore_and_gate(
                        self._scorer,
                        self._on_fail,
                        self._prompt,
                        "".join(self._buffer),
                    )
            yield chunk
        text = "".join(self._buffer)
        if text:
            await _ascore_and_gate(
                self._scorer,
                self._on_fail,
                self._prompt,
                text,
                injection_threshold=self._injection_threshold,
            )

    def _periodic_check(self):
        text = "".join(self._buffer)
        approved, cs = self._scorer.review(self._prompt, text)
        if not approved:
            _handle_failure(self._on_fail, self._prompt, text, cs)

    def _final_check(self):
        text = "".join(self._buffer)
        if text:
            _score_and_gate(
                self._scorer,
                self._on_fail,
                self._prompt,
                text,
                injection_threshold=self._injection_threshold,
            )


# â”€â”€ Cohere proxy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _has_cohere_shape(client) -> bool:
    """True if client exposes ``chat()`` without OpenAI-compatible shape (Cohere v2)."""
    if _has_openai_shape(client):
        return False
    return callable(getattr(client, "chat", None)) and not callable(
        getattr(getattr(client, "chat", None), "completions", None),
    )


class _CohereProxy:
    """Wraps a Cohere client with coherence scoring."""

    def __init__(self, client, scorer, on_fail, *, injection_threshold=None):
        self._client = client
        self._scorer = scorer
        self._on_fail = on_fail
        self._injection_threshold = injection_threshold

    def chat(self, **kwargs):
        prompt = kwargs.get("message", "")
        response = self._client.chat(**kwargs)
        text = getattr(response, "text", "") or ""
        _score_and_gate(
            self._scorer,
            self._on_fail,
            prompt,
            text,
            injection_threshold=self._injection_threshold,
        )
        return response

    def chat_stream(self, **kwargs):
        prompt = kwargs.get("message", "")
        response = self._client.chat_stream(**kwargs)
        return _GuardedCohereStream(
            response,
            self._scorer,
            self._on_fail,
            prompt,
            injection_threshold=self._injection_threshold,
        )

    def __getattr__(self, name):
        return getattr(self._client, name)


class _GuardedCohereStream:
    """Wraps a Cohere chat_stream with periodic coherence checks."""

    def __init__(self, stream, scorer, on_fail, prompt, *, injection_threshold=None):
        self._stream = stream
        self._scorer = scorer
        self._on_fail = on_fail
        self._prompt = prompt
        self._buffer: list[str] = []
        self._token_count = 0
        self._injection_threshold = injection_threshold

    def __iter__(self):
        for event in self._stream:
            text = getattr(event, "text", None)
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
            text = getattr(event, "text", None)
            if text:
                self._buffer.append(text)
                self._token_count += 1
                if self._token_count % STREAM_CHECK_INTERVAL == 0:
                    await _ascore_and_gate(
                        self._scorer,
                        self._on_fail,
                        self._prompt,
                        "".join(self._buffer),
                    )
            yield event
        text = "".join(self._buffer)
        if text:
            await _ascore_and_gate(
                self._scorer,
                self._on_fail,
                self._prompt,
                text,
                injection_threshold=self._injection_threshold,
            )

    def _periodic_check(self):
        text = "".join(self._buffer)
        approved, cs = self._scorer.review(self._prompt, text)
        if not approved:
            _handle_failure(self._on_fail, self._prompt, text, cs)

    def _final_check(self):
        text = "".join(self._buffer)
        if text:
            _score_and_gate(
                self._scorer,
                self._on_fail,
                self._prompt,
                text,
                injection_threshold=self._injection_threshold,
            )
