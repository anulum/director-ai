# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Native SDK interceptors for common LLM clients.

Usage::

    from director_ai import guard
    client = guard(OpenAI(), facts={"refund": "within 30 days"})
    resp = client.chat.completions.create(...)  # auto-scored
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from contextvars import ContextVar, copy_context
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
    require_model_backed_nli: bool = False,
    injection_require_model_backed_nli: bool = False,
    injection_fail_closed_on_error: bool = False,
) -> CoherenceScore:
    """Score a single prompt/response pair for hallucination.

    Returns a ``CoherenceScore`` without requiring an SDK client.
    When *injection_detection* is enabled, ``CoherenceScore.injection_risk``
    is populated with the intent-grounded injection risk score.
    """
    if profile is not None:
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig.from_profile(profile)
        if use_nli is not None:
            cfg.use_nli = use_nli
        if require_model_backed_nli:
            cfg.coherence_require_model_backed_nli = True
        if cfg.coherence_require_model_backed_nli and not cfg.use_nli:
            raise ValueError(
                "require_model_backed_nli=True requires use_nli=True "
                "when scoring with a profile",
            )
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
            require_model_backed_nli=require_model_backed_nli,
        )
    if injection_detection:
        scorer.enable_injection_detection(
            injection_threshold=injection_threshold,
            require_model_backed_nli=injection_require_model_backed_nli,
            fail_closed_on_error=injection_fail_closed_on_error,
        )
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
    require_model_backed_nli: bool = False,
    injection_require_model_backed_nli: bool = False,
    injection_fail_closed_on_error: bool = False,
) -> Any:
    """Wrap an LLM SDK client with coherence scoring.

    Supports seven SDK shapes:

    - **OpenAI-compatible** (``client.chat.completions.create``):
      OpenAI, vLLM, Groq, LiteLLM, Ollama, Together.
    - **Anthropic** (``client.messages.create``).
    - **AWS Bedrock** (``client.converse`` / ``client.converse_stream``).
    - **Google Gemini** (``client.generate_content``).
    - **Mistral** (``client.chat.complete``).
    - **Cohere** (``client.chat`` without ``client.completions``).
    - **Pydantic AI** (``agent.run_sync`` / ``agent.run``).

    When *injection_detection* is enabled, each response is additionally
    checked for prompt injection via intent-grounded NLI divergence.
    The *injection_threshold* controls sensitivity (0.0–1.0).

    Returns the guarded client. Some SDK clients are mutated in place; others
    return a proxy when their public surface cannot be safely patched.
    **Always use the return value**:
    ``client = guard(client, ...)``.
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
        require_model_backed_nli=require_model_backed_nli,
    )
    inj_threshold = injection_threshold if injection_detection else None
    if injection_detection:
        scorer.enable_injection_detection(
            injection_threshold=injection_threshold,
            require_model_backed_nli=injection_require_model_backed_nli,
            fail_closed_on_error=injection_fail_closed_on_error,
        )

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
    elif _has_mistral_shape(client):
        client.chat = _MistralChatProxy(
            client.chat,
            scorer,
            on_fail,
            injection_threshold=inj_threshold,
        )
    elif _has_pydantic_ai_shape(client):
        client = _PydanticAIProxy(
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
            "Expected a supported LLM SDK shape.",
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
    """Apply the configured hallucination failure policy."""
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
    """Synchronously score a response and enforce hallucination/injection gates."""
    result = scorer.review(query, response_text)
    if asyncio.iscoroutine(result):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures

            ctx = copy_context()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                approved, cs = pool.submit(ctx.run, asyncio.run, result).result()
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
    """Asynchronously score a response and enforce hallucination/injection gates."""
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
        """Create a guarded synchronous chat completion."""
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
        """Create a guarded asynchronous chat completion."""
        prompt = _extract_prompt(kwargs.get("messages", []))
        streaming = kwargs.get("stream", False)
        return await self._acreate(prompt, streaming, kwargs)

    async def _acreate(self, prompt, streaming, kwargs):
        """Await the original chat-completion call and gate the response."""
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
    """Extract assistant text from a chat completion response."""
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""


def _extract_stream_delta(chunk) -> str | None:
    """Extract text delta content from a streaming chat chunk."""
    choices = getattr(chunk, "choices", None)
    if not choices:
        return None
    delta_obj = getattr(choices[0], "delta", None)
    delta = getattr(delta_obj, "content", None)
    return str(delta) if delta is not None else None


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
        """Iterate an async chat stream while buffering emitted text."""
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
        """Run an asynchronous periodic score check for buffered text."""
        text = "".join(self._buffer)
        await _ascore_and_gate(self._scorer, self._on_fail, self._prompt, text)

    async def _afinal_check(self):
        """Run the final asynchronous score check for buffered text."""
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
        """Run a synchronous periodic score check for buffered text."""
        text = "".join(self._buffer)
        approved, cs = self._scorer.review(self._prompt, text)
        if not approved:
            _handle_failure(self._on_fail, self._prompt, text, cs)

    def _final_check(self):
        """Run the final synchronous score check for buffered text."""
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
        """Create a guarded synchronous vendor message."""
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
        """Create a guarded asynchronous vendor message."""
        prompt = _extract_prompt(kwargs.get("messages", []))
        streaming = kwargs.get("stream", False)
        return await self._acreate(prompt, streaming, kwargs)

    async def _acreate(self, prompt, streaming, kwargs):
        """Await the original vendor-message call and gate the response."""
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
    """Extract text from a vendor message response."""
    content = getattr(response, "content", None)
    if not content:
        return ""
    text = getattr(content[0], "text", None)
    return text if isinstance(text, str) else ""


def _extract_anthropic_event_text(event) -> str | None:
    """Extract text content from a vendor stream event."""
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
        """Iterate an async vendor stream while buffering emitted text."""
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
        """Run an asynchronous periodic score check for buffered text."""
        text = "".join(self._buffer)
        await _ascore_and_gate(self._scorer, self._on_fail, self._prompt, text)

    async def _afinal_check(self):
        """Run the final asynchronous score check for buffered text."""
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
        """Run a synchronous periodic score check for buffered text."""
        text = "".join(self._buffer)
        approved, cs = self._scorer.review(self._prompt, text)
        if not approved:
            _handle_failure(self._on_fail, self._prompt, text, cs)

    def _final_check(self):
        """Run the final synchronous score check for buffered text."""
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
    output = response.get("output") if isinstance(response, dict) else None
    message = output.get("message") if isinstance(output, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list) or not content or not isinstance(content[0], dict):
        return ""
    text = content[0].get("text")
    return text if isinstance(text, str) else ""


def _extract_bedrock_prompt(messages: list[dict]) -> str:
    """Extract the user prompt from Bedrock Converse messages."""
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
    """Extract text delta content from a Bedrock stream event."""
    block = event.get("contentBlockDelta") if isinstance(event, dict) else None
    delta = block.get("delta") if isinstance(block, dict) else None
    val = delta.get("text") if isinstance(delta, dict) else None
    return str(val) if val is not None else None


class _BedrockProxy:
    """Wraps a boto3 Bedrock Runtime client with coherence scoring."""

    def __init__(self, client, scorer, on_fail, *, injection_threshold=None):
        self._client = client
        self._scorer = scorer
        self._on_fail = on_fail
        self._injection_threshold = injection_threshold

    def converse(self, **kwargs):
        """Call Bedrock Converse and gate the returned message."""
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
        """Call Bedrock Converse streaming and wrap emitted events."""
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
        """Iterate an async Bedrock stream while buffering emitted text."""
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
        """Run a synchronous periodic score check for buffered text."""
        text = "".join(self._buffer)
        approved, cs = self._scorer.review(self._prompt, text)
        if not approved:
            _handle_failure(self._on_fail, self._prompt, text, cs)

    def _final_check(self):
        """Run the final synchronous score check for buffered text."""
        text = "".join(self._buffer)
        if text:
            _score_and_gate(
                self._scorer,
                self._on_fail,
                self._prompt,
                text,
                injection_threshold=self._injection_threshold,
            )


# Generate-content proxy


def _has_gemini_shape(client) -> bool:
    """True if client exposes ``generate_content()`` (google-generativeai)."""
    return callable(getattr(client, "generate_content", None))


def _extract_gemini_prompt(args: tuple, kwargs: dict) -> str:
    """Extract prompt text from generate_content inputs."""
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
        """Call generate_content and gate the response or stream."""
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
        """Iterate an async generate_content stream while buffering text."""
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
        """Run a synchronous periodic score check for buffered text."""
        text = "".join(self._buffer)
        approved, cs = self._scorer.review(self._prompt, text)
        if not approved:
            _handle_failure(self._on_fail, self._prompt, text, cs)

    def _final_check(self):
        """Run the final synchronous score check for buffered text."""
        text = "".join(self._buffer)
        if text:
            _score_and_gate(
                self._scorer,
                self._on_fail,
                self._prompt,
                text,
                injection_threshold=self._injection_threshold,
            )


# Mistral proxy


def _has_mistral_shape(client) -> bool:
    """True if client exposes ``client.chat.complete()`` (mistralai SDK)."""
    if _has_openai_shape(client):
        return False
    chat = getattr(client, "chat", None)
    return chat is not None and callable(getattr(chat, "complete", None))


class _MistralChatProxy:
    """Drop-in for ``client.chat`` in the Mistral Python SDK."""

    def __init__(self, original, scorer, on_fail, *, injection_threshold=None):
        self._original = original
        self._scorer = scorer
        self._on_fail = on_fail
        self._injection_threshold = injection_threshold
        self.complete: Any = (
            self._acomplete_entry
            if inspect.iscoroutinefunction(original.complete)
            else self._sync_complete
        )

    def _sync_complete(self, **kwargs):
        """Call a synchronous Mistral chat completion and gate it."""
        prompt = _extract_prompt(kwargs.get("messages", []))
        response = self._original.complete(**kwargs)
        text = _mistral_response_text(response)
        _score_and_gate(
            self._scorer,
            self._on_fail,
            prompt,
            text,
            injection_threshold=self._injection_threshold,
        )
        return response

    async def _acomplete_entry(self, **kwargs):
        """Call an asynchronous Mistral chat completion and gate it."""
        prompt = _extract_prompt(kwargs.get("messages", []))
        response = await self._original.complete(**kwargs)
        text = _mistral_response_text(response)
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


def _mistral_response_text(response) -> str:
    """Extract assistant text from a Mistral chat completion response."""
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, dict) and "text" in chunk:
                parts.append(str(chunk["text"]))
            else:
                text = getattr(chunk, "text", None)
                if text is not None:
                    parts.append(str(text))
        return "".join(parts)
    return ""


# Pydantic AI proxy


def _has_pydantic_ai_shape(client) -> bool:
    """True for Pydantic AI ``Agent`` instances with run APIs."""
    module = type(client).__module__
    if not module.startswith("pydantic_ai"):
        return False
    return callable(getattr(client, "run_sync", None)) and callable(
        getattr(client, "run", None),
    )


class _PydanticAIProxy:
    """Guard Pydantic AI ``Agent.run_sync`` and ``Agent.run`` outputs."""

    def __init__(self, agent, scorer, on_fail, *, injection_threshold=None):
        self._agent = agent
        self._scorer = scorer
        self._on_fail = on_fail
        self._injection_threshold = injection_threshold

    def run_sync(self, *args, **kwargs):
        """Run a synchronous Pydantic AI agent call and gate its output."""
        prompt = _extract_pydantic_ai_prompt(args, kwargs)
        result = self._agent.run_sync(*args, **kwargs)
        text = _pydantic_ai_output_text(result)
        _score_and_gate(
            self._scorer,
            self._on_fail,
            prompt,
            text,
            injection_threshold=self._injection_threshold,
        )
        return result

    async def run(self, *args, **kwargs):
        """Run an asynchronous Pydantic AI agent call and gate its output."""
        prompt = _extract_pydantic_ai_prompt(args, kwargs)
        result = await self._agent.run(*args, **kwargs)
        text = _pydantic_ai_output_text(result)
        await _ascore_and_gate(
            self._scorer,
            self._on_fail,
            prompt,
            text,
            injection_threshold=self._injection_threshold,
        )
        return result

    def __getattr__(self, name):
        return getattr(self._agent, name)


def _extract_pydantic_ai_prompt(args, kwargs) -> str:
    """Extract prompt text from Pydantic AI run arguments."""
    prompt = kwargs.get("user_prompt")
    if prompt is None and args:
        prompt = args[0]
    if prompt is None:
        return ""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list | tuple):
        return " ".join(_pydantic_ai_content_text(part) for part in prompt)
    return str(prompt)


def _pydantic_ai_content_text(content) -> str:
    """Return text for one Pydantic AI prompt content part."""
    if isinstance(content, str):
        return content
    text = getattr(content, "content", None)
    if text is not None:
        return str(text)
    return str(content)


def _pydantic_ai_output_text(result) -> str:
    """Serialise Pydantic AI run output for guard scoring."""
    output = getattr(result, "output", result)
    if isinstance(output, str):
        return output
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    model_dump_json = getattr(output, "model_dump_json", None)
    if callable(model_dump_json):
        return str(model_dump_json())
    if isinstance(output, dict | list | tuple):
        return json.dumps(output, sort_keys=True, default=str)
    return str(output)


# Cohere proxy


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
        """Call a Cohere chat completion and gate the response."""
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
        """Call a Cohere streaming chat completion and wrap emitted events."""
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
        """Iterate an async Cohere stream while buffering emitted text."""
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
        """Run a synchronous periodic score check for buffered text."""
        text = "".join(self._buffer)
        approved, cs = self._scorer.review(self._prompt, text)
        if not approved:
            _handle_failure(self._on_fail, self._prompt, text, cs)

    def _final_check(self):
        """Run the final synchronous score check for buffered text."""
        text = "".join(self._buffer)
        if text:
            _score_and_gate(
                self._scorer,
                self._on_fail,
                self._prompt,
                text,
                injection_threshold=self._injection_threshold,
            )
