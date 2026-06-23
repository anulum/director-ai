# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Anthropic Messages SDK guard proxy."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Iterator
from typing import Any

from .base import (
    STREAM_CHECK_INTERVAL,
    _ascore_and_gate,
    _extract_prompt,
    _score_and_gate,
)


def _has_anthropic_shape(client: Any) -> bool:
    """Return True if client exposes ``client.messages.create`` without ``client.chat``."""
    if getattr(client, "chat", None) is not None:
        return False
    messages = getattr(client, "messages", None)
    if messages is None:
        return False
    return callable(getattr(messages, "create", None))


class _AnthropicMessagesProxy:
    """Drop-in for ``client.messages``.

    Same sync / async dispatch pattern as
    :class:`_OpenAICompletionsProxy`.
    """

    def __init__(
        self,
        original: Any,
        scorer: Any,
        on_fail: str,
        *,
        injection_threshold: float | None = None,
    ) -> None:
        self._original = original
        self._scorer = scorer
        self._on_fail = on_fail
        self._injection_threshold = injection_threshold
        self.create: Any = (
            self._acreate_entry
            if inspect.iscoroutinefunction(original.create)
            else self._sync_create
        )

    def _sync_create(self, **kwargs: Any) -> Any:
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

    async def _acreate_entry(self, **kwargs: Any) -> Any:
        """Create a guarded asynchronous vendor message."""
        prompt = _extract_prompt(kwargs.get("messages", []))
        streaming = kwargs.get("stream", False)
        return await self._acreate(prompt, streaming, kwargs)

    async def _acreate(
        self, prompt: str, streaming: bool, kwargs: dict[str, Any]
    ) -> Any:
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

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def _anthropic_response_text(response: Any) -> str:
    """Extract text from a vendor message response."""
    content = getattr(response, "content", None)
    if not content:
        return ""
    text = getattr(content[0], "text", None)
    return text if isinstance(text, str) else ""


def _extract_anthropic_event_text(event: Any) -> str | None:
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

    def __init__(
        self,
        stream: Any,
        scorer: Any,
        on_fail: str,
        prompt: str,
        *,
        injection_threshold: float | None = None,
    ) -> None:
        self._stream = stream
        self._scorer = scorer
        self._on_fail = on_fail
        self._prompt = prompt
        self._buffer: list[str] = []
        self._token_count = 0
        self._injection_threshold = injection_threshold

    def __iter__(self) -> Iterator[Any]:
        for event in self._stream:
            text = _extract_anthropic_event_text(event)
            if text:
                self._buffer.append(text)
                self._token_count += 1
                if self._token_count % STREAM_CHECK_INTERVAL == 0:
                    self._periodic_check()
            yield event
        self._final_check()

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._aiter_impl()

    async def _aiter_impl(self) -> AsyncIterator[Any]:
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

    async def _aperiodic_check(self) -> None:
        """Run an asynchronous periodic score check for buffered text."""
        text = "".join(self._buffer)
        await _ascore_and_gate(
            self._scorer,
            self._on_fail,
            self._prompt,
            text,
            injection_threshold=self._injection_threshold,
        )

    async def _afinal_check(self) -> None:
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

    def _periodic_check(self) -> None:
        """Run a synchronous periodic score check for buffered text.

        Mirrors the final check (injection gate included) so a streamed
        prompt-injection is caught mid-stream rather than only on the last
        chunk after the whole response has already been yielded.
        """
        text = "".join(self._buffer)
        _score_and_gate(
            self._scorer,
            self._on_fail,
            self._prompt,
            text,
            injection_threshold=self._injection_threshold,
        )

    def _final_check(self) -> None:
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
