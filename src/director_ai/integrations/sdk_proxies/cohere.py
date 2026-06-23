# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Cohere SDK guard proxy."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from .base import STREAM_CHECK_INTERVAL, _ascore_and_gate, _score_and_gate
from .openai import _has_openai_shape


def _has_cohere_shape(client: Any) -> bool:
    """Return True if client exposes ``chat()`` without OpenAI-compatible shape (Cohere v2)."""
    if _has_openai_shape(client):
        return False
    return callable(getattr(client, "chat", None)) and not callable(
        getattr(getattr(client, "chat", None), "completions", None),
    )


class _CohereProxy:
    """Wraps a Cohere client with coherence scoring."""

    def __init__(
        self,
        client: Any,
        scorer: Any,
        on_fail: str,
        *,
        injection_threshold: float | None = None,
    ) -> None:
        self._client = client
        self._scorer = scorer
        self._on_fail = on_fail
        self._injection_threshold = injection_threshold

    def chat(self, **kwargs: Any) -> Any:
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

    def chat_stream(self, **kwargs: Any) -> Any:
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

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _GuardedCohereStream:
    """Wraps a Cohere chat_stream with periodic coherence checks."""

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
            text = getattr(event, "text", None)
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
                        injection_threshold=self._injection_threshold,
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
