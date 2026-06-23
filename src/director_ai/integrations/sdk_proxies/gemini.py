# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Google Gemini SDK guard proxy."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from .base import STREAM_CHECK_INTERVAL, _ascore_and_gate, _score_and_gate


def _has_gemini_shape(client: Any) -> bool:
    """Return True if client exposes ``generate_content()`` (google-generativeai)."""
    return callable(getattr(client, "generate_content", None))


def _extract_gemini_prompt(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
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

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
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

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _GuardedGeminiStream:
    """Wraps a Gemini streaming response with periodic coherence checks."""

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
        for chunk in self._stream:
            text = getattr(chunk, "text", None)
            if text:
                self._buffer.append(text)
                self._token_count += 1
                if self._token_count % STREAM_CHECK_INTERVAL == 0:
                    self._periodic_check()
            yield chunk
        self._final_check()

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._aiter_impl()

    async def _aiter_impl(self) -> AsyncIterator[Any]:
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
                        injection_threshold=self._injection_threshold,
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
