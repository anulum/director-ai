# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""AWS Bedrock SDK guard proxy."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from .base import STREAM_CHECK_INTERVAL, _ascore_and_gate, _score_and_gate


def _has_bedrock_shape(client: Any) -> bool:
    """Return True if client exposes ``converse()`` and ``invoke_model()`` (boto3 Bedrock)."""
    return callable(getattr(client, "converse", None)) and callable(
        getattr(client, "invoke_model", None),
    )


def _bedrock_response_text(response: dict[str, Any]) -> str:
    """Extract text from Bedrock Converse API response."""
    output = response.get("output") if isinstance(response, dict) else None
    message = output.get("message") if isinstance(output, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list) or not content or not isinstance(content[0], dict):
        return ""
    text = content[0].get("text")
    return text if isinstance(text, str) else ""


def _extract_bedrock_prompt(messages: list[dict[str, Any]]) -> str:
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


def _extract_bedrock_stream_delta(event: dict[str, Any]) -> str | None:
    """Extract text delta content from a Bedrock stream event."""
    block = event.get("contentBlockDelta") if isinstance(event, dict) else None
    delta = block.get("delta") if isinstance(block, dict) else None
    val = delta.get("text") if isinstance(delta, dict) else None
    return str(val) if val is not None else None


class _BedrockProxy:
    """Wraps a boto3 Bedrock Runtime client with coherence scoring."""

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

    def converse(self, **kwargs: Any) -> Any:
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

    def converse_stream(self, **kwargs: Any) -> Any:
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

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _GuardedBedrockStream:
    """Wraps Bedrock converse_stream with periodic coherence checks."""

    def __init__(
        self,
        response: Any,
        scorer: Any,
        on_fail: str,
        prompt: str,
        *,
        injection_threshold: float | None = None,
    ) -> None:
        self._response = response
        self._scorer = scorer
        self._on_fail = on_fail
        self._prompt = prompt
        self._buffer: list[str] = []
        self._token_count = 0
        self._injection_threshold = injection_threshold

    def __iter__(self) -> Iterator[Any]:
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

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._aiter_impl()

    async def _aiter_impl(self) -> AsyncIterator[Any]:
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
