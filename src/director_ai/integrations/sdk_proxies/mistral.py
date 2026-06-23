# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Mistral SDK guard proxy."""

from __future__ import annotations

import inspect
from typing import Any

from .base import _ascore_and_gate, _extract_prompt, _score_and_gate
from .openai import _has_openai_shape


def _has_mistral_shape(client: Any) -> bool:
    """Return True if client exposes ``client.chat.complete()`` (mistralai SDK)."""
    if _has_openai_shape(client):
        return False
    chat = getattr(client, "chat", None)
    return chat is not None and callable(getattr(chat, "complete", None))


class _MistralChatProxy:
    """Drop-in for ``client.chat`` in the Mistral Python SDK."""

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
        self.complete: Any = (
            self._acomplete_entry
            if inspect.iscoroutinefunction(original.complete)
            else self._sync_complete
        )

    def _sync_complete(self, **kwargs: Any) -> Any:
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

    async def _acomplete_entry(self, **kwargs: Any) -> Any:
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

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def _mistral_response_text(response: Any) -> str:
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
