# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Proxy OpenAI Wire-Format Content Extraction

"""OpenAI wire-format content extraction for the guardrail proxy.

Pure parsing helpers over chat/completions response and stream-delta
shapes — no I/O, no scoring. Every helper degrades to an empty string
on an unexpected shape instead of raising, so a malformed upstream
payload can never crash the review path.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def _chat_completion_content(data: object) -> str:
    """Extract OpenAI-compatible chat content without exception control flow."""
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _iter_choice_deltas(chunk: object) -> Iterator[object]:
    """Yield every choice's ``delta`` mapping in wire order.

    A chat completion may carry more than one choice (``n>1``); reviewing
    only ``choices[0]`` lets a sensitive payload on a later choice reach
    the client unreviewed, so every disclosure path walks the full list.
    """
    if not isinstance(chunk, dict):
        return
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if isinstance(choice, dict):
            yield choice.get("delta")


def _delta_content(delta: object) -> str:
    """Text content from a single delta mapping."""
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _delta_tool_call_text(delta: object) -> str:
    """Tool-call name and argument text from a single delta mapping.

    OpenAI tool-call streams carry their payload in
    ``delta.tool_calls[].function.{name,arguments}`` rather than
    ``delta.content``; a response that only calls tools still discloses
    model output (the tool it invokes and the arguments it passes) that
    must reach the review buffer.
    """
    if not isinstance(delta, dict):
        return ""
    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list):
        return ""
    parts: list[str] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            parts.append(name)
        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments:
            parts.append(arguments)
    return "".join(parts)


def _stream_delta_content(chunk: object) -> str:
    """Extract OpenAI-compatible stream delta content across every choice."""
    return "".join(_delta_content(delta) for delta in _iter_choice_deltas(chunk))


def _stream_tool_call_content(chunk: object) -> str:
    """Extract tool-call name and argument deltas across every choice."""
    return "".join(_delta_tool_call_text(delta) for delta in _iter_choice_deltas(chunk))


def _stream_chat_content(chunk: object) -> str:
    """Reviewable chat-stream text across ALL choices in wire order.

    For each choice's delta, in choice order, the message content and the
    tool-call name/argument deltas are folded into one reviewed string.
    Walking every choice keeps a multi-choice chunk whose sensitive tool
    call rides on a later choice — or whose first choice is empty — from
    leaving the review buffer empty and bypassing the terminal review.
    """
    parts: list[str] = []
    for delta in _iter_choice_deltas(chunk):
        parts.append(_delta_content(delta))
        parts.append(_delta_tool_call_text(delta))
    return "".join(parts)


def _completion_text(data: object) -> str:
    """Extract legacy ``/v1/completions`` text without exception control flow."""
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    text = first.get("text")
    return text if isinstance(text, str) else ""


def _stream_text_content(chunk: object) -> str:
    """Extract a legacy completions stream text delta."""
    if not isinstance(chunk, dict):
        return ""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    text = first.get("text")
    return text if isinstance(text, str) else ""


def _completion_prompt(body: dict[str, Any]) -> str:
    """Extract the legacy ``prompt`` field (string or list of strings)."""
    prompt = body.get("prompt", "")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
        return prompt[0]
    return ""


def _extract_prompt(messages: list[dict[str, Any]]) -> str:
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
    return ""
