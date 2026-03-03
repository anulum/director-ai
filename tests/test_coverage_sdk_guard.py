"""Coverage tests for sdk_guard.py — guard() + streaming proxies."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from director_ai.core.exceptions import HallucinationError
from director_ai.integrations.sdk_guard import (
    _anthropic_response_text,
    _extract_anthropic_event_text,
    _extract_prompt,
    _extract_stream_delta,
    _handle_failure,
    _has_anthropic_shape,
    _has_openai_shape,
    _openai_response_text,
    get_score,
    guard,
)


class TestGuardBasics:
    def test_invalid_on_fail(self):
        client = _make_openai_client()
        with pytest.raises(ValueError, match="on_fail"):
            guard(client, on_fail="invalid")

    def test_unsupported_client_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            guard(object())

    def test_guard_openai_client(self):
        client = _make_openai_client()
        result = guard(client, facts={"sky": "blue"})
        assert result is client

    def test_guard_anthropic_client(self):
        client = _make_anthropic_client()
        result = guard(client, facts={"sky": "blue"})
        assert result is client


class TestShapeDetection:
    def test_openai_shape_true(self):
        assert _has_openai_shape(_make_openai_client())

    def test_openai_shape_no_chat(self):
        assert not _has_openai_shape(SimpleNamespace())

    def test_openai_shape_no_completions(self):
        assert not _has_openai_shape(SimpleNamespace(chat=SimpleNamespace()))

    def test_anthropic_shape_true(self):
        assert _has_anthropic_shape(_make_anthropic_client())

    def test_anthropic_shape_false_has_chat(self):
        obj = SimpleNamespace(
            chat=MagicMock(), messages=SimpleNamespace(create=lambda: None)
        )
        assert not _has_anthropic_shape(obj)

    def test_anthropic_shape_no_messages(self):
        assert not _has_anthropic_shape(SimpleNamespace())


class TestExtractPrompt:
    def test_str_content(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert _extract_prompt(msgs) == "hello"

    def test_list_content(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert _extract_prompt(msgs) == "hi"

    def test_non_str_content(self):
        msgs = [{"role": "user", "content": 42}]
        assert _extract_prompt(msgs) == "42"

    def test_no_user_message(self):
        msgs = [{"role": "system", "content": "sys"}]
        result = _extract_prompt(msgs)
        assert "sys" in result


class TestHandleFailure:
    def test_raise(self):
        score = SimpleNamespace(score=0.1)
        with pytest.raises(HallucinationError):
            _handle_failure("raise", "q", "r", score)

    def test_log(self):
        _handle_failure("log", "q", "r", SimpleNamespace(score=0.1))

    def test_metadata(self):
        score = SimpleNamespace(score=0.1)
        _handle_failure("metadata", "q", "r", score)
        assert get_score() is score


class TestOpenAIProxy:
    def test_sync_create_non_streaming(self):
        client = _make_openai_client()
        client = guard(client, facts={"sky": "blue"}, on_fail="log")
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "color?"}]
        )
        assert resp is not None

    def test_sync_create_streaming(self):
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="word "))]
        )
        client = _make_openai_client(streaming=True, chunks=[chunk] * 10)
        client = guard(client, facts={"sky": "blue"}, on_fail="log")
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}], stream=True
        )
        collected = list(stream)
        assert len(collected) == 10

    def test_getattr_fallthrough(self):
        client = _make_openai_client()
        client.chat.completions.list = lambda: "models"
        client = guard(client, on_fail="log")
        assert client.chat.completions.list() == "models"


class TestAnthropicProxy:
    def test_sync_create(self):
        client = _make_anthropic_client()
        client = guard(client, on_fail="log")
        resp = client.messages.create(messages=[{"role": "user", "content": "hello"}])
        assert resp is not None

    def test_sync_streaming(self):
        event = SimpleNamespace(text="token ")
        client = _make_anthropic_client(streaming=True, events=[event] * 10)
        client = guard(client, on_fail="log")
        stream = client.messages.create(
            messages=[{"role": "user", "content": "hi"}], stream=True
        )
        collected = list(stream)
        assert len(collected) == 10

    def test_getattr_fallthrough(self):
        client = _make_anthropic_client()
        client.messages.list = lambda: "msgs"
        client = guard(client, on_fail="log")
        assert client.messages.list() == "msgs"


class TestExtractors:
    def test_openai_response_text(self):
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))]
        )
        assert _openai_response_text(resp) == "hi"

    def test_openai_response_text_empty(self):
        assert _openai_response_text(SimpleNamespace()) == ""

    def test_stream_delta_present(self):
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="tok"))]
        )
        assert _extract_stream_delta(chunk) == "tok"

    def test_stream_delta_none(self):
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
        )
        assert _extract_stream_delta(chunk) is None

    def test_stream_delta_error(self):
        assert _extract_stream_delta(SimpleNamespace()) is None

    def test_anthropic_response_text(self):
        resp = SimpleNamespace(content=[SimpleNamespace(text="ok")])
        assert _anthropic_response_text(resp) == "ok"

    def test_anthropic_response_text_empty(self):
        assert _anthropic_response_text(SimpleNamespace()) == ""

    def test_anthropic_event_text_attr(self):
        assert _extract_anthropic_event_text(SimpleNamespace(text="hi")) == "hi"

    def test_anthropic_event_delta_dict(self):
        event = SimpleNamespace(text=None, delta={"text": "hello"})
        assert _extract_anthropic_event_text(event) == "hello"

    def test_anthropic_event_none(self):
        assert (
            _extract_anthropic_event_text(SimpleNamespace(text=None, delta=None))
            is None
        )

    def test_anthropic_event_delta_none_val(self):
        event = SimpleNamespace(text=None, delta={"text": None})
        assert _extract_anthropic_event_text(event) is None


class TestAsyncOpenAIStream:
    def test_aiter(self):
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="word "))]
        )
        client = _make_openai_client(
            streaming=True, chunks=[chunk] * 10, async_mode=True
        )
        client = guard(client, on_fail="log")
        stream = asyncio.get_event_loop().run_until_complete(
            client.chat.completions.create(
                messages=[{"role": "user", "content": "hi"}], stream=True
            )
        )

        async def collect():
            return [c async for c in stream]

        collected = asyncio.get_event_loop().run_until_complete(collect())
        assert len(collected) == 10


class TestAsyncAnthropicStream:
    def test_aiter(self):
        event = SimpleNamespace(text="token ")
        client = _make_anthropic_client(
            streaming=True, events=[event] * 10, async_mode=True
        )
        client = guard(client, on_fail="log")
        stream = asyncio.get_event_loop().run_until_complete(
            client.messages.create(
                messages=[{"role": "user", "content": "hi"}], stream=True
            )
        )

        async def collect():
            return [e async for e in stream]

        collected = asyncio.get_event_loop().run_until_complete(collect())
        assert len(collected) == 10


# ── Helpers ──────────────────────────────────────────────────────


def _make_openai_client(streaming=False, chunks=None, async_mode=False):
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="The sky is blue."))]
    )

    if async_mode:

        async def create(**kwargs):
            if kwargs.get("stream"):
                return _AsyncIter(chunks or [])
            return resp
    else:

        def create(**kwargs):
            if kwargs.get("stream"):
                return chunks or []
            return resp

    completions = SimpleNamespace(create=create)
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


def _make_anthropic_client(streaming=False, events=None, async_mode=False):
    resp = SimpleNamespace(content=[SimpleNamespace(text="The sky is blue.")])

    if async_mode:

        async def create(**kwargs):
            if kwargs.get("stream"):
                return _AsyncIter(events or [])
            return resp
    else:

        def create(**kwargs):
            if kwargs.get("stream"):
                return events or []
            return resp

    messages = SimpleNamespace(create=create)
    return SimpleNamespace(messages=messages)


class _AsyncIter:
    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)
