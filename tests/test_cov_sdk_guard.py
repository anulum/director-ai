"""Coverage for sdk_guard.py — OpenAI/Anthropic proxy, streaming, async paths."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from director_ai.core.scorer import CoherenceScorer
from director_ai.integrations.sdk_guard import (
    _AnthropicMessagesProxy,
    _extract_anthropic_event_text,
    _extract_prompt,
    _extract_stream_delta,
    _handle_failure,
    _OpenAICompletionsProxy,
    guard,
)


class TestExtractPrompt:
    def test_user_string(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert _extract_prompt(msgs) == "hello"

    def test_user_list_blocks(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "block text"}]}]
        assert _extract_prompt(msgs) == "block text"

    def test_no_user_fallback(self):
        msgs = [{"role": "system", "content": "sys"}]
        result = _extract_prompt(msgs)
        assert "sys" in result


class TestHandleFailure:
    def test_raise(self):
        from director_ai.integrations.sdk_guard import HallucinationError

        score = SimpleNamespace(score=0.3)
        with pytest.raises(HallucinationError):
            _handle_failure("raise", "q", "a", score)

    def test_log(self):
        score = SimpleNamespace(score=0.3)
        _handle_failure("log", "q", "a", score)

    def test_metadata(self):
        score = SimpleNamespace(score=0.3)
        _handle_failure("metadata", "q", "a", score)


class TestOpenAIProxy:
    def test_sync_create(self):
        scorer = CoherenceScorer(use_nli=False)
        original = MagicMock()
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="The sky is blue.")),
            ],
        )
        original.create.return_value = response

        proxy = _OpenAICompletionsProxy(original, scorer, "log")
        result = proxy.create(messages=[{"role": "user", "content": "sky?"}])
        assert result is response

    def test_sync_streaming(self):
        scorer = CoherenceScorer(use_nli=False)
        original = MagicMock()

        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))],
        )
        original.create.return_value = [chunk]

        proxy = _OpenAICompletionsProxy(original, scorer, "log")
        result = proxy.create(messages=[{"role": "user", "content": "q"}], stream=True)
        chunks = list(result)
        assert len(chunks) == 1

    def test_getattr_passthrough(self):
        scorer = CoherenceScorer(use_nli=False)
        original = MagicMock()
        original.some_method = "value"

        proxy = _OpenAICompletionsProxy(original, scorer, "log")
        assert proxy.some_method == "value"


class TestAnthropicProxy:
    def test_sync_create(self):
        scorer = CoherenceScorer(use_nli=False)
        original = MagicMock()
        response = SimpleNamespace(content=[SimpleNamespace(text="The sky is blue.")])
        original.create.return_value = response

        proxy = _AnthropicMessagesProxy(original, scorer, "log")
        result = proxy.create(messages=[{"role": "user", "content": "sky?"}])
        assert result is response

    def test_sync_streaming(self):
        scorer = CoherenceScorer(use_nli=False)
        original = MagicMock()

        event = SimpleNamespace(text="hello")
        original.create.return_value = [event]

        proxy = _AnthropicMessagesProxy(original, scorer, "log")
        result = proxy.create(messages=[{"role": "user", "content": "q"}], stream=True)
        events = list(result)
        assert len(events) == 1


class TestExtractStreamDelta:
    def test_valid_chunk(self):
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))],
        )
        assert _extract_stream_delta(chunk) == "hi"

    def test_none_content(self):
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None))],
        )
        assert _extract_stream_delta(chunk) is None

    def test_no_choices(self):
        chunk = SimpleNamespace(choices=[])
        assert _extract_stream_delta(chunk) is None


class TestExtractAnthropicEventText:
    def test_text_attr(self):
        event = SimpleNamespace(text="hello")
        assert _extract_anthropic_event_text(event) == "hello"

    def test_delta_dict(self):
        event = SimpleNamespace(text=None, delta={"text": "world"})
        assert _extract_anthropic_event_text(event) == "world"

    def test_no_text(self):
        event = SimpleNamespace(text=None, delta=None)
        assert _extract_anthropic_event_text(event) is None


class TestGuardFunction:
    def test_guard_openai_shape(self):
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_completions = MagicMock()
        mock_client.chat = mock_chat
        mock_chat.completions = mock_completions

        guarded = guard(mock_client, on_fail="log")
        assert guarded is mock_client

    def test_guard_anthropic_shape(self):
        mock_client = MagicMock(spec=[])
        mock_client.messages = MagicMock()
        mock_client.messages.create = MagicMock()

        guarded = guard(mock_client, on_fail="log")
        assert guarded is mock_client

    def test_guard_unsupported(self):
        mock_client = MagicMock(spec=[])

        with pytest.raises(TypeError, match="Unsupported"):
            guard(mock_client, on_fail="log")
