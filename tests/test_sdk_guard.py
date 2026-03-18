# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” SDK Guard Tests

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from director_ai.core.exceptions import HallucinationError
from director_ai.core.types import CoherenceScore
from director_ai.integrations.sdk_guard import (
    _extract_prompt,
    get_score,
    guard,
    score,
)

# â”€â”€ Fake SDK scaffolding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_FakeOpenAI = type("OpenAI", (), {"__module__": "openai"})
_FakeAnthropic = type("Anthropic", (), {"__module__": "anthropic"})
_FakeUnknown = type("SomeClient", (), {"__module__": "some_lib"})
_FakeVLLM = type("VLLMClient", (), {"__module__": "vllm.client"})
_FakeGroq = type("Groq", (), {"__module__": "groq"})
_FakeLiteLLM = type("LiteLLM", (), {"__module__": "litellm"})


def _make_openai_client(response_text="The sky is blue."):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=response_text),
        delta=SimpleNamespace(content=None),
    )
    response = SimpleNamespace(choices=[choice])
    completions = MagicMock()
    completions.create = MagicMock(return_value=response)
    chat = SimpleNamespace(completions=completions)
    client = _FakeOpenAI()
    client.chat = chat
    return client, response


def _make_anthropic_client(response_text="The sky is blue."):
    block = SimpleNamespace(text=response_text)
    response = SimpleNamespace(content=[block])
    messages = MagicMock()
    messages.create = MagicMock(return_value=response)
    client = _FakeAnthropic()
    client.messages = messages
    return client, response


def _make_openai_stream_client(tokens):
    chunks = []
    for t in tokens:
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=t))],
        )
        chunks.append(chunk)
    completions = MagicMock()
    completions.create = MagicMock(return_value=iter(chunks))
    chat = SimpleNamespace(completions=completions)
    client = _FakeOpenAI()
    client.chat = chat
    return client


# â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@pytest.mark.consumer
class TestOpenAIGuard:
    def test_pass(self):
        client, resp = _make_openai_client("The sky is blue.")
        guarded = guard(
            client,
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        result = guarded.chat.completions.create(
            messages=[{"role": "user", "content": "What color is the sky?"}],
        )
        assert result is resp

    def test_fail_raises(self):
        client, _ = _make_openai_client("Mars has two moons named Phobos and Deimos.")
        guarded = guard(
            client,
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
        )
        with pytest.raises(HallucinationError) as exc_info:
            guarded.chat.completions.create(
                messages=[{"role": "user", "content": "What color is the sky?"}],
            )
        assert exc_info.value.score.score < 0.6

    def test_streaming_final_check(self):
        tokens = ["The ", "sky ", "is ", "blue."]
        client = _make_openai_stream_client(tokens)
        guarded = guard(
            client,
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        stream = guarded.chat.completions.create(
            messages=[{"role": "user", "content": "What color is the sky?"}],
            stream=True,
        )
        collected = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                collected.append(delta)
        assert "".join(collected) == "The sky is blue."


@pytest.mark.consumer
class TestAnthropicGuard:
    def test_pass(self):
        client, resp = _make_anthropic_client("The sky is blue.")
        guarded = guard(
            client,
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        result = guarded.messages.create(
            messages=[{"role": "user", "content": "What color is the sky?"}],
        )
        assert result is resp

    def test_fail_raises(self):
        client, _ = _make_anthropic_client(
            "Mars has two moons named Phobos and Deimos.",
        )
        guarded = guard(
            client,
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
        )
        with pytest.raises(HallucinationError):
            guarded.messages.create(
                messages=[{"role": "user", "content": "What color is the sky?"}],
            )


@pytest.mark.consumer
class TestOnFailModes:
    def test_log_mode(self, caplog):
        client, resp = _make_openai_client(
            "Mars has two moons named Phobos and Deimos.",
        )
        guarded = guard(
            client,
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
            on_fail="log",
        )
        with caplog.at_level(logging.WARNING, logger="DirectorAI.guard"):
            result = guarded.chat.completions.create(
                messages=[{"role": "user", "content": "What color is the sky?"}],
            )
        assert result is resp
        assert "Hallucination" in caplog.text

    def test_metadata_mode(self):
        client, resp = _make_openai_client(
            "Mars has two moons named Phobos and Deimos.",
        )
        guarded = guard(
            client,
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
            on_fail="metadata",
        )
        result = guarded.chat.completions.create(
            messages=[{"role": "user", "content": "What color is the sky?"}],
        )
        assert result is resp
        score = get_score()
        assert score is not None
        assert score.score < 0.6


@pytest.mark.consumer
class TestPromptExtraction:
    def test_single_user_message(self):
        msgs = [{"role": "user", "content": "Hello"}]
        assert _extract_prompt(msgs) == "Hello"

    def test_multi_message(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Sure."},
            {"role": "user", "content": "Follow-up"},
        ]
        assert _extract_prompt(msgs) == "Follow-up"

    def test_content_blocks(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {"type": "image_url", "image_url": {"url": "http://..."}},
                ],
            },
        ]
        assert _extract_prompt(msgs) == "Describe this."

    def test_no_user_message(self):
        msgs = [{"role": "system", "content": "Only system."}]
        assert _extract_prompt(msgs) == "Only system."


def _make_openai_shaped_client(cls, response_text="The sky is blue."):
    """Build a client with OpenAI-compatible shape from any class."""
    choice = SimpleNamespace(
        message=SimpleNamespace(content=response_text),
        delta=SimpleNamespace(content=None),
    )
    response = SimpleNamespace(choices=[choice])
    completions = MagicMock()
    completions.create = MagicMock(return_value=response)
    chat = SimpleNamespace(completions=completions)
    client = cls()
    client.chat = chat
    return client, response


@pytest.mark.consumer
class TestDuckTypeDetection:
    def test_vllm_client(self):
        client, resp = _make_openai_shaped_client(_FakeVLLM)
        guarded = guard(
            client,
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        result = guarded.chat.completions.create(
            messages=[{"role": "user", "content": "What color is the sky?"}],
        )
        assert result is resp

    def test_groq_client(self):
        client, resp = _make_openai_shaped_client(_FakeGroq)
        guarded = guard(
            client,
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        result = guarded.chat.completions.create(
            messages=[{"role": "user", "content": "What color is the sky?"}],
        )
        assert result is resp

    def test_litellm_client(self):
        client, resp = _make_openai_shaped_client(_FakeLiteLLM)
        guarded = guard(
            client,
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        result = guarded.chat.completions.create(
            messages=[{"role": "user", "content": "What color is the sky?"}],
        )
        assert result is resp

    def test_no_shape_raises(self):
        client = _FakeUnknown()
        with pytest.raises(TypeError, match="Unsupported client type"):
            guard(client, facts={"k": "v"})

    def test_anthropic_shape_with_no_chat(self):
        client, resp = _make_anthropic_client("The sky is blue.")
        assert not hasattr(client, "chat")
        guarded = guard(
            client,
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        result = guarded.messages.create(
            messages=[{"role": "user", "content": "What color is the sky?"}],
        )
        assert result is resp


@pytest.mark.consumer
class TestUnknownClient:
    def test_unknown_raises_type_error(self):
        client = _FakeUnknown()
        with pytest.raises(TypeError, match="Unsupported client type"):
            guard(client, facts={"k": "v"})


@pytest.mark.consumer
class TestHallucinationErrorReExport:
    def test_langchain_reexport(self):
        from director_ai.integrations.langchain import (
            HallucinationError as LcHallucinationError,
        )

        assert LcHallucinationError is HallucinationError

    def test_top_level_export(self):
        from director_ai import (
            HallucinationError as TopHallucinationError,
        )

        assert TopHallucinationError is HallucinationError


@pytest.mark.consumer
class TestScore:
    def test_score_basic(self):
        cs = score("What color is the sky?", "The sky is blue.", use_nli=False)
        assert isinstance(cs, CoherenceScore)
        assert 0.0 <= cs.score <= 1.0

    def test_score_with_facts_approved(self):
        cs = score(
            "What color is the sky?",
            "The sky is blue.",
            facts={"sky": "The sky is blue due to Rayleigh scattering."},
            use_nli=False,
        )
        assert cs.score >= 0.5

    def test_score_with_facts_hallucination(self):
        cs = score(
            "What color is the sky?",
            "Mars has two moons named Phobos and Deimos.",
            facts={"sky": "The sky is blue due to Rayleigh scattering."},
            threshold=0.6,
            use_nli=False,
        )
        assert cs.score < 0.6

    def test_score_with_profile(self):
        cs = score(
            "What is the refund policy?",
            "Refunds within 30 days.",
            facts={"refund": "Refunds within 30 days only."},
            profile="fast",
        )
        assert isinstance(cs, CoherenceScore)

    def test_score_returns_coherence_score(self):
        cs = score("Hello", "Hi there!", use_nli=False)
        assert isinstance(cs, CoherenceScore)
        assert hasattr(cs, "score")
        assert hasattr(cs, "h_logical")
        assert hasattr(cs, "h_factual")
