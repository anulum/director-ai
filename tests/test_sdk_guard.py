# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SDK Guard Tests
"""Multi-angle tests for SDK guard pipeline."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
_FakeMistral = type("Mistral", (), {"__module__": "mistralai"})
_FakePydanticAgent = type("Agent", (), {"__module__": "pydantic_ai.agent"})


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


def _make_mistral_client(response_text="The sky is blue."):
    choice = SimpleNamespace(message=SimpleNamespace(content=response_text))
    response = SimpleNamespace(choices=[choice])
    chat = SimpleNamespace(complete=MagicMock(return_value=response))
    client = _FakeMistral()
    client.chat = chat
    return client, response


def _make_async_mistral_client(response_text="The sky is blue."):
    choice = SimpleNamespace(message=SimpleNamespace(content=response_text))
    response = SimpleNamespace(choices=[choice])
    chat = SimpleNamespace(complete=AsyncMock(return_value=response))
    client = _FakeMistral()
    client.chat = chat
    return client, response


def _make_pydantic_ai_agent(response_output="The sky is blue."):
    response = SimpleNamespace(output=response_output)
    agent = _FakePydanticAgent()
    agent.run_sync = MagicMock(return_value=response)
    agent.run = AsyncMock(return_value=response)
    return agent, response


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
class TestMistralGuard:
    def test_pass(self):
        client, resp = _make_mistral_client("The sky is blue.")
        original_complete = client.chat.complete
        guarded = guard(
            client,
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        result = guarded.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "What color is the sky?"}],
        )
        assert result is resp
        original_complete.assert_called_once()

    def test_fail_raises(self):
        client, _ = _make_mistral_client("Mars has two moons named Phobos and Deimos.")
        guarded = guard(
            client,
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
        )
        with pytest.raises(HallucinationError):
            guarded.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": "What color is the sky?"}],
            )

    def test_response_content_chunks_are_scored(self):
        client, _ = _make_mistral_client(
            [
                {"type": "text", "text": "Mars has two moons "},
                SimpleNamespace(text="named Phobos and Deimos."),
            ],
        )
        guarded = guard(
            client,
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
        )
        with pytest.raises(HallucinationError):
            guarded.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": "What color is the sky?"}],
            )

    @pytest.mark.asyncio
    async def test_async_complete_is_guarded(self):
        client, resp = _make_async_mistral_client("The sky is blue.")
        original_complete = client.chat.complete
        guarded = guard(
            client,
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        result = await guarded.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "What color is the sky?"}],
        )
        assert result is resp
        original_complete.assert_awaited_once()


@pytest.mark.consumer
class TestPydanticAIGuard:
    def test_run_sync_pass(self):
        agent, resp = _make_pydantic_ai_agent("The sky is blue.")
        guarded = guard(
            agent,
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        result = guarded.run_sync("What color is the sky?")
        assert result is resp
        agent.run_sync.assert_called_once_with("What color is the sky?")

    def test_run_sync_fail_raises(self):
        agent, _ = _make_pydantic_ai_agent(
            "Mars has two moons named Phobos and Deimos."
        )
        guarded = guard(
            agent,
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
        )
        with pytest.raises(HallucinationError):
            guarded.run_sync(user_prompt="What color is the sky?")

    @pytest.mark.asyncio
    async def test_async_run_is_guarded(self):
        agent, resp = _make_pydantic_ai_agent("The sky is blue.")
        guarded = guard(
            agent,
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        result = await guarded.run("What color is the sky?")
        assert result is resp
        agent.run.assert_awaited_once_with("What color is the sky?")

    def test_structured_output_is_scored(self):
        structured = SimpleNamespace(
            model_dump_json=MagicMock(
                return_value='{"answer":"Mars has two moons named Phobos and Deimos."}',
            ),
        )
        agent, _ = _make_pydantic_ai_agent(structured)
        guarded = guard(
            agent,
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
        )
        with pytest.raises(HallucinationError):
            guarded.run_sync("What color is the sky?")
        structured.model_dump_json.assert_called_once()


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

    def test_score_with_profile_rejects_model_backed_without_nli(self):
        with pytest.raises(
            ValueError,
            match="require_model_backed_nli=True requires use_nli=True",
        ):
            score(
                "What color is the sky?",
                "The sky is blue.",
                profile="fast",
                require_model_backed_nli=True,
            )

    def test_score_with_profile_applies_model_backed_override(self, monkeypatch):
        captured: dict[str, object] = {}

        class _FakeCfg:
            use_nli = False
            coherence_require_model_backed_nli = False

            def build_scorer(self, store=None):
                captured["use_nli"] = self.use_nli
                captured["require_model_backed_nli"] = (
                    self.coherence_require_model_backed_nli
                )

                class _FakeScorer:
                    def review(self, _prompt, _response):
                        return True, CoherenceScore(
                            score=1.0,
                            approved=True,
                            h_logical=0.0,
                            h_factual=0.0,
                        )

                return _FakeScorer()

        monkeypatch.setattr(
            "director_ai.core.config.DirectorConfig.from_profile",
            lambda _name: _FakeCfg(),
        )

        cs = score(
            "Summarise this",
            "Summary text.",
            profile="fast",
            use_nli=True,
            require_model_backed_nli=True,
        )
        assert isinstance(cs, CoherenceScore)
        assert captured["use_nli"] is True
        assert captured["require_model_backed_nli"] is True

    def test_score_returns_coherence_score(self):
        cs = score("Hello", "Hi there!", use_nli=False)
        assert isinstance(cs, CoherenceScore)
        assert hasattr(cs, "score")
        assert hasattr(cs, "h_logical")
        assert hasattr(cs, "h_factual")

    def test_score_forwards_require_model_backed_nli(self, monkeypatch):
        captured: dict[str, object] = {}

        class _FakeScorer:
            def __init__(self, **kwargs):
                captured["init_kwargs"] = kwargs

            def review(self, _prompt, _response):
                return True, CoherenceScore(
                    score=1.0,
                    approved=True,
                    h_logical=0.0,
                    h_factual=0.0,
                )

        monkeypatch.setattr(
            "director_ai.integrations.sdk_guard.CoherenceScorer", _FakeScorer
        )

        cs = score(
            "What color is the sky?",
            "The sky is blue.",
            use_nli=False,
            require_model_backed_nli=True,
        )
        assert isinstance(cs, CoherenceScore)
        assert captured["init_kwargs"]["require_model_backed_nli"] is True

    def test_score_forwards_injection_fail_closed_flags(self, monkeypatch):
        captured: dict[str, object] = {}

        class _FakeScorer:
            def __init__(self, **kwargs):
                captured["init_kwargs"] = kwargs

            def enable_injection_detection(self, **kwargs):
                captured["injection_kwargs"] = kwargs

            def review(self, _prompt, _response):
                return True, CoherenceScore(
                    score=1.0,
                    approved=True,
                    h_logical=0.0,
                    h_factual=0.0,
                )

        monkeypatch.setattr(
            "director_ai.integrations.sdk_guard.CoherenceScorer", _FakeScorer
        )

        cs = score(
            "What color is the sky?",
            "The sky is blue.",
            use_nli=False,
            injection_detection=True,
            injection_require_model_backed_nli=True,
            injection_fail_closed_on_error=True,
        )
        assert isinstance(cs, CoherenceScore)
        assert captured["injection_kwargs"]["require_model_backed_nli"] is True
        assert captured["injection_kwargs"]["fail_closed_on_error"] is True

    def test_guard_forwards_hardening_flags_to_scorer(self, monkeypatch):
        captured: dict[str, object] = {}

        class _FakeScorer:
            def __init__(self, **kwargs):
                captured["init_kwargs"] = kwargs

            def enable_injection_detection(self, **kwargs):
                captured["injection_kwargs"] = kwargs

            def review(self, _prompt, _response):
                return True, CoherenceScore(
                    score=1.0,
                    approved=True,
                    h_logical=0.0,
                    h_factual=0.0,
                )

        monkeypatch.setattr(
            "director_ai.integrations.sdk_guard.CoherenceScorer", _FakeScorer
        )
        client, _ = _make_openai_client("The sky is blue.")
        guarded = guard(
            client,
            facts={"sky color": "The sky is blue."},
            use_nli=False,
            injection_detection=True,
            require_model_backed_nli=True,
            injection_require_model_backed_nli=True,
            injection_fail_closed_on_error=True,
        )
        guarded.chat.completions.create(
            messages=[{"role": "user", "content": "What color is the sky?"}],
        )

        assert captured["init_kwargs"]["require_model_backed_nli"] is True
        assert captured["injection_kwargs"]["require_model_backed_nli"] is True
        assert captured["injection_kwargs"]["fail_closed_on_error"] is True


class TestBedrockAdapterContracts:
    """Bedrock SDK guard helpers preserve prompt and stream text contracts."""

    def test_bedrock_response_text_extracts_first_text_block(self):
        from director_ai.integrations.sdk_guard import _bedrock_response_text

        response = {"output": {"message": {"content": [{"text": "Hello Bedrock"}]}}}

        assert _bedrock_response_text(response) == "Hello Bedrock"

    def test_bedrock_response_text_returns_empty_string_for_malformed_payload(self):
        from director_ai.integrations.sdk_guard import _bedrock_response_text

        assert _bedrock_response_text({}) == ""

    def test_bedrock_prompt_prefers_user_text_blocks(self):
        from director_ai.integrations.sdk_guard import _extract_bedrock_prompt

        messages = [{"role": "user", "content": [{"text": "hello bedrock"}]}]

        assert _extract_bedrock_prompt(messages) == "hello bedrock"

    def test_bedrock_prompt_accepts_plain_user_content(self):
        from director_ai.integrations.sdk_guard import _extract_bedrock_prompt

        messages = [{"role": "user", "content": "plain string"}]

        assert _extract_bedrock_prompt(messages) == "plain string"

    def test_bedrock_prompt_ignores_non_user_or_non_text_content(self):
        from director_ai.integrations.sdk_guard import _extract_bedrock_prompt

        assert _extract_bedrock_prompt(
            [{"role": "assistant", "content": [{"text": "answer"}]}],
        ) == ""
        assert _extract_bedrock_prompt(
            [{"role": "user", "content": [{"image": "data"}]}],
        ) == ""

    def test_bedrock_stream_delta_extracts_text_chunk(self):
        from director_ai.integrations.sdk_guard import _extract_bedrock_stream_delta

        event = {"contentBlockDelta": {"delta": {"text": "chunk"}}}

        assert _extract_bedrock_stream_delta(event) == "chunk"
        assert _extract_bedrock_stream_delta({}) is None


class TestGeminiAdapterPromptContracts:
    """Gemini SDK guard prompt extraction handles supported content shapes."""

    def test_gemini_prompt_uses_string_argument_or_keyword_contents(self):
        from director_ai.integrations.sdk_guard import _extract_gemini_prompt

        assert _extract_gemini_prompt(("tell me something",), {}) == "tell me something"
        assert _extract_gemini_prompt((), {"contents": "from kwargs"}) == "from kwargs"

    def test_gemini_prompt_uses_last_string_from_content_list(self):
        from director_ai.integrations.sdk_guard import _extract_gemini_prompt

        assert _extract_gemini_prompt((["first", "second"],), {}) == "second"

    def test_gemini_prompt_extracts_text_from_parts(self):
        from director_ai.integrations.sdk_guard import _extract_gemini_prompt

        assert _extract_gemini_prompt(([{"parts": ["part text"]}],), {}) == "part text"
        assert _extract_gemini_prompt(
            ([{"parts": [{"text": "dict part"}]}],),
            {},
        ) == "dict part"

    def test_gemini_prompt_falls_back_to_string_conversion(self):
        from director_ai.integrations.sdk_guard import _extract_gemini_prompt

        assert _extract_gemini_prompt((42,), {}) == "42"


class TestCohereAdapterShapeContracts:
    """Cohere detection must not capture OpenAI-compatible clients."""

    def test_cohere_shape_accepts_chat_client_without_completions(self):
        from unittest.mock import MagicMock

        from director_ai.integrations.sdk_guard import _has_cohere_shape

        client = MagicMock(spec=["chat"])
        client.chat = MagicMock(spec=[])

        assert _has_cohere_shape(client)

    def test_cohere_shape_rejects_openai_compatible_chat_client(self):
        from unittest.mock import MagicMock

        from director_ai.integrations.sdk_guard import _has_cohere_shape

        client = MagicMock()
        client.chat.completions.create = MagicMock()

        assert not _has_cohere_shape(client)
