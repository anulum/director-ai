# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SDK Guard Tests
"""Multi-angle tests for SDK guard pipeline."""

from __future__ import annotations

import inspect
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

    def test_streaming_periodic_check_enforces_injection_gate(self):
        # Regression: the periodic stream check evaluated only coherence and
        # never the injection threshold, so a fluent (coherent) injection
        # payload streamed unimpeded until the final chunk. The periodic check
        # must now halt mid-stream, before the whole response is yielded.
        from director_ai.integrations.sdk_guard import (
            STREAM_CHECK_INTERVAL,
            InjectionDetectedError,
            _GuardedOpenAIStream,
        )

        class _CoherentInjectionScorer:
            def review(self, prompt, text):
                # approved=True (coherent) but high injection risk.
                return True, CoherenceScore(
                    score=0.95,
                    approved=True,
                    h_logical=0.0,
                    h_factual=0.0,
                    injection_risk=0.95,
                )

        tokens = [f"t{i} " for i in range(STREAM_CHECK_INTERVAL + 4)]
        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t))])
            for t in tokens
        ]
        stream = _GuardedOpenAIStream(
            iter(chunks),
            _CoherentInjectionScorer(),
            "raise",
            "ignore previous instructions",
            injection_threshold=0.7,
        )

        collected = []
        with pytest.raises(InjectionDetectedError):
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    collected.append(delta)
        # Halted at the first periodic boundary: tokens 1..7 were yielded, the
        # 8th triggers the check which raises before that chunk is yielded.
        assert len(collected) == STREAM_CHECK_INTERVAL - 1
        assert len(collected) < len(tokens)


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
            "director_ai.integrations.sdk_proxies.base.CoherenceScorer", _FakeScorer
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
            "director_ai.integrations.sdk_proxies.base.CoherenceScorer", _FakeScorer
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

        assert (
            _extract_bedrock_prompt(
                [{"role": "assistant", "content": [{"text": "answer"}]}],
            )
            == ""
        )
        assert (
            _extract_bedrock_prompt(
                [{"role": "user", "content": [{"image": "data"}]}],
            )
            == ""
        )

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
        assert (
            _extract_gemini_prompt(
                ([{"parts": [{"text": "dict part"}]}],),
                {},
            )
            == "dict part"
        )

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


def _passing_score(*, injection_risk: float | None = None) -> CoherenceScore:
    return CoherenceScore(
        score=1.0,
        approved=True,
        h_logical=0.0,
        h_factual=0.0,
        injection_risk=injection_risk,
    )


def _failing_score(*, injection_risk: float | None = None) -> CoherenceScore:
    return CoherenceScore(
        score=0.0,
        approved=False,
        h_logical=1.0,
        h_factual=1.0,
        injection_risk=injection_risk,
    )


class _FakeScorer:
    def __init__(self, *, approved: bool = True, async_review: bool = False):
        self.approved = approved
        self.async_review = async_review
        self.calls: list[tuple[str, str]] = []

    def review(self, prompt, response):
        self.calls.append((prompt, response))
        result = (
            self.approved,
            _passing_score() if self.approved else _failing_score(),
        )
        if self.async_review:

            async def _result():
                return result

            return _result()
        return result


class _AsyncIterable:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        self._iter = iter(self._items)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class TestSdkGuardCoverageEdges:
    def test_guard_rejects_bad_on_fail_and_shape_helpers(self):
        from director_ai.integrations.sdk_guard import (
            _has_anthropic_shape,
            _has_mistral_shape,
            _has_openai_shape,
            _has_pydantic_ai_shape,
        )

        with pytest.raises(ValueError, match="on_fail"):
            guard(_FakeUnknown(), on_fail="ignore")

        assert not _has_openai_shape(SimpleNamespace())
        assert not _has_openai_shape(SimpleNamespace(chat=SimpleNamespace()))
        assert not _has_anthropic_shape(SimpleNamespace(chat=object()))
        assert not _has_anthropic_shape(SimpleNamespace())
        assert not _has_mistral_shape(SimpleNamespace(chat=SimpleNamespace()))
        assert not _has_pydantic_ai_shape(SimpleNamespace(run_sync=lambda: None))

    def test_prompt_and_text_extractors_cover_empty_and_object_paths(self):
        from director_ai.integrations.sdk_guard import (
            _anthropic_response_text,
            _extract_anthropic_event_text,
            _extract_prompt,
            _mistral_response_text,
            _openai_response_text,
            _pydantic_ai_output_text,
        )

        assert _extract_prompt([{"role": "user", "content": [{"type": "image"}]}]) == (
            "[{'type': 'image'}]"
        )
        assert _openai_response_text(SimpleNamespace(choices=[])) == ""
        assert (
            _openai_response_text(
                SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=123))]
                )
            )
            == ""
        )
        assert _anthropic_response_text(SimpleNamespace(content=[])) == ""
        assert (
            _anthropic_response_text(
                SimpleNamespace(content=[SimpleNamespace(text=123)])
            )
            == ""
        )
        assert _extract_anthropic_event_text(SimpleNamespace(text="direct")) == "direct"
        assert (
            _extract_anthropic_event_text(SimpleNamespace(delta={"text": "delta"}))
            == "delta"
        )
        assert _extract_anthropic_event_text(SimpleNamespace(delta={})) is None
        assert _mistral_response_text(SimpleNamespace(choices=[])) == ""
        assert (
            _mistral_response_text(
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=["a", {"text": "b"}, object()]
                            )
                        )
                    ]
                )
            )
            == "ab"
        )
        assert (
            _mistral_response_text(
                SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=3))]
                )
            )
            == ""
        )
        assert _pydantic_ai_output_text(SimpleNamespace(output=b"hi\xff")) == "hi�"
        assert _pydantic_ai_output_text(SimpleNamespace(output={"b": 1, "a": 2})) == (
            '{"a": 2, "b": 1}'
        )

    def test_failure_and_injection_policies(self, caplog):
        from director_ai.core.exceptions import InjectionDetectedError
        from director_ai.integrations.sdk_guard import (
            _check_injection,
            _handle_injection_failure,
        )

        cs = _passing_score(injection_risk=0.9)
        with pytest.raises(InjectionDetectedError):
            _handle_injection_failure("raise", "q", "r", cs)
        with caplog.at_level(logging.WARNING, logger="DirectorAI.guard"):
            _handle_injection_failure("log", "q", "r", cs)
        assert "Injection detected" in caplog.text
        _handle_injection_failure("metadata", "q", "r", cs)
        assert get_score() is cs
        _check_injection("metadata", "q", "r", _passing_score(), None)
        _check_injection("metadata", "q", "r", _passing_score(injection_risk=0.1), 0.7)

    @pytest.mark.asyncio
    async def test_sync_and_async_gate_coroutine_paths(self):
        from director_ai.integrations.sdk_guard import _ascore_and_gate, _score_and_gate

        sync_scorer = _FakeScorer(async_review=True)
        cs = _score_and_gate(sync_scorer, "metadata", "q", "r")
        assert isinstance(cs, CoherenceScore)
        assert get_score() is cs
        assert sync_scorer.calls == [("q", "r")]

        async_scorer = _FakeScorer(async_review=True)
        async_cs = await _ascore_and_gate(async_scorer, "metadata", "q", "r")
        assert get_score() is async_cs

        failing = _FakeScorer(approved=False)
        with pytest.raises(HallucinationError):
            await _ascore_and_gate(failing, "raise", "q", "bad")

    @pytest.mark.asyncio
    async def test_async_openai_and_anthropic_create_and_stream_paths(self):
        from director_ai.integrations.sdk_guard import (
            _AnthropicMessagesProxy,
            _OpenAICompletionsProxy,
        )

        openai_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )
        openai_original = SimpleNamespace(
            create=AsyncMock(return_value=openai_response)
        )
        openai_proxy = _OpenAICompletionsProxy(openai_original, _FakeScorer(), "raise")
        result = await openai_proxy.create(messages=[{"role": "user", "content": "q"}])
        assert result is openai_response
        assert inspect.iscoroutinefunction(openai_proxy.create)
        assert openai_proxy._original is openai_original

        stream_chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="tok"))]
        )
        openai_original.create = AsyncMock(
            return_value=_AsyncIterable([stream_chunk, SimpleNamespace(choices=[])])
        )
        stream = await openai_proxy.create(
            messages=[{"role": "user", "content": "q"}],
            stream=True,
        )
        assert [chunk async for chunk in stream] == [
            stream_chunk,
            SimpleNamespace(choices=[]),
        ]

        anthropic_response = SimpleNamespace(content=[SimpleNamespace(text="answer")])
        anthropic_original = SimpleNamespace(
            create=AsyncMock(return_value=anthropic_response)
        )
        anthropic_proxy = _AnthropicMessagesProxy(
            anthropic_original,
            _FakeScorer(),
            "raise",
        )
        result = await anthropic_proxy.create(
            messages=[{"role": "user", "content": "q"}]
        )
        assert result is anthropic_response

        anthropic_original.create = AsyncMock(
            return_value=_AsyncIterable(
                [SimpleNamespace(text="tok"), SimpleNamespace(delta={})]
            )
        )
        stream = await anthropic_proxy.create(
            messages=[{"role": "user", "content": "q"}],
            stream=True,
        )
        assert [event async for event in stream] == [
            SimpleNamespace(text="tok"),
            SimpleNamespace(delta={}),
        ]

    @pytest.mark.asyncio
    async def test_async_stream_wrappers_cover_periodic_and_final_paths(self):
        from director_ai.integrations.sdk_guard import (
            STREAM_CHECK_INTERVAL,
            _GuardedBedrockStream,
            _GuardedCohereStream,
            _GuardedGeminiStream,
            _GuardedOpenAIStream,
        )

        openai_chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=str(i)))]
            )
            for i in range(STREAM_CHECK_INTERVAL)
        ]
        scorer = _FakeScorer()
        assert [
            chunk
            async for chunk in _GuardedOpenAIStream(
                _AsyncIterable(openai_chunks),
                scorer,
                "raise",
                "prompt",
            )
        ] == openai_chunks
        assert scorer.calls[-1][1] == "".join(
            str(i) for i in range(STREAM_CHECK_INTERVAL)
        )

        bedrock_events = [
            {"contentBlockDelta": {"delta": {"text": str(i)}}}
            for i in range(STREAM_CHECK_INTERVAL)
        ]
        scorer = _FakeScorer()
        assert [
            event
            async for event in _GuardedBedrockStream(
                {"stream": _AsyncIterable(bedrock_events)},
                scorer,
                "raise",
                "prompt",
            )
        ] == bedrock_events
        assert scorer.calls

        gemini_events = [
            SimpleNamespace(text=str(i)) for i in range(STREAM_CHECK_INTERVAL)
        ]
        scorer = _FakeScorer()
        assert [
            event
            async for event in _GuardedGeminiStream(
                _AsyncIterable(gemini_events),
                scorer,
                "raise",
                "prompt",
            )
        ] == gemini_events
        assert scorer.calls

        cohere_events = [
            SimpleNamespace(text=str(i)) for i in range(STREAM_CHECK_INTERVAL)
        ]
        scorer = _FakeScorer()
        assert [
            event
            async for event in _GuardedCohereStream(
                _AsyncIterable(cohere_events),
                scorer,
                "raise",
                "prompt",
            )
        ] == cohere_events
        assert scorer.calls

    def test_bedrock_gemini_and_cohere_proxy_sync_paths(self):
        from director_ai.integrations.sdk_guard import (
            _BedrockProxy,
            _CohereProxy,
            _GeminiProxy,
        )

        bedrock_client = SimpleNamespace(
            converse=MagicMock(
                return_value={"output": {"message": {"content": [{"text": "answer"}]}}}
            ),
            converse_stream=MagicMock(
                return_value={
                    "stream": [
                        {"contentBlockDelta": {"delta": {"text": "a"}}},
                        {"not_text": True},
                    ]
                }
            ),
            extra="bedrock-extra",
        )
        bedrock = _BedrockProxy(bedrock_client, _FakeScorer(), "raise")
        assert bedrock.converse(messages=[{"role": "user", "content": "q"}])
        assert list(
            bedrock.converse_stream(messages=[{"role": "user", "content": "q"}])
        )
        assert bedrock.extra == "bedrock-extra"

        gemini_client = SimpleNamespace(
            generate_content=MagicMock(return_value=SimpleNamespace(text="answer")),
            extra="gemini-extra",
        )
        gemini = _GeminiProxy(gemini_client, _FakeScorer(), "raise")
        assert gemini.generate_content("prompt").text == "answer"
        gemini_client.generate_content.return_value = [
            SimpleNamespace(text="a"),
            SimpleNamespace(text=""),
        ]
        assert list(gemini.generate_content("prompt", stream=True))
        assert gemini.extra == "gemini-extra"

        cohere_client = SimpleNamespace(
            chat=MagicMock(return_value=SimpleNamespace(text="answer")),
            chat_stream=MagicMock(
                return_value=[SimpleNamespace(text="a"), SimpleNamespace(text="")]
            ),
            extra="cohere-extra",
        )
        cohere = _CohereProxy(cohere_client, _FakeScorer(), "raise")
        assert cohere.chat(message="prompt").text == "answer"
        assert list(cohere.chat_stream(message="prompt"))
        assert cohere.extra == "cohere-extra"

    def test_guard_selects_bedrock_gemini_and_cohere_shapes(self):
        bedrock_client = SimpleNamespace(
            converse=MagicMock(
                return_value={"output": {"message": {"content": [{"text": "answer"}]}}}
            ),
            converse_stream=MagicMock(return_value={"stream": []}),
            invoke_model=MagicMock(),
        )
        guarded = guard(
            bedrock_client,
            facts={"a": "answer"},
            threshold=0.0,
            use_nli=False,
        )
        assert guarded.converse(messages=[{"role": "user", "content": "q"}])

        gemini_client = SimpleNamespace(
            generate_content=MagicMock(return_value=SimpleNamespace(text="answer"))
        )
        guarded = guard(
            gemini_client,
            facts={"a": "answer"},
            threshold=0.0,
            use_nli=False,
        )
        assert guarded.generate_content("q").text == "answer"

        cohere_client = SimpleNamespace(
            chat=MagicMock(return_value=SimpleNamespace(text="answer"))
        )
        guarded = guard(
            cohere_client,
            facts={"a": "answer"},
            threshold=0.0,
            use_nli=False,
        )
        assert guarded.chat(message="q").text == "answer"

    def test_guard_selects_cohere_branch_without_facts(self, monkeypatch):
        class _CohereClient:
            extra = "cohere"

            def chat(self, **kwargs):
                return SimpleNamespace(text="answer")

        class _Scorer:
            def __init__(self, **kwargs):
                pass

            def review(self, prompt, response):
                return True, _passing_score()

        monkeypatch.setattr(
            "director_ai.integrations.sdk_guard.CoherenceScorer", _Scorer
        )

        guarded = guard(_CohereClient(), use_nli=False)

        assert guarded.chat(message="prompt").text == "answer"
        assert guarded.extra == "cohere"

    def test_score_and_gate_runs_coroutine_when_no_loop_is_running(self):
        from director_ai.integrations.sdk_guard import _score_and_gate

        scorer = _FakeScorer(async_review=True)

        result = _score_and_gate(scorer, "raise", "q", "r")

        assert isinstance(result, CoherenceScore)
        assert scorer.calls == [("q", "r")]

    def test_check_injection_high_risk_metadata_path(self):
        from director_ai.integrations.sdk_guard import _check_injection

        cs = _passing_score(injection_risk=0.95)

        _check_injection("metadata", "q", "r", cs, 0.7)

        assert get_score() is cs

    def test_proxy_getattr_delegates_for_openai_anthropic_mistral_and_pydantic(self):
        from director_ai.integrations.sdk_guard import (
            _AnthropicMessagesProxy,
            _MistralChatProxy,
            _OpenAICompletionsProxy,
            _PydanticAIProxy,
        )

        scorer = _FakeScorer()
        original = SimpleNamespace(create=MagicMock(), extra="openai")
        assert _OpenAICompletionsProxy(original, scorer, "raise").extra == "openai"

        original = SimpleNamespace(create=MagicMock(), extra="anthropic")
        assert _AnthropicMessagesProxy(original, scorer, "raise").extra == "anthropic"

        original = SimpleNamespace(complete=MagicMock(), extra="mistral")
        assert _MistralChatProxy(original, scorer, "raise").extra == "mistral"

        agent = SimpleNamespace(run_sync=MagicMock(), run=AsyncMock(), extra="agent")
        assert _PydanticAIProxy(agent, scorer, "raise").extra == "agent"

    def test_sync_stream_periodic_failures_and_empty_final_paths(self):
        from director_ai.integrations.sdk_guard import (
            STREAM_CHECK_INTERVAL,
            _GuardedAnthropicStream,
            _GuardedBedrockStream,
            _GuardedCohereStream,
            _GuardedGeminiStream,
            _GuardedOpenAIStream,
        )

        openai_chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=str(i)))]
            )
            for i in range(STREAM_CHECK_INTERVAL)
        ]
        with pytest.raises(HallucinationError):
            list(
                _GuardedOpenAIStream(
                    openai_chunks, _FakeScorer(approved=False), "raise", "p"
                )
            )
        assert list(
            _GuardedOpenAIStream(
                [SimpleNamespace(choices=[])], _FakeScorer(), "raise", "p"
            )
        )

        anthropic_events = [
            SimpleNamespace(text=str(i)) for i in range(STREAM_CHECK_INTERVAL)
        ]
        with pytest.raises(HallucinationError):
            list(
                _GuardedAnthropicStream(
                    anthropic_events,
                    _FakeScorer(approved=False),
                    "raise",
                    "p",
                )
            )
        assert list(
            _GuardedAnthropicStream(
                [SimpleNamespace(delta={})], _FakeScorer(), "raise", "p"
            )
        )

        bedrock_events = [
            {"contentBlockDelta": {"delta": {"text": str(i)}}}
            for i in range(STREAM_CHECK_INTERVAL)
        ]
        with pytest.raises(HallucinationError):
            list(
                _GuardedBedrockStream(
                    {"stream": bedrock_events},
                    _FakeScorer(approved=False),
                    "raise",
                    "p",
                )
            )
        assert list(
            _GuardedBedrockStream({"stream": [{}]}, _FakeScorer(), "raise", "p")
        )

        gemini_events = [
            SimpleNamespace(text=str(i)) for i in range(STREAM_CHECK_INTERVAL)
        ]
        with pytest.raises(HallucinationError):
            list(
                _GuardedGeminiStream(
                    gemini_events,
                    _FakeScorer(approved=False),
                    "raise",
                    "p",
                )
            )
        assert list(
            _GuardedGeminiStream(
                [SimpleNamespace(text="")], _FakeScorer(), "raise", "p"
            )
        )

        cohere_events = [
            SimpleNamespace(text=str(i)) for i in range(STREAM_CHECK_INTERVAL)
        ]
        with pytest.raises(HallucinationError):
            list(
                _GuardedCohereStream(
                    cohere_events,
                    _FakeScorer(approved=False),
                    "raise",
                    "p",
                )
            )
        assert list(
            _GuardedCohereStream(
                [SimpleNamespace(text="")], _FakeScorer(), "raise", "p"
            )
        )

    def test_sync_stream_periodic_approved_and_final_paths(self):
        from director_ai.integrations.sdk_guard import (
            STREAM_CHECK_INTERVAL,
            _GuardedAnthropicStream,
            _GuardedBedrockStream,
            _GuardedCohereStream,
            _GuardedGeminiStream,
            _GuardedOpenAIStream,
        )

        openai_chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=str(i)))]
            )
            for i in range(STREAM_CHECK_INTERVAL)
        ]
        scorer = _FakeScorer()
        list(_GuardedOpenAIStream(openai_chunks, scorer, "raise", "p"))
        assert scorer.calls

        anthropic_events = [SimpleNamespace(text=str(i)) for i in range(2)]
        scorer = _FakeScorer()
        list(_GuardedAnthropicStream(anthropic_events, scorer, "raise", "p"))
        assert scorer.calls[-1] == ("p", "01")

        bedrock_events = [
            {"contentBlockDelta": {"delta": {"text": str(i)}}}
            for i in range(STREAM_CHECK_INTERVAL)
        ]
        scorer = _FakeScorer()
        list(_GuardedBedrockStream({"stream": bedrock_events}, scorer, "raise", "p"))
        assert scorer.calls

        gemini_events = [
            SimpleNamespace(text=str(i)) for i in range(STREAM_CHECK_INTERVAL)
        ]
        scorer = _FakeScorer()
        list(_GuardedGeminiStream(gemini_events, scorer, "raise", "p"))
        assert scorer.calls

        cohere_events = [
            SimpleNamespace(text=str(i)) for i in range(STREAM_CHECK_INTERVAL)
        ]
        scorer = _FakeScorer()
        list(_GuardedCohereStream(cohere_events, scorer, "raise", "p"))
        assert scorer.calls

    @pytest.mark.asyncio
    async def test_anthropic_sync_stream_and_async_periodic_paths(self):
        import director_ai.integrations.sdk_guard as sdk_guard

        assert (
            sdk_guard._extract_anthropic_event_text(SimpleNamespace(delta="bad"))
            is None
        )

        original = SimpleNamespace(
            create=MagicMock(return_value=[SimpleNamespace(text="tok")])
        )
        proxy = sdk_guard._AnthropicMessagesProxy(original, _FakeScorer(), "raise")
        assert list(
            proxy.create(messages=[{"role": "user", "content": "q"}], stream=True)
        )

        scorer = _FakeScorer()
        events = [
            SimpleNamespace(text=str(i)) for i in range(sdk_guard.STREAM_CHECK_INTERVAL)
        ]
        assert [
            event
            async for event in sdk_guard._GuardedAnthropicStream(
                _AsyncIterable(events),
                scorer,
                "raise",
                "p",
            )
        ] == events
        assert scorer.calls

    @pytest.mark.asyncio
    async def test_async_stream_empty_final_paths(self):
        from director_ai.integrations.sdk_guard import (
            _GuardedAnthropicStream,
            _GuardedBedrockStream,
            _GuardedCohereStream,
            _GuardedGeminiStream,
            _GuardedOpenAIStream,
        )

        assert [
            chunk
            async for chunk in _GuardedOpenAIStream(
                _AsyncIterable([SimpleNamespace(choices=[])]),
                _FakeScorer(),
                "raise",
                "p",
            )
        ]
        assert [
            event
            async for event in _GuardedAnthropicStream(
                _AsyncIterable([SimpleNamespace(delta={})]),
                _FakeScorer(),
                "raise",
                "p",
            )
        ]
        assert [
            event
            async for event in _GuardedBedrockStream(
                {"stream": _AsyncIterable([{}])},
                _FakeScorer(),
                "raise",
                "p",
            )
        ]
        assert [
            event
            async for event in _GuardedGeminiStream(
                _AsyncIterable([SimpleNamespace(text="")]),
                _FakeScorer(),
                "raise",
                "p",
            )
        ]
        assert [
            event
            async for event in _GuardedCohereStream(
                _AsyncIterable([SimpleNamespace(text="")]),
                _FakeScorer(),
                "raise",
                "p",
            )
        ]

    def test_pydantic_prompt_and_output_fallbacks(self):
        from director_ai.integrations.sdk_guard import (
            _extract_pydantic_ai_prompt,
            _pydantic_ai_content_text,
            _pydantic_ai_output_text,
        )

        assert _extract_pydantic_ai_prompt((), {}) == ""
        assert (
            _extract_pydantic_ai_prompt((["a", SimpleNamespace(content="b")],), {})
            == "a b"
        )
        assert _extract_pydantic_ai_prompt((), {"user_prompt": 123}) == "123"
        assert _pydantic_ai_content_text(SimpleNamespace()) == "namespace()"
        assert _pydantic_ai_output_text(SimpleNamespace(output=object())).startswith(
            "<object object"
        )

    def test_mistral_shape_rejects_openai_compatible_client_first(self):
        from director_ai.integrations.sdk_guard import _has_mistral_shape

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=MagicMock()),
                complete=MagicMock(),
            )
        )

        assert not _has_mistral_shape(client)

    def test_remaining_prompt_extraction_branch_edges(self):
        from director_ai.integrations.sdk_guard import (
            _extract_gemini_prompt,
            _extract_prompt,
        )

        assert _extract_prompt([{"role": "user", "content": []}]) == "[]"
        assert _extract_prompt([{"role": "user", "content": {"kind": "tool"}}]) == (
            "{'kind': 'tool'}"
        )
        fallback_contents = [{"parts": [{"no_text": "x"}, 3]}]
        assert _extract_gemini_prompt((fallback_contents,), {}) == str(
            fallback_contents
        )
        object_contents = [object()]
        assert _extract_gemini_prompt((object_contents,), {}) == str(object_contents)

    def test_anthropic_sync_periodic_approved_branch(self):
        from director_ai.integrations.sdk_guard import (
            STREAM_CHECK_INTERVAL,
            _GuardedAnthropicStream,
        )

        scorer = _FakeScorer()
        events = [SimpleNamespace(text=str(i)) for i in range(STREAM_CHECK_INTERVAL)]

        assert list(_GuardedAnthropicStream(events, scorer, "raise", "p")) == events
        assert scorer.calls
