# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Agent Provider Tests (STRONG)
"""Multi-angle tests for CoherenceAgent provider routing.

Covers: default mock generator, LLM URL routing, mutual exclusion,
unknown provider guard, missing API key guard, OpenAI/Anthropic
instantiation, parametrised providers, pipeline integration,
and performance documentation.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from director_ai.core.agent import CoherenceAgent


class TestAgentProvider:
    def test_default_uses_mock(self):
        agent = CoherenceAgent()
        from director_ai.core.actor import MockGenerator

        assert isinstance(agent.generator, MockGenerator)

    def test_llm_api_url_uses_llm_generator(self):
        agent = CoherenceAgent(llm_api_url="http://localhost:8080/completion")
        from director_ai.core.actor import LLMGenerator

        assert isinstance(agent.generator, LLMGenerator)

    def test_provider_and_url_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            CoherenceAgent(llm_api_url="http://x", provider="openai")

    def test_unknown_provider_raises(self):
        with (
            patch.dict(os.environ, {"FAKE_KEY": "x"}),
            pytest.raises(ValueError, match="Unknown provider"),
        ):
            CoherenceAgent(provider="gemini")

    def test_missing_api_key_raises(self):
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(ValueError, match="OPENAI_API_KEY not set"),
        ):
            CoherenceAgent(provider="openai")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    def test_openai_provider_creates_openai(self):
        agent = CoherenceAgent(provider="openai")
        from director_ai.integrations.providers import OpenAIProvider

        assert isinstance(agent.generator, OpenAIProvider)
        assert agent.generator.api_key == "sk-test-key"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"})
    def test_anthropic_provider_creates_anthropic(self):
        agent = CoherenceAgent(provider="anthropic")
        from director_ai.integrations.providers import AnthropicProvider

        assert isinstance(agent.generator, AnthropicProvider)
        assert agent.generator.api_key == "sk-ant-test"

    def test_process_still_works_with_default(self):
        agent = CoherenceAgent()
        result = agent.process("What color is the sky?")
        assert result.output
        assert result.candidates_evaluated > 0

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8080/completion",
            "http://localhost:11434/v1/completions",
            "https://api.example.com/v1/completions",
        ],
    )
    def test_parametrised_llm_urls(self, url):
        from director_ai.core.actor import LLMGenerator

        agent = CoherenceAgent(llm_api_url=url)
        assert isinstance(agent.generator, LLMGenerator)

    @pytest.mark.parametrize("bad_provider", ["gemini", "cohere", "mistral", ""])
    def test_parametrised_unknown_providers(self, bad_provider):
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            if bad_provider:
                with pytest.raises(ValueError):
                    CoherenceAgent(provider=bad_provider)


class TestAgentProviderPerformanceDoc:
    """Document agent provider pipeline performance."""

    def test_default_agent_creates_fast(self):
        import time

        t0 = time.perf_counter()
        CoherenceAgent()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 1000, f"Agent creation took {elapsed_ms:.0f}ms"

    def test_process_returns_required_fields(self):
        agent = CoherenceAgent()
        result = agent.process("Test query")
        assert hasattr(result, "output")
        assert hasattr(result, "coherence")
        assert hasattr(result, "halted")
        assert hasattr(result, "candidates_evaluated")
