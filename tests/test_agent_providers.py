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
