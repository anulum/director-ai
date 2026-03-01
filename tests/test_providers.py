# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — LLM Provider Adapter Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from unittest.mock import MagicMock, patch

import pytest

from director_ai.integrations import (
    AnthropicProvider,
    HuggingFaceProvider,
    LLMProvider,
    LocalProvider,
    OpenAIProvider,
)


class TestLLMProviderProtocol:
    """Tests for LLMProvider base class."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_init_defaults(self):
        p = OpenAIProvider(api_key="sk-test")
        assert p.model == "gpt-4o-mini"
        assert p.base_url == "https://api.openai.com/v1"
        assert p.timeout == 30

    def test_name(self):
        p = OpenAIProvider(model="gpt-4o")
        assert p.name == "openai/gpt-4o"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_candidates_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [
                {"message": {"content": "Response 1"}},
                {"message": {"content": "Response 2"}},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        p = OpenAIProvider(api_key="sk-test")
        candidates = p.generate_candidates("Hello", n=2)
        assert len(candidates) == 2
        assert candidates[0]["text"] == "Response 1"
        assert candidates[0]["source"] == "openai/gpt-4o-mini"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_candidates_timeout(self, mock_post):
        import requests as _req

        mock_post.side_effect = _req.exceptions.Timeout("timed out")
        p = OpenAIProvider(api_key="sk-test")
        candidates = p.generate_candidates("Hello", n=1)
        assert len(candidates) == 1
        assert candidates[0]["text"] == "[Timeout]"
        assert candidates[0]["source"] == "error"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_candidates_http_error(self, mock_post):
        import requests as _req

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = _req.exceptions.HTTPError("429")
        mock_post.return_value = mock_resp
        p = OpenAIProvider(api_key="sk-test")
        candidates = p.generate_candidates("Hello", n=1)
        assert len(candidates) == 1
        assert "HTTP Error" in candidates[0]["text"]

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_candidates_connection_error(self, mock_post):
        import requests as _req

        mock_post.side_effect = _req.exceptions.ConnectionError("refused")
        p = OpenAIProvider(api_key="sk-test")
        candidates = p.generate_candidates("Hello", n=1)
        assert len(candidates) == 1
        assert candidates[0]["text"] == "[Connection Error]"


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_init_defaults(self):
        p = AnthropicProvider(api_key="sk-ant-test")
        assert "claude" in p.model
        assert p.timeout == 60

    def test_name(self):
        p = AnthropicProvider(model="claude-opus-4-6")
        assert p.name == "anthropic/claude-opus-4-6"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_candidates_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": [{"type": "text", "text": "Answer"}]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        p = AnthropicProvider(api_key="sk-ant-test")
        candidates = p.generate_candidates("Hello", n=2)
        assert len(candidates) == 2
        assert candidates[0]["text"] == "Answer"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_candidates_timeout(self, mock_post):
        import requests as _req

        mock_post.side_effect = _req.exceptions.Timeout("timed out")
        p = AnthropicProvider(api_key="bad-key")
        candidates = p.generate_candidates("Hello", n=1)
        assert len(candidates) == 1
        assert candidates[0]["text"] == "[Timeout]"


class TestHuggingFaceProvider:
    """Tests for HuggingFaceProvider."""

    def test_init_defaults(self):
        p = HuggingFaceProvider(api_key="hf-test")
        assert "Mistral" in p.model
        assert p.timeout == 60

    def test_name(self):
        p = HuggingFaceProvider(model="meta-llama/Llama-3-8B")
        assert p.name == "huggingface/meta-llama/Llama-3-8B"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_candidates_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"generated_text": "Output text"}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        p = HuggingFaceProvider(api_key="hf-test")
        candidates = p.generate_candidates("Hello", n=1)
        assert len(candidates) == 1
        assert candidates[0]["text"] == "Output text"


class TestLocalProvider:
    """Tests for LocalProvider."""

    def test_init_defaults(self):
        p = LocalProvider()
        assert "localhost" in p.api_url
        assert p.model == ""

    def test_name_default(self):
        p = LocalProvider()
        assert p.name == "local/default"

    def test_name_with_model(self):
        p = LocalProvider(model="llama3")
        assert p.name == "local/llama3"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_candidates_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [
                {"message": {"content": "Local response"}},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        p = LocalProvider()
        candidates = p.generate_candidates("Hello", n=1)
        assert len(candidates) == 1
        assert candidates[0]["text"] == "Local response"
