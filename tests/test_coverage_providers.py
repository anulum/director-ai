# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Coverage tests for providers.py — OpenAI/Anthropic/HF/Local adapters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from director_ai.integrations.providers import (
    AnthropicProvider,
    HuggingFaceProvider,
    LLMProvider,
    LocalProvider,
    OpenAIProvider,
)


class TestLLMProviderBase:
    def test_stream_generate_default(self):
        """Default stream_generate yields single candidate text."""

        class Stub(LLMProvider):
            @property
            def name(self):
                return "stub"

            def generate_candidates(self, prompt, n=3):
                return [{"text": "hello world"}]

        tokens = list(Stub().stream_generate("test"))
        assert tokens == ["hello world"]

    def test_stream_generate_empty_candidates(self):
        class Stub(LLMProvider):
            @property
            def name(self):
                return "stub"

            def generate_candidates(self, prompt, n=3):
                return []

        assert list(Stub().stream_generate("x")) == []


class TestOpenAIProvider:
    def test_name(self):
        p = OpenAIProvider(model="gpt-4o")
        assert p.name == "openai/gpt-4o"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "Yes"}}]}
        mock_post.return_value = mock_resp

        p = OpenAIProvider(api_key="k")
        result = p.generate_candidates("hi", n=1)
        assert len(result) == 1
        assert result[0]["text"] == "Yes"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("timeout")
        result = OpenAIProvider().generate_candidates("hi", n=1)
        assert result[0]["source"] == "error"
        assert "Timeout" in result[0]["text"]

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_http_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.HTTPError("403")
        result = OpenAIProvider().generate_candidates("hi", n=1)
        assert "HTTP Error" in result[0]["text"]

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("refused")
        result = OpenAIProvider().generate_candidates("hi", n=1)
        assert "Connection Error" in result[0]["text"]

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_parse_error(self, mock_post):
        mock_post.side_effect = ValueError("bad json")
        result = OpenAIProvider().generate_candidates("hi", n=1)
        assert "Parse Error" in result[0]["text"]

    @patch("director_ai.integrations.providers.requests.post")
    def test_stream_generate_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = [
            "",
            "data: " + '{"choices":[{"delta":{"content":"Hello"}}]}',
            "data: " + '{"choices":[{"delta":{"content":" World"}}]}',
            "data: [DONE]",
        ]
        mock_post.return_value = mock_resp

        tokens = list(OpenAIProvider().stream_generate("test"))
        assert tokens == ["Hello", " World"]

    @patch("director_ai.integrations.providers.requests.post")
    def test_stream_generate_bad_json_skipped(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = [
            "data: {bad json",
            "data: [DONE]",
        ]
        mock_post.return_value = mock_resp
        tokens = list(OpenAIProvider().stream_generate("test"))
        assert tokens == []

    @patch("director_ai.integrations.providers.requests.post")
    def test_stream_generate_request_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException("fail")
        tokens = list(OpenAIProvider().stream_generate("test"))
        assert len(tokens) == 1
        assert "Error" in tokens[0]


class TestAnthropicProvider:
    def test_name(self):
        p = AnthropicProvider(model="claude-3")
        assert p.name == "anthropic/claude-3"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"content": [{"type": "text", "text": "Answer"}]}
        mock_post.return_value = mock_resp

        result = AnthropicProvider().generate_candidates("hi", n=1)
        assert result[0]["text"] == "Answer"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("t")
        result = AnthropicProvider().generate_candidates("q", n=1)
        assert result[0]["source"] == "error"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_http_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.HTTPError("401")
        result = AnthropicProvider().generate_candidates("q", n=1)
        assert "HTTP Error" in result[0]["text"]

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("c")
        result = AnthropicProvider().generate_candidates("q", n=1)
        assert "Connection Error" in result[0]["text"]

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_value_error(self, mock_post):
        mock_post.side_effect = ValueError("v")
        result = AnthropicProvider().generate_candidates("q", n=1)
        assert "Parse Error" in result[0]["text"]


class TestHuggingFaceProvider:
    def test_name(self):
        p = HuggingFaceProvider(model="m/model")
        assert p.name == "huggingface/m/model"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_list_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [{"generated_text": "out"}]
        mock_post.return_value = mock_resp

        result = HuggingFaceProvider().generate_candidates("hi", n=1)
        assert result[0]["text"] == "out"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_non_list_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"error": "loading"}
        mock_post.return_value = mock_resp

        result = HuggingFaceProvider().generate_candidates("hi", n=1)
        assert "loading" in result[0]["text"]

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("t")
        result = HuggingFaceProvider().generate_candidates("hi", n=1)
        assert result[0]["source"] == "error"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_http_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.HTTPError("500")
        result = HuggingFaceProvider().generate_candidates("hi", n=1)
        assert "HTTP Error" in result[0]["text"]

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("c")
        result = HuggingFaceProvider().generate_candidates("hi", n=1)
        assert "Connection Error" in result[0]["text"]

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_parse_error(self, mock_post):
        mock_post.side_effect = KeyError("k")
        result = HuggingFaceProvider().generate_candidates("hi", n=1)
        assert "Parse Error" in result[0]["text"]


class TestLocalProvider:
    def test_name_default(self):
        assert LocalProvider().name == "local/default"

    def test_name_with_model(self):
        assert LocalProvider(model="llama3").name == "local/llama3"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_with_model(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
        mock_post.return_value = mock_resp

        result = LocalProvider(model="llama3").generate_candidates("q", n=1)
        assert result[0]["text"] == "hi"
        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "llama3"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("t")
        result = LocalProvider().generate_candidates("hi", n=1)
        assert result[0]["source"] == "error"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_http_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.HTTPError("500")
        result = LocalProvider().generate_candidates("hi", n=1)
        assert "HTTP Error" in result[0]["text"]

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("c")
        result = LocalProvider().generate_candidates("hi", n=1)
        assert "Connection Error" in result[0]["text"]

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_parse_error(self, mock_post):
        mock_post.side_effect = ValueError("v")
        result = LocalProvider().generate_candidates("hi", n=1)
        assert "Parse Error" in result[0]["text"]

    @patch("director_ai.integrations.providers.requests.post")
    def test_stream_generate_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = [
            "data: " + '{"choices":[{"delta":{"content":"A"}}]}',
            "data: " + '{"choices":[{"delta":{"content":"B"}}]}',
            "data: [DONE]",
        ]
        mock_post.return_value = mock_resp

        tokens = list(LocalProvider(model="m").stream_generate("test"))
        assert tokens == ["A", "B"]

    @patch("director_ai.integrations.providers.requests.post")
    def test_stream_generate_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException("e")
        tokens = list(LocalProvider().stream_generate("test"))
        assert "Error" in tokens[0]

    @patch("director_ai.integrations.providers.requests.post")
    def test_stream_generate_empty_content_skipped(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = [
            "data: " + '{"choices":[{"delta":{"content":""}}]}',
            "data: " + '{"choices":[{"delta":{}}]}',
            "data: [DONE]",
        ]
        mock_post.return_value = mock_resp

        tokens = list(LocalProvider().stream_generate("test"))
        assert tokens == []
