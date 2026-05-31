# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — LLM Provider Adapter Tests
"""Multi-angle tests for LLM provider adapter pipeline."""

from unittest.mock import MagicMock, patch

import pytest
import requests

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

    def test_default_stream_generate_yields_single_candidate_text(self):
        class _Provider(LLMProvider):
            @property
            def name(self):
                return "test/provider"

            def generate_candidates(self, prompt: str, n: int = 3):
                assert prompt == "prompt"
                assert n == 1
                return [{"text": "whole response", "source": self.name}]

        assert list(_Provider().stream_generate("prompt")) == ["whole response"]

    def test_default_stream_generate_suppresses_empty_candidates(self):
        class _Provider(LLMProvider):
            @property
            def name(self):
                return "test/provider"

            def generate_candidates(self, prompt: str, n: int = 3):
                return []

        assert list(_Provider().stream_generate("prompt")) == []


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
            ],
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
        mock_post.side_effect = requests.exceptions.ConnectionError("refused")
        p = OpenAIProvider(api_key="sk-test")
        candidates = p.generate_candidates("Hello", n=1)
        assert len(candidates) == 1
        assert candidates[0]["text"] == "[Connection Error]"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_candidates_parse_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = ValueError("not json")
        mock_post.return_value = mock_resp

        candidates = OpenAIProvider(api_key="sk-test").generate_candidates("Hello", n=1)

        assert candidates == [{"text": "[Parse Error]", "source": "error"}]

    @patch("director_ai.integrations.providers.requests.post")
    def test_stream_generate_yields_content_and_skips_non_sse_lines(self, mock_post):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.iter_lines.return_value = [
            "",
            "event: ping",
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            "data: [DONE]",
        ]
        mock_post.return_value = response

        assert list(OpenAIProvider(api_key="sk-test").stream_generate("Hello")) == [
            "Hel",
            "lo",
        ]
        assert mock_post.call_args.kwargs["stream"] is True

    @patch("director_ai.integrations.providers.requests.post")
    def test_stream_generate_request_exception_yields_error_token(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("slow")

        assert list(OpenAIProvider().stream_generate("Hello")) == ["[Error: slow]"]


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
    def test_fetch_one_ignores_non_text_blocks(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "content": [
                {"type": "tool_use", "text": "ignored"},
                {"type": "text", "text": "Answer"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        candidate = AnthropicProvider(api_key="sk-ant-test")._fetch_one("Hello")

        assert candidate["text"] == "Answer"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_candidates_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        p = AnthropicProvider(api_key="bad-key")
        candidates = p.generate_candidates("Hello", n=1)
        assert len(candidates) == 1
        assert candidates[0]["text"] == "[Timeout]"

    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (requests.exceptions.HTTPError("429"), "HTTP Error"),
            (requests.exceptions.ConnectionError("refused"), "[Connection Error]"),
        ],
    )
    @patch("director_ai.integrations.providers.requests.post")
    def test_fetch_one_http_and_connection_errors(self, mock_post, exc, expected):
        if isinstance(exc, requests.exceptions.HTTPError):
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = exc
            mock_post.return_value = mock_resp
        else:
            mock_post.side_effect = exc

        candidate = AnthropicProvider(api_key="sk-ant-test")._fetch_one("Hello")

        assert expected in candidate["text"]
        assert candidate["source"] == "error"

    @patch("director_ai.integrations.providers.requests.post")
    def test_fetch_one_parse_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = ValueError("bad json")
        mock_post.return_value = mock_resp

        assert AnthropicProvider(api_key="sk-ant-test")._fetch_one("Hello") == {
            "text": "[Parse Error]",
            "source": "error",
        }


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

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_candidates_non_list_payload_is_stringified(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"generated_text": "nested"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        candidates = HuggingFaceProvider(api_key="hf-test").generate_candidates(
            "Hello",
            n=1,
        )

        assert candidates[0]["text"] == "{'generated_text': 'nested'}"

    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (requests.exceptions.Timeout("slow"), "[Timeout]"),
            (requests.exceptions.HTTPError("503"), "HTTP Error"),
            (requests.exceptions.ConnectionError("refused"), "[Connection Error]"),
        ],
    )
    @patch("director_ai.integrations.providers.requests.post")
    def test_fetch_one_request_errors(self, mock_post, exc, expected):
        if isinstance(exc, requests.exceptions.HTTPError):
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = exc
            mock_post.return_value = mock_resp
        else:
            mock_post.side_effect = exc

        candidate = HuggingFaceProvider(api_key="hf-test")._fetch_one("Hello")

        assert expected in candidate["text"]
        assert candidate["source"] == "error"

    @patch("director_ai.integrations.providers.requests.post")
    def test_fetch_one_parse_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = ValueError("bad json")
        mock_post.return_value = mock_resp

        assert HuggingFaceProvider(api_key="hf-test")._fetch_one("Hello") == {
            "text": "[Parse Error]",
            "source": "error",
        }

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_candidates_concurrent_path_preserves_count(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"generated_text": "Output text"}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        candidates = HuggingFaceProvider(api_key="hf-test").generate_candidates(
            "Hello",
            n=3,
        )

        assert candidates == [
            {"text": "Output text", "source": "huggingface/mistralai/Mistral-7B-Instruct-v0.3"},
            {"text": "Output text", "source": "huggingface/mistralai/Mistral-7B-Instruct-v0.3"},
            {"text": "Output text", "source": "huggingface/mistralai/Mistral-7B-Instruct-v0.3"},
        ]


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
            ],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        p = LocalProvider()
        candidates = p.generate_candidates("Hello", n=1)
        assert len(candidates) == 1
        assert candidates[0]["text"] == "Local response"

    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (requests.exceptions.Timeout("slow"), "[Timeout]"),
            (requests.exceptions.HTTPError("500"), "HTTP Error"),
            (requests.exceptions.ConnectionError("refused"), "[Connection Error]"),
        ],
    )
    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_candidates_request_errors(self, mock_post, exc, expected):
        if isinstance(exc, requests.exceptions.HTTPError):
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = exc
            mock_post.return_value = mock_resp
        else:
            mock_post.side_effect = exc

        candidates = LocalProvider().generate_candidates("Hello", n=1)

        assert expected in candidates[0]["text"]
        assert candidates[0]["source"] == "error"

    @patch("director_ai.integrations.providers.requests.post")
    def test_generate_candidates_parse_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = ValueError("bad json")
        mock_post.return_value = mock_resp

        assert LocalProvider().generate_candidates("Hello", n=1) == [
            {"text": "[Parse Error]", "source": "error"}
        ]


def test_openai_stream_skips_malformed_sse_payloads() -> None:
    from unittest.mock import MagicMock, patch

    from director_ai.integrations.providers import OpenAIProvider

    with patch("director_ai.integrations.providers.requests.post") as post:
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.iter_lines.return_value = ["data: {bad json", "data: [DONE]"]
        post.return_value = response

        assert list(OpenAIProvider().stream_generate("prompt")) == []


def test_local_provider_includes_configured_model_in_generation_payload() -> None:
    from unittest.mock import MagicMock, patch

    from director_ai.integrations.providers import LocalProvider

    with patch("director_ai.integrations.providers.requests.post") as post:
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": "answer"}}]}
        post.return_value = response

        result = LocalProvider(model="llama3").generate_candidates("question", n=1)

    assert result[0]["text"] == "answer"
    assert post.call_args.kwargs["json"]["model"] == "llama3"


def test_local_stream_skips_empty_delta_content() -> None:
    from unittest.mock import MagicMock, patch

    from director_ai.integrations.providers import LocalProvider

    with patch("director_ai.integrations.providers.requests.post") as post:
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":""}}]}',
            'data: {"choices":[{"delta":{}}]}',
            "data: [DONE]",
        ]
        post.return_value = response

        tokens = list(LocalProvider().stream_generate("prompt"))

    assert tokens == []


def test_local_stream_skips_malformed_sse_payloads() -> None:
    from unittest.mock import MagicMock, patch

    from director_ai.integrations.providers import LocalProvider

    with patch("director_ai.integrations.providers.requests.post") as post:
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.iter_lines.return_value = ["data: {bad json", "data: [DONE]"]
        post.return_value = response

        tokens = list(LocalProvider().stream_generate("prompt"))

    assert tokens == []


def test_local_stream_yields_content_and_includes_configured_model() -> None:
    from unittest.mock import MagicMock, patch

    from director_ai.integrations.providers import LocalProvider

    with patch("director_ai.integrations.providers.requests.post") as post:
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.iter_lines.return_value = [
            "",
            "event: keepalive",
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            "data: [DONE]",
        ]
        post.return_value = response

        tokens = list(LocalProvider(model="llama3").stream_generate("prompt"))

    assert tokens == ["Hel", "lo"]
    assert post.call_args.kwargs["stream"] is True
    assert post.call_args.kwargs["json"]["model"] == "llama3"


def test_local_stream_request_exception_yields_error_token() -> None:
    from unittest.mock import patch

    from director_ai.integrations.providers import LocalProvider

    with patch("director_ai.integrations.providers.requests.post") as post:
        post.side_effect = requests.exceptions.ConnectionError("offline")

        tokens = list(LocalProvider().stream_generate("prompt"))

    assert tokens == ["[Error: offline]"]


def test_openai_stream_skips_empty_delta_content() -> None:
    from unittest.mock import MagicMock, patch

    from director_ai.integrations.providers import OpenAIProvider

    with patch("director_ai.integrations.providers.requests.post") as post:
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":""}}]}',
            'data: {"choices":[{"delta":{}}]}',
            "data: [DONE]",
        ]
        post.return_value = response

        tokens = list(OpenAIProvider().stream_generate("prompt"))

    assert tokens == []
