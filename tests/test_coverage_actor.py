# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Coverage tests for actor.py — MockGenerator, LLMGenerator."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import requests

from director_ai.core.actor import (
    LLMGenerator,
    MockGenerator,
)


class TestMockGenerator:
    def test_generate_default(self):
        gen = MockGenerator()
        candidates = gen.generate_candidates("test")
        assert len(candidates) == 3

    def test_generate_n1(self):
        candidates = MockGenerator().generate_candidates("test", n=1)
        assert len(candidates) == 1
        assert candidates[0]["type"] == "truth"

    def test_generate_n5(self):
        candidates = MockGenerator().generate_candidates("test", n=5)
        assert len(candidates) == 5

    def test_stream_tokens(self):
        gen = MockGenerator()

        async def run():
            tokens = []
            async for t in gen.stream_tokens("test"):
                tokens.append(t)
            return tokens

        tokens = asyncio.run(run())
        assert len(tokens) > 0


class TestLLMGenerator:
    def test_init(self):
        gen = LLMGenerator(api_url="http://localhost:8080")
        assert gen.api_url == "http://localhost:8080"

    @patch("director_ai.core.actor.requests.post")
    def test_generate_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"content": "Hello"}
        mock_post.return_value = mock_resp

        gen = LLMGenerator(api_url="http://test")
        result = gen.generate_candidates("hi", n=1)
        assert result[0]["text"] == "Hello"

    @patch("director_ai.core.actor.requests.post")
    def test_generate_choices_format(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"text": "World"}]}
        mock_post.return_value = mock_resp

        gen = LLMGenerator(api_url="http://test")
        result = gen.generate_candidates("hi", n=1)
        assert result[0]["text"] == "World"

    @patch("director_ai.core.actor.requests.post")
    def test_generate_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("timeout")

        gen = LLMGenerator(api_url="http://test", max_retries=1, base_delay=0.01)
        result = gen.generate_candidates("hi", n=1)
        assert "Error" in result[0]["text"]

    @patch("director_ai.core.actor.requests.post")
    def test_generate_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("refused")

        gen = LLMGenerator(api_url="http://test", max_retries=1, base_delay=0.01)
        result = gen.generate_candidates("hi", n=1)
        assert "Error" in result[0]["text"]

    @patch("director_ai.core.actor.requests.post")
    def test_generate_http_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_post.return_value = mock_resp

        gen = LLMGenerator(api_url="http://test", max_retries=1, base_delay=0.01)
        result = gen.generate_candidates("hi", n=1)
        assert "Error" in result[0]["text"]

    @patch("director_ai.core.actor.requests.post")
    def test_circuit_breaker(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("timeout")

        gen = LLMGenerator(api_url="http://test", max_retries=1, base_delay=0.01)
        gen._circuit_threshold = 2
        gen.generate_candidates("a", n=1)
        gen.generate_candidates("b", n=1)
        assert gen._circuit_open
        result = gen.generate_candidates("c", n=1)
        assert "Error" in result[0]["text"]

    def test_reset_circuit(self):
        gen = LLMGenerator(api_url="http://test")
        gen._circuit_open = True
        gen._consecutive_failures = 10
        gen.reset_circuit()
        assert not gen._circuit_open
        assert gen._consecutive_failures == 0

    def test_stream_tokens_fallback(self):
        gen = LLMGenerator(api_url="http://test")

        async def run():
            tokens = []
            async for t in gen.stream_tokens("test"):
                tokens.append(t)
            return tokens

        with patch("director_ai.core.actor.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"content": "hello world"}
            mock_post.return_value = mock_resp

            tokens = asyncio.run(run())
            assert len(tokens) >= 2
