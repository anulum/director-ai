# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for actor.py pipeline."""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from director_ai.core.actor import LLMGenerator, MockGenerator

URL = "http://localhost:8080/completion"


class TestMockGenerator:
    def test_default_candidates(self) -> None:
        g = MockGenerator()
        candidates = g.generate_candidates("test")
        assert len(candidates) == 3
        assert candidates[0]["type"] == "truth"
        assert candidates[1]["type"] == "hallucination"
        assert candidates[2]["type"] == "ambiguous"

    def test_n_less_than_pool(self) -> None:
        g = MockGenerator()
        assert len(g.generate_candidates("test", n=1)) == 1

    def test_n_greater_than_pool_cycles(self) -> None:
        g = MockGenerator()
        candidates = g.generate_candidates("test", n=5)
        assert len(candidates) == 5
        assert candidates[3]["type"] == candidates[0]["type"]

    def test_stream_tokens_yields_truthful_mock_response_words(self) -> None:
        async def collect_tokens() -> list[str]:
            return [token async for token in MockGenerator().stream_tokens("test")]

        assert asyncio.run(collect_tokens()) == [
            "Based",
            "on",
            "my",
            "training",
            "data,",
            "the",
            "answer",
            "is",
            "consistent",
            "with",
            "reality.",
        ]


class TestLLMGeneratorRetry:
    def test_generation_parameters_are_wired_into_payload(self) -> None:
        g = LLMGenerator(
            URL,
            max_tokens=256,
            temperature=0.2,
            stop_sequences=("END",),
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"content": "answer"}

        with patch(
            "director_ai.core.actor.requests.post",
            return_value=mock_resp,
        ) as mock_post:
            candidates = g.generate_candidates("test", n=1)

        assert candidates[0]["text"] == "answer"
        payload = mock_post.call_args.kwargs["json"]
        assert payload["n_predict"] == 256
        assert payload["temperature"] == 0.2
        assert payload["stop"] == ["END"]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"api_url": ""}, "api_url"),
            ({"api_url": URL, "max_retries": 0}, "max_retries"),
            ({"api_url": URL, "base_delay": -0.1}, "base_delay"),
            ({"api_url": URL, "timeout": 0}, "timeout"),
            ({"api_url": URL, "max_tokens": 0}, "max_tokens"),
            ({"api_url": URL, "temperature": -0.1}, "temperature"),
            ({"api_url": URL, "temperature": 2.1}, "temperature"),
            ({"api_url": URL, "stop_sequences": ("",)}, "stop_sequences"),
            ({"api_url": URL, "stop_sequences": "END"}, "stop_sequences"),
        ],
    )
    def test_invalid_generation_configuration_rejected(
        self, kwargs: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            LLMGenerator(**kwargs)

    @pytest.mark.parametrize("prompt", ["", "   ", 42])
    def test_invalid_prompt_rejected(self, prompt: object) -> None:
        g = LLMGenerator(URL)
        with pytest.raises(ValueError, match="prompt"):
            g.generate_candidates(cast(str, prompt), n=1)

    @pytest.mark.parametrize("n", [0, -1])
    def test_invalid_candidate_count_rejected(self, n: int) -> None:
        g = LLMGenerator(URL)
        with pytest.raises(ValueError, match="n"):
            g.generate_candidates("test", n=n)

    def test_success_first_try(self) -> None:
        g = LLMGenerator(URL, max_retries=3)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"content": "answer"}

        with patch(
            "director_ai.core.actor.requests.post",
            return_value=mock_resp,
        ):
            candidates = g.generate_candidates("test", n=1)
        assert candidates[0]["text"] == "answer"
        assert candidates[0]["source"] == "LLM"

    def test_retry_on_timeout(self) -> None:
        import requests as req

        g = LLMGenerator(URL, max_retries=2, base_delay=0.01)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"content": "ok"}

        with patch(
            "director_ai.core.actor.requests.post",
            side_effect=[req.exceptions.Timeout(), mock_resp],
        ):
            candidates = g.generate_candidates("test", n=1)
        assert candidates[0]["text"] == "ok"

    def test_all_retries_exhausted(self) -> None:
        import requests as req

        g = LLMGenerator(URL, max_retries=2, base_delay=0.01)

        with patch(
            "director_ai.core.actor.requests.post",
            side_effect=req.exceptions.Timeout(),
        ):
            candidates = g.generate_candidates("test", n=1)
        assert candidates[0]["source"] == "System"
        assert "unavailable" in candidates[0]["text"]

    def test_http_error_no_retry(self) -> None:
        g = LLMGenerator(URL, max_retries=2, base_delay=0.01)
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch(
            "director_ai.core.actor.requests.post",
            return_value=mock_resp,
        ):
            candidates = g.generate_candidates("test", n=1)
        assert "unavailable" in candidates[0]["text"]


class TestCircuitBreaker:
    def test_circuit_opens_after_threshold(self) -> None:
        import requests as req

        g = LLMGenerator(URL, max_retries=1, base_delay=0.01)
        g._circuit_threshold = 3

        with patch(
            "director_ai.core.actor.requests.post",
            side_effect=req.exceptions.ConnectionError(),
        ):
            for _ in range(3):
                g.generate_candidates("test", n=1)

        assert g._circuit_open is True

    def test_circuit_open_fast_fails(self) -> None:
        g = LLMGenerator(URL)
        g._circuit_open = True

        with patch(
            "director_ai.core.actor.requests.post",
        ) as mock_post:
            candidates = g.generate_candidates("test", n=1)
            mock_post.assert_not_called()
        assert "unavailable" in candidates[0]["text"]

    def test_reset_circuit(self) -> None:
        g = LLMGenerator(URL)
        g._circuit_open = True
        g._consecutive_failures = 10
        g.reset_circuit()
        assert g._circuit_open is False
        assert g._consecutive_failures == 0


def test_llm_generator_stream_tokens_falls_back_to_generated_text() -> None:
    async def collect_tokens() -> list[str]:
        tokens: list[str] = []
        async for token in LLMGenerator(api_url="http://test").stream_tokens("prompt"):
            tokens.append(token)
        return tokens

    with patch("director_ai.core.actor.requests.post") as post:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"content": "hello world"}
        post.return_value = response

        tokens = asyncio.run(collect_tokens())

    assert tokens == ["hello", "world"]


def test_llm_generator_stream_tokens_uses_sse_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeResponse:
        headers = {"content-type": "text/event-stream"}

        def raise_for_status(self) -> None:
            """Represent a successful HTTP response."""

        async def __aenter__(self) -> FakeResponse:
            return self

        async def __aexit__(self, *_exc: object) -> bool:
            return False

        async def aiter_lines(self) -> AsyncIterator[str]:
            for line in (
                'data: {"content": "hello"}',
                'data: {"token": "world"}',
                "event: ignored",
                'data: {"content": ""}',
            ):
                yield line

    class FakeClient:
        def __init__(self, *, timeout: float) -> None:
            calls["timeout"] = timeout

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_exc: object) -> bool:
            return False

        def stream(
            self, method: str, url: str, *, json: dict[str, object]
        ) -> FakeResponse:
            calls["method"] = method
            calls["url"] = url
            calls["json"] = json
            return FakeResponse()

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        types.SimpleNamespace(AsyncClient=FakeClient),
    )

    async def collect_tokens() -> list[str]:
        tokens: list[str] = []
        generator = LLMGenerator(api_url="http://test", timeout=7)
        async for token in generator.stream_tokens("prompt"):
            tokens.append(token)
        return tokens

    assert asyncio.run(collect_tokens()) == ["hello", "world"]
    assert calls["method"] == "POST"
    assert calls["url"] == "http://test"
    assert calls["timeout"] == 7
    assert calls["json"] == {
        "prompt": "prompt",
        "n_predict": 128,
        "temperature": 0.8,
        "stop": ["\nUser:", "\nSystem:"],
        "stream": True,
    }
