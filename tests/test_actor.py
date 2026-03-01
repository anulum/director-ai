"""Unit tests for actor.py (MockGenerator + LLMGenerator)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from director_ai.core.actor import LLMGenerator, MockGenerator

URL = "http://localhost:8080/completion"


class TestMockGenerator:
    def test_default_candidates(self):
        g = MockGenerator()
        candidates = g.generate_candidates("test")
        assert len(candidates) == 3
        assert candidates[0]["type"] == "truth"
        assert candidates[1]["type"] == "hallucination"
        assert candidates[2]["type"] == "ambiguous"

    def test_n_less_than_pool(self):
        g = MockGenerator()
        assert len(g.generate_candidates("test", n=1)) == 1

    def test_n_greater_than_pool_cycles(self):
        g = MockGenerator()
        candidates = g.generate_candidates("test", n=5)
        assert len(candidates) == 5
        assert candidates[3]["type"] == candidates[0]["type"]


class TestLLMGeneratorRetry:
    def test_success_first_try(self):
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

    def test_retry_on_timeout(self):
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

    def test_all_retries_exhausted(self):
        import requests as req

        g = LLMGenerator(URL, max_retries=2, base_delay=0.01)

        with patch(
            "director_ai.core.actor.requests.post",
            side_effect=req.exceptions.Timeout(),
        ):
            candidates = g.generate_candidates("test", n=1)
        assert candidates[0]["source"] == "System"
        assert "unavailable" in candidates[0]["text"]

    def test_http_error_no_retry(self):
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
    def test_circuit_opens_after_threshold(self):
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

    def test_circuit_open_fast_fails(self):
        g = LLMGenerator(URL)
        g._circuit_open = True

        with patch(
            "director_ai.core.actor.requests.post",
        ) as mock_post:
            candidates = g.generate_candidates("test", n=1)
            mock_post.assert_not_called()
        assert "unavailable" in candidates[0]["text"]

    def test_reset_circuit(self):
        g = LLMGenerator(URL)
        g._circuit_open = True
        g._consecutive_failures = 10
        g.reset_circuit()
        assert g._circuit_open is False
        assert g._consecutive_failures == 0
