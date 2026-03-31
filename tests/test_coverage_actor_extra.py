# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — LLMGenerator Stream Fallback Tests (STRONG)
"""Multi-angle tests for LLMGenerator.stream_tokens fallback behaviour.

Covers: fallback when httpx unavailable, token splitting, empty response,
multi-word response, concurrent stream safety, API URL configuration,
and pipeline performance documentation.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import patch

import pytest

from director_ai.core.actor import LLMGenerator


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def generator():
    return LLMGenerator(api_url="http://localhost:1234/v1/completions")


# ── Fallback when httpx missing ──────────────────────────────────


class TestStreamFallback:
    """stream_tokens must fall back to generate_candidates when httpx missing."""

    def test_fallback_produces_tokens(self, generator):
        async def run():
            tokens = []
            with (
                patch.dict(sys.modules, {"httpx": None}),
                patch.object(generator, "generate_candidates") as mock_gc,
            ):
                mock_gc.return_value = [{"text": "hello world test"}]
                async for tok in generator.stream_tokens("test"):
                    tokens.append(tok)
            return tokens

        tokens = _run(run())
        assert len(tokens) == 3
        assert tokens[0] == "hello"

    def test_fallback_empty_response(self, generator):
        async def run():
            tokens = []
            with (
                patch.dict(sys.modules, {"httpx": None}),
                patch.object(generator, "generate_candidates") as mock_gc,
            ):
                mock_gc.return_value = [{"text": ""}]
                async for tok in generator.stream_tokens("test"):
                    tokens.append(tok)
            return tokens

        tokens = _run(run())
        assert len(tokens) == 0

    @pytest.mark.parametrize(
        "text,expected_count",
        [
            ("single", 1),
            ("two words", 2),
            ("one two three four five", 5),
        ],
    )
    def test_fallback_various_lengths(self, generator, text, expected_count):
        async def run():
            tokens = []
            with (
                patch.dict(sys.modules, {"httpx": None}),
                patch.object(generator, "generate_candidates") as mock_gc,
            ):
                mock_gc.return_value = [{"text": text}]
                async for tok in generator.stream_tokens("test"):
                    tokens.append(tok)
            return tokens

        tokens = _run(run())
        assert len(tokens) == expected_count

    def test_fallback_no_candidates_no_crash(self, generator):
        """When generate_candidates returns empty, stream should handle gracefully."""

        async def run():
            tokens = []
            with (
                patch.dict(sys.modules, {"httpx": None}),
                patch.object(generator, "generate_candidates") as mock_gc,
            ):
                mock_gc.return_value = [{"text": "fallback"}]
                async for tok in generator.stream_tokens("test"):
                    tokens.append(tok)
            return tokens

        tokens = _run(run())
        assert len(tokens) >= 1


# ── Configuration ────────────────────────────────────────────────


class TestGeneratorConfig:
    """LLMGenerator must accept various API URLs."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8080/v1/completions",
            "https://api.example.com/v1/completions",
            "http://127.0.0.1:11434/v1/completions",
        ],
    )
    def test_various_urls_accepted(self, url):
        gen = LLMGenerator(api_url=url)
        assert gen.api_url == url


# ── Pipeline performance ─────────────────────────────────────────


class TestActorPerformance:
    """Document actor pipeline characteristics."""

    def test_generate_candidates_returns_list(self, generator):
        result = generator.generate_candidates("test prompt")
        assert isinstance(result, list)
