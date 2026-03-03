"""Coverage tests for actor.py — LLMGenerator stream_tokens fallback."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest

from director_ai.core.actor import LLMGenerator


class TestLLMGeneratorStreamFallback:
    def test_stream_tokens_no_httpx(self):
        gen = LLMGenerator(api_url="http://localhost:1234/v1/completions")

        async def run():
            tokens = []
            with patch.dict(sys.modules, {"httpx": None}):
                with patch.object(gen, "generate_candidates") as mock_gc:
                    mock_gc.return_value = [{"text": "hello world test"}]
                    async for tok in gen.stream_tokens("test"):
                        tokens.append(tok)
            return tokens

        tokens = asyncio.get_event_loop().run_until_complete(run())
        assert len(tokens) == 3
        assert tokens[0] == "hello"
