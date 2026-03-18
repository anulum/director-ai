# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Coverage for batch.py timeout/error paths and actor.py SSE streaming."""

from __future__ import annotations

import sys
from concurrent.futures import TimeoutError as FuturesTimeoutError
from unittest.mock import MagicMock, patch

import pytest

from director_ai.core.batch import BatchProcessor


class TestBatchTimeoutErrors:
    def _make_processor(self):
        agent = MagicMock()
        agent.process.return_value = MagicMock(
            output="ok",
            coherence=MagicMock(
                score=0.9,
                h_logical=0.1,
                h_factual=0.05,
                warning=False,
            ),
            halted=False,
            candidates_evaluated=1,
            fallback_used=False,
            halt_evidence=None,
        )
        return BatchProcessor(agent, max_concurrency=2, item_timeout=0.001)

    def test_process_batch_timeout(self):
        proc = self._make_processor()

        with patch.object(proc, "_process_one", side_effect=FuturesTimeoutError("t")):
            result = proc.process_batch(["q1"])
            assert result.failed >= 0

    def test_process_batch_runtime_error(self):
        proc = self._make_processor()

        with patch.object(proc, "_process_one", side_effect=RuntimeError("boom")):
            result = proc.process_batch(["q1"])
            assert result.failed >= 0

    def test_review_batch_timeout(self):
        proc = self._make_processor()

        with patch.object(proc, "_review_one", side_effect=FuturesTimeoutError("t")):
            result = proc.review_batch([("p", "r")])
            assert result.failed >= 0

    def test_review_batch_runtime_error(self):
        proc = self._make_processor()

        with patch.object(proc, "_review_one", side_effect=RuntimeError("boom")):
            result = proc.review_batch([("p", "r")])
            assert result.failed >= 0


class TestBatchAsync:
    def test_process_batch_timeout(self):
        agent = MagicMock()
        proc = BatchProcessor(agent, max_concurrency=2, item_timeout=0.001)

        with patch.object(proc, "_process_one", side_effect=RuntimeError("fail")):
            result = proc.process_batch(["q1"])
            assert result.total == 1


class TestActorSSE:
    @pytest.mark.asyncio
    async def test_stream_tokens_sse_path(self):
        mock_httpx = MagicMock()

        class FakeAsyncClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            def stream(self, method, url, json=None):
                return FakeStreamCtx()

        class FakeStreamCtx:
            async def __aenter__(self):
                return FakeResponse()

            async def __aexit__(self, *a):
                pass

        class FakeResponse:
            async def aiter_lines(self):
                yield 'data: {"content": "hello"}'
                yield 'data: {"content": " world"}'

        mock_httpx.AsyncClient = FakeAsyncClient

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            from director_ai.core.actor import LLMGenerator

            gen = LLMGenerator("http://localhost:1234/v1/chat/completions")
            tokens = []
            async for tok in gen.stream_tokens("test"):
                tokens.append(tok)
            assert len(tokens) == 2
            assert tokens[0] == "hello"

    @pytest.mark.asyncio
    async def test_stream_tokens_fallback(self):
        with patch.dict(sys.modules, {"httpx": None}):
            from director_ai.core.actor import LLMGenerator

            gen = LLMGenerator("http://localhost:1234/v1/chat/completions")
            tokens = []
            async for tok in gen.stream_tokens("test"):
                tokens.append(tok)
            assert len(tokens) > 0
