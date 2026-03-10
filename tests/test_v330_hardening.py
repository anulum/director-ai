# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — v3.3.0 Hardening Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from director_ai.core.agent import CoherenceAgent
from director_ai.core.config import DirectorConfig

_HAS_GRPC = __import__("importlib").util.find_spec("grpc") is not None

# ── Item 1: version sync ────────────────────────────────────────────


class TestVersionSync:
    def test_version_is_3_3_0(self):
        from director_ai import __version__

        assert __version__ == "3.7.0"


# ── Item 3: gRPC proto stubs ────────────────────────────────────────


@pytest.mark.skipif(not _HAS_GRPC, reason="grpc not installed")
class TestGRPCProtoStubs:
    def test_director_pb2_importable(self):
        from director_ai import director_pb2

        assert hasattr(director_pb2, "ReviewResponse")
        assert hasattr(director_pb2, "ProcessResponse")
        assert hasattr(director_pb2, "BatchReviewResponse")
        assert hasattr(director_pb2, "TokenEvent")

    def test_director_pb2_grpc_importable(self):
        from director_ai import director_pb2_grpc

        assert hasattr(director_pb2_grpc, "DirectorServiceStub")
        assert hasattr(director_pb2_grpc, "DirectorServiceServicer")


# ── Item 4: async agent ─────────────────────────────────────────────


class TestAsyncAgent:
    def test_aprocess_returns_review_result(self):
        scorer = MagicMock()
        scorer.review.return_value = (
            True,
            MagicMock(score=0.9, warning=False),
        )
        agent = CoherenceAgent(_scorer=scorer)
        # Replace generator with mock that returns a single candidate
        agent.generator = MagicMock()
        agent.generator.generate_candidates.return_value = [{"text": "response"}]

        result = asyncio.run(agent.aprocess("test prompt"))
        assert not result.halted
        assert result.coherence.score == 0.9


# ── Item 5: CLI chunk-size validation ────────────────────────────────


class TestCLIChunkSizeValidation:
    def test_chunk_size_zero_rejected(self, monkeypatch):
        import sys

        from director_ai.cli import _cmd_ingest

        monkeypatch.setattr(
            sys, "exit", lambda code: (_ for _ in ()).throw(SystemExit(code))
        )
        try:
            _cmd_ingest(["somefile.txt", "--chunk-size", "0"])
        except SystemExit as e:
            assert e.code == 1

    def test_chunk_size_negative_rejected(self, monkeypatch):
        import sys

        from director_ai.cli import _cmd_ingest

        monkeypatch.setattr(
            sys, "exit", lambda code: (_ for _ in ()).throw(SystemExit(code))
        )
        try:
            _cmd_ingest(["somefile.txt", "--chunk-size", "-5"])
        except SystemExit as e:
            assert e.code == 1


# ── Item 6: CORS default ────────────────────────────────────────────


class TestCORSDefault:
    def test_default_cors_empty(self):
        cfg = DirectorConfig()
        assert cfg.cors_origins == ""

    def test_explicit_cors_preserved(self):
        cfg = DirectorConfig(cors_origins="https://example.com")
        assert cfg.cors_origins == "https://example.com"
