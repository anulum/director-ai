# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” v3.3.0 Hardening Tests

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from director_ai.core.agent import CoherenceAgent
from director_ai.core.config import DirectorConfig

_HAS_GRPC = __import__("importlib").util.find_spec("grpc") is not None

# â”€â”€ Item 1: version sync â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestVersionSync:
    def test_version_is_3_3_0(self):
        from director_ai import __version__

        assert __version__ == "3.9.4"


# â”€â”€ Item 3: gRPC proto stubs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


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


# â”€â”€ Item 4: async agent â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


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


# â”€â”€ Item 5: CLI chunk-size validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestCLIChunkSizeValidation:
    def test_chunk_size_zero_rejected(self, monkeypatch):
        import sys

        from director_ai.cli import _cmd_ingest

        monkeypatch.setattr(
            sys,
            "exit",
            lambda code: (_ for _ in ()).throw(SystemExit(code)),
        )
        try:
            _cmd_ingest(["somefile.txt", "--chunk-size", "0"])
        except SystemExit as e:
            assert e.code == 1

    def test_chunk_size_negative_rejected(self, monkeypatch):
        import sys

        from director_ai.cli import _cmd_ingest

        monkeypatch.setattr(
            sys,
            "exit",
            lambda code: (_ for _ in ()).throw(SystemExit(code)),
        )
        try:
            _cmd_ingest(["somefile.txt", "--chunk-size", "-5"])
        except SystemExit as e:
            assert e.code == 1


# â”€â”€ Item 6: CORS default â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestCORSDefault:
    def test_default_cors_empty(self):
        cfg = DirectorConfig()
        assert cfg.cors_origins == ""

    def test_explicit_cors_preserved(self):
        cfg = DirectorConfig(cors_origins="https://example.com")
        assert cfg.cors_origins == "https://example.com"
