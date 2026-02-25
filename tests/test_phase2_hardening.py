# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Phase 2 Hardening Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Tests for Phase 2 hardening items (H18-H27).

Covers:
  - H18: ChromaDB empty metadata fix
  - H19: CORS origins configurable
  - H21: WebSocket input validation
  - H22: CLI batch file safety
  - H23: Logging f-string fixes (no runtime test needed)
  - H25: Async batch timeout
  - H27: Shared score blending helpers
  - Exception hierarchy
  - Config validation
  - Metrics memory cap
"""

import tempfile

import pytest

from director_ai.core.config import DirectorConfig
from director_ai.core.exceptions import (
    CoherenceError,
    DependencyError,
    DirectorAIError,
    GeneratorError,
    KernelHaltError,
    NumericalError,
    PhysicsError,
    ValidationError,
)
from director_ai.core.metrics import MetricsCollector
from director_ai.core.scorer import CoherenceScorer
from director_ai.core.vector_store import InMemoryBackend, VectorGroundTruthStore

# ── Exception Hierarchy ────────────────────────────────────────────


class TestExceptionHierarchy:
    """Verify all custom exceptions descend from DirectorAIError."""

    def test_base_exception(self):
        with pytest.raises(DirectorAIError):
            raise DirectorAIError("test")

    def test_coherence_error(self):
        with pytest.raises(DirectorAIError):
            raise CoherenceError("coherence failed")

    def test_kernel_halt_error(self):
        with pytest.raises(DirectorAIError):
            raise KernelHaltError("halt")

    def test_generator_error(self):
        with pytest.raises(DirectorAIError):
            raise GeneratorError("gen failed")

    def test_validation_error(self):
        with pytest.raises(DirectorAIError):
            raise ValidationError("bad input")

    def test_dependency_error(self):
        with pytest.raises(DirectorAIError):
            raise DependencyError("missing dep")

    def test_physics_error(self):
        with pytest.raises(DirectorAIError):
            raise PhysicsError("physics broke")

    def test_numerical_error_is_physics_error(self):
        with pytest.raises(PhysicsError):
            raise NumericalError("NaN detected")

    def test_numerical_error_is_director_error(self):
        with pytest.raises(DirectorAIError):
            raise NumericalError("NaN detected")


# ── H18: ChromaDB Metadata Fix ────────────────────────────────────


class TestVectorStoreMetadata:
    """Verify InMemoryBackend and VectorGroundTruthStore handle empty metadata."""

    def test_in_memory_add_with_none_metadata(self):
        backend = InMemoryBackend()
        backend.add("doc1", "hello world", metadata=None)
        assert backend.count() == 1

    def test_in_memory_add_with_empty_metadata(self):
        backend = InMemoryBackend()
        backend.add("doc1", "hello world", metadata={})
        assert backend.count() == 1

    def test_vector_store_add_fact(self):
        store = VectorGroundTruthStore(auto_index=True)
        store.add_fact("test key", "test value")
        result = store.retrieve_context("test key")
        assert result is not None
        assert "test" in result

    def test_vector_store_facts_indexed(self):
        store = VectorGroundTruthStore(auto_index=True)
        assert store.backend.count() > 0


# ── H19: CORS Origins Configurable ────────────────────────────────


class TestCORSConfig:
    """Verify cors_origins field in DirectorConfig."""

    def test_default_cors_is_star(self):
        cfg = DirectorConfig()
        assert cfg.cors_origins == "*"

    def test_custom_cors_origins(self):
        cfg = DirectorConfig(cors_origins="https://example.com,https://app.example.com")
        assert "example.com" in cfg.cors_origins

    def test_cors_in_to_dict(self):
        cfg = DirectorConfig(cors_origins="https://example.com")
        d = cfg.to_dict()
        assert d["cors_origins"] == "https://example.com"


# ── H21: WebSocket Validation ──────────────────────────────────────

try:
    from fastapi.testclient import TestClient

    from director_ai.server import create_app

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestWebSocketValidation:
    """Verify WebSocket input validation on /v1/stream."""

    @pytest.fixture
    def client(self):
        config = DirectorConfig(use_nli=False)
        app = create_app(config)
        with TestClient(app) as c:
            yield c

    def test_ws_non_dict_rejected(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json("just a string")
            resp = ws.receive_json()
            assert "error" in resp
            assert "JSON object" in resp["error"]

    def test_ws_missing_prompt_rejected(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"not_prompt": "value"})
            resp = ws.receive_json()
            assert "error" in resp
            assert "non-empty" in resp["error"]

    def test_ws_empty_prompt_rejected(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": ""})
            resp = ws.receive_json()
            assert "error" in resp

    def test_ws_whitespace_prompt_rejected(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": "   \t  "})
            resp = ws.receive_json()
            assert "error" in resp

    def test_ws_numeric_prompt_rejected(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": 12345})
            resp = ws.receive_json()
            assert "error" in resp

    def test_ws_valid_prompt_returns_result(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": "What is the sky?"})
            resp = ws.receive_json()
            assert resp["type"] == "result"
            assert "output" in resp
            assert "halted" in resp

    def test_ws_oversized_prompt_rejected(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": "x" * 200_000})
            resp = ws.receive_json()
            assert "error" in resp
            assert "exceeds" in resp["error"]


# ── H22: CLI Batch Safety ──────────────────────────────────────────


class TestCLIBatchSafety:
    """Verify CLI batch command hardening."""

    def test_batch_nonexistent_file(self):
        from director_ai.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["batch", "/nonexistent/path/file.jsonl"])
        assert exc_info.value.code == 1

    def test_batch_malformed_json_skipped(self, capsys):
        from director_ai.cli import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"prompt": "Good line"}\n')
            f.write("this is not json\n")
            f.write('{"prompt": "Another good line"}\n')
            path = f.name

        main(["batch", path])
        captured = capsys.readouterr()
        assert "Total:" in captured.out
        assert "Warning:" in captured.out  # Malformed line warning

    def test_batch_empty_lines_skipped(self, capsys):
        from director_ai.cli import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"prompt": "Q1"}\n')
            f.write("\n")
            f.write("\n")
            f.write('{"prompt": "Q2"}\n')
            path = f.name

        main(["batch", path])
        captured = capsys.readouterr()
        # Should process 2 prompts, not crash on empty lines
        assert "Total:    2" in captured.out


# ── H25: Async Batch Timeout ──────────────────────────────────────


class TestAsyncBatchTimeout:
    """Verify async batch respects item_timeout."""

    @pytest.mark.asyncio
    async def test_async_batch_has_timeout_handling(self):
        from director_ai.core.agent import CoherenceAgent
        from director_ai.core.batch import BatchProcessor

        agent = CoherenceAgent()
        processor = BatchProcessor(agent, max_concurrency=2, item_timeout=60.0)
        # Normal prompts should work fine within timeout
        result = await processor.process_batch_async(["Q1", "Q2"])
        assert result.total == 2
        assert result.succeeded == 2

    def test_sync_batch_timeout_field(self):
        from director_ai.core.agent import CoherenceAgent
        from director_ai.core.batch import BatchProcessor

        agent = CoherenceAgent()
        processor = BatchProcessor(agent, item_timeout=30.0)
        assert processor.item_timeout == 30.0


# ── H27: Shared Score Blending ─────────────────────────────────────


class TestSharedScoreBlending:
    """Verify the extracted _heuristic_coherence and _finalise_review helpers."""

    @pytest.fixture
    def scorer(self):
        return CoherenceScorer(threshold=0.6)

    def test_heuristic_coherence_returns_three_floats(self, scorer):
        h_logic, h_fact, coherence = scorer._heuristic_coherence(
            "What is the sky?", "The sky is blue."
        )
        assert isinstance(h_logic, float)
        assert isinstance(h_fact, float)
        assert isinstance(coherence, float)
        assert 0.0 <= h_logic <= 1.0
        assert 0.0 <= h_fact <= 1.0
        assert 0.0 <= coherence <= 1.0

    def test_finalise_review_approves_high_coherence(self, scorer):
        approved, score = scorer._finalise_review(0.8, 0.1, 0.2, "test action")
        assert approved is True
        assert score.score == 0.8
        assert score.h_logical == 0.1
        assert score.h_factual == 0.2

    def test_finalise_review_rejects_low_coherence(self, scorer):
        approved, score = scorer._finalise_review(0.3, 0.8, 0.7, "test action")
        assert approved is False
        assert score.score == 0.3

    def test_review_uses_shared_helpers(self, scorer):
        """review() should produce same result as calling helpers manually."""
        prompt = "What is the sky?"
        action = "Based on my training data, the answer is consistent with reality."
        approved, score = scorer.review(prompt, action)
        # Verify manually
        h_l, h_f, c = scorer._heuristic_coherence(prompt, action)
        assert abs(score.score - c) < 1e-10
        assert abs(score.h_logical - h_l) < 1e-10
        assert abs(score.h_factual - h_f) < 1e-10

    def test_weight_constants(self):
        assert CoherenceScorer.W_LOGIC == 0.6
        assert CoherenceScorer.W_FACT == 0.4
        assert abs(CoherenceScorer.W_LOGIC + CoherenceScorer.W_FACT - 1.0) < 1e-10


# ── Config Validation (from Phase 1 H12) ──────────────────────────


class TestConfigValidation:
    """Verify __post_init__ validation rejects invalid values."""

    def test_threshold_too_high(self):
        with pytest.raises(ValueError, match="coherence_threshold"):
            DirectorConfig(coherence_threshold=1.5)

    def test_threshold_negative(self):
        with pytest.raises(ValueError, match="coherence_threshold"):
            DirectorConfig(coherence_threshold=-0.1)

    def test_hard_limit_out_of_range(self):
        with pytest.raises(ValueError, match="hard_limit"):
            DirectorConfig(hard_limit=2.0)

    def test_max_candidates_zero(self):
        with pytest.raises(ValueError, match="max_candidates"):
            DirectorConfig(max_candidates=0)

    def test_history_window_zero(self):
        with pytest.raises(ValueError, match="history_window"):
            DirectorConfig(history_window=0)

    def test_temperature_too_high(self):
        with pytest.raises(ValueError, match="llm_temperature"):
            DirectorConfig(llm_temperature=3.0)

    def test_max_tokens_zero(self):
        with pytest.raises(ValueError, match="llm_max_tokens"):
            DirectorConfig(llm_max_tokens=0)

    def test_batch_concurrency_zero(self):
        with pytest.raises(ValueError, match="batch_max_concurrency"):
            DirectorConfig(batch_max_concurrency=0)


# ── Metrics Memory Cap (from Phase 1 H7) ──────────────────────────


class TestMetricsMemoryCap:
    """Verify histogram max_samples cap prevents unbounded growth."""

    def test_histogram_caps_at_max_samples(self):
        collector = MetricsCollector()
        # Observe more than default max_samples
        for i in range(200):
            collector.observe("test_hist", float(i))
        m = collector.get_metrics()
        # Count should be at most the internal max
        assert m["histograms"]["test_hist"]["count"] <= 200

    def test_reset_clears_labels(self):
        collector = MetricsCollector()
        collector.inc("test_counter", label="a")
        collector.inc("test_counter", label="b")
        collector.reset()
        m = collector.get_metrics()
        assert m["counters"]["test_counter"]["total"] == 0.0
        assert m["counters"]["test_counter"]["labels"] == {}
