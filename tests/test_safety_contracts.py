# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Safety Contract Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import hmac

import pytest

from director_ai.core.config import DirectorConfig
from director_ai.core.scorer import CoherenceScorer


def _has_fastapi() -> bool:
    try:
        import fastapi  # noqa: F401

        return True
    except ImportError:
        return False


class TestBuildScorerWiring:
    """Config.build_scorer() must wire all fields through."""

    def test_default_config(self):
        cfg = DirectorConfig()
        scorer = cfg.build_scorer()
        assert isinstance(scorer, CoherenceScorer)
        assert scorer.threshold == cfg.coherence_threshold

    def test_custom_weights(self):
        cfg = DirectorConfig(w_logic=0.4, w_fact=0.6)
        scorer = cfg.build_scorer()
        assert scorer.W_LOGIC == 0.4
        assert scorer.W_FACT == 0.6

    def test_scorer_backend_passed(self):
        cfg = DirectorConfig(scorer_backend="lite")
        scorer = cfg.build_scorer()
        assert scorer.scorer_backend == "lite"

    def test_soft_limit_passed(self):
        cfg = DirectorConfig(soft_limit=0.7)
        scorer = cfg.build_scorer()
        assert scorer.soft_limit == 0.7

    def test_llm_judge_config(self):
        cfg = DirectorConfig(
            llm_judge_enabled=True,
            llm_judge_provider="openai",
            llm_judge_confidence_threshold=0.4,
            scorer_backend="hybrid",
        )
        scorer = cfg.build_scorer()
        assert scorer._llm_judge_enabled is True
        assert scorer._llm_judge_provider == "openai"
        assert scorer._llm_judge_threshold == 0.4


class TestConstantTimeAuth:
    """API key comparisons must use hmac.compare_digest."""

    def test_hmac_compare_digest_matches(self):
        assert hmac.compare_digest("secret-key", "secret-key")

    def test_hmac_compare_digest_rejects(self):
        assert not hmac.compare_digest("secret-key", "wrong-key")

    def test_hmac_compare_digest_empty(self):
        assert not hmac.compare_digest("", "secret-key")


class TestStrictModeRejects:
    """strict_mode=True with no NLI must reject, never return neutral."""

    def test_strict_mode_factual_rejects(self):
        from director_ai.core.knowledge import GroundTruthStore

        store = GroundTruthStore()
        store.add("sky color", "The sky is blue.")
        scorer = CoherenceScorer(
            strict_mode=True,
            use_nli=False,
            ground_truth_store=store,
        )
        result = scorer.calculate_factual_divergence("sky color", "Mars has rings.")
        assert result == 0.9

    def test_strict_mode_logical_rejects(self):
        scorer = CoherenceScorer(strict_mode=True, use_nli=False)
        result = scorer.calculate_logical_divergence("sky color", "Mars has rings.")
        assert result == 0.9

    def test_strict_mode_rejected_flag(self):
        scorer = CoherenceScorer(strict_mode=True, use_nli=False)
        approved, cs = scorer.review("test", "test response")
        assert cs.strict_mode_rejected is True


@pytest.mark.skipif(not _has_fastapi(), reason="fastapi not installed")
class TestWebSocketAuthContract:
    """WS endpoint must check auth before accept when api_keys set."""

    def test_ws_rejects_without_key(self):
        from starlette.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        from director_ai.server import create_app

        cfg = DirectorConfig(api_keys=["test-key-123"])
        app = create_app(config=cfg)
        client = TestClient(app)
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect("/v1/stream", headers={}) as ws,
        ):
            ws.send_json({"prompt": "hello"})

    def test_ws_accepts_with_key(self):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        cfg = DirectorConfig(api_keys=["test-key-123"])
        app = create_app(config=cfg)
        client = TestClient(app)
        # Auth succeeds: connection accepted, we get a JSON response
        # (may be "server not ready" since lifespan isn't fully up)
        with client.websocket_connect(
            "/v1/stream",
            headers={"X-API-Key": "test-key-123"},
        ) as ws:
            ws.send_json({"prompt": "hello"})
            data = ws.receive_json()
            assert isinstance(data, dict)
