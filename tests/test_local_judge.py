# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Local DeBERTa Judge Tests
"""Multi-angle tests for local DeBERTa-base binary judge pipeline."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from director_ai.core import CoherenceScorer  # noqa: E402, I001
from director_ai.core.config import DirectorConfig  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────


def _mock_judge_model(approve_prob: float = 0.8) -> MagicMock:
    """Build mock judge model that returns fixed approve probability."""
    model = MagicMock()
    logits = torch.tensor([[approve_prob, 1 - approve_prob]])
    model.return_value = MagicMock(logits=logits)
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)
    return model


def _mock_judge_tokenizer() -> MagicMock:
    """Build mock tokenizer returning dict with input_ids."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    return tokenizer


def _make_local_scorer(approve_prob: float = 0.8, **kw: Any) -> CoherenceScorer:
    """Build scorer with mocked local judge."""
    scorer = CoherenceScorer(
        use_nli=False,
        llm_judge_enabled=True,
        llm_judge_provider="local",
        llm_judge_model="",
        scorer_backend="hybrid",
        **kw,
    )
    scorer._local_judge_model = _mock_judge_model(approve_prob)
    scorer._local_judge_tokenizer = _mock_judge_tokenizer()
    scorer._local_judge_device = "cpu"
    return scorer


# ── Tests ─────────────────────────────────────────────────────────────


class TestLocalJudgeCheck:
    """Local judge score-adjustment checks for direct scorer proxy calls."""

    def test_known_hallucinated_rejects(self) -> None:
        """High reject probability → divergence shifts toward rejection."""
        scorer = _make_local_scorer(approve_prob=0.2)
        result = scorer._local_judge_check(
            "What is the capital of France?",
            "The capital of France is Berlin.",
            nli_score=0.5,
        )
        assert result > 0.5

    def test_known_correct_approves(self) -> None:
        """High approve probability → divergence shifts toward approval."""
        scorer = _make_local_scorer(approve_prob=0.9)
        result = scorer._local_judge_check(
            "What is the capital of France?",
            "The capital of France is Paris.",
            nli_score=0.3,
        )
        assert result < 0.4

    def test_borderline_nli_hallucinated_rejects(self) -> None:
        """Borderline NLI + hallucinated → judge tips to reject."""
        scorer = _make_local_scorer(approve_prob=0.15)
        result = scorer._local_judge_check(
            "Context about topic",
            "Hallucinated response",
            nli_score=0.5,
        )
        assert result >= 0.55

    def test_borderline_nli_correct_approves(self) -> None:
        """Borderline NLI + correct → judge tips to approve."""
        scorer = _make_local_scorer(approve_prob=0.85)
        result = scorer._local_judge_check(
            "Context about topic",
            "Correct response",
            nli_score=0.5,
        )
        assert result <= 0.45


class TestLocalJudgeBatch:
    """Batch-equivalence checks for repeated local judge calls."""

    def test_batch_matches_serial(self) -> None:
        """16 calls produce same results as serial execution."""
        scorer = _make_local_scorer(approve_prob=0.7)
        prompts = [f"prompt_{i}" for i in range(16)]
        responses = [f"response_{i}" for i in range(16)]

        results: list[float] = []
        for p, r in zip(prompts, responses, strict=True):
            scorer._judge_cache.clear()
            results.append(scorer._local_judge_check(p, r, nli_score=0.5))

        assert len(set(results)) == 1
        assert len(results) == 16


class TestLocalJudgeConfig:
    """Configuration and routing checks for local judge setup."""

    def test_config_local_provider_wires(self) -> None:
        """llm_judge_provider='local' sets up local judge path."""
        scorer = _make_local_scorer()
        assert scorer._llm_judge_provider == "local"
        assert scorer._local_judge_model is not None

    def test_should_escalate_with_local(self) -> None:
        """Local judge escalates when NLI confidence is low (near 0.5)."""
        scorer = _make_local_scorer()
        assert scorer._should_escalate(0.5) is True
        assert scorer._should_escalate(0.45) is True
        # High confidence → no escalation needed
        assert scorer._should_escalate(0.1) is False

    def test_should_not_escalate_without_model(self) -> None:
        """No model loaded → no escalation."""
        scorer = CoherenceScorer(
            use_nli=False,
            llm_judge_enabled=True,
            llm_judge_provider="local",
            llm_judge_model="",
            scorer_backend="hybrid",
        )
        assert scorer._should_escalate(0.5) is False

    def test_llm_judge_check_routes_to_local(self) -> None:
        """_llm_judge_check routes to _local_judge_check when provider='local'."""
        scorer = _make_local_scorer(approve_prob=0.9)
        result = scorer._llm_judge_check("prompt", "response", 0.5)
        assert 0.0 <= result <= 1.0
        scorer._local_judge_model.assert_called()


class TestLocalJudgeFallback:
    """Fallback checks when the local judge model is unavailable."""

    def test_missing_model_falls_back_to_nli(self) -> None:
        """No checkpoint → graceful fallback to raw NLI score."""
        scorer = CoherenceScorer(
            use_nli=False,
            llm_judge_enabled=True,
            llm_judge_provider="local",
            llm_judge_model="",
            scorer_backend="hybrid",
        )
        result = scorer._local_judge_check("prompt", "response", nli_score=0.42)
        assert result == 0.42


class TestLocalJudgeLatency:
    """Latency checks for the local judge compatibility path."""

    def test_latency_under_100ms(self) -> None:
        """Mock inference should complete in < 100ms."""
        scorer = _make_local_scorer()
        t0 = time.monotonic()
        for _ in range(10):
            scorer._judge_cache.clear()
            scorer._local_judge_check("test prompt", "test response", 0.5)
        elapsed = (time.monotonic() - t0) / 10
        assert elapsed < 0.1, f"Average latency {elapsed:.3f}s exceeds 100ms"


class TestLocalJudgeCaching:
    """Cache behavior checks for repeated local judge inputs."""

    def test_cache_hit(self) -> None:
        """Second call with same input hits cache."""
        scorer = _make_local_scorer()
        r1 = scorer._local_judge_check("prompt", "response", 0.5)
        r2 = scorer._local_judge_check("prompt", "response", 0.5)
        assert r1 == r2
        assert scorer._local_judge_model.call_count == 1


class TestDirectorConfigLocalJudge:
    """DirectorConfig local judge profile and model-field checks."""

    def test_config_has_local_model_field(self) -> None:
        """DirectorConfig accepts llm_judge_local_model."""
        cfg = DirectorConfig(llm_judge_local_model="/path/to/model")
        assert cfg.llm_judge_local_model == "/path/to/model"

    def test_thorough_profile_uses_local(self) -> None:
        """Thorough profile defaults to local judge provider."""
        cfg = DirectorConfig.from_profile("thorough")
        assert cfg.llm_judge_provider == "local"

    def test_research_profile_uses_local(self) -> None:
        """Research profile defaults to local judge provider."""
        cfg = DirectorConfig.from_profile("research")
        assert cfg.llm_judge_provider == "local"
