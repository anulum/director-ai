# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for hybrid scorer hardening: escalation, caching, retries."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from director_ai.core import CoherenceScorer, GroundTruthStore


def _make_hybrid_scorer(**kw):
    return CoherenceScorer(
        use_nli=False,
        llm_judge_enabled=True,
        llm_judge_provider="openai",
        scorer_backend="hybrid",
        **kw,
    )


def _mock_openai_yes():
    mock = MagicMock()
    client = MagicMock()
    mock.OpenAI.return_value = client
    choice = MagicMock()
    choice.message.content = '{"verdict": "YES"}'
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return mock, client


class TestShouldEscalate:
    def test_hybrid_escalates_near_boundary(self):
        scorer = _make_hybrid_scorer()
        # High-confidence scores (far from 0.5) should NOT escalate
        assert scorer._should_escalate(0.1) is False
        assert scorer._should_escalate(0.9) is False
        # Low-confidence scores (near 0.5) should escalate
        assert scorer._should_escalate(0.5) is True
        assert scorer._should_escalate(0.45) is True

    def test_disabled_never_escalates(self):
        scorer = CoherenceScorer(use_nli=False, llm_judge_enabled=False)
        assert scorer._should_escalate(0.5) is False

    def test_no_provider_never_escalates(self):
        scorer = CoherenceScorer(
            use_nli=False,
            llm_judge_enabled=True,
            llm_judge_provider="",
        )
        assert scorer._should_escalate(0.5) is False

    def test_non_hybrid_escalates_near_boundary(self):
        scorer = CoherenceScorer(
            use_nli=False,
            llm_judge_enabled=True,
            llm_judge_provider="openai",
            llm_judge_confidence_threshold=0.3,
        )
        assert scorer._should_escalate(0.5) is True
        assert scorer._should_escalate(0.45) is True
        assert scorer._should_escalate(0.05) is False
        assert scorer._should_escalate(0.95) is False


class TestNonEvidenceEscalation:
    def test_calculate_factual_divergence_triggers_judge(self):
        store = GroundTruthStore()
        # Use mismatched text so heuristic score lands near the ambiguity boundary
        store.add("weather", "Today is partly cloudy with mild temperatures.")
        scorer = _make_hybrid_scorer(
            ground_truth_store=store,
            llm_judge_confidence_threshold=0.51,
        )

        mock_openai, client = _mock_openai_yes()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            result = scorer.calculate_factual_divergence(
                "weather?",
                "The weather will be sunny and warm tomorrow.",
            )
        assert 0.0 <= result <= 1.0
        client.chat.completions.create.assert_called_once()

    def test_no_escalation_without_hybrid(self):
        store = GroundTruthStore()
        store.add("sky", "The sky is blue.")
        scorer = CoherenceScorer(
            use_nli=False,
            ground_truth_store=store,
            llm_judge_enabled=False,
        )
        result = scorer.calculate_factual_divergence("sky?", "The sky is blue.")
        assert 0.0 <= result <= 1.0


class TestJudgeCache:
    def test_cache_hit_avoids_api_call(self):
        store = GroundTruthStore()
        store.add("weather", "Today is partly cloudy with mild temperatures.")
        scorer = _make_hybrid_scorer(
            ground_truth_store=store,
            llm_judge_confidence_threshold=0.51,
        )

        mock_openai, client = _mock_openai_yes()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            r1 = scorer.calculate_factual_divergence(
                "weather?",
                "The weather will be sunny and warm tomorrow.",
            )
            r2 = scorer.calculate_factual_divergence(
                "weather?",
                "The weather will be sunny and warm tomorrow.",
            )

        assert r1 == r2
        client.chat.completions.create.assert_called_once()

    def test_cache_eviction_at_max(self):
        scorer = _make_hybrid_scorer()
        for i in range(scorer._JUDGE_CACHE_MAX + 10):
            scorer._judge_cache[i] = float(i)
        assert len(scorer._judge_cache) == scorer._JUDGE_CACHE_MAX + 10

        mock_openai, _ = _mock_openai_yes()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            scorer._llm_judge_check(f"prompt_{999}", "response", 0.5)
        assert len(scorer._judge_cache) <= scorer._JUDGE_CACHE_MAX + 10


class TestJudgeRetry:
    def test_retry_succeeds_on_second_attempt(self):
        scorer = _make_hybrid_scorer()

        mock_openai = MagicMock()
        client = MagicMock()
        mock_openai.OpenAI.return_value = client

        choice = MagicMock()
        choice.message.content = '{"verdict": "YES"}'
        success = MagicMock(choices=[choice])

        client.chat.completions.create.side_effect = [
            ConnectionError("network"),
            success,
        ]

        with (
            patch.dict(sys.modules, {"openai": mock_openai}),
            patch("time.sleep"),
        ):
            result = scorer._llm_judge_check("q", "a", 0.5)
        assert result != 0.5
        assert client.chat.completions.create.call_count == 2

    def test_all_retries_exhausted_returns_fallback(self):
        scorer = _make_hybrid_scorer()

        mock_openai = MagicMock()
        client = MagicMock()
        mock_openai.OpenAI.return_value = client
        client.chat.completions.create.side_effect = ConnectionError("down")

        with (
            patch.dict(sys.modules, {"openai": mock_openai}),
            patch("time.sleep"),
        ):
            result = scorer._llm_judge_check("q", "a", 0.5)
        assert result == 0.5
        assert client.chat.completions.create.call_count == scorer._JUDGE_RETRY_MAX

    def test_import_error_no_retry(self):
        scorer = _make_hybrid_scorer()

        with patch.dict(sys.modules, {"openai": None}):
            result = scorer._llm_judge_check("q", "a", 0.5)
        assert result == 0.5
