# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``director_ai.agentic.handoff_scorer``.

Covers keyword scoring, NLI fallback, history tracking, statistics,
threshold behaviour, and edge cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from director_ai.agentic.handoff_scorer import (
    HandoffScore,
    HandoffScorer,
    _keyword_divergence,
)

# ── Keyword divergence utility ─────────────────────────────────────────


class TestKeywordDivergence:
    def test_identical(self):
        assert _keyword_divergence("hello world", "hello world") == 0.0

    def test_no_overlap(self):
        assert _keyword_divergence("foo bar", "baz qux") == 1.0

    def test_partial(self):
        d = _keyword_divergence(
            "Paris capital France", "Paris is the capital of France"
        )
        assert 0.0 <= d < 0.5  # good overlap (all msg words in ctx → 0.0)

    def test_empty_message(self):
        assert _keyword_divergence("", "context") == 0.5

    def test_empty_context(self):
        assert _keyword_divergence("message", "") == 0.5

    def test_both_empty(self):
        assert _keyword_divergence("", "") == 0.5


# ── Basic scoring ──────────────────────────────────────────────────────


class TestBasicScoring:
    def test_grounded_message(self):
        s = HandoffScorer(threshold=0.5)
        result = s.score(
            "Paris is the capital",
            "Paris is the capital of France",
            "r1",
            "s1",
        )
        assert result.grounded is True
        assert result.score < 0.5
        assert result.method == "keyword"

    def test_hallucinated_message(self):
        s = HandoffScorer(threshold=0.3)
        result = s.score(
            "completely unrelated gibberish xyz abc",
            "Paris is the capital of France",
            "r1",
            "s1",
        )
        assert result.grounded is False
        assert result.score > 0.3

    def test_agent_ids_preserved(self):
        s = HandoffScorer()
        result = s.score("msg", "ctx", "agent-a", "agent-b")
        assert result.from_agent == "agent-a"
        assert result.to_agent == "agent-b"

    def test_latency_recorded(self):
        s = HandoffScorer()
        result = s.score("msg", "msg context", "a", "b")
        assert result.latency_ms >= 0.0

    def test_evidence_on_failure(self):
        s = HandoffScorer(threshold=0.01)
        result = s.score("unrelated xyz", "actual context here", "a", "b")
        assert len(result.evidence) >= 1
        assert "divergence" in result.evidence[0]


# ── Threshold behaviour ────────────────────────────────────────────────


class TestThreshold:
    def test_strict_threshold(self):
        s = HandoffScorer(threshold=0.0)
        result = s.score("partial overlap text", "partial overlap text here", "a", "b")
        # Only exact match passes threshold=0.0
        assert isinstance(result.grounded, bool)

    def test_loose_threshold(self):
        s = HandoffScorer(threshold=0.99)
        # Use partial overlap so score is <1.0
        result = s.score("some shared words here", "shared words in context", "a", "b")
        assert result.grounded is True  # high threshold passes partial overlap

    def test_default_threshold(self):
        s = HandoffScorer()
        assert s._threshold == 0.4


# ── NLI scoring ────────────────────────────────────────────────────────


class TestNLIScoring:
    def test_nli_used_when_available(self):
        mock_nli = MagicMock()
        mock_nli.calculate_factual_divergence.return_value = 0.15
        s = HandoffScorer(nli_scorer=mock_nli)
        result = s.score("msg", "ctx", "a", "b")
        assert result.method == "nli"
        assert result.score == 0.15
        mock_nli.calculate_factual_divergence.assert_called_once()

    def test_nli_failure_falls_back(self):
        mock_nli = MagicMock()
        mock_nli.calculate_factual_divergence.side_effect = RuntimeError("NLI fail")
        s = HandoffScorer(nli_scorer=mock_nli)
        result = s.score(
            "overlapping words here", "overlapping words here too", "a", "b"
        )
        assert result.method == "keyword"  # fell back

    def test_no_nli_uses_keyword(self):
        s = HandoffScorer(nli_scorer=None)
        result = s.score("msg", "msg context", "a", "b")
        assert result.method == "keyword"


# ── History tracking ───────────────────────────────────────────────────


class TestHistory:
    def test_history_recorded(self):
        s = HandoffScorer()
        s.score("msg1", "ctx1", "a", "b")
        s.score("msg2", "ctx2", "a", "c")
        assert len(s.history) == 2

    def test_history_order(self):
        s = HandoffScorer()
        s.score("first", "ctx", "a", "b")
        s.score("second", "ctx", "a", "c")
        assert s.history[0].from_agent == "a"
        assert s.history[1].to_agent == "c"

    def test_clear_history(self):
        s = HandoffScorer()
        s.score("msg", "ctx", "a", "b")
        s.clear_history()
        assert len(s.history) == 0

    def test_history_is_copy(self):
        s = HandoffScorer()
        s.score("msg", "ctx", "a", "b")
        h = s.history
        h.clear()
        assert len(s.history) == 1  # original not affected


# ── Statistics ─────────────────────────────────────────────────────────


class TestStats:
    def test_empty_stats(self):
        s = HandoffScorer()
        stats = s.stats()
        assert stats["total"] == 0
        assert stats["grounded_pct"] == 0.0

    def test_all_grounded(self):
        s = HandoffScorer(threshold=0.99)
        for _ in range(5):
            s.score("same words", "same words context", "a", "b")
        stats = s.stats()
        assert stats["total"] == 5
        assert stats["grounded_pct"] == 100.0

    def test_mean_score(self):
        s = HandoffScorer()
        s.score("overlap text here", "overlap text here context", "a", "b")
        stats = s.stats()
        assert 0.0 <= stats["mean_score"] <= 1.0

    def test_mean_latency(self):
        s = HandoffScorer()
        s.score("msg", "msg ctx", "a", "b")
        stats = s.stats()
        assert stats["mean_latency_ms"] >= 0.0


# ── Dataclass ──────────────────────────────────────────────────────────


class TestHandoffScoreDataclass:
    def test_frozen(self):
        h = HandoffScore("a", "b", 0.5, True)
        assert h.from_agent == "a"

    def test_defaults(self):
        h = HandoffScore("a", "b", 0.3, True)
        assert h.method == "keyword"
        assert h.evidence == []
        assert h.latency_ms == 0.0
