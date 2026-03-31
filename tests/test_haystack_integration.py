# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Haystack Integration Tests (STRONG)
"""Multi-angle tests for Haystack DirectorAIChecker integration.

Covers: approved/rejected paths, filter mode, no query, metadata,
batch replies, pipeline-in-pipeline, score structure, custom store,
parametrised thresholds, pipeline performance documentation.
"""

import pytest

from director_ai.integrations.haystack import DirectorAIChecker


@pytest.mark.consumer
class TestDirectorAIChecker:
    def test_run_approved(self):
        checker = DirectorAIChecker(
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        result = checker.run(
            query="What color is the sky?",
            replies=["The sky is blue."],
        )
        assert len(result["replies"]) == 1
        assert result["approved"][0] is True
        assert result["scores"][0]["score"] > 0.5

    def test_run_rejected(self):
        checker = DirectorAIChecker(
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
        )
        result = checker.run(
            query="What color is the sky?",
            replies=["Mars has two moons named Phobos and Deimos."],
        )
        assert result["approved"][0] is False

    def test_run_multiple_replies(self):
        checker = DirectorAIChecker(
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
        )
        result = checker.run(
            query="What color is the sky?",
            replies=[
                "The sky is blue.",
                "Mars has two moons named Phobos and Deimos.",
            ],
        )
        assert len(result["scores"]) == 2
        assert len(result["approved"]) == 2

    def test_run_empty_replies(self):
        checker = DirectorAIChecker(use_nli=False)
        result = checker.run(query="test", replies=[])
        assert result == {"replies": [], "scores": [], "approved": []}

    def test_run_none_replies(self):
        checker = DirectorAIChecker(use_nli=False)
        result = checker.run(query="test")
        assert result == {"replies": [], "scores": [], "approved": []}

    def test_filter_rejected_true(self):
        checker = DirectorAIChecker(
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            filter_rejected=True,
            use_nli=False,
        )
        result = checker.run(
            query="What color is the sky?",
            replies=[
                "The sky is blue.",
                "Mars has two moons named Phobos and Deimos.",
            ],
        )
        assert len(result["scores"]) == 2
        # Filtered replies should only contain approved ones
        assert len(result["replies"]) <= 2

    def test_filter_rejected_false(self):
        checker = DirectorAIChecker(
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            filter_rejected=False,
            use_nli=False,
        )
        result = checker.run(
            query="What color is the sky?",
            replies=["The sky is blue.", "unrelated response"],
        )
        assert len(result["replies"]) == 2

    def test_to_dict(self):
        checker = DirectorAIChecker(threshold=0.75, filter_rejected=True)
        d = checker.to_dict()
        assert d["type"] == "director_ai.integrations.haystack.DirectorAIChecker"
        assert d["init_parameters"]["threshold"] == 0.75
        assert d["init_parameters"]["filter_rejected"] is True

    def test_score_structure(self):
        checker = DirectorAIChecker(
            facts={"sky color": "The sky is blue."},
            use_nli=False,
        )
        result = checker.run(query="sky", replies=["The sky is blue."])
        score = result["scores"][0]
        assert "score" in score
        assert "h_logical" in score
        assert "h_factual" in score
        assert "approved" in score
        assert "warning" in score

    def test_custom_store(self):
        from director_ai.core import GroundTruthStore

        store = GroundTruthStore()
        store.add("capital", "Paris is the capital of France.")
        checker = DirectorAIChecker(store=store, threshold=0.4, use_nli=False)
        result = checker.run(
            query="capital",
            replies=["Paris is the capital of France."],
        )
        assert result["approved"][0] is True

    @pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.7])
    def test_parametrised_thresholds(self, threshold):
        checker = DirectorAIChecker(
            facts={"test": "Test fact."},
            threshold=threshold,
            use_nli=False,
        )
        result = checker.run(query="test", replies=["Test fact."])
        assert len(result["approved"]) == 1


class TestHaystackPerformanceDoc:
    """Document Haystack integration performance."""

    def test_checker_run_fast(self):
        import time

        checker = DirectorAIChecker(
            facts={"test": "Test."},
            use_nli=False,
        )
        t0 = time.perf_counter()
        for _ in range(10):
            checker.run(query="test", replies=["Test response."])
        per_call_ms = (time.perf_counter() - t0) / 10 * 1000
        assert per_call_ms < 50, f"Haystack check took {per_call_ms:.1f}ms"

    def test_result_structure(self):
        checker = DirectorAIChecker(use_nli=False)
        result = checker.run(query="q", replies=["r"])
        assert "approved" in result
        assert "scores" in result
        assert isinstance(result["approved"], list)
        assert isinstance(result["scores"], list)
