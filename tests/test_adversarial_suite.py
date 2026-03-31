# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for Phase 5 Gem 9: Adversarial Robustness Testing.

Covers: pattern generation, unicode injection, homoglyph substitution,
perfect/broken/partial guardrails, custom patterns, report structure,
robust/not-robust thresholds, parametrised detection rates, pipeline
integration with CoherenceScorer, and performance documentation.
"""

from __future__ import annotations

import pytest

from director_ai.testing.adversarial_suite import (
    AdversarialTester,
    RobustnessReport,
    _build_patterns,
    _homoglyph_replace,
    _inject_zero_width,
)


class TestPatternGeneration:
    def test_builds_patterns(self):
        patterns = _build_patterns()
        assert len(patterns) >= 20  # 5 samples × 5 transforms
        categories = {p.category for p in patterns}
        assert "unicode" in categories
        assert "encoding" in categories
        assert "injection" in categories

    def test_zero_width_injection(self):
        result = _inject_zero_width("hello")
        assert len(result) > len("hello")
        assert "\u200b" in result

    def test_homoglyph_changes_chars(self):
        result = _homoglyph_replace("ace")
        assert result != "ace"
        assert len(result) == 3


class TestAdversarialTester:
    def test_perfect_guardrail(self):
        def always_reject(prompt, response):
            return False, 0.1

        tester = AdversarialTester(review_fn=always_reject)
        report = tester.run()
        assert report.detection_rate == 1.0
        assert report.bypassed == 0
        assert report.is_robust

    def test_broken_guardrail(self):
        def always_approve(prompt, response):
            return True, 0.9

        tester = AdversarialTester(review_fn=always_approve)
        report = tester.run()
        assert report.detection_rate == 0.0
        assert report.bypassed == report.total_patterns
        assert not report.is_robust

    def test_partial_detection(self):
        call_count = 0

        def detect_some(prompt, response):
            nonlocal call_count
            call_count += 1
            # Reject (detect) every 3rd call; others pass through
            return call_count % 3 != 0, 0.5

        tester = AdversarialTester(review_fn=detect_some)
        report = tester.run()
        # Some detected, some not
        assert report.detected > 0
        assert report.bypassed > 0

    def test_custom_patterns(self):
        from director_ai.testing.adversarial_suite import AdversarialPattern

        patterns = [
            AdversarialPattern(
                name="test",
                category="custom",
                transform="identity",
                original="fake claim",
                adversarial="fake claim",
            )
        ]

        def reject_all(prompt, response):
            return False, 0.2

        tester = AdversarialTester(review_fn=reject_all)
        report = tester.run(patterns=patterns)
        assert report.total_patterns == 1
        assert report.detected == 1


class TestRobustnessReport:
    def test_report_structure(self):
        report = RobustnessReport(
            total_patterns=10,
            detected=9,
            bypassed=1,
            detection_rate=0.9,
            vulnerable_categories=["unicode"],
        )
        assert report.is_robust
        assert "unicode" in report.vulnerable_categories

    def test_not_robust(self):
        report = RobustnessReport(
            total_patterns=10,
            detected=5,
            bypassed=5,
            detection_rate=0.5,
        )
        assert not report.is_robust

    def test_empty_report(self):
        report = RobustnessReport(
            total_patterns=0,
            detected=0,
            bypassed=0,
            detection_rate=1.0,
        )
        assert report.is_robust

    @pytest.mark.parametrize(
        "detection_rate,expected_robust",
        [(1.0, True), (0.95, True), (0.9, True), (0.5, False), (0.0, False)],
    )
    def test_robustness_threshold(self, detection_rate, expected_robust):
        report = RobustnessReport(
            total_patterns=100,
            detected=int(100 * detection_rate),
            bypassed=int(100 * (1 - detection_rate)),
            detection_rate=detection_rate,
        )
        assert report.is_robust == expected_robust


class TestAdversarialPipelineIntegration:
    """Verify adversarial tester works with real CoherenceScorer."""

    def test_scorer_as_review_fn(self):
        from director_ai.core import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.5, use_nli=False)

        def review_fn(prompt, response):
            approved, score = scorer.review(prompt, response)
            return approved, score.score

        tester = AdversarialTester(review_fn=review_fn)
        report = tester.run()
        assert isinstance(report, RobustnessReport)
        assert report.total_patterns > 0


class TestAdversarialPerformanceDoc:
    """Document adversarial testing pipeline performance."""

    def test_pattern_generation_fast(self):
        import time

        t0 = time.perf_counter()
        patterns = _build_patterns()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 1000, f"Pattern generation took {elapsed_ms:.0f}ms"
        assert len(patterns) >= 20

    def test_report_has_all_fields(self):
        report = RobustnessReport(
            total_patterns=10,
            detected=8,
            bypassed=2,
            detection_rate=0.8,
        )
        assert hasattr(report, "total_patterns")
        assert hasattr(report, "detected")
        assert hasattr(report, "bypassed")
        assert hasattr(report, "detection_rate")
        assert hasattr(report, "is_robust")
        assert hasattr(report, "vulnerable_categories")
