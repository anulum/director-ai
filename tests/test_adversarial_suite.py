# SPDX-License-Identifier: Apache-2.0
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
    grounded_suite,
)


class TestPatternGeneration:
    def test_builds_patterns(self):
        patterns = _build_patterns()
        # 5 samples × 5 transforms + NLI-evasion classes (KIMI2-I):
        # 5 paraphrase + 5 temporal + 3 negation.
        assert len(patterns) >= 33
        categories = {p.category for p in patterns}
        assert "unicode" in categories
        assert "encoding" in categories
        assert "injection" in categories
        assert "paraphrase" in categories
        assert "temporal" in categories
        assert "negation" in categories

    def test_nli_evasion_patterns_target_the_detector(self):
        # KIMI2-I: the NLI-evasion adversarials are plain-text falsehoods —
        # no encoding/wrapping — so a bypass means the DETECTOR itself missed
        # a reworded/authority-framed/negated falsehood.
        patterns = _build_patterns()
        temporal = [p for p in patterns if p.category == "temporal"]
        assert temporal
        for p in temporal:
            assert p.adversarial.startswith("As of 2025, researchers confirmed")
            assert p.original[0].lower() in p.adversarial
        negation = [p for p in patterns if p.category == "negation"]
        assert negation
        for p in negation:
            assert " not " in p.adversarial
            assert p.original != p.adversarial
        paraphrase = [p for p in patterns if p.category == "paraphrase"]
        assert len(paraphrase) == 5
        for p in paraphrase:
            assert p.adversarial != p.original

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

    def test_score_extracted_from_object_with_score_attribute(self):
        class _Verdict:
            def __init__(self, score):
                self.score = score

        def review(prompt, response):
            return False, _Verdict(0.42)

        tester = AdversarialTester(review_fn=review)
        report = tester.run()
        # Every pattern is rejected (approved=False) -> fully detected.
        assert report.detection_rate == 1.0
        assert report.bypassed == 0

    def test_score_defaults_when_object_score_is_none(self):
        class _Verdict:
            score = None

        def review(prompt, response):
            return True, _Verdict()

        tester = AdversarialTester(review_fn=review)
        report = tester.run()
        # approved=True everywhere -> nothing detected.
        assert report.detection_rate == 0.0

    def test_score_defaults_when_review_returns_non_tuple(self):
        def review(prompt, response):
            return "approved-without-score"

        tester = AdversarialTester(review_fn=review)
        report = tester.run()
        # A non-(bool, score) result is treated as approved with full score, so
        # nothing is detected.
        assert report.detection_rate == 0.0

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


class TestGroundedSuite:
    def test_grounded_suite_attaches_relevant_prompts_and_benign_arm(self):
        """Every pattern carries its relevant prompt; the control arm exists."""
        suite = grounded_suite()
        assert len(suite.patterns) == len(_build_patterns())
        assert all(p.prompt for p in suite.patterns)
        assert len(suite.facts) == 8
        assert len(suite.benign_pairs) == 8
        # Prompts follow the pattern's origin sample, not one fixed string.
        assert len({p.prompt for p in suite.patterns}) > 1

    def test_per_pattern_prompt_takes_precedence_over_fixed(self):
        """run() must send each pattern's own prompt to the guardrail."""
        seen: list[str] = []

        def review_fn(prompt: str, response: str):
            seen.append(prompt)
            return False, 0.1

        suite = grounded_suite()
        tester = AdversarialTester(review_fn=review_fn, prompt="FIXED")
        tester.run(suite.patterns)

        assert "FIXED" not in seen
        assert set(seen) == {p.prompt for p in suite.patterns}

    def test_constant_reject_passes_robustness_but_fails_discrimination(self):
        """The vacuousness regression: reject-everything must not certify.

        Measured 2026-07-18: an ungrounded scorer returned the 0.5 neutral
        for every input, so it rejected benign truths and falsehoods alike
        and still scored 100 % detection. The benign arm makes that visible.
        """
        suite = grounded_suite()
        tester = AdversarialTester(review_fn=lambda _p, _r: (False, 0.5))
        report = tester.run(suite.patterns, benign_pairs=suite.benign_pairs)

        assert report.is_robust  # 100 % "detection"...
        assert report.benign_total == 8
        assert report.benign_approved == 0  # ...but approves nothing
        assert report.benign_approval_rate == 0.0
        assert not report.is_discriminative

    def test_discriminating_guardrail_passes_both_arms(self):
        """A guardrail that rejects attacks and approves truths certifies.

        The attack set is the ADVERSARIAL texts only — negation patterns'
        ``original`` is a true fact shared with the benign arm, so a
        discriminating guardrail must approve it, not reject it.
        """
        suite = grounded_suite()
        attacks = {p.adversarial for p in suite.patterns}

        def review_fn(prompt: str, response: str):
            return response not in attacks, 0.2 if response in attacks else 0.8

        tester = AdversarialTester(review_fn=review_fn)
        report = tester.run(suite.patterns, benign_pairs=suite.benign_pairs)

        assert report.is_robust
        assert report.benign_approved == report.benign_total == 8
        assert report.is_discriminative

    def test_report_without_benign_arm_is_never_discriminative(self):
        """Structure-only runs cannot claim efficacy."""
        tester = AdversarialTester(review_fn=lambda _p, _r: (False, 0.1))
        report = tester.run()
        assert report.is_robust
        assert report.benign_total == 0
        assert not report.is_discriminative


class TestGroundedEfficacyPinned:
    """Measured grounded-heuristic efficacy (2026-07-18 probe), pinned.

    The heuristic path is deterministic (BM25 retrieval + lexical overlap,
    no RNG), so the per-class counts are exact pins: a change here means
    the detector's real efficacy moved and must be re-measured, not
    explained away.
    """

    def test_grounded_heuristic_per_class_efficacy_and_benign_approval(self):
        from director_ai.core.scoring.scorer import CoherenceScorer
        from director_ai.core.vector_store import VectorGroundTruthStore

        suite = grounded_suite()
        store = VectorGroundTruthStore()
        store.ingest(suite.facts)
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)

        def review_fn(prompt: str, response: str):
            approved, score = scorer.review(prompt, response)
            return approved, score.score

        tester = AdversarialTester(review_fn=review_fn)
        report = tester.run(suite.patterns, benign_pairs=suite.benign_pairs)

        per_class: dict[str, list[bool]] = {}
        for result in report.results:
            per_class.setdefault(result.pattern.category, []).append(result.detected)

        assert sum(per_class["encoding"]) == 10
        assert sum(per_class["unicode"]) == 10
        assert sum(per_class["injection"]) == 5
        assert sum(per_class["paraphrase"]) == 5
        assert sum(per_class["temporal"]) == 5
        # Pure negation is the known heuristic limit (high lexical overlap);
        # the model-backed NLI tier handles it (KIMI3-negation, task #52).
        assert sum(per_class["negation"]) >= 2
        # Benign truths APPROVE — the arm that kills constant-reject.
        assert report.benign_approved == report.benign_total == 8
        assert report.is_discriminative
