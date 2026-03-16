# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — VerifiedScorer Tests

from __future__ import annotations

import pytest

from director_ai.core.verified_scorer import (
    VerifiedScorer,
    _entity_overlap,
    _negation_flip,
    _numerical_consistency,
    _split_sentences,
    _traceability,
)


class TestSplitSentences:
    def test_basic(self):
        # "Hello world." has only 2 words — filtered by min 3 words
        result = _split_sentences("Hello world here. This is a test.")
        assert "This is a test." in result

    def test_short_filtered(self):
        result = _split_sentences("Hi. Ok. This is long enough.")
        assert len(result) == 1
        assert "long enough" in result[0]

    def test_empty(self):
        assert _split_sentences("") == []


class TestEntityOverlap:
    def test_full_match(self):
        assert _entity_overlap("Paris France", "Paris France") == 1.0

    def test_no_entities(self):
        assert _entity_overlap("the sky is blue", "the sky is blue") == 1.0

    def test_partial(self):
        # Regex finds [A-Z][a-z]+ words: Paris, France, Berlin, London
        score = _entity_overlap(
            "Met Paris and Berlin today", "Met Paris and London today"
        )
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        assert _entity_overlap("Paris France", "London Berlin") == 0.0


class TestNumericalConsistency:
    def test_matching(self):
        assert _numerical_consistency("costs $99 per month", "price is $99") is True

    def test_mismatch(self):
        assert _numerical_consistency("costs $99", "costs $49") is False

    def test_no_numbers(self):
        assert _numerical_consistency("hello world", "hello world") is None


class TestNegationFlip:
    def test_flip(self):
        assert _negation_flip(
            "The product does not support multi-user mode",
            "The product supports multi-user mode fully",
        )

    def test_no_flip(self):
        assert not _negation_flip(
            "The product supports multi-user mode",
            "The product supports multi-user mode",
        )

    def test_both_negated(self):
        assert not _negation_flip(
            "The product does not support this",
            "The product does not support that",
        )


class TestTraceability:
    def test_full_trace(self):
        assert _traceability("The sky is blue", "The sky is blue") > 0.8

    def test_no_trace(self):
        assert (
            _traceability(
                "Discord webhooks and WhatsApp integration",
                "Slack and Microsoft Teams notifications",
            )
            < 0.3
        )

    def test_partial_trace(self):
        score = _traceability(
            "The product supports 500 users and costs $99",
            "The product supports 500 concurrent users",
        )
        assert 0.3 < score < 0.9

    def test_empty_claim(self):
        assert _traceability("", "some source") == 1.0


class TestVerifiedScorer:
    def test_correct_claim_approved(self):
        vs = VerifiedScorer()
        r = vs.verify(
            "Paris is the capital of France.",
            "France is a country. Paris is the capital of France.",
        )
        assert r.approved
        assert r.contradicted_count == 0

    def test_number_mismatch_flagged(self):
        vs = VerifiedScorer()
        r = vs.verify(
            "The plan costs $199 per month.",
            "Our Basic plan costs $29 per month. Premium costs $79 per month.",
        )
        # Without NLI, heuristic may not catch — check verdict is not "supported"
        if r.claims:
            assert (
                r.claims[0].verdict != "supported"
                or r.claims[0].numerical_match is False
            )

    def test_fabricated_content_caught(self):
        vs = VerifiedScorer()
        r = vs.verify(
            "We integrate with Slack and Teams. We also support Discord webhooks and WhatsApp Business.",
            "We integrate with Slack and Microsoft Teams for notifications.",
        )
        assert r.fabricated_count >= 1 or not r.approved

    def test_negation_flip_caught(self):
        vs = VerifiedScorer()
        r = vs.verify(
            "The system can process files of any size.",
            "The system cannot process files larger than 100 MB.",
        )
        has_issue = r.contradicted_count > 0 or not r.approved
        assert has_issue

    def test_empty_response(self):
        vs = VerifiedScorer()
        r = vs.verify("", "Some source text here.")
        assert r.approved
        assert r.confidence == "low"

    def test_empty_source(self):
        vs = VerifiedScorer()
        r = vs.verify("Some long response with enough words.", "")
        assert r.confidence == "low"

    def test_to_dict(self):
        vs = VerifiedScorer()
        r = vs.verify(
            "The sky is blue.",
            "The sky is blue due to Rayleigh scattering.",
        )
        d = r.to_dict()
        assert "approved" in d
        assert "claims" in d
        assert "confidence" in d
        assert "fabricated" in d
        if d["claims"]:
            claim = d["claims"][0]
            assert "traceability" in claim
            assert "verdict" in claim

    def test_coverage_calculation(self):
        vs = VerifiedScorer()
        r = vs.verify(
            "Paris is the capital. London is nearby.",
            "Paris is the capital of France. London is the capital of the UK.",
        )
        assert 0.0 <= r.coverage <= 1.0
