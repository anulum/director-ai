# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — VerifiedScorer Tests
"""Multi-angle tests for verified scorer pipeline."""

from __future__ import annotations

import pytest

import director_ai.core.scoring._claim_signals as signals_mod
import director_ai.core.scoring.verified_scorer as verified_mod
from director_ai.core.verified_scorer import (
    VerifiedScorer,
    _decompose_atomic,
    _entity_overlap,
    _negation_flip,
    _numerical_consistency,
    _split_sentences,
    _traceability,
    _word_overlap,
)


class TestSignalReExports:
    def test_verified_scorer_re_exports_claim_signal_kernels(self):
        """The historical import surface must resolve to _claim_signals.

        Compares by defining module rather than object identity: the
        mandatory-runtime-policy suite reloads _claim_signals to exercise
        the kernel-absent stubs, so identity across modules is test-order
        dependent while ``__module__`` provenance is not.
        """
        for name in (
            "_split_sentences",
            "_decompose_atomic",
            "_entity_overlap",
            "_numerical_consistency",
            "_negation_flip",
            "_traceability",
            "_word_overlap",
            "_sum_int",
            "_sum_float",
        ):
            assert getattr(verified_mod, name).__module__ == signals_mod.__name__, name


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

    def test_python_sentence_splitter_path(self, monkeypatch):
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", False)

        assert _split_sentences("Tiny. This is a python fallback sentence.") == [
            "This is a python fallback sentence."
        ]

    def test_python_sum_paths_without_rust(self, monkeypatch):
        # With the Rust signal kernels disabled the integer/float reducers fall
        # back to the built-in sum.
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", False)
        assert verified_mod._sum_int([1, 2, 3, 4]) == 10
        assert verified_mod._sum_float([0.5, 0.25, 0.25]) == pytest.approx(1.0)

    def test_rust_sentence_splitter_delegation(self, monkeypatch):
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", True)
        monkeypatch.setattr(
            signals_mod,
            "rust_split_sentences",
            lambda text: ["Rust sentence one.", "Rust sentence two has enough words."],
            raising=False,
        )
        result = _split_sentences("ignored")
        assert result == ["Rust sentence one.", "Rust sentence two has enough words."]

    def test_rust_sentence_splitter_runtime_fallback(self, monkeypatch):
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", True)

        def _raise_runtime(text):
            raise RuntimeError("ffi unavailable")

        monkeypatch.setattr(
            signals_mod,
            "rust_split_sentences",
            _raise_runtime,
            raising=False,
        )
        result = _split_sentences("Tiny. This is a fallback sentence.")
        assert result == ["This is a fallback sentence."]

    def test_rust_sentence_splitter_empty_filtered_result_uses_python_fallback(
        self, monkeypatch
    ):
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", True)
        monkeypatch.setattr(
            signals_mod,
            "rust_split_sentences",
            lambda _text: ["Tiny."],
            raising=False,
        )

        result = _split_sentences("Tiny. This is a fallback sentence.")

        assert result == ["This is a fallback sentence."]

    def test_rust_sentence_splitter_non_runtime_fallback(self, monkeypatch):
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", True)

        def _raise_value_error(text):
            raise ValueError("ffi unavailable")

        monkeypatch.setattr(
            signals_mod,
            "rust_split_sentences",
            _raise_value_error,
            raising=False,
        )
        result = _split_sentences("Tiny. This is a fallback sentence.")
        assert result == ["This is a fallback sentence."]


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

    def test_rust_exception_falls_back_to_python(self, monkeypatch):
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", True)
        monkeypatch.setattr(
            signals_mod,
            "rust_entity_overlap",
            lambda _a, _b: (_ for _ in ()).throw(RuntimeError("ffi fail")),
            raising=False,
        )
        score = _entity_overlap("Met Paris and Berlin", "Met Paris and London")
        assert 0.0 <= score <= 1.0

    def test_rust_non_runtime_exception_falls_back_to_python(self, monkeypatch):
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", True)
        monkeypatch.setattr(
            signals_mod,
            "rust_entity_overlap",
            lambda _a, _b: (_ for _ in ()).throw(ValueError("ffi fail")),
            raising=False,
        )
        score = _entity_overlap("Met Paris and Berlin", "Met Paris and London")
        assert 0.0 <= score <= 1.0

    def test_rust_type_error_falls_back_to_python(self, monkeypatch):
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", True)
        monkeypatch.setattr(
            signals_mod,
            "rust_entity_overlap",
            lambda _a, _b: (_ for _ in ()).throw(TypeError("ffi fail")),
            raising=False,
        )
        score = _entity_overlap("Met Paris and Berlin", "Met Paris and London")
        assert 0.0 <= score <= 1.0


class TestNumericalConsistency:
    def test_matching(self):
        assert _numerical_consistency("costs $99 per month", "price is $99") is True

    def test_mismatch(self):
        assert _numerical_consistency("costs $99", "costs $49") is False

    def test_no_numbers(self):
        assert _numerical_consistency("hello world", "hello world") is None

    def test_rust_numerical_non_runtime_exception_falls_back_to_python(
        self, monkeypatch
    ):
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", True)
        monkeypatch.setattr(
            signals_mod,
            "rust_numerical_consistency",
            lambda _a, _b: (_ for _ in ()).throw(ValueError("ffi fail")),
            raising=False,
        )
        assert _numerical_consistency("costs $99", "costs $49") is False

    def test_rust_numerical_type_error_falls_back_to_python(self, monkeypatch):
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", True)
        monkeypatch.setattr(
            signals_mod,
            "rust_numerical_consistency",
            lambda _a, _b: (_ for _ in ()).throw(TypeError("ffi fail")),
            raising=False,
        )
        assert _numerical_consistency("costs $99", "costs $49") is False


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

    def test_rust_negation_non_runtime_exception_falls_back_to_python(
        self, monkeypatch
    ):
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", True)
        monkeypatch.setattr(
            signals_mod,
            "rust_negation_flip",
            lambda _claim, _source: (_ for _ in ()).throw(ValueError("ffi fail")),
            raising=False,
        )
        assert _negation_flip(
            "The product does not support multi-user mode",
            "The product supports multi-user mode fully",
        )

    def test_rust_negation_type_error_falls_back_to_python(self, monkeypatch):
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", True)
        monkeypatch.setattr(
            signals_mod,
            "rust_negation_flip",
            lambda _claim, _source: (_ for _ in ()).throw(TypeError("ffi fail")),
            raising=False,
        )
        assert _negation_flip(
            "The product does not support multi-user mode",
            "The product supports multi-user mode fully",
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

    def test_rust_traceability_exception_falls_back_to_python(self, monkeypatch):
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", True)
        monkeypatch.setattr(
            signals_mod,
            "rust_traceability",
            lambda _a, _b: (_ for _ in ()).throw(RuntimeError("ffi fail")),
            raising=False,
        )
        score = _traceability("The sky is blue", "The sky is blue")
        assert 0.0 <= score <= 1.0

    def test_rust_traceability_non_runtime_exception_falls_back_to_python(
        self, monkeypatch
    ):
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", True)
        monkeypatch.setattr(
            signals_mod,
            "rust_traceability",
            lambda _a, _b: (_ for _ in ()).throw(ValueError("ffi fail")),
            raising=False,
        )
        score = _traceability("The sky is blue", "The sky is blue")
        assert 0.0 <= score <= 1.0

    def test_rust_traceability_type_error_falls_back_to_python(self, monkeypatch):
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", True)
        monkeypatch.setattr(
            signals_mod,
            "rust_traceability",
            lambda _a, _b: (_ for _ in ()).throw(TypeError("ffi fail")),
            raising=False,
        )
        score = _traceability("The sky is blue", "The sky is blue")
        assert 0.0 <= score <= 1.0


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

    def test_atomic_mode_decomposes_compound_claims(self):
        vs = VerifiedScorer()
        r = vs.verify(
            "Paris is the capital of France and Berlin is the capital of Germany.",
            "Paris is the capital of France. Berlin is the capital of Germany.",
            atomic=True,
        )
        assert [claim.is_atomic for claim in r.claims] == [True, True]
        assert [claim.claim for claim in r.claims] == [
            "Paris is the capital of France",
            "Berlin is the capital of Germany.",
        ]

    def test_multi_span_evidence_supports_cross_sentence_claim(self):
        vs = VerifiedScorer()
        r = vs.verify(
            "The mission used Falcon 9 from Cape Canaveral.",
            "The mission used Falcon 9. The launch site was Cape Canaveral.",
            evidence_top_k=2,
        )

        assert r.claims[0].evidence_mode == "multi_span"
        assert r.claims[0].source_indices == [0, 1]
        assert r.claims[0].aggregated_source == (
            "The mission used Falcon 9. The launch site was Cape Canaveral."
        )
        assert r.claims[0].traceability >= 0.8
        assert r.claims[0].verdict == "supported"

        payload = r.to_dict()["claims"][0]
        assert payload["evidence_mode"] == "multi_span"
        assert payload["source_indices"] == [0, 1]
        assert payload["aggregated_source"] == r.claims[0].aggregated_source

    def test_short_claims_are_ignored_after_sentence_split(self):
        vs = VerifiedScorer()
        r = vs.verify("Too short.", "The source has enough words.")
        assert r.approved
        assert r.overall_score == 1.0
        assert r.claims == []

    def test_nli_backed_matching_uses_lowest_divergence(self):
        class FakeNLI:
            model_available = True

            def score_batch(self, pairs):
                assert pairs == [
                    ("The ocean is blue.", "The sky is blue."),
                    ("The sky is blue.", "The sky is blue."),
                ]
                return [0.8, 0.05]

        vs = VerifiedScorer(nli_scorer=FakeNLI())

        assert vs._find_best_match(
            "The sky is blue.",
            ["The ocean is blue.", "The sky is blue."],
        ) == (1, 0.05)
        spans = vs._find_top_k_matches(
            "The sky is blue.",
            ["The ocean is blue.", "The sky is blue."],
            k=1,
        )
        assert spans[0].index == 1
        assert spans[0].nli_divergence == 0.05

    def test_decisive_nli_support_is_not_overridden_by_lexical_traceability(self):
        class SupportingNLI:
            model_available = True

            def score_batch(self, pairs):
                return [0.05 for _ in pairs]

        vs = VerifiedScorer(nli_scorer=SupportingNLI())
        result = vs.verify(
            "The client ended service following the subscription extension.",
            "The customer cancelled after the renewal.",
        )

        assert result.approved
        assert result.fabricated_count == 0
        assert result.claims[0].traceability < 0.15
        assert result.claims[0].verdict == "supported"
        assert result.claims[0].traceability_mode == "lexical"

    def test_uncertain_nli_still_uses_traceability_as_fabrication_signal(self):
        class UncertainNLI:
            model_available = True

            def score_batch(self, pairs):
                return [0.5 for _ in pairs]

        vs = VerifiedScorer(nli_scorer=UncertainNLI())
        result = vs.verify(
            "WhatsApp Business approvals ship now.",
            "Slack notifications work today.",
        )

        assert not result.approved
        assert result.fabricated_count == 1
        assert result.claims[0].traceability < 0.2

    def test_semantic_traceability_backend_is_used_when_configured(self):
        class SemanticTrace:
            def score(self, premise, hypothesis):
                assert premise == "The customer cancelled after the renewal."
                assert (
                    hypothesis
                    == "The client ended service following the subscription extension."
                )
                return 0.91

        vs = VerifiedScorer(
            traceability_scorer=SemanticTrace(),
            traceability_mode="semantic",
        )
        result = vs.verify(
            "The client ended service following the subscription extension.",
            "The customer cancelled after the renewal.",
        )

        assert result.claims[0].traceability == 0.91
        assert result.claims[0].traceability_mode == "semantic"

    def test_semantic_traceability_requires_backend(self):
        try:
            VerifiedScorer(traceability_mode="semantic")
        except ValueError as exc:
            assert "traceability_scorer" in str(exc)
        else:
            raise AssertionError("semantic traceability without scorer must fail")

    def test_invalid_traceability_mode_is_rejected(self):
        with pytest.raises(ValueError, match="traceability_mode"):
            VerifiedScorer(traceability_mode="unknown")

    def test_fallback_best_match_uses_word_overlap(self):
        vs = VerifiedScorer()
        best_idx, divergence = vs._find_best_match(
            "Alpha beta gamma",
            ["Delta epsilon zeta.", "Alpha beta gamma delta."],
        )
        assert best_idx == 1
        assert 0.0 <= divergence < 0.5

    def test_multi_signal_verdict_entity_contradiction(self):
        verdict, confidence = VerifiedScorer()._multi_signal_verdict(
            nli_div=0.5,
            entity_score=0.1,
            num_match=None,
            neg_flip=True,
            traceability=0.8,
        )
        assert verdict == "contradicted"
        assert confidence >= 0.5

    def test_multi_signal_verdict_fabrication_ratio_path(self):
        verdict, confidence = VerifiedScorer()._multi_signal_verdict(
            nli_div=0.5,
            entity_score=0.0,
            num_match=None,
            neg_flip=False,
            traceability=0.18,
        )
        assert verdict == "fabricated"
        assert confidence > 0.8

    def test_multi_signal_verdict_numeric_support(self):
        verdict, confidence = VerifiedScorer()._multi_signal_verdict(
            nli_div=0.1,
            entity_score=0.0,
            num_match=True,
            neg_flip=False,
            traceability=0.6,
        )
        assert verdict == "supported"
        assert confidence >= 0.5

    def test_verify_skips_short_claims_from_sentence_splitter(self, monkeypatch):
        monkeypatch.setattr(verified_mod, "_split_sentences", lambda _text: ["short"])

        result = VerifiedScorer().verify("ignored", "ignored")

        assert result.approved is True
        assert result.claims == []

    def test_aggregate_premise_is_scored_with_nli(self):
        calls = []

        class FakeNLI:
            model_available = True

            def score_batch(self, pairs):
                calls.append(pairs)
                if len(pairs) == 2:
                    return [0.2, 0.3]
                return [0.1]

        result = VerifiedScorer(nli_scorer=FakeNLI()).verify(
            "The Falcon mission launched from Cape Canaveral.",
            "The Falcon mission launched successfully. Cape Canaveral hosted the launch.",
            evidence_top_k=2,
        )

        assert calls[-1] == [
            (
                "The Falcon mission launched successfully. Cape Canaveral hosted the launch.",
                "The Falcon mission launched from Cape Canaveral.",
            )
        ]
        assert result.claims[0].nli_divergence == 0.1

    def test_traceability_can_be_disabled(self):
        score, mode = VerifiedScorer(traceability_mode="disabled")._score_traceability(
            "Unsupported claim text",
            "Different source text",
        )

        assert (score, mode) == (1.0, "disabled")

    def test_multi_signal_unverifiable_when_ratios_are_indecisive(self):
        verdict, confidence = VerifiedScorer()._multi_signal_verdict(
            nli_div=0.5,
            entity_score=0.0,
            num_match=None,
            neg_flip=False,
            traceability=0.3,
        )

        assert verdict == "unverifiable"
        assert confidence == 0.0


class TestAtomicDecomposition:
    def test_short_continuation_is_attached_to_previous_claim(self):
        assert _decompose_atomic(
            "Alpha beta gamma delta and too short. Echo zeta eta theta.",
        ) == ["Alpha beta gamma delta too short.", "Echo zeta eta theta."]

    def test_short_first_clause_keeps_original_sentence(self):
        assert _decompose_atomic("Alpha beta gamma and delta echo zeta.") == [
            "Alpha beta gamma and delta echo zeta."
        ]


class TestSignalImplementations:
    def test_python_signal_fallback_paths(self, monkeypatch):
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", False)

        assert _entity_overlap("Paris France", "Paris France") == 1.0
        assert _entity_overlap("Paris France", "Berlin Germany") == 0.0
        assert _entity_overlap("lowercase only", "still lowercase") == 1.0
        assert _numerical_consistency("value 42", "value 42") is True
        assert _numerical_consistency("value 42", "value 43") is False
        assert _numerical_consistency("value 42", "no number here") is None
        assert _numerical_consistency("no number here", "also no number") is None
        assert _negation_flip(
            "The service does not support offline mode",
            "The service supports offline mode",
        )
        assert _traceability("and the to", "irrelevant source") == 1.0
        assert _traceability("Alpha Beta", "Alpha Gamma") == 0.5
        assert _word_overlap("alpha beta", "alpha gamma") == 1.0 / 3.0
        assert verified_mod._sum_int([1, 2, 3]) == 6
        assert verified_mod._sum_float([1.25, 2.75]) == 4.0

    def test_aggregate_evidence_spans_ignores_duplicates_and_blank_text(self):
        text, indices = verified_mod._aggregate_evidence_spans(
            [
                verified_mod.SourceSpan("Second source.", 1, 0.2),
                verified_mod.SourceSpan("   ", 2, 0.3),
                verified_mod.SourceSpan("First source.", 0, 0.1),
                verified_mod.SourceSpan("Duplicate first.", 0, 0.4),
            ]
        )

        assert text == "First source. Second source."
        assert indices == [0, 1]

    def test_rust_signal_delegation_paths(self, monkeypatch):
        monkeypatch.setattr(signals_mod, "_RUST_SIGNALS", True)
        monkeypatch.setattr(
            signals_mod,
            "rust_entity_overlap",
            lambda text_a, text_b: 0.25 if text_a and text_b else 0.0,
            raising=False,
        )
        monkeypatch.setattr(
            signals_mod,
            "rust_numerical_consistency",
            lambda text_a, text_b: "42" in text_a and "42" in text_b,
            raising=False,
        )
        monkeypatch.setattr(
            signals_mod,
            "rust_negation_flip",
            lambda claim, source: claim != source,
            raising=False,
        )
        monkeypatch.setattr(
            signals_mod,
            "rust_traceability",
            lambda claim, source: 0.75 if claim in source else 0.0,
            raising=False,
        )

        assert _entity_overlap("a", "b") == 0.25
        assert _numerical_consistency("42", "42") is True
        assert _negation_flip("not same", "same")
        assert _traceability("claim", "source claim") == 0.75
        # _word_overlap now delegates to the shared text_overlap helper (covered
        # by test_text_overlap); its value is exercised below and in the fallback.
        assert _word_overlap("alpha beta", "alpha gamma") == pytest.approx(1.0 / 3.0)
        # Drive the native accelerated reducers (rust_sum_i64/rust_sum_f64 are the
        # real kernel here, not stubbed) so the _RUST_SIGNALS=True branch of the
        # sum helpers is exercised end-to-end through mandatory_execution.
        assert verified_mod._sum_int([1, 2, 3, 4]) == 10
        assert verified_mod._sum_float([0.5, 0.25, 0.25]) == pytest.approx(1.0)

    def test_fallback_matchers_use_word_overlap(self, monkeypatch):
        # _word_overlap delegates to the shared helper; drive ranking with real
        # lexical overlap. The claim shares "target source" with index 1.
        monkeypatch.setattr(
            signals_mod,
            "rust_entity_overlap",
            lambda text_a, text_b: 0.0,
            raising=False,
        )
        monkeypatch.setattr(
            signals_mod,
            "rust_numerical_consistency",
            lambda text_a, text_b: None,
            raising=False,
        )
        vs = VerifiedScorer()
        best_idx, divergence = vs._find_best_match(
            "target source claim",
            ["irrelevant unrelated text", "target source sentence"],
        )
        assert best_idx == 1
        assert divergence < 1.0  # some overlap with the target source

        spans = vs._find_top_k_matches(
            "target source claim",
            ["irrelevant unrelated text", "target source sentence", "another none"],
            k=2,
        )
        assert spans[0].index == 1  # the highest-overlap source ranks first


def test_verify_emits_claim_verdicts_with_full_signal_contract():
    from director_ai.core.scoring.verified_scorer import ClaimVerdict

    r = VerifiedScorer().verify(
        "Paris is the capital of France.",
        "France is a country. Paris is the capital of France.",
    )

    assert r.claims and all(isinstance(c, ClaimVerdict) for c in r.claims)
    first = r.claims[0]
    assert first.claim
    assert first.claim_index == 0
    assert first.verdict in {"supported", "contradicted", "unverifiable", "fabricated"}
    assert 0.0 <= first.nli_divergence <= 1.0
    assert 0.0 <= first.traceability <= 1.0
