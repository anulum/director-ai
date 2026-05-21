# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — task-specific scoring path tests

"""Behavioural coverage for task-specific scoring fallbacks."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from director_ai.core.scoring import _task_scoring
from director_ai.core.scoring._task_scoring import (
    detect_task_type,
    dialogue_factual_divergence,
    minicheck_claim_coverage,
    summarization_factual_divergence,
)
from director_ai.core.types import ScoringEvidence


def _evidence() -> ScoringEvidence:
    return ScoringEvidence(
        chunks=[],
        nli_premise="source",
        nli_hypothesis="claim",
        nli_score=0.1,
    )


class _NliScorer:
    def __init__(self, *, reverse_divergence: float = 0.15) -> None:
        self.reverse_divergence = reverse_divergence
        self.score_chunked_calls = []

    def score_chunked(self, *args, **kwargs):
        self.score_chunked_calls.append((args, kwargs))
        return self.reverse_divergence, None


class TestTaskDetectionFallback:
    def test_rust_detector_is_used_when_available(self, monkeypatch):
        monkeypatch.setattr(_task_scoring, "_RUST_TASK", True)
        monkeypatch.setattr(
            _task_scoring,
            "rust_detect_task_type",
            lambda prompt, response: "qa",
            raising=False,
        )

        assert detect_task_type("Question?", "Answer") == "qa"

    def test_python_dialogue_detection_precedes_summarisation(self, monkeypatch):
        monkeypatch.setattr(_task_scoring, "_RUST_TASK", False)

        prompt = "User: Please summarise this.\nAssistant: I can help."

        assert detect_task_type(prompt, "short reply") == "dialogue"

    def test_python_summary_keyword_detection(self, monkeypatch):
        monkeypatch.setattr(_task_scoring, "_RUST_TASK", False)

        assert detect_task_type("Please write a TLDR for the deployment notes") == (
            "summarization"
        )

    def test_python_length_ratio_detects_summarisation(self, monkeypatch):
        monkeypatch.setattr(_task_scoring, "_RUST_TASK", False)

        prompt = "Policy text. " * 120
        response = "Short operational summary."

        assert detect_task_type(prompt, response) == "summarization"

    def test_python_rag_fact_check_qa_and_default_order(self, monkeypatch):
        monkeypatch.setattr(_task_scoring, "_RUST_TASK", False)

        assert detect_task_type("Based on the following source document, answer") == (
            "rag"
        )
        assert detect_task_type("Please verify this claim before release") == (
            "fact_check"
        )
        assert detect_task_type("What is the deployment status?") == "qa"
        assert detect_task_type("Write a neutral paragraph") == "default"


class TestDialogueFactualDivergence:
    def test_bidirectional_dialogue_uses_lenient_direction_and_baseline(self):
        evidence = _evidence()
        nli = _NliScorer(reverse_divergence=0.85)

        def calculate(prompt, response, tenant_id, **kwargs):
            assert kwargs == {"_inner_agg": "min", "_outer_agg": "mean"}
            assert tenant_id == "tenant-a"
            return 0.90, evidence

        adjusted, returned_evidence = dialogue_factual_divergence(
            nli,
            "User: question\nAssistant: context",
            "grounded reply",
            "tenant-a",
            calculate_factual_with_evidence=calculate,
            baseline=0.80,
        )

        assert adjusted == pytest.approx(0.25)
        assert returned_evidence is evidence
        assert nli.score_chunked_calls[0][1] == {
            "inner_agg": "min",
            "outer_agg": "mean",
            "premise_ratio": 0.4,
        }

    def test_dialogue_zero_denominator_uses_raw_divergence(self):
        adjusted, _ = dialogue_factual_divergence(
            _NliScorer(reverse_divergence=0.4),
            "context",
            "reply",
            "tenant-a",
            calculate_factual_with_evidence=lambda *args, **kwargs: (0.6, None),
            baseline=1.0,
        )

        assert adjusted == 0.4


class TestSummarizationFactualDivergence:
    def test_minicheck_layer_populates_claim_metadata(self):
        evidence = _evidence()
        nli = _NliScorer(reverse_divergence=0.32)
        mc = SimpleNamespace(
            score=lambda source, sentence: (
                0.8 if sentence.startswith("unsupported") else 0.2
            )
        )

        adjusted, returned_evidence = summarization_factual_divergence(
            nli,
            "source text",
            "supported claim. unsupported claim.",
            "tenant-a",
            calculate_factual_with_evidence=lambda *args, **kwargs: (0.36, evidence),
            baseline=0.20,
            get_minicheck_scorer=lambda: mc,
        )

        assert adjusted == pytest.approx(0.6 * 0.5 + 0.4 * 0.15)
        assert returned_evidence is evidence
        assert evidence.claim_coverage == 0.5
        assert evidence.per_claim_divergences == [0.2, 0.8]
        assert evidence.claims == ["supported claim.", "unsupported claim."]

    def test_minicheck_layer_allows_absent_evidence(self):
        mc = SimpleNamespace(score=lambda source, sentence: 0.2)

        adjusted, returned_evidence = summarization_factual_divergence(
            _NliScorer(reverse_divergence=0.20),
            "source text",
            "supported claim.",
            "tenant-a",
            calculate_factual_with_evidence=lambda *args, **kwargs: (0.30, None),
            baseline=0.20,
            get_minicheck_scorer=lambda: mc,
        )

        assert adjusted == 0.0
        assert returned_evidence is None

    def test_factcg_claim_coverage_populates_attributions(self):
        evidence = _evidence()

        class _CoverageNli(_NliScorer):
            def score_claim_coverage_with_attribution(
                self, premise, response, **kwargs
            ):
                assert premise == "source text"
                assert kwargs == {"support_threshold": 0.7}
                return 0.25, [0.1, 0.9], ["a", "b"], ["attr-a", "attr-b"]

        adjusted, _ = summarization_factual_divergence(
            _CoverageNli(reverse_divergence=0.4),
            "source text",
            "summary",
            "tenant-a",
            calculate_factual_with_evidence=lambda *args, **kwargs: (0.6, evidence),
            baseline=0.20,
            claim_support_threshold=0.7,
            claim_coverage_alpha=0.4,
        )

        assert adjusted == pytest.approx(0.4 * 0.75 + 0.6 * 0.25)
        assert evidence.claim_coverage == 0.25
        assert evidence.per_claim_divergences == [0.1, 0.9]
        assert evidence.claims == ["a", "b"]
        assert evidence.attributions == ["attr-a", "attr-b"]

    def test_factcg_claim_coverage_allows_absent_evidence(self):
        class _CoverageNli(_NliScorer):
            def score_claim_coverage_with_attribution(
                self, premise, response, **kwargs
            ):
                return 1.0, [0.1], ["supported"], ["source-span"]

        adjusted, evidence = summarization_factual_divergence(
            _CoverageNli(reverse_divergence=0.4),
            "source text",
            "summary",
            "tenant-a",
            calculate_factual_with_evidence=lambda *args, **kwargs: (0.6, None),
            baseline=0.20,
            claim_coverage_alpha=0.4,
        )

        assert adjusted == pytest.approx(0.6 * 0.25)
        assert evidence is None

    def test_claim_coverage_oom_clears_cache_and_returns_layer_a(self, monkeypatch):
        cleared = []
        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(
                cuda=SimpleNamespace(empty_cache=lambda: cleared.append(True))
            ),
        )

        class _OomNli(_NliScorer):
            def score_claim_coverage_with_attribution(self, *args, **kwargs):
                raise RuntimeError("CUDA out of memory while scoring claims")

        adjusted, evidence = summarization_factual_divergence(
            _OomNli(reverse_divergence=0.4),
            "source text",
            "summary",
            "tenant-a",
            calculate_factual_with_evidence=lambda *args, **kwargs: (0.6, None),
            baseline=0.20,
        )

        assert adjusted == pytest.approx(0.25)
        assert evidence is None
        assert cleared == [True]

    def test_claim_coverage_non_oom_error_is_propagated(self):
        class _BrokenNli(_NliScorer):
            def score_claim_coverage_with_attribution(self, *args, **kwargs):
                raise RuntimeError("tokeniser unavailable")

        with pytest.raises(RuntimeError, match="tokeniser"):
            summarization_factual_divergence(
                _BrokenNli(),
                "source text",
                "summary",
                "tenant-a",
                calculate_factual_with_evidence=lambda *args, **kwargs: (0.2, None),
            )

    def test_claim_coverage_can_be_disabled(self):
        adjusted, evidence = summarization_factual_divergence(
            _NliScorer(reverse_divergence=0.4),
            "source text",
            "summary",
            "tenant-a",
            calculate_factual_with_evidence=lambda *args, **kwargs: (0.6, None),
            baseline=0.0,
            claim_coverage_enabled=False,
        )

        assert adjusted == 0.4
        assert evidence is None


class TestMiniCheckClaimCoverage:
    def test_empty_summary_has_full_coverage(self):
        assert minicheck_claim_coverage(
            SimpleNamespace(score=lambda *_: 0.0), "src", ""
        ) == (
            1.0,
            [],
            [],
        )

    def test_fallback_sentence_splitter_when_nltk_unavailable(self, monkeypatch):
        real_import = __import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "nltk.tokenize":
                raise ImportError("nltk unavailable")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr("builtins.__import__", guarded_import)
        scorer = SimpleNamespace(score=lambda source, sentence: 0.4)

        coverage, divs, sentences = minicheck_claim_coverage(
            scorer, "source", "One claim. Two claim"
        )

        assert coverage == 1.0
        assert divs == [0.4, 0.4]
        assert sentences == ["One claim.", "Two claim."]

    def test_rust_reducer_used_when_available(self, monkeypatch):
        monkeypatch.setattr(_task_scoring, "_RUST_TASK", True)

        called = {"count": 0}

        def _reduce(divs, threshold):
            called["count"] += 1
            assert threshold == 0.5
            assert divs == [0.2, 0.9]
            return 0.5, 1

        monkeypatch.setattr(
            _task_scoring,
            "rust_coverage_from_divergences",
            _reduce,
            raising=True,
        )

        scorer = SimpleNamespace(score=lambda source, sentence: 0.2 if "One" in sentence else 0.9)
        coverage, divs, sentences = minicheck_claim_coverage(
            scorer,
            "source",
            "One claim. Two claim.",
        )

        assert called["count"] == 1
        assert coverage == pytest.approx(0.5)
        assert divs == [0.2, 0.9]
        assert sentences == ["One claim.", "Two claim."]

    def test_python_reducer_fallback_on_rust_runtime_error(self, monkeypatch):
        monkeypatch.setattr(_task_scoring, "_RUST_TASK", True)

        def _raise_runtime(divs, threshold):
            raise RuntimeError("ffi unavailable")

        monkeypatch.setattr(
            _task_scoring,
            "rust_coverage_from_divergences",
            _raise_runtime,
            raising=True,
        )

        scorer = SimpleNamespace(score=lambda source, sentence: 0.2 if "One" in sentence else 0.9)
        coverage, divs, sentences = minicheck_claim_coverage(
            scorer,
            "source",
            "One claim. Two claim.",
        )

        assert coverage == pytest.approx(0.5)
        assert divs == [0.2, 0.9]
        assert sentences == ["One claim.", "Two claim."]

    def test_rust_sentence_splitter_used_when_available(self, monkeypatch):
        monkeypatch.setattr(_task_scoring, "_RUST_TASK", True)
        monkeypatch.setattr(
            _task_scoring,
            "rust_split_sentences",
            lambda text: ["Claim from rust A.", "Claim from rust B."],
            raising=True,
        )
        monkeypatch.setattr(
            _task_scoring,
            "rust_coverage_from_divergences",
            lambda divs, threshold: (0.5, 1),
            raising=True,
        )

        scorer = SimpleNamespace(
            score=lambda source, sentence: 0.2 if "A" in sentence else 0.9
        )
        coverage, divs, sentences = minicheck_claim_coverage(
            scorer,
            "source",
            "ignored by rust splitter",
        )

        assert coverage == pytest.approx(0.5)
        assert divs == [0.2, 0.9]
        assert sentences == ["Claim from rust A.", "Claim from rust B."]

    def test_sentence_splitter_fallback_when_rust_runtime_error(self, monkeypatch):
        monkeypatch.setattr(_task_scoring, "_RUST_TASK", True)

        def _raise_runtime(text):
            raise RuntimeError("ffi unavailable")

        monkeypatch.setattr(
            _task_scoring,
            "rust_split_sentences",
            _raise_runtime,
            raising=True,
        )
        monkeypatch.setattr(
            _task_scoring,
            "rust_coverage_from_divergences",
            lambda divs, threshold: (0.5, 1),
            raising=True,
        )

        scorer = SimpleNamespace(
            score=lambda source, sentence: 0.2 if "One" in sentence else 0.9
        )
        coverage, divs, sentences = minicheck_claim_coverage(
            scorer,
            "source",
            "One claim. Two claim.",
        )

        assert coverage == pytest.approx(0.5)
        assert divs == [0.2, 0.9]
        assert sentences == ["One claim.", "Two claim."]
