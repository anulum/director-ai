# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — task-specific scoring path tests

"""Behavioural coverage for task-specific scoring fallbacks."""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import pytest

from director_ai.core.scoring import _task_accel, _task_scoring
from director_ai.core.scoring._task_scoring import (
    _normalize_claim_sentence,
    _sum_int,
    detect_task_type,
    dialogue_factual_divergence,
    minicheck_claim_coverage,
    summarization_factual_divergence,
)
from director_ai.core.types import ScoringEvidence

_HAS_RUST = importlib.util.find_spec("backfire_kernel") is not None
_needs_rust = pytest.mark.skipif(
    not _HAS_RUST, reason="requires the compiled backfire-kernel (director-ai[rust])"
)


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


@_needs_rust
class TestTaskScoringRustParity:
    """The pure-Python floor reproduces the Rust kernel bit-for-bit (ADR-0001).

    ``_task_scoring`` became fallback-eligible: with the kernel absent the flag is
    ``False`` and each accelerated path uses its pure-Python equivalent. These
    tests prove — against the real ``backfire_kernel`` binary — that the fallback
    is not a silent degradation but an exact reproduction.
    """

    _TASK_CASES = (
        ("What is the capital of France?", "Paris."),
        ("Summarize the quarterly report.", "Revenue rose."),
        ("Please write a TLDR of the notes.", "short"),
        ("User: hi\nAssistant: hello\nUser: bye", "ok"),
        ("Human: q\nAI: a\nHuman: q2\nAI: a2", "resp"),
        ("Based on the context, what is X?", "X is Y."),
        ("Given the document, answer the query.", "answer"),
        ("According to the passage, respond.", "resp"),
        ("Verify this claim before release.", "false"),
        ("Is it true that water boils at 100C?", "Yes at sea level."),
        ("Explain the deployment architecture.", "It has layers."),
        ("Write a neutral paragraph about the sea.", "waves crash"),
        ("A" * 1500, "short reply"),
        ("", ""),
        ("?", "x"),
    )

    def test_detect_task_type_labels_match_rust(self, monkeypatch):
        from backfire_kernel import rust_detect_task_type

        monkeypatch.setattr(_task_accel, "_RUST_TASK", False)
        for prompt, response in self._TASK_CASES:
            assert detect_task_type(prompt, response) == rust_detect_task_type(
                prompt, response
            ), f"label divergence for prompt {prompt[:40]!r}"

    def test_sum_int_matches_rust_including_overflow(self, monkeypatch):
        from backfire_kernel import rust_sum_i64

        monkeypatch.setattr(_task_accel, "_RUST_TASK", False)
        for values in (
            [1, 2, 3],
            [0],
            [2**63 - 1, 1],  # wraps to i64::MIN
            [2**63 - 1, 2**63 - 1],  # -> -2
            [-(2**63), -1],  # wraps to i64::MAX
        ):
            assert _sum_int(values) == rust_sum_i64(values)


class TestTaskDetectionFallback:
    def test_rust_detector_is_used_when_available(self, monkeypatch):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", True)
        monkeypatch.setattr(
            _task_accel,
            "rust_detect_task_type",
            lambda prompt, response: "qa",
            raising=False,
        )

        assert detect_task_type("Question?", "Answer") == "qa"

    def test_rust_detector_exception_falls_back_to_python(self, monkeypatch):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", True)
        monkeypatch.setattr(
            _task_accel,
            "rust_detect_task_type",
            lambda _prompt, _response: (_ for _ in ()).throw(RuntimeError("ffi fail")),
            raising=False,
        )
        assert detect_task_type("What is the deployment status?", "Answer") == "qa"

    def test_rust_detector_non_runtime_exception_falls_back_to_python(
        self, monkeypatch
    ):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", True)
        monkeypatch.setattr(
            _task_accel,
            "rust_detect_task_type",
            lambda _prompt, _response: (_ for _ in ()).throw(ValueError("ffi fail")),
            raising=False,
        )
        assert detect_task_type("What is the deployment status?", "Answer") == "qa"

    def test_python_dialogue_detection_precedes_summarisation(self, monkeypatch):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", False)

        prompt = "User: Please summarise this.\nAssistant: I can help."

        assert detect_task_type(prompt, "short reply") == "dialogue"

    def test_python_summary_keyword_detection(self, monkeypatch):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", False)

        assert detect_task_type("Please write a TLDR for the deployment notes") == (
            "summarization"
        )

    def test_python_length_ratio_detects_summarisation(self, monkeypatch):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", False)

        prompt = "Policy text. " * 120
        response = "Short operational summary."

        assert detect_task_type(prompt, response) == "summarization"

    def test_python_rag_fact_check_qa_and_default_order(self, monkeypatch):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", False)

        assert detect_task_type("Based on the following source document, answer") == (
            "rag"
        )
        assert detect_task_type("Please verify this claim before release") == (
            "fact_check"
        )
        assert detect_task_type("What is the deployment status?") == "qa"
        assert detect_task_type("Write a neutral paragraph") == "default"

    def test_python_question_detection_uses_instruction_keywords(self, monkeypatch):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", False)

        assert detect_task_type("Answer the question using the brief") == "qa"
        assert detect_task_type("According to the cited source, respond") == "qa"


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
    def test_minicheck_uses_nltk_sentence_tokeniser_when_available(self, monkeypatch):
        # With the rust splitter disabled the coverage scorer falls through to
        # nltk; inject a stand-in tokeniser so the nltk-present branch runs even
        # though nltk is not installed in the test environment.
        import sys
        import types

        fake_tokenize = types.ModuleType("nltk.tokenize")
        fake_tokenize.sent_tokenize = lambda text: ["Sentence one.", "Sentence two."]
        monkeypatch.setitem(sys.modules, "nltk", types.ModuleType("nltk"))
        monkeypatch.setitem(sys.modules, "nltk.tokenize", fake_tokenize)
        monkeypatch.setattr(_task_accel, "_RUST_TASK", False)

        mc = SimpleNamespace(score=lambda source, sentence: 0.2)
        coverage, divs, sentences = _task_scoring.minicheck_claim_coverage(
            mc, "source text", "Sentence one. Sentence two."
        )
        assert sentences == ["Sentence one.", "Sentence two."]
        assert divs == [0.2, 0.2]
        assert coverage == 1.0

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


class TestSumIntPaths:
    def test_sum_int_uses_rust_when_available(self, monkeypatch):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", True)
        monkeypatch.setattr(
            _task_accel, "rust_sum_i64", lambda values: sum(values), raising=False
        )
        assert _sum_int([1, 2, 3, 4]) == 10

    def test_sum_int_falls_back_to_python_without_rust(self, monkeypatch):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", False)
        assert _sum_int([5, 6]) == 11


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
        monkeypatch.setattr(_task_accel, "_RUST_TASK", True)

        called = {"count": 0}

        def _reduce(divs, threshold):
            called["count"] += 1
            assert threshold == 0.5
            assert divs == [0.2, 0.9]
            return 0.5, 1

        monkeypatch.setattr(
            _task_accel,
            "rust_coverage_from_divergences",
            _reduce,
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

        assert called["count"] == 1
        assert coverage == pytest.approx(0.5)
        assert divs == [0.2, 0.9]
        assert sentences == ["One claim.", "Two claim."]

    def test_python_reducer_fallback_on_rust_runtime_error(self, monkeypatch):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", True)

        def _raise_runtime(divs, threshold):
            raise RuntimeError("ffi unavailable")

        monkeypatch.setattr(
            _task_accel,
            "rust_coverage_from_divergences",
            _raise_runtime,
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

    def test_python_reducer_fallback_on_non_runtime_rust_error(self, monkeypatch):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", True)
        monkeypatch.setattr(
            _task_accel,
            "rust_coverage_from_divergences",
            lambda _divs, _threshold: (_ for _ in ()).throw(ValueError("ffi fail")),
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

    def test_python_reducer_fallback_on_rust_type_error(self, monkeypatch):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", True)
        monkeypatch.setattr(
            _task_accel,
            "rust_coverage_from_divergences",
            lambda _divs, _threshold: (_ for _ in ()).throw(
                TypeError("ffi signature mismatch")
            ),
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

    def test_rust_sentence_splitter_used_when_available(self, monkeypatch):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", True)
        monkeypatch.setattr(
            _task_accel,
            "rust_split_sentences",
            lambda text: ["Claim from rust A.", "Claim from rust B."],
            raising=True,
        )
        monkeypatch.setattr(
            _task_accel,
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
        monkeypatch.setattr(_task_accel, "_RUST_TASK", True)

        def _raise_runtime(text):
            raise RuntimeError("ffi unavailable")

        monkeypatch.setattr(
            _task_accel,
            "rust_split_sentences",
            _raise_runtime,
            raising=True,
        )
        monkeypatch.setattr(
            _task_accel,
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

    def test_sentence_splitter_fallback_when_rust_non_runtime_error(self, monkeypatch):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", True)
        monkeypatch.setattr(
            _task_accel,
            "rust_split_sentences",
            lambda _text: (_ for _ in ()).throw(ValueError("ffi fail")),
            raising=True,
        )
        monkeypatch.setattr(
            _task_accel,
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

    def test_sentence_splitter_fallback_when_rust_type_error(self, monkeypatch):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", True)
        monkeypatch.setattr(
            _task_accel,
            "rust_split_sentences",
            lambda _text: (_ for _ in ()).throw(TypeError("ffi signature mismatch")),
            raising=True,
        )
        monkeypatch.setattr(
            _task_accel,
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

    def test_normalize_claim_sentence_preserves_terminal_punctuation(self):
        assert _normalize_claim_sentence(" Already punctuated! ") == (
            "Already punctuated!"
        )
        assert _normalize_claim_sentence("Needs punctuation") == "Needs punctuation."
        assert _normalize_claim_sentence("   ") == ""

    def test_sum_int_uses_python_when_rust_disabled(self, monkeypatch):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", False)

        assert _sum_int([1, 0, 1, 1]) == 3

    def test_minicheck_uses_python_reducer_when_rust_disabled(self, monkeypatch):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", False)
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

    def test_minicheck_returns_full_coverage_when_splitters_find_no_sentences(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(_task_accel, "_RUST_TASK", False)
        real_import = __import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "nltk.tokenize":
                raise ImportError("nltk unavailable")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr("builtins.__import__", guarded_import)

        assert minicheck_claim_coverage(
            SimpleNamespace(score=lambda *_: 0.0),
            "source",
            "...",
        ) == (1.0, [], [])
