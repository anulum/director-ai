# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — DivergenceMixin composition and heuristic contracts

"""Contract tests for the divergence-signal mixin behind CoherenceScorer.

The divergence calculators live in ``director_ai.core.scoring._divergence``
and are composed into :class:`CoherenceScorer` as a mixin. These tests pin
the composition (methods resolve to the mixin, constants re-export from the
scorer module) and the heuristic fallback behaviour, including the module
whose globals gate the Rust-accelerated dispatch.
"""

from __future__ import annotations

import pytest

import director_ai.core.scoring._divergence as divergence_module
import director_ai.core.scoring._divergence_factual as divergence_factual_module
import director_ai.core.scoring._divergence_routing as divergence_routing_module
import director_ai.core.scoring.scorer as scorer_module
from director_ai.core import CoherenceScorer
from director_ai.core.scoring._divergence import DivergenceMixin


class TestDivergenceComposition:
    def test_coherence_scorer_composes_divergence_mixin(self):
        assert issubclass(CoherenceScorer, DivergenceMixin)
        for name in (
            "calculate_factual_divergence",
            "calculate_factual_divergence_with_evidence",
            "calculate_logical_divergence",
            "_calculate_prompt_premise_divergence_with_evidence",
            "_dialogue_factual_divergence",
            "_summarization_factual_divergence",
            "_heuristic_factual",
            "_heuristic_logical",
            "_heuristic_coherence",
            "_resolve_agg_profile",
            "_detect_task_type",
            "_has_grounding_query",
        ):
            assert getattr(CoherenceScorer, name) is getattr(DivergenceMixin, name)

    def test_divergence_mixin_composes_factual_and_routing_bases(self):
        assert issubclass(
            DivergenceMixin, divergence_factual_module.FactualDivergenceMixin
        )
        assert issubclass(
            DivergenceMixin, divergence_routing_module.TaskRoutedCoherenceMixin
        )
        for name in (
            "calculate_factual_divergence",
            "calculate_factual_divergence_with_evidence",
            "_calculate_prompt_premise_divergence_with_evidence",
            "_has_grounding_query",
        ):
            assert getattr(DivergenceMixin, name) is getattr(
                divergence_factual_module.FactualDivergenceMixin, name
            )
        for name in (
            "_heuristic_coherence",
            "_dialogue_factual_divergence",
            "_summarization_factual_divergence",
            "_resolve_agg_profile",
            "_detect_task_type",
        ):
            assert getattr(DivergenceMixin, name) is getattr(
                divergence_routing_module.TaskRoutedCoherenceMixin, name
            )

    def test_divergence_constants_re_export_from_scorer_module(self):
        assert scorer_module.DIVERGENCE_NEUTRAL is divergence_module.DIVERGENCE_NEUTRAL
        assert scorer_module.DIVERGENCE_ALIGNED is divergence_module.DIVERGENCE_ALIGNED
        assert (
            scorer_module.DIVERGENCE_CONTRADICTED
            is divergence_module.DIVERGENCE_CONTRADICTED
        )
        assert divergence_module.DIVERGENCE_NEUTRAL == 0.5
        assert divergence_module.DIVERGENCE_ALIGNED == 0.1
        assert divergence_module.DIVERGENCE_CONTRADICTED == 0.9

    def test_accelerated_dispatch_reads_divergence_module_globals(self, monkeypatch):
        monkeypatch.setattr(
            divergence_module,
            "rust_heuristic_factual_divergence",
            lambda _context, _output: 0.42,
        )
        monkeypatch.setattr(
            divergence_module,
            "rust_heuristic_logical_divergence",
            lambda _output, _prompt: 0.17,
        )

        assert CoherenceScorer._heuristic_factual("ctx", "out") == pytest.approx(0.42)
        assert CoherenceScorer._heuristic_logical("out", "prompt") == pytest.approx(
            0.17
        )


class TestHeuristicFallbacks:
    @pytest.fixture(autouse=True)
    def _force_python_heuristics(self, monkeypatch):
        monkeypatch.setattr(
            divergence_module, "rust_heuristic_factual_divergence", None
        )
        monkeypatch.setattr(
            divergence_module, "rust_heuristic_logical_divergence", None
        )

    def test_heuristic_logical_keyword_routes(self):
        assert CoherenceScorer._heuristic_logical(
            "This is consistent with reality",
        ) == pytest.approx(divergence_module.DIVERGENCE_ALIGNED)
        assert CoherenceScorer._heuristic_logical("The opposite is true") == (
            divergence_module.DIVERGENCE_CONTRADICTED
        )
        assert CoherenceScorer._heuristic_logical(
            "It depends on your perspective",
        ) == (divergence_module.DIVERGENCE_NEUTRAL)

    def test_heuristic_logical_overlap_and_neutral_guards(self):
        assert CoherenceScorer._heuristic_logical("an answer") == (
            divergence_module.DIVERGENCE_NEUTRAL
        )
        assert CoherenceScorer._heuristic_logical("!!!", "???") == (
            divergence_module.DIVERGENCE_NEUTRAL
        )
        identical = CoherenceScorer._heuristic_logical(
            "the sky is blue",
            "the sky is blue",
        )
        assert identical == pytest.approx(0.0)

    def test_heuristic_factual_negation_and_novel_entity_signals(self):
        baseline = CoherenceScorer._heuristic_factual(
            "The tower stands in Paris.",
            "The tower stands in Paris.",
        )
        negated = CoherenceScorer._heuristic_factual(
            "The tower stands in Paris.",
            "The tower does not stand in Paris.",
        )
        novel_entity = CoherenceScorer._heuristic_factual(
            "The tower stands in Paris.",
            "The tower stands in Berlin according to Napoleon.",
        )
        assert baseline == pytest.approx(0.0)
        assert negated >= baseline + 0.25
        assert novel_entity >= 0.15

    def test_heuristic_factual_empty_token_sets_return_neutral(self):
        assert CoherenceScorer._heuristic_factual("", "answer") == (
            divergence_module.DIVERGENCE_NEUTRAL
        )
        assert CoherenceScorer._heuristic_factual("the a of", "the a of") == (
            divergence_module.DIVERGENCE_NEUTRAL
        )


class TestTaskRouting:
    def test_resolve_agg_profile_switches_to_dialogue_profile(self):
        scorer = CoherenceScorer(use_nli=False)
        dialogue_prompt = "User: hi\nAssistant: hello\nUser: how are you?"
        assert scorer._resolve_agg_profile(dialogue_prompt) == (
            "min",
            "mean",
            "min",
            "mean",
        )

    def test_resolve_agg_profile_keeps_defaults_for_plain_prompts(self):
        scorer = CoherenceScorer(use_nli=False)
        assert scorer._resolve_agg_profile("What is the capital of France?") == (
            "max",
            "max",
            "max",
            "max",
        )

    def test_resolve_agg_profile_preserves_custom_settings(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._fact_inner_agg = "mean"
        dialogue_prompt = "User: hi\nAssistant: hello\nUser: how are you?"
        assert scorer._resolve_agg_profile(dialogue_prompt) == (
            "mean",
            "max",
            "max",
            "max",
        )

    def test_has_grounding_query_requires_non_blank_prompt(self):
        assert CoherenceScorer._has_grounding_query("What is X?") is True
        assert CoherenceScorer._has_grounding_query("   \n\t  ") is False

    def test_factual_divergence_without_store_is_neutral(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._rust_scorer = None
        assert scorer.calculate_factual_divergence("prompt", "output") == (
            divergence_module.DIVERGENCE_NEUTRAL
        )


class _ClaimSupportNLI:
    """NLI double exposing the claim-coverage surface of the raw routes."""

    model_available = True

    def __init__(self, divs):
        self.divs = list(divs)
        self.coverage_calls = []

    def score_claim_coverage(self, premise, response, support_threshold=0.6):
        self.coverage_calls.append((premise, response))
        return 0.5, list(self.divs), [f"claim-{i}" for i in range(len(self.divs))]

    def score_chunked(self, *args, **kwargs):
        raise AssertionError("raw-support routes must not run chunked passes")

    def _ensure_model(self):
        return True


_DIALOGUE_PROMPT = "User: hi\nAssistant: hello\nUser: how are you?"


class TestRawSupportRouting:
    """WCS-2a: raw-support routes and their operating-point mirror."""

    def _scorer(self, divs=(0.1, 0.7)):
        scorer = CoherenceScorer(use_nli=False)
        scorer._nli = _ClaimSupportNLI(divs)
        return scorer

    def test_dialogue_route_defaults_to_raw_support(self):
        scorer = self._scorer(divs=(0.1, 0.7))

        divergence, evidence = scorer._dialogue_factual_divergence(
            _DIALOGUE_PROMPT, "reply"
        )

        assert divergence == pytest.approx(0.7)
        assert evidence is not None
        assert evidence.per_claim_divergences == [0.1, 0.7]
        assert scorer._nli.coverage_calls == [(_DIALOGUE_PROMPT, "reply")]

    def test_dialogue_baseline_squeeze_mode_restores_the_old_path(self):
        scorer = self._scorer()
        scorer._dialogue_scoring = "baseline_squeeze"
        scorer._nli.score_chunked = lambda *args, **kwargs: (0.85, None)
        scorer.calculate_factual_divergence_with_evidence = lambda *args, **kwargs: (
            0.90,
            None,
        )

        divergence, _ = scorer._dialogue_factual_divergence(_DIALOGUE_PROMPT, "reply")

        assert divergence == pytest.approx((0.85 - 0.80) / 0.20)
        assert scorer._nli.coverage_calls == []

    def test_operating_point_mirrors_the_dialogue_route(self):
        scorer = self._scorer()

        assert scorer._raw_support_operating_point(
            _DIALOGUE_PROMPT, "reply"
        ) == pytest.approx(0.0091)

    def test_operating_point_is_none_in_baseline_squeeze_mode(self):
        scorer = self._scorer()
        scorer._dialogue_scoring = "baseline_squeeze"

        assert scorer._raw_support_operating_point(_DIALOGUE_PROMPT, "reply") is None

    def test_operating_point_is_none_without_a_model_backed_nli(self):
        scorer = self._scorer()
        scorer._nli.model_available = False

        assert scorer._raw_support_operating_point(_DIALOGUE_PROMPT, "reply") is None

    def test_operating_point_is_none_for_composite_routes(self):
        scorer = self._scorer()

        assert (
            scorer._raw_support_operating_point(
                "What is the capital of France?", "Paris."
            )
            is None
        )

    def test_summarisation_blend_default_keeps_composite_thresholds(self):
        scorer = self._scorer()

        assert (
            scorer._raw_support_operating_point(
                "Summarize the quarterly report.", "Revenue rose."
            )
            is None
        )

    def test_summarisation_weakest_link_exposes_its_operating_point(self):
        scorer = self._scorer()
        scorer._summarization_aggregation = "weakest_link"

        assert scorer._raw_support_operating_point(
            "Summarize the quarterly report.", "Revenue rose."
        ) == pytest.approx(0.0402)

    def test_prompt_as_premise_summarisation_route_is_mirrored(self):
        # The SUMMARISATION route also triggers for prompt-as-premise
        # zero-logic deployments regardless of the detected task type —
        # the operating-point mirror must follow the ROUTE, not the task.
        scorer = self._scorer()
        scorer._summarization_aggregation = "weakest_link"
        scorer._use_prompt_as_premise = True
        scorer.W_LOGIC = 0.0
        scorer.W_FACT = 1.0

        assert scorer._raw_support_operating_point(
            "What is the capital of France?", "Paris."
        ) == pytest.approx(0.0402)

    def test_heuristic_coherence_returns_raw_support_for_dialogue(self):
        scorer = self._scorer(divs=(0.1, 0.7))

        h_logic, h_fact, coherence, evidence = scorer._heuristic_coherence(
            _DIALOGUE_PROMPT, "reply"
        )

        assert h_logic == 0.0
        assert h_fact == pytest.approx(0.7)
        assert coherence == pytest.approx(0.3)
        assert evidence is not None

    def test_raw_task_support_scores_the_dialogue_context(self):
        scorer = self._scorer(divs=(0.1, 0.4))

        task, support = scorer.raw_task_support(_DIALOGUE_PROMPT, "reply")

        assert task == "dialogue"
        assert support == pytest.approx(0.6)
        assert scorer._nli.coverage_calls == [(_DIALOGUE_PROMPT, "reply")]

    def test_raw_task_support_applies_the_summarisation_premise_budget(self):
        scorer = self._scorer(divs=(0.2,))
        scorer._summarization_premise_chars = 10
        prompt = "Summarize: " + "x" * 100

        task, support = scorer.raw_task_support(prompt, "short")

        assert task == "summarization"
        assert support == pytest.approx(0.8)
        assert scorer._nli.coverage_calls == [(prompt[:10], "short")]

    def test_raw_task_support_requires_a_model_backed_nli(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._nli = None

        with pytest.raises(RuntimeError, match="NLI model required"):
            scorer.raw_task_support(_DIALOGUE_PROMPT, "reply")


class _KeywordStore:
    """Non-vector grounding store: abstention gating must not apply."""

    def retrieve_context(self, prompt, top_k=3, tenant_id=""):
        del prompt, top_k, tenant_id
        return "verified context"


class _EmptyChunkVectorStore:
    """Vector-store double built lazily as a real subclass (isinstance path)."""

    def __new__(cls):
        from director_ai.core.retrieval.vector_store import VectorGroundTruthStore

        class _Store(VectorGroundTruthStore):
            def __init__(self):
                pass

            def retrieve_context(self, prompt, top_k=3, tenant_id=""):
                del prompt, top_k, tenant_id
                return "verified context"

            def retrieve_context_with_chunks(self, prompt, top_k=3, tenant_id=""):
                del prompt, top_k, tenant_id
                return []

        return _Store()


class _SingleClaimNLI:
    """NLI double whose sentence splitter always yields one claim."""

    model_available = True

    def score_chunked(self, *args, **kwargs):
        del args, kwargs
        return 0.2, [0.2]

    def _split_sentences(self, text):
        return [text]


class TestFactualDivergenceBranches:
    def test_abstention_gate_skips_non_vector_stores(self):
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=_KeywordStore())
        scorer._rust_scorer = None
        scorer._retrieval_abstention_threshold = 0.4

        assert scorer.calculate_factual_divergence(
            "verified context", "verified context"
        ) == pytest.approx(0.0)

    def test_abstention_gate_continues_when_vector_store_returns_no_chunks(self):
        scorer = CoherenceScorer(
            use_nli=False,
            ground_truth_store=_EmptyChunkVectorStore(),
        )
        scorer._rust_scorer = None
        scorer._retrieval_abstention_threshold = 0.4

        assert scorer.calculate_factual_divergence(
            "verified context", "verified context"
        ) == pytest.approx(0.0)

    def test_claim_decomposition_skips_single_claim_outputs(self):
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=_KeywordStore())
        scorer._rust_scorer = None
        scorer._nli = _SingleClaimNLI()
        scorer._rag_claim_decomposition = True
        long_single_claim = "word " * 30

        assert scorer.calculate_factual_divergence(
            "prompt", long_single_claim
        ) == pytest.approx(0.2)

    def test_factual_with_evidence_blank_prompt_is_neutral(self):
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=_KeywordStore())
        scorer._rust_scorer = None

        score, evidence = scorer.calculate_factual_divergence_with_evidence(
            "   \n\t  ",
            "answer",
        )

        assert score == divergence_module.DIVERGENCE_NEUTRAL
        assert evidence is None

    def test_factual_with_evidence_vector_store_without_chunks_is_neutral(self):
        scorer = CoherenceScorer(
            use_nli=False,
            ground_truth_store=_EmptyChunkVectorStore(),
        )
        scorer._rust_scorer = None

        score, evidence = scorer.calculate_factual_divergence_with_evidence(
            "prompt",
            "answer",
        )

        assert score == divergence_module.DIVERGENCE_NEUTRAL
        assert evidence is None
