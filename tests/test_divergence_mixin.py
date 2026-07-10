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
