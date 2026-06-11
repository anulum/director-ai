# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Summarization Bidirectional NLI Tests

"""Multi-angle tests for bidirectional NLI summarization pipeline."""

from __future__ import annotations

import pytest

from director_ai.core.scorer import CoherenceScorer

# â”€â”€ Attribute defaults â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestSummarizationBaseline:
    """_summarization_nli_baseline attribute and calibration formula."""

    def test_default_baseline(self):
        scorer = CoherenceScorer(use_nli=False)
        assert scorer._summarization_nli_baseline == 0.20

    def test_custom_baseline(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._summarization_nli_baseline = 0.20
        assert scorer._summarization_nli_baseline == 0.20

    def test_calibration_zero_baseline(self):
        """With baseline=0.0, raw score passes through unchanged."""
        baseline = 0.0
        raw = 0.45
        if baseline > 0.0:
            adjusted = max(0.0, (raw - baseline) / (1.0 - baseline))
        else:
            adjusted = raw
        assert adjusted == pytest.approx(0.45)

    def test_calibration_at_baseline(self):
        """Score at baseline â†’ adjusted = 0."""
        baseline = 0.20
        raw = 0.20
        adjusted = max(0.0, (raw - baseline) / (1.0 - baseline))
        assert adjusted == pytest.approx(0.0)

    def test_calibration_above_baseline(self):
        """Score above baseline â†’ proportional adjusted value."""
        baseline = 0.20
        raw = 0.60
        adjusted = max(0.0, (raw - baseline) / (1.0 - baseline))
        assert adjusted == pytest.approx(0.50)

    def test_calibration_below_baseline(self):
        """Score below baseline â†’ clamped to 0."""
        baseline = 0.20
        raw = 0.10
        adjusted = max(0.0, (raw - baseline) / (1.0 - baseline))
        assert adjusted == pytest.approx(0.0)

    def test_calibration_at_max(self):
        """Score at 1.0 â†’ adjusted = 1.0."""
        baseline = 0.20
        raw = 1.0
        adjusted = max(0.0, (raw - baseline) / (1.0 - baseline))
        assert adjusted == pytest.approx(1.0)


# â”€â”€ Routing logic â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestSummarizationRouting:
    """_heuristic_coherence routes summarization through bidir path."""

    def test_summ_profile_without_nli_uses_heuristic(self):
        """Without NLI, summarization falls through to heuristic path."""
        scorer = CoherenceScorer(threshold=0.15, use_nli=False, w_logic=0.0, w_fact=1.0)
        scorer._use_prompt_as_premise = True
        doc = "The sky is blue. Water is wet."
        summary = "The sky is blue and water is wet."
        h_logic, h_fact, coherence, evidence = scorer._heuristic_coherence(doc, summary)
        assert h_logic == pytest.approx(0.0)
        assert isinstance(h_fact, float)
        assert 0.0 <= coherence <= 1.0

    def test_non_summ_profile_not_affected(self):
        """Non-summarization (use_prompt_as_premise=False) uses standard path."""
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        scorer._use_prompt_as_premise = False
        prompt = "What is the capital of France?"
        response = "Paris is the capital of France."
        h_logic, h_fact, coherence, evidence = scorer._heuristic_coherence(
            prompt,
            response,
        )
        assert isinstance(h_logic, float)
        assert isinstance(h_fact, float)

    def test_dialogue_takes_precedence_over_summ(self):
        """Dialogue detection fires before summarization path."""
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        scorer._use_prompt_as_premise = True
        scorer._auto_dialogue_profile = True
        # Dialogue prompt — should be detected as dialogue, not summarization
        prompt = "User: Hi\nAssistant: Hello\nUser: How are you?"
        task = CoherenceScorer._detect_task_type(prompt)
        assert task == "dialogue"

    def test_default_summarization_uses_prompt_as_evidence_premise(self):
        """Auto-detected summarization uses source prompt without a KB store."""

        class _FakeNli:
            model_available = True
            last_token_count = 11
            last_estimated_cost = 0.001
            _cost_per_token = 0.0

            def _ensure_model(self):
                return None

            def reset_token_counter(self):
                return None

            def _score_chunked_with_counts(self, premise, hypothesis, **kwargs):
                assert "Source document" in premise
                assert "short summary" in hypothesis
                return 0.24, [0.24], 1, 1

            def score_chunked(self, premise, hypothesis, **kwargs):
                assert "short summary" in premise
                assert "Source document" in hypothesis
                return 0.22, None

            def score_claim_coverage_with_attribution(
                self, premise, response, **kwargs
            ):
                return 1.0, [0.1], ["short summary"], ["source span"]

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        scorer._nli = _FakeNli()
        scorer._minicheck_nli = None
        scorer._use_prompt_as_premise = False
        prompt = "Summarise this.\n\nSource document: " + ("calibrated fact. " * 80)
        action = "short summary with calibrated fact."

        _h_logic, _h_fact, _coherence, evidence = scorer._heuristic_coherence(
            prompt,
            action,
        )

        assert evidence is not None
        assert evidence.nli_premise == prompt
        assert evidence.chunks[0].source == "prompt"


# â”€â”€ Config wiring â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestSummarizationConfig:
    """DirectorConfig wires nli_summarization_baseline to scorer."""

    def test_config_field_exists(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig()
        assert hasattr(cfg, "nli_summarization_baseline")
        assert cfg.nli_summarization_baseline == 0.20

    def test_summarization_profile_has_baseline(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig.from_profile("summarization")
        assert hasattr(cfg, "nli_summarization_baseline")

    def test_build_scorer_passes_baseline(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(
            use_nli=False,
            nli_summarization_baseline=0.15,
        )
        scorer = cfg.build_scorer()
        assert scorer._summarization_nli_baseline == 0.15
