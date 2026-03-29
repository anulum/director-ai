# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Dialogue FPR Profile Tests

"""Tests for automatic dialogue detection and bidirectional NLI scoring."""

from __future__ import annotations

import pytest

from director_ai.core.scorer import _DIALOGUE_TURN_RE, CoherenceScorer

# â”€â”€ Task detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestDetectTaskType:
    """_detect_task_type correctly classifies dialogue vs default prompts."""

    def test_single_speaker_is_not_dialogue(self):
        prompt = "User: What is the capital of France?"
        assert CoherenceScorer._detect_task_type(prompt) != "dialogue"

    def test_two_speakers_is_dialogue(self):
        prompt = "User: Hi\nAssistant: Hello, how can I help?"
        assert CoherenceScorer._detect_task_type(prompt) == "dialogue"

    def test_multi_turn_dialogue(self):
        prompt = (
            "User: What is the weather today?\n"
            "Assistant: It's sunny and warm.\n"
            "User: Should I bring an umbrella?\n"
            "Assistant: No, you should be fine."
        )
        assert CoherenceScorer._detect_task_type(prompt) == "dialogue"

    def test_human_ai_markers(self):
        prompt = "Human: Tell me a joke.\nAI: Why did the chicken cross the road?"
        assert CoherenceScorer._detect_task_type(prompt) == "dialogue"

    def test_customer_agent_markers(self):
        prompt = "Customer: I need help.\nAgent: Sure, what's the issue?"
        assert CoherenceScorer._detect_task_type(prompt) == "dialogue"

    def test_speaker_with_number(self):
        prompt = "Speaker 1: Hello.\nSpeaker 2: Hi there."
        assert CoherenceScorer._detect_task_type(prompt) == "dialogue"

    def test_bracket_format(self):
        prompt = "[User] What time is it?\n[Assistant] It is 3pm."
        assert CoherenceScorer._detect_task_type(prompt) == "dialogue"

    def test_plain_text_is_default(self):
        prompt = "The capital of France is Paris. It is known for the Eiffel Tower."
        assert CoherenceScorer._detect_task_type(prompt) == "default"

    def test_document_with_colon_is_not_dialogue(self):
        prompt = "Title: Climate Change\nAbstract: Global temperatures are rising."
        assert CoherenceScorer._detect_task_type(prompt) != "dialogue"

    def test_case_insensitive(self):
        prompt = "user: Hello\nassistant: Hi"
        assert CoherenceScorer._detect_task_type(prompt) == "dialogue"

    def test_system_prefix(self):
        prompt = "System: You are a helpful assistant.\nUser: Hi.\nAssistant: Hello!"
        assert CoherenceScorer._detect_task_type(prompt) == "dialogue"

    def test_single_line_dialogue(self):
        """HaluEval format: all turns on one line separated by spaces."""
        prompt = "[Human]: Do you like Iron Man [Assistant]: Sure do!"
        assert CoherenceScorer._detect_task_type(prompt) == "dialogue"

    def test_single_line_with_colon(self):
        prompt = "User: Hi there Assistant: Hello, how can I help?"
        assert CoherenceScorer._detect_task_type(prompt) == "dialogue"


# â”€â”€ Dialogue regex â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestDialogueRegex:
    """_DIALOGUE_TURN_RE matches expected speaker patterns."""

    @pytest.mark.parametrize(
        "text",
        [
            "User: hello",
            "  User: hello",
            "\nUser: hello",
            "Assistant: hi",
            "Human: hi",
            "AI: response",
            "Bot: answer",
            "Customer: help",
            "Agent: sure",
            "System: ready",
            "Speaker 1: hello",
            "Speaker 2: hi",
            "[User] hello",
            "[Assistant] hi",
        ],
    )
    def test_matches_speaker(self, text):
        assert _DIALOGUE_TURN_RE.search(text) is not None

    @pytest.mark.parametrize(
        "text",
        [
            "Title: Something",
            "Abstract: Some text",
            "Note: important",
            "hello world",
        ],
    )
    def test_no_match_non_speaker(self, text):
        assert _DIALOGUE_TURN_RE.search(text) is None


# â”€â”€ Profile resolution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestResolveAggProfile:
    """_resolve_agg_profile returns min-mean for dialogue, defaults otherwise."""

    def test_dialogue_prompt_gets_min_mean(self):
        scorer = CoherenceScorer(use_nli=False)
        prompt = "User: What is AI?\nAssistant: Artificial Intelligence.\nUser: Tell me more."
        fi, fo, li, lo = scorer._resolve_agg_profile(prompt)
        assert fi == "min"
        assert fo == "mean"
        assert li == "min"
        assert lo == "mean"

    def test_plain_prompt_gets_defaults(self):
        scorer = CoherenceScorer(use_nli=False)
        prompt = "What is the capital of France?"
        fi, fo, li, lo = scorer._resolve_agg_profile(prompt)
        assert fi == "max"
        assert fo == "max"
        assert li == "max"
        assert lo == "max"

    def test_disabled_auto_profile(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._auto_dialogue_profile = False
        prompt = "User: Hi\nAssistant: Hello"
        fi, fo, li, lo = scorer._resolve_agg_profile(prompt)
        assert fi == "max"
        assert fo == "max"

    def test_custom_agg_not_overridden(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._fact_inner_agg = "min"
        scorer._fact_outer_agg = "trimmed_mean"
        prompt = "User: Hi\nAssistant: Hello"
        fi, fo, li, lo = scorer._resolve_agg_profile(prompt)
        # User's custom settings preserved — auto-profile only applies to defaults
        assert fi == "min"
        assert fo == "trimmed_mean"

    def test_summarization_profile_not_overridden(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._use_prompt_as_premise = True
        prompt = "User: Hi\nAssistant: Hello"
        fi, fo, li, lo = scorer._resolve_agg_profile(prompt)
        assert fi == "max"
        assert fo == "max"


# â”€â”€ Dialogue baseline calibration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestDialogueBaseline:
    """Dialogue NLI baseline calibration attribute and logic."""

    def test_default_baseline(self):
        scorer = CoherenceScorer(use_nli=False)
        assert scorer._dialogue_nli_baseline == 0.80

    def test_custom_baseline(self):
        scorer = CoherenceScorer(use_nli=False)
        scorer._dialogue_nli_baseline = 0.85
        assert scorer._dialogue_nli_baseline == 0.85

    def test_calibration_formula_zero(self):
        """Score at baseline â†’ adjusted divergence = 0."""
        baseline = 0.80
        raw = 0.80
        adjusted = max(0.0, (raw - baseline) / (1.0 - baseline))
        assert adjusted == pytest.approx(0.0)

    def test_calibration_formula_midpoint(self):
        """Score at midpoint between baseline and 1.0."""
        baseline = 0.80
        raw = 0.90
        adjusted = max(0.0, (raw - baseline) / (1.0 - baseline))
        assert adjusted == pytest.approx(0.5)

    def test_calibration_formula_max(self):
        """Score at 1.0 â†’ adjusted divergence = 1.0."""
        baseline = 0.80
        raw = 1.0
        adjusted = max(0.0, (raw - baseline) / (1.0 - baseline))
        assert adjusted == pytest.approx(1.0)

    def test_calibration_below_baseline(self):
        """Score below baseline â†’ clamped to 0."""
        baseline = 0.80
        raw = 0.60
        adjusted = max(0.0, (raw - baseline) / (1.0 - baseline))
        assert adjusted == pytest.approx(0.0)


# â”€â”€ Dialogue detection in heuristic_coherence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestDialogueDetectionInCoherence:
    """_heuristic_coherence correctly identifies dialogue for special path."""

    def test_dialogue_skips_logical_divergence(self):
        """For dialogue without NLI, the path falls through to standard scoring.
        NLI is required for the dialogue bidirectional path.
        """
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        prompt = (
            "User: What color is the sky?\nAssistant: It's blue.\nUser: Are you sure?"
        )
        response = "Yes, the sky is typically blue during clear days."
        h_logic, h_fact, coherence, evidence = scorer._heuristic_coherence(
            prompt,
            response,
        )
        # Without NLI, standard heuristic path is used
        assert isinstance(h_logic, float)
        assert isinstance(h_fact, float)
        assert 0.0 <= coherence <= 1.0

    def test_non_dialogue_uses_standard_path(self):
        """Non-dialogue prompts always use the standard scoring path."""
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        prompt = "The capital of France is Paris."
        response = "Paris is the capital of France."
        h_logic, h_fact, coherence, evidence = scorer._heuristic_coherence(
            prompt,
            response,
        )
        assert isinstance(h_logic, float)
        assert isinstance(h_fact, float)


# â”€â”€ Integration: review() with dialogue â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestDialogueReview:
    """review() applies dialogue profile transparently."""

    def test_dialogue_review_runs(self):
        """Smoke test: dialogue prompt doesn't crash review()."""
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        prompt = (
            "User: What color is the sky?\nAssistant: It's blue.\nUser: Are you sure?"
        )
        response = "Yes, the sky is typically blue during clear days."
        approved, score = scorer.review(prompt, response)
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_auto_profile_attribute_default(self):
        scorer = CoherenceScorer(use_nli=False)
        assert scorer._auto_dialogue_profile is True

    def test_dialogue_baseline_attribute_default(self):
        scorer = CoherenceScorer(use_nli=False)
        assert scorer._dialogue_nli_baseline == 0.80
