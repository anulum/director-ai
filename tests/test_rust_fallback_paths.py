# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Python Fallback Path Tests
"""Test Python fallback paths when Rust accelerators are disabled.

Monkeypatches `_RUST_*` flags to False, forcing Python codepaths that
are normally bypassed when backfire_kernel is installed. Ensures both
Rust and Python paths produce identical results and that all branches
(threshold boundaries, edge cases, error paths) are covered.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

# ── LiteScorer Python fallback ──────────────────────────────────────


class TestLiteScorerPythonFallback:
    """Test Python path of LiteScorer.score() with _RUST_LITE=False."""

    def _score_py(self, premise: str, hypothesis: str) -> float:
        """Force Python path by patching _RUST_LITE."""
        import director_ai.core.scoring.lite_scorer as mod

        with patch.object(mod, "_RUST_LITE", False):
            scorer = mod.LiteScorer()
            return scorer.score(premise, hypothesis)

    def test_identical_texts(self):
        s = self._score_py("The sky is blue.", "The sky is blue.")
        assert s < 0.01

    def test_empty_premise(self):
        assert abs(self._score_py("", "something") - 0.5) < 1e-9

    def test_empty_hypothesis(self):
        assert abs(self._score_py("hello", "") - 0.5) < 1e-9

    def test_both_empty(self):
        assert abs(self._score_py("", "") - 0.5) < 1e-9

    def test_no_word_tokens(self):
        # Strings with no \w+ matches
        assert abs(self._score_py("...", "???") - 0.5) < 1e-9

    def test_negation_asymmetry(self):
        base = self._score_py("The product works.", "The product works.")
        neg = self._score_py("The product works.", "The product does not work.")
        assert neg > base

    def test_entity_both_present(self):
        s = self._score_py("Apple is great.", "Samsung is great.")
        # Different entities → ent_overlap < 1 → higher divergence
        assert s > 0.05

    def test_entity_one_side_only(self):
        s = self._score_py("Apple released a phone.", "someone released a phone.")
        # p_ents has Apple, h_ents is empty → ent_overlap = 0.0
        assert s > 0.0

    def test_entity_neither_side(self):
        s = self._score_py("the sky is blue today.", "the sky is blue today.")
        # No capitalised entities → ent_overlap = 0.5 (neutral)
        # Identical text → low divergence but not zero due to ent_overlap=0.5 vs 1.0
        assert s < 0.15

    def test_completely_different(self):
        s = self._score_py(
            "Quantum computing uses qubits.",
            "The recipe calls for flour and sugar.",
        )
        assert s > 0.5

    def test_negation_both_sides(self):
        # Both have negation → no penalty
        s = self._score_py(
            "The system does not crash.",
            "The system never fails.",
        )
        # neg_penalty = 0.0 (both have negation)
        base = self._score_py(
            "The system works well.",
            "The system never fails.",
        )
        # base has negation asymmetry (one side only)
        assert base > s

    def test_score_batch_python(self):
        import director_ai.core.scoring.lite_scorer as mod

        with patch.object(mod, "_RUST_LITE", False):
            scorer = mod.LiteScorer()
            pairs = [
                ("The sky is blue.", "The sky is blue."),
                ("Yes it works.", "No it does not work."),
            ]
            results = scorer.score_batch(pairs)
            assert len(results) == 2
            assert results[0] < results[1]

    def test_review_approved(self):
        import director_ai.core.scoring.lite_scorer as mod

        with patch.object(mod, "_RUST_LITE", False):
            scorer = mod.LiteScorer()
            approved, score = scorer.review("The sky is blue.", "The sky is blue.")
            assert approved
            assert score.score > 0.5
            assert score.approved

    def test_review_rejected(self):
        import director_ai.core.scoring.lite_scorer as mod

        with patch.object(mod, "_RUST_LITE", False):
            scorer = mod.LiteScorer()
            approved, score = scorer.review(
                "Quantum computing uses qubits.",
                "The recipe calls for flour.",
                threshold=0.9,
            )
            assert not approved
            assert not score.approved

    def test_review_custom_threshold(self):
        import director_ai.core.scoring.lite_scorer as mod

        with patch.object(mod, "_RUST_LITE", False):
            scorer = mod.LiteScorer()
            approved, score = scorer.review(
                "The sky is blue.", "The sky is blue.", threshold=0.99
            )
            assert approved
            assert score.h_logical == score.h_factual

    def test_parity_rust_vs_python(self):
        """Verify Python fallback produces same results as Rust path."""
        import director_ai.core.scoring.lite_scorer as mod

        cases = [
            ("The sky is blue.", "The sky is green."),
            ("Hello world.", "Hello world."),
            ("", "something"),
            ("text", ""),
            ("Apple is great.", "Samsung is great."),
            ("No entities here.", "no entities here either."),
        ]
        scorer = mod.LiteScorer()
        for premise, hypothesis in cases:
            rust_result = scorer.score(premise, hypothesis)  # uses Rust
            py_result = self._score_py(premise, hypothesis)  # forces Python
            assert abs(rust_result - py_result) < 1e-9, (
                f"Parity failed for ({premise!r}, {hypothesis!r}): "
                f"rust={rust_result}, python={py_result}"
            )


# ── NLI helpers Python fallback ─────────────────────────────────────


class TestNLISoftmaxFallback:
    """Test _softmax_np Python path (small batch < 100 elements)."""

    def test_small_batch_uses_python(self):
        from director_ai.core.scoring.nli import _softmax_np

        # 3x3 = 9 elements < 100 threshold → Python path
        x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.0, 0.0, 10.0]])
        result = _softmax_np(x)
        assert result.shape == (3, 3)
        # Each row sums to 1
        for i in range(3):
            assert abs(result[i].sum() - 1.0) < 1e-9

    def test_large_batch_uses_rust(self):
        from director_ai.core.scoring.nli import _softmax_np

        # 50x3 = 150 elements >= 100 threshold → Rust path
        rng = np.random.default_rng(42)
        x = rng.standard_normal((50, 3))
        result = _softmax_np(x)
        assert result.shape == (50, 3)
        for i in range(50):
            assert abs(result[i].sum() - 1.0) < 1e-9

    def test_boundary_99_elements_python(self):
        from director_ai.core.scoring.nli import _softmax_np

        # 33x3 = 99 elements < 100 → Python
        x = np.ones((33, 3))
        result = _softmax_np(x)
        # Uniform → each element ~0.333
        assert abs(result[0, 0] - 1 / 3) < 1e-6

    def test_boundary_100_elements_rust(self):
        from director_ai.core.scoring.nli import _softmax_np

        # 50x2 = 100 elements >= 100 → Rust
        x = np.zeros((50, 2))
        result = _softmax_np(x)
        assert abs(result[0, 0] - 0.5) < 1e-6

    def test_forced_python_path(self):
        import director_ai.core.scoring.nli as mod

        with patch.object(mod, "_RUST_NLI", False):
            x = np.array([[1.0, 2.0, 3.0]] * 50)  # 150 elements
            result = mod._softmax_np(x)
            assert result.shape == (50, 3)
            for i in range(50):
                assert abs(result[i].sum() - 1.0) < 1e-9


class TestNLIDivergenceFallback:
    """Test _probs_to_divergence Python path (< 10 rows)."""

    def test_small_batch_2class(self):
        from director_ai.core.scoring.nli import _probs_to_divergence

        probs = np.array([[0.3, 0.7], [0.8, 0.2]])
        divs = _probs_to_divergence(probs)
        assert abs(divs[0] - 0.3) < 1e-9  # 1 - 0.7
        assert abs(divs[1] - 0.8) < 1e-9  # 1 - 0.2

    def test_small_batch_3class(self):
        from director_ai.core.scoring.nli import _probs_to_divergence

        probs = np.array([[0.2, 0.3, 0.5]])  # 1 row < 10
        divs = _probs_to_divergence(probs)
        # P(contra=0.5) + 0.5 * P(neutral=0.3) = 0.65
        assert abs(divs[0] - 0.65) < 1e-9

    def test_custom_label_indices(self):
        from director_ai.core.scoring.nli import _probs_to_divergence

        probs = np.array([[0.1, 0.6, 0.3]])
        # contra=0, neutral=2 instead of default 2,1
        divs = _probs_to_divergence(probs, label_indices=(0, 2))
        # P(contra=0.1) + 0.5 * P(neutral=0.3) = 0.25
        assert abs(divs[0] - 0.25) < 1e-9

    def test_large_batch_uses_rust(self):
        from director_ai.core.scoring.nli import _probs_to_divergence

        # 15 rows >= 10 → Rust path
        probs = np.full((15, 3), 1 / 3)
        divs = _probs_to_divergence(probs)
        assert len(divs) == 15

    def test_forced_python_path(self):
        import director_ai.core.scoring.nli as mod

        with patch.object(mod, "_RUST_NLI", False):
            probs = np.full((20, 3), 1 / 3)
            divs = mod._probs_to_divergence(probs)
            assert len(divs) == 20
            # Uniform 3-class: contra(1/3) + 0.5*neutral(1/3) ≈ 0.5
            assert abs(divs[0] - 0.5) < 0.01


class TestNLIConfidenceFallback:
    """Test _probs_to_confidence Python path (< 10 rows)."""

    def test_uniform_low_confidence(self):
        from director_ai.core.scoring.nli import _probs_to_confidence

        probs = np.array([[0.5, 0.5]])  # 1 row < 10 → Python
        confs = _probs_to_confidence(probs)
        assert confs[0] < 0.01  # max entropy

    def test_certain_high_confidence(self):
        from director_ai.core.scoring.nli import _probs_to_confidence

        probs = np.array([[0.001, 0.999]])
        confs = _probs_to_confidence(probs)
        assert confs[0] > 0.9

    def test_3class_uniform(self):
        from director_ai.core.scoring.nli import _probs_to_confidence

        probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
        confs = _probs_to_confidence(probs)
        assert confs[0] < 0.01

    def test_large_batch_uses_rust(self):
        from director_ai.core.scoring.nli import _probs_to_confidence

        probs = np.full((15, 3), 1 / 3)
        confs = _probs_to_confidence(probs)
        assert len(confs) == 15

    def test_forced_python_path(self):
        import director_ai.core.scoring.nli as mod

        with patch.object(mod, "_RUST_NLI", False):
            probs = np.full((20, 2), 0.5)
            confs = mod._probs_to_confidence(probs)
            assert len(confs) == 20
            assert all(c < 0.01 for c in confs)


# ── Temporal freshness Python fallback ──────────────────────────────


class TestTemporalFreshnessFallback:
    """Test Python path with source_timestamp (bypasses Rust)."""

    def test_with_source_timestamp_uses_python(self):
        import time

        from director_ai.core.scoring.temporal_freshness import score_temporal_freshness

        # source_timestamp is provided → always Python path
        result = score_temporal_freshness(
            "The CEO of Apple is Tim Cook.",
            source_timestamp=time.time() - 86400 * 30,  # 30 days old
        )
        assert result.has_temporal_claims
        assert any(c.claim_type == "position" for c in result.claims)

    def test_stale_source_increases_risk(self):
        import time

        from director_ai.core.scoring.temporal_freshness import score_temporal_freshness

        fresh = score_temporal_freshness(
            "The CEO of Apple is Tim Cook.",
            source_timestamp=time.time(),  # just now
        )
        stale = score_temporal_freshness(
            "The CEO of Apple is Tim Cook.",
            source_timestamp=time.time() - 86400 * 365,  # 1 year old
        )
        assert stale.overall_staleness_risk > fresh.overall_staleness_risk

    def test_forced_python_path(self):
        import director_ai.core.scoring.temporal_freshness as mod

        with patch.object(mod, "_RUST_TEMPORAL", False):
            result = mod.score_temporal_freshness("The CEO of Apple is Tim Cook.")
            assert result.has_temporal_claims
            assert any(c.claim_type == "position" for c in result.claims)

    def test_statistic_claim_python(self):
        import director_ai.core.scoring.temporal_freshness as mod

        with patch.object(mod, "_RUST_TEMPORAL", False):
            result = mod.score_temporal_freshness(
                "The population of Japan is 125 million."
            )
            assert any(c.claim_type == "statistic" for c in result.claims)

    def test_current_reference_python(self):
        import director_ai.core.scoring.temporal_freshness as mod

        with patch.object(mod, "_RUST_TEMPORAL", False):
            result = mod.score_temporal_freshness(
                "As of 2024, the market share was 15%."
            )
            assert any(c.claim_type == "current_reference" for c in result.claims)

    def test_record_claim_python(self):
        import director_ai.core.scoring.temporal_freshness as mod

        with patch.object(mod, "_RUST_TEMPORAL", False):
            result = mod.score_temporal_freshness(
                "The tallest building in the world is Burj Khalifa."
            )
            assert any(c.claim_type == "record" for c in result.claims)

    def test_stale_claims_property(self):
        import director_ai.core.scoring.temporal_freshness as mod

        with patch.object(mod, "_RUST_TEMPORAL", False):
            result = mod.score_temporal_freshness("The CEO of Apple is Tim Cook.")
            # staleness_risk > 0.5 for position claims with unknown source
            stale = result.stale_claims
            assert len(stale) >= 1


# ── Reasoning verifier Python fallback ──────────────────────────────


class TestReasoningVerifierFallback:
    """Test Python path of extract_steps and _word_overlap."""

    def test_extract_steps_python_numbered(self):
        import director_ai.core.verification.reasoning_verifier as mod

        with patch.object(mod, "_RUST_REASONING", False):
            steps = mod.extract_steps(
                "Step 1: First premise.\nStep 2: Second premise.\nStep 3: Conclusion."
            )
            assert len(steps) >= 2

    def test_extract_steps_python_bullets(self):
        import director_ai.core.verification.reasoning_verifier as mod

        with patch.object(mod, "_RUST_REASONING", False):
            steps = mod.extract_steps("- Point A\n- Point B\n- Point C")
            assert len(steps) == 3

    def test_extract_steps_python_sentence_fallback(self):
        import director_ai.core.verification.reasoning_verifier as mod

        with patch.object(mod, "_RUST_REASONING", False):
            steps = mod.extract_steps(
                "First we examine the data carefully. "
                "Then we draw a reasonable conclusion from the analysis."
            )
            assert len(steps) >= 2

    def test_word_overlap_python(self):
        import director_ai.core.verification.reasoning_verifier as mod

        with patch.object(mod, "_RUST_REASONING", False):
            assert abs(mod._word_overlap("hello world", "hello world") - 1.0) < 1e-9
            assert mod._word_overlap("hello world", "foo bar") == 0.0

    def test_word_overlap_empty(self):
        import director_ai.core.verification.reasoning_verifier as mod

        with patch.object(mod, "_RUST_REASONING", False):
            assert mod._word_overlap("", "hello") == 0.0
            assert mod._word_overlap("hello", "") == 0.0

    def test_verify_chain_python(self):
        import director_ai.core.verification.reasoning_verifier as mod

        with patch.object(mod, "_RUST_REASONING", False):
            result = mod.verify_reasoning_chain(
                "Step 1: All mammals are warm-blooded animals.\n"
                "Step 2: A whale is a warm-blooded mammal animal.\n"
                "Step 3: Therefore, a whale is warm-blooded mammal."
            )
            assert result.steps_found >= 3
            # Heuristic overlap should find support between steps
            assert len(result.verdicts) >= 3

    def test_conclusion_detection(self):
        import director_ai.core.verification.reasoning_verifier as mod

        with patch.object(mod, "_RUST_REASONING", False):
            steps = mod.extract_steps(
                "Step 1: Premise.\n"
                "Step 2: Evidence.\n"
                "Step 3: Therefore the conclusion follows."
            )
            conclusions = [s for s in steps if s.is_conclusion]
            assert len(conclusions) >= 1


# ── Sanitizer scrub() coverage ──────────────────────────────────────


class TestSanitizerScrubCoverage:
    """Additional scrub() tests for full coverage."""

    def test_scrub_bidi_override(self):
        from director_ai.core.safety.sanitizer import InputSanitizer

        san = InputSanitizer()
        text = "Hello \u202eworld\u202c normal"
        cleaned = san.scrub(text)
        assert "\u202e" not in cleaned

    def test_scrub_mixed_control_and_normal(self):
        from director_ai.core.safety.sanitizer import InputSanitizer

        san = InputSanitizer()
        text = "Normal\x00text\x01with\x02controls"
        cleaned = san.scrub(text)
        assert "\x00" not in cleaned
        assert "\x01" not in cleaned
        assert "Normal" in cleaned

    def test_scrub_preserves_newlines(self):
        from director_ai.core.safety.sanitizer import InputSanitizer

        san = InputSanitizer()
        text = "Line 1\nLine 2\tTabbed"
        cleaned = san.scrub(text)
        assert "\n" in cleaned
        assert "\t" in cleaned


class TestSanitizerUnicodeFallback:
    """Test _has_suspicious_unicode Python path with _RUST_SANITIZER=False."""

    def test_clean_text_python(self):
        import director_ai.core.safety.sanitizer as mod

        with patch.object(mod, "_RUST_SANITIZER", False):
            san = mod.InputSanitizer()
            assert not san._has_suspicious_unicode("Normal ASCII text")

    def test_empty_text_python(self):
        import director_ai.core.safety.sanitizer as mod

        with patch.object(mod, "_RUST_SANITIZER", False):
            san = mod.InputSanitizer()
            assert not san._has_suspicious_unicode("")

    def test_bidi_override_python(self):
        import director_ai.core.safety.sanitizer as mod

        with patch.object(mod, "_RUST_SANITIZER", False):
            san = mod.InputSanitizer()
            # > 15% suspicious chars (Cf category)
            text = "\u202e\u202e\u202eab"  # 3/5 = 60%
            assert san._has_suspicious_unicode(text)

    def test_low_ratio_not_suspicious(self):
        import director_ai.core.safety.sanitizer as mod

        with patch.object(mod, "_RUST_SANITIZER", False):
            san = mod.InputSanitizer()
            # 1 suspicious char in long text → below threshold
            text = "A" * 100 + "\u202e"
            assert not san._has_suspicious_unicode(text)

    def test_private_use_area(self):
        import director_ai.core.safety.sanitizer as mod

        with patch.object(mod, "_RUST_SANITIZER", False):
            san = mod.InputSanitizer()
            # Co (private use) chars — high ratio
            text = "\ue000\ue001\ue002ab"  # 3/5 = 60%
            assert san._has_suspicious_unicode(text)


class TestLiteScorerBatchRustPath:
    """Test Rust batch path (line 95) which returns list via FFI."""

    def test_batch_rust_path(self):
        from director_ai.core.scoring.lite_scorer import LiteScorer

        scorer = LiteScorer()
        pairs = [
            ("The sky is blue.", "The sky is blue."),
            ("Yes.", "No."),
            ("Apple is great.", "Samsung is great."),
        ]
        results = scorer.score_batch(pairs)
        assert len(results) == 3
        assert all(isinstance(r, float) for r in results)
