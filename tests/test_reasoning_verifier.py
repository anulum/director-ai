# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for reasoning chain verification pipeline (STRONG)."""

from __future__ import annotations

from director_ai.core.verification.reasoning_verifier import (
    ReasoningChainResult,
    extract_steps,
    verify_reasoning_chain,
)


class TestExtractSteps:
    def test_numbered_steps(self):
        text = "1. The sky is blue.\n2. Blue light scatters more.\n3. Therefore the sky appears blue."
        steps = extract_steps(text)
        assert len(steps) >= 2
        assert steps[-1].is_conclusion

    def test_bullet_steps(self):
        text = "- First we gather the data.\n- Then we analyze the patterns.\n- Finally we draw conclusions."
        steps = extract_steps(text)
        assert len(steps) >= 2

    def test_keyword_steps(self):
        text = "First, we identify the problem. Then, we propose a solution. Therefore, the issue is resolved."
        steps = extract_steps(text)
        assert len(steps) >= 2

    def test_single_sentence_no_steps(self):
        text = "The answer is 42."
        steps = extract_steps(text)
        assert len(steps) < 2

    def test_sentence_fallback(self):
        text = "The economy grew by 3% last year. This growth was driven by exports. Consequently unemployment fell."
        steps = extract_steps(text)
        assert len(steps) >= 2


class TestVerifyReasoningChain:
    def test_coherent_chain(self):
        text = (
            "1. Water boils at 100 degrees Celsius at sea level.\n"
            "2. At higher altitudes, atmospheric pressure is lower.\n"
            "3. Lower pressure means water boils at a lower temperature.\n"
            "4. Therefore, water boils below 100 degrees at high altitude."
        )
        result = verify_reasoning_chain(text, support_threshold=0.8)
        assert result.steps_found >= 3
        assert isinstance(result, ReasoningChainResult)

    def test_non_sequitur_detected(self):
        text = (
            "1. The stock market rose by 5% today.\n"
            "2. My cat prefers tuna over salmon.\n"
            "3. Therefore, inflation will decrease."
        )
        result = verify_reasoning_chain(text, support_threshold=0.3)
        assert result.issues_found > 0

    def test_circular_reasoning(self):
        # Steps 1 and 2 share > 85% word overlap (near-identical)
        text = (
            "1. The policy is good and effective for the company.\n"
            "2. The policy is good and effective for the company.\n"
            "3. Therefore the policy should be adopted."
        )
        result = verify_reasoning_chain(text, support_threshold=0.8)
        circular = [v for v in result.verdicts if v.verdict == "circular"]
        assert len(circular) >= 1

    def test_short_text_no_verification(self):
        result = verify_reasoning_chain("Just one sentence.")
        assert result.steps_found < 2
        assert result.chain_valid

    def test_custom_score_fn(self):
        def always_supported(premise, hypothesis):
            return 0.1  # low divergence = supported

        text = "1. First step. 2. Second step. 3. Third step."
        result = verify_reasoning_chain(text, score_fn=always_supported)
        assert all(v.verdict == "supported" for v in result.verdicts)

    def test_custom_score_fn_all_unsupported(self):
        def never_supported(premise, hypothesis):
            return 0.99

        text = "1. Premise is stated here clearly.\n2. Completely different conclusion about weather.\n3. Yet another unrelated topic about music."
        result = verify_reasoning_chain(
            text, score_fn=never_supported, support_threshold=0.3
        )
        unsupported = [v for v in result.verdicts if v.verdict != "supported"]
        assert len(unsupported) >= 1


class TestReasoningChainResult:
    def test_properties(self):
        result = verify_reasoning_chain(
            "1. A causes B.\n2. Completely unrelated statement about Z.\n3. Therefore C."
        )
        assert result.steps_found >= 2
        assert isinstance(result.non_sequiturs, list)
        assert isinstance(result.unsupported_leaps, list)

    def test_empty_result(self):
        result = ReasoningChainResult(steps_found=0)
        assert result.chain_valid
        assert result.non_sequiturs == []
        assert result.unsupported_leaps == []
