# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Neuro-Symbolic Verifier Tests

from director_ai.core import NeuroSymbolicVerifier, NeuroSymbolicVerifierInput
from director_ai.core.formal_verification import Not, ReasoningStep, Variable


def test_neural_score_passes_but_numeric_error_rejects():
    verifier = NeuroSymbolicVerifier(neural_accept_threshold=0.6)

    result = verifier.verify(
        NeuroSymbolicVerifierInput(
            text="There is a 150% probability of success.",
            neural_score=0.91,
            evidence_ref="claim://probability",
        )
    )

    assert result.decision == "reject"
    assert result.numeric_result is not None
    assert result.numeric_result.valid is False
    assert result.symbolic_verdict is None
    assert result.to_dict()["text"] is None


def test_symbolic_contradiction_rejects_even_with_good_neural_score():
    verifier = NeuroSymbolicVerifier(neural_accept_threshold=0.6)

    result = verifier.verify(
        NeuroSymbolicVerifierInput(
            text="A and not A.",
            neural_score=0.95,
            symbolic_steps=(
                ReasoningStep("A", Variable("A")),
                ReasoningStep("not-A", Not(Variable("A"))),
            ),
            evidence_ref="formal://claim-a",
        )
    )

    assert result.decision == "reject"
    assert result.symbolic_verdict is not None
    assert result.symbolic_verdict.contradictory is True
    assert "symbolic_contradiction" in result.reasons


def test_low_neural_score_warns_when_symbolic_checks_pass():
    verifier = NeuroSymbolicVerifier(neural_accept_threshold=0.8)

    result = verifier.verify(
        NeuroSymbolicVerifierInput(
            text="There is a 60% probability of success.",
            neural_score=0.7,
            symbolic_steps=(ReasoningStep("A", Variable("A")),),
        )
    )

    assert result.decision == "warn"
    assert result.numeric_result is not None
    assert result.numeric_result.valid is True
    assert result.symbolic_verdict is not None
    assert result.symbolic_verdict.consistent is True


def test_valid_high_confidence_claim_is_allowed():
    verifier = NeuroSymbolicVerifier(neural_accept_threshold=0.6)

    result = verifier.verify(
        NeuroSymbolicVerifierInput(
            text="There is a 60% probability of success.",
            neural_score=0.9,
            symbolic_steps=(ReasoningStep("A", Variable("A")),),
        )
    )

    assert result.decision == "allow"
    assert result.reasons == ()


def test_serialisation_can_include_text_only_when_requested():
    verifier = NeuroSymbolicVerifier(neural_accept_threshold=0.6)
    result = verifier.verify(
        NeuroSymbolicVerifierInput(
            text="There is a 60% probability of success.",
            neural_score=0.9,
        )
    )

    assert "60% probability" not in str(result.to_dict())
    assert "60% probability" in str(result.to_dict(include_text=True))
