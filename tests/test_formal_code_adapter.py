# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Formal math and code verifier adapter tests."""

from __future__ import annotations

from time import sleep

from director_ai.core.formal_verification import (
    And,
    FormalCodeVerifierAdapter,
    Not,
    Variable,
)
from director_ai.core.guard_control import RiskEnvelope


def _math_envelope() -> RiskEnvelope:
    return RiskEnvelope(
        action_category="code",
        reversibility="reversible",
        domain="regulated",
        calibrated_threshold=0.5,
        no_go_threshold=0.85,
    )


def test_contradictory_formula_halts_and_serialises_without_formula_text():
    adapter = FormalCodeVerifierAdapter()

    result = adapter.verify_formula(
        formula=And(Variable("approved"), Not(Variable("approved"))),
        risk_envelope=_math_envelope(),
        policy_id="policy.formal.regulated",
        evidence_ref="formal://claim-1",
    )

    assert result.guard_decision.decision == "halt"
    assert result.signal.verdict == "contradictory"
    assert "approved" not in str(result.to_dict())
    assert result.to_safety_event(hook_id="formal.code").hook_scope == "agent"


def test_unsupported_formal_backend_warns_instead_of_allowing():
    adapter = FormalCodeVerifierAdapter(formal_verifier=None)

    result = adapter.verify_formula(
        formula=Variable("claim"),
        risk_envelope=_math_envelope(),
        policy_id="policy.formal.regulated",
        evidence_ref="formal://claim-2",
    )

    assert result.guard_decision.decision == "warn"
    assert result.guard_decision.reason == "formal_backend_unsupported"
    assert result.signal.verdict == "unsupported"


def test_code_verifier_detects_hallucinated_api_without_executing_code():
    adapter = FormalCodeVerifierAdapter()
    source = "import pandas as pd\ndf = pd.read_quantum_csv('data.csv')"

    result = adapter.verify_code(
        code=source,
        risk_envelope=_math_envelope(),
        policy_id="policy.code.regulated",
        api_manifest={"pd": {"read_csv", "DataFrame"}},
        evidence_ref="code://snippet-1",
    )

    assert result.guard_decision.decision == "halt"
    assert result.signal.verdict == "invalid"
    assert "read_quantum_csv" not in str(result.to_dict())
    assert result.to_dict()["sandbox"]["execution_allowed"] is False


def test_code_verifier_exception_warns_not_passes():
    def broken_verifier(**kwargs):
        raise RuntimeError("backend failed")

    adapter = FormalCodeVerifierAdapter(code_verifier=broken_verifier)

    result = adapter.verify_code(
        code="print('safe')",
        risk_envelope=_math_envelope(),
        policy_id="policy.code.regulated",
        evidence_ref="code://snippet-2",
    )

    assert result.guard_decision.decision == "warn"
    assert result.guard_decision.reason == "code_verifier_failed"
    assert result.signal.verdict == "verifier_failed"


def test_verifier_timeout_warns_not_passes():
    def slow_verifier(**kwargs):
        sleep(0.01)
        return type(
            "CodeResult",
            (),
            {
                "syntax_valid": True,
                "unknown_imports": [],
                "hallucinated_apis": [],
                "error_count": 0,
            },
        )()

    adapter = FormalCodeVerifierAdapter(code_verifier=slow_verifier, timeout_ms=1.0)

    result = adapter.verify_code(
        code="print('safe')",
        risk_envelope=_math_envelope(),
        policy_id="policy.code.regulated",
        evidence_ref="code://snippet-3",
    )

    assert result.guard_decision.decision == "warn"
    assert result.guard_decision.reason == "code_verifier_timeout"
    assert result.signal.verdict == "timeout"
