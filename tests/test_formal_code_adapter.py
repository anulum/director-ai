# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Formal math and code verifier adapter tests."""

from __future__ import annotations

from time import sleep

import pytest

from director_ai.core.formal_verification import (
    And,
    FormalCodeVerifierAdapter,
    Implies,
    Not,
    ReasoningVerdict,
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


def test_adapter_rejects_invalid_timeout_and_theorem_backend_inputs():
    with pytest.raises(ValueError, match="timeout_ms must be positive"):
        FormalCodeVerifierAdapter(timeout_ms=0)

    with pytest.raises(ValueError, match="lean_runner is required"):
        FormalCodeVerifierAdapter.with_theorem_backend("lean")

    with pytest.raises(ValueError, match="unsupported theorem backend"):
        FormalCodeVerifierAdapter.with_theorem_backend("isabelle")


def test_verify_formula_requires_evidence_reference():
    adapter = FormalCodeVerifierAdapter()

    with pytest.raises(ValueError, match="evidence_ref is required"):
        adapter.verify_formula(
            formula=Variable("claim"),
            risk_envelope=_math_envelope(),
            policy_id="policy.formal.regulated",
            evidence_ref=" ",
        )


def test_formal_verifier_exception_warns_not_passes():
    class BrokenVerifier:
        def verify(self, steps):
            raise RuntimeError("solver unavailable")

    adapter = FormalCodeVerifierAdapter(formal_verifier=BrokenVerifier())

    result = adapter.verify_formula(
        formula=Variable("claim"),
        risk_envelope=_math_envelope(),
        policy_id="policy.formal.regulated",
        evidence_ref="formal://claim-broken",
    )

    assert result.guard_decision.decision == "warn"
    assert result.guard_decision.reason == "formal_verifier_failed"
    assert result.signal.verdict == "verifier_failed"


def test_formal_verifier_timeout_warns_not_passes():
    class SlowVerifier:
        def verify(self, steps):
            sleep(0.01)
            return ReasoningVerdict(
                consistent=True,
                model={},
                step_count=len(steps),
                backend="slow",
            )

    adapter = FormalCodeVerifierAdapter(
        formal_verifier=SlowVerifier(),
        timeout_ms=1.0,
    )

    result = adapter.verify_formula(
        formula=Variable("claim"),
        risk_envelope=_math_envelope(),
        policy_id="policy.formal.regulated",
        evidence_ref="formal://claim-slow",
    )

    assert result.guard_decision.decision == "warn"
    assert result.guard_decision.reason == "formal_verifier_timeout"
    assert result.signal.verdict == "timeout"


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


def test_code_verifier_allows_clean_structural_result_and_preserves_sandbox():
    observed = {}

    def verifier(**kwargs):
        observed.update(kwargs)
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

    adapter = FormalCodeVerifierAdapter(code_verifier=verifier)

    result = adapter.verify_code(
        code="SELECT 1",
        language="sql",
        known_modules={"warehouse"},
        api_manifest={"warehouse": {"query"}},
        risk_envelope=_math_envelope(),
        policy_id="policy.code.regulated",
        evidence_ref="code://snippet-clean",
    )

    assert result.guard_decision.decision == "allow"
    assert result.guard_decision.reason == "code_verifier_supported"
    assert result.signal.verdict == "valid"
    assert result.sandbox == {
        "execution_allowed": False,
        "timeout_ms": 1000.0,
        "language": "sql",
    }
    assert observed["known_modules"] == {"warehouse"}
    assert observed["api_manifest"] == {"warehouse": {"query"}}


def test_verify_code_requires_evidence_reference():
    adapter = FormalCodeVerifierAdapter()

    with pytest.raises(ValueError, match="evidence_ref is required"):
        adapter.verify_code(
            code="print('safe')",
            risk_envelope=_math_envelope(),
            policy_id="policy.code.regulated",
            evidence_ref=" ",
        )


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


def test_named_dpll_backend_records_theorem_backend_in_sandbox():
    adapter = FormalCodeVerifierAdapter.with_theorem_backend("dpll")

    result = adapter.verify_formula(
        formula=Implies(Variable("p"), Variable("p")),
        risk_envelope=_math_envelope(),
        policy_id="policy.formal.regulated",
        evidence_ref="formal://claim-4",
    )

    assert result.guard_decision.decision == "allow"
    assert result.sandbox["backend"] == "dpll"
    assert result.signal.verifier == "formal.dpll"


def test_named_lean_backend_uses_runner_without_external_binary():
    def runner(source: str) -> dict:
        assert "def target" in source
        return {"sat": False}

    adapter = FormalCodeVerifierAdapter.with_theorem_backend(
        "lean",
        lean_runner=runner,
    )

    result = adapter.verify_formula(
        formula=And(Variable("p"), Not(Variable("p"))),
        risk_envelope=_math_envelope(),
        policy_id="policy.formal.regulated",
        evidence_ref="formal://lean-claim",
    )

    assert result.guard_decision.decision == "halt"
    assert result.sandbox["backend"] == "lean"
    assert result.signal.verifier == "formal.lean"


def test_code_contract_uses_code_verifier_before_theorem_backend():
    calls = []

    def verifier(**kwargs):
        calls.append(kwargs["code"])
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

    adapter = FormalCodeVerifierAdapter.with_theorem_backend(
        "dpll",
        code_verifier=verifier,
    )

    result = adapter.verify_code_contract(
        code="def identity(x):\n    return x\n",
        contract=Implies(Variable("input_valid"), Variable("output_valid")),
        risk_envelope=_math_envelope(),
        policy_id="policy.code.contract.regulated",
        evidence_ref="code-contract://identity",
    )

    assert calls == ["def identity(x):\n    return x\n"]
    assert result.kind == "code_contract"
    assert result.guard_decision.decision == "allow"
    assert result.sandbox["code_verifier"] == "structural"
    assert result.sandbox["theorem_backend"] == "dpll"
    assert "def identity" not in str(result.to_dict())


def test_code_contract_halts_before_theorem_backend_when_code_is_invalid():
    adapter = FormalCodeVerifierAdapter.with_theorem_backend("dpll")

    result = adapter.verify_code_contract(
        code="def broken(:\n    pass",
        contract=Variable("unreached_contract"),
        risk_envelope=_math_envelope(),
        policy_id="policy.code.contract.regulated",
        evidence_ref="code-contract://broken",
    )

    assert result.kind == "code_contract"
    assert result.guard_decision.decision == "halt"
    assert result.guard_decision.reason == "code_contract_rejected"
    assert "unreached_contract" not in str(result.to_dict())
