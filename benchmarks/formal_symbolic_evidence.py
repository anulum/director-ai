# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — formal symbolic evidence packet

"""Generate local evidence for formal and symbolic guard paths.

The packet checks the production-relevant R16 primitives without executing
generated code:

* DPLL-backed formula verification maps contradictions to halts and tautologies
  to allows;
* the Lean profile calls only the operator-owned runner and records the backend
  in tenant-safe evidence;
* the Z3 profile either executes when the formal extra is installed or records
  the optional dependency gate explicitly;
* code-contract verification runs structural checks before theorem backends and
  omits raw source/formula text from serialised reports.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks._common import save_results
from benchmarks._provenance import resolve_git_sha
from director_ai.core.formal_verification import (
    And,
    FormalCodeVerifierAdapter,
    Implies,
    Not,
    Variable,
)
from director_ai.core.guard_control import RiskEnvelope


def _risk_envelope() -> RiskEnvelope:
    return RiskEnvelope(
        action_category="code",
        reversibility="reversible",
        domain="regulated",
        calibrated_threshold=0.5,
        no_go_threshold=0.85,
    )


def run_dpll_formula_probe() -> dict[str, Any]:
    """Return DPLL guard-decision evidence for SAT and UNSAT formulae."""
    adapter = FormalCodeVerifierAdapter.with_theorem_backend("dpll")
    contradiction = adapter.verify_formula(
        formula=And(Variable("private_claim"), Not(Variable("private_claim"))),
        risk_envelope=_risk_envelope(),
        policy_id="policy.formal.regulated",
        evidence_ref="formal://dpll-contradiction",
    )
    tautology = adapter.verify_formula(
        formula=Implies(Variable("private_claim"), Variable("private_claim")),
        risk_envelope=_risk_envelope(),
        policy_id="policy.formal.regulated",
        evidence_ref="formal://dpll-tautology",
    )
    serialised = json.dumps(
        {
            "contradiction": contradiction.to_dict(),
            "tautology": tautology.to_dict(),
        },
        sort_keys=True,
    )
    return {
        "name": "dpll_formula_guard",
        "contradiction_decision": contradiction.guard_decision.decision,
        "contradiction_verdict": contradiction.signal.verdict,
        "tautology_decision": tautology.guard_decision.decision,
        "tautology_verdict": tautology.signal.verdict,
        "backend": contradiction.sandbox["backend"],
        "raw_formula_leaked": "private_claim" in serialised,
        "passed": bool(
            contradiction.guard_decision.decision == "halt"
            and contradiction.signal.verdict == "contradictory"
            and tautology.guard_decision.decision == "allow"
            and tautology.signal.verdict == "consistent"
            and contradiction.sandbox["backend"] == "dpll"
            and "private_claim" not in serialised
        ),
    }


def run_lean_runner_probe() -> dict[str, Any]:
    """Return Lean adapter evidence using an operator-owned runner."""
    observed_sources: list[str] = []

    def runner(source: str) -> dict[str, Any]:
        observed_sources.append(source)
        return {"sat": False, "model": {}}

    adapter = FormalCodeVerifierAdapter.with_theorem_backend(
        "lean",
        lean_runner=runner,
    )
    result = adapter.verify_formula(
        formula=And(
            Variable("lean_private_claim"), Not(Variable("lean_private_claim"))
        ),
        risk_envelope=_risk_envelope(),
        policy_id="policy.formal.regulated",
        evidence_ref="formal://lean-contradiction",
    )
    serialised = json.dumps(result.to_dict(), sort_keys=True)
    return {
        "name": "lean_runner_contract",
        "runner_invoked": len(observed_sources) == 1,
        "runner_source_contains_target": bool(
            observed_sources and "def target" in observed_sources[0]
        ),
        "decision": result.guard_decision.decision,
        "verdict": result.signal.verdict,
        "backend": result.sandbox["backend"],
        "raw_formula_leaked": "lean_private_claim" in serialised,
        "passed": bool(
            len(observed_sources) == 1
            and "def target" in observed_sources[0]
            and result.guard_decision.decision == "halt"
            and result.signal.verdict == "contradictory"
            and result.sandbox["backend"] == "lean"
            and "lean_private_claim" not in serialised
        ),
    }


def run_z3_profile_probe() -> dict[str, Any]:
    """Return actual Z3 evidence when installed, otherwise the dependency gate."""
    if importlib.util.find_spec("z3") is None:
        error = ""
        try:
            FormalCodeVerifierAdapter.with_theorem_backend("z3")
        except ImportError as exc:
            error = str(exc)
        return {
            "name": "z3_profile_contract",
            "z3_installed": False,
            "actual_z3_run": False,
            "optional_dependency_gate": "director-ai[formal]" in error,
            "decision": "",
            "verdict": "",
            "passed": bool("director-ai[formal]" in error),
        }

    adapter = FormalCodeVerifierAdapter.with_theorem_backend("z3")
    result = adapter.verify_formula(
        formula=And(Variable("z3_private_claim"), Not(Variable("z3_private_claim"))),
        risk_envelope=_risk_envelope(),
        policy_id="policy.formal.regulated",
        evidence_ref="formal://z3-contradiction",
    )
    serialised = json.dumps(result.to_dict(), sort_keys=True)
    return {
        "name": "z3_profile_contract",
        "z3_installed": True,
        "actual_z3_run": True,
        "optional_dependency_gate": False,
        "decision": result.guard_decision.decision,
        "verdict": result.signal.verdict,
        "backend": result.sandbox["backend"],
        "raw_formula_leaked": "z3_private_claim" in serialised,
        "passed": bool(
            result.guard_decision.decision == "halt"
            and result.signal.verdict == "contradictory"
            and result.sandbox["backend"] == "z3"
            and "z3_private_claim" not in serialised
        ),
    }


def run_code_contract_probe() -> dict[str, Any]:
    """Return structural-code + theorem-contract guard evidence."""
    adapter = FormalCodeVerifierAdapter.with_theorem_backend("dpll")
    valid = adapter.verify_code_contract(
        code="def identity(x):\n    return x\n",
        contract=Implies(Variable("input_valid"), Variable("input_valid")),
        risk_envelope=_risk_envelope(),
        policy_id="policy.code.contract.regulated",
        evidence_ref="code-contract://identity",
    )
    invalid = adapter.verify_code_contract(
        code="def broken(:\n    pass",
        contract=Variable("unreached_private_contract"),
        risk_envelope=_risk_envelope(),
        policy_id="policy.code.contract.regulated",
        evidence_ref="code-contract://broken",
    )
    serialised = json.dumps(
        {"valid": valid.to_dict(), "invalid": invalid.to_dict()},
        sort_keys=True,
    )
    raw_payload_leaked = (
        "def identity" in serialised
        or "def broken" in serialised
        or "unreached_private_contract" in serialised
    )
    return {
        "name": "code_contract_guard",
        "valid_decision": valid.guard_decision.decision,
        "valid_contract_checked": valid.sandbox["contract_checked"],
        "invalid_decision": invalid.guard_decision.decision,
        "invalid_contract_checked": invalid.sandbox["contract_checked"],
        "raw_payload_leaked": raw_payload_leaked,
        "passed": bool(
            valid.guard_decision.decision == "allow"
            and valid.sandbox["contract_checked"] is True
            and invalid.guard_decision.decision == "halt"
            and invalid.sandbox["contract_checked"] is False
            and not raw_payload_leaked
        ),
    }


def run_formal_symbolic_evidence() -> dict[str, Any]:
    """Return the complete local R16 formal-symbolic evidence packet."""
    dpll = run_dpll_formula_probe()
    lean = run_lean_runner_probe()
    z3 = run_z3_profile_probe()
    code_contract = run_code_contract_probe()
    passed = bool(
        dpll["passed"] and lean["passed"] and z3["passed"] and code_contract["passed"]
    )
    return {
        "schema_version": "director-ai.formal-symbolic-evidence.v1",
        "benchmark": "formal_symbolic_evidence",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": resolve_git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "acceptance": {
            "passed": passed,
            "checks": {
                "dpll_formula_guard": bool(dpll["passed"]),
                "lean_runner_contract": bool(lean["passed"]),
                "z3_profile_contract": bool(z3["passed"]),
                "code_contract_guard": bool(code_contract["passed"]),
            },
            "limits": {
                "local_only": True,
                "external_lean_binary_run_included": False,
                "z3_actual_run_included": bool(z3["actual_z3_run"]),
                "operator_domain_contracts_included": False,
            },
        },
        "probes": {
            "dpll_formula_guard": dpll,
            "lean_runner_contract": lean,
            "z3_profile_contract": z3,
            "code_contract_guard": code_contract,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Director-AI formal symbolic evidence packet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to benchmarks/results/.",
    )
    args = parser.parse_args(argv)

    payload = run_formal_symbolic_evidence()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results saved to {args.output}")
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        save_results(payload, f"formal_symbolic_evidence_{stamp}.json")
    print(json.dumps(payload["acceptance"], indent=2))
    return 0 if payload["acceptance"]["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
