# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — formal symbolic evidence tests

from __future__ import annotations

import json

from benchmarks import formal_symbolic_evidence as evidence


def test_dpll_formula_probe_halts_contradictions_without_formula_leak() -> None:
    packet = evidence.run_dpll_formula_probe()

    assert packet["passed"] is True
    assert packet["contradiction_decision"] == "halt"
    assert packet["contradiction_verdict"] == "contradictory"
    assert packet["tautology_decision"] == "allow"
    assert packet["tautology_verdict"] == "consistent"
    assert packet["backend"] == "dpll"
    assert packet["raw_formula_leaked"] is False


def test_lean_runner_probe_uses_runner_without_external_binary() -> None:
    packet = evidence.run_lean_runner_probe()

    assert packet["passed"] is True
    assert packet["runner_invoked"] is True
    assert packet["runner_source_contains_target"] is True
    assert packet["decision"] == "halt"
    assert packet["backend"] == "lean"
    assert packet["raw_formula_leaked"] is False


def test_z3_profile_probe_records_actual_run_or_optional_gate() -> None:
    packet = evidence.run_z3_profile_probe()

    assert packet["passed"] is True
    assert packet["name"] == "z3_profile_contract"
    assert packet["actual_z3_run"] is packet["z3_installed"]
    assert packet["optional_dependency_gate"] is (not packet["z3_installed"])
    if packet["z3_installed"]:
        assert packet["decision"] == "halt"
        assert packet["backend"] == "z3"


def test_code_contract_probe_checks_code_before_contract_and_omits_raw_text() -> None:
    packet = evidence.run_code_contract_probe()

    assert packet["passed"] is True
    assert packet["valid_decision"] == "allow"
    assert packet["valid_contract_checked"] is True
    assert packet["invalid_decision"] == "halt"
    assert packet["invalid_contract_checked"] is False
    assert packet["raw_payload_leaked"] is False


def test_formal_symbolic_evidence_payload_has_acceptance_summary(monkeypatch) -> None:
    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")

    packet = evidence.run_formal_symbolic_evidence()

    assert packet["schema_version"] == "director-ai.formal-symbolic-evidence.v1"
    assert packet["benchmark"] == "formal_symbolic_evidence"
    assert packet["git_commit"] == "abc123"
    assert packet["acceptance"]["passed"] is True
    assert packet["acceptance"]["checks"] == {
        "dpll_formula_guard": True,
        "lean_runner_contract": True,
        "z3_profile_contract": True,
        "code_contract_guard": True,
    }
    assert set(packet["probes"]) == {
        "dpll_formula_guard",
        "lean_runner_contract",
        "z3_profile_contract",
        "code_contract_guard",
    }


def test_main_writes_requested_output_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")
    output = tmp_path / "formal-symbolic.json"

    exit_code = evidence.main(["--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["acceptance"]["passed"] is True
