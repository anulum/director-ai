# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — trajectory rollback evidence tests

from __future__ import annotations

import json

from benchmarks import trajectory_rollback_evidence as evidence


def test_preflight_rollback_probe_reports_all_action_bands() -> None:
    packet = evidence.run_preflight_rollback_probe(simulations=4)

    assert packet["passed"] is True
    assert packet["repeat_status"] == "already_executed"
    assert packet["repeat_executed"] is False
    assert packet["hook_calls"] == [("rollback-halt", "trajectory_preflight_halt")]
    assert packet["raw_prompt_payload_included"] is False
    assert [(record["case"], record["status"]) for record in packet["records"]] == [
        ("proceed", "not_required"),
        ("warn", "armed"),
        ("halt", "executed"),
    ]
    assert all(record["matched"] for record in packet["records"])


def test_preflight_rollback_probe_validates_simulation_count() -> None:
    try:
        evidence.run_preflight_rollback_probe(simulations=3)
    except ValueError as exc:
        assert "simulations" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_failure_probe_reports_tenant_safe_error_type() -> None:
    packet = evidence.run_failure_probe()

    assert packet == {
        "name": "rollback_failure_sanitisation",
        "status": "failed",
        "executed": False,
        "error_type": "RuntimeError",
        "raw_error_payload_included": False,
        "passed": True,
    }


def test_trajectory_rollback_evidence_payload_has_acceptance_summary(
    monkeypatch,
) -> None:
    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")

    packet = evidence.run_trajectory_rollback_evidence(simulations=4)

    assert packet["schema_version"] == "director-ai.trajectory-rollback-evidence.v1"
    assert packet["benchmark"] == "trajectory_rollback_evidence"
    assert packet["git_commit"] == "abc123"
    assert packet["acceptance"] == {
        "passed": True,
        "checks": {
            "preflight_rollback_paths": True,
            "rollback_failure_sanitisation": True,
        },
        "limits": {
            "local_only": True,
            "external_operator_signoff_included": False,
            "live_undo_backend_included": False,
        },
    }
    assert set(packet["probes"]) == {
        "preflight_rollback_paths",
        "rollback_failure_sanitisation",
    }


def test_main_writes_requested_output_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")
    output = tmp_path / "trajectory-rollback.json"

    exit_code = evidence.main(
        [
            "--simulations",
            "4",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["acceptance"]["passed"] is True
