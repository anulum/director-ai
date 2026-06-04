# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — conformal routing evidence tests

from __future__ import annotations

import json

from benchmarks import conformal_routing_evidence as evidence


def test_coverage_probe_meets_target_with_reliable_intervals() -> None:
    packet = evidence.run_coverage_probe(
        coverage=0.95,
        calibration_samples=40,
        validation_samples=20,
        min_samples=30,
    )

    assert packet["passed"] is True
    assert packet["target_coverage"] == 0.95
    assert packet["empirical_coverage"] >= 0.95
    assert packet["coverage_failures"] == 0
    assert packet["reliable"] is True


def test_routing_probe_reports_all_expected_operational_paths() -> None:
    packet = evidence.run_routing_probe(
        coverage=0.95,
        calibration_samples=40,
        min_samples=30,
    )

    assert packet["passed"] is True
    assert packet["action_counts"] == {
        "allow": 1,
        "human_review": 2,
        "escalate": 1,
        "reject": 1,
    }
    assert {decision["case"] for decision in packet["decisions"]} == {
        "low_risk",
        "ambiguous_mid",
        "uncertain_high",
        "high_risk",
        "uncalibrated",
    }
    assert all(decision["matched"] for decision in packet["decisions"])


def test_conformal_routing_evidence_payload_has_acceptance_summary(monkeypatch) -> None:
    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")

    packet = evidence.run_conformal_routing_evidence(
        coverage=0.95,
        calibration_samples=40,
        validation_samples=20,
        min_samples=30,
    )

    assert packet["schema_version"] == "director-ai.conformal-routing-evidence.v1"
    assert packet["benchmark"] == "conformal_routing_evidence"
    assert packet["git_commit"] == "abc123"
    assert packet["acceptance"] == {
        "passed": True,
        "checks": {
            "coverage_calibration": True,
            "routing_decisions": True,
        },
        "limits": {
            "local_only": True,
            "external_operator_signoff_included": False,
            "representative_domain_dataset_included": False,
        },
    }
    assert set(packet["probes"]) == {"coverage_calibration", "routing_decisions"}


def test_main_writes_requested_output_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")
    output = tmp_path / "conformal-routing.json"

    exit_code = evidence.main(
        [
            "--calibration-samples",
            "40",
            "--validation-samples",
            "20",
            "--min-samples",
            "30",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["acceptance"]["passed"] is True
