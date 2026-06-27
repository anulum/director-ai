# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — conformal routing evidence tests

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from pytest import MonkeyPatch

from benchmarks import conformal_routing_evidence as evidence


def test_git_commit_falls_back_when_git_is_unavailable(
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify conformal evidence handles missing and failing git clients."""

    module = cast(Any, evidence)
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)
    assert module._git_commit() == "unknown"

    monkeypatch.setattr(module.shutil, "which", lambda _name: "git")

    def raise_subprocess(*_args: object, **_kwargs: object) -> None:
        raise module.subprocess.SubprocessError()

    monkeypatch.setattr(module.subprocess, "run", raise_subprocess)
    assert module._git_commit() == "unknown"

    class Completed:
        stdout = "abc123\n"

    def complete_subprocess(*_args: object, **_kwargs: object) -> Completed:
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", complete_subprocess)
    assert module._git_commit() == "abc123"


def test_coverage_probe_meets_target_with_reliable_intervals() -> None:
    """Verify conformal calibration reaches the requested coverage target."""

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


def test_coverage_probe_validates_sample_counts() -> None:
    """Verify conformal coverage evidence rejects undersized samples."""

    for kwargs in (
        {"calibration_samples": 1, "validation_samples": 20},
        {"calibration_samples": 40, "validation_samples": 1},
    ):
        try:
            evidence.run_coverage_probe(**kwargs)
        except ValueError as exc:
            assert "samples" in str(exc) or "sample_count" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


def test_routing_probe_reports_all_expected_operational_paths() -> None:
    """Verify conformal routing emits every production routing action."""

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


def test_conformal_routing_evidence_payload_has_acceptance_summary(
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify the R10 packet records acceptance checks and release limits."""

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


def test_main_writes_requested_output_path(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify the R10 CLI writes the requested evidence artifact."""

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


def test_main_uses_default_results_path(monkeypatch: MonkeyPatch) -> None:
    """Verify the R10 CLI saves to the default benchmark results path."""

    saved: list[str] = []

    def save_results(payload: object, filename: str) -> None:
        saved.append(filename)

    monkeypatch.setattr(evidence, "save_results", save_results)
    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")

    assert evidence.main(
        [
            "--calibration-samples",
            "40",
            "--validation-samples",
            "20",
            "--min-samples",
            "30",
        ]
    ) == 0
    assert len(saved) == 1
    assert saved[0].startswith("conformal_routing_evidence_")
