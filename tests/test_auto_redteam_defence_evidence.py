# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — auto-redteam defence evidence tests

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from pytest import MonkeyPatch

from benchmarks import auto_redteam_defence_evidence as evidence


def test_git_commit_falls_back_when_git_is_unavailable(
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify redteam evidence handles missing and failing git clients."""

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


def test_repeated_cycle_probe_promotes_two_versions_without_prompt_leak() -> None:
    """Verify repeated redteam cycles promote guards without prompt leakage."""

    packet = evidence.run_repeated_cycle_probe(
        min_failures=8,
        min_detection_uplift=0.5,
    )

    assert packet["passed"] is True
    assert packet["cycles_run"] == 2
    assert packet["active_version"] == 3
    assert packet["history_versions"] == [1, 2]
    assert packet["promoted_versions"] == [2, 3]
    assert all(rate == 0.0 for rate in packet["baseline_detection_rates"])
    assert all(rate == 1.0 for rate in packet["candidate_detection_rates"])
    assert packet["tenant_safe_reports"] is True
    assert packet["raw_prompt_leaked"] is False


def test_repeated_cycle_probe_validates_failure_count() -> None:
    """Verify redteam evidence rejects empty failure windows."""

    try:
        evidence.run_repeated_cycle_probe(min_failures=0, min_detection_uplift=0.5)
    except ValueError as exc:
        assert "min_failures" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_failure_builder_validates_positive_count() -> None:
    """Verify the redteam failure builder rejects empty event counts."""

    module = cast(Any, evidence)
    try:
        module._failures("marker", 0)
    except ValueError as exc:
        assert "count" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_auto_redteam_defence_evidence_payload_has_acceptance_summary(
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify the R15 packet records acceptance checks and release limits."""

    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")

    packet = evidence.run_auto_redteam_defence_evidence(
        min_failures=8,
        min_detection_uplift=0.5,
    )

    assert packet["schema_version"] == "director-ai.auto-redteam-defence-evidence.v1"
    assert packet["benchmark"] == "auto_redteam_defence_evidence"
    assert packet["git_commit"] == "abc123"
    assert packet["acceptance"] == {
        "passed": True,
        "checks": {
            "repeated_auto_redteam_cycles": True,
            "tenant_safe_reports": True,
            "registry_promotions": True,
        },
        "limits": {
            "local_only": True,
            "live_nightly_workflow_included": False,
            "operator_patch_signoff_included": False,
            "external_adversarial_corpus_included": False,
        },
    }
    assert set(packet["probes"]) == {"repeated_auto_redteam_cycles"}


def test_main_writes_requested_output_path(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify the R15 CLI writes the requested evidence artifact."""

    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")
    output = tmp_path / "auto-redteam-defence.json"

    exit_code = evidence.main(["--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["acceptance"]["passed"] is True


def test_main_uses_default_results_path(monkeypatch: MonkeyPatch) -> None:
    """Verify the R15 CLI saves to the default benchmark results path."""

    saved: list[str] = []

    def save_results(payload: object, filename: str) -> None:
        saved.append(filename)

    monkeypatch.setattr(evidence, "save_results", save_results)
    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")

    assert evidence.main(["--min-failures", "8"]) == 0
    assert len(saved) == 1
    assert saved[0].startswith("auto_redteam_defence_evidence_")
