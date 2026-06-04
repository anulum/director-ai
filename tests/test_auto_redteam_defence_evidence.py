# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — auto-redteam defence evidence tests

from __future__ import annotations

import json

from benchmarks import auto_redteam_defence_evidence as evidence


def test_repeated_cycle_probe_promotes_two_versions_without_prompt_leak() -> None:
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


def test_auto_redteam_defence_evidence_payload_has_acceptance_summary(
    monkeypatch,
) -> None:
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


def test_main_writes_requested_output_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")
    output = tmp_path / "auto-redteam-defence.json"

    exit_code = evidence.main(["--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["acceptance"]["passed"] is True
