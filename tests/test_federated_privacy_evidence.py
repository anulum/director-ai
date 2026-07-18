# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — federated privacy evidence tests

from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from benchmarks import federated_privacy_evidence as evidence


def test_dp_signal_probe_caps_tenant_categories_and_omits_raw_payloads() -> None:
    """Verify DP signal aggregation stays tenant-safe and deduplicated."""

    packet = evidence.run_dp_signal_probe()

    assert packet["passed"] is True
    assert packet["accepted_first"] == ("decision:halt", "scope:streaming")
    assert packet["accepted_duplicate"] == ()
    assert packet["accepted_same_tenant"] == ()
    assert packet["accepted_second"] == ("decision:warn", "scope:streaming")
    assert packet["signal_count"] == 2
    assert packet["distinct_tenants"] == 2
    assert packet["payload_raw_counts_included"] is False
    assert packet["tenant_ids_leaked"] is False


def test_min_tenant_probe_blocks_release_without_charging_budget() -> None:
    """Verify the minimum-tenant gate blocks release without budget spend."""

    packet = evidence.run_min_tenant_probe()

    assert packet == {
        "name": "minimum_tenant_gate",
        "release_blocked": True,
        "accountant_epsilon": 0.0,
        "passed": True,
    }


def test_secret_sharing_probe_reconstructs_only_aggregate() -> None:
    """Verify secure aggregation exposes only the aggregate total."""

    packet = evidence.run_secret_sharing_probe()

    assert packet["passed"] is True
    assert packet["party_count"] == 3
    assert packet["submissions"] == 3
    assert packet["aggregate_total"] == 10
    assert packet["individual_party_values_included"] is False


def test_federated_privacy_evidence_payload_has_acceptance_summary(
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify the R13 packet records acceptance checks and release limits."""

    monkeypatch.setattr(evidence, "resolve_git_sha", lambda: "abc123")

    packet = evidence.run_federated_privacy_evidence()

    assert packet["schema_version"] == "director-ai.federated-privacy-evidence.v1"
    assert packet["benchmark"] == "federated_privacy_evidence"
    assert packet["git_commit"] == "abc123"
    assert packet["acceptance"] == {
        "passed": True,
        "checks": {
            "dp_signal_aggregation": True,
            "minimum_tenant_gate": True,
            "secure_additive_aggregation": True,
        },
        "limits": {
            "local_only": True,
            "external_federation_included": False,
            "malicious_secure_aggregation_proof_included": False,
        },
    }
    assert set(packet["probes"]) == {
        "dp_signal_aggregation",
        "minimum_tenant_gate",
        "secure_additive_aggregation",
    }


def test_main_writes_requested_output_path(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify the R13 CLI writes the requested evidence artifact."""

    monkeypatch.setattr(evidence, "resolve_git_sha", lambda: "abc123")
    output = tmp_path / "federated-privacy.json"

    exit_code = evidence.main(["--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["acceptance"]["passed"] is True


def test_main_uses_default_results_path(monkeypatch: MonkeyPatch) -> None:
    """Verify the R13 CLI saves to the default benchmark results path."""

    saved: list[str] = []

    def save_results(payload: object, filename: str) -> None:
        saved.append(filename)

    monkeypatch.setattr(evidence, "save_results", save_results)
    monkeypatch.setattr(evidence, "resolve_git_sha", lambda: "abc123")

    assert evidence.main([]) == 0
    assert len(saved) == 1
    assert saved[0].startswith("federated_privacy_evidence_")
