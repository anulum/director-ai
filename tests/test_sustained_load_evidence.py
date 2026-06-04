# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — sustained load evidence tests

from __future__ import annotations

import json

from benchmarks import sustained_load_evidence as evidence


def test_async_ordering_probe_reports_clean_concurrent_streams() -> None:
    packet = evidence.run_async_ordering_probe(streams=4, tokens_per_stream=8)

    assert packet["passed"] is True
    assert packet["streams"] == 4
    assert packet["tokens_per_stream"] == 8
    assert packet["total_events"] == 32
    assert packet["failed_streams"] == 0
    assert packet["events_per_second"] > 0


def test_tenant_poisoning_probe_blocks_same_key_cross_tenant_contamination() -> None:
    packet = evidence.run_tenant_poisoning_probe(cases=4)

    assert packet["passed"] is True
    assert packet["cases"] == 4
    assert packet["writes"] == 8
    assert packet["queries"] == 12
    assert packet["failed_cases"] == 0


def test_tenant_poisoning_probe_default_scale_is_not_rank_saturation() -> None:
    packet = evidence.run_tenant_poisoning_probe()

    assert packet["passed"] is True
    assert packet["cases"] == 64
    assert packet["failed_cases"] == 0


def test_sustained_load_evidence_payload_has_acceptance_summary(monkeypatch) -> None:
    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")

    packet = evidence.run_sustained_load_evidence(
        streams=2,
        tokens_per_stream=4,
        tenant_cases=2,
    )

    assert packet["benchmark"] == "sustained_load_evidence"
    assert packet["git_commit"] == "abc123"
    assert packet["acceptance"] == {
        "passed": True,
        "async_ordering": True,
        "tenant_poisoning": True,
        "limits": {
            "local_only": True,
            "staging_or_production_telemetry_included": False,
            "external_operator_signoff_included": False,
        },
    }
    assert packet["probes"]["async_ordering"]["total_events"] == 8
    assert packet["probes"]["tenant_poisoning"]["cases"] == 2


def test_main_writes_requested_output_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")
    output = tmp_path / "evidence.json"

    exit_code = evidence.main(
        [
            "--streams",
            "2",
            "--tokens-per-stream",
            "4",
            "--tenant-cases",
            "2",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["acceptance"]["passed"] is True
