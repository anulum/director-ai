# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - external security test packet tests

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "security" / "external_security_test_packet.toml"
PACKET_DOC = ROOT / "security" / "EXTERNAL_SECURITY_TEST_PACKET.md"


def _packet() -> dict:
    return tomllib.loads(PACKET.read_text(encoding="utf-8"))


def test_external_security_packet_tracks_required_surfaces():
    packet = _packet()
    tracks = {track["id"]: track for track in packet["test_tracks"]}

    assert set(tracks) == {
        "streaming_interception",
        "multi_tenant_isolation",
        "knowledge_ingestion",
        "physical_hooks",
        "attestation",
        "cross_language_trust_boundary",
    }
    assert "/v1/stream" in tracks["streaming_interception"]["surfaces"]
    assert (
        "/v1/tenants/{tenant_id}/facts" in tracks["multi_tenant_isolation"]["surfaces"]
    )
    assert "/v1/knowledge/ingest" in tracks["knowledge_ingestion"]["surfaces"]
    assert "GroundingHook.evaluate" in tracks["physical_hooks"]["surfaces"]
    assert "PassportVerifier" in tracks["attestation"]["surfaces"]
    assert "director.v1 protobuf" in tracks["cross_language_trust_boundary"]["surfaces"]


def test_external_security_packet_paths_exist():
    packet = _packet()

    assert (ROOT / packet["packet_doc"]).exists()
    assert (ROOT / packet["runbook_doc"]).exists()
    assert (ROOT / packet["evidence_validator"]).exists()
    assert (ROOT / packet["policy_doc"]).exists()
    assert packet["run_status"] == "not_run"

    for track in packet["test_tracks"]:
        for source_file in track["source_files"]:
            assert (ROOT / source_file).exists(), source_file
        for test_file in track["existing_regression_tests"]:
            assert (ROOT / test_file).exists(), test_file
        assert track["required_checks"]
        assert track["required_evidence"]


def test_external_security_packet_required_outputs():
    packet = _packet()
    outputs = {item["path"] for item in packet["required_outputs"]}

    assert {
        "security-validation/environment.json",
        "security-validation/http_transcripts/",
        "security-validation/websocket_frames.jsonl",
        "security-validation/tenant_matrix.csv",
        "security-validation/ingestion_matrix.csv",
        "security-validation/physical_matrix.csv",
        "security-validation/attestation_matrix.csv",
        "security-validation/contract_matrix.csv",
        "security-validation/findings.jsonl",
        "security-validation/summary.md",
    } <= outputs

    acceptance = packet["acceptance"]
    assert acceptance["minimum_tracks"] == 6
    assert acceptance["require_raw_http_transcripts"]
    assert acceptance["require_websocket_frame_log"]
    assert acceptance["require_tenant_matrix"]
    assert acceptance["require_ingestion_matrix"]
    assert acceptance["require_physical_matrix"]
    assert acceptance["require_attestation_matrix"]
    assert acceptance["require_contract_matrix"]


def test_external_security_doc_contains_required_sections():
    doc = PACKET_DOC.read_text(encoding="utf-8")
    runbook = (ROOT / "security" / "EXTERNAL_SECURITY_TEST_RUNBOOK.md").read_text(
        encoding="utf-8"
    )

    for heading in [
        "## Test Scope",
        "## Streaming Interception Checks",
        "## Multi-Tenant Isolation Checks",
        "## Knowledge-Base Ingestion Checks",
        "## Physical Hook Checks",
        "## Attestation Checks",
        "## Cross-Language Trust Boundary Checks",
        "## Required Outputs",
        "## Report Rules",
    ]:
        assert heading in doc

    assert "security/external_security_test_packet.toml" in doc
    assert "tools/validate_external_security_run.py" in doc
    assert "## Completion Rule" in runbook
    assert "streaming_interception" in runbook
    assert "multi_tenant_isolation" in runbook
    assert "physical_hooks" in runbook
    assert "attestation" in runbook
    assert "cross_language_trust_boundary" in runbook
