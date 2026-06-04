# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — edge/mobile evidence tests

from __future__ import annotations

import json

from benchmarks import edge_mobile_evidence as evidence


def test_edge_mobile_evidence_payload_records_truthful_release_limits(
    monkeypatch,
) -> None:
    monkeypatch.setattr(evidence, "_git_commit", lambda _repo: "abc123")

    packet = evidence.run_edge_mobile_evidence()

    assert packet["schema_version"] == "director-ai.edge-mobile-evidence.v1"
    assert packet["benchmark"] == "edge_mobile_evidence"
    assert packet["git_commit"] == "abc123"
    assert packet["acceptance"]["passed"] is True
    assert packet["acceptance"]["checks"]["browser_worker_local_trial_ready"] is True
    assert packet["acceptance"]["checks"]["wasm_source_contract"] is True
    assert packet["acceptance"]["checks"]["quantised_nli_contract"] is True
    assert packet["acceptance"]["limits"]["local_only"] is True
    assert packet["acceptance"]["limits"]["package_publish_included"] is False
    assert packet["profiles"]["browser-worker"]["ready_for_release"] is False


def test_edge_mobile_evidence_omits_raw_external_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(evidence, "_git_commit", lambda _repo: "abc123")
    outside = tmp_path.parent / "edge-model.onnx"

    packet = evidence.run_edge_mobile_evidence(quantised_model_path=outside)
    serialised = json.dumps(packet, sort_keys=True)

    assert packet["acceptance"]["checks"]["tenant_safe_serialisation"] is True
    assert "external path not serialised" in serialised
    assert str(outside) not in serialised


def test_main_writes_requested_output_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evidence, "_git_commit", lambda _repo: "abc123")
    output = tmp_path / "edge-mobile.json"

    exit_code = evidence.main(["--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["acceptance"]["passed"] is True
