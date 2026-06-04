# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — local release evidence gate tests

from __future__ import annotations

import json

from tools.check_local_release_evidence import (
    REQUIRED_PACKETS,
    evaluate_release_evidence,
    main,
)


def _write_packet(
    root,
    benchmark: str,
    *,
    passed: bool = True,
    release_ready: bool = False,
) -> None:
    limits = (
        {
            "local_only": False,
            "external_operator_signoff_included": True,
            "representative_domain_dataset_included": True,
            "actual_wasm_build_included": True,
            "quantised_model_artefact_included": True,
            "browser_worker_smoke_included": True,
            "mobile_device_smoke_included": True,
            "package_publish_included": True,
        }
        if release_ready
        else {
            "local_only": True,
            "external_operator_signoff_included": False,
        }
    )
    if benchmark == "sustained_load_evidence" and release_ready:
        limits["staging_or_production_telemetry_included"] = True
    payload = {
        "benchmark": benchmark,
        "acceptance": {
            "passed": passed,
            "limits": limits,
        },
    }
    if benchmark == "edge_mobile_evidence":
        payload["profiles"] = {
            "browser-worker": {
                "ready_for_release": release_ready,
            }
        }
    path = root / "benchmarks" / "results" / f"{benchmark}_20260604T000000Z.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_all_packets(root, *, release_ready: bool = False) -> None:
    for packet in REQUIRED_PACKETS:
        _write_packet(root, packet.benchmark, release_ready=release_ready)


def test_gate_allows_local_mode_for_passed_local_packets(tmp_path) -> None:
    _write_all_packets(tmp_path)

    gate = evaluate_release_evidence(tmp_path, mode="local")

    assert gate.ready is True
    assert gate.local_ready is True
    assert gate.release_ready is False
    assert all(packet.local_ready for packet in gate.packets)
    assert gate.blockers == ()
    assert any(
        blocker["code"] == "local_only_evidence"
        for blocker in gate.release_blockers
    )


def test_gate_blocks_release_mode_for_local_only_packets(tmp_path) -> None:
    _write_all_packets(tmp_path)

    gate = evaluate_release_evidence(tmp_path, mode="release")

    assert gate.ready is False
    assert gate.local_ready is True
    assert gate.release_ready is False
    codes = {blocker["code"] for blocker in gate.blockers}
    assert "local_only_evidence" in codes
    assert "edge_runtime_not_release_ready" in codes
    assert "missing_external_operator_signoff_included" in codes
    assert "missing_staging_or_production_telemetry_included" in codes


def test_gate_allows_release_when_all_release_limits_are_ready(tmp_path) -> None:
    _write_all_packets(tmp_path, release_ready=True)

    gate = evaluate_release_evidence(tmp_path, mode="release")

    assert gate.ready is True
    assert gate.local_ready is True
    assert gate.release_ready is True
    assert gate.blockers == ()
    assert gate.release_blockers == ()


def test_gate_blocks_release_when_sustained_load_packet_lacks_staging_limits(
    tmp_path,
) -> None:
    _write_all_packets(tmp_path, release_ready=True)
    path = (
        tmp_path
        / "benchmarks"
        / "results"
        / "sustained_load_evidence_20260604T000001Z.json"
    )
    path.write_text(
        json.dumps(
            {
                "benchmark": "sustained_load_evidence",
                "acceptance": {
                    "passed": True,
                    "async_ordering": True,
                    "tenant_poisoning": True,
                },
            },
        ),
        encoding="utf-8",
    )

    gate = evaluate_release_evidence(tmp_path, mode="release")

    assert gate.ready is False
    codes = {blocker["code"] for blocker in gate.blockers}
    assert "missing_staging_or_production_telemetry_included" in codes
    assert "missing_external_operator_signoff_included" in codes


def test_gate_blocks_missing_packet(tmp_path) -> None:
    _write_packet(tmp_path, REQUIRED_PACKETS[0].benchmark)

    gate = evaluate_release_evidence(tmp_path, mode="local")

    assert gate.ready is False
    assert gate.local_ready is False
    assert any(
        blocker["code"] == "evidence_packet_missing" for blocker in gate.blockers
    )


def test_cli_writes_json_report_for_local_mode(tmp_path) -> None:
    _write_all_packets(tmp_path)
    output = tmp_path / "report.json"

    exit_code = main(["--root", str(tmp_path), "--mode", "local", "--json", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["schema_version"] == "director-ai.local-release-evidence-gate.v1"
    assert report["ready"] is True
    assert report["release_ready"] is False


def test_cli_prints_json_report_to_stdout_for_ci(tmp_path, capsys) -> None:
    _write_all_packets(tmp_path)

    exit_code = main(["--root", str(tmp_path), "--mode", "local", "--format", "json"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert report["mode"] == "local"
    assert report["ready"] is True
    assert report["release_ready"] is False
    assert "# Local Release Evidence Gate" not in captured.out
