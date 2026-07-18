# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — edge/mobile evidence tests

from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from benchmarks import edge_mobile_evidence as evidence
from director_ai.core.edge.runtime_profile import EdgeRuntimeCheck, EdgeRuntimeReadiness


def test_edge_mobile_evidence_payload_records_truthful_release_limits(
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify the R14 packet records local readiness and release blockers."""

    monkeypatch.setattr(evidence, "resolve_git_sha", lambda _repo: "abc123")

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


def test_edge_mobile_evidence_detects_leaked_raw_paths(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify tenant-safe serialisation fails if a raw absolute path leaks."""

    raw_path = tmp_path / "model.onnx"

    def passed_check(name: str, *items: str) -> EdgeRuntimeCheck:
        return EdgeRuntimeCheck(
            name=name,
            status="pass",
            summary="test check",
            evidence=tuple(items),
        )

    def build_readiness(_repo: Path, **_kwargs: object) -> EdgeRuntimeReadiness:
        return EdgeRuntimeReadiness(
            schema_version="director-ai.edge-runtime-readiness.v1",
            target_id="browser-worker",
            runtime="Browser Web Worker",
            wasm_target="web",
            ready_for_local_trial=True,
            ready_for_release=False,
            checks=(
                passed_check("wasm_release_plan"),
                passed_check("wasm_source_contract"),
                passed_check("quantised_nli_contract", str(raw_path)),
                passed_check("rust_kernel_source_contract"),
                passed_check("edge_deployment_docs"),
                passed_check("latency_benchmark_surface"),
            ),
            findings=(),
            limits={"local_only": True},
        )

    monkeypatch.setattr(evidence, "build_edge_runtime_readiness", build_readiness)
    monkeypatch.setattr(evidence, "resolve_git_sha", lambda _repo: "abc123")

    packet = evidence.run_edge_mobile_evidence(quantised_model_path=raw_path)

    assert packet["acceptance"]["checks"]["tenant_safe_serialisation"] is False
    assert packet["acceptance"]["passed"] is False


def test_edge_mobile_evidence_omits_raw_external_paths(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify external model paths are redacted from edge evidence."""

    monkeypatch.setattr(evidence, "resolve_git_sha", lambda _repo: "abc123")
    outside = tmp_path.parent / "edge-model.onnx"

    packet = evidence.run_edge_mobile_evidence(quantised_model_path=outside)
    serialised = json.dumps(packet, sort_keys=True)

    assert packet["acceptance"]["checks"]["tenant_safe_serialisation"] is True
    assert "external path not serialised" in serialised
    assert str(outside) not in serialised


def test_main_writes_requested_output_path(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify the R14 CLI writes the requested evidence artifact."""

    monkeypatch.setattr(evidence, "resolve_git_sha", lambda _repo: "abc123")
    output = tmp_path / "edge-mobile.json"

    exit_code = evidence.main(["--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["acceptance"]["passed"] is True


def test_main_uses_default_results_path(monkeypatch: MonkeyPatch) -> None:
    """Verify the R14 CLI saves to the default benchmark results path."""

    saved: list[str] = []

    def save_results(payload: object, filename: str) -> None:
        saved.append(filename)

    monkeypatch.setattr(evidence, "save_results", save_results)
    monkeypatch.setattr(evidence, "resolve_git_sha", lambda _repo: "abc123")

    assert evidence.main([]) == 0
    assert len(saved) == 1
    assert saved[0].startswith("edge_mobile_evidence_")
