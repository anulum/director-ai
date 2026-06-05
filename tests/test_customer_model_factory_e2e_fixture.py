# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory end-to-end fixture tests

from __future__ import annotations

import json
from pathlib import Path

from tools.generate_customer_model_factory_fixture import main as fixture_main
from tools.verify_customer_model_factory_docs_freeze import main as freeze_main

ROOT = Path(__file__).resolve().parents[1]


EXPECTED_MANIFESTS = {
    "dataset_report.json",
    "training_manifest.json",
    "benchmark_result.json",
    "selection_report.json",
    "deployment_manifest.json",
    "sector_evidence_mapping.json",
    "evidence_pack.json",
    "runtime_package.json",
    "monitoring_manifest.json",
    "risk_register.json",
    "observability_operations_evidence.json",
    "provenance_lineage_evidence.json",
    "conformal_routing_evidence.json",
    "trajectory_rollback_evidence.json",
    "multimodal_temporal_evidence.json",
    "deployment_hardening_evidence.json",
    "enterprise_readiness.json",
    "release_gate.json",
}


def test_e2e_fixture_generator_writes_all_release_artifacts(tmp_path: Path):
    output_dir = tmp_path / "fixture"

    exit_code = fixture_main(["--output-dir", str(output_dir)])

    assert exit_code == 0
    assert {path.name for path in output_dir.iterdir()} == EXPECTED_MANIFESTS
    release_gate = json.loads(
        (output_dir / "release_gate.json").read_text(encoding="utf-8")
    )
    runtime_package = json.loads(
        (output_dir / "runtime_package.json").read_text(encoding="utf-8")
    )
    risk_register = json.loads(
        (output_dir / "risk_register.json").read_text(encoding="utf-8")
    )

    assert release_gate["promotion_allowed"] is True
    assert (
        release_gate["observability_operations_evidence"]["environment"] == "staging"
    )
    assert release_gate["provenance_lineage_evidence"]["environment"] == "staging"
    assert release_gate["conformal_routing_evidence"]["environment"] == "staging"
    assert release_gate["trajectory_rollback_evidence"]["environment"] == "staging"
    assert release_gate["multimodal_temporal_evidence"]["environment"] == "staging"
    assert release_gate["deployment_hardening_evidence"]["environment"] == "staging"
    assert (
        release_gate["artifact_hashes"]["runtime_hash"]
        == runtime_package["runtime_hash"]
    )
    assert (
        release_gate["artifact_hashes"]["risk_register_hash"]
        == risk_register["register_hash"]
    )
    assert release_gate["customer_id"] == "customer-alpha"
    assert release_gate["tenant_id"] == "customer-alpha-tenant"


def test_e2e_fixture_generator_is_deterministic(tmp_path: Path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    assert fixture_main(["--output-dir", str(first_dir)]) == 0
    assert fixture_main(["--output-dir", str(second_dir)]) == 0

    for filename in EXPECTED_MANIFESTS:
        assert (first_dir / filename).read_text(encoding="utf-8") == (
            second_dir / filename
        ).read_text(encoding="utf-8")


def test_docs_freeze_verifier_accepts_current_customer_model_factory_surface():
    exit_code = freeze_main(["--root", str(ROOT)])

    assert exit_code == 0


def test_docs_freeze_verifier_reports_missing_required_surface(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("Customer Model Factory\n", encoding="utf-8")

    exit_code = freeze_main(["--root", str(root)])

    assert exit_code == 1
