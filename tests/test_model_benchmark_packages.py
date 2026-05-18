# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - per-model benchmark package tests

from __future__ import annotations

from pathlib import Path

from benchmarks.model_benchmark_packages import (
    PACKAGE_MANIFEST_PATH,
    load_package_manifest,
    next_model_package_work,
    package_manifest_to_dict,
    validate_package_manifest,
)
from director_ai.core.scoring.model_choices import list_scorer_model_choices

ROOT = Path(__file__).resolve().parents[1]


def test_model_package_manifest_is_valid():
    manifest = load_package_manifest(PACKAGE_MANIFEST_PATH)
    findings = validate_package_manifest(manifest, root=ROOT)

    assert findings == []
    assert manifest.package_id == "director-ai-model-benchmark-packages-2026-05"
    assert manifest.default_status == "pending_external_suite"


def test_stable_runtime_models_have_required_benchmark_packages():
    manifest = load_package_manifest(PACKAGE_MANIFEST_PATH)
    packages = {package.model_alias: package for package in manifest.packages}
    stable_aliases = {choice.alias for choice in list_scorer_model_choices()}
    required_stage_ids = {stage.stage_id for stage in manifest.required_stages}

    assert stable_aliases <= set(packages)
    assert {
        "model_choice_general_gate",
        "aggrefact_anchor_vertex",
        "ragtruth_vertex",
        "halueval_vertex",
        "financebench_vertex",
        "legal_contractnli_vertex",
        "medical_mednli_pubmedqa_vertex",
        "patronus_halubench_wire",
    } <= required_stage_ids

    for alias in stable_aliases:
        package = packages[alias]
        assert package.status == "pending_external_suite"
        assert package.model_id
        assert package.runtime_model.startswith("gs://")
        assert {item.stage_id for item in package.evidence} == required_stage_ids


def test_next_model_package_work_returns_first_missing_evidence_item():
    manifest = load_package_manifest(PACKAGE_MANIFEST_PATH)

    work = next_model_package_work(manifest, completed_evidence_ids=set())

    assert work is not None
    assert work.model_alias == "balanced-default"
    assert work.stage_id == "aggrefact_anchor_vertex"
    assert work.vertex_allowed is True
    assert "DIRECTOR_SCORER_MODEL=balanced-default" in work.command


def test_package_manifest_serialises_without_token_placeholders():
    manifest = load_package_manifest(PACKAGE_MANIFEST_PATH)
    payload = package_manifest_to_dict(manifest)
    encoded = repr(payload)

    assert "model_benchmark_packages" in payload
    assert "<accepted-access-token>" not in encoded
    assert "secret" not in encoded.lower()


def test_readme_advertises_selectable_scorer_models_without_overclaiming():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Selectable scorer models" in readme
    assert "DIRECTOR_SCORER_MODEL=balanced-default" in readme
    assert "GET /v1/scorer/models" in readme
    assert "`balanced-default`" in readme
    assert "`deberta-small`" in readme
    assert "`deberta-large-nli`" in readme
    assert "per-model benchmark package" in readme
    assert "before public model-specific claims" in readme
