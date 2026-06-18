# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory documentation compliance tests

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mkdocs_nav_exposes_customer_model_factory_guide_and_api():
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    assert "Customer Model Factory: guide/customer-model-factory.md" in mkdocs
    assert "Customer Model Factory: api/customer-model-factory.md" in mkdocs


def test_customer_model_factory_guide_documents_operator_workflow():
    guide = (ROOT / "docs-site" / "guide" / "customer-model-factory.md").read_text(
        encoding="utf-8"
    )

    required_tokens = {
        "Customer Model Factory",
        "Dataset validation",
        "Training manifest",
        "Benchmark selection",
        "Deployment manifest",
        "Sector-extension boundary",
        "Evidence pack",
        "Runtime package",
        "Monitoring manifest",
        "Risk register",
        "Release gate",
        "tools/generate_customer_model_factory_fixture.py",
        "tools/assemble_customer_model_factory_release.py",
        "tools/verify_customer_model_factory_compliance.py",
        "tools/verify_customer_model_factory_docs_freeze.py",
        "tools/verify_public_sector_boundary.py",
        "Customer-specific accuracy claims require package-specific benchmark evidence",
    }
    assert required_tokens <= set(_matched_tokens(guide, required_tokens))


def test_customer_model_factory_api_page_documents_all_new_modules():
    api_page = (ROOT / "docs-site" / "api" / "customer-model-factory.md").read_text(
        encoding="utf-8"
    )

    for module in (
        "dataset_contract",
        "training_manifest",
        "benchmark_selection",
        "deployment_manifest",
        "sector_extension",
        "evidence_pack",
        "runtime_package",
        "monitoring_manifest",
        "risk_register",
        "release_gate",
    ):
        assert f"director_ai.core.customer_model_factory.{module}" in api_page


def _matched_tokens(text: str, tokens: set[str]) -> tuple[str, ...]:
    return tuple(token for token in tokens if token in text)
