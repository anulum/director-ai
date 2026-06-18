# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — pricing tier-boundary documentation tests
"""Regression tests for the public package, pricing, and tier boundary."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _doc(path: str) -> str:
    """Read a repository document and collapse Markdown line wrapping."""

    return " ".join((ROOT / path).read_text(encoding="utf-8").split())


def test_readme_declares_three_tier_product_ladder() -> None:
    doc = _doc("README.md")

    assert "Director-Lite" in doc
    assert "pip install director-ai-lite" in doc
    assert "Director-AI Pro" in doc
    assert "Director-Class AI" in doc
    assert "not a separate wheel" in doc


def test_pricing_page_uses_usd_and_package_boundary() -> None:
    doc = _doc("docs-site/pricing.md")

    assert "Director-Lite" in doc
    assert "USD 0" in doc
    assert "Director-AI Pro self-host" in doc
    assert "USD 199/mo" in doc
    assert "Director-Class AI" in doc
    assert "not a separate wheel" in doc
    assert "CHF 199/mo" not in doc
    assert "CHF 49/mo" not in doc
    assert "polar.sh/checkout" not in doc
    assert "Request USD checkout" in doc


def test_lite_package_readme_points_to_upgrade_path() -> None:
    doc = _doc("packages/director-ai-lite/README.md")

    assert "Director-Lite" in doc
    assert "Director-AI" in doc
    assert "Director-Class AI" in doc
    assert "only the first two are Python packages" in doc


def test_readme_leads_with_evidence_and_buried_differentiators() -> None:
    doc = _doc("README.md")

    assert "Evidence-first deployment surfaces" in doc
    assert "Voice Guard" in doc
    assert "Inference-server hooks" in doc
    assert "Supply-chain controls" in doc
    assert "director-ai evidence --emit" in doc
    assert "docs-site/deployment/supply-chain.md" in doc
    assert "PayPal" not in doc
    assert "TWINT" not in doc
    assert "Crypto" not in doc


def test_public_architecture_collapses_internal_decision_maps() -> None:
    doc = _doc("ARCHITECTURE.md")

    assert "Public Runtime Boundary" in doc
    assert "Ownership Boundary" in doc
    assert "internal planning records" in doc
    assert "Safety Surface Map" not in doc
    assert "Responsibility Consolidation Map" not in doc


def test_public_docs_explain_market_applications_and_onboarding_spine() -> None:
    """Keep the public docs understandable for buyers and first-time builders."""

    map_doc = _doc("docs-site/guide/applications-and-market-map.md")
    readme = _doc("README.md")
    landing = _doc("docs-site/index.md")
    api_index = _doc("docs-site/api/index.md")
    tutorials = _doc("docs-site/tutorials.md")
    gallery = _doc("docs-site/notebook-gallery.md")

    for required in (
        "factual-coherence control plane",
        "What The Software Is",
        "Who Needs It",
        "Application Lanes",
        "Market Value",
        "What Ships Publicly",
        "First Evidence Packet",
        "Director-Lite",
        "Director-AI",
        "Director-Class AI",
        "Customer support",
        "Enterprise knowledge assistant",
        "Regulated review",
        "Streaming assistant",
        "Agent workflow",
        "Evaluation pipeline",
        "Platform deployment",
    ):
        assert required in map_doc

    for surface in (readme, landing, api_index, tutorials, gallery):
        assert "Applications and Market Map" in surface


def test_priority_notebooks_carry_product_context() -> None:
    """Notebook first cells must state the application and market context."""

    quickstart = _doc("notebooks/quickstart.ipynb")
    production = _doc("notebooks/09_production_guardrails.ipynb")
    enterprise = _doc("notebooks/14_enterprise_multi_tenant.ipynb")

    assert "Market lanes" in quickstart
    assert "applications-and-market-map.md" in quickstart
    assert "market question" in production
    assert "wrapped SDK path" in production
    assert "value beyond a local demo" in enterprise
    assert "tenant isolation" in enterprise
