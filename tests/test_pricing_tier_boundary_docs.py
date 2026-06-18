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
