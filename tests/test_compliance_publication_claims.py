# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — final EU AI Act publication claim guard
"""Lock the public compliance guides to the published final instrument."""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_GUIDES = (
    _ROOT / "docs" / "COMPLIANCE.md",
    _ROOT / "docs-site" / "guide" / "eu-ai-act-whitepaper.md",
    _ROOT / "docs-site" / "guide" / "compliance-reporting.md",
)
_OFFICIAL_URL = "https://eur-lex.europa.eu/eli/reg/2026/1744/oj"
_STALE_WORDING = (
    "awaiting formal publication",
    "awaiting publication in official journal",
    "verify current dates against the official journal",
)


@pytest.mark.parametrize("guide", _GUIDES, ids=lambda path: path.name)
def test_compliance_guide_cites_the_published_final_instrument(guide: Path) -> None:
    """Every affected public guide should identify and link the final act."""
    text = guide.read_text(encoding="utf-8")

    assert "Regulation (EU) 2026/1744" in text
    assert _OFFICIAL_URL in text
    assert "24 July 2026" in text
    assert "27 July 2026" in text
    assert "2 December 2027" in text
    assert "2 August 2028" in text


@pytest.mark.parametrize("guide", _GUIDES, ids=lambda path: path.name)
def test_compliance_guide_does_not_retain_prepublication_language(
    guide: Path,
) -> None:
    """Published-law guides must not ask readers to await the publication."""
    lowered = guide.read_text(encoding="utf-8").lower()

    for stale in _STALE_WORDING:
        assert stale not in lowered
