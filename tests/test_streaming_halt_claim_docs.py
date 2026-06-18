# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — streaming-halt claim documentation tests
"""Regression tests for the public streaming-halt claim boundary."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_doc(path: str) -> str:
    """Read a public documentation page from the repository root."""

    return (ROOT / path).read_text(encoding="utf-8")


def _normalise_whitespace(text: str) -> str:
    """Collapse Markdown wrapping for phrase-level documentation assertions."""

    return " ".join(text.split())


def test_streaming_guide_documents_contradiction_replacement() -> None:
    doc = _normalise_whitespace(_read_doc("docs-site/guide/streaming.md"))

    assert "Contradiction halt replaces the old coherence callback" in doc
    assert "streaming_contradiction_halt_base.json" in doc
    assert "false-halt rate 0.0148" in doc
    assert "recall 0.6667" in doc


def test_market_docs_keep_unsupported_vs_contradiction_boundary() -> None:
    market = _normalise_whitespace(
        _read_doc("docs-site/guide/market-value-and-positioning.md")
    )
    landscape = _normalise_whitespace(
        _read_doc("docs-site/guide/guardrail-landscape.md")
    )

    assert "factual-coherence gate" in market
    assert "The streaming moat" not in market
    assert "streaming interlock is contradiction-driven" in market
    assert "contradiction signals can be applied during streaming" in market
    assert "absence of evidence is not the same signal as contradiction" in market
    assert "Opt-in, contradiction-driven" in landscape
    assert "coverage of every unsupported addition" in landscape
