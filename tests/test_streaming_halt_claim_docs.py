# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — streaming-halt claim documentation tests
"""Regression tests for the public streaming-halt claim boundary."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_doc(path: str) -> str:
    """Read a public documentation page from the repository root."""
    return (ROOT / path).read_text(encoding="utf-8")


def _normalise_whitespace(text: str) -> str:
    """Collapse Markdown wrapping for phrase-level documentation assertions."""
    return " ".join(text.split())


def test_streaming_guide_documents_grounded_contradiction_halt() -> None:
    """Keep the opt-in halt, evidence boundary and recorded numbers explicit."""
    doc = _normalise_whitespace(_read_doc("docs-site/guide/streaming.md"))

    assert (
        "With contradiction halt enabled, Director-AI scores completed claims "
        "for `P(contradiction)` against retrieved grounding facts." in doc
    )
    assert (
        "The stream stops when that probability reaches the configured threshold."
        in doc
    )
    assert "when retrieval returns no usable facts" in doc
    assert "An exception raised by retrieval or scoring interrupts" in doc
    assert "Previously reviewed and released windows remain with the client." in doc


def test_public_streaming_counts_follow_archived_benchmark() -> None:
    """Tie public rates and operating point to archived passage decisions."""
    archive = "benchmarks/results/streaming_contradiction_halt_base.json"
    report = json.loads(_read_doc(archive))
    command = report["benchmark_context"]["command"]
    threshold = command[command.index("--threshold") + 1]
    good = report["per_good"]
    bad = report["per_bad"]
    false_halts = sum(row["halted"] for row in good)
    caught = sum(row["halted"] for row in bad)
    summary = (
        f"measured {false_halts}/{len(good)} false halts "
        f"and caught {caught}/{len(bad)} contradictions"
    )
    readme = _normalise_whitespace(_read_doc("README.md"))
    why = _normalise_whitespace(_read_doc("docs-site/guide/why-director-ai.md"))
    guide = _normalise_whitespace(_read_doc("docs-site/guide/streaming.md"))
    assert f"benchmark at threshold {threshold} {summary}" in readme
    assert f"benchmark at threshold {threshold} {summary}" in why
    assert f"{summary} at threshold {threshold}" in guide
    assert archive in guide
    assert (
        f"| Grounded | {false_halts} false halts | {len(good)} | "
        f"{false_halts / len(good):.4f} |" in guide
    )
    assert (
        f"| All contradictions | {len(bad) - caught} misses | "
        f"{len(bad)} | {(len(bad) - caught) / len(bad):.4f} |" in guide
    )
    for kind in ("semantic", "numeric"):
        rows = [row for row in bad if row["kind"] == kind]
        misses = sum(not row["halted"] for row in rows)
        assert (
            f"| {kind.capitalize()} contradiction | {misses} misses | "
            f"{len(rows)} | {misses / len(rows):.4f} |" in guide
        )


def test_market_docs_keep_unsupported_vs_contradiction_boundary() -> None:
    """Keep public claims explicit about contradiction and missing evidence."""
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
