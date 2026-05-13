# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Roadmap Reconciliation Tests
"""Regression checks that completed roadmap surfaces stay documented."""

from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_self_improving_guard_loop_roadmap_has_public_api_docs() -> None:
    roadmap = _read("ROADMAP.md")
    api_doc = _read("docs-site/api/self-improving-guard-loop.md")
    mkdocs = _read("mkdocs.yml")

    assert "[x] Design a human-reviewed self-improving guard loop" in roadmap
    assert "Self-Improving Guard Loop: api/self-improving-guard-loop.md" in mkdocs
    assert "reviewer_id" in api_doc
    assert "LoRA jobs are proposal-only" in api_doc
    assert "It does not mutate runtime configuration." in api_doc
