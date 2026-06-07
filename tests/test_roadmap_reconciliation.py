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


def test_roadmap_status_lists_only_current_open_actionable_items() -> None:
    roadmap = _read("ROADMAP.md")
    status = _read("docs/ROADMAP_STATUS.md")

    assert (
        "Current open-item reconciliation lives in `docs/ROADMAP_STATUS.md`" in roadmap
    )
    assert "| R1 | Independent external security test" in status
    # R1-R7 are the operational public status items that stay public.
    for rid in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]:
        assert f"| {rid} |" in status
    # R8-R17 (the competitive differentiator queue, each with its per-item
    # "next evidence needed" forward plan) and the "Future And Strategic Items"
    # table were moved to the internal TODO on 2026-06-07 to avoid premature
    # disclosure of forward plans. They must not reappear in the public status.
    for rid in ["R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15", "R16", "R17"]:
        assert f"| {rid} |" not in status
    assert "tracked internally" in status
    assert "docs/internal/TODO_CONSOLIDATED.md" in status
    assert "Older unchecked internal checklists" in status


def test_public_roadmap_has_single_unchecked_item_after_reconciliation() -> None:
    roadmap = _read("ROADMAP.md")
    unchecked = [line for line in roadmap.splitlines() if line.startswith("- [ ] ")]

    assert unchecked == [
        "- [ ] Run an external security test focused on streaming paths and tenant isolation"
    ]
