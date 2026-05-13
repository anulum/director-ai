# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Experimental Deployment Documentation Tests
"""Regression tests for deployment warnings around experimental hook surfaces."""

from __future__ import annotations

from pathlib import Path


def test_experimental_hooks_document_isolation_and_human_review() -> None:
    guide = Path("docs-site/deployment/production.md").read_text(encoding="utf-8")
    required_terms = (
        "meta_guard",
        "self_evolving",
        "continual_adversarial",
        "isolation",
        "human review",
        "live deployments",
    )

    missing = [term for term in required_terms if term not in guide]

    assert missing == []
