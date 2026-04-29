# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - physical risk documentation tests

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_security_policy_documents_physical_action_residual_risks() -> None:
    text = (ROOT / "SECURITY.md").read_text(encoding="utf-8")

    for phrase in [
        "Physical-action residual risks",
        "Hardware damage",
        "Malformed action payloads",
        "Expensive solver payloads",
        "Simulator dependency isolation",
        "TenantPhysicalBudget",
        "director-ai[physical]",
    ]:
        assert phrase in text
