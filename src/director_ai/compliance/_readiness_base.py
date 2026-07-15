# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — compliance readiness shared base

"""Shared readiness vocabulary for the compliance readiness deliverables.

Split out of :mod:`director_ai.compliance.readiness` so the SOC 2 / ISO report
(:mod:`director_ai.compliance._readiness_soc2_iso`) and the HIPAA documentation
packet (:mod:`director_ai.compliance._readiness_hipaa`) share one status enum,
one risk grader, and the HIPAA reference prefix without importing each other.
"""

from __future__ import annotations

from enum import StrEnum


class ReadinessStatus(StrEnum):
    """Readiness status for a control row."""

    # Readiness status label, not a password or credential.
    PASS = "passed"  # nosec B105
    WARNING = "warning"
    FAIL = "failing"
    NOT_APPLICABLE = "not_applicable"


_HIPAA_REF_PREFIX = "45 CFR 164."


def _risk_level(*, failures: int, warnings: int) -> str:
    if failures:
        return "critical"
    if warnings:
        return "attention_required"
    return "ready"
