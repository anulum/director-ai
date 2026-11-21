# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Graders

"""Grader specifications and their roles.

A grader may train, select or score sealed looks. A grader that is not
exact and deterministic needs a calibration report digest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._checks import (
    require_bool,
    require_choice,
    require_sha256,
    require_text,
    require_tuple,
    require_unique,
)
from .errors import EvaluationContractError

GraderRole = Literal["training", "selection", "sealed"]


GRADER_ROLES: frozenset[str] = frozenset({"training", "selection", "sealed"})


@dataclass(frozen=True, slots=True)
class GraderSpec:
    """One grader and the roles it may play.

    Attributes
    ----------
    grader_id : str
        Grader identifier.
    implementation_digest : str
        SHA-256 of the grader's implementation.
    roles : tuple[GraderRole, ...]
        Roles among ``training``, ``selection`` and ``sealed``.
    is_exact_deterministic : bool
        Whether the grader is exact and deterministic.
    calibration_report_digest : str or None
        SHA-256 of its calibration report; required when not exact.
    """

    grader_id: str
    implementation_digest: str
    roles: tuple[GraderRole, ...]
    is_exact_deterministic: bool
    calibration_report_digest: str | None


def validate_grader(grader: GraderSpec) -> GraderSpec:
    """Check one grader's fields and its calibration requirement.

    Parameters
    ----------
    grader : GraderSpec
        The grader to check.

    Returns
    -------
    GraderSpec
        The same grader.

    Raises
    ------
    EvaluationContractError
        If a field is malformed, or a non-exact grader has no calibration
        report digest.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(grader, GraderSpec):
        raise EvaluationContractError("a GraderSpec is required")
    require_text(grader.grader_id, "grader_id", EvaluationContractError)
    require_sha256(grader.implementation_digest, "implementation_digest")
    roles = require_tuple(grader.roles, "roles", EvaluationContractError)
    if not roles:
        raise EvaluationContractError(
            f"grader {grader.grader_id!r} needs at least one role"
        )
    for role in roles:
        require_choice(role, "roles", GRADER_ROLES, EvaluationContractError)
    require_unique(roles, "roles", EvaluationContractError)
    exact = require_bool(
        grader.is_exact_deterministic, "is_exact_deterministic", EvaluationContractError
    )
    if not exact or grader.calibration_report_digest is not None:
        require_sha256(grader.calibration_report_digest, "calibration_report_digest")
    return grader
