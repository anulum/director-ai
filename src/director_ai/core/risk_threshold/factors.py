# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Risk factors for adaptive thresholding

"""The per-request risk profile that drives the adaptive threshold.

Each field is one documented input to the threshold. The continuous fields are
in ``[0, 1]`` with a fixed orientation (``1`` = the safe end) so the adapter can
map them to threshold deltas with a single sign convention. Categorical fields
(role, domain) are looked up in the policy's maps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["RiskFactors"]


def _check_unit(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")


@dataclass(frozen=True)
class RiskFactors:
    """The risk profile of one request.

    Parameters
    ----------
    user_role:
        The caller's role; looked up in the policy's role map (unknown roles
        contribute nothing).
    tenant_risk:
        The tenant's risk level in ``[0, 1]`` (``0`` = low-risk tenant).
    domain:
        The request domain (e.g. ``"medical"``); looked up in the policy's
        domain map.
    retrieval_confidence:
        Confidence of the supporting retrieval in ``[0, 1]`` (``1`` = strong
        match). Low confidence raises the threshold.
    action_reversibility:
        Reversibility of the action the answer drives in ``[0, 1]`` (``1`` =
        fully reversible). Low reversibility raises the threshold.
    external_exposure:
        Whether the answer leaves the organisation (shown to an end customer,
        published). Raises the threshold.
    pii_present:
        Whether the request or answer involves personal data. Raises the
        threshold.
    freshness:
        Freshness of the supporting evidence in ``[0, 1]`` (``1`` = current).
        Stale evidence raises the threshold.
    historical_fpr:
        The guard's historical false-positive (false-halt) rate for this slice
        in ``[0, 1]``. A high rate *lowers* the threshold to cut over-blocking.
    """

    user_role: str = ""
    tenant_risk: float = 0.0
    domain: str = ""
    retrieval_confidence: float = 1.0
    action_reversibility: float = 1.0
    external_exposure: bool = False
    pii_present: bool = False
    freshness: float = 1.0
    historical_fpr: float = 0.0

    def __post_init__(self) -> None:
        _check_unit("tenant_risk", self.tenant_risk)
        _check_unit("retrieval_confidence", self.retrieval_confidence)
        _check_unit("action_reversibility", self.action_reversibility)
        _check_unit("freshness", self.freshness)
        _check_unit("historical_fpr", self.historical_fpr)
