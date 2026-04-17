# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — federated privacy-preserving sharing

"""Differential privacy + additive secret sharing for
multi-tenant failure-pattern aggregation.

Three layers:

* :class:`LaplaceMechanism` — adds calibrated Laplace noise to a
  numeric query so the released value obeys an ``ε``-differential
  privacy guarantee against the caller's declared
  ``sensitivity``. Also ships a :class:`GaussianMechanism` for
  ``(ε, δ)``-DP.
* :class:`SecretShare` + :class:`SecureAggregator` — additive
  secret sharing over a prime modulus. Each party splits its
  value into ``n`` random shares that sum (modulo ``p``) back
  to the original; the aggregator sums the per-party shares
  componentwise and the reconstruction gives the multi-party
  total without any single party seeing another's contribution.
* :class:`PrivacyAccountant` — tracks the cumulative ``(ε, δ)``
  budget across queries. Simple composition is the default; an
  ``advanced=True`` mode applies the Dwork-Rothblum-Vadhan bound
  for a tighter budget at larger query counts.

High-level helpers:

* :class:`FederatedCounter` — DP-noised count aggregation across
  tenants with a per-tenant ε draw.
* :class:`FederatedHistogram` — DP-noised histogram over a fixed
  category set, composing counts via the Laplace mechanism and
  the accountant.
"""

from .accountant import AccountantEntry, PrivacyAccountant
from .aggregator import FederatedCounter, FederatedHistogram
from .mechanisms import GaussianMechanism, LaplaceMechanism
from .secret_sharing import SecretShare, SecureAggregator, ShareError

__all__ = [
    "AccountantEntry",
    "FederatedCounter",
    "FederatedHistogram",
    "GaussianMechanism",
    "LaplaceMechanism",
    "PrivacyAccountant",
    "SecretShare",
    "SecureAggregator",
    "ShareError",
]
