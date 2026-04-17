# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — simulation containment + reality anchor

"""Proofs of which execution layer an agent is running in.

An agent that has graduated from a sandbox to production touches
different systems (real billing, real vehicles, real medical
devices). When the host supervisor tells the agent ``you are
in sandbox``, a compromised or misaligned model may ignore the
claim and issue production-grade commands anyway. Simulation
containment puts a cryptographic anchor on every session so the
gateway can refuse actions whose scope does not match the anchor,
and an event stream monitor can raise a breakout alarm the moment
the agent attempts to cross the boundary.

Two primitives:

* :class:`RealityAnchor` — HMAC-signed attestation of
  ``(session_id, scope, issuer, created_at)``; verifier rejects
  anchors whose MAC does not round-trip and anchors whose
  ``scope`` disagrees with the live host's ground truth.
* :class:`BreakoutDetector` — analyses an event stream for the
  three breakout patterns we have seen in practice: production
  API hostname in a sandbox context, prompt-injection patterns
  that ask the model to ignore the anchor, anchor/claim scope
  mismatch.

:class:`ContainmentGuard` composes the two: call ``check(event,
anchor)`` before every outbound action and consult the verdict.
"""

from __future__ import annotations

from .anchor import AnchorVerification, ContainmentAttestor, RealityAnchor
from .detector import BreakoutDetector, BreakoutFinding
from .guard import ContainmentGuard, ContainmentVerdict
from .scope import ContainmentScope, scope_allows_real_effects

__all__ = [
    "AnchorVerification",
    "BreakoutDetector",
    "BreakoutFinding",
    "ContainmentAttestor",
    "ContainmentGuard",
    "ContainmentScope",
    "ContainmentVerdict",
    "RealityAnchor",
    "scope_allows_real_effects",
]
