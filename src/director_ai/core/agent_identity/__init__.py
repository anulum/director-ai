# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — persistent agent identity + behavioral provenance

"""Cryptographically signed agent passports plus a behavioural
fingerprint so the guardrail can detect identity hijacking.

* :class:`AgentPassport` — immutable identity record with role,
  tenant, capabilities, issued-at / expires-at timestamps, and
  an HMAC-SHA-256 signature over a canonical JSON serialisation.
* :class:`PassportSigner` — issues and verifies passports with a
  caller-supplied secret. Constant-time signature comparison via
  :func:`hmac.compare_digest`.
* :class:`BehavioralFingerprint` — rolling statistics over prompt
  length, tool-call counts, token rates, time-of-day. Welford's
  algorithm for mean / variance so long-running agents do not
  lose numerical precision.
* :class:`IdentityMonitor` — checks a new observation against the
  rolling fingerprint and raises when the z-score across any
  tracked feature exceeds a caller-supplied threshold. Hooks
  into the existing HMAC audit chain via :class:`AuditChain`.
* :class:`AuditChain` — hash-chained event log. Each entry
  carries a parent-hash + event-hash + HMAC-SHA-256 tag so the
  chain is tamper-evident offline.
"""

from .audit import AuditChain, AuditEntry
from .fingerprint import BehavioralFingerprint, BehaviorObservation, IdentityMonitor
from .passport import AgentPassport, PassportSigner, PassportVerificationError

__all__ = [
    "AgentPassport",
    "AuditChain",
    "AuditEntry",
    "BehaviorObservation",
    "BehavioralFingerprint",
    "IdentityMonitor",
    "PassportSigner",
    "PassportVerificationError",
]
