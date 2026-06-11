# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Evidence firewall checks

"""The eight admission checks, each a pure function over one chunk.

A check returns a :class:`CheckOutcome` when the policy enables it, or ``None``
when the policy switches it off (the firewall then records nothing for it). Every
failure ``reason`` is a stable code from this module's vocabulary so it can be
counted and audited without exposing chunk text.
"""

from __future__ import annotations

from collections.abc import Callable

from .chunk import RetrievedChunk
from .policy import CheckOutcome, FirewallContext, FirewallPolicy

__all__ = [
    "check_allowed_use_case",
    "check_content_hash",
    "check_expiry",
    "check_max_age",
    "check_poisoning",
    "check_provenance_present",
    "check_sensitivity",
    "check_signature_verified",
    "check_source_owner",
    "check_tenant_authorisation",
]

_TENANT = "tenant_authorisation"
_PROVENANCE = "provenance_present"
_SIGNATURE = "signature_verified"
_CONTENT_HASH = "content_hash_match"
_EXPIRY = "freshness_expiry"
_MAX_AGE = "freshness_age"
_SOURCE_OWNER = "source_owner_known"
_SENSITIVITY = "sensitivity_allowed"
_POISON = "poisoning_scan"
_USE_CASE = "allowed_use_case"


def check_tenant_authorisation(
    chunk: RetrievedChunk,
    policy: FirewallPolicy,
    context: FirewallContext,
) -> CheckOutcome | None:
    """Fail when the chunk is owned by a tenant the request may not read."""
    if not policy.require_tenant_match:
        return None
    if context.tenant_allowed(chunk.tenant_id):
        return CheckOutcome(_TENANT, passed=True)
    return CheckOutcome(_TENANT, passed=False, reason="tenant_mismatch")


def check_provenance_present(
    chunk: RetrievedChunk,
    policy: FirewallPolicy,
    context: FirewallContext,
) -> CheckOutcome | None:
    """Fail when the chunk carries no provenance marker at all."""
    if not policy.require_provenance:
        return None
    if chunk.has_provenance:
        return CheckOutcome(_PROVENANCE, passed=True)
    return CheckOutcome(_PROVENANCE, passed=False, reason="provenance_missing")


def check_signature_verified(
    chunk: RetrievedChunk,
    policy: FirewallPolicy,
    context: FirewallContext,
) -> CheckOutcome | None:
    """Fail when the chunk's write was not signature-verified."""
    if not policy.require_signature:
        return None
    if chunk.signature_verified:
        return CheckOutcome(_SIGNATURE, passed=True)
    return CheckOutcome(_SIGNATURE, passed=False, reason="signature_unverified")


def check_content_hash(
    chunk: RetrievedChunk,
    policy: FirewallPolicy,
    context: FirewallContext,
) -> CheckOutcome | None:
    """Fail when a recorded text digest does not match the current text.

    Passes (rather than fails) when no text digest is recorded — absence is
    handled by :func:`check_provenance_present`, not here; this check only
    catches tampering between write and read.
    """
    if not policy.verify_content_hash:
        return None
    recorded = chunk.recorded_text_digest
    if not recorded:
        return CheckOutcome(_CONTENT_HASH, passed=True)
    if recorded.lower() == chunk.computed_text_digest():
        return CheckOutcome(_CONTENT_HASH, passed=True)
    return CheckOutcome(_CONTENT_HASH, passed=False, reason="content_hash_mismatch")


def check_expiry(
    chunk: RetrievedChunk,
    policy: FirewallPolicy,
    context: FirewallContext,
) -> CheckOutcome | None:
    """Fail when the chunk's recorded expiry has passed."""
    if not policy.enforce_expiry or context.now_unix <= 0.0:
        return None
    expires = chunk.expires_at_unix
    if expires is None or expires > context.now_unix:
        return CheckOutcome(_EXPIRY, passed=True)
    return CheckOutcome(_EXPIRY, passed=False, reason="expired")


def check_max_age(
    chunk: RetrievedChunk,
    policy: FirewallPolicy,
    context: FirewallContext,
) -> CheckOutcome | None:
    """Fail when the chunk is older than the policy's max age."""
    if policy.max_age_seconds <= 0.0 or context.now_unix <= 0.0:
        return None
    created = chunk.created_at_unix
    if created is None:
        return CheckOutcome(_MAX_AGE, passed=True)
    if context.now_unix - created <= policy.max_age_seconds:
        return CheckOutcome(_MAX_AGE, passed=True)
    return CheckOutcome(_MAX_AGE, passed=False, reason="too_old")


def check_source_owner(
    chunk: RetrievedChunk,
    policy: FirewallPolicy,
    context: FirewallContext,
) -> CheckOutcome | None:
    """Fail when no source owner/key is recorded for the chunk."""
    if not policy.require_source_owner:
        return None
    if chunk.source_owner:
        return CheckOutcome(_SOURCE_OWNER, passed=True)
    return CheckOutcome(_SOURCE_OWNER, passed=False, reason="source_owner_unknown")


def check_sensitivity(
    chunk: RetrievedChunk,
    policy: FirewallPolicy,
    context: FirewallContext,
) -> CheckOutcome | None:
    """Fail when the chunk's sensitivity label is not admissible."""
    if not policy.enforce_sensitivity:
        return None
    if chunk.sensitivity in policy.allowed_sensitivity:
        return CheckOutcome(_SENSITIVITY, passed=True)
    return CheckOutcome(_SENSITIVITY, passed=False, reason="sensitivity_blocked")


def check_allowed_use_case(
    chunk: RetrievedChunk,
    policy: FirewallPolicy,
    context: FirewallContext,
) -> CheckOutcome | None:
    """Fail when the chunk's use-case list excludes the request use case.

    A chunk with an empty allowed-use-case list is unrestricted and passes.
    """
    if not policy.enforce_use_case:
        return None
    allowed = chunk.allowed_use_cases
    if not allowed:
        return CheckOutcome(_USE_CASE, passed=True)
    if context.use_case.strip().lower() in allowed:
        return CheckOutcome(_USE_CASE, passed=True)
    return CheckOutcome(_USE_CASE, passed=False, reason="use_case_not_allowed")


def check_poisoning(
    chunk: RetrievedChunk,
    policy: FirewallPolicy,
    context: FirewallContext,
    *,
    scan: Callable[[str], float],
) -> CheckOutcome | None:
    """Fail when the chunk's poisoning score meets the policy threshold."""
    if not policy.scan_poisoning:
        return None
    score = scan(chunk.text)
    if score < policy.poison_threshold:
        return CheckOutcome(_POISON, passed=True)
    return CheckOutcome(_POISON, passed=False, reason="poisoning_detected")
