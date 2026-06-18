# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Evidence firewall policy

"""Policy, request context, and per-check outcome types for the evidence firewall.

The firewall runs on retrieved chunks *before* they reach the model. A
:class:`FirewallPolicy` says which of the eight admission checks are enforced
and with what bounds; a :class:`FirewallContext` carries the per-request facts a
check needs (calling tenant, declared use case, wall-clock now). Each check
yields a :class:`CheckOutcome` whose ``reason`` is a stable, tenant-safe code —
never raw chunk text — so the outcome can be logged and shipped to a customer
audit trail without leaking another tenant's data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "CheckOutcome",
    "FirewallContext",
    "FirewallPolicy",
]


@dataclass(frozen=True)
class CheckOutcome:
    """Result of one admission check on one chunk.

    Parameters
    ----------
    name:
        Stable check identifier, e.g. ``"tenant_authorisation"``.
    passed:
        ``True`` when the chunk satisfied the check.
    reason:
        Tenant-safe failure code when ``passed`` is ``False`` (e.g.
        ``"tenant_mismatch"``); empty string when the check passed. The code is
        drawn from a closed vocabulary so dashboards can aggregate it.
    """

    name: str
    passed: bool
    reason: str = ""

    def __post_init__(self) -> None:
        """Require a check name and forbid a reason on a passing outcome."""
        if not self.name.strip():
            raise ValueError("check name is required")
        if self.passed and self.reason:
            raise ValueError("a passing outcome must not carry a failure reason")
        if not self.passed and not self.reason.strip():
            raise ValueError("a failing outcome must carry a reason code")


@dataclass(frozen=True)
class FirewallContext:
    """Per-request facts the admission checks read.

    Parameters
    ----------
    tenant_id:
        The tenant the answer is being produced for. A chunk is tenant-authorised
        only when its own ``tenant_id`` is empty (shared corpus) or appears in
        ``authorised_tenants``.
    use_case:
        The declared use case of the request (e.g. ``"support"``,
        ``"underwriting"``). Matched against each chunk's allowed-use-case list
        when use-case enforcement is on.
    now_unix:
        Wall-clock seconds since the epoch, supplied by the caller so freshness
        checks stay deterministic and testable. ``0.0`` disables age/expiry
        evaluation regardless of policy.
    authorised_tenants:
        Tenants whose chunks this request may read. Defaults to ``{tenant_id}``
        when left empty.
    """

    tenant_id: str
    use_case: str = ""
    now_unix: float = 0.0
    authorised_tenants: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Reject a negative timestamp and default the authorised-tenant set."""
        if self.now_unix < 0.0:
            raise ValueError("now_unix must be non-negative")
        if not self.authorised_tenants:
            object.__setattr__(self, "authorised_tenants", frozenset({self.tenant_id}))
        else:
            object.__setattr__(
                self,
                "authorised_tenants",
                frozenset(self.authorised_tenants) | {self.tenant_id},
            )

    def tenant_allowed(self, chunk_tenant: str) -> bool:
        """Return whether a chunk owned by ``chunk_tenant`` may be read.

        An empty ``chunk_tenant`` denotes a shared, non-tenant corpus and is
        always allowed; otherwise the owner must be in ``authorised_tenants``.
        """
        if not chunk_tenant:
            return True
        return chunk_tenant in self.authorised_tenants


@dataclass(frozen=True)
class FirewallPolicy:
    """Which admission checks the firewall enforces, and their bounds.

    Defaults are fail-closed on the integrity-critical checks (tenant,
    provenance, signature, expiry) and opt-in on the corpus-shape checks
    (sensitivity labels, declared use case) that depend on a customer taxonomy.

    Parameters
    ----------
    require_tenant_match:
        Drop a chunk owned by a tenant the request is not authorised for.
    require_provenance:
        Drop a chunk that carries no provenance metadata at all (no content
        hash, version, or signature marker).
    require_signature:
        Drop a chunk whose write was not signature-verified
        (``kb_signature_verified`` is not ``True``).
    verify_content_hash:
        When a chunk records a digest *of its own text*
        (``text_sha256``/``content_sha256``), recompute and require equality.
    enforce_expiry:
        Drop a chunk whose ``expires_at`` has passed.
    max_age_seconds:
        When > 0, drop a chunk older than this many seconds relative to
        ``now_unix`` (uses the chunk ``created_at``). ``0`` disables the age
        bound.
    require_source_owner:
        Drop a chunk with no recorded source owner/key.
    enforce_sensitivity:
        Drop a chunk whose sensitivity label is not in ``allowed_sensitivity``.
    allowed_sensitivity:
        Admissible sensitivity labels when ``enforce_sensitivity`` is on. A
        chunk with no label is treated as ``"unclassified"``.
    scan_poisoning:
        Run the poisoning/sentinel scan over chunk text and drop a chunk whose
        score meets ``poison_threshold``.
    poison_threshold:
        Score in ``[0, 1]`` at or above which a chunk is treated as poisoned.
    enforce_use_case:
        Drop a chunk whose allowed-use-case list excludes the request's
        ``use_case``. A chunk with no list is treated as unrestricted.
    """

    require_tenant_match: bool = True
    require_provenance: bool = True
    require_signature: bool = True
    verify_content_hash: bool = True
    enforce_expiry: bool = True
    max_age_seconds: float = 0.0
    require_source_owner: bool = False
    enforce_sensitivity: bool = False
    allowed_sensitivity: frozenset[str] = field(
        default_factory=lambda: frozenset({"unclassified", "public", "internal"})
    )
    scan_poisoning: bool = True
    poison_threshold: float = 0.6
    enforce_use_case: bool = False

    def __post_init__(self) -> None:
        """Range-check the maximum age bound and the poison threshold."""
        if self.max_age_seconds < 0.0:
            raise ValueError("max_age_seconds must be non-negative")
        if not 0.0 <= self.poison_threshold <= 1.0:
            raise ValueError("poison_threshold must be in [0, 1]")
        object.__setattr__(
            self,
            "allowed_sensitivity",
            frozenset(label.strip().lower() for label in self.allowed_sensitivity),
        )

    @classmethod
    def permissive(cls) -> FirewallPolicy:
        """Return a policy that admits everything (checks off).

        Useful as an explicit, named "firewall disabled" posture for non-tenant
        development corpora; production should never use it.
        """
        return cls(
            require_tenant_match=False,
            require_provenance=False,
            require_signature=False,
            verify_content_hash=False,
            enforce_expiry=False,
            require_source_owner=False,
            enforce_sensitivity=False,
            scan_poisoning=False,
            enforce_use_case=False,
        )
