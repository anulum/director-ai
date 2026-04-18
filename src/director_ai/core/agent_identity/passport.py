# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AgentPassport + PassportSigner

"""Signed agent identity records.

The signer derives an HMAC-SHA-256 tag over a canonical JSON
serialisation of the passport fields. Verification uses
:func:`hmac.compare_digest` for constant-time comparison against
forgery attempts. Secrets must be at least 32 bytes — short
secrets are rejected at signer construction.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace


class PassportVerificationError(ValueError):
    """Signature mismatch, expired passport, or unknown key id."""


@dataclass(frozen=True)
class AgentPassport:
    """Immutable signed identity.

    Parameters
    ----------
    agent_id :
        Globally unique ID. Conventionally ``tenant/role/name``
        but the signer treats it as an opaque string.
    role :
        Non-empty role label. Mirrors
        :class:`~director_ai.core.knowledge_graph.Principal.role`.
    tenant_id :
        Empty means "cross-tenant"; non-empty scopes the passport
        to one tenant.
    capabilities :
        Sorted tuple of capability tokens. Sorting at construction
        keeps the canonical JSON stable across platforms.
    issued_at :
        POSIX timestamp when the passport was issued.
    expires_at :
        POSIX timestamp of expiry. ``0.0`` means "no expiry" — use
        sparingly, default signers set a 24 h lifetime.
    key_id :
        Caller-chosen key identifier that lets the signer rotate
        keys without invalidating every outstanding passport.
    signature :
        Base-16 HMAC-SHA-256 tag. Empty until the signer attaches it.
    """

    agent_id: str
    role: str
    tenant_id: str
    capabilities: tuple[str, ...]
    issued_at: float
    expires_at: float
    key_id: str
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id must be non-empty")
        if not self.role:
            raise ValueError("role must be non-empty")
        if not self.key_id:
            raise ValueError("key_id must be non-empty")
        if self.issued_at < 0.0:
            raise ValueError("issued_at must be non-negative")
        if self.expires_at < 0.0:
            raise ValueError("expires_at must be non-negative")
        if self.expires_at and self.expires_at < self.issued_at:
            raise ValueError("expires_at must be >= issued_at")

    def canonical(self) -> bytes:
        """Deterministic JSON-encoded bytes used as the HMAC
        payload. Excludes the signature itself and uses sorted
        keys + compact separators so bit-for-bit reproducibility
        is guaranteed across Python versions and platforms."""
        data = asdict(self)
        data.pop("signature", None)
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


class PassportSigner:
    """Issue and verify :class:`AgentPassport` records.

    The signer holds one active key and optional inactive keys for
    verification-only during rotation. ``default_ttl_seconds`` is
    the lifetime attached to passports issued without an explicit
    expiry (default 24 h).

    Parameters
    ----------
    active_key :
        Bytes used to sign new passports. Must be at least 32
        bytes — shorter keys are rejected.
    active_key_id :
        Identifier stored inside the passport so rotation can
        still verify old passports after ``active_key`` changes.
    inactive_keys :
        Optional mapping of ``key_id → key_bytes`` used for
        verification only. Callers rotate by promoting an inactive
        key to active and moving the old active into inactive.
    default_ttl_seconds :
        Lifetime for passports issued via :meth:`issue` without
        an explicit ``expires_at``. Default 24 h.
    clock :
        Callable returning the current POSIX timestamp. Injection
        point for tests.
    """

    def __init__(
        self,
        *,
        active_key: bytes,
        active_key_id: str,
        inactive_keys: dict[str, bytes] | None = None,
        default_ttl_seconds: float = 24 * 60 * 60.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if len(active_key) < 32:
            raise ValueError(
                f"active_key must be at least 32 bytes; got {len(active_key)}"
            )
        if not active_key_id:
            raise ValueError("active_key_id must be non-empty")
        if default_ttl_seconds < 0:
            raise ValueError("default_ttl_seconds must be non-negative")
        self._active_key = active_key
        self._active_key_id = active_key_id
        self._inactive_keys = dict(inactive_keys or {})
        self._default_ttl = default_ttl_seconds
        self._clock = clock or time.time

    def issue(
        self,
        *,
        agent_id: str,
        role: str,
        tenant_id: str = "",
        capabilities: tuple[str, ...] = (),
        ttl_seconds: float | None = None,
    ) -> AgentPassport:
        """Mint a fresh passport signed under the active key."""
        now = float(self._clock())
        ttl = self._default_ttl if ttl_seconds is None else ttl_seconds
        if ttl < 0:
            raise ValueError("ttl_seconds must be non-negative")
        expires_at = now + ttl if ttl > 0 else 0.0
        partial = AgentPassport(
            agent_id=agent_id,
            role=role,
            tenant_id=tenant_id,
            capabilities=tuple(sorted(capabilities)),
            issued_at=now,
            expires_at=expires_at,
            key_id=self._active_key_id,
            signature="",
        )
        signature = self._compute_signature(self._active_key, partial)
        return replace(partial, signature=signature)

    def verify(self, passport: AgentPassport) -> None:
        """Raise :class:`PassportVerificationError` when the
        passport is expired, carries an unknown key id, or fails
        signature comparison. Returns ``None`` on success so the
        caller's ``try``/``except`` is the branching path."""
        if not passport.signature:
            raise PassportVerificationError("passport has no signature")
        key = self._resolve_key(passport.key_id)
        expected = self._compute_signature(key, passport)
        if not hmac.compare_digest(expected, passport.signature):
            raise PassportVerificationError("signature mismatch")
        if passport.expires_at and passport.expires_at < float(self._clock()):
            raise PassportVerificationError(
                f"passport expired at {passport.expires_at:.0f}"
            )

    def is_valid(self, passport: AgentPassport) -> bool:
        """Boolean wrapper for :meth:`verify`. Useful in
        expression contexts; prefer :meth:`verify` when the
        caller needs the failure reason."""
        try:
            self.verify(passport)
        except PassportVerificationError:
            return False
        return True

    def rotate(self, *, new_active_key: bytes, new_active_key_id: str) -> None:
        """Promote a new key to active and move the previous one
        into the inactive set so outstanding passports signed
        under it still verify."""
        if len(new_active_key) < 32:
            raise ValueError("new_active_key must be at least 32 bytes")
        if not new_active_key_id:
            raise ValueError("new_active_key_id must be non-empty")
        self._inactive_keys[self._active_key_id] = self._active_key
        self._active_key = new_active_key
        self._active_key_id = new_active_key_id

    def _resolve_key(self, key_id: str) -> bytes:
        if key_id == self._active_key_id:
            return self._active_key
        if key_id in self._inactive_keys:
            return self._inactive_keys[key_id]
        raise PassportVerificationError(f"unknown key id {key_id!r}")

    @staticmethod
    def _compute_signature(key: bytes, passport: AgentPassport) -> str:
        tag = hmac.new(key, passport.canonical(), hashlib.sha256).hexdigest()
        return tag

    @property
    def active_key_id(self) -> str:
        return self._active_key_id
