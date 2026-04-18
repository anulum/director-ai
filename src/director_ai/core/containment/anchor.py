# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — reality anchor (HMAC-signed scope attestation)

"""HMAC-SHA256 attestation binding a session to a containment scope.

The host supervisor (the only party that holds the secret key)
signs a :class:`RealityAnchor` at session-open time. The anchor
accompanies every outbound action proposal. The gateway verifies
the anchor's MAC and the anchor's ``scope`` against the
supervisor's live record before dispatching the action.

The anchor is deliberately small and binary-stable: canonicalising
the payload as ``session_id|scope|issuer|created_at|nonce`` keeps
verification free of JSON encoder drift.

Keys must be at least 32 bytes — shorter keys are rejected on
construction to avoid silent downgrade attacks.
"""

from __future__ import annotations

import hmac
import secrets
import time
from dataclasses import dataclass, field
from hashlib import sha256

from .scope import ContainmentScope, validate_scope

_MIN_KEY_LEN = 32
_MAC_LEN_HEX = 64  # sha256 digest in hex


@dataclass(frozen=True)
class RealityAnchor:
    """Signed claim that *session_id* runs under *scope*.

    Attributes
    ----------
    session_id : str
        Opaque identifier minted by the host for this agent run.
    scope : ContainmentScope
        One of ``sandbox`` / ``simulator`` / ``shadow`` / ``production``.
    issuer : str
        Host supervisor identifier (``"host://edge-11"`` is typical).
    created_at : int
        Unix epoch seconds at issue time. Used for anchor freshness.
    nonce : str
        128-bit hex nonce bound into the MAC; prevents replay of
        anchors between sessions sharing the same scope.
    mac : str
        Lowercase hex HMAC-SHA256 over the canonical payload.
    """

    session_id: str
    scope: ContainmentScope
    issuer: str
    created_at: int
    nonce: str
    mac: str

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        if not self.issuer:
            raise ValueError("issuer must be non-empty")
        validate_scope(self.scope)
        if self.created_at < 0:
            raise ValueError("created_at must be non-negative unix epoch")
        if len(self.nonce) != 32:
            raise ValueError("nonce must be a 32-char hex string (128 bits)")
        if len(self.mac) != _MAC_LEN_HEX:
            raise ValueError(
                f"mac must be {_MAC_LEN_HEX}-char hex (sha256); got {len(self.mac)}"
            )

    @property
    def canonical_payload(self) -> bytes:
        """Byte string signed by the MAC. Exposed for diagnostics."""
        return _canonicalise(
            self.session_id,
            self.scope,
            self.issuer,
            self.created_at,
            self.nonce,
        )


@dataclass(frozen=True)
class AnchorVerification:
    """Outcome of :meth:`ContainmentAttestor.verify`.

    ``valid`` is the single boolean answer; ``reason`` is filled on
    rejection so the caller can log the failure mode without
    branching on string equality.
    """

    valid: bool
    reason: str = ""


@dataclass
class ContainmentAttestor:
    """Mints and verifies :class:`RealityAnchor` instances.

    One attestor per host. Callers construct with a symmetric
    secret at least :data:`_MIN_KEY_LEN` bytes long (typically
    loaded from the host secret store) and an optional
    ``max_age_seconds`` for anchor freshness. Anchor MAC uses
    HMAC-SHA256 with constant-time comparison on verify.

    Parameters
    ----------
    key : bytes
        Shared HMAC secret. Rejected if shorter than 32 bytes.
    issuer : str
        Default issuer stamped on minted anchors.
    max_age_seconds : int
        Anchors older than this are rejected with
        ``reason="expired"``. 0 disables the freshness check.
    clock : callable, optional
        Returns current unix seconds. Defaults to
        :func:`time.time`; tests inject a deterministic source.
    """

    key: bytes
    issuer: str
    max_age_seconds: int = 3600
    clock: object = field(default=time.time)

    def __post_init__(self) -> None:
        if len(self.key) < _MIN_KEY_LEN:
            raise ValueError(
                f"HMAC key must be at least {_MIN_KEY_LEN} bytes (got {len(self.key)})"
            )
        if not self.issuer:
            raise ValueError("issuer must be non-empty")
        if self.max_age_seconds < 0:
            raise ValueError("max_age_seconds must be non-negative")
        if not callable(self.clock):
            raise ValueError("clock must be callable returning seconds")

    def mint(
        self,
        session_id: str,
        scope: ContainmentScope,
        nonce: str | None = None,
        created_at: int | None = None,
    ) -> RealityAnchor:
        """Produce a freshly signed anchor for *session_id*/*scope*."""
        if not session_id:
            raise ValueError("session_id must be non-empty")
        validate_scope(scope)
        resolved_nonce = nonce if nonce is not None else secrets.token_hex(16)
        resolved_time = created_at if created_at is not None else int(self._now())
        payload = _canonicalise(
            session_id, scope, self.issuer, resolved_time, resolved_nonce
        )
        mac = hmac.new(self.key, payload, sha256).hexdigest()
        return RealityAnchor(
            session_id=session_id,
            scope=scope,
            issuer=self.issuer,
            created_at=resolved_time,
            nonce=resolved_nonce,
            mac=mac,
        )

    def verify(
        self,
        anchor: RealityAnchor,
        expected_scope: ContainmentScope | None = None,
    ) -> AnchorVerification:
        """Constant-time MAC check plus optional scope assertion.

        Returns :class:`AnchorVerification` rather than raising so
        callers can decide whether a bad anchor should fail-closed
        (reject the action) or fail-open-with-alarm (log + continue,
        when the anchor was optional).
        """
        expected_mac = hmac.new(self.key, anchor.canonical_payload, sha256).hexdigest()
        if not hmac.compare_digest(expected_mac, anchor.mac):
            return AnchorVerification(valid=False, reason="mac_mismatch")
        if anchor.issuer != self.issuer:
            return AnchorVerification(valid=False, reason="issuer_mismatch")
        if self.max_age_seconds > 0:
            age = int(self._now()) - anchor.created_at
            if age > self.max_age_seconds:
                return AnchorVerification(
                    valid=False,
                    reason=f"expired (age={age}s > max={self.max_age_seconds}s)",
                )
            if age < -5:
                # Small negative ages are permitted (clock skew), but
                # an anchor claiming the distant future is tampered.
                return AnchorVerification(
                    valid=False, reason=f"future_timestamp (age={age}s)"
                )
        if expected_scope is not None:
            validate_scope(expected_scope)
            if anchor.scope != expected_scope:
                return AnchorVerification(
                    valid=False,
                    reason=(
                        f"scope_mismatch (anchor={anchor.scope}, "
                        f"expected={expected_scope})"
                    ),
                )
        return AnchorVerification(valid=True)

    def _now(self) -> float:
        # ``clock`` is validated callable in __post_init__.
        fn = self.clock
        result = fn() if callable(fn) else time.time()
        return float(result)


def _canonicalise(
    session_id: str,
    scope: str,
    issuer: str,
    created_at: int,
    nonce: str,
) -> bytes:
    """Build the byte string that gets HMACed.

    The delimiter ``|`` never appears in a valid scope or nonce
    (both are constrained sets) and is escaped from the two free
    fields to avoid ambiguity.
    """
    return "|".join(
        [
            _escape(session_id),
            scope,
            _escape(issuer),
            str(created_at),
            nonce,
        ]
    ).encode("utf-8")


def _escape(field_value: str) -> str:
    return field_value.replace("\\", "\\\\").replace("|", "\\|")
