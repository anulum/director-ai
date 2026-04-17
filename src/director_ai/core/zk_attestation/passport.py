# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — cross-org passport issuer + verifier

"""Signed bundle of proved statements for an agent handoff.

An :class:`PassportIssuer` builds passports on the source org's
side: for each statement the issuer calls its matching backend's
``prove`` method, collects the proofs, and signs the whole
bundle with the issuer's HMAC key. A :class:`PassportVerifier`
on the receiving side verifies the signature then iterates
``(statement, proof)`` pairs, handing each off to the appropriate
backend.

The passport format is designed for JSON serialisation across
organisational boundaries — none of the fields carry raw samples,
and the proof payload is either a :class:`CommitmentProof`
dataclass (for the commitment backend) or a ``bytes`` blob
(for plug-in zk-SNARK adapters).
"""

from __future__ import annotations

import hmac
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from hashlib import sha256

from .backends import AttestationBackend, CommitmentBackend
from .statements import AttestationStatement, HistorySample

_MIN_KEY_LEN = 32
_MAC_HEX_LEN = 64


@dataclass(frozen=True)
class _StatementEntry:
    """One ``(statement, proof, backend_kind)`` tuple inside a passport."""

    statement: AttestationStatement
    proof: object
    backend_kind: str


@dataclass(frozen=True)
class CrossOrgPassport:
    """A frozen attestation bundle ready to hand to another org.

    Attributes
    ----------
    agent_id : str
        Stable identifier for the agent across orgs.
    issuing_org : str
        Source org's canonical identifier (``"org://example.com"``).
    created_at : int
        Unix epoch seconds at issue time.
    entries : tuple[_StatementEntry, ...]
        Proved statements in the order they were added.
    mac : str
        Lowercase-hex HMAC-SHA256 over the canonical header +
        per-entry statement kinds + backend kinds. The verifier
        re-derives the MAC before trusting any entry.
    """

    agent_id: str
    issuing_org: str
    created_at: int
    entries: tuple[_StatementEntry, ...]
    mac: str

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id must be non-empty")
        if not self.issuing_org:
            raise ValueError("issuing_org must be non-empty")
        if self.created_at < 0:
            raise ValueError("created_at must be non-negative")
        if not self.entries:
            raise ValueError("passport must contain at least one entry")
        if len(self.mac) != _MAC_HEX_LEN:
            raise ValueError("mac must be a SHA-256 hex digest")

    @property
    def canonical_header(self) -> bytes:
        """Byte string that participates in the MAC."""
        return _canonical_header(
            self.agent_id,
            self.issuing_org,
            self.created_at,
            tuple(
                (entry.statement.kind, entry.statement.name, entry.backend_kind)
                for entry in self.entries
            ),
        )


@dataclass(frozen=True)
class PassportVerdict:
    """Aggregate answer from :meth:`PassportVerifier.verify`.

    ``accepted`` is True only when every statement's proof
    verifies. ``failures`` is the ordered list of
    ``(statement_name, reason)`` for entries that failed.
    """

    accepted: bool
    signature_ok: bool
    failures: tuple[tuple[str, str], ...]

    def summary(self) -> str:
        if self.accepted:
            return "all statements proved"
        if not self.signature_ok:
            return "passport signature failed"
        return f"{len(self.failures)} statement(s) failed"


@dataclass
class PassportIssuer:
    """Mints a :class:`CrossOrgPassport` for the source org.

    Parameters
    ----------
    key : bytes
        HMAC secret (≥ 32 bytes) used both to sign the passport
        header *and* — by default — to seed the commitment
        backend. Distinct keys for the two roles are supported by
        passing a pre-configured ``default_backend``.
    issuing_org : str
        Org identifier stamped on every passport.
    default_backend : AttestationBackend, optional
        Backend used when an entry does not explicitly specify
        one. Defaults to a :class:`CommitmentBackend` constructed
        from ``key``.
    clock : callable, optional
        Returns current unix seconds; tests inject a deterministic
        source.
    """

    key: bytes
    issuing_org: str
    default_backend: object = field(default=None)
    clock: object = field(default=time.time)

    def __post_init__(self) -> None:
        if len(self.key) < _MIN_KEY_LEN:
            raise ValueError("HMAC key must be at least 32 bytes")
        if not self.issuing_org:
            raise ValueError("issuing_org must be non-empty")
        if not callable(self.clock):
            raise ValueError("clock must be callable returning seconds")
        if self.default_backend is None:
            self.default_backend = CommitmentBackend(key=self.key)
        if not isinstance(self.default_backend, AttestationBackend):
            raise TypeError(
                "default_backend must implement AttestationBackend"
            )

    def issue(
        self,
        agent_id: str,
        samples: Sequence[HistorySample],
        statements: Iterable[AttestationStatement],
        backends: dict[str, AttestationBackend] | None = None,
    ) -> CrossOrgPassport:
        """Produce a signed passport for *agent_id*.

        ``backends`` is an optional map from statement name to a
        backend override. Any statement without an override uses
        ``self.default_backend``.
        """
        if not agent_id:
            raise ValueError("agent_id must be non-empty")
        if not samples:
            raise ValueError("samples must be non-empty")

        backends = backends or {}
        entries: list[_StatementEntry] = []
        for statement in statements:
            backend = backends.get(statement.name, None)
            if backend is None:
                backend_obj = self.default_backend
                if not isinstance(backend_obj, AttestationBackend):
                    raise TypeError(
                        "default_backend must implement AttestationBackend"
                    )
                backend = backend_obj
            proof = backend.prove(statement, samples)
            entries.append(
                _StatementEntry(
                    statement=statement,
                    proof=proof,
                    backend_kind=backend.kind,
                ),
            )
        if not entries:
            raise ValueError("must provide at least one statement")

        created_at = int(self._now())
        header = _canonical_header(
            agent_id,
            self.issuing_org,
            created_at,
            tuple(
                (entry.statement.kind, entry.statement.name, entry.backend_kind)
                for entry in entries
            ),
        )
        mac = hmac.new(self.key, header, sha256).hexdigest()
        return CrossOrgPassport(
            agent_id=agent_id,
            issuing_org=self.issuing_org,
            created_at=created_at,
            entries=tuple(entries),
            mac=mac,
        )

    def _now(self) -> float:
        fn = self.clock
        return float(fn()) if callable(fn) else time.time()


@dataclass
class PassportVerifier:
    """Verifies a passport against the receiving org's copy of the
    issuer's public verification key.

    The verifier holds a map from ``issuing_org`` to the shared
    HMAC secret — in practice this map is hydrated from a PKI /
    out-of-band key exchange. It also holds a map from
    ``backend_kind`` to the backend instance that can verify
    proofs of that kind.
    """

    issuer_keys: dict[str, bytes]
    backends: dict[str, AttestationBackend]

    def __post_init__(self) -> None:
        if not self.issuer_keys:
            raise ValueError("issuer_keys must not be empty")
        for org, key in self.issuer_keys.items():
            if not org:
                raise ValueError("issuer_keys map contains empty org id")
            if len(key) < _MIN_KEY_LEN:
                raise ValueError(
                    f"issuer key for {org!r} must be at least 32 bytes"
                )
        if not self.backends:
            raise ValueError("backends must not be empty")
        for kind, backend in self.backends.items():
            if not kind:
                raise ValueError("backends map contains empty kind")
            if not isinstance(backend, AttestationBackend):
                raise TypeError(
                    f"backend {kind!r} does not implement AttestationBackend"
                )

    def verify(self, passport: CrossOrgPassport) -> PassportVerdict:
        key = self.issuer_keys.get(passport.issuing_org)
        if key is None:
            return PassportVerdict(
                accepted=False,
                signature_ok=False,
                failures=(("_passport", "unknown_issuing_org"),),
            )
        expected = hmac.new(key, passport.canonical_header, sha256).hexdigest()
        if not hmac.compare_digest(expected, passport.mac):
            return PassportVerdict(
                accepted=False,
                signature_ok=False,
                failures=(("_passport", "mac_mismatch"),),
            )

        failures: list[tuple[str, str]] = []
        for entry in passport.entries:
            backend = self.backends.get(entry.backend_kind)
            if backend is None:
                failures.append(
                    (entry.statement.name, f"no_backend_for_{entry.backend_kind}"),
                )
                continue
            ok, reason = backend.verify(entry.statement, entry.proof)
            if not ok:
                failures.append((entry.statement.name, reason))

        return PassportVerdict(
            accepted=not failures,
            signature_ok=True,
            failures=tuple(failures),
        )


def _canonical_header(
    agent_id: str,
    issuing_org: str,
    created_at: int,
    entries: tuple[tuple[str, str, str], ...],
) -> bytes:
    """Deterministic header for MAC computation.

    Escape ``|`` and ``\\`` in the free-text fields so a
    statement whose name contains the delimiter cannot collide
    with a different passport layout.
    """
    header_fields = [
        _escape(agent_id),
        _escape(issuing_org),
        str(created_at),
    ]
    for kind, name, backend_kind in entries:
        header_fields.append(
            "::".join([_escape(kind), _escape(name), _escape(backend_kind)])
        )
    return "|".join(header_fields).encode("utf-8")


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|")
