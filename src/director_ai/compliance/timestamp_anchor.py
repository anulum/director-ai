# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RFC 3161 external timestamp anchoring for the audit chain

"""External RFC 3161 trusted-timestamp anchoring for the compliance audit chain.

The compliance :class:`~director_ai.compliance.audit_log.AuditLog` seals every
interaction into a tamper-evident hash chain (SHA-256 content hash + ``prev_hash``
linkage + HMAC ``chain_tag``). That proves **tamper-evidence** but not
**existence-at-time**: whoever holds the HMAC secret can rebuild the whole chain
with back-dated timestamps.

This module narrows that gap. A Timestamp Authority (TSA) signs an RFC 3161
timestamp token over the current chain **head** (``entry_hash`` — which, through
the ``prev_hash`` linkage, commits to the whole prior history). A stored, verified
token binds the chain up to that head to the token's ``genTime``: it is
TSA-token-anchored. Verification here is internal-consistency only — the token is
signed by the certificate it carries over exactly our digest. Chaining that
certificate to a *trusted* TSA root (root-of-trust pinning), which is what turns
"some certificate attested this time" into "a trusted TSA attested this time" and
completes back-dating resistance, is a documented follow-up. Do not describe an
anchor as trusted-TSA-attested until that lands.

The feature is **opt-in and offline-graceful**: :func:`try_anchor_chain_head`
never raises into a caller near the audit path — a down or unreachable TSA yields
``None`` and a logged warning. Publishing to a public transparency log (Rekor) is
deliberately out of scope; anchoring here is a private TSA round-trip only.

The ASN.1/CMS handling needs :mod:`asn1crypto` (the ``crypto`` extra); imports are
lazy so the free/core surface never pulls it. Install with
``pip install director-ai[crypto]``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import time as _now
from typing import Any

__all__ = [
    "TimestampAnchor",
    "AnchorError",
    "AnchorDependencyError",
    "TsaUnreachableError",
    "TsaResponseError",
    "ImprintMismatchError",
    "Rfc3161Anchorer",
    "AnchorStore",
    "verify_token",
    "try_anchor_chain_head",
]

_logger = logging.getLogger("DirectorAI.ComplianceAnchor")

_GENESIS_HASH = "0" * 64
_TIMESTAMP_QUERY = "application/timestamp-query"
_GRANTED = ("granted", "granted_with_mods")

# A transport turns a DER ``TimeStampReq`` into the raw ``TimeStampResp`` bytes.
Transport = Callable[[bytes, str, float], bytes]


class AnchorError(Exception):
    """Base class for every anchoring failure."""


class AnchorDependencyError(AnchorError):
    """The optional ``asn1crypto`` / ``cryptography`` dependency is missing."""


class TsaUnreachableError(AnchorError):
    """The Timestamp Authority could not be reached (network/transport error)."""


class TsaResponseError(AnchorError):
    """The TSA returned a non-granted status or an unparsable response."""


class ImprintMismatchError(AnchorError):
    """The token's message imprint does not match the anchored digest."""


@dataclass(frozen=True)
class TimestampAnchor:
    """A stored RFC 3161 anchor over one chain head.

    ``token_der`` is the DER of the RFC 3161 timestamp token (the CMS
    ``ContentInfo``); ``imprint_sha256`` is the SHA-256 of the anchored head
    bytes that was sent to the TSA; ``gen_time`` is the TSA's asserted time as a
    UTC epoch; ``created_at`` is when this record was stored locally.
    """

    anchored_hash: str
    imprint_sha256: str
    gen_time: float
    serial_number: int
    tsa_url: str
    token_der: bytes
    created_at: float


def _require_asn1crypto() -> Any:
    """Return the ``asn1crypto`` ``(cms, tsp)`` modules or raise an install hint."""
    try:
        from asn1crypto import cms, tsp

        return cms, tsp
    except ImportError as exc:
        raise AnchorDependencyError(
            "RFC 3161 anchoring requires asn1crypto — install director-ai[crypto]"
        ) from exc


def _default_transport(request_der: bytes, tsa_url: str, timeout_s: float) -> bytes:
    """POST a DER ``TimeStampReq`` to ``tsa_url`` and return the response bytes."""
    request = urllib.request.Request(  # noqa: S310 - operator-configured TSA URL
        tsa_url,
        data=request_der,
        headers={"Content-Type": _TIMESTAMP_QUERY},
        method="POST",
    )
    with urllib.request.urlopen(  # noqa: S310  # nosec B310 (operator-set TSA URL)
        request, timeout=timeout_s
    ) as response:
        return bytes(response.read())


class Rfc3161Anchorer:
    """Builds RFC 3161 requests, submits them, and returns verified anchors.

    Parameters
    ----------
    tsa_url:
        The Timestamp Authority endpoint (``https``). Operator-configured.
    timeout_s:
        Network timeout for the TSA round-trip.
    transport:
        Injectable ``(request_der, tsa_url, timeout_s) -> response_der`` callable;
        defaults to a stdlib ``urllib`` POST. Tests inject a synthetic transport
        so no network is touched.
    """

    def __init__(
        self,
        tsa_url: str,
        timeout_s: float = 10.0,
        transport: Transport | None = None,
    ) -> None:
        self.tsa_url = tsa_url
        self.timeout_s = timeout_s
        self._transport = transport or _default_transport

    def build_request(self, imprint: bytes) -> bytes:
        """Return the DER ``TimeStampReq`` for a SHA-256 ``imprint``.

        A fresh 64-bit nonce binds the response to this request, and
        ``cert_req`` asks the TSA to embed its signing certificate so the token
        can be verified offline.
        """
        _cms, tsp = _require_asn1crypto()
        request = tsp.TimeStampReq(
            {
                "version": "v1",
                "message_imprint": tsp.MessageImprint(
                    {
                        "hash_algorithm": {"algorithm": "sha256"},
                        "hashed_message": imprint,
                    }
                ),
                "cert_req": True,
                "nonce": int.from_bytes(os.urandom(8), "big"),
            }
        )
        return bytes(request.dump())

    def submit(self, anchored_hash_hex: str) -> TimestampAnchor:
        """Anchor ``anchored_hash_hex`` (a chain head) and return the anchor.

        Raises :class:`TsaUnreachableError` on a transport failure,
        :class:`TsaResponseError` on a non-granted or unparsable response, and
        :class:`ImprintMismatchError` if the token does not commit to our digest.
        """
        _cms, tsp = _require_asn1crypto()
        head_bytes = bytes.fromhex(anchored_hash_hex)
        imprint = hashlib.sha256(head_bytes).digest()
        request_der = self.build_request(imprint)

        try:
            response_der = self._transport(request_der, self.tsa_url, self.timeout_s)
        except (urllib.error.URLError, OSError) as exc:
            raise TsaUnreachableError(f"TSA unreachable: {exc}") from exc

        try:
            response = tsp.TimeStampResp.load(response_der)
            status = response["status"]["status"].native
        except Exception as exc:  # noqa: BLE001 - any ASN.1 error is a bad response
            raise TsaResponseError(f"unparsable TSA response: {exc}") from exc
        if status not in _GRANTED:
            raise TsaResponseError(f"TSA status not granted: {status!r}")

        token = response["time_stamp_token"]
        tst_info = _extract_tst_info(token)
        token_imprint = tst_info["message_imprint"]["hashed_message"].native
        token_algo = tst_info["message_imprint"]["hash_algorithm"]["algorithm"].native
        if token_algo != "sha256" or token_imprint != imprint:  # nosec B105 (algo name)
            raise ImprintMismatchError(
                "token message imprint does not match the anchored digest"
            )
        gen_time = tst_info["gen_time"].native.timestamp()
        serial = int(tst_info["serial_number"].native)
        return TimestampAnchor(
            anchored_hash=anchored_hash_hex,
            imprint_sha256=imprint.hex(),
            gen_time=gen_time,
            serial_number=serial,
            tsa_url=self.tsa_url,
            token_der=bytes(token.dump()),
            created_at=_now(),
        )


def _extract_tst_info(token: Any) -> Any:
    """Return the ``TSTInfo`` carried by an RFC 3161 token ``ContentInfo``."""
    signed_data = token["content"]
    encap = signed_data["encap_content_info"]
    if encap["content_type"].native != "tst_info":
        raise TsaResponseError("token does not encapsulate a TSTInfo")
    return encap["content"].parsed


def _reencode_signed_attrs(signer_info: Any) -> bytes:
    """Return the signed attributes re-tagged as an explicit ``SET OF`` for signing."""
    # In a SignerInfo the attributes are ``[0] IMPLICIT`` (0xA0); the CMS
    # signature is computed over the same octets tagged as an explicit SET (0x31).
    der = bytes(signer_info["signed_attrs"].dump())
    return b"\x31" + der[1:]


def verify_token(anchor: TimestampAnchor, expected_hash_hex: str) -> bool:
    """Return ``True`` when the token is internally consistent over the digest.

    Four checks, all of which must pass (fail-closed — any parse error or
    mismatch returns ``False``): the token's message imprint equals
    ``SHA-256(expected_hash_hex bytes)``; the ``content-type`` signed attribute
    is ``id-ct-TSTInfo``; the ``message-digest`` signed attribute equals the hash
    of the encapsulated ``TSTInfo``; and the CMS signature over the signed
    attributes verifies against the certificate embedded in the token (RSA or
    ECDSA; other key types fail closed).

    Scope (honest): this establishes that *some* certificate signed this digest
    at the claimed ``genTime`` — the token is internally consistent and signed by
    the certificate it carries. It does NOT yet chain that certificate to a
    trusted TSA root, so back-dating resistance depends on the certificate being
    trusted out of band. Root-of-trust pinning to a configured TSA root — which
    would upgrade the guarantee to "a *trusted* TSA attested" — is a documented
    follow-up.
    """
    try:
        cms, _tsp = _require_asn1crypto()
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
        from cryptography.x509 import load_der_x509_certificate

        expected_imprint = hashlib.sha256(bytes.fromhex(expected_hash_hex)).digest()
        token = cms.ContentInfo.load(anchor.token_der)
        tst_info = _extract_tst_info(token)
        if tst_info["message_imprint"]["hashed_message"].native != expected_imprint:
            return False

        signed_data = token["content"]
        signer_info = signed_data["signer_infos"][0]
        econtent_der = bytes(signed_data["encap_content_info"]["content"].parsed.dump())

        # The signed attributes must bind the content: content-type must be
        # id-ct-TSTInfo (RFC 3161) and message-digest = SHA-256(the TSTInfo).
        # A missing attribute reads as None and fails the corresponding check.
        signed_attrs = {
            attr["type"].native: attr["values"][0].native
            for attr in signer_info["signed_attrs"]
        }
        if signed_attrs.get("content_type") != "tst_info":
            return False
        if signed_attrs.get("message_digest") != hashlib.sha256(econtent_der).digest():
            return False

        cert_der = bytes(signed_data["certificates"][0].chosen.dump())
        public_key = load_der_x509_certificate(cert_der).public_key()
        signature = signer_info["signature"].native
        to_verify = _reencode_signed_attrs(signer_info)
        if isinstance(public_key, ec.EllipticCurvePublicKey):
            public_key.verify(signature, to_verify, ec.ECDSA(hashes.SHA256()))
        elif isinstance(public_key, rsa.RSAPublicKey):
            public_key.verify(signature, to_verify, padding.PKCS1v15(), hashes.SHA256())
        else:
            # Unsupported TSA key type (e.g. DSA/Ed25519) — fail closed.
            return False
    except Exception:  # noqa: BLE001 - fail-closed on any verification error
        return False
    return True


class AnchorStore:
    """Durable SQLite storage for timestamp anchors, in the audit database.

    Opens its own connection to ``db_path`` (the same file the compliance
    :class:`~director_ai.compliance.audit_log.AuditLog` writes) and manages an
    ``audit_anchor`` table alongside the ``audit_log`` chain.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            self._db_path, check_same_thread=False
        )
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_anchor (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anchored_hash TEXT NOT NULL,
                imprint_sha256 TEXT NOT NULL,
                gen_time REAL NOT NULL,
                serial_number INTEGER NOT NULL,
                tsa_url TEXT NOT NULL,
                token_der BLOB NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        self._conn.commit()

    def record(self, anchor: TimestampAnchor) -> int:
        """Persist ``anchor`` and return its row id."""
        if self._conn is None:
            raise AnchorError("anchor store is closed")
        cursor = self._conn.execute(
            """INSERT INTO audit_anchor
               (anchored_hash, imprint_sha256, gen_time, serial_number,
                tsa_url, token_der, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                anchor.anchored_hash,
                anchor.imprint_sha256,
                anchor.gen_time,
                anchor.serial_number,
                anchor.tsa_url,
                anchor.token_der,
                anchor.created_at,
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def all(self) -> list[TimestampAnchor]:
        """Return every stored anchor, oldest first."""
        if self._conn is None:
            return []
        rows = self._conn.execute(
            "SELECT anchored_hash, imprint_sha256, gen_time, serial_number, "
            "tsa_url, token_der, created_at FROM audit_anchor ORDER BY id ASC"
        ).fetchall()
        return [
            TimestampAnchor(
                anchored_hash=r[0],
                imprint_sha256=r[1],
                gen_time=r[2],
                serial_number=r[3],
                tsa_url=r[4],
                token_der=bytes(r[5]),
                created_at=r[6],
            )
            for r in rows
        ]

    def latest(self) -> TimestampAnchor | None:
        """Return the most recently stored anchor, or ``None`` when empty."""
        anchors = self.all()
        return anchors[-1] if anchors else None

    def _chain_has_head(self, entry_hash: str) -> bool:
        """Return whether ``entry_hash`` is a sealed row in the audit chain.

        Only called from :meth:`verify_against_chain` after :meth:`all` has
        returned at least one row, so the connection is live here.
        """
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT 1 FROM audit_log WHERE entry_hash = ? LIMIT 1", (entry_hash,)
        ).fetchone()
        return row is not None

    def verify_against_chain(self) -> tuple[bool, str | None]:
        """Verify every stored anchor against the audit chain in the same DB.

        For each anchor the ``anchored_hash`` must be a real sealed
        ``entry_hash`` in ``audit_log`` **and** its token must verify against
        that hash. Returns ``(ok, first_bad_anchored_hash)``.
        """
        for anchor in self.all():
            if not self._chain_has_head(anchor.anchored_hash):
                return False, anchor.anchored_hash
            if not verify_token(anchor, anchor.anchored_hash):
                return False, anchor.anchored_hash
        return True, None

    def close(self) -> None:
        """Close the database connection. Safe to call more than once."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def try_anchor_chain_head(
    head_hex: str,
    anchorer: Rfc3161Anchorer,
    store: AnchorStore,
) -> TimestampAnchor | None:
    """Anchor ``head_hex`` and store it, offline-graceful.

    Returns the stored :class:`TimestampAnchor`, or ``None`` when the head is the
    genesis/empty hash (nothing to anchor) or when anchoring fails (an
    :class:`AnchorError` is logged, never raised — so a caller near the audit
    path is never broken by a down TSA).
    """
    if not head_hex or head_hex == _GENESIS_HASH:
        return None
    try:
        anchor = anchorer.submit(head_hex)
    except AnchorError as exc:
        _logger.warning("timestamp anchoring failed for %s: %s", head_hex[:12], exc)
        return None
    store.record(anchor)
    return anchor
