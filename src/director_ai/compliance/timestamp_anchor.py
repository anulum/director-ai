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
token binds the chain up to that head to the token's ``genTime``.

Verification has two tiers (see :func:`verify_token`). By default it is
internal-consistency only — the token is signed by the certificate it carries over
exactly our digest — which is **TSA-token-anchored**. When trusted TSA root
certificates are configured (``audit_anchor_tsa_roots`` /
:func:`load_trusted_roots`), the signing certificate must additionally chain to a
pinned root, valid at ``genTime`` with the time-stamping EKU — which is
**trusted-TSA-attested** and completes back-dating resistance. Do not describe an
anchor as trusted-TSA-attested unless it was verified against pinned roots.

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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from director_ai.compliance.anchor_revocation import RevocationEvidence

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
    "load_trusted_roots",
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
    """POST a DER ``TimeStampReq`` to ``tsa_url`` and return the response bytes.

    The TSA endpoint is an operator-configured deployment setting, but it is
    restricted to ``http(s)`` here so the ``urllib`` ``file://`` (and similar
    local-scheme) read vector cannot be reached even from a misconfigured value.
    The linters below are silenced because that check makes the call safe.
    """
    if not tsa_url.lower().startswith(("https://", "http://")):
        raise TsaUnreachableError(f"TSA URL scheme must be http(s): {tsa_url!r}")
    request = urllib.request.Request(  # noqa: S310
        tsa_url,
        data=request_der,
        headers={"Content-Type": _TIMESTAMP_QUERY},
        method="POST",
    )
    with urllib.request.urlopen(  # noqa: S310  # nosec B310  # nosemgrep
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


def _find_signer_cert(certs: Any, signer_info: Any) -> Any:
    """Return the embedded certificate identified by the ``SignerInfo`` sid.

    The CMS ``certificates`` field is a SET (unordered on the wire — a multi-cert
    token's DER can reorder the signer past position 0), so the signing cert is
    located by its issuer name + serial number, not by position. Only the
    ``IssuerAndSerialNumber`` sid form is handled (what RFC 3161 TSAs use); any
    other form raises and the caller fails closed.
    """
    sid = signer_info["sid"].chosen
    issuer_der = sid["issuer"].dump()
    serial = sid["serial_number"].native
    for choice in certs:
        cert = choice.chosen
        if cert.serial_number == serial and cert.issuer.dump() == issuer_der:
            return cert
    raise TsaResponseError("signer certificate not found in the token")


def load_trusted_roots(pem_path: str | Path) -> list[Any]:
    """Load trusted TSA root certificates from a PEM bundle.

    Returns the parsed ``cryptography`` ``Certificate`` objects. Raises
    :class:`AnchorError` when the file is missing or holds no certificate.
    """
    from cryptography.x509 import load_pem_x509_certificates

    path = Path(pem_path)
    if not path.is_file():
        raise AnchorError(f"trusted-roots PEM not found: {pem_path}")
    try:
        certs = load_pem_x509_certificates(path.read_bytes())
    except ValueError as exc:
        raise AnchorError(
            f"no valid certificates in trusted-roots PEM: {pem_path}"
        ) from exc
    return list(certs)


def _leaf_has_timestamping_eku(leaf: Any) -> bool:
    """Return whether ``leaf``'s EKU is exactly ``{id-kp-timeStamping}`` and critical.

    RFC 3161 §2.3 requires the time-stamping EKU to be the SOLE extended key usage
    and marked critical; a leaf carrying timeStamping among other usages, or a
    non-critical EKU, is not a conformant TSA signing certificate and is rejected.
    """
    from cryptography.x509 import ExtensionNotFound
    from cryptography.x509.oid import ExtendedKeyUsageOID, ExtensionOID

    try:
        ext = leaf.extensions.get_extension_for_oid(ExtensionOID.EXTENDED_KEY_USAGE)
    except ExtensionNotFound:
        return False
    return bool(ext.critical) and list(ext.value) == [ExtendedKeyUsageOID.TIME_STAMPING]


def _cert_time_valid(cert: Any, at_time: Any) -> bool:
    """Return whether ``cert`` is inside its validity window at ``at_time`` (UTC)."""
    return bool(cert.not_valid_before_utc <= at_time <= cert.not_valid_after_utc)


def _valid_ca_issuer(cert: Any, intermediates_below: int) -> bool:
    """Return whether ``cert`` may act as a CA issuer for a path (RFC 5280).

    ``cryptography``'s ``verify_directly_issued_by`` checks only issuer name +
    signature, so a non-CA end-entity certificate would otherwise be accepted as
    an issuer. Because intermediates come from the attacker-controlled token, each
    non-leaf certificate must independently satisfy the CA constraints before it
    is trusted to have issued the cert below it: ``basicConstraints`` CA=True, a
    ``pathLenConstraint`` (if present) that permits ``intermediates_below``
    subordinate CAs, and — when a ``KeyUsage`` extension is present — the
    ``keyCertSign`` bit.
    """
    from cryptography.x509 import ExtensionNotFound
    from cryptography.x509.oid import ExtensionOID

    try:
        basic = cert.extensions.get_extension_for_oid(
            ExtensionOID.BASIC_CONSTRAINTS
        ).value
    except ExtensionNotFound:
        return False
    if not basic.ca:
        return False
    if basic.path_length is not None and intermediates_below > basic.path_length:
        return False
    try:
        key_usage = cert.extensions.get_extension_for_oid(ExtensionOID.KEY_USAGE).value
    except ExtensionNotFound:
        return True
    return bool(key_usage.key_cert_sign)


def _path_to_trusted_root(
    leaf: Any,
    extra_certs: list[Any],
    trusted_roots: list[Any],
    at_time: Any,
    max_depth: int = 4,
) -> list[Any] | None:
    """Return a validated leaf-to-root path, or ``None`` when none exists.

    The leaf must carry a sole, critical time-stamping EKU and be time-valid. Each
    hop is a ``verify_directly_issued_by`` (issuer-name match + signature); because
    the intermediates come from the attacker-controlled token, every candidate
    ISSUER (each intermediate and the pinned root) must also satisfy the CA
    constraints (:func:`_valid_ca_issuer`: basicConstraints CA=True, keyCertSign,
    pathLenConstraint) so a leaked non-CA end-entity key cannot forge a chain.
    Every cert on the path must be time-valid at ``at_time``. Bounded depth guards
    a malformed certificate set. There is no revocation or name-constraint
    checking: trust is anchored on the operator-pinned root, the appropriate model
    for a pinned-root TSA anchor.
    """
    if not _leaf_has_timestamping_eku(leaf) or not _cert_time_valid(leaf, at_time):
        return None
    frontier: list[tuple[Any, int, list[Any]]] = [(leaf, 0, [leaf])]
    while frontier:
        node, depth, path = frontier.pop()
        for root in trusted_roots:
            if not _cert_time_valid(root, at_time) or not _valid_ca_issuer(root, depth):
                continue
            try:
                node.verify_directly_issued_by(root)
                return [*path, root]
            except Exception:  # noqa: BLE001  # nosec B112 - not this root; keep searching
                continue
        if depth >= max_depth:
            continue
        for inter in extra_certs:
            if not _cert_time_valid(inter, at_time) or not _valid_ca_issuer(
                inter, depth
            ):
                continue
            try:
                node.verify_directly_issued_by(inter)
            except Exception:  # noqa: BLE001  # nosec B112 - not the issuer; skip
                continue
            frontier.append((inter, depth + 1, [*path, inter]))
    return None


def _chain_to_trusted_root(
    leaf: Any,
    extra_certs: list[Any],
    trusted_roots: list[Any],
    at_time: Any,
    max_depth: int = 4,
) -> bool:
    """Return whether ``leaf`` has a valid path to a pinned root."""
    return (
        _path_to_trusted_root(
            leaf,
            extra_certs,
            trusted_roots,
            at_time,
            max_depth=max_depth,
        )
        is not None
    )


def verify_token(
    anchor: TimestampAnchor,
    expected_hash_hex: str,
    *,
    trusted_roots: list[Any] | None = None,
    revocation_evidence: RevocationEvidence | None = None,
) -> bool:
    """Return ``True`` when the token verifies over the digest (fail-closed).

    Always applies four internal-consistency checks — any parse error or mismatch
    returns ``False``: the token's message imprint equals
    ``SHA-256(expected_hash_hex bytes)``; the ``content-type`` signed attribute is
    ``id-ct-TSTInfo``; the ``message-digest`` signed attribute equals the hash of
    the encapsulated ``TSTInfo``; and the CMS signature over the signed attributes
    verifies against the certificate embedded in the token (RSA or ECDSA; other
    key types fail closed).

    When ``trusted_roots`` is a non-empty list of ``cryptography`` ``Certificate``
    roots, a fifth check is required: the embedded signing certificate must chain
    to one of those roots — validated as of the token's ``genTime``, with the
    time-stamping EKU on the leaf and every cert on the path time-valid (see
    :func:`_path_to_trusted_root`). When ``revocation_evidence`` is supplied, a
    sixth fail-closed check requires fresh, signed CRL or OCSP coverage for every
    non-root certificate on that exact path. Revocation checking is invalid
    without pinned roots because there would be no authenticated issuer path.

    Scope (honest): WITHOUT ``trusted_roots`` the result means only that *some*
    certificate signed this digest at the claimed ``genTime`` — the chain is
    **TSA-token-anchored**, and back-dating resistance depends on the certificate
    being trusted out of band. WITH pinned ``trusted_roots`` the result means a
    *trusted* TSA signed it — **trusted-TSA-attested**. With fresh revocation
    evidence, the stronger result is **trusted-TSA-attested + revocation-evidenced**.
    Do not claim either tier unless its inputs were supplied and verified. Name
    constraints remain outside this pinned-root profile.
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

        certs = signed_data["certificates"]
        leaf_der = bytes(_find_signer_cert(certs, signer_info).dump())
        leaf_cert = load_der_x509_certificate(leaf_der)
        public_key = leaf_cert.public_key()
        signature = signer_info["signature"].native
        to_verify = _reencode_signed_attrs(signer_info)
        if isinstance(public_key, ec.EllipticCurvePublicKey):
            public_key.verify(signature, to_verify, ec.ECDSA(hashes.SHA256()))
        elif isinstance(public_key, rsa.RSAPublicKey):
            public_key.verify(signature, to_verify, padding.PKCS1v15(), hashes.SHA256())
        else:
            # Unsupported TSA key type (e.g. DSA/Ed25519) — fail closed.
            return False

        if revocation_evidence is not None and not trusted_roots:
            return False
        if trusted_roots:
            extra_certs = [
                load_der_x509_certificate(bytes(c.chosen.dump()))
                for c in certs
                if bytes(c.chosen.dump()) != leaf_der
            ]
            certificate_path = _path_to_trusted_root(
                leaf_cert,
                extra_certs,
                trusted_roots,
                tst_info["gen_time"].native,
            )
            if certificate_path is None:
                return False
            if revocation_evidence is not None:
                from director_ai.compliance.anchor_revocation import (
                    verify_certificate_path_revocation,
                )

                if not verify_certificate_path_revocation(
                    certificate_path,
                    revocation_evidence,
                    token_time=tst_info["gen_time"].native,
                ):
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

    def verify_against_chain(
        self,
        *,
        trusted_roots: list[Any] | None = None,
        revocation_evidence: RevocationEvidence | None = None,
    ) -> tuple[bool, str | None]:
        """Verify every stored anchor against the audit chain in the same DB.

        For each anchor the ``anchored_hash`` must be a real sealed
        ``entry_hash`` in ``audit_log`` **and** its token must verify against
        that hash. When ``trusted_roots`` is supplied, each token must also chain
        to a pinned root (root-of-trust pinning). When ``revocation_evidence`` is
        supplied, every non-root path certificate must additionally have fresh,
        signed CRL or OCSP coverage. Returns
        ``(ok, first_bad_anchored_hash)``.
        """
        for anchor in self.all():
            if not self._chain_has_head(anchor.anchored_hash):
                return False, anchor.anchored_hash
            if not verify_token(
                anchor,
                anchor.anchored_hash,
                trusted_roots=trusted_roots,
                revocation_evidence=revocation_evidence,
            ):
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
