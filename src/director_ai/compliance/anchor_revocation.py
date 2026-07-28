# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — offline revocation evidence for RFC 3161 anchors

"""Fail-closed CRL and OCSP evidence checks for RFC 3161 TSA certificates.

The timestamp token verifier builds a certification path to an operator-pinned
root. This module evaluates operator-supplied, offline revocation evidence for
every non-root certificate on that exact path. It deliberately does not fetch
URLs from certificate extensions: evidence retrieval and custody remain an
operator responsibility, while verification stays deterministic and bounded.

CRLs are restricted to direct, complete CRLs. Indirect and delta CRLs fail
closed because applying their distribution-point scope incompletely would be
worse than rejecting them. OCSP responses may be signed by the certificate
issuer or by a directly issued responder with the OCSP-signing EKU. A delegated
responder without ``id-pkix-ocsp-nocheck`` must itself have valid CRL coverage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from pathlib import Path
from typing import cast

from cryptography import x509
from cryptography.hazmat.primitives.asymmetric.types import (
    CertificateIssuerPublicKeyTypes,
)
from cryptography.x509 import ocsp
from cryptography.x509.oid import (
    CRLEntryExtensionOID,
    ExtensionOID,
)

__all__ = [
    "RevocationEvidence",
    "RevocationEvidenceError",
    "load_revocation_evidence",
    "verify_certificate_path_revocation",
]

_MAX_EVIDENCE_BYTES = 8 * 1024 * 1024
_TIME_PRESERVING_REASONS = frozenset(
    {
        x509.ReasonFlags.unspecified,
        x509.ReasonFlags.affiliation_changed,
        x509.ReasonFlags.superseded,
        x509.ReasonFlags.cessation_of_operation,
    }
)


class RevocationEvidenceError(ValueError):
    """Raised when an operator-supplied CRL or OCSP artefact cannot be loaded."""


@dataclass(frozen=True)
class RevocationEvidence:
    """Parsed offline revocation evidence and its verification clock.

    Parameters
    ----------
    crls:
        Direct, complete X.509 CRLs. Supply one file per CRL through
        :func:`load_revocation_evidence`.
    ocsp_responses:
        DER OCSP responses. Responses must remain fresh at ``checked_at``.
    checked_at:
        A timezone-aware verification time. Freshness is evaluated against this
        value, while revocation effective times are compared with the timestamp
        token's ``genTime``.
    """

    crls: tuple[x509.CertificateRevocationList, ...] = ()
    ocsp_responses: tuple[ocsp.OCSPResponse, ...] = ()
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Reject naïve clocks and empty evidence sets."""
        if self.checked_at.tzinfo is None or self.checked_at.utcoffset() is None:
            raise RevocationEvidenceError("checked_at must be timezone-aware")
        if not self.crls and not self.ocsp_responses:
            raise RevocationEvidenceError(
                "at least one CRL or OCSP response is required"
            )


class _EvidenceStatus(Enum):
    NOT_APPLICABLE = auto()
    INVALID = auto()
    GOOD = auto()
    REVOKED = auto()


def _read_bounded(path_value: str | Path) -> bytes:
    """Read one regular evidence file with a bounded size."""
    path = Path(path_value)
    if not path.is_file():
        raise RevocationEvidenceError(f"revocation evidence file not found: {path}")
    size = path.stat().st_size
    if size <= 0 or size > _MAX_EVIDENCE_BYTES:
        raise RevocationEvidenceError(
            f"revocation evidence file size must be 1..{_MAX_EVIDENCE_BYTES} bytes: {path}"
        )
    return path.read_bytes()


def _load_crl(path_value: str | Path) -> x509.CertificateRevocationList:
    """Load one PEM or DER CRL from ``path_value``."""
    data = _read_bounded(path_value)
    try:
        if data.lstrip().startswith(b"-----BEGIN"):
            return x509.load_pem_x509_crl(data)
        return x509.load_der_x509_crl(data)
    except ValueError as exc:
        raise RevocationEvidenceError(f"invalid CRL: {path_value}") from exc


def _load_ocsp(path_value: str | Path) -> ocsp.OCSPResponse:
    """Load one DER OCSP response from ``path_value``."""
    data = _read_bounded(path_value)
    try:
        return ocsp.load_der_ocsp_response(data)
    except ValueError as exc:
        raise RevocationEvidenceError(
            f"invalid DER OCSP response: {path_value}"
        ) from exc


def load_revocation_evidence(
    *,
    crl_paths: Sequence[str | Path] = (),
    ocsp_paths: Sequence[str | Path] = (),
    checked_at: datetime | None = None,
) -> RevocationEvidence:
    """Load bounded, operator-supplied CRL and OCSP evidence files.

    Parameters
    ----------
    crl_paths:
        Paths to individual PEM or DER CRLs.
    ocsp_paths:
        Paths to individual DER OCSP responses.
    checked_at:
        Optional timezone-aware verification time; defaults to current UTC.

    Returns
    -------
    RevocationEvidence
        Parsed evidence ready for deterministic path verification.

    Raises
    ------
    RevocationEvidenceError
        If a path is missing, oversized, malformed, or the evidence set is empty.
    """
    return RevocationEvidence(
        crls=tuple(_load_crl(path) for path in crl_paths),
        ocsp_responses=tuple(_load_ocsp(path) for path in ocsp_paths),
        checked_at=checked_at or datetime.now(UTC),
    )


def _has_unknown_critical(extensions: x509.Extensions) -> bool:
    """Return whether ``extensions`` contains an unhandled critical value."""
    return any(
        extension.critical and isinstance(extension.value, x509.UnrecognizedExtension)
        for extension in extensions
    )


def _crl_issuer_authorized(issuer: x509.Certificate) -> bool:
    """Return whether ``issuer`` is certified to sign CRLs.

    RFC 10007 updates RFC 5280 to require the KeyUsage extension with cRLSign
    for v3 CRL-issuer certificates. Legacy v1 certificates have no extension
    field and retain the RFC exception.
    """
    if issuer.version is not x509.Version.v3:
        return True
    try:
        key_usage = cast(
            x509.KeyUsage,
            issuer.extensions.get_extension_for_oid(ExtensionOID.KEY_USAGE).value,
        )
    except x509.ExtensionNotFound:
        return False
    return bool(key_usage.crl_sign)


def _revocation_invalidates(
    reason: x509.ReasonFlags | None,
    revoked_at: datetime,
    token_time: datetime,
) -> bool:
    """Apply RFC 3161 TSA-revocation reason semantics."""
    if reason not in _TIME_PRESERVING_REASONS:
        # Missing reason and key/CA/AA compromise invalidate every token. Other
        # unexpected reasons also fail closed rather than inventing semantics.
        return True
    return revoked_at <= token_time


def _crl_status(
    certificate: x509.Certificate,
    issuer: x509.Certificate,
    crl: x509.CertificateRevocationList,
    *,
    token_time: datetime,
    checked_at: datetime,
) -> _EvidenceStatus:
    """Return the status one direct, complete CRL gives ``certificate``."""
    if crl.issuer != issuer.subject:
        return _EvidenceStatus.NOT_APPLICABLE
    if not _crl_issuer_authorized(issuer):
        return _EvidenceStatus.INVALID
    if _has_unknown_critical(crl.extensions):
        return _EvidenceStatus.INVALID
    unsupported_scope_oids = {
        ExtensionOID.DELTA_CRL_INDICATOR,
        ExtensionOID.ISSUING_DISTRIBUTION_POINT,
    }
    if any(extension.oid in unsupported_scope_oids for extension in crl.extensions):
        return _EvidenceStatus.INVALID
    next_update = crl.next_update_utc
    if (
        next_update is None
        or not crl.last_update_utc <= checked_at <= next_update
        or not issuer.not_valid_before_utc
        <= crl.last_update_utc
        <= issuer.not_valid_after_utc
        or not crl.is_signature_valid(
            cast(CertificateIssuerPublicKeyTypes, issuer.public_key())
        )
    ):
        return _EvidenceStatus.INVALID

    revoked = crl.get_revoked_certificate_by_serial_number(certificate.serial_number)
    if revoked is None:
        return _EvidenceStatus.GOOD
    if _has_unknown_critical(revoked.extensions):
        return _EvidenceStatus.INVALID
    if any(
        extension.oid == CRLEntryExtensionOID.CERTIFICATE_ISSUER
        for extension in revoked.extensions
    ):
        return _EvidenceStatus.INVALID

    reason: x509.ReasonFlags | None = None
    revoked_at = revoked.revocation_date_utc
    reason_extension = None
    invalidity_extension = None
    try:
        reason_extension = revoked.extensions.get_extension_for_oid(
            CRLEntryExtensionOID.CRL_REASON
        )
    except x509.ExtensionNotFound:
        reason_extension = None
    try:
        invalidity_extension = revoked.extensions.get_extension_for_oid(
            CRLEntryExtensionOID.INVALIDITY_DATE
        )
    except x509.ExtensionNotFound:
        invalidity_extension = None
    if reason_extension is not None:
        reason = cast(x509.CRLReason, reason_extension.value).reason
    if invalidity_extension is not None:
        revoked_at = cast(
            x509.InvalidityDate, invalidity_extension.value
        ).invalidity_date_utc
    if _revocation_invalidates(reason, revoked_at, token_time):
        return _EvidenceStatus.REVOKED
    return _EvidenceStatus.GOOD


def verify_certificate_path_revocation(
    certificate_path: Sequence[x509.Certificate],
    evidence: RevocationEvidence,
    *,
    token_time: datetime,
) -> bool:
    """Verify revocation coverage for every non-root certificate on a path.

    Parameters
    ----------
    certificate_path:
        Ordered leaf-to-root path already validated cryptographically.
    evidence:
        Fresh offline CRL/OCSP evidence.
    token_time:
        RFC 3161 token ``genTime``. Ordinary retirement reasons preserve tokens
        created before revocation; missing reasons and key compromise do not.

    Returns
    -------
    bool
        ``True`` only when every non-root certificate has at least one valid
        good-at-``token_time`` status source and no valid source reports a
        revocation that invalidates the token.
    """
    if token_time.tzinfo is None or token_time.utcoffset() is None:
        return False
    checked_at = evidence.checked_at.astimezone(UTC)
    token_time = token_time.astimezone(UTC)
    if len(certificate_path) < 2 or token_time > checked_at:
        return False

    # Local import avoids exposing implementation details while keeping the
    # bounded OCSP verifier independent from the public evidence container.
    from director_ai.compliance._anchor_revocation_ocsp import _ocsp_status

    for certificate, issuer in zip(
        certificate_path, certificate_path[1:], strict=False
    ):
        statuses = [
            _crl_status(
                certificate,
                issuer,
                crl,
                token_time=token_time,
                checked_at=checked_at,
            )
            for crl in evidence.crls
        ]
        statuses.extend(
            _ocsp_status(
                certificate,
                issuer,
                response,
                crls=evidence.crls,
                checked_at=checked_at,
                token_time=token_time,
            )
            for response in evidence.ocsp_responses
        )
        if _EvidenceStatus.REVOKED in statuses:
            return False
        if _EvidenceStatus.INVALID in statuses:
            return False
        if _EvidenceStatus.GOOD not in statuses:
            return False
    return True
