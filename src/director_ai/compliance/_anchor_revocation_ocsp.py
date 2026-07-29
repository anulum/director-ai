# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — bounded OCSP verification internals

"""Internal RFC 6960 verification helpers for offline anchor evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime
from typing import Any, cast

from asn1crypto import ocsp as asn1_ocsp
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.x509 import ocsp
from cryptography.x509.oid import ExtendedKeyUsageOID, ExtensionOID

from director_ai.compliance.anchor_revocation import (
    _crl_status,
    _EvidenceStatus,
    _has_unknown_critical,
    _revocation_invalidates,
)


def _responder_key_hash(certificate: x509.Certificate) -> bytes:
    """Return the RFC 6960 SHA-1 hash of the subjectPublicKey BIT STRING."""
    from asn1crypto import keys

    spki = certificate.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_key_bits = keys.PublicKeyInfo.load(spki)["public_key"].contents[1:]
    # RFC 6960 section 4.2.2.3 requires SHA-1 for ResponderID byKey.
    return hashlib.sha1(  # noqa: S324  # nosemgrep: python.lang.security.insecure-hash-algorithms.insecure-hash-algorithm-sha1
        public_key_bits, usedforsecurity=False
    ).digest()


def _matches_responder_id(
    response: ocsp.OCSPResponse, certificate: x509.Certificate
) -> bool:
    """Return whether ``certificate`` matches the OCSP responder identifier."""
    if response.responder_name is not None:
        return bool(response.responder_name == certificate.subject)
    if response.responder_key_hash is not None:
        return bool(response.responder_key_hash == _responder_key_hash(certificate))
    return False


def _key_allows_digital_signature(certificate: x509.Certificate) -> bool:
    """Return whether KeyUsage, when present, permits digital signatures."""
    try:
        key_usage = cast(
            x509.KeyUsage,
            certificate.extensions.get_extension_for_oid(ExtensionOID.KEY_USAGE).value,
        )
    except x509.ExtensionNotFound:
        return True
    return bool(key_usage.digital_signature)


def _verify_ocsp_signature(
    response: ocsp.OCSPResponse, signer: x509.Certificate
) -> bool:
    """Verify an OCSP response signature for the supported RSA/ECDSA keys."""
    public_key = signer.public_key()
    hash_algorithm = response.signature_hash_algorithm
    try:
        if isinstance(public_key, rsa.RSAPublicKey) and hash_algorithm is not None:
            public_key.verify(
                response.signature,
                response.tbs_response_bytes,
                padding.PKCS1v15(),
                hash_algorithm,
            )
        elif isinstance(public_key, ec.EllipticCurvePublicKey) and hash_algorithm:
            public_key.verify(
                response.signature,
                response.tbs_response_bytes,
                ec.ECDSA(hash_algorithm),
            )
        else:
            return False
    except Exception:  # noqa: BLE001 - invalid signature is a status, not an error
        return False
    return True


def _has_critical_single_extensions(response: ocsp.OCSPResponse) -> bool:
    """Reject unprocessed critical extensions on any SingleResponse."""
    try:
        response_data: Any = asn1_ocsp.ResponseData.load(response.tbs_response_bytes)
        for single in response_data["responses"]:
            extensions = single["single_extensions"]
            if extensions.native is None:
                continue
            if any(bool(extension["critical"].native) for extension in extensions):
                return True
    except Exception:  # noqa: BLE001 - malformed extension data fails closed
        return True
    return False


def _delegated_responder_authorized(
    responder: x509.Certificate,
    issuer: x509.Certificate,
    *,
    crls: Sequence[x509.CertificateRevocationList],
    checked_at: datetime,
    produced_at: datetime,
) -> bool:
    """Return whether a delegated OCSP responder is authorized and current."""
    try:
        responder.verify_directly_issued_by(issuer)
    except Exception:  # noqa: BLE001 - fail closed on any chain error
        return False
    try:
        eku = cast(
            x509.ExtendedKeyUsage,
            responder.extensions.get_extension_for_oid(
                ExtensionOID.EXTENDED_KEY_USAGE
            ).value,
        )
    except x509.ExtensionNotFound:
        return False
    if list(eku) != [ExtendedKeyUsageOID.OCSP_SIGNING]:
        return False
    if not _key_allows_digital_signature(responder):
        return False
    try:
        basic = cast(
            x509.BasicConstraints,
            responder.extensions.get_extension_for_oid(
                ExtensionOID.BASIC_CONSTRAINTS
            ).value,
        )
        if basic.ca:
            return False
    except x509.ExtensionNotFound:
        pass
    try:
        responder.extensions.get_extension_for_oid(ExtensionOID.OCSP_NO_CHECK)
        return True
    except x509.ExtensionNotFound:
        pass

    statuses = [
        _crl_status(
            responder,
            issuer,
            crl,
            token_time=produced_at,
            checked_at=checked_at,
        )
        for crl in crls
    ]
    return bool(
        _EvidenceStatus.GOOD in statuses
        and _EvidenceStatus.INVALID not in statuses
        and _EvidenceStatus.REVOKED not in statuses
    )


def _ocsp_signer(
    response: ocsp.OCSPResponse,
    issuer: x509.Certificate,
    *,
    crls: Sequence[x509.CertificateRevocationList],
    checked_at: datetime,
) -> x509.Certificate | None:
    """Return the authorized OCSP response signer, or ``None``."""
    candidates_by_fingerprint = {
        cert.fingerprint(hashes.SHA256()): cert
        for cert in (issuer, *response.certificates)
    }
    candidates = candidates_by_fingerprint.values()
    signers = [cert for cert in candidates if _matches_responder_id(response, cert)]
    if len(signers) != 1:
        return None
    signer = signers[0]
    produced_at = response.produced_at_utc
    if produced_at is None or not (
        signer.not_valid_before_utc <= produced_at <= signer.not_valid_after_utc
    ):
        return None
    if signer.fingerprint(hashes.SHA256()) != issuer.fingerprint(hashes.SHA256()):
        if not _delegated_responder_authorized(
            signer,
            issuer,
            crls=crls,
            checked_at=checked_at,
            produced_at=produced_at,
        ):
            return None
    elif not _key_allows_digital_signature(signer):
        return None
    if not _verify_ocsp_signature(response, signer):
        return None
    return signer


def _ocsp_status(
    certificate: x509.Certificate,
    issuer: x509.Certificate,
    response: ocsp.OCSPResponse,
    *,
    crls: Sequence[x509.CertificateRevocationList],
    checked_at: datetime,
    token_time: datetime,
) -> _EvidenceStatus:
    """Return the status one OCSP response gives ``certificate``."""
    if response.response_status is not ocsp.OCSPResponseStatus.SUCCESSFUL:
        return _EvidenceStatus.INVALID
    if _has_critical_single_extensions(response):
        return _EvidenceStatus.INVALID

    matching: list[ocsp.OCSPSingleResponse] = []
    for single in response.responses:
        if single.serial_number != certificate.serial_number:
            continue
        try:
            request = (
                ocsp.OCSPRequestBuilder()
                .add_certificate(certificate, issuer, single.hash_algorithm)
                .build()
            )
        except (TypeError, ValueError):
            return _EvidenceStatus.INVALID
        if (
            request.issuer_name_hash == single.issuer_name_hash
            and request.issuer_key_hash == single.issuer_key_hash
        ):
            matching.append(single)
    if not matching:
        return _EvidenceStatus.NOT_APPLICABLE
    if len(matching) != 1 or _has_unknown_critical(response.extensions):
        return _EvidenceStatus.INVALID
    single = matching[0]

    this_update = single.this_update_utc
    next_update = single.next_update_utc
    produced_at = response.produced_at_utc
    if (
        next_update is None
        or produced_at is None
        or not this_update <= produced_at <= checked_at <= next_update
    ):
        return _EvidenceStatus.INVALID
    if _ocsp_signer(response, issuer, crls=crls, checked_at=checked_at) is None:
        return _EvidenceStatus.INVALID

    if single.certificate_status is ocsp.OCSPCertStatus.GOOD:
        return _EvidenceStatus.GOOD
    if single.certificate_status is not ocsp.OCSPCertStatus.REVOKED:
        return _EvidenceStatus.INVALID
    revoked_at = single.revocation_time_utc
    if revoked_at is None:
        return _EvidenceStatus.INVALID
    if _revocation_invalidates(single.revocation_reason, revoked_at, token_time):
        return _EvidenceStatus.REVOKED
    return _EvidenceStatus.GOOD
