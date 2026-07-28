# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RFC 3161 anchor OCSP evidence tests

"""Real-signature OCSP tests for timestamp-anchor revocation checks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from cryptography.x509 import ocsp
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID, ObjectIdentifier

from director_ai.compliance._anchor_revocation_ocsp import (
    _key_allows_digital_signature,
    _matches_responder_id,
    _verify_ocsp_signature,
)
from director_ai.compliance.anchor_revocation import (
    RevocationEvidence,
    verify_certificate_path_revocation,
)

_TOKEN_TIME = datetime(2026, 7, 22, 3, 30, tzinfo=UTC)
_CHECKED_AT = datetime.now(UTC) + timedelta(minutes=1)


def _key_usage(
    *, digital_signature: bool, key_cert_sign: bool, crl_sign: bool
) -> x509.KeyUsage:
    return x509.KeyUsage(
        digital_signature=digital_signature,
        content_commitment=False,
        key_encipherment=False,
        data_encipherment=False,
        key_agreement=False,
        key_cert_sign=key_cert_sign,
        crl_sign=crl_sign,
        encipher_only=False,
        decipher_only=False,
    )


def _issue_certificate(
    subject: str,
    subject_key: rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey,
    issuer: str,
    issuer_key: rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey,
    *,
    ca: bool,
    key_cert_sign: bool = False,
    crl_sign: bool = False,
    digital_signature: bool = True,
    eku: ObjectIdentifier | None = None,
    ocsp_no_check: bool = False,
    include_key_usage: bool = True,
    include_basic_constraints: bool = True,
    extra_eku: bool = False,
    valid_after: datetime | None = None,
) -> x509.Certificate:
    builder = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)]))
        .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer)]))
        .public_key(subject_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_TOKEN_TIME - timedelta(days=2))
        .not_valid_after(valid_after or _CHECKED_AT + timedelta(days=365))
    )
    if include_basic_constraints:
        builder = builder.add_extension(
            x509.BasicConstraints(ca=ca, path_length=None), critical=True
        )
    if include_key_usage:
        builder = builder.add_extension(
            _key_usage(
                digital_signature=digital_signature,
                key_cert_sign=key_cert_sign,
                crl_sign=crl_sign,
            ),
            critical=True,
        )
    if eku is not None:
        usages = [eku]
        if extra_eku:
            usages.append(ExtendedKeyUsageOID.SERVER_AUTH)
        builder = builder.add_extension(x509.ExtendedKeyUsage(usages), critical=True)
    if ocsp_no_check:
        builder = builder.add_extension(x509.OCSPNoCheck(), critical=False)
    return builder.sign(issuer_key, hashes.SHA256())


@pytest.fixture(scope="module")
def certificate_path():
    root_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    root = _issue_certificate(
        "Root",
        root_key,
        "Root",
        root_key,
        ca=True,
        key_cert_sign=True,
        crl_sign=True,
    )
    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    leaf = _issue_certificate(
        "TSA",
        leaf_key,
        "Root",
        root_key,
        ca=False,
        eku=ExtendedKeyUsageOID.TIME_STAMPING,
    )
    return {
        "root_key": root_key,
        "root": root,
        "leaf": leaf,
        "path": [leaf, root],
    }


def _crl(
    issuer: x509.Certificate,
    issuer_key: rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey,
) -> x509.CertificateRevocationList:
    return (
        x509.CertificateRevocationListBuilder()
        .issuer_name(issuer.subject)
        .last_update(_CHECKED_AT - timedelta(hours=1))
        .next_update(_CHECKED_AT + timedelta(hours=1))
        .sign(issuer_key, hashes.SHA256())
    )


def _ocsp_response(
    certificate: x509.Certificate,
    issuer: x509.Certificate,
    responder: x509.Certificate,
    responder_key: rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey,
    *,
    status: ocsp.OCSPCertStatus = ocsp.OCSPCertStatus.GOOD,
    reason: x509.ReasonFlags | None = None,
    revoked_at: datetime | None = None,
    this_update: datetime | None = None,
    next_update: datetime | None = None,
    encoding: ocsp.OCSPResponderEncoding = ocsp.OCSPResponderEncoding.HASH,
    include_responder: bool = False,
    omit_next_update: bool = False,
    unknown_critical: bool = False,
) -> ocsp.OCSPResponse:
    builder = (
        ocsp.OCSPResponseBuilder()
        .add_response(
            cert=certificate,
            issuer=issuer,
            algorithm=hashes.SHA256(),
            cert_status=status,
            this_update=this_update or _CHECKED_AT - timedelta(minutes=5),
            next_update=(
                None
                if omit_next_update
                else next_update or _CHECKED_AT + timedelta(minutes=5)
            ),
            revocation_time=revoked_at,
            revocation_reason=reason,
        )
        .responder_id(encoding, responder)
    )
    if include_responder:
        builder = builder.certificates([responder])
    if unknown_critical:
        builder = builder.add_extension(
            x509.UnrecognizedExtension(ObjectIdentifier("1.2.3.4.7"), b"\x05\x00"),
            critical=True,
        )
    return builder.sign(responder_key, hashes.SHA256())


def _evidence(
    *,
    crls: tuple[x509.CertificateRevocationList, ...] = (),
    responses: tuple[ocsp.OCSPResponse, ...] = (),
) -> RevocationEvidence:
    return RevocationEvidence(
        crls=crls,
        ocsp_responses=responses,
        checked_at=_CHECKED_AT,
    )


def test_fresh_issuer_signed_ocsp_covers_leaf(certificate_path):
    response = _ocsp_response(
        certificate_path["leaf"],
        certificate_path["root"],
        certificate_path["root"],
        certificate_path["root_key"],
        encoding=ocsp.OCSPResponderEncoding.NAME,
    )
    assert verify_certificate_path_revocation(
        certificate_path["path"],
        _evidence(responses=(response,)),
        token_time=_TOKEN_TIME,
    )


def test_fresh_ec_issuer_signed_ocsp_covers_leaf():
    root_key = ec.generate_private_key(ec.SECP256R1())
    root = _issue_certificate(
        "EC Root",
        root_key,
        "EC Root",
        root_key,
        ca=True,
        key_cert_sign=True,
        crl_sign=True,
    )
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf = _issue_certificate("EC TSA", leaf_key, "EC Root", root_key, ca=False)
    response = _ocsp_response(leaf, root, root, root_key)
    assert verify_certificate_path_revocation(
        [leaf, root], _evidence(responses=(response,)), token_time=_TOKEN_TIME
    )


def test_ocsp_helper_fail_closed_invariants(certificate_path):
    no_usage_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    no_usage = _issue_certificate(
        "No Usage",
        no_usage_key,
        "Root",
        certificate_path["root_key"],
        ca=False,
        include_key_usage=False,
    )
    assert _key_allows_digital_signature(no_usage)
    assert not _matches_responder_id(
        SimpleNamespace(responder_name=None, responder_key_hash=None), no_usage
    )

    unsupported = SimpleNamespace(
        public_key=lambda: ed25519.Ed25519PrivateKey.generate().public_key()
    )
    response = SimpleNamespace(
        signature=b"invalid",
        tbs_response_bytes=b"payload",
        signature_hash_algorithm=None,
    )
    assert not _verify_ocsp_signature(response, unsupported)


def test_ocsp_rejects_unsuccessful_and_unsupported_extensions(certificate_path):
    unsuccessful = ocsp.OCSPResponseBuilder.build_unsuccessful(
        ocsp.OCSPResponseStatus.UNAUTHORIZED
    )
    assert not verify_certificate_path_revocation(
        certificate_path["path"],
        _evidence(responses=(unsuccessful,)),
        token_time=_TOKEN_TIME,
    )

    for option in ("omit-next-update", "unknown-critical"):
        response = _ocsp_response(
            certificate_path["leaf"],
            certificate_path["root"],
            certificate_path["root"],
            certificate_path["root_key"],
            omit_next_update=option == "omit-next-update",
            unknown_critical=option == "unknown-critical",
        )
        assert not verify_certificate_path_revocation(
            certificate_path["path"],
            _evidence(responses=(response,)),
            token_time=_TOKEN_TIME,
        )


@pytest.mark.parametrize(
    ("status", "reason", "revoked_at", "expected"),
    [
        (
            ocsp.OCSPCertStatus.REVOKED,
            x509.ReasonFlags.affiliation_changed,
            _TOKEN_TIME + timedelta(1),
            True,
        ),
        (
            ocsp.OCSPCertStatus.REVOKED,
            x509.ReasonFlags.affiliation_changed,
            _TOKEN_TIME - timedelta(1),
            False,
        ),
        (
            ocsp.OCSPCertStatus.REVOKED,
            x509.ReasonFlags.key_compromise,
            _TOKEN_TIME + timedelta(1),
            False,
        ),
        (ocsp.OCSPCertStatus.UNKNOWN, None, None, False),
    ],
)
def test_ocsp_status_and_reason_semantics(
    certificate_path, status, reason, revoked_at, expected
):
    response = _ocsp_response(
        certificate_path["leaf"],
        certificate_path["root"],
        certificate_path["root"],
        certificate_path["root_key"],
        status=status,
        reason=reason,
        revoked_at=revoked_at,
    )
    assert (
        verify_certificate_path_revocation(
            certificate_path["path"],
            _evidence(responses=(response,)),
            token_time=_TOKEN_TIME,
        )
        is expected
    )


@pytest.mark.parametrize("variant", ["stale", "future", "bad-signature", "wrong-cert"])
def test_ocsp_rejects_invalid_or_unrelated_evidence(certificate_path, variant):
    certificate = certificate_path["leaf"]
    this_update = _CHECKED_AT - timedelta(minutes=5)
    next_update = _CHECKED_AT + timedelta(minutes=5)
    signer_key = certificate_path["root_key"]
    if variant == "stale":
        next_update = _CHECKED_AT - timedelta(seconds=1)
    elif variant == "future":
        this_update = _CHECKED_AT + timedelta(seconds=1)
    elif variant == "wrong-cert":
        other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        certificate = _issue_certificate(
            "Other Leaf", other_key, "Root", certificate_path["root_key"], ca=False
        )
    response = _ocsp_response(
        certificate,
        certificate_path["root"],
        certificate_path["root"],
        signer_key,
        this_update=this_update,
        next_update=next_update,
    )
    if variant == "bad-signature":
        encoded = bytearray(response.public_bytes(serialization.Encoding.DER))
        encoded[-1] ^= 1
        response = ocsp.load_der_ocsp_response(bytes(encoded))
    assert not verify_certificate_path_revocation(
        certificate_path["path"],
        _evidence(responses=(response,)),
        token_time=_TOKEN_TIME,
    )


def test_delegated_ocsp_responder_with_nocheck(certificate_path):
    responder_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    responder = _issue_certificate(
        "Responder",
        responder_key,
        "Root",
        certificate_path["root_key"],
        ca=False,
        eku=ExtendedKeyUsageOID.OCSP_SIGNING,
        ocsp_no_check=True,
    )
    response = _ocsp_response(
        certificate_path["leaf"],
        certificate_path["root"],
        responder,
        responder_key,
        include_responder=True,
    )
    assert verify_certificate_path_revocation(
        certificate_path["path"],
        _evidence(responses=(response,)),
        token_time=_TOKEN_TIME,
    )


def test_delegated_ocsp_responder_requires_own_status(certificate_path):
    responder_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    responder = _issue_certificate(
        "Responder",
        responder_key,
        "Root",
        certificate_path["root_key"],
        ca=False,
        eku=ExtendedKeyUsageOID.OCSP_SIGNING,
    )
    response = _ocsp_response(
        certificate_path["leaf"],
        certificate_path["root"],
        responder,
        responder_key,
        include_responder=True,
    )
    assert not verify_certificate_path_revocation(
        certificate_path["path"],
        _evidence(responses=(response,)),
        token_time=_TOKEN_TIME,
    )
    responder_crl = _crl(certificate_path["root"], certificate_path["root_key"])
    assert verify_certificate_path_revocation(
        certificate_path["path"],
        _evidence(crls=(responder_crl,), responses=(response,)),
        token_time=_TOKEN_TIME,
    )


def test_delegated_responder_without_key_usage_is_permitted(certificate_path):
    responder_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    responder = _issue_certificate(
        "Responder",
        responder_key,
        "Root",
        certificate_path["root_key"],
        ca=False,
        eku=ExtendedKeyUsageOID.OCSP_SIGNING,
        ocsp_no_check=True,
        include_key_usage=False,
    )
    response = _ocsp_response(
        certificate_path["leaf"],
        certificate_path["root"],
        responder,
        responder_key,
        include_responder=True,
    )
    assert verify_certificate_path_revocation(
        certificate_path["path"],
        _evidence(responses=(response,)),
        token_time=_TOKEN_TIME,
    )


def test_delegated_responder_without_basic_constraints_is_permitted(
    certificate_path,
):
    responder_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    responder = _issue_certificate(
        "Responder",
        responder_key,
        "Root",
        certificate_path["root_key"],
        ca=False,
        eku=ExtendedKeyUsageOID.OCSP_SIGNING,
        ocsp_no_check=True,
        include_basic_constraints=False,
    )
    response = _ocsp_response(
        certificate_path["leaf"],
        certificate_path["root"],
        responder,
        responder_key,
        include_responder=True,
    )
    assert verify_certificate_path_revocation(
        certificate_path["path"],
        _evidence(responses=(response,)),
        token_time=_TOKEN_TIME,
    )


def test_conflicting_revocation_source_wins(certificate_path):
    good_crl = _crl(certificate_path["root"], certificate_path["root_key"])
    revoked_response = _ocsp_response(
        certificate_path["leaf"],
        certificate_path["root"],
        certificate_path["root"],
        certificate_path["root_key"],
        status=ocsp.OCSPCertStatus.REVOKED,
        reason=x509.ReasonFlags.key_compromise,
        revoked_at=_CHECKED_AT,
    )
    assert not verify_certificate_path_revocation(
        certificate_path["path"],
        _evidence(crls=(good_crl,), responses=(revoked_response,)),
        token_time=_TOKEN_TIME,
    )
