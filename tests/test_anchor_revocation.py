# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RFC 3161 anchor revocation evidence tests

"""Real-signature CRL and OCSP tests for timestamp-anchor revocation checks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509 import ocsp
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID, ObjectIdentifier

from director_ai.compliance.anchor_revocation import (
    RevocationEvidence,
    RevocationEvidenceError,
    _crl_issuer_authorized,
    load_revocation_evidence,
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
) -> x509.Certificate:
    builder = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)]))
        .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer)]))
        .public_key(subject_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_TOKEN_TIME - timedelta(days=2))
        .not_valid_after(_CHECKED_AT + timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
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
        builder = builder.add_extension(x509.ExtendedKeyUsage([eku]), critical=True)
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
        "leaf_key": leaf_key,
        "leaf": leaf,
        "path": [leaf, root],
    }


def _crl(
    issuer: x509.Certificate,
    issuer_key: rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey,
    *,
    revoked: x509.Certificate | None = None,
    reason: x509.ReasonFlags | None = None,
    revoked_at: datetime = _TOKEN_TIME,
    invalidity_at: datetime | None = None,
    last_update: datetime = _CHECKED_AT - timedelta(hours=1),
    next_update: datetime = _CHECKED_AT + timedelta(hours=1),
    delta: bool = False,
    unknown_critical: bool = False,
    unknown_entry_critical: bool = False,
    indirect_entry: bool = False,
) -> x509.CertificateRevocationList:
    builder = (
        x509.CertificateRevocationListBuilder()
        .issuer_name(issuer.subject)
        .last_update(last_update)
        .next_update(next_update)
    )
    if revoked is not None:
        entry = (
            x509.RevokedCertificateBuilder()
            .serial_number(revoked.serial_number)
            .revocation_date(revoked_at)
        )
        if reason is not None:
            entry = entry.add_extension(x509.CRLReason(reason), critical=False)
        if invalidity_at is not None:
            entry = entry.add_extension(
                x509.InvalidityDate(invalidity_at), critical=False
            )
        if unknown_entry_critical:
            entry = entry.add_extension(
                x509.UnrecognizedExtension(ObjectIdentifier("1.2.3.4.6"), b"\x05\x00"),
                critical=True,
            )
        if indirect_entry:
            entry = entry.add_extension(
                x509.CertificateIssuer([x509.DirectoryName(issuer.subject)]),
                critical=True,
            )
        builder = builder.add_revoked_certificate(entry.build())
    if delta:
        builder = builder.add_extension(x509.DeltaCRLIndicator(1), critical=True)
    if unknown_critical:
        builder = builder.add_extension(
            x509.UnrecognizedExtension(ObjectIdentifier("1.2.3.4.5"), b"\x05\x00"),
            critical=True,
        )
    return builder.sign(issuer_key, hashes.SHA256())


def _ocsp_response(
    certificate: x509.Certificate,
    issuer: x509.Certificate,
    responder: x509.Certificate,
    responder_key: rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey,
    *,
    status: ocsp.OCSPCertStatus = ocsp.OCSPCertStatus.GOOD,
    reason: x509.ReasonFlags | None = None,
    revoked_at: datetime | None = None,
    this_update: datetime = _CHECKED_AT - timedelta(minutes=5),
    next_update: datetime = _CHECKED_AT + timedelta(minutes=5),
    encoding: ocsp.OCSPResponderEncoding = ocsp.OCSPResponderEncoding.HASH,
    include_responder: bool = False,
) -> ocsp.OCSPResponse:
    builder = (
        ocsp.OCSPResponseBuilder()
        .add_response(
            cert=certificate,
            issuer=issuer,
            algorithm=hashes.SHA256(),
            cert_status=status,
            this_update=this_update,
            next_update=next_update,
            revocation_time=revoked_at,
            revocation_reason=reason,
        )
        .responder_id(encoding, responder)
    )
    if include_responder:
        builder = builder.certificates([responder])
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


def test_evidence_requires_content_and_aware_clock(certificate_path):
    with pytest.raises(RevocationEvidenceError, match="at least one"):
        RevocationEvidence(checked_at=_CHECKED_AT)
    crl = _crl(certificate_path["root"], certificate_path["root_key"])
    with pytest.raises(RevocationEvidenceError, match="timezone-aware"):
        RevocationEvidence(crls=(crl,), checked_at=datetime(2026, 7, 28))


def test_loader_accepts_pem_der_crl_and_der_ocsp(tmp_path, certificate_path):
    crl = _crl(certificate_path["root"], certificate_path["root_key"])
    response = _ocsp_response(
        certificate_path["leaf"],
        certificate_path["root"],
        certificate_path["root"],
        certificate_path["root_key"],
    )
    pem = tmp_path / "status.pem"
    der = tmp_path / "status.crl"
    ocsp_der = tmp_path / "status.ocsp"
    pem.write_bytes(crl.public_bytes(serialization.Encoding.PEM))
    der.write_bytes(crl.public_bytes(serialization.Encoding.DER))
    ocsp_der.write_bytes(response.public_bytes(serialization.Encoding.DER))

    evidence = load_revocation_evidence(
        crl_paths=[pem, der], ocsp_paths=[ocsp_der], checked_at=_CHECKED_AT
    )
    assert len(evidence.crls) == 2
    assert len(evidence.ocsp_responses) == 1


@pytest.mark.parametrize("payload", [b"", b"not-der"])
def test_loader_rejects_empty_and_invalid_files(tmp_path, payload):
    path = tmp_path / "bad.crl"
    path.write_bytes(payload)
    with pytest.raises(RevocationEvidenceError):
        load_revocation_evidence(crl_paths=[path], checked_at=_CHECKED_AT)


def test_loader_rejects_missing_and_oversized_files(tmp_path):
    with pytest.raises(RevocationEvidenceError, match="not found"):
        load_revocation_evidence(
            crl_paths=[tmp_path / "missing.crl"], checked_at=_CHECKED_AT
        )
    oversized = tmp_path / "oversized.crl"
    oversized.write_bytes(b"x" * (8 * 1024 * 1024 + 1))
    with pytest.raises(RevocationEvidenceError, match="size"):
        load_revocation_evidence(crl_paths=[oversized], checked_at=_CHECKED_AT)


def test_loader_rejects_invalid_ocsp(tmp_path):
    path = tmp_path / "bad.ocsp"
    path.write_bytes(b"not-ocsp")
    with pytest.raises(RevocationEvidenceError, match="OCSP"):
        load_revocation_evidence(ocsp_paths=[path], checked_at=_CHECKED_AT)


def test_fresh_direct_crl_covers_leaf(certificate_path):
    crl = _crl(certificate_path["root"], certificate_path["root_key"])
    assert verify_certificate_path_revocation(
        certificate_path["path"], _evidence(crls=(crl,)), token_time=_TOKEN_TIME
    )


@pytest.mark.parametrize(
    ("reason", "revoked_at", "expected"),
    [
        (x509.ReasonFlags.cessation_of_operation, _TOKEN_TIME - timedelta(1), False),
        (x509.ReasonFlags.cessation_of_operation, _TOKEN_TIME + timedelta(1), True),
        (x509.ReasonFlags.key_compromise, _TOKEN_TIME + timedelta(1), False),
        (None, _TOKEN_TIME + timedelta(1), False),
    ],
)
def test_crl_applies_rfc3161_reason_semantics(
    certificate_path, reason, revoked_at, expected
):
    crl = _crl(
        certificate_path["root"],
        certificate_path["root_key"],
        revoked=certificate_path["leaf"],
        reason=reason,
        revoked_at=revoked_at,
    )
    assert (
        verify_certificate_path_revocation(
            certificate_path["path"], _evidence(crls=(crl,)), token_time=_TOKEN_TIME
        )
        is expected
    )


def test_crl_invalidity_date_precedes_token(certificate_path):
    crl = _crl(
        certificate_path["root"],
        certificate_path["root_key"],
        revoked=certificate_path["leaf"],
        reason=x509.ReasonFlags.superseded,
        revoked_at=_TOKEN_TIME + timedelta(days=2),
        invalidity_at=_TOKEN_TIME - timedelta(days=1),
    )
    assert not verify_certificate_path_revocation(
        certificate_path["path"], _evidence(crls=(crl,)), token_time=_TOKEN_TIME
    )


@pytest.mark.parametrize(
    "variant", ["stale", "future", "delta", "unknown", "bad-signature"]
)
def test_crl_rejects_invalid_evidence(certificate_path, variant):
    last_update = _CHECKED_AT - timedelta(hours=1)
    next_update = _CHECKED_AT + timedelta(hours=1)
    delta = False
    unknown_critical = False
    signing_key = certificate_path["root_key"]
    if variant == "stale":
        next_update = _CHECKED_AT - timedelta(seconds=1)
    elif variant == "future":
        last_update = _CHECKED_AT + timedelta(seconds=1)
    elif variant == "delta":
        delta = True
    elif variant == "unknown":
        unknown_critical = True
    else:
        signing_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    crl = _crl(
        certificate_path["root"],
        signing_key,
        last_update=last_update,
        next_update=next_update,
        delta=delta,
        unknown_critical=unknown_critical,
    )
    assert not verify_certificate_path_revocation(
        certificate_path["path"], _evidence(crls=(crl,)), token_time=_TOKEN_TIME
    )


def test_crl_requires_certified_crl_sign_key():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    issuer = _issue_certificate(
        "No CRL Sign",
        key,
        "No CRL Sign",
        key,
        ca=True,
        key_cert_sign=True,
        crl_sign=False,
    )
    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    leaf = _issue_certificate("Leaf", leaf_key, "No CRL Sign", key, ca=False)
    crl = _crl(issuer, key)
    assert not verify_certificate_path_revocation(
        [leaf, issuer], _evidence(crls=(crl,)), token_time=_TOKEN_TIME
    )


def test_crl_requires_key_usage_extension():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    issuer = _issue_certificate(
        "No KeyUsage",
        key,
        "No KeyUsage",
        key,
        ca=True,
        key_cert_sign=True,
        crl_sign=True,
        include_key_usage=False,
    )
    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    leaf = _issue_certificate("Leaf", leaf_key, "No KeyUsage", key, ca=False)
    assert not verify_certificate_path_revocation(
        [leaf, issuer], _evidence(crls=(_crl(issuer, key),)), token_time=_TOKEN_TIME
    )


def test_legacy_v1_crl_issuer_retains_rfc_exception():
    assert _crl_issuer_authorized(SimpleNamespace(version=x509.Version.v1))


@pytest.mark.parametrize("variant", ["unknown-entry", "indirect-entry"])
def test_crl_rejects_unsupported_entry_extensions(certificate_path, variant):
    crl = _crl(
        certificate_path["root"],
        certificate_path["root_key"],
        revoked=certificate_path["leaf"],
        unknown_entry_critical=variant == "unknown-entry",
        indirect_entry=variant == "indirect-entry",
    )
    assert not verify_certificate_path_revocation(
        certificate_path["path"], _evidence(crls=(crl,)), token_time=_TOKEN_TIME
    )


def test_path_requires_complete_coverage_and_valid_times(certificate_path):
    unrelated_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    unrelated = _issue_certificate(
        "Other",
        unrelated_key,
        "Other",
        unrelated_key,
        ca=True,
        key_cert_sign=True,
        crl_sign=True,
    )
    unrelated_crl = _crl(unrelated, unrelated_key)
    evidence = _evidence(crls=(unrelated_crl,))
    assert not verify_certificate_path_revocation(
        certificate_path["path"], evidence, token_time=_TOKEN_TIME
    )
    assert not verify_certificate_path_revocation(
        [certificate_path["leaf"]], evidence, token_time=_TOKEN_TIME
    )
    assert not verify_certificate_path_revocation(
        certificate_path["path"], evidence, token_time=datetime(2026, 7, 22)
    )
    assert not verify_certificate_path_revocation(
        certificate_path["path"], evidence, token_time=_CHECKED_AT + timedelta(1)
    )


def test_intermediate_path_requires_status_at_every_hop(certificate_path):
    intermediate_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    intermediate = _issue_certificate(
        "Intermediate",
        intermediate_key,
        "Root",
        certificate_path["root_key"],
        ca=True,
        key_cert_sign=True,
        crl_sign=True,
    )
    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    leaf = _issue_certificate(
        "TSA",
        leaf_key,
        "Intermediate",
        intermediate_key,
        ca=False,
        eku=ExtendedKeyUsageOID.TIME_STAMPING,
    )
    root_crl = _crl(certificate_path["root"], certificate_path["root_key"])
    intermediate_crl = _crl(intermediate, intermediate_key)
    path = [leaf, intermediate, certificate_path["root"]]

    assert not verify_certificate_path_revocation(
        path, _evidence(crls=(intermediate_crl,)), token_time=_TOKEN_TIME
    )
    assert verify_certificate_path_revocation(
        path,
        _evidence(crls=(intermediate_crl, root_crl)),
        token_time=_TOKEN_TIME,
    )
