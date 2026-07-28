# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — delegated OCSP responder profile tests
# test-surface: approved-protocol-fake
# real-surface-companion: tests/test_anchor_revocation_ocsp.py

"""Fail-closed delegated-responder authorization tests."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509 import ocsp
from cryptography.x509.oid import ExtendedKeyUsageOID
from test_anchor_revocation_ocsp import (
    _CHECKED_AT,
    _TOKEN_TIME,
    _evidence,
    _issue_certificate,
    _ocsp_response,
)

from director_ai.compliance import _anchor_revocation_ocsp as ocsp_module
from director_ai.compliance.anchor_revocation import (
    _EvidenceStatus,
    verify_certificate_path_revocation,
)

pytest_plugins = ("test_anchor_revocation_ocsp",)


def test_issuer_signed_ocsp_deduplicates_embedded_issuer(certificate_path):
    response = _ocsp_response(
        certificate_path["leaf"],
        certificate_path["root"],
        certificate_path["root"],
        certificate_path["root_key"],
        include_responder=True,
    )
    assert verify_certificate_path_revocation(
        certificate_path["path"],
        _evidence(responses=(response,)),
        token_time=_TOKEN_TIME,
    )


@pytest.mark.parametrize(
    "variant", ["expired", "wrong-issuer", "missing-eku", "extra-eku", "no-ds", "ca"]
)
def test_delegated_ocsp_responder_rejects_invalid_profile(certificate_path, variant):
    responder_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    issuer_name = "Root"
    issuer_key = certificate_path["root_key"]
    if variant == "wrong-issuer":
        issuer_name = "Responder"
        issuer_key = responder_key
    responder = _issue_certificate(
        "Responder",
        responder_key,
        issuer_name,
        issuer_key,
        ca=variant == "ca",
        eku=None if variant == "missing-eku" else ExtendedKeyUsageOID.OCSP_SIGNING,
        ocsp_no_check=True,
        digital_signature=variant != "no-ds",
        extra_eku=variant == "extra-eku",
        valid_after=(_TOKEN_TIME + timedelta(days=1) if variant == "expired" else None),
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


def test_ocsp_rejects_ambiguous_and_non_signing_issuer(certificate_path):
    alias = _issue_certificate(
        "Root",
        certificate_path["root_key"],
        "Root",
        certificate_path["root_key"],
        ca=False,
    )
    ambiguous = (
        ocsp.OCSPResponseBuilder()
        .add_response(
            cert=certificate_path["leaf"],
            issuer=certificate_path["root"],
            algorithm=hashes.SHA256(),
            cert_status=ocsp.OCSPCertStatus.GOOD,
            this_update=_CHECKED_AT - timedelta(minutes=5),
            next_update=_CHECKED_AT + timedelta(minutes=5),
            revocation_time=None,
            revocation_reason=None,
        )
        .responder_id(ocsp.OCSPResponderEncoding.NAME, certificate_path["root"])
        .certificates([alias])
        .sign(certificate_path["root_key"], hashes.SHA256())
    )
    assert not verify_certificate_path_revocation(
        certificate_path["path"],
        _evidence(responses=(ambiguous,)),
        token_time=_TOKEN_TIME,
    )

    no_ds_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    no_ds_root = _issue_certificate(
        "No DS Root",
        no_ds_key,
        "No DS Root",
        no_ds_key,
        ca=True,
        key_cert_sign=True,
        crl_sign=True,
        digital_signature=False,
    )
    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    leaf = _issue_certificate("Leaf", leaf_key, "No DS Root", no_ds_key, ca=False)
    response = _ocsp_response(leaf, no_ds_root, no_ds_root, no_ds_key)
    assert not verify_certificate_path_revocation(
        [leaf, no_ds_root], _evidence(responses=(response,)), token_time=_TOKEN_TIME
    )


def test_ocsp_rejects_issuer_expired_at_produced_at():
    root_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    root = _issue_certificate(
        "Expired Root",
        root_key,
        "Expired Root",
        root_key,
        ca=True,
        key_cert_sign=True,
        crl_sign=True,
        valid_after=_TOKEN_TIME + timedelta(days=1),
    )
    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    leaf = _issue_certificate("Leaf", leaf_key, "Expired Root", root_key, ca=False)
    response = _ocsp_response(leaf, root, root, root_key)
    assert not verify_certificate_path_revocation(
        [leaf, root], _evidence(responses=(response,)), token_time=_TOKEN_TIME
    )


def test_ocsp_rejects_unusable_cert_id_hash(monkeypatch, certificate_path):
    class RejectingBuilder:
        def add_certificate(self, *args):
            raise ValueError("unsupported digest")

    monkeypatch.setattr(ocsp_module.ocsp, "OCSPRequestBuilder", RejectingBuilder)
    monkeypatch.setattr(
        ocsp_module, "_has_critical_single_extensions", lambda _response: False
    )
    response = SimpleNamespace(
        response_status=ocsp.OCSPResponseStatus.SUCCESSFUL,
        responses=[
            SimpleNamespace(
                serial_number=certificate_path["leaf"].serial_number,
                hash_algorithm=hashes.SHA256(),
            )
        ],
    )
    status = ocsp_module._ocsp_status(
        certificate_path["leaf"],
        certificate_path["root"],
        response,
        crls=(),
        checked_at=_CHECKED_AT,
        token_time=_TOKEN_TIME,
    )
    assert status is _EvidenceStatus.INVALID


def test_ocsp_rejects_critical_or_malformed_single_extensions(
    monkeypatch, certificate_path
):
    class Extensions(list):
        native = True

    parsed = {
        "responses": [
            {
                "single_extensions": Extensions(
                    [{"critical": SimpleNamespace(native=False)}]
                )
            },
            {
                "single_extensions": Extensions(
                    [{"critical": SimpleNamespace(native=True)}]
                )
            },
        ]
    }
    monkeypatch.setattr(
        ocsp_module.asn1_ocsp.ResponseData, "load", lambda _payload: parsed
    )
    response = SimpleNamespace(
        response_status=ocsp.OCSPResponseStatus.SUCCESSFUL,
        tbs_response_bytes=b"synthetic",
    )
    status = ocsp_module._ocsp_status(
        certificate_path["leaf"],
        certificate_path["root"],
        response,
        crls=(),
        checked_at=_CHECKED_AT,
        token_time=_TOKEN_TIME,
    )
    assert status is _EvidenceStatus.INVALID

    def malformed(_payload):
        raise ValueError("malformed response data")

    monkeypatch.setattr(ocsp_module.asn1_ocsp.ResponseData, "load", malformed)
    status = ocsp_module._ocsp_status(
        certificate_path["leaf"],
        certificate_path["root"],
        response,
        crls=(),
        checked_at=_CHECKED_AT,
        token_time=_TOKEN_TIME,
    )
    assert status is _EvidenceStatus.INVALID


def test_ocsp_rejects_revoked_status_without_time(monkeypatch, certificate_path):
    class MatchingBuilder:
        def add_certificate(self, *args):
            return self

        def build(self):
            return SimpleNamespace(issuer_name_hash=b"name", issuer_key_hash=b"key")

    monkeypatch.setattr(ocsp_module.ocsp, "OCSPRequestBuilder", MatchingBuilder)
    monkeypatch.setattr(
        ocsp_module, "_has_critical_single_extensions", lambda _response: False
    )
    monkeypatch.setattr(
        ocsp_module, "_ocsp_signer", lambda *args, **kwargs: certificate_path["root"]
    )
    matching = SimpleNamespace(
        serial_number=certificate_path["leaf"].serial_number,
        hash_algorithm=hashes.SHA256(),
        issuer_name_hash=b"name",
        issuer_key_hash=b"key",
        this_update_utc=_CHECKED_AT - timedelta(minutes=5),
        next_update_utc=_CHECKED_AT + timedelta(minutes=5),
        certificate_status=ocsp.OCSPCertStatus.REVOKED,
        revocation_time_utc=None,
    )
    wrong_hash = SimpleNamespace(
        serial_number=certificate_path["leaf"].serial_number,
        hash_algorithm=hashes.SHA256(),
        issuer_name_hash=b"wrong",
        issuer_key_hash=b"wrong",
    )
    response = SimpleNamespace(
        response_status=ocsp.OCSPResponseStatus.SUCCESSFUL,
        responses=[SimpleNamespace(serial_number=-1), wrong_hash, matching],
        extensions=[],
        produced_at_utc=_CHECKED_AT - timedelta(minutes=1),
    )
    status = ocsp_module._ocsp_status(
        certificate_path["leaf"],
        certificate_path["root"],
        response,
        crls=(),
        checked_at=_CHECKED_AT,
        token_time=_TOKEN_TIME,
    )
    assert status is _EvidenceStatus.INVALID
