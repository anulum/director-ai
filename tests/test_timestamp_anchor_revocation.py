# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RFC 3161 revocation integration tests

"""Exercise offline revocation evidence through timestamp and CLI surfaces."""

from __future__ import annotations

import datetime

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.x509 import ocsp
from test_timestamp_anchor import (
    _audit_db_with_head,
    _head,
    _pinned_anchor,
    _pinned_transport,
    _populated_audit_db,
)
from test_timestamp_anchor import pinned_tsa as pinned_tsa

from director_ai.compliance.anchor_revocation import RevocationEvidence
from director_ai.compliance.timestamp_anchor import AnchorStore, verify_token

pytest_plugins = ("test_timestamp_anchor",)


def _root_crl(pinned_tsa, *, revoke_leaf: bool = False):
    checked_at = datetime.datetime.now(datetime.UTC)
    builder = (
        x509.CertificateRevocationListBuilder()
        .issuer_name(pinned_tsa["root"].subject)
        .last_update(checked_at - datetime.timedelta(hours=1))
        .next_update(checked_at + datetime.timedelta(hours=1))
    )
    if revoke_leaf:
        revoked = (
            x509.RevokedCertificateBuilder()
            .serial_number(pinned_tsa["leaf"].serial_number)
            .revocation_date(checked_at)
            .add_extension(
                x509.CRLReason(x509.ReasonFlags.key_compromise), critical=False
            )
            .build()
        )
        builder = builder.add_revoked_certificate(revoked)
    return builder.sign(pinned_tsa["root_key"], hashes.SHA256()), checked_at


def _evidence(pinned_tsa, *, revoke_leaf: bool = False) -> RevocationEvidence:
    crl, checked_at = _root_crl(pinned_tsa, revoke_leaf=revoke_leaf)
    return RevocationEvidence(crls=(crl,), checked_at=checked_at)


def _root_ocsp(pinned_tsa) -> ocsp.OCSPResponse:
    checked_at = datetime.datetime.now(datetime.UTC)
    return (
        ocsp.OCSPResponseBuilder()
        .add_response(
            cert=pinned_tsa["leaf"],
            issuer=pinned_tsa["root"],
            algorithm=hashes.SHA256(),
            cert_status=ocsp.OCSPCertStatus.GOOD,
            this_update=checked_at - datetime.timedelta(minutes=5),
            next_update=checked_at + datetime.timedelta(minutes=5),
            revocation_time=None,
            revocation_reason=None,
        )
        .responder_id(ocsp.OCSPResponderEncoding.HASH, pinned_tsa["root"])
        .sign(pinned_tsa["root_key"], hashes.SHA256())
    )


def test_verify_token_requires_roots_and_accepts_fresh_crl(pinned_tsa):
    anchor = _pinned_anchor(pinned_tsa, [pinned_tsa["leaf"]])
    evidence = _evidence(pinned_tsa)

    assert not verify_token(anchor, _head(), revocation_evidence=evidence)
    assert verify_token(
        anchor,
        _head(),
        trusted_roots=[pinned_tsa["root"]],
        revocation_evidence=evidence,
    )


def test_verify_token_rejects_key_compromise_even_after_gen_time(pinned_tsa):
    anchor = _pinned_anchor(pinned_tsa, [pinned_tsa["leaf"]])
    assert not verify_token(
        anchor,
        _head(),
        trusted_roots=[pinned_tsa["root"]],
        revocation_evidence=_evidence(pinned_tsa, revoke_leaf=True),
    )


def test_anchor_store_passes_revocation_evidence(tmp_path, pinned_tsa):
    db = _audit_db_with_head(tmp_path)
    anchor = _pinned_anchor(pinned_tsa, [pinned_tsa["leaf"]])
    store = AnchorStore(db)
    store.record(anchor)
    try:
        ok, bad = store.verify_against_chain(
            trusted_roots=[pinned_tsa["root"]],
            revocation_evidence=_evidence(pinned_tsa),
        )
    finally:
        store.close()
    assert ok is True and bad is None


def test_cli_reports_revocation_evidenced_mode(
    tmp_path, monkeypatch, capsys, pinned_tsa
):
    from director_ai.cli import main as cli_main
    from director_ai.compliance import timestamp_anchor as timestamp_module

    db = _populated_audit_db(tmp_path)
    monkeypatch.setattr(
        timestamp_module,
        "_default_transport",
        _pinned_transport(pinned_tsa["leaf_key"], [pinned_tsa["leaf"]]),
    )
    cli_main(["compliance", "anchor", "--db", db, "--tsa-url", "https://x/tsr"])
    capsys.readouterr()

    roots_path = tmp_path / "roots.pem"
    roots_path.write_bytes(pinned_tsa["root"].public_bytes(serialization.Encoding.PEM))
    crl, _ = _root_crl(pinned_tsa)
    crl_path = tmp_path / "root.crl"
    crl_path.write_bytes(crl.public_bytes(serialization.Encoding.PEM))
    ocsp_path = tmp_path / "root.ocsp"
    ocsp_path.write_bytes(
        _root_ocsp(pinned_tsa).public_bytes(serialization.Encoding.DER)
    )

    cli_main(
        [
            "compliance",
            "verify-anchors",
            "--db",
            db,
            "--tsa-roots",
            str(roots_path),
            "--tsa-crl",
            str(crl_path),
            "--tsa-ocsp",
            str(ocsp_path),
        ]
    )
    assert "root-pinned + revocation-evidenced" in capsys.readouterr().out


def test_cli_rejects_revocation_evidence_without_roots(tmp_path, capsys):
    from director_ai.cli import main as cli_main

    db = _populated_audit_db(tmp_path)
    crl_path = tmp_path / "status.crl"
    crl_path.write_bytes(b"not reached without roots")

    try:
        cli_main(
            [
                "compliance",
                "verify-anchors",
                "--db",
                db,
                "--tsa-crl",
                str(crl_path),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("CLI accepted revocation evidence without roots")
    assert "Revocation evidence requires --tsa-roots" in capsys.readouterr().out


def test_cli_reports_invalid_revocation_evidence(tmp_path, capsys, pinned_tsa):
    from director_ai.cli import main as cli_main

    db = _populated_audit_db(tmp_path)
    roots_path = tmp_path / "roots.pem"
    roots_path.write_bytes(pinned_tsa["root"].public_bytes(serialization.Encoding.PEM))
    crl_path = tmp_path / "invalid.crl"
    crl_path.write_bytes(b"not a CRL")

    try:
        cli_main(
            [
                "compliance",
                "verify-anchors",
                "--db",
                db,
                "--tsa-roots",
                str(roots_path),
                "--tsa-crl",
                str(crl_path),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("CLI accepted malformed revocation evidence")
    assert "Revocation evidence invalid" in capsys.readouterr().out
