# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real-surface tests for RFC 3161 audit-chain timestamp anchoring.

A synthetic Timestamp Authority is minted in-process (self-signed RSA and EC
certificates, real CMS ``SignedData`` over a real ``TSTInfo``) so the anchorer,
verifier and store are exercised end-to-end with no network. Transports are
injected, not mocked.
"""

from __future__ import annotations

import builtins
import datetime
import hashlib
import sqlite3

import pytest
from asn1crypto import cms, tsp
from asn1crypto import x509 as a_x509
from cryptography import x509 as c_x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from cryptography.x509.oid import NameOID

from director_ai.compliance.audit_log import AuditEntry, AuditLog
from director_ai.compliance.timestamp_anchor import (
    AnchorDependencyError,
    AnchorError,
    AnchorStore,
    ImprintMismatchError,
    Rfc3161Anchorer,
    TimestampAnchor,
    TsaResponseError,
    TsaUnreachableError,
    _default_transport,
    _require_asn1crypto,
    try_anchor_chain_head,
    verify_token,
)

_GEN_TIME = datetime.datetime(2026, 7, 22, 3, 30, tzinfo=datetime.UTC)


def _make_cert(key, public_key):
    name = c_x509.Name([c_x509.NameAttribute(NameOID.COMMON_NAME, "Test TSA")])
    base = datetime.datetime(2026, 7, 22, tzinfo=datetime.UTC)
    # Ed25519 certificates are signed with algorithm=None; RSA/EC use a hash.
    algorithm = None if isinstance(key, ed25519.Ed25519PrivateKey) else hashes.SHA256()
    return (
        c_x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(public_key)
        .serial_number(c_x509.random_serial_number())
        .not_valid_before(base - datetime.timedelta(days=1))
        .not_valid_after(base + datetime.timedelta(days=3650))
        .sign(key, algorithm)
    )


@pytest.fixture(scope="module")
def rsa_tsa():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, _make_cert(key, key.public_key())


@pytest.fixture(scope="module")
def ec_tsa():
    key = ec.generate_private_key(ec.SECP256R1())
    return key, _make_cert(key, key.public_key())


@pytest.fixture(scope="module")
def ed25519_tsa():
    key = ed25519.Ed25519PrivateKey.generate()
    return key, _make_cert(key, key.public_key())


def _mint(
    imprint,
    tsa,
    *,
    serial=42,
    status="granted",
    content_type="tst_info",
    signed_content_type=None,
    algo="sha256",
    wrong_message_digest=False,
    use_ec=False,
    use_ed25519=False,
):
    """Return DER of a TimeStampResp over ``imprint`` from the synthetic TSA."""
    key, cert = tsa
    tst = tsp.TSTInfo(
        {
            "version": "v1",
            "policy": "1.2.3.4.1",
            "message_imprint": tsp.MessageImprint(
                {"hash_algorithm": {"algorithm": algo}, "hashed_message": imprint}
            ),
            "serial_number": serial,
            "gen_time": _GEN_TIME,
        }
    )
    tst_der = tst.dump()
    a_cert = a_x509.Certificate.load(cert.public_bytes(serialization.Encoding.DER))
    digest_value = (
        b"\x00" * 32 if wrong_message_digest else hashlib.sha256(tst_der).digest()
    )
    attr_content_type = (
        content_type if signed_content_type is None else signed_content_type
    )
    signed_attrs = cms.CMSAttributes(
        [
            cms.CMSAttribute({"type": "content_type", "values": [attr_content_type]}),
            cms.CMSAttribute({"type": "message_digest", "values": [digest_value]}),
        ]
    )
    to_sign = b"\x31" + signed_attrs.dump()[1:]
    if use_ed25519:
        signature = key.sign(to_sign)
        sig_algo = "ed25519"
    elif use_ec:
        signature = key.sign(to_sign, ec.ECDSA(hashes.SHA256()))
        sig_algo = "sha256_ecdsa"
    else:
        signature = key.sign(to_sign, padding.PKCS1v15(), hashes.SHA256())
        sig_algo = "rsassa_pkcs1v15"
    signer = cms.SignerInfo(
        {
            "version": "v1",
            "sid": cms.SignerIdentifier(
                {
                    "issuer_and_serial_number": cms.IssuerAndSerialNumber(
                        {"issuer": a_cert.issuer, "serial_number": a_cert.serial_number}
                    )
                }
            ),
            "digest_algorithm": {"algorithm": "sha256"},
            "signed_attrs": signed_attrs,
            "signature_algorithm": {"algorithm": sig_algo},
            "signature": signature,
        }
    )
    signed_data = cms.SignedData(
        {
            "version": "v3",
            "digest_algorithms": [{"algorithm": "sha256"}],
            "encap_content_info": {
                "content_type": content_type,
                "content": tst if content_type == "tst_info" else tst_der,
            },
            "certificates": [a_cert],
            "signer_infos": [signer],
        }
    )
    token = cms.ContentInfo({"content_type": "signed_data", "content": signed_data})
    return tsp.TimeStampResp(
        {"status": {"status": status}, "time_stamp_token": token}
    ).dump()


def _transport_for(tsa, **mint_kw):
    """A transport that mints a response over whatever imprint the request carries."""

    def transport(request_der, url, timeout):
        request = tsp.TimeStampReq.load(request_der)
        imprint = request["message_imprint"]["hashed_message"].native
        return _mint(imprint, tsa, **mint_kw)

    return transport


def _head(seed: bytes = b"chain-head") -> str:
    return hashlib.sha256(seed).hexdigest()


# --------------------------------------------------------------------------- #
# build_request
# --------------------------------------------------------------------------- #
def test_build_request_round_trips(rsa_tsa):
    anchorer = Rfc3161Anchorer("https://tsa.example/tsr")
    imprint = hashlib.sha256(b"x").digest()
    der = anchorer.build_request(imprint)
    request = tsp.TimeStampReq.load(der)
    assert request["version"].native == "v1"
    assert request["message_imprint"]["hash_algorithm"]["algorithm"].native == "sha256"
    assert request["message_imprint"]["hashed_message"].native == imprint
    assert request["cert_req"].native is True
    assert isinstance(request["nonce"].native, int)


# --------------------------------------------------------------------------- #
# submit
# --------------------------------------------------------------------------- #
def test_submit_happy_path(rsa_tsa):
    head = _head()
    anchorer = Rfc3161Anchorer(
        "https://tsa.example/tsr", transport=_transport_for(rsa_tsa, serial=99)
    )
    anchor = anchorer.submit(head)
    assert anchor.anchored_hash == head
    assert anchor.serial_number == 99
    assert anchor.gen_time == pytest.approx(_GEN_TIME.timestamp())
    assert anchor.tsa_url == "https://tsa.example/tsr"
    assert anchor.imprint_sha256 == hashlib.sha256(bytes.fromhex(head)).hexdigest()
    assert anchor.token_der


def test_submit_transport_failure_raises_unreachable(rsa_tsa):
    def offline(request_der, url, timeout):
        raise OSError("network down")

    anchorer = Rfc3161Anchorer("https://tsa.example/tsr", transport=offline)
    with pytest.raises(TsaUnreachableError):
        anchorer.submit(_head())


def test_submit_unparsable_response_raises(rsa_tsa):
    def garbage(request_der, url, timeout):
        return b"not-a-der-response"

    anchorer = Rfc3161Anchorer("https://tsa.example/tsr", transport=garbage)
    with pytest.raises(TsaResponseError):
        anchorer.submit(_head())


def test_submit_non_granted_status_raises(rsa_tsa):
    anchorer = Rfc3161Anchorer(
        "https://tsa.example/tsr",
        transport=_transport_for(rsa_tsa, status="rejection"),
    )
    with pytest.raises(TsaResponseError):
        anchorer.submit(_head())


def test_submit_imprint_mismatch_raises(rsa_tsa):
    # Transport mints over a DIFFERENT imprint than the request carried.
    def wrong(request_der, url, timeout):
        return _mint(hashlib.sha256(b"unrelated").digest(), rsa_tsa)

    anchorer = Rfc3161Anchorer("https://tsa.example/tsr", transport=wrong)
    with pytest.raises(ImprintMismatchError):
        anchorer.submit(_head())


def test_submit_wrong_algorithm_raises(rsa_tsa):
    def wrong_algo(request_der, url, timeout):
        request = tsp.TimeStampReq.load(request_der)
        imprint = request["message_imprint"]["hashed_message"].native
        return _mint(imprint, rsa_tsa, algo="sha1")

    anchorer = Rfc3161Anchorer("https://tsa.example/tsr", transport=wrong_algo)
    with pytest.raises(ImprintMismatchError):
        anchorer.submit(_head())


# --------------------------------------------------------------------------- #
# verify_token
# --------------------------------------------------------------------------- #
def test_verify_token_rsa_positive_and_negative(rsa_tsa):
    head = _head()
    anchor = Rfc3161Anchorer("u", transport=_transport_for(rsa_tsa)).submit(head)
    assert verify_token(anchor, head) is True
    assert verify_token(anchor, _head(b"other")) is False


def test_verify_token_ec_signature(ec_tsa):
    head = _head(b"ec")
    anchor = Rfc3161Anchorer("u", transport=_transport_for(ec_tsa, use_ec=True)).submit(
        head
    )
    assert verify_token(anchor, head) is True


def test_verify_token_rejects_unsupported_key_type(ed25519_tsa):
    # A well-formed token whose TSA cert uses an unsupported key (Ed25519)
    # must fail closed rather than fall through to the RSA path.
    head = _head(b"ed")
    anchor = Rfc3161Anchorer(
        "u", transport=_transport_for(ed25519_tsa, use_ed25519=True)
    ).submit(head)
    assert verify_token(anchor, head) is False


def test_verify_token_rejects_tampered_signature(rsa_tsa):
    head = _head()
    anchor = Rfc3161Anchorer("u", transport=_transport_for(rsa_tsa)).submit(head)
    flipped = bytearray(anchor.token_der)
    flipped[-1] ^= 0xFF
    tampered = TimestampAnchor(
        anchored_hash=anchor.anchored_hash,
        imprint_sha256=anchor.imprint_sha256,
        gen_time=anchor.gen_time,
        serial_number=anchor.serial_number,
        tsa_url=anchor.tsa_url,
        token_der=bytes(flipped),
        created_at=anchor.created_at,
    )
    assert verify_token(tampered, head) is False


def test_verify_token_rejects_wrong_message_digest(rsa_tsa):
    head = _head()
    anchor = Rfc3161Anchorer(
        "u", transport=_transport_for(rsa_tsa, wrong_message_digest=True)
    ).submit(head)
    # imprint + signature verify, but the message-digest attr does not bind eContent.
    assert verify_token(anchor, head) is False


def test_verify_token_rejects_wrong_content_type_attr(rsa_tsa):
    head = _head()
    # Encapsulated content is a real TSTInfo (so submit accepts it), but the
    # signed content-type attribute claims 'data' instead of id-ct-TSTInfo.
    anchor = Rfc3161Anchorer(
        "u", transport=_transport_for(rsa_tsa, signed_content_type="data")
    ).submit(head)
    assert verify_token(anchor, head) is False


def test_submit_rejects_non_tstinfo_content(rsa_tsa):
    anchorer = Rfc3161Anchorer(
        "u", transport=_transport_for(rsa_tsa, content_type="data")
    )
    with pytest.raises(TsaResponseError):
        anchorer.submit(_head())


def test_verify_token_rejects_garbage(rsa_tsa):
    anchor = TimestampAnchor(
        anchored_hash=_head(),
        imprint_sha256="00",
        gen_time=0.0,
        serial_number=1,
        tsa_url="u",
        token_der=b"garbage",
        created_at=0.0,
    )
    assert verify_token(anchor, _head()) is False


# --------------------------------------------------------------------------- #
# AnchorStore
# --------------------------------------------------------------------------- #
def _audit_db_with_head(tmp_path, seed=b"chain-head"):
    """Build an audit DB whose chain head equals SHA-256(seed)."""
    db = str(tmp_path / "audit.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE audit_log (id INTEGER PRIMARY KEY, entry_hash TEXT)")
    conn.execute("INSERT INTO audit_log (entry_hash) VALUES (?)", (_head(seed),))
    conn.commit()
    conn.close()
    return db


def test_store_record_latest_all(tmp_path, rsa_tsa):
    db = _audit_db_with_head(tmp_path)
    anchor = Rfc3161Anchorer("u", transport=_transport_for(rsa_tsa)).submit(_head())
    store = AnchorStore(db)
    assert store.latest() is None
    row_id = store.record(anchor)
    assert row_id > 0
    assert len(store.all()) == 1
    assert store.latest().anchored_hash == _head()
    store.close()


def test_store_verify_against_chain_ok(tmp_path, rsa_tsa):
    db = _audit_db_with_head(tmp_path)
    anchor = Rfc3161Anchorer("u", transport=_transport_for(rsa_tsa)).submit(_head())
    store = AnchorStore(db)
    store.record(anchor)
    ok, bad = store.verify_against_chain()
    assert ok is True and bad is None
    store.close()


def test_store_verify_rejects_head_not_in_chain(tmp_path, rsa_tsa):
    db = _audit_db_with_head(tmp_path)
    other = _head(b"not-in-chain")
    anchor = Rfc3161Anchorer("u", transport=_transport_for(rsa_tsa)).submit(other)
    store = AnchorStore(db)
    store.record(anchor)
    ok, bad = store.verify_against_chain()
    assert ok is False and bad == other
    store.close()


def test_store_verify_rejects_tampered_token(tmp_path, rsa_tsa):
    db = _audit_db_with_head(tmp_path)
    anchor = Rfc3161Anchorer("u", transport=_transport_for(rsa_tsa)).submit(_head())
    store = AnchorStore(db)
    store.record(anchor)
    # Corrupt the stored token bytes directly in the DB, then reopen.
    conn = sqlite3.connect(db)
    conn.execute("UPDATE audit_anchor SET token_der = ?", (b"broken",))
    conn.commit()
    conn.close()
    store2 = AnchorStore(db)
    ok, bad = store2.verify_against_chain()
    assert ok is False and bad == _head()
    store2.close()


def test_store_empty_verifies_true(tmp_path):
    db = _audit_db_with_head(tmp_path)
    store = AnchorStore(db)
    ok, bad = store.verify_against_chain()
    assert ok is True and bad is None
    store.close()


def test_store_operations_after_close(tmp_path, rsa_tsa):
    db = _audit_db_with_head(tmp_path)
    anchor = Rfc3161Anchorer("u", transport=_transport_for(rsa_tsa)).submit(_head())
    store = AnchorStore(db)
    store.close()
    store.close()  # idempotent
    assert store.all() == []
    with pytest.raises(AnchorError):
        store.record(anchor)


# --------------------------------------------------------------------------- #
# try_anchor_chain_head
# --------------------------------------------------------------------------- #
def test_try_anchor_success(tmp_path, rsa_tsa):
    db = _audit_db_with_head(tmp_path)
    store = AnchorStore(db)
    anchorer = Rfc3161Anchorer("u", transport=_transport_for(rsa_tsa))
    anchor = try_anchor_chain_head(_head(), anchorer, store)
    assert anchor is not None
    assert store.latest().anchored_hash == _head()
    store.close()


def test_try_anchor_genesis_and_empty_return_none(tmp_path, rsa_tsa):
    db = _audit_db_with_head(tmp_path)
    store = AnchorStore(db)
    anchorer = Rfc3161Anchorer("u", transport=_transport_for(rsa_tsa))
    assert try_anchor_chain_head("0" * 64, anchorer, store) is None
    assert try_anchor_chain_head("", anchorer, store) is None
    assert store.all() == []
    store.close()


def test_try_anchor_swallows_anchor_error(tmp_path, caplog):
    db = _audit_db_with_head(tmp_path)
    store = AnchorStore(db)

    def offline(request_der, url, timeout):
        raise OSError("down")

    anchorer = Rfc3161Anchorer("u", transport=offline)
    assert try_anchor_chain_head(_head(), anchorer, store) is None
    assert store.all() == []
    store.close()


# --------------------------------------------------------------------------- #
# dependency guard + default transport
# --------------------------------------------------------------------------- #
def test_require_asn1crypto_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "asn1crypto" or name.startswith("asn1crypto."):
            raise ImportError("no asn1crypto")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(AnchorDependencyError):
        _require_asn1crypto()


def test_default_transport_posts_and_reads(monkeypatch):
    captured = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"response-bytes"

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["data"] = request.data
        captured["timeout"] = timeout
        captured["content_type"] = request.headers.get("Content-type")
        return _FakeResponse()

    import urllib.request as urlreq

    monkeypatch.setattr(urlreq, "urlopen", fake_urlopen)
    out = _default_transport(b"req", "https://tsa.example/tsr", 5.0)
    assert out == b"response-bytes"
    assert captured["url"] == "https://tsa.example/tsr"
    assert captured["data"] == b"req"
    assert captured["timeout"] == 5.0
    assert captured["content_type"] == "application/timestamp-query"


# --------------------------------------------------------------------------- #
# AuditLog.current_head + config validation
# --------------------------------------------------------------------------- #
def test_audit_log_current_head(tmp_path):
    log = AuditLog(str(tmp_path / "chain.db"))
    assert log.current_head() == "0" * 64
    log.log(
        AuditEntry(
            prompt="p",
            response="r",
            model="m",
            provider="v",
            score=0.1,
            approved=True,
            verdict_confidence=0.9,
            task_type="qa",
            domain="d",
            latency_ms=1.0,
            timestamp=1.0,
        )
    )
    head = log.current_head()
    assert len(head) == 64 and head != "0" * 64
    log.close()


def test_config_validation_anchor_requires_url():
    from director_ai.core.config import DirectorConfig

    with pytest.raises(ValueError, match="audit_anchor_tsa_url"):
        DirectorConfig(audit_anchor_enabled=True, audit_anchor_tsa_url="")


def test_config_validation_anchor_timeout_positive():
    from director_ai.core.config import DirectorConfig

    with pytest.raises(ValueError, match="audit_anchor_timeout_s"):
        DirectorConfig(audit_anchor_timeout_s=0.0)


def test_config_env_overlay(monkeypatch):
    from director_ai.core.config import DirectorConfig

    monkeypatch.setenv("DIRECTOR_AUDIT_ANCHOR_ENABLED", "true")
    monkeypatch.setenv("DIRECTOR_AUDIT_ANCHOR_TSA_URL", "https://env.tsa/tsr")
    monkeypatch.setenv("DIRECTOR_AUDIT_ANCHOR_TIMEOUT_S", "15")
    cfg = DirectorConfig.from_env()
    assert cfg.audit_anchor_enabled is True
    assert cfg.audit_anchor_tsa_url == "https://env.tsa/tsr"
    assert cfg.audit_anchor_timeout_s == 15.0


# --------------------------------------------------------------------------- #
# CLI: director-ai compliance anchor / verify-anchors
# --------------------------------------------------------------------------- #
def _populated_audit_db(tmp_path):
    """Return an audit DB path whose chain has one sealed entry."""
    db = str(tmp_path / "cli_audit.db")
    log = AuditLog(db, hmac_secret="test-secret")
    log.log(
        AuditEntry(
            prompt="p",
            response="r",
            model="m",
            provider="v",
            score=0.1,
            approved=True,
            verdict_confidence=0.9,
            task_type="qa",
            domain="d",
            latency_ms=1.0,
            timestamp=1.0,
        )
    )
    log.close()
    return db


def _patch_transport(monkeypatch, tsa, **mint_kw):
    from director_ai.compliance import timestamp_anchor as _mod

    monkeypatch.setattr(_mod, "_default_transport", _transport_for(tsa, **mint_kw))


def test_cli_anchor_and_verify(tmp_path, monkeypatch, capsys, rsa_tsa):
    from director_ai.cli import main as cli_main

    db = _populated_audit_db(tmp_path)
    _patch_transport(monkeypatch, rsa_tsa)
    cli_main(["compliance", "anchor", "--db", db, "--tsa-url", "https://x/tsr"])
    assert "Anchored chain head" in capsys.readouterr().out

    cli_main(["compliance", "verify-anchors", "--db", db])
    out = capsys.readouterr().out
    assert "verify against the audit chain" in out


def test_cli_anchor_tsa_url_from_env(tmp_path, monkeypatch, capsys, rsa_tsa):
    from director_ai.cli import main as cli_main

    db = _populated_audit_db(tmp_path)
    _patch_transport(monkeypatch, rsa_tsa)
    monkeypatch.setenv("DIRECTOR_AUDIT_ANCHOR_TSA_URL", "https://env/tsr")
    cli_main(["compliance", "anchor", "--db", db])
    assert "Anchored chain head" in capsys.readouterr().out


def test_cli_anchor_missing_tsa_url_exits(tmp_path, monkeypatch, capsys):
    from director_ai.cli import main as cli_main

    db = _populated_audit_db(tmp_path)
    monkeypatch.delenv("DIRECTOR_AUDIT_ANCHOR_TSA_URL", raising=False)
    with pytest.raises(SystemExit):
        cli_main(["compliance", "anchor", "--db", db])
    assert "No TSA URL" in capsys.readouterr().out


def test_cli_anchor_empty_chain_exits(tmp_path, monkeypatch, capsys, rsa_tsa):
    from director_ai.cli import main as cli_main

    db = str(tmp_path / "empty.db")
    AuditLog(db, hmac_secret="s").close()  # exists but no sealed entries
    _patch_transport(monkeypatch, rsa_tsa)
    with pytest.raises(SystemExit):
        cli_main(["compliance", "anchor", "--db", db, "--tsa-url", "https://x/tsr"])
    assert "empty" in capsys.readouterr().out


def test_cli_anchor_failure_exits(tmp_path, monkeypatch, capsys):
    from director_ai.cli import main as cli_main
    from director_ai.compliance import timestamp_anchor as _mod

    db = _populated_audit_db(tmp_path)

    def offline(request_der, url, timeout):
        raise OSError("down")

    monkeypatch.setattr(_mod, "_default_transport", offline)
    with pytest.raises(SystemExit):
        cli_main(["compliance", "anchor", "--db", db, "--tsa-url", "https://x/tsr"])
    assert "Anchoring failed" in capsys.readouterr().out


def test_cli_verify_anchors_none_recorded(tmp_path, capsys):
    from director_ai.cli import main as cli_main

    db = _populated_audit_db(tmp_path)
    cli_main(["compliance", "verify-anchors", "--db", db])
    assert "No timestamp anchors recorded yet" in capsys.readouterr().out


def test_cli_verify_anchors_failure_exits(tmp_path, monkeypatch, capsys, rsa_tsa):
    from director_ai.cli import main as cli_main

    db = _populated_audit_db(tmp_path)
    _patch_transport(monkeypatch, rsa_tsa)
    cli_main(["compliance", "anchor", "--db", db, "--tsa-url", "https://x/tsr"])
    capsys.readouterr()
    # Corrupt the stored token so verification fails.
    conn = sqlite3.connect(db)
    conn.execute("UPDATE audit_anchor SET token_der = ?", (b"broken",))
    conn.commit()
    conn.close()
    with pytest.raises(SystemExit):
        cli_main(["compliance", "verify-anchors", "--db", db])
    assert "FAILED" in capsys.readouterr().out
