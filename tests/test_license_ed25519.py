# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Ed25519 License Signing Tests
"""Tests for asymmetric Ed25519 license signing and verification (SEC-1)."""

from __future__ import annotations

import json
import stat

import pytest

from director_ai.core import license as lic
from tools.generate_license_keypair import generate_keypair
from tools.generate_license_keypair import main as keygen_main


@pytest.fixture
def signing_keypair(monkeypatch):
    """Embed a fresh test public key and expose its private key for signing."""
    private_hex, public_hex = generate_keypair()
    monkeypatch.setattr(lic, "_LICENSE_ED25519_PUBLIC_KEY_HEX", public_hex)
    monkeypatch.setenv("DIRECTOR_LICENSE_PRIVATE_KEY", private_hex)
    monkeypatch.delenv("DIRECTOR_LICENSE_SIGNING_KEY", raising=False)
    return private_hex, public_hex


class TestKeygenTool:
    def test_generate_keypair_returns_two_distinct_32_byte_keys(self):
        private_hex, public_hex = generate_keypair()
        assert len(bytes.fromhex(private_hex)) == 32
        assert len(bytes.fromhex(public_hex)) == 32
        assert private_hex != public_hex

    def test_keypair_round_trips_sign_and_verify(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )

        private_hex, public_hex = generate_keypair()
        message = b"licence-payload"
        signature = Ed25519PrivateKey.from_private_bytes(
            bytes.fromhex(private_hex)
        ).sign(message)
        # verify() raises on mismatch; a clean call means the pair is consistent.
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_hex)).verify(
            signature, message
        )

    def test_main_writes_private_key_0600_and_prints_public(self, tmp_path, capsys):
        out = tmp_path / "priv.hex"
        assert keygen_main(["--private-out", str(out)]) == 0
        assert stat.S_IMODE(out.stat().st_mode) == 0o600
        assert len(bytes.fromhex(out.read_text().strip())) == 32
        assert "Public key" in capsys.readouterr().out


class TestEd25519SigningAndVerification:
    def test_generate_signs_with_ed25519_when_private_key_present(
        self, signing_keypair
    ):
        payload = lic.generate_license("pro", "Acme", "a@acme.example")
        assert "ed25519_signature" in payload
        assert "signature" not in payload

    def test_signed_license_validates(self, signing_keypair, tmp_path):
        payload = lic.generate_license("pro", "Acme", "a@acme.example")
        path = tmp_path / "lic.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        info = lic.validate_file(path)

        assert info.valid is True
        assert info.tier == "pro"

    def test_tampered_tier_is_rejected(self, signing_keypair, tmp_path):
        payload = lic.generate_license("pro", "Acme", "a@acme.example")
        payload["tier"] = "enterprise"  # attempt a self-serve tier upgrade
        path = tmp_path / "lic.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        info = lic.validate_file(path)

        assert info.valid is False
        assert "signature" in info.message.lower()

    def test_forged_signature_is_rejected(self, signing_keypair, tmp_path):
        payload = lic.generate_license("pro", "Acme", "a@acme.example")
        payload["ed25519_signature"] = "00" * 64
        path = tmp_path / "lic.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        info = lic.validate_file(path)

        assert info.valid is False

    def test_signature_from_a_different_key_is_rejected(self, tmp_path, monkeypatch):
        # A client with only the public key cannot forge: signing with their own
        # private key fails verification against ANULUM's embedded public key.
        attacker_priv, _attacker_pub = generate_keypair()
        _anulum_priv, anulum_pub = generate_keypair()
        monkeypatch.setattr(lic, "_LICENSE_ED25519_PUBLIC_KEY_HEX", anulum_pub)
        monkeypatch.setenv("DIRECTOR_LICENSE_PRIVATE_KEY", attacker_priv)
        monkeypatch.delenv("DIRECTOR_LICENSE_SIGNING_KEY", raising=False)

        payload = lic.generate_license("enterprise", "Attacker", "x@x.example")
        path = tmp_path / "forged.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        info = lic.validate_file(path)

        assert info.valid is False
        assert "signature" in info.message.lower()


class TestEmbeddedKeyAndBackwardCompatibility:
    def test_signed_license_without_embedded_key_cannot_verify(
        self, tmp_path, monkeypatch
    ):
        private_hex, _public_hex = generate_keypair()
        monkeypatch.setenv("DIRECTOR_LICENSE_PRIVATE_KEY", private_hex)
        monkeypatch.setattr(lic, "_LICENSE_ED25519_PUBLIC_KEY_HEX", "")
        payload = lic.generate_license("pro", "Acme", "a@acme.example")
        path = tmp_path / "lic.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        info = lic.validate_file(path)

        assert info.valid is False
        assert "public key" in info.message.lower()

    def test_legacy_hmac_license_still_validates(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DIRECTOR_LICENSE_PRIVATE_KEY", raising=False)
        monkeypatch.setenv("DIRECTOR_LICENSE_SIGNING_KEY", "legacy-secret")
        payload = lic.generate_license("indie", "Old Corp", "old@x.example")
        assert "signature" in payload
        assert "ed25519_signature" not in payload
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        info = lic.validate_file(path)

        assert info.valid is True
        assert info.tier == "indie"

    def test_default_build_ships_no_embedded_public_key(self):
        # Empty until ANULUM completes the key ceremony (see the module comment);
        # this test guards against an accidental placeholder key being committed.
        assert lic._LICENSE_ED25519_PUBLIC_KEY_HEX == ""
