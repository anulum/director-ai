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

    def test_legacy_hmac_license_validates_only_with_explicit_opt_in(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("DIRECTOR_LICENSE_PRIVATE_KEY", raising=False)
        monkeypatch.setenv("DIRECTOR_LICENSE_SIGNING_KEY", "legacy-secret")
        monkeypatch.setenv("DIRECTOR_LICENSE_ALLOW_LEGACY_HMAC", "1")
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


class TestLegacyHmacDowngradeIsClosed:
    """The HMAC path must not offer a downgrade forge around Ed25519 (SEC-1)."""

    @pytest.fixture
    def legacy_minting_env(self, monkeypatch, tmp_path):
        """Mint a legacy HMAC licence, then drop the migration opt-in."""
        monkeypatch.delenv("DIRECTOR_LICENSE_PRIVATE_KEY", raising=False)
        monkeypatch.setenv("DIRECTOR_LICENSE_SIGNING_KEY", "shared-secret")
        monkeypatch.setenv("DIRECTOR_LICENSE_ALLOW_LEGACY_HMAC", "1")
        payload = lic.generate_license("enterprise", "Forger", "f@x.example")
        path = tmp_path / "hmac.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.delenv("DIRECTOR_LICENSE_ALLOW_LEGACY_HMAC", raising=False)
        return path

    def test_hmac_licence_rejected_by_default(self, legacy_minting_env):
        # Holding the shared HMAC secret must NOT be enough to mint a licence:
        # without the migration opt-in the HMAC branch is closed outright.
        info = lic.validate_file(legacy_minting_env)

        assert info.valid is False
        assert "deprecated" in info.message.lower()
        assert "DIRECTOR_LICENSE_ALLOW_LEGACY_HMAC" in info.message

    def test_stripping_ed25519_signature_cannot_downgrade_to_hmac(
        self, signing_keypair, tmp_path, monkeypatch
    ):
        # Downgrade attack: take a signed licence, strip the Ed25519 signature,
        # re-sign the upgraded payload with the known shared HMAC secret.
        import hashlib
        import hmac as hmac_mod

        payload = lic.generate_license("pro", "Acme", "a@acme.example")
        del payload["ed25519_signature"]
        payload["tier"] = "enterprise"
        payload["key"] = payload["key"].replace("DAI-PRO-", "DAI-ENTERPRISE-")
        secret = b"shared-secret"
        payload["signature"] = hmac_mod.new(
            secret, json.dumps(payload, sort_keys=True).encode(), hashlib.sha256
        ).hexdigest()
        monkeypatch.setenv("DIRECTOR_LICENSE_SIGNING_KEY", "shared-secret")
        monkeypatch.delenv("DIRECTOR_LICENSE_ALLOW_LEGACY_HMAC", raising=False)
        path = tmp_path / "downgraded.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        info = lic.validate_file(path)

        assert info.valid is False
        assert "deprecated" in info.message.lower()

    def test_generate_refuses_hmac_minting_by_default(self, monkeypatch):
        monkeypatch.delenv("DIRECTOR_LICENSE_PRIVATE_KEY", raising=False)
        monkeypatch.delenv("DIRECTOR_LICENSE_ALLOW_LEGACY_HMAC", raising=False)
        monkeypatch.setenv("DIRECTOR_LICENSE_SIGNING_KEY", "shared-secret")

        with pytest.raises(RuntimeError, match="Ed25519"):
            lic.generate_license("pro", "Acme", "a@acme.example")
