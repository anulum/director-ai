# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for license validation pipeline."""

import hashlib
import hmac
import json

import pytest

from director_ai.core.license import (
    LicenseInfo,
    generate_license,
    load_license,
    validate_file,
    validate_key,
)


@pytest.fixture(autouse=True)
def _set_signing_key(monkeypatch):
    monkeypatch.setenv("DIRECTOR_LICENSE_SIGNING_KEY", "test-license-key-for-ci")


def _resign(payload: dict) -> dict:
    signed = dict(payload)
    unsigned = {k: v for k, v in signed.items() if k != "signature"}
    signed["signature"] = hmac.new(
        b"test-license-key-for-ci",
        json.dumps(unsigned, sort_keys=True).encode(),
        hashlib.sha256,
    ).hexdigest()
    return signed


class TestValidateKey:
    _UUID = "550e8400-e29b-41d4-a716-446655440000"

    def test_empty_key(self):
        info = validate_key("")
        assert not info.valid
        assert info.tier == "community"

    def test_no_prefix(self):
        info = validate_key("PRO-abc-123")
        assert not info.valid

    def test_valid_indie(self):
        info = validate_key(f"DAI-INDIE-{self._UUID}")
        assert info.valid
        assert info.tier == "indie"
        assert info.is_commercial

    def test_valid_pro(self):
        info = validate_key(f"DAI-PRO-{self._UUID}")
        assert info.valid
        assert info.tier == "pro"
        assert info.is_commercial

    def test_valid_trial(self):
        info = validate_key(f"DAI-TRIAL-{self._UUID}")
        assert info.valid
        assert info.tier == "trial"
        assert info.is_trial
        assert not info.is_commercial

    def test_valid_enterprise(self):
        info = validate_key(f"DAI-ENTERPRISE-{self._UUID}")
        assert info.valid
        assert info.tier == "enterprise"

    def test_unknown_tier(self):
        info = validate_key("DAI-GOLD-test")
        assert not info.valid
        assert "Unknown tier" in info.message

    def test_too_few_parts(self):
        info = validate_key("DAI-PRO")
        assert not info.valid

    def test_invalid_uuid(self):
        info = validate_key("DAI-PRO-not-a-uuid")
        assert not info.valid
        assert "UUID" in info.message

    def test_braced_uuid_is_rejected_as_noncanonical(self):
        info = validate_key(f"DAI-PRO-{{{self._UUID}}}")
        assert not info.valid
        assert "UUID" in info.message


class TestValidateFile:
    def test_missing_file(self, tmp_path):
        info = validate_file(tmp_path / "nonexistent.json")
        assert not info.valid
        assert "not found" in info.message

    def test_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json")
        info = validate_file(p)
        assert not info.valid

    def test_invalid_signature(self, tmp_path):
        p = tmp_path / "forged.json"
        p.write_text(
            json.dumps(
                {
                    "key": "DAI-PRO-fake",
                    "tier": "pro",
                    "licensee": "Fake Corp",
                    "signature": "wrong",
                }
            )
        )
        info = validate_file(p)
        assert not info.valid
        assert "signature" in info.message.lower()

    def test_valid_generated_file(self, tmp_path):
        data = generate_license("pro", "Test Corp", "test@test.com", days=365)
        p = tmp_path / "license.json"
        p.write_text(json.dumps(data))
        info = validate_file(p)
        assert info.valid
        assert info.tier == "pro"
        assert info.licensee == "Test Corp"
        assert not info.expired

    def test_expired_license(self, tmp_path):
        data = generate_license("indie", "Old Corp", "old@test.com", days=365)
        # Manually set expiry to the past
        data["expires"] = "2020-01-01T00:00:00+00:00"
        # Re-sign
        import hashlib
        import hmac

        payload = {k: v for k, v in data.items() if k != "signature"}
        data["signature"] = hmac.new(
            b"test-license-key-for-ci",
            json.dumps(payload, sort_keys=True).encode(),
            hashlib.sha256,
        ).hexdigest()
        p = tmp_path / "expired.json"
        p.write_text(json.dumps(data))
        info = validate_file(p)
        assert not info.valid
        assert "expired" in info.message.lower()

    def test_malformed_expiry_treated_as_expired(self, tmp_path):
        data = generate_license("pro", "Bad Date", "bad@test.com", days=365)
        data["expires"] = "not-a-date"
        # Re-sign with the malformed expiry
        import hashlib
        import hmac

        payload = {k: v for k, v in data.items() if k != "signature"}
        data["signature"] = hmac.new(
            b"test-license-key-for-ci",
            json.dumps(payload, sort_keys=True).encode(),
            hashlib.sha256,
        ).hexdigest()
        p = tmp_path / "malformed.json"
        p.write_text(json.dumps(data))
        info = validate_file(p)
        # File validates (signature ok) but LicenseInfo.expired returns True
        assert info.expired

    def test_missing_signing_key_fails_closed(self, monkeypatch, tmp_path):
        data = generate_license("pro", "Test Corp", "test@test.com", days=365)
        p = tmp_path / "license.json"
        p.write_text(json.dumps(data))
        monkeypatch.delenv("DIRECTOR_LICENSE_SIGNING_KEY", raising=False)

        info = validate_file(p)

        assert not info.valid
        assert "Signing key not configured" in info.message

    def test_unknown_signed_tier_is_rejected(self, tmp_path):
        data = generate_license("pro", "Test Corp", "test@test.com", days=365)
        data["tier"] = "gold"
        p = tmp_path / "gold.json"
        p.write_text(json.dumps(_resign(data)))

        info = validate_file(p)

        assert not info.valid
        assert "Unknown tier in license: gold" in info.message

    def test_signed_file_rejects_invalid_embedded_key(self, tmp_path):
        data = generate_license("pro", "Test Corp", "test@test.com", days=365)
        data["key"] = "DAI-PRO-not-a-uuid"
        p = tmp_path / "bad-key.json"
        p.write_text(json.dumps(_resign(data)))

        info = validate_file(p)

        assert not info.valid
        assert "Invalid license key in file" in info.message

    def test_signed_file_rejects_key_tier_mismatch(self, tmp_path):
        data = generate_license("pro", "Test Corp", "test@test.com", days=365)
        data["tier"] = "indie"
        p = tmp_path / "tier-mismatch.json"
        p.write_text(json.dumps(_resign(data)))

        info = validate_file(p)

        assert not info.valid
        assert "License tier does not match key tier" in info.message


class TestGenerateLicense:
    def test_generates_valid_structure(self):
        data = generate_license("pro", "Acme", "a@b.com", days=30, deployments=5)
        assert data["tier"] == "pro"
        assert data["licensee"] == "Acme"
        assert data["email"] == "a@b.com"
        assert data["deployments"] == 5
        assert data["key"].startswith("DAI-PRO-")
        assert "signature" in data
        assert data["version"] == "1"

    def test_roundtrip(self, tmp_path):
        data = generate_license("enterprise", "Big Co", "e@big.com", days=365)
        p = tmp_path / "roundtrip.json"
        p.write_text(json.dumps(data))
        info = validate_file(p)
        assert info.valid
        assert info.tier == "enterprise"
        assert info.licensee == "Big Co"
        assert info.is_commercial


class TestLicenseInfo:
    def test_community_default(self):
        info = LicenseInfo()
        assert info.tier == "community"
        assert not info.valid
        assert not info.is_commercial
        assert not info.is_trial
        assert not info.expired


class TestLoadLicense:
    def test_signed_file_env_wins_after_bare_key_warning(self, monkeypatch, tmp_path):
        data = generate_license("enterprise", "Ops", "ops@example.com", days=365)
        p = tmp_path / "license.json"
        p.write_text(json.dumps(data))
        monkeypatch.setenv("DIRECTOR_LICENSE_KEY", data["key"])
        monkeypatch.setenv("DIRECTOR_LICENSE_FILE", str(p))

        info = load_license()

        assert info.valid
        assert info.tier == "enterprise"
        assert info.licensee == "Ops"

    def test_invalid_env_file_falls_back_to_default_file(
        self,
        monkeypatch,
        tmp_path,
    ):
        invalid = tmp_path / "invalid.json"
        invalid.write_text("{}")
        home = tmp_path / "home"
        default_dir = home / ".director-ai"
        default_dir.mkdir(parents=True)
        default_file = default_dir / "license.json"
        default_file.write_text(
            json.dumps(generate_license("pro", "Default", "default@example.com")),
        )
        monkeypatch.setenv("DIRECTOR_LICENSE_FILE", str(invalid))
        monkeypatch.setattr("pathlib.Path.home", lambda: home)

        info = load_license()

        assert info.valid
        assert info.tier == "pro"
        assert info.licensee == "Default"
