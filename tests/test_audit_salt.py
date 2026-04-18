# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Audit salt loader tests

"""VULN-DAI-003 regression: per-installation audit salt resolution.

Covers env var, env file, legacy fallback with one-shot warning,
error paths, and fingerprint-divergence invariants across salts.
"""

from __future__ import annotations

import hashlib
import hmac
import logging

import pytest

from director_ai.core.safety import audit_salt as audit_salt_mod
from director_ai.core.safety.audit_salt import get_audit_salt, reset_warning_for_tests


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.delenv("DIRECTOR_AUDIT_SALT", raising=False)
    monkeypatch.delenv("DIRECTOR_AUDIT_SALT_FILE", raising=False)
    reset_warning_for_tests()
    yield
    reset_warning_for_tests()


class TestAuditSaltResolution:
    def test_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("DIRECTOR_AUDIT_SALT", "deployment-alpha")
        assert get_audit_salt() == b"deployment-alpha"

    def test_env_file_used(self, monkeypatch, tmp_path):
        f = tmp_path / "salt"
        f.write_text("from-file-salt\n")
        monkeypatch.setenv("DIRECTOR_AUDIT_SALT_FILE", str(f))
        assert get_audit_salt() == b"from-file-salt"

    def test_env_var_overrides_file(self, monkeypatch, tmp_path):
        f = tmp_path / "salt"
        f.write_text("from-file")
        monkeypatch.setenv("DIRECTOR_AUDIT_SALT", "from-env")
        monkeypatch.setenv("DIRECTOR_AUDIT_SALT_FILE", str(f))
        assert get_audit_salt() == b"from-env"

    def test_legacy_fallback_returns_bytes(self):
        result = get_audit_salt()
        assert isinstance(result, bytes)
        assert result  # non-empty
        assert result == b"director-ai-audit-v1"

    def test_legacy_warning_emitted_once(self, caplog):
        caplog.set_level(logging.WARNING, logger="DirectorAI.AuditSalt")
        get_audit_salt()
        get_audit_salt()
        get_audit_salt()
        warnings = [
            r for r in caplog.records if "legacy default audit salt" in r.message
        ]
        assert len(warnings) == 1

    def test_set_env_silences_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("DIRECTOR_AUDIT_SALT", "quiet-prod")
        caplog.set_level(logging.WARNING, logger="DirectorAI.AuditSalt")
        get_audit_salt()
        assert not [
            r for r in caplog.records if "legacy default audit salt" in r.message
        ]

    def test_missing_file_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DIRECTOR_AUDIT_SALT_FILE", str(tmp_path / "nope"))
        with pytest.raises(RuntimeError, match="cannot be read"):
            get_audit_salt()

    def test_empty_file_raises(self, monkeypatch, tmp_path):
        f = tmp_path / "salt"
        f.write_text("   \n")
        monkeypatch.setenv("DIRECTOR_AUDIT_SALT_FILE", str(f))
        with pytest.raises(RuntimeError, match="is empty"):
            get_audit_salt()

    def test_unicode_env_var_encoded_utf8(self, monkeypatch):
        monkeypatch.setenv("DIRECTOR_AUDIT_SALT", "soľ-α-β")
        assert get_audit_salt() == "soľ-α-β".encode()


class TestFingerprintInvariants:
    """Different salts must yield different fingerprints for the same key."""

    @staticmethod
    def _fingerprint(key: str, salt: bytes) -> str:
        return hashlib.sha256(salt + key.encode()).hexdigest()[:16]

    def test_same_salt_same_fingerprint(self, monkeypatch):
        monkeypatch.setenv("DIRECTOR_AUDIT_SALT", "stable")
        salt = get_audit_salt()
        assert self._fingerprint("sk-x", salt) == self._fingerprint("sk-x", salt)

    def test_different_salt_different_fingerprint(self, monkeypatch):
        monkeypatch.setenv("DIRECTOR_AUDIT_SALT", "deployment-a")
        salt_a = get_audit_salt()
        monkeypatch.setenv("DIRECTOR_AUDIT_SALT", "deployment-b")
        salt_b = get_audit_salt()
        assert salt_a != salt_b
        assert self._fingerprint("sk-x", salt_a) != self._fingerprint("sk-x", salt_b)


class TestMiddlewareHashIntegration:
    """VULN-DAI-003 regression: middleware fingerprint must follow the salt."""

    def test_middleware_hash_follows_env_salt(self, monkeypatch):
        from director_ai.middleware.api_key import _hash_key

        monkeypatch.setenv("DIRECTOR_AUDIT_SALT", "tenant-alpha")
        h_alpha = _hash_key("sk-live-abc")
        monkeypatch.setenv("DIRECTOR_AUDIT_SALT", "tenant-beta")
        h_beta = _hash_key("sk-live-abc")
        assert h_alpha != h_beta, "salt rotation must change fingerprint"

    def test_middleware_hash_matches_hmac_recompute(self, monkeypatch):
        from director_ai.middleware.api_key import _hash_key

        monkeypatch.setenv("DIRECTOR_AUDIT_SALT", "explicit-salt")
        expected = hmac.new(b"explicit-salt", b"sk-xyz", "sha512").hexdigest()[:16]
        assert _hash_key("sk-xyz") == expected


def test_module_api_exports():
    assert hasattr(audit_salt_mod, "get_audit_salt")
    assert hasattr(audit_salt_mod, "reset_warning_for_tests")
    assert "get_audit_salt" in audit_salt_mod.__all__
