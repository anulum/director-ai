# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

# Director-Class AI — License Edge Case Tests
"""Multi-angle tests for license validation edge cases.

Covers: special chars, very long keys, unicode keys, empty/binary/huge files,
community tier fallback, parametrised invalid keys, and pipeline performance.
"""

import pytest

from director_ai.core.license import load_license, validate_file, validate_key


@pytest.fixture(autouse=True)
def _set_signing_key(monkeypatch):
    monkeypatch.setenv("DIRECTOR_LICENSE_SIGNING_KEY", "test-license-key-for-ci")


def test_key_with_special_chars():
    info = validate_key("DAI-PRO-abc!@#$%^&*()")
    assert not info.valid


def test_key_very_long():
    info = validate_key("DAI-PRO-" + "a" * 10000)
    assert not info.valid


def test_key_unicode():
    info = validate_key("DAI-PRO-ĂĽnĂŻcĂ¶dĂ©-key")
    assert not info.valid


def test_file_empty(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text("")
    info = validate_file(p)
    assert not info.valid


def test_file_binary(tmp_path):
    p = tmp_path / "binary.json"
    p.write_bytes(b"\x00\x01\x02\x03")
    info = validate_file(p)
    assert not info.valid


def test_file_huge(tmp_path):
    p = tmp_path / "huge.json"
    p.write_text('{"key": "' + "a" * 1_000_000 + '"}')
    info = validate_file(p)
    assert not info.valid  # invalid signature


def test_load_with_no_env(monkeypatch):
    monkeypatch.delenv("DIRECTOR_LICENSE_KEY", raising=False)
    monkeypatch.delenv("DIRECTOR_LICENSE_FILE", raising=False)
    info = load_license()
    assert info.tier == "community"
    assert info.valid


def test_load_with_key_only_falls_back_to_community(monkeypatch):
    monkeypatch.setenv(
        "DIRECTOR_LICENSE_KEY",
        "DAI-PRO-550e8400-e29b-41d4-a716-446655440000",
    )
    monkeypatch.delenv("DIRECTOR_LICENSE_FILE", raising=False)
    info = load_license()
    assert info.tier == "community"
    assert info.valid


@pytest.mark.parametrize(
    "key",
    [
        "",
        "DAI",
        "not-a-valid-key",
        "DAI-PRO-",
        "DAI-PRO-abc!@#",
        "DAI-PRO-" + "x" * 10000,
    ],
)
def test_parametrised_invalid_keys(key):
    info = validate_key(key)
    assert not info.valid


class TestLicensePerformanceDoc:
    """Document license validation performance."""

    def test_validate_key_fast(self):
        import time

        t0 = time.perf_counter()
        for _ in range(1000):
            validate_key("DAI-PRO-test-key")
        per_call_us = (time.perf_counter() - t0) / 1000 * 1_000_000
        assert per_call_us < 100, f"validate_key took {per_call_us:.1f}µs"

    def test_community_tier_is_default(self):
        info = load_license()
        assert info.tier == "community"
        assert info.valid
