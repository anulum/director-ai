# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for Polar licence validation and offline fallback."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import requests

from director_ai.core.license import generate_license, load_license
from director_ai.core.polar_license import validate_polar_key


class _Response:
    def __init__(self, status_code: int, payload: object):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload


@pytest.fixture(autouse=True)
def _license_env(monkeypatch):
    monkeypatch.setenv("DIRECTOR_LICENSE_SIGNING_KEY", "test-license-key-for-ci")
    monkeypatch.delenv("DIRECTOR_LICENSE_KEY", raising=False)
    monkeypatch.delenv("DIRECTOR_LICENSE_FILE", raising=False)
    monkeypatch.delenv("DIRECTOR_AI_POLAR_ORG_ID", raising=False)
    monkeypatch.delenv("DIRECTOR_AI_POLAR_DEFAULT_TIER", raising=False)
    monkeypatch.delenv("DIRECTOR_AI_POLAR_BENEFIT_TIERS", raising=False)


def test_polar_granted_license_is_commercial(monkeypatch):
    captured = {}

    def fake_post(url, *, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _Response(
            200,
            {
                "key": "polar-key",
                "status": "granted",
                "expires_at": "2999-01-01T00:00:00Z",
                "limit_activations": 3,
                "metadata": {"director_ai_tier": "pro"},
                "customer": {"name": "Acme", "email": "ops@example.com"},
            },
        )

    monkeypatch.setattr("director_ai.core.polar_license.requests.post", fake_post)

    info = validate_polar_key(
        "polar-key",
        "550e8400-e29b-41d4-a716-446655440000",
        timeout_seconds=1.25,
    )

    assert info.valid
    assert info.tier == "pro"
    assert info.is_commercial
    assert info.licensee == "Acme"
    assert info.email == "ops@example.com"
    assert info.deployments == 3
    assert captured["json"] == {
        "key": "polar-key",
        "organization_id": "550e8400-e29b-41d4-a716-446655440000",
    }
    assert captured["timeout"] == 1.25


def test_polar_revoked_license_is_not_valid(monkeypatch):
    monkeypatch.setattr(
        "director_ai.core.polar_license.requests.post",
        lambda *args, **kwargs: _Response(200, {"status": "revoked"}),
    )

    info = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")

    assert not info.valid
    assert "revoked" in info.message


def test_polar_benefit_id_maps_to_tier(monkeypatch):
    monkeypatch.setenv("DIRECTOR_AI_POLAR_BENEFIT_TIERS", "benefit-pro=enterprise")
    monkeypatch.setattr(
        "director_ai.core.polar_license.requests.post",
        lambda *args, **kwargs: _Response(
            200,
            {
                "status": "granted",
                "benefit_id": "benefit-pro",
                "expires_at": "2999-01-01T00:00:00Z",
            },
        ),
    )

    info = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")

    assert info.valid
    assert info.tier == "enterprise"


def test_polar_network_error_falls_back_to_signed_file(monkeypatch, tmp_path):
    def raise_timeout(*args, **kwargs):
        raise requests.Timeout("network unavailable")

    license_file = tmp_path / "license.json"
    license_file.write_text(
        json.dumps(generate_license("pro", "Fallback Corp", "ops@example.com")),
        encoding="utf-8",
    )
    monkeypatch.setenv("DIRECTOR_LICENSE_KEY", "polar-key")
    monkeypatch.setenv(
        "DIRECTOR_AI_POLAR_ORG_ID", "550e8400-e29b-41d4-a716-446655440000"
    )
    monkeypatch.setenv("DIRECTOR_LICENSE_FILE", str(license_file))
    monkeypatch.setattr("director_ai.core.polar_license.requests.post", raise_timeout)

    info = load_license()

    assert info.valid
    assert info.tier == "pro"
    assert info.licensee == "Fallback Corp"


def test_load_license_accepts_polar_before_offline_key_syntax(monkeypatch):
    monkeypatch.setenv("DIRECTOR_LICENSE_KEY", "polar-generated-key")
    monkeypatch.setenv(
        "DIRECTOR_AI_POLAR_ORG_ID", "550e8400-e29b-41d4-a716-446655440000"
    )
    monkeypatch.setattr(
        "director_ai.core.polar_license.requests.post",
        lambda *args, **kwargs: _Response(
            200,
            {
                "status": "granted",
                "expires_at": "2999-01-01T00:00:00Z",
                "customer": {"name": "Polar Customer"},
            },
        ),
    )

    info = load_license()

    assert info.valid
    assert info.tier == "indie"
    assert info.licensee == "Polar Customer"


def test_polar_invalid_json_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "director_ai.core.polar_license.requests.post",
        lambda *args, **kwargs: _Response(200, ValueError("bad json")),
    )

    info = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")

    assert not info.valid
    assert "invalid JSON" in info.message


def test_polar_missing_org_is_rejected(monkeypatch):
    post = SimpleNamespace(called=False)

    def fake_post(*args, **kwargs):
        post.called = True
        return _Response(200, {"status": "granted"})

    monkeypatch.setattr("director_ai.core.polar_license.requests.post", fake_post)

    info = validate_polar_key("polar-key")

    assert not info.valid
    assert "ORG_ID" in info.message
    assert post.called is False
