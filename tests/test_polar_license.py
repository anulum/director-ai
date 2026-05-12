# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for Polar licence validation and offline fallback."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from types import SimpleNamespace

import pytest
import requests

from director_ai.core.license import generate_license, load_license
from director_ai.core.polar_license import (
    activate_polar_key,
    create_polar_customer_portal_session,
    deactivate_polar_key,
    validate_polar_deployment_env,
    validate_polar_key,
    validate_polar_webhook,
)


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
    monkeypatch.delenv("DIRECTOR_AI_POLAR_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("POLAR_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("DIRECTOR_AI_POLAR_ACTIVATION_ID", raising=False)
    monkeypatch.delenv("DIRECTOR_AI_POLAR_BENEFIT_ID", raising=False)
    monkeypatch.delenv("DIRECTOR_AI_POLAR_CONDITIONS", raising=False)
    monkeypatch.delenv("DIRECTOR_AI_POLAR_CUSTOMER_ID", raising=False)
    monkeypatch.delenv("DIRECTOR_AI_POLAR_INCREMENT_USAGE", raising=False)
    monkeypatch.delenv("DIRECTOR_AI_POLAR_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("DIRECTOR_AI_POLAR_VALIDATE_URL", raising=False)
    monkeypatch.delenv("DIRECTOR_AI_POLAR_API_BASE", raising=False)
    monkeypatch.delenv("DIRECTOR_AI_POLAR_DEFAULT_TIER", raising=False)
    monkeypatch.delenv("DIRECTOR_AI_POLAR_BENEFIT_TIERS", raising=False)
    monkeypatch.delenv("DIRECTOR_AI_POLAR_ACTIVATION_LABEL", raising=False)
    monkeypatch.delenv("DIRECTOR_AI_POLAR_WEBHOOK_SECRET", raising=False)


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


def test_polar_server_validation_uses_bearer_token_and_conditions(monkeypatch):
    captured = {}

    def fake_post(url, *, json, timeout, headers):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _Response(
            200,
            {
                "status": "granted",
                "expires_at": "2999-01-01T00:00:00Z",
                "metadata": {"tier": "pro"},
            },
        )

    monkeypatch.setenv("DIRECTOR_AI_POLAR_ACCESS_TOKEN", "polar-token")
    monkeypatch.setenv("DIRECTOR_AI_POLAR_ACTIVATION_ID", "activation-id")
    monkeypatch.setenv("DIRECTOR_AI_POLAR_BENEFIT_ID", "benefit-id")
    monkeypatch.setenv("DIRECTOR_AI_POLAR_CUSTOMER_ID", "customer-id")
    monkeypatch.setenv("DIRECTOR_AI_POLAR_INCREMENT_USAGE", "3")
    monkeypatch.setenv(
        "DIRECTOR_AI_POLAR_CONDITIONS",
        '{"major_version": 1, "edition": "enterprise"}',
    )
    monkeypatch.setattr("director_ai.core.polar_license.requests.post", fake_post)

    info = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")

    assert info.valid
    assert captured["url"] == "https://api.polar.sh/v1/license-keys/validate"
    assert captured["headers"] == {"Authorization": "Bearer polar-token"}
    assert captured["json"] == {
        "key": "polar-key",
        "organization_id": "550e8400-e29b-41d4-a716-446655440000",
        "activation_id": "activation-id",
        "benefit_id": "benefit-id",
        "customer_id": "customer-id",
        "increment_usage": 3,
        "conditions": {"major_version": 1, "edition": "enterprise"},
    }


def test_polar_invalid_conditions_are_rejected_before_network(monkeypatch):
    post = SimpleNamespace(called=False)

    def fake_post(*args, **kwargs):
        post.called = True
        return _Response(200, {"status": "granted"})

    monkeypatch.setenv("DIRECTOR_AI_POLAR_CONDITIONS", "[]")
    monkeypatch.setattr("director_ai.core.polar_license.requests.post", fake_post)

    info = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")

    assert not info.valid
    assert "JSON object" in info.message
    assert post.called is False


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


def test_polar_blank_key_is_rejected_before_network(monkeypatch):
    post = SimpleNamespace(called=False)

    def fake_post(*args, **kwargs):
        post.called = True
        return _Response(200, {"status": "granted"})

    monkeypatch.setattr("director_ai.core.polar_license.requests.post", fake_post)

    info = validate_polar_key("  ", "550e8400-e29b-41d4-a716-446655440000")

    assert not info.valid
    assert "No Polar license key" in info.message
    assert post.called is False


def test_polar_increment_usage_must_be_integer(monkeypatch):
    post = SimpleNamespace(called=False)

    def fake_post(*args, **kwargs):
        post.called = True
        return _Response(200, {"status": "granted"})

    monkeypatch.setenv("DIRECTOR_AI_POLAR_INCREMENT_USAGE", "one")
    monkeypatch.setattr("director_ai.core.polar_license.requests.post", fake_post)

    info = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")

    assert not info.valid
    assert "INCREMENT_USAGE must be int" in info.message
    assert post.called is False


def test_polar_http_statuses_return_actionable_messages(monkeypatch):
    responses = iter(
        [
            _Response(404, {}),
            _Response(422, {}),
            _Response(500, {}),
        ]
    )
    monkeypatch.setattr(
        "director_ai.core.polar_license.requests.post",
        lambda *args, **kwargs: next(responses),
    )

    missing = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")
    rejected = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")
    failed = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")

    assert missing.message == "Polar license key not found"
    assert rejected.message == "Polar license validation request rejected"
    assert failed.message == "Polar validation failed with HTTP 500"


def test_polar_invalid_payload_shape_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "director_ai.core.polar_license.requests.post",
        lambda *args, **kwargs: _Response(200, ["granted"]),
    )

    info = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")

    assert not info.valid
    assert info.message == "Polar validation returned invalid payload"


def test_polar_network_request_exception_is_reported(monkeypatch):
    def raise_connection_error(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(
        "director_ai.core.polar_license.requests.post",
        raise_connection_error,
    )

    info = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")

    assert not info.valid
    assert "Polar validation unavailable: offline" in info.message


@pytest.mark.parametrize(
    ("conditions", "message"),
    [
        ("{", "must be JSON"),
        (json.dumps({f"k{i}": i for i in range(51)}), "at most 50 entries"),
        (json.dumps({"x" * 41: 1}), "keys must be strings up to 40 characters"),
        (json.dumps({"nested": {"bad": True}}), "values must be scalar"),
        (json.dumps({"long": "x" * 501}), "string condition values"),
    ],
)
def test_polar_condition_validation_rejects_bad_policy_inputs(
    monkeypatch, conditions, message
):
    post = SimpleNamespace(called=False)

    def fake_post(*args, **kwargs):
        post.called = True
        return _Response(200, {"status": "granted"})

    monkeypatch.setenv("DIRECTOR_AI_POLAR_CONDITIONS", conditions)
    monkeypatch.setattr("director_ai.core.polar_license.requests.post", fake_post)

    info = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")

    assert not info.valid
    assert message in info.message
    assert post.called is False


def test_polar_timeout_env_and_explicit_endpoint_without_auth_header(monkeypatch):
    captured = {}

    def fake_post(url, *, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _Response(
            200,
            {
                "status": "granted",
                "expires_at": "",
                "metadata": {"license_tier": "pro"},
            },
        )

    monkeypatch.setenv("DIRECTOR_AI_POLAR_ACCESS_TOKEN", "ignored-for-portal")
    monkeypatch.setenv("DIRECTOR_AI_POLAR_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setattr("director_ai.core.polar_license.requests.post", fake_post)

    info = validate_polar_key(
        "polar-key",
        "550e8400-e29b-41d4-a716-446655440000",
        endpoint="https://api.polar.sh/v1/customer-portal/license-keys/validate/",
    )

    assert info.valid
    assert info.tier == "pro"
    assert captured["timeout"] == 0.1
    assert captured["url"].endswith("/validate/")


def test_polar_invalid_timeout_env_falls_back_to_default(monkeypatch):
    captured = {}

    def fake_post(url, *, json, timeout):
        captured["timeout"] = timeout
        return _Response(
            200,
            {"status": "granted", "metadata": {"director_ai_tier": "indie"}},
        )

    monkeypatch.setenv("DIRECTOR_AI_POLAR_TIMEOUT_SECONDS", "slow")
    monkeypatch.setattr("director_ai.core.polar_license.requests.post", fake_post)

    info = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")

    assert info.valid
    assert captured["timeout"] == 5.0


def test_polar_explicit_validate_url_and_legacy_token_use_server_auth(monkeypatch):
    captured = {}

    def fake_post(url, *, json, timeout, headers):
        captured["url"] = url
        captured["headers"] = headers
        return _Response(
            200,
            {"status": "granted", "metadata": {"director_ai_tier": "pro"}},
        )

    monkeypatch.setenv("POLAR_ACCESS_TOKEN", "legacy-token")
    monkeypatch.setenv("DIRECTOR_AI_POLAR_VALIDATE_URL", "https://polar.internal/keys")
    monkeypatch.setattr("director_ai.core.polar_license.requests.post", fake_post)

    info = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")

    assert info.valid
    assert captured == {
        "url": "https://polar.internal/keys",
        "headers": {"Authorization": "Bearer legacy-token"},
    }


def test_polar_expired_and_malformed_expiry_are_invalid(monkeypatch):
    responses = iter(
        [
            _Response(
                200,
                {
                    "status": "granted",
                    "expires_at": "2000-01-01T00:00:00",
                    "metadata": {"director_ai_tier": "pro"},
                },
            ),
            _Response(
                200,
                {
                    "status": "granted",
                    "expires_at": "not-a-date",
                    "metadata": {"director_ai_tier": "pro"},
                },
            ),
        ]
    )
    monkeypatch.setattr(
        "director_ai.core.polar_license.requests.post",
        lambda *args, **kwargs: next(responses),
    )

    expired = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")
    malformed = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")

    assert not expired.valid
    assert "expired on 2000-01-01T00:00:00" in expired.message
    assert not malformed.valid
    assert "expired on not-a-date" in malformed.message


def test_polar_unknown_tier_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "director_ai.core.polar_license.requests.post",
        lambda *args, **kwargs: _Response(
            200,
            {
                "status": "granted",
                "metadata": {"tier": "galactic"},
            },
        ),
    )

    info = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")

    assert not info.valid
    assert info.message == "Unknown Polar license tier: galactic"


def test_polar_tier_resolution_uses_customer_metadata_key_prefix_and_default(
    monkeypatch,
):
    responses = iter(
        [
            _Response(
                200,
                {
                    "status": "granted",
                    "customer": {"metadata": {"license_tier": "enterprise"}},
                    "limit_activations": "unlimited",
                },
            ),
            _Response(200, {"status": "granted"}),
            _Response(200, {"status": "granted"}),
        ]
    )
    monkeypatch.setenv("DIRECTOR_AI_POLAR_DEFAULT_TIER", "pro")
    monkeypatch.setattr(
        "director_ai.core.polar_license.requests.post",
        lambda *args, **kwargs: next(responses),
    )

    customer_metadata = validate_polar_key(
        "polar-key", "550e8400-e29b-41d4-a716-446655440000"
    )
    key_prefix = validate_polar_key(
        "DAI-INDIE-abc123", "550e8400-e29b-41d4-a716-446655440000"
    )
    default = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")

    assert customer_metadata.valid
    assert customer_metadata.tier == "enterprise"
    assert customer_metadata.deployments == 0
    assert key_prefix.valid
    assert key_prefix.tier == "indie"
    assert default.valid
    assert default.tier == "pro"


def test_polar_benefit_map_skips_blank_and_malformed_entries(monkeypatch):
    monkeypatch.setenv(
        "DIRECTOR_AI_POLAR_BENEFIT_TIERS",
        " , malformed, ignored:indie, benefit-pro:pro",
    )
    monkeypatch.setattr(
        "director_ai.core.polar_license.requests.post",
        lambda *args, **kwargs: _Response(
            200,
            {
                "status": "granted",
                "benefit_id": "benefit-pro",
            },
        ),
    )

    info = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")

    assert info.valid
    assert info.tier == "pro"


def test_polar_benefit_map_falls_back_when_unconfigured_or_unmatched(monkeypatch):
    responses = iter(
        [
            _Response(200, {"status": "granted", "benefit_id": "benefit-pro"}),
            _Response(200, {"status": "granted", "benefit_id": "benefit-pro"}),
        ]
    )
    monkeypatch.setenv("DIRECTOR_AI_POLAR_DEFAULT_TIER", "enterprise")
    monkeypatch.setattr(
        "director_ai.core.polar_license.requests.post",
        lambda *args, **kwargs: next(responses),
    )

    unconfigured = validate_polar_key(
        "polar-key", "550e8400-e29b-41d4-a716-446655440000"
    )

    monkeypatch.setenv("DIRECTOR_AI_POLAR_BENEFIT_TIERS", "other=pro")
    unmatched = validate_polar_key("polar-key", "550e8400-e29b-41d4-a716-446655440000")

    assert unconfigured.valid
    assert unconfigured.tier == "enterprise"
    assert unmatched.valid
    assert unmatched.tier == "enterprise"


def test_polar_activation_uses_server_auth_conditions_and_meta(monkeypatch):
    captured = {}

    def fake_post(url, *, json, timeout, headers):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        captured["headers"] = headers
        return _Response(
            200,
            {
                "id": "activation-id",
                "license_key_id": "license-id",
                "label": "node-a",
                "license_key": {
                    "status": "granted",
                    "metadata": {"tier": "pro"},
                    "customer": {"name": "Acme"},
                    "limit_activations": 4,
                },
            },
        )

    monkeypatch.setenv("DIRECTOR_AI_POLAR_ACCESS_TOKEN", "polar-token")
    monkeypatch.setattr("director_ai.core.polar_license.requests.post", fake_post)

    activation = activate_polar_key(
        "polar-key",
        "550e8400-e29b-41d4-a716-446655440000",
        label="node-a",
        conditions={"major_version": 3},
        meta={"host": "worker-1"},
        timeout_seconds=2.5,
    )

    assert activation.activation_id == "activation-id"
    assert activation.license_key_id == "license-id"
    assert activation.label == "node-a"
    assert activation.license.valid
    assert activation.license.tier == "pro"
    assert activation.license.licensee == "Acme"
    assert activation.license.deployments == 4
    assert captured == {
        "url": "https://api.polar.sh/v1/license-keys/activate",
        "json": {
            "key": "polar-key",
            "organization_id": "550e8400-e29b-41d4-a716-446655440000",
            "label": "node-a",
            "conditions": {"major_version": 3},
            "meta": {"host": "worker-1"},
        },
        "timeout": 2.5,
        "headers": {"Authorization": "Bearer polar-token"},
    }


def test_polar_deactivation_accepts_204_and_requires_activation_id(monkeypatch):
    captured = {}

    def fake_post(url, *, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _Response(204, {})

    monkeypatch.setattr("director_ai.core.polar_license.requests.post", fake_post)

    result = deactivate_polar_key(
        "polar-key",
        "activation-id",
        "550e8400-e29b-41d4-a716-446655440000",
        timeout_seconds=1.5,
    )

    assert result.valid
    assert result.message == "Polar activation deactivated"
    assert captured == {
        "url": "https://api.polar.sh/v1/customer-portal/license-keys/deactivate",
        "json": {
            "key": "polar-key",
            "organization_id": "550e8400-e29b-41d4-a716-446655440000",
            "activation_id": "activation-id",
        },
        "timeout": 1.5,
    }

    missing = deactivate_polar_key(
        "polar-key", "", "550e8400-e29b-41d4-a716-446655440000"
    )
    assert not missing.valid
    assert "activation_id" in missing.message


def test_polar_customer_portal_session_uses_org_token(monkeypatch):
    captured = {}

    def fake_post(url, *, json, timeout, headers):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _Response(
            201,
            {
                "customer_portal_url": "https://polar.sh/acme/portal/session",
                "token": "customer-session-token",
                "customer_id": "customer-id",
                "expires_at": "2999-01-01T00:00:00Z",
            },
        )

    monkeypatch.setenv("DIRECTOR_AI_POLAR_ACCESS_TOKEN", "polar-token")
    monkeypatch.setattr("director_ai.core.polar_license.requests.post", fake_post)

    session = create_polar_customer_portal_session(
        customer_external_id="tenant-42",
        timeout_seconds=3.0,
    )

    assert session.customer_portal_url == "https://polar.sh/acme/portal/session"
    assert session.token == "customer-session-token"
    assert session.customer_id == "customer-id"
    assert captured == {
        "url": "https://api.polar.sh/v1/customer-sessions/",
        "json": {"external_customer_id": "tenant-42"},
        "headers": {"Authorization": "Bearer polar-token"},
        "timeout": 3.0,
    }


def test_polar_webhook_validation_uses_standard_webhooks(monkeypatch):
    secret = base64.b64encode(b"webhook-secret-32-bytes-minimum").decode()
    body = b'{"type":"license_key.updated","data":{"id":"lk_1"}}'
    timestamp = str(int(time.time()))
    signed = b"msg_1." + timestamp.encode() + b"." + body
    digest = hmac.new(base64.b64decode(secret), signed, hashlib.sha256).digest()
    headers = {
        "webhook-id": "msg_1",
        "webhook-timestamp": timestamp,
        "webhook-signature": "v1," + base64.b64encode(digest).decode(),
    }

    event = validate_polar_webhook(body, headers, "whsec_" + secret)

    assert event.webhook_id == "msg_1"
    assert event.event_type == "license_key.updated"
    assert event.payload["data"] == {"id": "lk_1"}

    headers["webhook-signature"] = "v1,invalid"
    with pytest.raises(ValueError, match="Invalid Polar webhook signature"):
        validate_polar_webhook(body, headers, "whsec_" + secret)

    headers["webhook-signature"] = "v1," + base64.b64encode(digest).decode()
    headers["webhook-timestamp"] = str(int(time.time()) - 1_000)
    with pytest.raises(ValueError, match="timestamp outside tolerance"):
        validate_polar_webhook(body, headers, "whsec_" + secret)


def test_polar_deployment_env_validation_surfaces_operational_gaps(monkeypatch):
    report = validate_polar_deployment_env()

    assert not report.ready
    assert "DIRECTOR_LICENSE_KEY is not configured" in report.errors
    assert "DIRECTOR_AI_POLAR_ORG_ID is not configured" in report.errors
    assert "DIRECTOR_AI_POLAR_WEBHOOK_SECRET is not configured" in report.warnings

    monkeypatch.setenv("DIRECTOR_LICENSE_KEY", "polar-key")
    monkeypatch.setenv(
        "DIRECTOR_AI_POLAR_ORG_ID", "550e8400-e29b-41d4-a716-446655440000"
    )
    monkeypatch.setenv("DIRECTOR_AI_POLAR_ACCESS_TOKEN", "polar-token")
    monkeypatch.setenv("DIRECTOR_AI_POLAR_ACTIVATION_ID", "activation-id")
    monkeypatch.setenv(
        "DIRECTOR_AI_POLAR_WEBHOOK_SECRET",
        "whsec_" + base64.b64encode(b"webhook-secret-32-bytes-minimum").decode(),
    )

    ready = validate_polar_deployment_env()

    assert ready.ready
    assert ready.errors == []
    assert ready.warnings == []
