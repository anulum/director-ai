# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Polar licence validation for commercial deployments."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import requests

from director_ai.core.license import TIERS, LicenseInfo

POLAR_VALIDATE_URL = "https://api.polar.sh/v1/customer-portal/license-keys/validate"
POLAR_SERVER_VALIDATE_URL = "https://api.polar.sh/v1/license-keys/validate"
POLAR_ACTIVATE_PATH = "/license-keys/activate"
POLAR_DEACTIVATE_PATH = "/license-keys/deactivate"
POLAR_CUSTOMER_PORTAL_PREFIX = "/customer-portal"
POLAR_CUSTOMER_SESSIONS_PATH = "/customer-sessions/"
POLAR_DEFAULT_TIMEOUT_SECONDS = 5.0
POLAR_DEFAULT_WEBHOOK_TOLERANCE_SECONDS = 300


@dataclass(frozen=True)
class PolarActivationInfo:
    """Result returned by a Polar license-key activation call."""

    activation_id: str
    license_key_id: str
    label: str
    license: LicenseInfo
    payload: dict[str, Any]


@dataclass(frozen=True)
class PolarOperationResult:
    """Result for Polar operations that do not return a license object."""

    valid: bool
    message: str
    status_code: int = 0
    payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class PolarCustomerPortalSession:
    """Pre-authenticated Polar customer portal session."""

    customer_portal_url: str
    token: str
    customer_id: str
    expires_at: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class PolarWebhookEvent:
    """Verified Polar webhook payload with Standard Webhooks metadata."""

    webhook_id: str
    timestamp: int
    event_type: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class PolarDeploymentReport:
    """Readiness report for production Polar licensing configuration."""

    ready: bool
    errors: list[str]
    warnings: list[str]


def validate_polar_key(
    key: str,
    organization_id: str | None = None,
    *,
    timeout_seconds: float | None = None,
    endpoint: str | None = None,
) -> LicenseInfo:
    """Validate a licence key with Polar's public customer portal endpoint."""

    clean_key = key.strip()
    if not clean_key:
        return LicenseInfo(message="No Polar license key provided")

    org_id = (organization_id or os.environ.get("DIRECTOR_AI_POLAR_ORG_ID", "")).strip()
    if not org_id:
        return LicenseInfo(message="DIRECTOR_AI_POLAR_ORG_ID not configured")

    request_body: dict[str, object] = {
        "key": clean_key,
        "organization_id": org_id,
    }
    activation_id = os.environ.get("DIRECTOR_AI_POLAR_ACTIVATION_ID", "").strip()
    if activation_id:
        request_body["activation_id"] = activation_id

    benefit_id = os.environ.get("DIRECTOR_AI_POLAR_BENEFIT_ID", "").strip()
    if benefit_id:
        request_body["benefit_id"] = benefit_id

    customer_id = os.environ.get("DIRECTOR_AI_POLAR_CUSTOMER_ID", "").strip()
    if customer_id:
        request_body["customer_id"] = customer_id

    increment_usage = os.environ.get("DIRECTOR_AI_POLAR_INCREMENT_USAGE", "").strip()
    if increment_usage:
        try:
            request_body["increment_usage"] = int(increment_usage)
        except ValueError:
            return LicenseInfo(message="DIRECTOR_AI_POLAR_INCREMENT_USAGE must be int")

    conditions_info = _conditions_from_env()
    if not conditions_info.valid:
        return LicenseInfo(message=conditions_info.message)
    if conditions_info.conditions:
        request_body["conditions"] = conditions_info.conditions

    url = endpoint or _validate_url_from_env()
    headers = _auth_headers(url)

    timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else _env_timeout_seconds()
    )

    try:
        if headers:
            response = requests.post(
                url,
                json=request_body,
                timeout=timeout,
                headers=headers,
            )
        else:
            response = requests.post(url, json=request_body, timeout=timeout)
    except requests.RequestException as exc:
        return LicenseInfo(message=f"Polar validation unavailable: {exc}")

    if response.status_code == 404:
        return LicenseInfo(message="Polar license key not found")
    if response.status_code == 422:
        return LicenseInfo(message="Polar license validation request rejected")
    if response.status_code < 200 or response.status_code >= 300:
        return LicenseInfo(
            message=f"Polar validation failed with HTTP {response.status_code}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        return LicenseInfo(message=f"Polar validation returned invalid JSON: {exc}")
    if not isinstance(payload, dict):
        return LicenseInfo(message="Polar validation returned invalid payload")

    return _license_info_from_payload(clean_key, payload)


def activate_polar_key(
    key: str,
    organization_id: str | None = None,
    *,
    label: str | None = None,
    conditions: dict[str, object] | None = None,
    meta: dict[str, object] | None = None,
    timeout_seconds: float | None = None,
    endpoint: str | None = None,
) -> PolarActivationInfo:
    """Activate a Polar license key and return the activation identifier."""

    clean_key = key.strip()
    activation_label = (
        label or os.environ.get("DIRECTOR_AI_POLAR_ACTIVATION_LABEL") or "director-ai"
    ).strip()
    if not clean_key:
        return _activation_error("No Polar license key provided", activation_label)
    if not activation_label:
        return _activation_error("Polar activation label is required", activation_label)

    org_id = (organization_id or os.environ.get("DIRECTOR_AI_POLAR_ORG_ID", "")).strip()
    if not org_id:
        return _activation_error(
            "DIRECTOR_AI_POLAR_ORG_ID not configured", activation_label
        )

    conditions_info = (
        _conditions_from_env()
        if conditions is None
        else _map_info("Polar activation conditions", conditions)
    )
    if not conditions_info.valid:
        return _activation_error(conditions_info.message, activation_label)
    meta_info = _map_info("Polar activation meta", meta or {})
    if not meta_info.valid:
        return _activation_error(meta_info.message, activation_label)

    request_body: dict[str, object] = {
        "key": clean_key,
        "organization_id": org_id,
        "label": activation_label,
    }
    if conditions_info.conditions:
        request_body["conditions"] = conditions_info.conditions
    if meta_info.conditions:
        request_body["meta"] = meta_info.conditions

    url = endpoint or _polar_endpoint(POLAR_ACTIVATE_PATH)
    response = _post_polar(url, request_body, timeout_seconds)
    if isinstance(response, LicenseInfo):
        return _activation_error(response.message, activation_label)
    if response.status_code < 200 or response.status_code >= 300:
        return _activation_error(
            f"Polar activation failed with HTTP {response.status_code}",
            activation_label,
        )

    payload = _response_payload(response)
    if payload is None:
        return _activation_error(
            "Polar activation returned invalid payload", activation_label
        )
    license_payload = payload.get("license_key")
    license_info = _license_info_from_payload(
        clean_key, license_payload if isinstance(license_payload, dict) else payload
    )
    return PolarActivationInfo(
        activation_id=_string_field(payload, "id"),
        license_key_id=_string_field(payload, "license_key_id"),
        label=_string_field(payload, "label") or activation_label,
        license=license_info,
        payload=payload,
    )


def deactivate_polar_key(
    key: str,
    activation_id: str,
    organization_id: str | None = None,
    *,
    timeout_seconds: float | None = None,
    endpoint: str | None = None,
) -> PolarOperationResult:
    """Deactivate a Polar license-key activation."""

    clean_key = key.strip()
    clean_activation = activation_id.strip()
    if not clean_key:
        return PolarOperationResult(False, "No Polar license key provided")
    if not clean_activation:
        return PolarOperationResult(False, "Polar activation_id is required")

    org_id = (organization_id or os.environ.get("DIRECTOR_AI_POLAR_ORG_ID", "")).strip()
    if not org_id:
        return PolarOperationResult(False, "DIRECTOR_AI_POLAR_ORG_ID not configured")

    request_body: dict[str, object] = {
        "key": clean_key,
        "organization_id": org_id,
        "activation_id": clean_activation,
    }
    url = endpoint or _polar_endpoint(POLAR_DEACTIVATE_PATH)
    response = _post_polar(url, request_body, timeout_seconds)
    if isinstance(response, LicenseInfo):
        return PolarOperationResult(False, response.message)
    if response.status_code == 204:
        return PolarOperationResult(True, "Polar activation deactivated", 204)
    payload = _response_payload(response) or {}
    return PolarOperationResult(
        False,
        f"Polar deactivation failed with HTTP {response.status_code}",
        response.status_code,
        payload,
    )


def create_polar_customer_portal_session(
    *,
    customer_id: str | None = None,
    customer_external_id: str | None = None,
    timeout_seconds: float | None = None,
    endpoint: str | None = None,
) -> PolarCustomerPortalSession:
    """Create a pre-authenticated Polar customer portal session."""

    token = _polar_access_token()
    if not token:
        raise ValueError("DIRECTOR_AI_POLAR_ACCESS_TOKEN is required")
    if bool(customer_id) == bool(customer_external_id):
        raise ValueError("provide exactly one of customer_id or customer_external_id")

    request_body: dict[str, object]
    if customer_id:
        request_body = {"customer_id": customer_id.strip()}
    elif customer_external_id:
        request_body = {"external_customer_id": customer_external_id.strip()}
    else:
        raise ValueError("provide exactly one of customer_id or customer_external_id")
    if not next(iter(request_body.values())):
        raise ValueError("Polar customer identifier must be non-empty")

    url = endpoint or _polar_api_url(POLAR_CUSTOMER_SESSIONS_PATH)
    response = _post_polar(url, request_body, timeout_seconds)
    if isinstance(response, LicenseInfo):
        raise ValueError(response.message)
    if response.status_code < 200 or response.status_code >= 300:
        raise ValueError(
            f"Polar customer session failed with HTTP {response.status_code}"
        )
    payload = _response_payload(response)
    if payload is None:
        raise ValueError("Polar customer session returned invalid payload")
    portal_url = _string_field(payload, "customer_portal_url")
    if not portal_url:
        raise ValueError("Polar customer session response missing customer_portal_url")
    return PolarCustomerPortalSession(
        customer_portal_url=portal_url,
        token=_string_field(payload, "token"),
        customer_id=_string_field(payload, "customer_id"),
        expires_at=_string_field(payload, "expires_at"),
        payload=payload,
    )


def validate_polar_webhook(
    body: bytes | str,
    headers: Mapping[str, str],
    secret: str | None = None,
    *,
    tolerance_seconds: int = POLAR_DEFAULT_WEBHOOK_TOLERANCE_SECONDS,
    now: float | None = None,
) -> PolarWebhookEvent:
    """Validate a Polar webhook using the Standard Webhooks HMAC scheme."""

    raw_body = body.encode("utf-8") if isinstance(body, str) else body
    webhook_id = _header_value(headers, "webhook-id")
    timestamp_raw = _header_value(headers, "webhook-timestamp")
    signature_header = _header_value(headers, "webhook-signature")
    if not webhook_id or not timestamp_raw or not signature_header:
        raise ValueError("Missing Polar webhook signature headers")
    if "." in webhook_id or "." in timestamp_raw:
        raise ValueError("Invalid Polar webhook signature metadata")
    try:
        timestamp = int(timestamp_raw)
    except ValueError as exc:
        raise ValueError("Invalid Polar webhook timestamp") from exc
    clock = time.time() if now is None else now
    if abs(clock - timestamp) > tolerance_seconds:
        raise ValueError("Polar webhook timestamp outside tolerance")

    secret_bytes = _decode_webhook_secret(
        secret or os.environ.get("DIRECTOR_AI_POLAR_WEBHOOK_SECRET", "")
    )
    signed = b".".join([webhook_id.encode(), timestamp_raw.encode(), raw_body])
    expected = hmac.new(secret_bytes, signed, hashlib.sha256).digest()
    if not _signature_matches(signature_header, expected):
        raise ValueError("Invalid Polar webhook signature")

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Polar webhook JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Invalid Polar webhook payload")
    return PolarWebhookEvent(
        webhook_id=webhook_id,
        timestamp=timestamp,
        event_type=_string_field(payload, "type"),
        payload=payload,
    )


def validate_polar_deployment_env() -> PolarDeploymentReport:
    """Check Polar-related environment variables before production deployment."""

    errors: list[str] = []
    warnings: list[str] = []
    if not os.environ.get("DIRECTOR_LICENSE_KEY", "").strip():
        errors.append("DIRECTOR_LICENSE_KEY is not configured")
    if not os.environ.get("DIRECTOR_AI_POLAR_ORG_ID", "").strip():
        errors.append("DIRECTOR_AI_POLAR_ORG_ID is not configured")

    if not _polar_access_token():
        warnings.append(
            "DIRECTOR_AI_POLAR_ACCESS_TOKEN is not configured; "
            "server-side validation, activations, and portal sessions are unavailable"
        )
    if not os.environ.get("DIRECTOR_AI_POLAR_ACTIVATION_ID", "").strip():
        warnings.append(
            "DIRECTOR_AI_POLAR_ACTIVATION_ID is not configured; "
            "device activation limits will not be bound to this deployment"
        )
    if not os.environ.get("DIRECTOR_AI_POLAR_WEBHOOK_SECRET", "").strip():
        warnings.append("DIRECTOR_AI_POLAR_WEBHOOK_SECRET is not configured")
    else:
        try:
            _decode_webhook_secret(os.environ["DIRECTOR_AI_POLAR_WEBHOOK_SECRET"])
        except ValueError as exc:
            errors.append(str(exc))

    conditions_info = _conditions_from_env()
    if not conditions_info.valid:
        errors.append(conditions_info.message)
    increment_usage = os.environ.get("DIRECTOR_AI_POLAR_INCREMENT_USAGE", "").strip()
    if increment_usage:
        try:
            int(increment_usage)
        except ValueError:
            errors.append("DIRECTOR_AI_POLAR_INCREMENT_USAGE must be int")
    return PolarDeploymentReport(ready=not errors, errors=errors, warnings=warnings)


def _env_timeout_seconds() -> float:
    """Return the configured Polar HTTP timeout with a safe floor."""
    raw = os.environ.get("DIRECTOR_AI_POLAR_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return POLAR_DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except ValueError:
        return POLAR_DEFAULT_TIMEOUT_SECONDS
    return max(0.1, timeout)


def _polar_api_url(path: str) -> str:
    """Build an absolute Polar API URL for a relative path."""
    base = os.environ.get("DIRECTOR_AI_POLAR_API_BASE", "https://api.polar.sh/v1")
    return base.rstrip("/") + "/" + path.lstrip("/")


def _polar_endpoint(path: str) -> str:
    """Return server or customer-portal endpoint based on auth mode."""
    if _polar_access_token():
        return _polar_api_url(path)
    return _polar_api_url(POLAR_CUSTOMER_PORTAL_PREFIX + path)


def _validate_url_from_env() -> str:
    """Return the effective Polar licence validation endpoint."""
    explicit = os.environ.get("DIRECTOR_AI_POLAR_VALIDATE_URL", "").strip()
    if explicit:
        return explicit
    if _polar_access_token():
        return POLAR_SERVER_VALIDATE_URL
    return POLAR_VALIDATE_URL


def _polar_access_token() -> str:
    """Return the configured server-side Polar access token."""
    return (
        os.environ.get("DIRECTOR_AI_POLAR_ACCESS_TOKEN", "").strip()
        or os.environ.get("POLAR_ACCESS_TOKEN", "").strip()
    )


def _auth_headers(url: str) -> dict[str, str]:
    """Return bearer auth headers when endpoint mode requires them."""
    token = _polar_access_token()
    if not token:
        return {}
    if url.rstrip("/") == POLAR_VALIDATE_URL.rstrip("/"):
        return {}
    return {"Authorization": f"Bearer {token}"}


class _ConditionsInfo:
    """Validated Polar conditions parsed from environment JSON."""

    def __init__(
        self,
        *,
        valid: bool,
        conditions: dict[str, object] | None = None,
        message: str = "",
    ) -> None:
        self.valid = valid
        self.conditions = conditions or {}
        self.message = message


def _conditions_from_env() -> _ConditionsInfo:
    """Parse and validate Polar condition metadata from the environment."""
    raw = os.environ.get("DIRECTOR_AI_POLAR_CONDITIONS", "").strip()
    if not raw:
        return _ConditionsInfo(valid=True)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _ConditionsInfo(
            valid=False,
            message=f"DIRECTOR_AI_POLAR_CONDITIONS must be JSON: {exc}",
        )
    return _map_info("DIRECTOR_AI_POLAR_CONDITIONS", parsed)


def _map_info(name: str, parsed: object) -> _ConditionsInfo:
    """Validate a Polar conditions JSON object and scalar values."""
    if not isinstance(parsed, dict):
        return _ConditionsInfo(valid=False, message=f"{name} must be a JSON object")
    if len(parsed) > 50:
        return _ConditionsInfo(
            valid=False, message=f"{name} supports at most 50 entries"
        )
    for key, value in parsed.items():
        if not isinstance(key, str) or len(key) > 40:
            return _ConditionsInfo(
                valid=False,
                message=f"{name} keys must be strings up to 40 characters",
            )
        if not isinstance(value, str | int | float | bool):
            return _ConditionsInfo(
                valid=False,
                message=f"{name} values must be scalar JSON values",
            )
        if isinstance(value, str) and len(value) > 500:
            return _ConditionsInfo(
                valid=False,
                message=f"{name} string condition values must be up to 500 characters",
            )
    return _ConditionsInfo(valid=True, conditions=parsed)


def _post_polar(
    url: str,
    request_body: dict[str, object],
    timeout_seconds: float | None,
) -> requests.Response | LicenseInfo:
    """POST to Polar and convert network failures into LicenseInfo."""
    timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else _env_timeout_seconds()
    )
    headers = _auth_headers(url)
    try:
        if headers:
            return requests.post(
                url,
                json=request_body,
                timeout=timeout,
                headers=headers,
            )
        return requests.post(url, json=request_body, timeout=timeout)
    except requests.RequestException as exc:
        return LicenseInfo(message=f"Polar request unavailable: {exc}")


def _response_payload(response: requests.Response) -> dict[str, Any] | None:
    """Return a JSON object payload or None for invalid response bodies."""
    if response.status_code == 204:
        return {}
    try:
        payload = response.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _activation_error(message: str, label: str = "") -> PolarActivationInfo:
    """Return a Polar activation result that carries an error message."""
    return PolarActivationInfo("", "", label, LicenseInfo(message=message), {})


def _header_value(headers: Mapping[str, str], name: str) -> str:
    """Return a case-insensitive Standard Webhooks header value."""
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value.strip()
    return ""


def _decode_webhook_secret(secret: str) -> bytes:
    """Decode and validate a Standard Webhooks secret."""
    clean = secret.strip()
    if not clean:
        raise ValueError("DIRECTOR_AI_POLAR_WEBHOOK_SECRET is not configured")
    if clean.startswith("whsec_"):
        clean = clean[6:]
    try:
        decoded = base64.b64decode(clean, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("DIRECTOR_AI_POLAR_WEBHOOK_SECRET must be base64") from exc
    if len(decoded) < 24:
        raise ValueError(
            "DIRECTOR_AI_POLAR_WEBHOOK_SECRET must decode to at least 24 bytes"
        )
    return decoded


def _signature_matches(signature_header: str, expected: bytes) -> bool:
    """Return whether any v1 signature matches the expected digest."""
    for item in signature_header.split():
        if "," not in item:
            continue
        scheme, signature = item.split(",", 1)
        if scheme != "v1":
            continue
        try:
            received = base64.b64decode(signature, validate=True)
        except (ValueError, binascii.Error):
            continue
        if hmac.compare_digest(received, expected):
            return True
    return False


def _license_info_from_payload(key: str, payload: dict[str, Any]) -> LicenseInfo:
    """Convert a Polar validation payload into LicenseInfo."""
    status = str(payload.get("status", "")).lower()
    if status != "granted":
        return LicenseInfo(message=f"Polar license status: {status or 'unknown'}")

    expires_at = _string_field(payload, "expires_at")
    if _is_expired(expires_at):
        return LicenseInfo(
            key=key,
            expires=expires_at,
            message=f"Polar license expired on {expires_at}",
        )

    tier = _resolve_tier(key, payload)
    if tier not in TIERS:
        return LicenseInfo(message=f"Unknown Polar license tier: {tier}")

    customer = payload.get("customer")
    customer_payload = customer if isinstance(customer, dict) else {}
    email = _string_field(customer_payload, "email")
    licensee = _string_field(customer_payload, "name")
    return LicenseInfo(
        tier=tier,
        licensee=licensee,
        email=email,
        key=key,
        expires=expires_at,
        deployments=_int_field(payload, "limit_activations"),
        valid=True,
        message=f"Polar license valid ({tier})",
    )


def _resolve_tier(key: str, payload: dict[str, Any]) -> str:
    """Resolve Director-AI tier from benefit, metadata, key, or default."""
    mapped_tier = _tier_from_benefit_map(_string_field(payload, "benefit_id"))
    if mapped_tier:
        return mapped_tier

    for source in (payload, payload.get("metadata"), _customer_metadata(payload)):
        if isinstance(source, dict):
            tier = _metadata_tier(source)
            if tier:
                return tier

    key_parts = key.split("-", 2)
    if len(key_parts) == 3 and key_parts[0].upper() == "DAI":
        return key_parts[1].lower()

    return os.environ.get("DIRECTOR_AI_POLAR_DEFAULT_TIER", "indie").strip().lower()


def _tier_from_benefit_map(benefit_id: str) -> str:
    """Map a Polar benefit id to a configured Director-AI tier."""
    if not benefit_id:
        return ""
    raw_map = os.environ.get("DIRECTOR_AI_POLAR_BENEFIT_TIERS", "").strip()
    if not raw_map:
        return ""
    for item in raw_map.split(","):
        if not item.strip():
            continue
        separator = "=" if "=" in item else ":"
        if separator not in item:
            continue
        raw_benefit, raw_tier = item.split(separator, 1)
        if raw_benefit.strip() == benefit_id:
            return raw_tier.strip().lower()
    return ""


def _metadata_tier(payload: dict[str, Any]) -> str:
    """Extract tier metadata from known Polar metadata fields."""
    for key in ("director_ai_tier", "tier", "license_tier"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def _customer_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Return nested Polar customer metadata when available."""
    customer = payload.get("customer")
    if not isinstance(customer, dict):
        return {}
    metadata = customer.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _is_expired(expires_at: str) -> bool:
    """Return whether an ISO expiry timestamp is in the past."""
    if not expires_at:
        return False
    try:
        expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    return datetime.now(UTC) > expires


def _string_field(payload: dict[str, Any], key: str) -> str:
    """Return a string payload field or an empty string."""
    value = payload.get(key, "")
    return value if isinstance(value, str) else ""


def _int_field(payload: dict[str, Any], key: str) -> int:
    """Return an integer payload field or zero."""
    value = payload.get(key, 0)
    return value if isinstance(value, int) else 0
