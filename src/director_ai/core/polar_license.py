# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Polar licence validation for commercial deployments."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import requests

from director_ai.core.license import TIERS, LicenseInfo

POLAR_VALIDATE_URL = "https://api.polar.sh/v1/customer-portal/license-keys/validate"
POLAR_DEFAULT_TIMEOUT_SECONDS = 5.0


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

    increment_usage = os.environ.get("DIRECTOR_AI_POLAR_INCREMENT_USAGE", "").strip()
    if increment_usage:
        try:
            request_body["increment_usage"] = int(increment_usage)
        except ValueError:
            return LicenseInfo(message="DIRECTOR_AI_POLAR_INCREMENT_USAGE must be int")

    url = endpoint or os.environ.get("DIRECTOR_AI_POLAR_VALIDATE_URL", "").strip()
    if not url:
        url = POLAR_VALIDATE_URL

    timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else _env_timeout_seconds()
    )

    try:
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


def _env_timeout_seconds() -> float:
    raw = os.environ.get("DIRECTOR_AI_POLAR_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return POLAR_DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except ValueError:
        return POLAR_DEFAULT_TIMEOUT_SECONDS
    return max(0.1, timeout)


def _license_info_from_payload(key: str, payload: dict[str, Any]) -> LicenseInfo:
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
    for key in ("director_ai_tier", "tier", "license_tier"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def _customer_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    customer = payload.get("customer")
    if not isinstance(customer, dict):
        return {}
    metadata = customer.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _is_expired(expires_at: str) -> bool:
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
    value = payload.get(key, "")
    return value if isinstance(value, str) else ""


def _int_field(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key, 0)
    return value if isinstance(value, int) else 0
