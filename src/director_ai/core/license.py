# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""License validation for commercial Director-AI deployments.

License keys follow the format: DAI-{TIER}-{UUID}
Example: DAI-PRO-a8f3e2b1-9c4d-4e5f-b6a7-1234567890ab

License files are signed JSON containing key, tier, licensee, and dates.
Validation is offline-only — no phone-home, no DRM.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

logger = logging.getLogger("director_ai.license")

TIERS = {"community", "indie", "pro", "enterprise", "trial"}

# Capability rank of each tier for gating: a higher rank unlocks everything at
# or below it. Community/open-core is 0; ``trial`` grants pro-level access for
# the trial window. The exact tier-to-capability matrix (which features require
# which tier, and whether enforcement is on by default) is a commercial policy
# decision layered on top of this mechanism.
_TIER_RANK = {"community": 0, "indie": 1, "trial": 2, "pro": 2, "enterprise": 3}
_ENFORCE_ENV = "DIRECTOR_ENFORCE_LICENSE_TIER"


class LicenseError(RuntimeError):
    """Raised when the active license does not cover a requested capability."""


def tier_rank(tier: str) -> int:
    """Return the capability rank of a tier; unknown tiers rank as community (0)."""
    return _TIER_RANK.get(tier.strip().lower(), 0)


def require_tier(info: LicenseInfo, minimum: str, *, capability: str = "") -> None:
    """Raise :class:`LicenseError` unless the license reaches ``minimum``.

    Rank-based, so a higher tier satisfies a lower minimum. A ``minimum`` at
    community rank (0) always passes. An invalid license is treated as
    community. The message names the required and active tiers so the caller can
    surface an actionable upgrade path.
    """
    required = tier_rank(minimum)
    if required <= 0:
        return
    current = tier_rank(info.tier) if info.valid else 0
    if current < required:
        detail = f" for {capability}" if capability else ""
        active = info.tier if info.valid else "community"
        raise LicenseError(
            f"This capability{detail} requires the '{minimum}' tier or higher; "
            f"the active license is '{active}'. See https://anulum.li for licensing."
        )


def tier_enforcement_enabled() -> bool:
    """Whether runtime license-tier enforcement is opted in via the environment."""
    return os.environ.get(_ENFORCE_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def enforce_capability_tier(capability: str, *, minimum: str = "pro") -> None:
    """Enforce the license tier for an advanced capability when opted in.

    No-op unless ``DIRECTOR_ENFORCE_LICENSE_TIER`` is truthy, so the default
    stays backward-compatible (advisory). When enabled, loads the active license
    and raises :class:`LicenseError` if it does not reach ``minimum``.
    """
    if not tier_enforcement_enabled():
        return
    require_tier(load_license(), minimum, capability=capability)


def _signing_secret() -> bytes:
    secret = os.environ.get("DIRECTOR_LICENSE_SIGNING_KEY", "").strip()
    if not secret:
        raise RuntimeError(
            "DIRECTOR_LICENSE_SIGNING_KEY not set. "
            "License generation/validation requires the signing key in the environment."
        )
    return secret.encode("utf-8")


@dataclass(frozen=True)
class LicenseInfo:
    """Validated license state returned by local license checks."""

    tier: str = "community"
    licensee: str = ""
    email: str = ""
    key: str = ""
    issued: str = ""
    expires: str = ""
    deployments: int = 0
    valid: bool = False
    message: str = ""

    @property
    def is_commercial(self) -> bool:
        """Return whether the license unlocks paid deployment rights."""
        return self.valid and self.tier in ("indie", "pro", "enterprise")

    @property
    def is_trial(self) -> bool:
        """Return whether the license is an active trial license."""
        return self.valid and self.tier == "trial"

    @property
    def expired(self) -> bool:
        """Return whether the license expiry is missing or already past."""
        if not self.expires:
            return False
        try:
            exp = datetime.fromisoformat(self.expires)
            return datetime.now(UTC) > exp
        except (ValueError, TypeError):
            return True  # malformed expiry = expired (fail closed)


def validate_key(key: str) -> LicenseInfo:
    """Validate a license key string (DAI-TIER-UUID format)."""
    if not key or not key.startswith("DAI-"):
        return LicenseInfo(message="No license key provided")

    parts = key.split("-", 2)
    if len(parts) != 3:
        return LicenseInfo(message="Invalid key format")

    tier = parts[1].lower()
    if tier not in TIERS:
        return LicenseInfo(message=f"Unknown tier: {tier}")

    key_id = parts[2].strip()
    if not key_id:
        return LicenseInfo(message="Invalid key format")

    try:
        parsed = UUID(key_id)
    except ValueError:
        return LicenseInfo(message="Invalid key UUID")

    if str(parsed) != key_id.lower():
        return LicenseInfo(message="Invalid key UUID")

    return LicenseInfo(
        tier=tier,
        key=key,
        valid=True,
        message=f"License key accepted ({tier})",
    )


def validate_file(path: str | Path) -> LicenseInfo:
    """Validate a signed license file (JSON)."""
    p = Path(path).expanduser()
    if not p.exists():
        return LicenseInfo(message=f"License file not found: {p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return LicenseInfo(message=f"Cannot read license file: {exc}")

    try:
        secret = _signing_secret()
    except RuntimeError:
        return LicenseInfo(
            message="Signing key not configured; cannot validate license file"
        )

    sig = data.get("signature", "")
    payload = {k: v for k, v in data.items() if k != "signature"}
    expected_sig = hmac.new(
        secret, json.dumps(payload, sort_keys=True).encode(), hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(sig, expected_sig):
        return LicenseInfo(message="Invalid license signature")

    tier = data.get("tier", "").lower()
    if tier not in TIERS:
        return LicenseInfo(message=f"Unknown tier in license: {tier}")

    key_info = validate_key(str(data.get("key", "")).strip())
    if not key_info.valid:
        return LicenseInfo(message=f"Invalid license key in file: {key_info.message}")
    if key_info.tier != tier:
        return LicenseInfo(message="License tier does not match key tier")

    info = LicenseInfo(
        tier=tier,
        licensee=data.get("licensee", ""),
        email=data.get("email", ""),
        key=data.get("key", ""),
        issued=data.get("issued", ""),
        expires=data.get("expires", ""),
        deployments=data.get("deployments", 0),
        valid=True,
        message=f"License valid ({tier})",
    )

    if info.expired:
        return LicenseInfo(
            tier=info.tier,
            licensee=info.licensee,
            email=info.email,
            key=info.key,
            expires=info.expires,
            valid=False,
            message=f"License expired on {info.expires}",
        )

    return info


def load_license() -> LicenseInfo:
    """Load license from environment or file.

    Checks in order:
    1. DIRECTOR_LICENSE_KEY + DIRECTOR_AI_POLAR_ORG_ID via Polar validation
    2. DIRECTOR_LICENSE_KEY env var (syntax check only; does not activate)
    3. DIRECTOR_LICENSE_FILE env var (path to signed JSON)
    4. ~/.director-ai/license.json (default file location)
    5. Falls back to the open-core community tier (Apache-2.0 core + BUSL-1.1
       advanced, free for non-production)
    """
    key = os.environ.get("DIRECTOR_LICENSE_KEY", "").strip()
    if key:
        polar_org_id = os.environ.get("DIRECTOR_AI_POLAR_ORG_ID", "").strip()
        if polar_org_id:
            from director_ai.core.polar_license import validate_polar_key

            polar_info = validate_polar_key(key, polar_org_id)
            if polar_info.valid:
                logger.info("License: %s (Polar)", polar_info.message)
                return polar_info
            logger.warning("Polar license invalid: %s", polar_info.message)

        key_info = validate_key(key)
        if key_info.valid:
            logger.warning(
                "Bare DIRECTOR_LICENSE_KEY is not sufficient; "
                "provide DIRECTOR_LICENSE_FILE with a signed license.",
            )
        else:
            logger.warning("License key invalid: %s", key_info.message)

    file_path = os.environ.get("DIRECTOR_LICENSE_FILE", "").strip()
    if file_path:
        info = validate_file(file_path)
        if info.valid:
            logger.info("License: %s (file)", info.message)
            return info
        logger.warning("License file invalid: %s", info.message)

    default_path = Path.home() / ".director-ai" / "license.json"
    if default_path.exists():
        info = validate_file(default_path)
        if info.valid:
            logger.info("License: %s (default file)", info.message)
            return info

    logger.info("License: open core (Apache-2.0 + BUSL-1.1, community)")
    return LicenseInfo(
        tier="community",
        valid=True,
        message="open core: Apache-2.0 + BUSL-1.1 (no commercial license)",
    )


def generate_license(
    tier: str,
    licensee: str,
    email: str,
    days: int = 365,
    deployments: int = 1,
) -> dict[str, str | int]:
    """Generate a signed license file (for admin/sales use).

    Returns a dict ready to be written as JSON.
    """
    import uuid

    now = datetime.now(UTC)
    from datetime import timedelta

    payload: dict[str, str | int] = {
        "key": f"DAI-{tier.upper()}-{uuid.uuid4()}",
        "tier": tier.lower(),
        "licensee": licensee,
        "email": email,
        "issued": now.isoformat(),
        "expires": (now + timedelta(days=days)).isoformat() if days > 0 else "",
        "deployments": deployments,
        "version": "1",
    }
    payload["signature"] = hmac.new(
        _signing_secret(), json.dumps(payload, sort_keys=True).encode(), hashlib.sha256
    ).hexdigest()
    return payload
