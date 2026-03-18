# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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

logger = logging.getLogger("director_ai.license")

TIERS = {"community", "indie", "pro", "enterprise", "trial"}
_SIGNING_SECRET = b"director-ai-license-v1"


@dataclass(frozen=True)
class LicenseInfo:
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
        return self.valid and self.tier in ("indie", "pro", "enterprise")

    @property
    def is_trial(self) -> bool:
        return self.valid and self.tier == "trial"

    @property
    def expired(self) -> bool:
        if not self.expires:
            return False
        try:
            exp = datetime.fromisoformat(self.expires)
            return datetime.now(UTC) > exp
        except ValueError:
            return False


def validate_key(key: str) -> LicenseInfo:
    """Validate a license key string (DAI-TIER-UUID format)."""
    if not key or not key.startswith("DAI-"):
        return LicenseInfo(message="No license key provided")

    parts = key.split("-", 2)
    if len(parts) < 3:
        return LicenseInfo(message="Invalid key format")

    tier = parts[1].lower()
    if tier not in TIERS:
        return LicenseInfo(message=f"Unknown tier: {tier}")

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

    sig = data.get("signature", "")
    payload = {k: v for k, v in data.items() if k != "signature"}
    expected_sig = hmac.new(
        _SIGNING_SECRET, json.dumps(payload, sort_keys=True).encode(), hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(sig, expected_sig):
        return LicenseInfo(message="Invalid license signature")

    tier = data.get("tier", "").lower()
    if tier not in TIERS:
        return LicenseInfo(message=f"Unknown tier in license: {tier}")

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
    1. DIRECTOR_LICENSE_KEY env var (key string)
    2. DIRECTOR_LICENSE_FILE env var (path to signed JSON)
    3. ~/.director-ai/license.json (default file location)
    4. Falls back to community (AGPL) tier
    """
    key = os.environ.get("DIRECTOR_LICENSE_KEY", "").strip()
    if key:
        info = validate_key(key)
        if info.valid:
            logger.info("License: %s (key)", info.message)
            return info
        logger.warning("License key invalid: %s", info.message)

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

    logger.info("License: AGPL-3.0-or-later (community)")
    return LicenseInfo(
        tier="community",
        valid=True,
        message="AGPL-3.0-or-later (no commercial license)",
    )


def generate_license(
    tier: str,
    licensee: str,
    email: str,
    days: int = 365,
    deployments: int = 1,
) -> dict:
    """Generate a signed license file (for admin/sales use).

    Returns a dict ready to be written as JSON.
    """
    import uuid

    now = datetime.now(UTC)
    from datetime import timedelta

    payload = {
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
        _SIGNING_SECRET, json.dumps(payload, sort_keys=True).encode(), hashlib.sha256
    ).hexdigest()
    return payload
