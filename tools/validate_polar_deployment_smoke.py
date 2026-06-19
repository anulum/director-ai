#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Polar deployment smoke packet validator

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any

PACKET = Path("security/polar_deployment_smoke_packet.toml")
REQUIRED_FIELDS = {
    "schema_version",
    "packet_id",
    "provider",
    "pricing_currency",
    "pricing_status",
    "operator_approval_required",
    "no_committed_secrets",
    "public_commercial_claim",
    "live_checkout_claim",
    "checkout_status",
    "customer_portal_status",
    "webhook_status",
    "license_validation_status",
    "smoke_evidence_path",
    "env_preflight_command",
    "claim_boundary",
    "required_smoke_checks",
}
STATUS_FIELDS = {
    "checkout_status",
    "customer_portal_status",
    "webhook_status",
    "license_validation_status",
}
SMOKE_STATUSES = {"pending", "passed"}
REQUIRED_SMOKE_CHECKS = {
    "usd_checkout_created",
    "customer_portal_session",
    "webhook_signature_validation",
    "license_key_validation",
}


def _load_packet(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, [f"{PACKET}: missing Polar deployment smoke packet"]
    try:
        packet = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return {}, [f"{PACKET}: invalid TOML: {exc}"]
    if not isinstance(packet, dict):
        return {}, [f"{PACKET}: packet must be a TOML table"]
    return packet, []


def _all_live_checks_passed(packet: dict[str, Any]) -> bool:
    return all(packet.get(field) == "passed" for field in STATUS_FIELDS)


def _validate_packet(
    root: Path,
    packet: dict[str, Any],
    require_live: bool,
) -> list[str]:
    missing = sorted(REQUIRED_FIELDS - set(packet))
    if missing:
        return [f"{PACKET}: missing required fields {', '.join(missing)}"]

    errors: list[str] = []
    for field in ("schema_version", "packet_id", "claim_boundary"):
        value = packet[field]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{PACKET}: {field} must be a non-empty string")

    if packet["provider"] != "polar":
        errors.append(f"{PACKET}: provider must be polar")
    if packet["pricing_currency"] != "USD":
        errors.append(f"{PACKET}: pricing_currency must remain USD")
    if packet["pricing_status"] not in {"request_checkout_links", "live_checkout"}:
        errors.append(
            f"{PACKET}: pricing_status must be request_checkout_links or live_checkout"
        )

    if packet["operator_approval_required"] is not True:
        errors.append(f"{PACKET}: operator_approval_required must remain true")
    if packet["no_committed_secrets"] is not True:
        errors.append(f"{PACKET}: no_committed_secrets must remain true")

    for field in STATUS_FIELDS:
        if packet[field] not in SMOKE_STATUSES:
            errors.append(f"{PACKET}: {field} must be pending or passed")

    live_ready = _all_live_checks_passed(packet)
    if packet["pricing_status"] == "live_checkout" and not live_ready:
        errors.append(
            f"{PACKET}: live_checkout pricing requires all smoke checks passed"
        )
    if packet["public_commercial_claim"] is True and not live_ready:
        errors.append(
            f"{PACKET}: public_commercial_claim requires all smoke checks passed"
        )
    if packet["live_checkout_claim"] is True and not live_ready:
        errors.append(f"{PACKET}: live_checkout_claim requires all smoke checks passed")

    evidence_path = packet["smoke_evidence_path"]
    if not isinstance(evidence_path, str):
        errors.append(f"{PACKET}: smoke_evidence_path must be a string")
    elif live_ready:
        if not evidence_path.strip():
            errors.append(f"{PACKET}: passed live smoke requires smoke_evidence_path")
        elif not (root / evidence_path).exists():
            errors.append(f"{PACKET}: smoke_evidence_path does not exist")
    elif evidence_path.strip():
        errors.append(f"{PACKET}: pending live smoke must not point at smoke evidence")

    command = packet["env_preflight_command"]
    if command != "director-ai license polar-env --json":
        errors.append(
            f"{PACKET}: env_preflight_command must be director-ai license polar-env --json"
        )

    boundary = packet["claim_boundary"]
    if (
        not isinstance(boundary, str)
        or "no public" not in boundary.lower()
        or "no committed secrets" not in boundary.lower()
    ):
        errors.append(
            f"{PACKET}: claim_boundary must state no public claim and no committed secrets"
        )

    checks = packet["required_smoke_checks"]
    if not isinstance(checks, list) or not checks:
        errors.append(f"{PACKET}: required_smoke_checks must be a non-empty list")
    elif set(checks) != REQUIRED_SMOKE_CHECKS:
        errors.append(
            f"{PACKET}: required_smoke_checks must be exactly "
            f"{', '.join(sorted(REQUIRED_SMOKE_CHECKS))}"
        )
    elif not all(isinstance(check, str) and check.strip() for check in checks):
        errors.append(f"{PACKET}: required_smoke_checks must contain strings")

    if require_live and not live_ready:
        errors.append(f"{PACKET}: --require-live requires all smoke checks passed")
    return errors


def validate_polar_deployment_smoke(
    root: Path,
    *,
    require_live: bool = False,
) -> list[str]:
    root = root.resolve()
    packet, errors = _load_packet(root / PACKET)
    if errors:
        return errors
    errors.extend(_validate_packet(root, packet, require_live))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the Polar deployment smoke packet."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing the Polar deployment smoke packet",
    )
    parser.add_argument(
        "--require-live",
        action="store_true",
        help="Fail unless live checkout, portal, webhook, and licence smoke passed",
    )
    args = parser.parse_args(argv)
    errors = validate_polar_deployment_smoke(args.root, require_live=args.require_live)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("polar_deployment_smoke_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
