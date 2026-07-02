#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Hugging Face Space deployment smoke packet validator
"""Validate the Hugging Face Space deployment-smoke gate packet."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any

PACKET = Path("demo/hf_space_deployment_smoke.toml")
SPACE_MANIFEST = Path("demo/hf_space_manifest.toml")
REQUIRED_FIELDS = {
    "schema_version",
    "packet_id",
    "space_slug",
    "space_url",
    "space_repo",
    "publish_by_default",
    "operator_approval_required",
    "public_demo_claim",
    "deployment_status",
    "smoke_status",
    "smoke_evidence_path",
    "claim_boundary",
    "demo_manifest",
    "package_validator",
    "push_script",
    "required_smoke_checks",
}
DEPLOYMENT_STATUSES = {"not_published", "published"}
SMOKE_STATUSES = {"pending", "passed"}
REQUIRED_SMOKE_CHECKS = {
    "space_http_200",
    "gradio_app_loads",
    "score_response_smoke",
    "streaming_halt_tab_smoke",
}


def _load_toml(path: Path, label: Path) -> tuple[dict[str, Any], list[str]]:
    """Load TOML from ``path`` and report errors against ``label``."""
    if not path.exists():
        return {}, [f"{label.as_posix()}: missing TOML file"]
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return {}, [f"{label.as_posix()}: invalid TOML: {exc}"]
    return data, []


def _non_empty_string(packet: dict[str, Any], field: str) -> bool:
    """Return whether ``field`` contains a non-empty string value."""
    value = packet.get(field)
    return isinstance(value, str) and bool(value.strip())


def _validate_required_fields(packet: dict[str, Any]) -> list[str]:
    """Return missing-field errors for the deployment-smoke packet."""
    missing = sorted(REQUIRED_FIELDS - set(packet))
    if missing:
        return [f"{PACKET}: missing required fields {', '.join(missing)}"]
    return []


def _validate_status(
    packet: dict[str, Any],
    field: str,
    allowed: set[str],
) -> list[str]:
    """Validate that ``field`` is one of the allowed string statuses."""
    value = packet[field]
    if isinstance(value, str) and value in allowed:
        return []
    return [
        f"{PACKET}: {field} must be one of {', '.join(sorted(allowed))}",
    ]


def _validate_required_smoke_checks(checks: Any) -> list[str]:
    """Validate the exact set of smoke checks required for publication."""
    if not isinstance(checks, list) or not checks:
        return [f"{PACKET}: required_smoke_checks must be a non-empty list"]
    if not all(isinstance(check, str) and check.strip() for check in checks):
        return [f"{PACKET}: required_smoke_checks must contain strings"]
    if set(checks) != REQUIRED_SMOKE_CHECKS or len(checks) != len(
        REQUIRED_SMOKE_CHECKS
    ):
        return [
            f"{PACKET}: required_smoke_checks must be exactly "
            f"{', '.join(sorted(REQUIRED_SMOKE_CHECKS))}",
        ]
    return []


def _validate_packet(
    root: Path,
    packet: dict[str, Any],
    manifest: dict[str, Any],
    require_published: bool,
) -> list[str]:
    """Validate deployment-smoke packet fields against local repo surfaces."""
    errors = _validate_required_fields(packet)
    if errors:
        return errors

    for field in ("schema_version", "packet_id", "space_slug", "claim_boundary"):
        if not _non_empty_string(packet, field):
            errors.append(f"{PACKET}: {field} must be a non-empty string")

    for field in ("space_url", "space_repo"):
        value = packet[field]
        if not isinstance(value, str) or not value.startswith(
            "https://huggingface.co/spaces/"
        ):
            errors.append(f"{PACKET}: {field} must be a Hugging Face Spaces HTTPS URL")

    expected_space_url = f"https://huggingface.co/spaces/{packet['space_slug']}"
    if packet["space_url"] != expected_space_url:
        errors.append(f"{PACKET}: space_url must match space_slug")
    if packet["space_repo"] != expected_space_url:
        errors.append(f"{PACKET}: space_repo must match space_slug")

    if packet["publish_by_default"] is not False:
        errors.append(f"{PACKET}: publish_by_default must remain false")
    if packet["operator_approval_required"] is not True:
        errors.append(f"{PACKET}: operator_approval_required must remain true")

    deployment_status = packet["deployment_status"]
    errors.extend(_validate_status(packet, "deployment_status", DEPLOYMENT_STATUSES))
    smoke_status = packet["smoke_status"]
    errors.extend(_validate_status(packet, "smoke_status", SMOKE_STATUSES))

    if packet["public_demo_claim"] is True and (
        deployment_status != "published" or smoke_status != "passed"
    ):
        errors.append(
            f"{PACKET}: public_demo_claim requires published deployment and passed smoke"
        )

    evidence_path = packet["smoke_evidence_path"]
    if not isinstance(evidence_path, str):
        errors.append(f"{PACKET}: smoke_evidence_path must be a string")
    elif smoke_status == "passed":
        if not evidence_path.strip():
            errors.append(f"{PACKET}: passed smoke requires smoke_evidence_path")
        elif not (root / evidence_path).exists():
            errors.append(f"{PACKET}: smoke_evidence_path does not exist")
    elif evidence_path.strip():
        errors.append(f"{PACKET}: pending smoke must not point at smoke evidence")

    boundary = packet["claim_boundary"]
    if (
        not isinstance(boundary, str)
        or "no public" not in boundary.lower()
        or "smoke" not in boundary.lower()
    ):
        errors.append(
            f"{PACKET}: claim_boundary must state no public claim without smoke"
        )

    errors.extend(_validate_required_smoke_checks(packet["required_smoke_checks"]))

    for field in ("demo_manifest", "package_validator", "push_script"):
        value = packet[field]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{PACKET}: {field} must be a non-empty path")
        elif not (root / value).exists():
            errors.append(f"{PACKET}: {field} path does not exist")

    manifest_slug = manifest.get("space_slug")
    if isinstance(manifest_slug, str) and manifest_slug != packet["space_slug"]:
        errors.append(f"{PACKET}: space_slug must match demo/hf_space_manifest.toml")

    manifest_publish = manifest.get("publish_by_default")
    if manifest_publish is not False:
        errors.append(f"{SPACE_MANIFEST}: publish_by_default must remain false")

    if require_published and (
        deployment_status != "published" or smoke_status != "passed"
    ):
        errors.append(
            f"{PACKET}: --require-published requires published deployment and passed smoke"
        )
    return errors


def validate_hf_space_deployment_smoke(
    root: Path,
    *,
    require_published: bool = False,
) -> list[str]:
    """Validate the deployment-smoke packet rooted at ``root``."""
    root = root.resolve()
    packet, errors = _load_toml(root / PACKET, PACKET)
    if errors:
        return errors
    manifest, manifest_errors = _load_toml(root / SPACE_MANIFEST, SPACE_MANIFEST)
    errors.extend(manifest_errors)
    if manifest_errors:
        return errors
    errors.extend(_validate_packet(root, packet, manifest, require_published))
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run the command-line deployment-smoke validator."""
    parser = argparse.ArgumentParser(
        description="Validate the HF Space deployment smoke packet."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing the Space deployment smoke packet",
    )
    parser.add_argument(
        "--require-published",
        action="store_true",
        help="Fail unless deployment and smoke evidence are complete",
    )
    args = parser.parse_args(argv)
    errors = validate_hf_space_deployment_smoke(
        args.root,
        require_published=args.require_published,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("hf_space_deployment_smoke_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
