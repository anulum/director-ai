# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Polar deployment smoke packet tests

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_polar_deployment_smoke.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_polar_deployment_smoke", VALIDATOR
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

validate_polar_deployment_smoke = MODULE.validate_polar_deployment_smoke


def _write_packet(
    root: Path,
    *,
    pricing_status: str = "request_checkout_links",
    public_commercial_claim: bool = False,
    live_checkout_claim: bool = False,
    statuses: dict[str, str] | None = None,
    smoke_evidence_path: str = "",
    required_smoke_checks: list[str] | None = None,
) -> None:
    security = root / "security"
    security.mkdir(parents=True, exist_ok=True)
    values = {
        "checkout_status": "pending",
        "customer_portal_status": "pending",
        "webhook_status": "pending",
        "license_validation_status": "pending",
    }
    values.update(statuses or {})
    checks = required_smoke_checks or [
        "usd_checkout_created",
        "customer_portal_session",
        "webhook_signature_validation",
        "license_key_validation",
    ]
    rendered_checks = ", ".join(f'"{check}"' for check in checks)
    (security / "polar_deployment_smoke_packet.toml").write_text(
        f"""
schema_version = "1.0.0"
packet_id = "polar-deployment-smoke-test"
provider = "polar"
pricing_currency = "USD"
pricing_status = "{pricing_status}"
operator_approval_required = true
no_committed_secrets = true
public_commercial_claim = {str(public_commercial_claim).lower()}
live_checkout_claim = {str(live_checkout_claim).lower()}
checkout_status = "{values["checkout_status"]}"
customer_portal_status = "{values["customer_portal_status"]}"
webhook_status = "{values["webhook_status"]}"
license_validation_status = "{values["license_validation_status"]}"
smoke_evidence_path = "{smoke_evidence_path}"
env_preflight_command = "director-ai license polar-env --json"
claim_boundary = "No public live checkout claim and no committed secrets until all Polar smoke checks pass."
required_smoke_checks = [{rendered_checks}]
""".strip(),
        encoding="utf-8",
    )


def test_polar_deployment_smoke_validates_current_packet() -> None:
    assert validate_polar_deployment_smoke(ROOT) == []


def test_polar_deployment_smoke_require_live_fails_pending() -> None:
    errors = validate_polar_deployment_smoke(ROOT, require_live=True)

    assert (
        "security/polar_deployment_smoke_packet.toml: --require-live requires all smoke checks passed"
        in errors
    )


def test_polar_deployment_smoke_rejects_public_claim_before_checks(
    tmp_path: Path,
) -> None:
    _write_packet(tmp_path, public_commercial_claim=True, live_checkout_claim=True)

    errors = validate_polar_deployment_smoke(tmp_path)

    assert (
        "security/polar_deployment_smoke_packet.toml: public_commercial_claim requires all smoke checks passed"
        in errors
    )
    assert (
        "security/polar_deployment_smoke_packet.toml: live_checkout_claim requires all smoke checks passed"
        in errors
    )


def test_polar_deployment_smoke_rejects_live_checkout_without_all_checks(
    tmp_path: Path,
) -> None:
    _write_packet(tmp_path, pricing_status="live_checkout")

    errors = validate_polar_deployment_smoke(tmp_path)

    assert (
        "security/polar_deployment_smoke_packet.toml: live_checkout pricing requires all smoke checks passed"
        in errors
    )


def test_polar_deployment_smoke_requires_evidence_when_checks_passed(
    tmp_path: Path,
) -> None:
    _write_packet(
        tmp_path,
        pricing_status="live_checkout",
        public_commercial_claim=True,
        live_checkout_claim=True,
        statuses={
            "checkout_status": "passed",
            "customer_portal_status": "passed",
            "webhook_status": "passed",
            "license_validation_status": "passed",
        },
    )

    errors = validate_polar_deployment_smoke(tmp_path)

    assert (
        "security/polar_deployment_smoke_packet.toml: passed live smoke requires smoke_evidence_path"
        in errors
    )


def test_polar_deployment_smoke_rejects_missing_required_check(
    tmp_path: Path,
) -> None:
    _write_packet(
        tmp_path,
        required_smoke_checks=[
            "usd_checkout_created",
            "customer_portal_session",
            "webhook_signature_validation",
        ],
    )

    errors = validate_polar_deployment_smoke(tmp_path)

    assert (
        "security/polar_deployment_smoke_packet.toml: required_smoke_checks must be exactly customer_portal_session, license_key_validation, usd_checkout_created, webhook_signature_validation"
        in errors
    )


def test_polar_deployment_smoke_accepts_live_packet_with_evidence(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "docs" / "internal" / "polar-smoke.md"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(
        "USD checkout, customer portal, webhook, and licence validation passed.\n",
        encoding="utf-8",
    )
    _write_packet(
        tmp_path,
        pricing_status="live_checkout",
        public_commercial_claim=True,
        live_checkout_claim=True,
        statuses={
            "checkout_status": "passed",
            "customer_portal_status": "passed",
            "webhook_status": "passed",
            "license_validation_status": "passed",
        },
        smoke_evidence_path="docs/internal/polar-smoke.md",
    )

    assert validate_polar_deployment_smoke(tmp_path, require_live=True) == []
