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

import pytest

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


def _packet_path(root: Path) -> Path:
    """Return the repository-relative Polar smoke packet path in a test root."""
    return root / "security" / "polar_deployment_smoke_packet.toml"


def _write_raw_packet(root: Path, text: str) -> None:
    """Write a raw Polar smoke packet for validator edge-case tests."""
    packet = _packet_path(root)
    packet.parent.mkdir(parents=True, exist_ok=True)
    packet.write_text(text.strip(), encoding="utf-8")


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


def test_polar_deployment_smoke_reports_missing_and_invalid_packet(
    tmp_path: Path,
) -> None:
    assert validate_polar_deployment_smoke(tmp_path) == [
        "security/polar_deployment_smoke_packet.toml: missing Polar deployment smoke packet"
    ]

    _write_raw_packet(tmp_path, "not = [valid")

    errors = validate_polar_deployment_smoke(tmp_path)

    assert len(errors) == 1
    assert errors[0].startswith(
        "security/polar_deployment_smoke_packet.toml: invalid TOML:"
    )


def test_polar_deployment_smoke_rejects_non_table_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_raw_packet(tmp_path, 'schema_version = "1.0.0"')
    monkeypatch.setattr(MODULE.tomllib, "loads", lambda _text: ["not-a-table"])

    assert MODULE._load_packet(_packet_path(tmp_path)) == (
        {},
        ["security/polar_deployment_smoke_packet.toml: packet must be a TOML table"],
    )


def test_polar_deployment_smoke_rejects_missing_required_fields(
    tmp_path: Path,
) -> None:
    _write_raw_packet(tmp_path, 'schema_version = "1.0.0"')

    errors = validate_polar_deployment_smoke(tmp_path)

    assert len(errors) == 1
    assert errors[0].startswith(
        "security/polar_deployment_smoke_packet.toml: missing required fields"
    )
    assert "provider" in errors[0]


def test_polar_deployment_smoke_reports_malformed_static_fields(
    tmp_path: Path,
) -> None:
    _write_raw_packet(
        tmp_path,
        """
schema_version = ""
packet_id = ""
provider = "stripe"
pricing_currency = "EUR"
pricing_status = "preview"
operator_approval_required = false
no_committed_secrets = false
public_commercial_claim = false
live_checkout_claim = false
checkout_status = "queued"
customer_portal_status = "passed"
webhook_status = "pending"
license_validation_status = "pending"
smoke_evidence_path = 123
env_preflight_command = "director-ai license polar-env"
claim_boundary = "public claims allowed"
required_smoke_checks = []
""",
    )

    errors = validate_polar_deployment_smoke(tmp_path)

    assert (
        "security/polar_deployment_smoke_packet.toml: schema_version must be a non-empty string"
        in errors
    )
    assert (
        "security/polar_deployment_smoke_packet.toml: packet_id must be a non-empty string"
        in errors
    )
    assert "security/polar_deployment_smoke_packet.toml: provider must be polar" in errors
    assert (
        "security/polar_deployment_smoke_packet.toml: pricing_currency must remain USD"
        in errors
    )
    assert (
        "security/polar_deployment_smoke_packet.toml: pricing_status must be request_checkout_links or live_checkout"
        in errors
    )
    assert (
        "security/polar_deployment_smoke_packet.toml: operator_approval_required must remain true"
        in errors
    )
    assert (
        "security/polar_deployment_smoke_packet.toml: no_committed_secrets must remain true"
        in errors
    )
    assert (
        "security/polar_deployment_smoke_packet.toml: checkout_status must be pending or passed"
        in errors
    )
    assert (
        "security/polar_deployment_smoke_packet.toml: smoke_evidence_path must be a string"
        in errors
    )
    assert (
        "security/polar_deployment_smoke_packet.toml: env_preflight_command must be director-ai license polar-env --json"
        in errors
    )
    assert (
        "security/polar_deployment_smoke_packet.toml: claim_boundary must state no public claim and no committed secrets"
        in errors
    )
    assert (
        "security/polar_deployment_smoke_packet.toml: required_smoke_checks must be a non-empty list"
        in errors
    )


def test_polar_deployment_smoke_rejects_bad_evidence_paths(
    tmp_path: Path,
) -> None:
    _write_packet(tmp_path, smoke_evidence_path="docs/internal/polar-smoke.md")
    pending_errors = validate_polar_deployment_smoke(tmp_path)
    assert (
        "security/polar_deployment_smoke_packet.toml: pending live smoke must not point at smoke evidence"
        in pending_errors
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
        smoke_evidence_path="docs/internal/missing-polar-smoke.md",
    )
    live_errors = validate_polar_deployment_smoke(tmp_path)
    assert (
        "security/polar_deployment_smoke_packet.toml: smoke_evidence_path does not exist"
        in live_errors
    )


def test_polar_deployment_smoke_rejects_non_string_required_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_raw_packet(
        tmp_path,
        """
schema_version = "1.0.0"
packet_id = "polar-deployment-smoke-test"
provider = "polar"
pricing_currency = "USD"
pricing_status = "request_checkout_links"
operator_approval_required = true
no_committed_secrets = true
public_commercial_claim = false
live_checkout_claim = false
checkout_status = "pending"
customer_portal_status = "pending"
webhook_status = "pending"
license_validation_status = "pending"
smoke_evidence_path = ""
env_preflight_command = "director-ai license polar-env --json"
claim_boundary = "No public live checkout claim and no committed secrets until all Polar smoke checks pass."
required_smoke_checks = ["usd_checkout_created", "customer_portal_session", "webhook_signature_validation", 1]
""",
    )
    monkeypatch.setattr(
        MODULE,
        "REQUIRED_SMOKE_CHECKS",
        {
            "usd_checkout_created",
            "customer_portal_session",
            "webhook_signature_validation",
            1,
        },
    )

    errors = validate_polar_deployment_smoke(tmp_path)

    assert (
        "security/polar_deployment_smoke_packet.toml: required_smoke_checks must contain strings"
        in errors
    )


def test_polar_deployment_smoke_cli_reports_success_and_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_packet(tmp_path)

    assert MODULE.main(["--root", str(tmp_path)]) == 0
    ok_output = capsys.readouterr()
    assert ok_output.out == "polar_deployment_smoke_ok\n"
    assert ok_output.err == ""

    assert MODULE.main(["--root", str(tmp_path), "--require-live"]) == 1
    failed_output = capsys.readouterr()
    assert failed_output.out == ""
    assert "--require-live requires all smoke checks passed" in failed_output.err
