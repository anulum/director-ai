# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Hugging Face Space deployment smoke packet tests
"""Unit guard for Hugging Face Space deployment-smoke packet rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.validate_hf_space_deployment_smoke import (
    main,
    validate_hf_space_deployment_smoke,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(root: Path) -> None:
    """Write the minimal Space manifest surfaces required by the validator."""
    demo = root / "demo"
    demo.mkdir(parents=True, exist_ok=True)
    (demo / "hf_space_manifest.toml").write_text(
        """
schema_version = "1.0.0"
space_slug = "anulum/director-ai-guardrail"
publish_by_default = false
files = ["app.py", "requirements.txt", "README.md"]
""".strip(),
        encoding="utf-8",
    )
    (demo / "push_to_hf.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (root / "tools").mkdir(parents=True, exist_ok=True)
    (root / "tools" / "validate_hf_space_demo.py").write_text(
        "#!/usr/bin/env python3\n",
        encoding="utf-8",
    )


def _write_packet(
    root: Path,
    *,
    deployment_status: str = "not_published",
    smoke_status: str = "pending",
    public_demo_claim: bool = False,
    smoke_evidence_path: str = "",
    required_smoke_checks: list[str] | None = None,
) -> None:
    """Write a deployment-smoke packet fixture under ``root``."""
    _write_manifest(root)
    checks = (
        [
            "space_http_200",
            "gradio_app_loads",
            "score_response_smoke",
            "streaming_halt_tab_smoke",
        ]
        if required_smoke_checks is None
        else required_smoke_checks
    )
    rendered_checks = ", ".join(f'"{check}"' for check in checks)
    (root / "demo" / "hf_space_deployment_smoke.toml").write_text(
        f"""
schema_version = "1.0.0"
packet_id = "hf-space-deployment-smoke-test"
space_slug = "anulum/director-ai-guardrail"
space_url = "https://huggingface.co/spaces/anulum/director-ai-guardrail"
space_repo = "https://huggingface.co/spaces/anulum/director-ai-guardrail"
publish_by_default = false
operator_approval_required = true
public_demo_claim = {str(public_demo_claim).lower()}
deployment_status = "{deployment_status}"
smoke_status = "{smoke_status}"
smoke_evidence_path = "{smoke_evidence_path}"
claim_boundary = "No public live demo claim until deployment and smoke evidence exist."
demo_manifest = "demo/hf_space_manifest.toml"
package_validator = "tools/validate_hf_space_demo.py"
push_script = "demo/push_to_hf.sh"
required_smoke_checks = [{rendered_checks}]
""".strip(),
        encoding="utf-8",
    )


def test_hf_space_deployment_smoke_validates_current_packet() -> None:
    """The checked-in deployment-smoke packet should satisfy the validator."""
    assert validate_hf_space_deployment_smoke(ROOT) == []


def test_hf_space_deployment_smoke_require_published_fails_pending() -> None:
    """The publish-required mode should fail while the packet is pending."""
    errors = validate_hf_space_deployment_smoke(ROOT, require_published=True)

    assert (
        "demo/hf_space_deployment_smoke.toml: --require-published requires published deployment and passed smoke"
        in errors
    )


def test_hf_space_deployment_smoke_rejects_public_claim_before_smoke(
    tmp_path: Path,
) -> None:
    """A public demo claim should require published deployment and smoke evidence."""
    _write_packet(tmp_path, public_demo_claim=True)

    errors = validate_hf_space_deployment_smoke(tmp_path)

    assert (
        "demo/hf_space_deployment_smoke.toml: public_demo_claim requires published deployment and passed smoke"
        in errors
    )


def test_hf_space_deployment_smoke_requires_evidence_when_passed(
    tmp_path: Path,
) -> None:
    """A passed smoke status should point to archived evidence."""
    _write_packet(
        tmp_path,
        deployment_status="published",
        smoke_status="passed",
        public_demo_claim=True,
    )

    errors = validate_hf_space_deployment_smoke(tmp_path)

    assert (
        "demo/hf_space_deployment_smoke.toml: passed smoke requires smoke_evidence_path"
        in errors
    )


def test_hf_space_deployment_smoke_rejects_missing_required_check(
    tmp_path: Path,
) -> None:
    """The smoke packet should enumerate every required live Space check."""
    _write_packet(
        tmp_path,
        required_smoke_checks=[
            "space_http_200",
            "gradio_app_loads",
            "score_response_smoke",
        ],
    )

    errors = validate_hf_space_deployment_smoke(tmp_path)

    assert (
        "demo/hf_space_deployment_smoke.toml: required_smoke_checks must be exactly gradio_app_loads, score_response_smoke, space_http_200, streaming_halt_tab_smoke"
        in errors
    )


def test_hf_space_deployment_smoke_rejects_empty_required_checks(
    tmp_path: Path,
) -> None:
    """The smoke packet should reject an empty live-check list."""
    _write_packet(tmp_path, required_smoke_checks=[])

    errors = validate_hf_space_deployment_smoke(tmp_path)

    assert (
        "demo/hf_space_deployment_smoke.toml: required_smoke_checks must be a non-empty list"
        in errors
    )


def test_hf_space_deployment_smoke_accepts_published_packet_with_evidence(
    tmp_path: Path,
) -> None:
    """Published packets with archived smoke evidence should pass strict mode."""
    evidence = tmp_path / "docs" / "internal" / "hf-space-smoke.md"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("Space HTTP 200 and both tabs passed.\n", encoding="utf-8")
    _write_packet(
        tmp_path,
        deployment_status="published",
        smoke_status="passed",
        public_demo_claim=True,
        smoke_evidence_path="docs/internal/hf-space-smoke.md",
    )

    assert (
        validate_hf_space_deployment_smoke(
            tmp_path,
            require_published=True,
        )
        == []
    )


def test_hf_space_deployment_smoke_rejects_missing_packet(tmp_path: Path) -> None:
    """A missing deployment-smoke packet should be reported without crashing."""
    errors = validate_hf_space_deployment_smoke(tmp_path)

    assert errors == ["demo/hf_space_deployment_smoke.toml: missing TOML file"]


def test_hf_space_deployment_smoke_rejects_invalid_packet_toml(
    tmp_path: Path,
) -> None:
    """Invalid packet TOML should be reported as a validation error."""
    _write_manifest(tmp_path)
    (tmp_path / "demo" / "hf_space_deployment_smoke.toml").write_text(
        "[invalid\n",
        encoding="utf-8",
    )

    errors = validate_hf_space_deployment_smoke(tmp_path)

    assert errors
    assert errors[0].startswith("demo/hf_space_deployment_smoke.toml: invalid TOML:")


def test_hf_space_deployment_smoke_rejects_missing_manifest(
    tmp_path: Path,
) -> None:
    """The deployment packet should require the Space manifest it references."""
    _write_packet(tmp_path)
    (tmp_path / "demo" / "hf_space_manifest.toml").unlink()

    errors = validate_hf_space_deployment_smoke(tmp_path)

    assert errors == ["demo/hf_space_manifest.toml: missing TOML file"]


def test_hf_space_deployment_smoke_rejects_missing_required_fields(
    tmp_path: Path,
) -> None:
    """The deployment packet should fail closed when required keys are absent."""
    _write_packet(tmp_path)
    packet = tmp_path / "demo" / "hf_space_deployment_smoke.toml"
    packet.write_text(
        packet.read_text(encoding="utf-8").replace(
            'packet_id = "hf-space-deployment-smoke-test"\n',
            "",
        ),
        encoding="utf-8",
    )

    errors = validate_hf_space_deployment_smoke(tmp_path)

    assert errors == [
        "demo/hf_space_deployment_smoke.toml: missing required fields packet_id"
    ]


def test_hf_space_deployment_smoke_reports_invalid_packet_values(
    tmp_path: Path,
) -> None:
    """Malformed packet values should produce explicit validation errors."""
    _write_packet(tmp_path)
    packet = tmp_path / "demo" / "hf_space_deployment_smoke.toml"
    packet.write_text(
        """
schema_version = ""
packet_id = ""
space_slug = "anulum/director-ai-guardrail"
space_url = "https://example.invalid/anulum/director-ai-guardrail"
space_repo = "https://huggingface.co/spaces/anulum/other"
publish_by_default = true
operator_approval_required = false
public_demo_claim = false
deployment_status = "shipping"
smoke_status = "done"
smoke_evidence_path = "docs/internal/not-yet.md"
claim_boundary = "Ready for launch."
demo_manifest = ""
package_validator = "missing.py"
push_script = "demo/missing.sh"
required_smoke_checks = ["space_http_200", 42]
""".strip(),
        encoding="utf-8",
    )

    errors = validate_hf_space_deployment_smoke(tmp_path)

    assert (
        "demo/hf_space_deployment_smoke.toml: schema_version must be a non-empty string"
        in errors
    )
    assert (
        "demo/hf_space_deployment_smoke.toml: packet_id must be a non-empty string"
        in errors
    )
    assert (
        "demo/hf_space_deployment_smoke.toml: space_url must be a Hugging Face Spaces HTTPS URL"
        in errors
    )
    assert (
        "demo/hf_space_deployment_smoke.toml: space_url must match space_slug" in errors
    )
    assert (
        "demo/hf_space_deployment_smoke.toml: space_repo must match space_slug"
        in errors
    )
    assert (
        "demo/hf_space_deployment_smoke.toml: publish_by_default must remain false"
        in errors
    )
    assert (
        "demo/hf_space_deployment_smoke.toml: operator_approval_required must remain true"
        in errors
    )
    assert (
        "demo/hf_space_deployment_smoke.toml: deployment_status must be one of not_published, published"
        in errors
    )
    assert (
        "demo/hf_space_deployment_smoke.toml: smoke_status must be one of passed, pending"
        in errors
    )
    assert (
        "demo/hf_space_deployment_smoke.toml: pending smoke must not point at smoke evidence"
        in errors
    )
    assert (
        "demo/hf_space_deployment_smoke.toml: claim_boundary must state no public claim without smoke"
        in errors
    )
    assert (
        "demo/hf_space_deployment_smoke.toml: required_smoke_checks must contain strings"
        in errors
    )
    assert (
        "demo/hf_space_deployment_smoke.toml: demo_manifest must be a non-empty path"
        in errors
    )
    assert (
        "demo/hf_space_deployment_smoke.toml: package_validator path does not exist"
        in errors
    )
    assert (
        "demo/hf_space_deployment_smoke.toml: push_script path does not exist" in errors
    )


def test_hf_space_deployment_smoke_rejects_non_string_evidence_path(
    tmp_path: Path,
) -> None:
    """Smoke evidence paths should be strings, not arbitrary TOML values."""
    _write_packet(tmp_path)
    packet = tmp_path / "demo" / "hf_space_deployment_smoke.toml"
    packet.write_text(
        packet.read_text(encoding="utf-8").replace(
            'smoke_evidence_path = ""',
            "smoke_evidence_path = 42",
        ),
        encoding="utf-8",
    )

    errors = validate_hf_space_deployment_smoke(tmp_path)

    assert (
        "demo/hf_space_deployment_smoke.toml: smoke_evidence_path must be a string"
        in errors
    )


def test_hf_space_deployment_smoke_reports_invalid_manifest_values(
    tmp_path: Path,
) -> None:
    """Manifest drift should be detected alongside packet validation."""
    _write_packet(tmp_path)
    (tmp_path / "demo" / "hf_space_manifest.toml").write_text(
        """
schema_version = "1.0.0"
space_slug = "anulum/other"
publish_by_default = true
files = ["app.py", "requirements.txt", "README.md"]
""".strip(),
        encoding="utf-8",
    )

    errors = validate_hf_space_deployment_smoke(tmp_path)

    assert (
        "demo/hf_space_deployment_smoke.toml: space_slug must match demo/hf_space_manifest.toml"
        in errors
    )
    assert "demo/hf_space_manifest.toml: publish_by_default must remain false" in errors


def test_hf_space_deployment_smoke_rejects_unhashable_status_values(
    tmp_path: Path,
) -> None:
    """Array statuses should be validation errors rather than type crashes."""
    _write_packet(tmp_path)
    packet = tmp_path / "demo" / "hf_space_deployment_smoke.toml"
    packet.write_text(
        packet.read_text(encoding="utf-8")
        .replace('deployment_status = "not_published"', 'deployment_status = ["bad"]')
        .replace('smoke_status = "pending"', 'smoke_status = ["bad"]'),
        encoding="utf-8",
    )

    errors = validate_hf_space_deployment_smoke(tmp_path)

    assert (
        "demo/hf_space_deployment_smoke.toml: deployment_status must be one of not_published, published"
        in errors
    )
    assert (
        "demo/hf_space_deployment_smoke.toml: smoke_status must be one of passed, pending"
        in errors
    )


def test_hf_space_deployment_smoke_main_reports_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI entrypoint should print the success marker for a valid packet."""
    assert main(["--root", str(ROOT)]) == 0

    captured = capsys.readouterr()
    assert captured.out == "hf_space_deployment_smoke_ok\n"
    assert captured.err == ""


def test_hf_space_deployment_smoke_main_reports_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI entrypoint should return non-zero and print validation errors."""
    assert main(["--root", str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "demo/hf_space_deployment_smoke.toml: missing TOML file\n"
