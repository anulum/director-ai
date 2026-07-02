# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Hugging Face Space deployment smoke real-surface tests
"""Real subprocess coverage for the Space deployment-smoke validator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_hf_space_deployment_smoke.py"


def _run_validator(
    root: Path,
    *,
    require_published: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run the production deployment-smoke validator CLI for ``root``."""
    command = [sys.executable, str(VALIDATOR), "--root", str(root)]
    if require_published:
        command.append("--require-published")
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )


def _write_manifest(root: Path) -> None:
    """Write the minimal deployment manifest surfaces required by the CLI."""
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
) -> None:
    """Write a deployment-smoke packet fixture for subprocess validation."""
    _write_manifest(root)
    (root / "demo" / "hf_space_deployment_smoke.toml").write_text(
        f"""
schema_version = "1.0.0"
packet_id = "hf-space-deployment-smoke-real-surface-test"
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
required_smoke_checks = [
  "space_http_200",
  "gradio_app_loads",
  "score_response_smoke",
  "streaming_halt_tab_smoke",
]
""".strip(),
        encoding="utf-8",
    )


def test_hf_space_deployment_smoke_unit_guard_has_real_cli_companion() -> None:
    """The unit guard should be reclassified only with a real CLI companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_hf_space_deployment_smoke.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_hf_space_deployment_smoke_real_surface.py" in category


def test_hf_space_deployment_smoke_cli_accepts_checked_in_packet() -> None:
    """The production CLI should validate the checked-in pending packet."""
    result = _run_validator(ROOT)

    assert result.returncode == 0
    assert result.stdout == "hf_space_deployment_smoke_ok\n"
    assert result.stderr == ""


def test_hf_space_deployment_smoke_cli_rejects_unpublished_strict_mode() -> None:
    """The production CLI should fail strict mode before manual publication."""
    result = _run_validator(ROOT, require_published=True)

    assert result.returncode == 1
    assert result.stdout == ""
    assert (
        "demo/hf_space_deployment_smoke.toml: --require-published requires published deployment and passed smoke"
        in result.stderr
    )


def test_hf_space_deployment_smoke_cli_accepts_published_packet(
    tmp_path: Path,
) -> None:
    """The production CLI should accept a published packet with smoke evidence."""
    evidence = tmp_path / "docs" / "internal" / "hf-space-smoke.md"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(
        "Space HTTP 200 and both app smoke checks passed.\n", encoding="utf-8"
    )
    _write_packet(
        tmp_path,
        deployment_status="published",
        smoke_status="passed",
        public_demo_claim=True,
        smoke_evidence_path="docs/internal/hf-space-smoke.md",
    )

    result = _run_validator(tmp_path, require_published=True)

    assert result.returncode == 0
    assert result.stdout == "hf_space_deployment_smoke_ok\n"
    assert result.stderr == ""


def test_hf_space_deployment_smoke_cli_rejects_invalid_packet_toml(
    tmp_path: Path,
) -> None:
    """The production CLI should reject malformed packet TOML."""
    _write_manifest(tmp_path)
    (tmp_path / "demo" / "hf_space_deployment_smoke.toml").write_text(
        "[invalid\n",
        encoding="utf-8",
    )

    result = _run_validator(tmp_path)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith(
        "demo/hf_space_deployment_smoke.toml: invalid TOML:"
    )
