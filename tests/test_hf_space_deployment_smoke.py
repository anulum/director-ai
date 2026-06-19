# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Hugging Face Space deployment smoke packet tests

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_hf_space_deployment_smoke.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_hf_space_deployment_smoke", VALIDATOR
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

validate_hf_space_deployment_smoke = MODULE.validate_hf_space_deployment_smoke


def _write_manifest(root: Path) -> None:
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
    _write_manifest(root)
    checks = required_smoke_checks or [
        "space_http_200",
        "gradio_app_loads",
        "score_response_smoke",
        "streaming_halt_tab_smoke",
    ]
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
    assert validate_hf_space_deployment_smoke(ROOT) == []


def test_hf_space_deployment_smoke_require_published_fails_pending() -> None:
    errors = validate_hf_space_deployment_smoke(ROOT, require_published=True)

    assert (
        "demo/hf_space_deployment_smoke.toml: --require-published requires published deployment and passed smoke"
        in errors
    )


def test_hf_space_deployment_smoke_rejects_public_claim_before_smoke(
    tmp_path: Path,
) -> None:
    _write_packet(tmp_path, public_demo_claim=True)

    errors = validate_hf_space_deployment_smoke(tmp_path)

    assert (
        "demo/hf_space_deployment_smoke.toml: public_demo_claim requires published deployment and passed smoke"
        in errors
    )


def test_hf_space_deployment_smoke_requires_evidence_when_passed(
    tmp_path: Path,
) -> None:
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


def test_hf_space_deployment_smoke_accepts_published_packet_with_evidence(
    tmp_path: Path,
) -> None:
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
