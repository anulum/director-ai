# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Hugging Face Space real-surface tests
"""Real subprocess coverage for the Hugging Face Space package validator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_hf_space_demo.py"


def _run_validator(root: Path) -> subprocess.CompletedProcess[str]:
    """Run the production HF Space validator CLI for ``root``."""
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(root)],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )


def _write_minimal_space(root: Path, push_script: str) -> None:
    """Write a minimal Hugging Face Space package fixture."""
    demo = root / "demo"
    demo.mkdir(parents=True, exist_ok=True)
    (demo / "README_HF.md").write_text(
        """---
title: Director-AI Guardrail
sdk: gradio
sdk_version: "6.7.0"
app_file: app.py
license: apache-2.0
---

# Demo
""",
        encoding="utf-8",
    )
    (demo / "requirements.txt").write_text(
        "director-ai>=3.10.0,<4.0.0\ngradio>=6.7.0,<7.0\n",
        encoding="utf-8",
    )
    (demo / "hf_space_manifest.toml").write_text(
        """schema_version = "1.0.0"
space_slug = "anulum/director-ai-guardrail"
readme = "demo/README_HF.md"
app_file = "demo/app.py"
requirements = "demo/requirements.txt"
deploy_script = "demo/push_to_hf.sh"
publish_by_default = false
files = ["app.py", "requirements.txt", "README.md"]
""",
        encoding="utf-8",
    )
    (demo / "app.py").write_text(
        "import gradio as gr\n\n"
        "def build_app() -> gr.Blocks:\n"
        "    return gr.Blocks()\n\n"
        'if __name__ == "__main__":\n'
        "    build_app().launch()\n",
        encoding="utf-8",
    )
    (demo / "push_to_hf.sh").write_text(push_script, encoding="utf-8")


def _valid_push_script() -> str:
    """Return the explicit-file deploy script shape required by the validator."""
    return (
        "#!/usr/bin/env bash\n"
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        'TMP_DIR="$(mktemp -d)"\n'
        'cp "$SCRIPT_DIR/app.py" "$TMP_DIR/app.py"\n'
        'cp "$SCRIPT_DIR/requirements.txt" "$TMP_DIR/requirements.txt"\n'
        'cp "$SCRIPT_DIR/README_HF.md" "$TMP_DIR/README.md"\n'
        'cd "$TMP_DIR"\n'
        "git add app.py requirements.txt README.md\n"
    )


def test_hf_space_demo_unit_guard_has_real_cli_companion() -> None:
    """The package guard should be reclassified only with a real CLI companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_hf_space_demo.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_hf_space_demo_real_surface.py" in category


def test_hf_space_demo_cli_accepts_checked_in_package() -> None:
    """The production CLI should validate the checked-in Space package."""
    result = _run_validator(ROOT)

    assert result.returncode == 0
    assert result.stdout == "hf_space_demo_ok\n"
    assert result.stderr == ""


def test_hf_space_demo_cli_rejects_implicit_git_add(tmp_path: Path) -> None:
    """The production CLI should reject broad git staging in the push helper."""
    _write_minimal_space(
        tmp_path,
        _valid_push_script().replace(
            "git add app.py requirements.txt README.md",
            "git add -A",
        ),
    )

    result = _run_validator(tmp_path)

    assert result.returncode == 1
    assert result.stdout == ""
    assert (
        "demo/push_to_hf.sh: use explicit Space files instead of git add -A"
        in result.stderr
    )
    assert (
        "demo/push_to_hf.sh: must stage only app.py requirements.txt README.md"
        in result.stderr
    )


def test_hf_space_demo_cli_rejects_unlaunchable_app(tmp_path: Path) -> None:
    """The production CLI should reject app packages without main-guard launch."""
    _write_minimal_space(tmp_path, _valid_push_script())
    app = tmp_path / "demo" / "app.py"
    app.write_text(
        "import gradio as gr\n\n"
        "def build_app() -> gr.Blocks:\n"
        "    return gr.Blocks()\n",
        encoding="utf-8",
    )

    result = _run_validator(tmp_path)

    assert result.returncode == 1
    assert result.stdout == ""
    assert "demo/app.py: must launch only from the main guard" in result.stderr
