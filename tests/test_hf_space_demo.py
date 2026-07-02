# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Hugging Face Space package validation tests
"""Unit guard for Hugging Face Space package validation rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.validate_hf_space_demo import main, validate_hf_space_demo

ROOT = Path(__file__).resolve().parents[1]


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


def test_hf_space_demo_validates_current_package() -> None:
    """The checked-in Space package should satisfy the package validator."""
    assert validate_hf_space_demo(ROOT) == []


def test_hf_space_demo_rejects_implicit_git_add(tmp_path: Path) -> None:
    """The validator should reject broad git staging in the push helper."""
    _write_minimal_space(
        tmp_path,
        _valid_push_script().replace(
            "git add app.py requirements.txt README.md",
            "git add -A",
        ),
    )

    errors = validate_hf_space_demo(tmp_path)

    assert (
        "demo/push_to_hf.sh: use explicit Space files instead of git add -A" in errors
    )


def test_hf_space_demo_rejects_missing_app_file_metadata(tmp_path: Path) -> None:
    """The validator should reject README metadata without app_file."""
    _write_minimal_space(
        tmp_path,
        _valid_push_script(),
    )
    readme = tmp_path / "demo" / "README_HF.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace("app_file: app.py\n", ""),
        encoding="utf-8",
    )

    errors = validate_hf_space_demo(tmp_path)

    assert "demo/README_HF.md: app_file must be app.py" in errors


def test_hf_space_demo_rejects_missing_package_files(tmp_path: Path) -> None:
    """The validator should report every missing package surface."""
    errors = validate_hf_space_demo(tmp_path)

    assert "demo/README_HF.md: missing file" in errors
    assert "demo/requirements.txt: missing file" in errors
    assert "demo/hf_space_manifest.toml: missing file" in errors
    assert "demo/app.py: missing file" in errors
    assert "demo/push_to_hf.sh: missing file" in errors


def test_hf_space_demo_rejects_missing_front_matter(tmp_path: Path) -> None:
    """The validator should reject README files without closed metadata."""
    _write_minimal_space(tmp_path, _valid_push_script())
    (tmp_path / "demo" / "README_HF.md").write_text("# Demo\n", encoding="utf-8")

    assert validate_hf_space_demo(tmp_path) == [
        "demo/README_HF.md: missing YAML front matter"
    ]

    (tmp_path / "demo" / "README_HF.md").write_text(
        "---\ntitle: Demo\n",
        encoding="utf-8",
    )

    assert validate_hf_space_demo(tmp_path) == [
        "demo/README_HF.md: missing YAML front matter"
    ]


def test_hf_space_demo_rejects_readme_metadata_drift(tmp_path: Path) -> None:
    """The validator should reject drifted README metadata fields."""
    _write_minimal_space(tmp_path, _valid_push_script())
    (tmp_path / "demo" / "README_HF.md").write_text(
        """---
# comment without colon
title: ""
sdk: streamlit
sdk_version: "0.0.0"
app_file: demo.py
license: mit
---

# Demo
""",
        encoding="utf-8",
    )

    errors = validate_hf_space_demo(tmp_path)

    assert "demo/README_HF.md: sdk must be gradio" in errors
    assert "demo/README_HF.md: sdk_version must be 6.7.0" in errors
    assert "demo/README_HF.md: app_file must be app.py" in errors
    assert "demo/README_HF.md: license must be apache-2.0" in errors
    assert "demo/README_HF.md: title must be set" in errors


def test_hf_space_demo_rejects_requirement_drift(tmp_path: Path) -> None:
    """The validator should require pinned Space runtime dependencies."""
    _write_minimal_space(tmp_path, _valid_push_script())
    (tmp_path / "demo" / "requirements.txt").write_text(
        "director-ai>=3.10.0,<4.0.0\n",
        encoding="utf-8",
    )

    assert validate_hf_space_demo(tmp_path) == [
        "demo/requirements.txt: missing requirement gradio>=6.7.0,<7.0"
    ]


def test_hf_space_demo_rejects_invalid_manifest(tmp_path: Path) -> None:
    """The validator should reject invalid TOML in the manifest."""
    _write_minimal_space(tmp_path, _valid_push_script())
    (tmp_path / "demo" / "hf_space_manifest.toml").write_text(
        "schema_version = [",
        encoding="utf-8",
    )

    errors = validate_hf_space_demo(tmp_path)

    assert len(errors) == 1
    assert errors[0].startswith("demo/hf_space_manifest.toml: invalid TOML:")


def test_hf_space_demo_rejects_manifest_drift(tmp_path: Path) -> None:
    """The validator should reject stale manifest paths and file lists."""
    _write_minimal_space(tmp_path, _valid_push_script())
    manifest = tmp_path / "demo" / "hf_space_manifest.toml"
    manifest.write_text(
        """schema_version = "1.0.0"
space_slug = "anulum/director-ai-guardrail"
readme = "README.md"
app_file = "demo.py"
requirements = "requirements.txt"
deploy_script = "push.sh"
publish_by_default = true
files = ["app.py"]
""",
        encoding="utf-8",
    )

    errors = validate_hf_space_demo(tmp_path)

    assert "demo/hf_space_manifest.toml: readme must be demo/README_HF.md" in errors
    assert "demo/hf_space_manifest.toml: app_file must be demo/app.py" in errors
    assert (
        "demo/hf_space_manifest.toml: requirements must be demo/requirements.txt"
        in errors
    )
    assert (
        "demo/hf_space_manifest.toml: deploy_script must be demo/push_to_hf.sh"
        in errors
    )
    assert "demo/hf_space_manifest.toml: publish_by_default must be false" in errors
    assert (
        "demo/hf_space_manifest.toml: files must be ['app.py', 'requirements.txt', 'README.md']"
        in errors
    )


def test_hf_space_demo_rejects_manifest_missing_declared_path(
    tmp_path: Path,
) -> None:
    """The validator should reject manifest-declared files that are absent."""
    _write_minimal_space(tmp_path, _valid_push_script())
    (tmp_path / "demo" / "app.py").unlink()

    errors = validate_hf_space_demo(tmp_path)

    assert "demo/hf_space_manifest.toml: declared path is missing: demo/app.py" in (
        errors
    )
    assert "demo/app.py: missing file" in errors


def test_hf_space_demo_rejects_invalid_app_python(tmp_path: Path) -> None:
    """The validator should report invalid Python syntax in the Space app."""
    _write_minimal_space(tmp_path, _valid_push_script())
    (tmp_path / "demo" / "app.py").write_text(
        "def build_app(:\n",
        encoding="utf-8",
    )

    errors = validate_hf_space_demo(tmp_path)

    assert len(errors) == 1
    assert errors[0].startswith("demo/app.py: invalid Python syntax:")


def test_hf_space_demo_rejects_unwired_app(tmp_path: Path) -> None:
    """The validator should require build_app, Gradio, and guarded launch."""
    _write_minimal_space(tmp_path, _valid_push_script())
    (tmp_path / "demo" / "app.py").write_text(
        "def other() -> object:\n    return object()\n",
        encoding="utf-8",
    )

    errors = validate_hf_space_demo(tmp_path)

    assert "demo/app.py: build_app must be defined" in errors
    assert "demo/app.py: must import gradio" in errors
    assert "demo/app.py: must launch only from the main guard" in errors


def test_hf_space_demo_accepts_gradio_from_import(tmp_path: Path) -> None:
    """The validator should accept ``from gradio`` imports."""
    _write_minimal_space(tmp_path, _valid_push_script())
    (tmp_path / "demo" / "app.py").write_text(
        "from gradio import Blocks\n\n"
        "def build_app() -> Blocks:\n"
        "    return Blocks()\n\n"
        'if __name__ == "__main__":\n'
        "    build_app().launch()\n",
        encoding="utf-8",
    )

    assert validate_hf_space_demo(tmp_path) == []


def test_hf_space_demo_rejects_push_script_drift(tmp_path: Path) -> None:
    """The validator should require safe shell shape and explicit copies."""
    _write_minimal_space(
        tmp_path,
        "#!/bin/sh\ngit add .\n",
    )

    errors = validate_hf_space_demo(tmp_path)

    assert "demo/push_to_hf.sh: must start with a bash shebang" in errors
    assert (
        "demo/push_to_hf.sh: use explicit Space files instead of git add -A" in errors
    )
    assert (
        "demo/push_to_hf.sh: must stage only app.py requirements.txt README.md"
        in errors
    )
    assert "demo/push_to_hf.sh: must copy app.py to app.py" in errors
    assert (
        "demo/push_to_hf.sh: must copy requirements.txt to requirements.txt" in errors
    )
    assert "demo/push_to_hf.sh: must copy README_HF.md to README.md" in errors


def test_hf_space_demo_main_reports_success_and_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI entrypoint should print operator-readable results."""
    assert main([str(ROOT)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "hf_space_demo_ok\n"
    assert captured.err == ""

    assert main([str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "demo/README_HF.md: missing file\n" in captured.err
