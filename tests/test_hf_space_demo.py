# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Hugging Face Space package validation tests

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_hf_space_demo.py"
SPEC = importlib.util.spec_from_file_location("validate_hf_space_demo", VALIDATOR)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

validate_hf_space_demo = MODULE.validate_hf_space_demo


def _write_minimal_space(root: Path, push_script: str) -> None:
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


def test_hf_space_demo_validates_current_package() -> None:
    assert validate_hf_space_demo(ROOT) == []


def test_hf_space_demo_rejects_implicit_git_add(tmp_path: Path) -> None:
    _write_minimal_space(
        tmp_path,
        "#!/usr/bin/env bash\n"
        "cp demo/app.py app.py\n"
        "cp demo/requirements.txt requirements.txt\n"
        "cp demo/README_HF.md README.md\n"
        "git add -A\n",
    )

    errors = validate_hf_space_demo(tmp_path)

    assert (
        "demo/push_to_hf.sh: use explicit Space files instead of git add -A" in errors
    )


def test_hf_space_demo_rejects_missing_app_file_metadata(tmp_path: Path) -> None:
    _write_minimal_space(
        tmp_path,
        "#!/usr/bin/env bash\n"
        "cp demo/app.py app.py\n"
        "cp demo/requirements.txt requirements.txt\n"
        "cp demo/README_HF.md README.md\n"
        "git add app.py requirements.txt README.md\n",
    )
    readme = tmp_path / "demo" / "README_HF.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace("app_file: app.py\n", ""),
        encoding="utf-8",
    )

    errors = validate_hf_space_demo(tmp_path)

    assert "demo/README_HF.md: app_file must be app.py" in errors
