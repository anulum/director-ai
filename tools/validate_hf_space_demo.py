#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Hugging Face Space demo package validator

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

README = Path("demo/README_HF.md")
REQUIREMENTS = Path("demo/requirements.txt")
MANIFEST = Path("demo/hf_space_manifest.toml")
APP = Path("demo/app.py")
PUSH_SCRIPT = Path("demo/push_to_hf.sh")
REQUIRED_SPACE_FILES = ("app.py", "requirements.txt", "README.md")
REQUIRED_METADATA = {
    "sdk": "gradio",
    "sdk_version": "6.7.0",
    "app_file": "app.py",
    "license": "agpl-3.0",
}


def _read_text(path: Path, label: Path) -> tuple[str, list[str]]:
    if not path.exists():
        return "", [f"{label}: missing file"]
    return path.read_text(encoding="utf-8"), []


def _parse_front_matter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end == -1:
        return {}
    metadata: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata


def _validate_readme(root: Path) -> list[str]:
    text, errors = _read_text(root / README, README)
    if errors:
        return errors
    metadata = _parse_front_matter(text)
    if not metadata:
        return [f"{README}: missing YAML front matter"]

    for key, expected in REQUIRED_METADATA.items():
        actual = metadata.get(key)
        if actual != expected:
            errors.append(f"{README}: {key} must be {expected}")
    if not metadata.get("title", "").strip():
        errors.append(f"{README}: title must be set")
    return errors


def _validate_requirements(root: Path) -> list[str]:
    text, errors = _read_text(root / REQUIREMENTS, REQUIREMENTS)
    if errors:
        return errors
    lines = {line.strip() for line in text.splitlines() if line.strip()}
    required = {
        "director-ai>=3.10.0,<4.0.0",
        "gradio>=6.7.0,<7.0",
    }
    missing = sorted(required - lines)
    return [f"{REQUIREMENTS}: missing requirement {line}" for line in missing]


def _load_manifest(root: Path) -> tuple[dict[str, Any], list[str]]:
    text, errors = _read_text(root / MANIFEST, MANIFEST)
    if errors:
        return {}, errors
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return {}, [f"{MANIFEST}: invalid TOML: {exc}"]
    return data, []


def _validate_manifest(root: Path) -> list[str]:
    manifest, errors = _load_manifest(root)
    if errors:
        return errors

    expected_paths = {
        "readme": README.as_posix(),
        "app_file": APP.as_posix(),
        "requirements": REQUIREMENTS.as_posix(),
        "deploy_script": PUSH_SCRIPT.as_posix(),
    }
    for key, expected in expected_paths.items():
        if manifest.get(key) != expected:
            errors.append(f"{MANIFEST}: {key} must be {expected}")
    if manifest.get("publish_by_default") is not False:
        errors.append(f"{MANIFEST}: publish_by_default must be false")
    files = manifest.get("files")
    if files != list(REQUIRED_SPACE_FILES):
        errors.append(f"{MANIFEST}: files must be {list(REQUIRED_SPACE_FILES)!r}")
    for path in (README, REQUIREMENTS, APP, PUSH_SCRIPT):
        if not (root / path).exists():
            errors.append(f"{MANIFEST}: declared path is missing: {path}")
    return errors


def _validate_app(root: Path) -> list[str]:
    text, errors = _read_text(root / APP, APP)
    if errors:
        return errors
    try:
        tree = ast.parse(text, filename=APP.as_posix())
    except SyntaxError as exc:
        return [f"{APP}: invalid Python syntax: {exc}"]

    functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    imports_gradio = any(
        isinstance(node, ast.Import)
        and any(alias.name == "gradio" for alias in node.names)
        or isinstance(node, ast.ImportFrom)
        and node.module == "gradio"
        for node in ast.walk(tree)
    )
    if "build_app" not in functions:
        errors.append(f"{APP}: build_app must be defined")
    if not imports_gradio:
        errors.append(f"{APP}: must import gradio")
    if 'if __name__ == "__main__"' not in text or ".launch(" not in text:
        errors.append(f"{APP}: must launch only from the main guard")
    return errors


def _validate_push_script(root: Path) -> list[str]:
    text, errors = _read_text(root / PUSH_SCRIPT, PUSH_SCRIPT)
    if errors:
        return errors
    if not text.startswith("#!/usr/bin/env bash\n"):
        errors.append(f"{PUSH_SCRIPT}: must start with a bash shebang")
    for pattern in (r"\bgit\s+add\s+-A\b", r"\bgit\s+add\s+\.\b"):
        if re.search(pattern, text):
            errors.append(
                f"{PUSH_SCRIPT}: use explicit Space files instead of git add -A"
            )
            break
    if "git add app.py requirements.txt README.md" not in text:
        errors.append(
            f"{PUSH_SCRIPT}: must stage only app.py requirements.txt README.md"
        )
    for source, target in (
        ("app.py", "app.py"),
        ("requirements.txt", "requirements.txt"),
        ("README_HF.md", "README.md"),
    ):
        if f'cp "$SCRIPT_DIR/{source}" "$TMP_DIR/{target}"' not in text:
            errors.append(f"{PUSH_SCRIPT}: must copy {source} to {target}")
    return errors


def validate_hf_space_demo(root: Path) -> list[str]:
    root = root.resolve()
    errors: list[str] = []
    errors.extend(_validate_readme(root))
    errors.extend(_validate_requirements(root))
    errors.extend(_validate_manifest(root))
    errors.extend(_validate_app(root))
    errors.extend(_validate_push_script(root))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="Repository root containing the demo Space package",
    )
    args = parser.parse_args(argv)

    errors = validate_hf_space_demo(args.root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("hf_space_demo_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
