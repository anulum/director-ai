#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - notebook gallery consistency validator

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

MANIFEST = Path("notebooks/gallery.toml")
GALLERY_PAGE = Path("docs-site/notebook-gallery.md")
GENERATED_MARKER = "<!-- notebook-gallery:generated from notebooks/gallery.toml -->"
REQUIRED_FIELDS = {
    "id",
    "path",
    "title",
    "track",
    "audience",
    "duration_minutes",
    "use_case",
    "extras",
}


def _load_manifest(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"{MANIFEST}: missing notebook gallery manifest"]
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return [], [f"{MANIFEST}: invalid TOML: {exc}"]
    entries = data.get("notebook")
    if not isinstance(entries, list):
        return [], [f"{MANIFEST}: missing [[notebook]] entries"]
    typed_entries = [entry for entry in entries if isinstance(entry, dict)]
    if len(typed_entries) != len(entries):
        return [], [f"{MANIFEST}: every [[notebook]] entry must be a table"]
    return typed_entries, []


def _validate_entry_schema(entry: dict[str, Any], index: int) -> list[str]:
    prefix = f"{MANIFEST}: notebook[{index}]"
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(entry))
    if missing:
        errors.append(f"{prefix}: missing required fields {', '.join(missing)}")
        return errors

    for field in ("id", "path", "title", "track", "audience", "use_case"):
        value = entry[field]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{prefix}: {field} must be a non-empty string")

    duration = entry["duration_minutes"]
    if not isinstance(duration, int) or duration <= 0:
        errors.append(f"{prefix}: duration_minutes must be a positive integer")

    extras = entry["extras"]
    if not isinstance(extras, list) or not all(
        isinstance(item, str) for item in extras
    ):
        errors.append(f"{prefix}: extras must be a list of strings")

    return errors


def _safe_relative_path(
    root: Path, value: str, index: int
) -> tuple[Path | None, list[str]]:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None, [
            f"{MANIFEST}: notebook[{index}]: path must stay inside repository"
        ]
    if (
        relative.suffix != ".ipynb"
        or not relative.parts
        or relative.parts[0] != "notebooks"
    ):
        return None, [f"{MANIFEST}: notebook[{index}]: path must be notebooks/*.ipynb"]
    return root / relative, []


def _validate_notebook_file(root: Path, entry: dict[str, Any], index: int) -> list[str]:
    errors: list[str] = []
    notebook_path, path_errors = _safe_relative_path(
        root, str(entry.get("path", "")), index
    )
    errors.extend(path_errors)
    if notebook_path is None:
        return errors

    relative_path = notebook_path.relative_to(root).as_posix()
    if not notebook_path.exists():
        return [f"{MANIFEST}: missing notebook file {relative_path}"]
    try:
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{relative_path}: invalid notebook JSON: {exc}"]
    if payload.get("nbformat") != 4 or not isinstance(payload.get("cells"), list):
        return [f"{relative_path}: expected nbformat 4 notebook with cells"]
    if not payload["cells"]:
        return [f"{relative_path}: notebook must contain at least one cell"]
    return []


def _validate_manifest_coverage(root: Path, entries: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    actual = {
        path.relative_to(root).as_posix()
        for path in sorted((root / "notebooks").glob("*.ipynb"))
    }
    declared: list[str] = []
    ids: set[str] = set()
    for entry in entries:
        notebook_id = entry.get("id")
        if isinstance(notebook_id, str):
            if notebook_id in ids:
                errors.append(f"{MANIFEST}: duplicate notebook id {notebook_id}")
            ids.add(notebook_id)
        path_value = entry.get("path")
        if isinstance(path_value, str):
            declared.append(path_value)

    declared_set = set(declared)
    for duplicate in sorted({path for path in declared if declared.count(path) > 1}):
        errors.append(f"{MANIFEST}: duplicate manifest entry for {duplicate}")
    for missing in sorted(actual - declared_set):
        errors.append(f"{MANIFEST}: missing manifest entry for {missing}")
    for stale in sorted(declared_set - actual):
        errors.append(f"{MANIFEST}: manifest entry points to missing notebook {stale}")
    return errors


def _validate_gallery_page(root: Path, entries: list[dict[str, Any]]) -> list[str]:
    page = root / GALLERY_PAGE
    if not page.exists():
        return [f"{GALLERY_PAGE}: missing notebook gallery page"]
    text = page.read_text(encoding="utf-8")
    errors: list[str] = []
    if GENERATED_MARKER not in text:
        errors.append(f"{GALLERY_PAGE}: missing generated manifest marker")
    for entry in entries:
        path_value = entry.get("path")
        if not isinstance(path_value, str):
            continue
        accepted_targets = (
            f"../{path_value}",
            f"https://github.com/anulum/director-ai/blob/main/{path_value}",
        )
        if not any(target in text for target in accepted_targets):
            errors.append(f"{GALLERY_PAGE}: missing link for {path_value}")
    return errors


def validate_notebook_gallery(root: Path) -> list[str]:
    root = root.resolve()
    entries, errors = _load_manifest(root / MANIFEST)
    if errors:
        return errors

    for index, entry in enumerate(entries):
        errors.extend(_validate_entry_schema(entry, index))
        errors.extend(_validate_notebook_file(root, entry, index))
    errors.extend(_validate_manifest_coverage(root, entries))
    errors.extend(_validate_gallery_page(root, entries))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="Repository root containing notebooks/gallery.toml",
    )
    args = parser.parse_args(argv)

    errors = validate_notebook_gallery(args.root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("notebook_gallery_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
