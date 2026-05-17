# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - API reference consistency gate tests

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_api_reference.py"
SPEC = importlib.util.spec_from_file_location("validate_api_reference", VALIDATOR)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

validate_api_reference = MODULE.validate_api_reference


def test_api_reference_index_matches_current_docs_and_imports() -> None:
    assert validate_api_reference(ROOT) == []


def test_api_reference_gate_rejects_missing_markdown_target(tmp_path: Path) -> None:
    docs = tmp_path / "docs-site" / "api"
    docs.mkdir(parents=True)
    (docs / "index.md").write_text(
        "# API Reference\n\n"
        "| Symbol | Module | Purpose |\n"
        "|--------|--------|---------|\n"
        "| [`guard()`](missing.md) | `director_ai` | stale link |\n",
        encoding="utf-8",
    )

    errors = validate_api_reference(tmp_path)

    assert errors == [
        "docs-site/api/index.md:5: missing markdown target missing.md"
    ]


def test_api_reference_gate_rejects_missing_importable_symbol(tmp_path: Path) -> None:
    docs = tmp_path / "docs-site" / "api"
    docs.mkdir(parents=True)
    (docs / "guard.md").write_text("# Guard\n\n", encoding="utf-8")
    (docs / "index.md").write_text(
        "# API Reference\n\n"
        "| Symbol | Module | Purpose |\n"
        "|--------|--------|---------|\n"
        "| [`definitely_missing()`](guard.md) | `director_ai` | stale symbol |\n",
        encoding="utf-8",
    )

    errors = validate_api_reference(tmp_path)

    assert errors == [
        "docs-site/api/index.md:5: director_ai does not expose definitely_missing"
    ]
