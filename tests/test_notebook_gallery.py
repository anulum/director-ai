# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - notebook gallery consistency tests

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_notebook_gallery.py"
SPEC = importlib.util.spec_from_file_location("validate_notebook_gallery", VALIDATOR)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

validate_notebook_gallery = MODULE.validate_notebook_gallery


def _write_notebook(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": ["# Test Notebook\n"],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


def test_notebook_gallery_covers_current_notebooks_and_docs() -> None:
    assert validate_notebook_gallery(ROOT) == []


def test_notebook_gallery_is_linked_from_mkdocs_navigation() -> None:
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    assert "Notebook Gallery: notebook-gallery.md" in mkdocs


def test_notebook_gallery_rejects_unlisted_notebook(tmp_path: Path) -> None:
    _write_notebook(tmp_path / "notebooks" / "listed.ipynb")
    _write_notebook(tmp_path / "notebooks" / "missing.ipynb")
    (tmp_path / "notebooks" / "gallery.toml").write_text(
        """
[[notebook]]
id = "listed"
path = "notebooks/listed.ipynb"
title = "Listed"
track = "Foundations"
audience = "Evaluator"
duration_minutes = 5
use_case = "Baseline walkthrough."
extras = []
""".strip(),
        encoding="utf-8",
    )
    docs = tmp_path / "docs-site"
    docs.mkdir()
    (docs / "notebook-gallery.md").write_text(
        "<!-- notebook-gallery:generated from notebooks/gallery.toml -->\n"
        "[Listed](../notebooks/listed.ipynb)\n",
        encoding="utf-8",
    )

    errors = validate_notebook_gallery(tmp_path)

    assert errors == [
        "notebooks/gallery.toml: missing manifest entry for notebooks/missing.ipynb"
    ]


def test_notebook_gallery_rejects_stale_docs_page(tmp_path: Path) -> None:
    _write_notebook(tmp_path / "notebooks" / "listed.ipynb")
    (tmp_path / "notebooks" / "gallery.toml").write_text(
        """
[[notebook]]
id = "listed"
path = "notebooks/listed.ipynb"
title = "Listed"
track = "Foundations"
audience = "Evaluator"
duration_minutes = 5
use_case = "Baseline walkthrough."
extras = []
""".strip(),
        encoding="utf-8",
    )
    docs = tmp_path / "docs-site"
    docs.mkdir()
    (docs / "notebook-gallery.md").write_text(
        "<!-- notebook-gallery:generated from notebooks/gallery.toml -->\n",
        encoding="utf-8",
    )

    errors = validate_notebook_gallery(tmp_path)

    assert errors == [
        "docs-site/notebook-gallery.md: missing link for notebooks/listed.ipynb"
    ]
