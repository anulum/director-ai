# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Director-Lite package tests
"""Release-contract tests for the ``director-ai-lite`` package split.

The package is intentionally a thin distribution wrapper around
``director_ai.lite``. These tests pin the public import surface, static package
metadata, and one model-free halt path so the Lite wheel cannot drift away from
the canonical implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:  # pragma: no cover - exercised on Python 3.11+ in CI.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
LITE_PACKAGE = ROOT / "packages" / "director-ai-lite"
LITE_SRC = LITE_PACKAGE / "src"

if str(LITE_SRC) not in sys.path:
    sys.path.insert(0, str(LITE_SRC))


def _tokens(text: str) -> list[str]:
    """Return deterministic whitespace-suffixed tokens for streaming tests."""

    return [token + " " for token in text.split()]


def test_lite_package_metadata_tracks_director_ai_core_version() -> None:
    pyproject = tomllib.loads((LITE_PACKAGE / "pyproject.toml").read_text())
    project = pyproject["project"]

    assert project["name"] == "director-ai-lite"
    assert project["version"] == "3.15.3"
    assert project["requires-python"] == ">=3.11"
    assert "director-ai>=3.15.3,<4" in project["dependencies"]
    assert pyproject["tool"]["setuptools"]["package-data"]["director_ai_lite"] == [
        "py.typed"
    ]


def test_lite_package_exposes_canonical_stream_guard() -> None:
    import director_ai_lite

    from director_ai.lite import StreamGuard as CanonicalStreamGuard

    assert director_ai_lite.StreamGuard is CanonicalStreamGuard
    assert director_ai_lite.__version__ == "3.15.3"
    assert sorted(director_ai_lite.__all__) == [
        "StreamGuard",
        "__version__",
        "guard",
        "streaming_guard",
    ]


def test_lite_guard_runs_model_free_halt_path() -> None:
    from director_ai_lite import guard

    session = guard(
        _tokens("The capital of France is Berlin and then Tokyo"),
        facts={"capital": "Paris is the capital of France."},
        prompt="What is the capital of France?",
        threshold=0.5,
    )

    assert session.halted is True
    assert "Berlin" not in session.output
