# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Director-Lite package tests
"""Release-contract tests for the standalone ``director-ai-lite`` package.

The Lite package is a self-contained distribution: it has no ``director-ai``
runtime dependency and ships its own streaming-halt guard. These tests pin that
standalone contract from the main repository so the wheel cannot drift back into
a facade. The package's own behavioural suite lives in
``packages/director-ai-lite/tests/``.
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


def test_lite_package_is_standalone_apache() -> None:
    pyproject = tomllib.loads((LITE_PACKAGE / "pyproject.toml").read_text())
    project = pyproject["project"]

    assert project["name"] == "director-ai-lite"
    assert project["version"] == "3.18.1"
    assert project["license"] == "Apache-2.0"
    assert project["requires-python"] == ">=3.11"
    # Standalone: no runtime dependencies. The full package is an opt-in extra.
    assert project["dependencies"] == []
    assert "director-ai>=3.18.1,<4" in project["optional-dependencies"]["full"]
    assert pyproject["tool"]["setuptools"]["package-data"]["director_ai_lite"] == [
        "py.typed"
    ]


def test_lite_exposes_its_own_standalone_surface() -> None:
    import director_ai_lite

    assert director_ai_lite.__version__ == "3.18.1"
    assert sorted(director_ai_lite.__all__) == [
        "StreamGuard",
        "StreamResult",
        "__version__",
        "guard",
        "streaming_guard",
    ]
    # The guard is the package's own implementation, not a director_ai re-export.
    assert director_ai_lite.StreamGuard.__module__ == "director_ai_lite.guard"


def test_lite_guard_runs_model_free_halt_path() -> None:
    from director_ai_lite import guard

    # The grounding heuristic is gradual: it halts once the share of ungrounded
    # content words pushes coherence below the threshold, removing the drifted
    # tail rather than a single mid-stream token.
    session = guard(
        _tokens("Paris is the capital of France Berlin Tokyo Mars Jupiter banana"),
        facts={"capital": "Paris is the capital of France."},
        prompt="What is the capital of France?",
        threshold=0.5,
    )

    assert session.halted is True
    assert "banana" not in session.output
