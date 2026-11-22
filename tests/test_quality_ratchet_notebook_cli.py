# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native notebook typing CLI tests
"""Exercise real native notebook diagnostics and committed baseline comparison.

The capture runs under the pinned quality interpreter alone; one test runs it
with the lane's own interpreter and expects the engine check's refusal.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.quality_ratchet_python_support import quality_python
from tests.test_quality_ratchet_typing_cli import commit_typing_sources
from tests.test_quality_ratchet_typing_cli import typing_repository as typing_repository
from tools.quality_ratchet_contracts import parse_snapshot

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = "notebooks/streaming.ipynb"
METADATA = {
    "kernelspec": {"language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def _cli(
    root: Path, command: str, *arguments: str, python: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the notebook profile via the production CLI and native MyPy.

    The pinned quality interpreter runs it unless ``python`` names another.
    """
    return subprocess.run(
        [
            python or quality_python(),
            "-B",
            "-m",
            "tools.check_quality_ratchet",
            "--profile",
            "notebook-types",
            "--root",
            str(root),
            command,
            *arguments,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )


def _write(root: Path, *cells: object) -> Path:
    """Create a versioned notebook with one Markdown cell and ordered Python cells."""
    path = root / NOTEBOOK
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": METADATA,
                "cells": [
                    {
                        "cell_type": "markdown",
                        "id": "heading",
                        "metadata": {},
                        "source": "# Stream evaluation",
                    },
                    *(
                        {
                            "cell_type": "code",
                            "id": f"code-{index}",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": text,
                        }
                        for index, text in enumerate(cells)
                    ),
                ],
            }
        )
    )
    return path


def test_notebook_typing_keeps_cell_positions_and_checks_await(
    typing_repository: Path, tmp_path: Path
) -> None:
    """Report the actual error after a native top-level await with cell provenance."""
    _write(
        typing_repository,
        [
            "!pip install director-ai\n",
            "from typing import Literal\n",
            'command: Literal["!pip install director-ai"] = """!pip install director-ai"""\n',
            'text: Literal["start\\n!pip install director-ai"] = """start\n',
            '!pip install director-ai"""\n',
            "async def read_count() -> int:\n",
            "    return 42\n",
            "# Evaluate the returned count.\n",
        ],
        "count: str = await read_count()\n",
    )
    commit_typing_sources(typing_repository)
    output = tmp_path / "capture.json"
    result = _cli(typing_repository, "capture", "--output", str(output))
    assert result.returncode == 0, result.stderr
    snapshot = parse_snapshot(json.loads(output.read_bytes()))
    assert set(snapshot.sources) == {NOTEBOOK}
    assert [(f.path, f.cell, f.line, f.rule) for f in snapshot.findings] == [
        (NOTEBOOK, 3, 1, "assignment"),
    ]
    assert any(
        advisory == f"notebook-native-top-level-await:{NOTEBOOK}:3:1"
        for advisory in snapshot.advisories
    )
    assert any(
        advisory.startswith(f"notebook-projection-sha256:{NOTEBOOK}:")
        for advisory in snapshot.advisories
    )


def test_notebook_typing_rejects_new_debt_from_changed_snapshot(
    typing_repository: Path,
) -> None:
    """Allow a clean historical notebook and refuse a new incompatible assignment."""
    _write(typing_repository, 'label: str = "score"')
    commit_typing_sources(typing_repository)
    baseline = typing_repository / "tools/quality_ratchet_baseline.json"
    recorded = _cli(typing_repository, "record", "--output", str(baseline))
    assert recorded.returncode == 0, recorded.stderr
    commit_typing_sources(typing_repository)
    checked = _cli(typing_repository, "check")
    assert checked.returncode == 0, checked.stderr
    _write(typing_repository, "label: str = 3")
    refused = _cli(typing_repository, "check")
    assert refused.returncode == 1, refused.stderr
    assert f"New or modified source has native findings: {NOTEBOOK}" in refused.stdout


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        ("!pip install {package}", "Unsupported notebook shell escape"),
        ("!pip install it's", "Unsupported notebook shell escape"),
        ("!ls 'x", "Unsupported notebook shell escape"),
        ("%%bash\necho example\n", "Unsupported notebook Python syntax"),
        (
            "value = (\n!pip install director-ai\n)\n",
            "Unsupported notebook Python syntax",
        ),
        (
            "value: int = 'wrong' # type: ignore[assignment]",
            "Notebook typing suppression",
        ),
        (
            "# mypy: disable-error-code=assignment\nvalue: int = 'bad'",
            "Notebook typing suppression",
        ),
        (
            "def run() -> None:\n    await other()\n",
            "Unsupported notebook Python syntax",
        ),
        (["value = 1\n", 2], "Invalid notebook cell source"),
    ],
)
def test_notebook_typing_refuses_source_the_projection_cannot_qualify(
    typing_repository: Path, tmp_path: Path, source: object, reason: str
) -> None:
    """Leave no snapshot when a cell uses unsupported or masking syntax."""
    _write(typing_repository, source)
    commit_typing_sources(typing_repository)
    output = tmp_path / "capture.json"
    result = _cli(typing_repository, "capture", "--output", str(output))
    assert result.returncode == 2 and reason in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"nbformat": 3}, "Unsupported notebook format or kernel"),
        (
            {
                "metadata": {
                    "kernelspec": {"language": "julia"},
                    "language_info": {"name": "julia"},
                },
            },
            "Unsupported notebook format or kernel",
        ),
        (
            {"cells": [{"cell_type": "script", "source": "value = 3"}]},
            "Unsupported notebook cell type",
        ),
        (
            {"cells": [{"cell_type": "markdown", "source": "# Overview"}]},
            "No notebook code cells",
        ),
    ],
)
def test_notebook_typing_requires_supported_notebook_structure(
    typing_repository: Path,
    tmp_path: Path,
    changes: dict[str, object],
    reason: str,
) -> None:
    """Check that an unsupported notebook format or cell layout produces no capture."""
    notebook = _write(typing_repository, "value = 3")
    payload: dict[str, object] = json.loads(notebook.read_bytes())
    payload.update(changes)
    notebook.write_text(json.dumps(payload))
    commit_typing_sources(typing_repository)
    output = tmp_path / "capture.json"
    result = _cli(typing_repository, "capture", "--output", str(output))
    assert result.returncode == 2 and reason in result.stderr
    assert not output.exists()


def test_notebook_typing_preserves_imported_native_findings(
    typing_repository: Path, tmp_path: Path
) -> None:
    """Keep a dependency error under its original source identity."""
    imported = typing_repository / "src/legacy.py"
    imported.write_text('value: int = "wrong"\n')
    _write(
        typing_repository,
        "from typing import reveal_type\n"
        "from legacy import value\n"
        "reveal_type(value)\n",
    )
    commit_typing_sources(typing_repository)
    output = tmp_path / "capture.json"
    result = _cli(typing_repository, "capture", "--output", str(output))
    assert result.returncode == 0, result.stderr
    snapshot = parse_snapshot(json.loads(output.read_bytes()))
    assert set(snapshot.sources) == {NOTEBOOK, "src/legacy.py"}
    assert [(f.path, f.cell, f.line, f.rule) for f in snapshot.findings] == [
        ("src/legacy.py", None, 1, "assignment"),
    ]


def test_notebook_typing_accepts_native_notes_without_errors(
    typing_repository: Path, tmp_path: Path
) -> None:
    """Preserve a successful native check that also reports an inferred type."""
    _write(typing_repository, "from typing import reveal_type\nreveal_type(42)\n")
    commit_typing_sources(typing_repository)
    output = tmp_path / "capture.json"
    result = _cli(typing_repository, "capture", "--output", str(output))
    assert result.returncode == 0, result.stderr
    snapshot = parse_snapshot(json.loads(output.read_bytes()))
    assert set(snapshot.sources) == {NOTEBOOK}
    assert not snapshot.findings


def test_notebook_typing_preserves_findings_with_ambient_pythonpath(
    typing_repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep an imported repository diagnostic when the caller sets PYTHONPATH."""
    benchmarks = typing_repository / "benchmarks"
    benchmarks.mkdir()
    (benchmarks / "__init__.py").write_text('"""Notebook benchmark inputs."""\n')
    imported = benchmarks / "inputs.py"
    imported.write_text('value: int = "wrong"\n')
    _write(typing_repository, "from benchmarks.inputs import value\n")
    commit_typing_sources(typing_repository)

    monkeypatch.delenv("PYTHONPATH", raising=False)
    clean = tmp_path / "clean.json"
    result = _cli(typing_repository, "capture", "--output", str(clean))
    assert result.returncode == 0, result.stderr
    original = parse_snapshot(json.loads(clean.read_bytes()))
    assert [(f.path, f.rule) for f in original.findings] == [
        ("benchmarks/inputs.py", "assignment"),
    ]

    monkeypatch.setenv("PYTHONPATH", str(typing_repository))
    ambient = tmp_path / "ambient.json"
    result = _cli(typing_repository, "capture", "--output", str(ambient))
    assert result.returncode == 0, result.stderr
    assert parse_snapshot(json.loads(ambient.read_bytes())) == original


def test_the_lane_interpreter_is_refused_by_the_engine_check(
    typing_repository: Path, tmp_path: Path
) -> None:
    """Refuse by name a capture run by the lane's own setup-python build."""
    _write(typing_repository, 'label: str = "score"')
    commit_typing_sources(typing_repository)
    output = tmp_path / "capture.json"
    result = _cli(
        typing_repository, "capture", "--output", str(output), python=sys.executable
    )
    assert result.returncode == 2
    assert re.search(
        r"Native quality refused: The running Python (binary|tokenize|token) "
        r"differs from its verified artefact",
        result.stderr,
    ), result.stderr
    assert not output.exists()
