# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real subprocess coverage for the ingest CLI boundary."""

from __future__ import annotations

import subprocess
import sys

import pytest

from director_ai.cli import main


def test_ingest_help_dispatcher_prints_storage_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public dispatcher should expose ingest options without side effects."""
    main(["ingest", "--help"])

    captured = capsys.readouterr()
    assert "Usage: director-ai ingest" in captured.out
    assert "--persist <dir>" in captured.out
    assert "--chunk-size <tokens>" in captured.out
    assert "path not found" not in captured.out


def test_ingest_help_exits_without_opening_input_path() -> None:
    """Ingest help must return through the CLI entrypoint without file access."""
    result = subprocess.run(
        [sys.executable, "-m", "director_ai.cli", "ingest", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert result.returncode == 0
    assert "Usage: director-ai ingest" in result.stdout
    assert "--persist <dir>" in result.stdout
    assert "--chunk-size <tokens>" in result.stdout
    assert "path not found" not in result.stdout
    assert result.stderr == ""
