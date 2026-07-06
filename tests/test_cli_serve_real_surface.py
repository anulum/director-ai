# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real subprocess coverage for the serve CLI boundary."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from director_ai.cli import main


def test_serve_help_dispatcher_prints_transport_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public CLI dispatcher should expose serve options without side effects."""
    main(["serve", "--help"])

    captured = capsys.readouterr()
    assert "Usage: director-ai serve" in captured.out
    assert "--transport http|grpc" in captured.out
    assert "Starting Director AI server" not in captured.out


def test_serve_help_exits_without_starting_server() -> None:
    """Serve help must print transport options without importing server runtime."""
    env = {
        **os.environ,
        "DIRECTOR_AUDIT_SALT": "test-serve-help-salt",
        "DIRECTOR_FORCE_CPU": "1",
    }

    result = subprocess.run(
        [sys.executable, "-m", "director_ai.cli", "serve", "--help"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=8,
    )

    assert result.returncode == 0
    assert "Usage: director-ai serve" in result.stdout
    assert "--transport http|grpc" in result.stdout
    assert "Starting Director AI server" not in result.stdout
    assert "Started server process" not in result.stderr


def test_serve_invalid_port_exits_before_runtime_start() -> None:
    """Invalid serve ports should fail through the installed CLI boundary."""
    env = {
        **os.environ,
        "DIRECTOR_AUDIT_SALT": "test-serve-invalid-port-salt",
        "DIRECTOR_FORCE_CPU": "1",
    }

    result = subprocess.run(
        [sys.executable, "-m", "director_ai.cli", "serve", "--port", "not-a-port"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=8,
    )

    assert result.returncode == 1
    assert "invalid port number: not-a-port" in result.stdout
    assert "Starting Director AI server" not in result.stdout
    assert "Started server process" not in result.stderr
