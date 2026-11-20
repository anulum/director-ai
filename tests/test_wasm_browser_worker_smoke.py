# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — WASM browser worker smoke tests

"""Exercise the WebAssembly browser Worker result parser and smoke CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_wasm_browser_worker_smoke import (
    _parse_worker_result_text,
    main,
    run_wasm_browser_worker_smoke,
)


def test_parse_worker_result_text_extracts_worker_json() -> None:
    """Decode a passing Worker message into the result mapping."""
    payload = _parse_worker_result_text('{"passed":true}')

    assert payload == {"passed": True}


def test_parse_worker_result_text_reports_invalid_json() -> None:
    """Keep malformed Worker output with its JSON parser error."""
    payload = _parse_worker_result_text("pending")

    assert payload["passed"] is False
    assert payload["error"] == "result JSON invalid: Expecting value"
    assert payload["raw"] == "pending"


def test_browser_worker_smoke_blocks_missing_package(tmp_path: Path) -> None:
    """Report a missing package without starting a browser."""
    report = run_wasm_browser_worker_smoke(tmp_path / "missing")

    assert report.passed is False
    assert any(blocker["code"] == "package_dir_missing" for blocker in report.blockers)


def test_browser_worker_smoke_cli_writes_json_with_mocked_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write the serialised report through the CLI entry point."""
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    output = tmp_path / "smoke.json"

    monkeypatch.setattr(
        "tools.run_wasm_browser_worker_smoke._find_chrome",
        lambda: tmp_path / "chrome",
    )
    monkeypatch.setattr(
        "tools.run_wasm_browser_worker_smoke._chrome_version",
        lambda _chrome: "Chrome 1.0",
    )
    monkeypatch.setattr(
        "tools.run_wasm_browser_worker_smoke._run_chrome_worker_result",
        lambda *_args, **_kwargs: {
            "passed": True,
            "first_halted": False,
            "second_halted": True,
            "third_halted": True,
            "halt_reason": "hard_limit: 0.1",
            "active_after_halt": False,
            "token_count": 2,
        },
    )

    exit_code = main(
        [
            "--package-dir",
            str(package_dir),
            "--json",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["schema_version"] == "director-ai.wasm-browser-worker-smoke.v1"
    assert payload["passed"] is True
    assert payload["runtime"] == "headless-chrome-web-worker"


def test_browser_worker_smoke_blocks_worker_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Translate a failed Worker result into a smoke blocker."""
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()

    monkeypatch.setattr(
        "tools.run_wasm_browser_worker_smoke._find_chrome",
        lambda: tmp_path / "chrome",
    )
    monkeypatch.setattr(
        "tools.run_wasm_browser_worker_smoke._chrome_version",
        lambda _chrome: "Chrome 1.0",
    )
    monkeypatch.setattr(
        "tools.run_wasm_browser_worker_smoke._run_chrome_worker_result",
        lambda *_args, **_kwargs: {"passed": False, "error": "worker boom"},
    )

    report = run_wasm_browser_worker_smoke(package_dir)

    assert report.passed is False
    assert report.worker_result["error"] == "worker boom"
    assert any(blocker["code"] == "worker_smoke_failed" for blocker in report.blockers)


def test_browser_worker_smoke_keeps_sandbox_on_startup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launch the configured browser once with its sandbox settings intact."""
    launches = tmp_path / "launches.txt"
    monkeypatch.setenv("DIRECTOR_BROWSER_TEST_LAUNCHES", str(launches))
    monkeypatch.chdir(tmp_path)
    browser = tmp_path / "launchers/chrome"
    browser.parent.mkdir()
    browser.write_text(
        """#!/bin/sh
if [ "$1" = "--version" ]; then
    printf 'Test Chrome\\n'
    exit 0
fi
printf '%s\\n' "$PWD" "$TMPDIR" "$@" >> "$DIRECTOR_BROWSER_TEST_LAUNCHES"
printf 'Browser sandbox startup failed\\n' >&2
exit 1
""",
        encoding="utf-8",
    )
    browser.chmod(0o700)

    report = run_wasm_browser_worker_smoke(
        tmp_path, chrome_path=browser.relative_to(tmp_path), timeout_seconds=3
    )

    directory, temporary, *args = launches.read_text(encoding="utf-8").splitlines()
    assert temporary == "."
    assert f"--user-data-dir={directory}" in args
    assert args.count("--headless=new") == 1
    assert "--no-sandbox" not in args
    assert "--disable-setuid-sandbox" not in args
    assert not report.passed
    assert report.blockers[0]["code"] == "worker_smoke_failed"
    assert "Browser sandbox startup failed" in report.worker_result["error"]
