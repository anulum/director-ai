# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — production scaffold contract tests

from __future__ import annotations

import json
from pathlib import Path

import pytest

from director_ai._cli_production import validate_production_scaffold
from director_ai.cli import main


def _generate_production_scaffold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.chdir(tmp_path)
    main(["quickstart", "--profile", "production"])
    return tmp_path / "director_guard"


def test_production_check_accepts_generated_scaffold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaffold = _generate_production_scaffold(tmp_path, monkeypatch)

    report = validate_production_scaffold(scaffold)

    assert report.passed is True
    assert report.blockers == ()


def test_production_check_cli_outputs_json_for_generated_scaffold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scaffold = _generate_production_scaffold(tmp_path, monkeypatch)
    capsys.readouterr()

    with pytest.raises(SystemExit) as exc_info:
        main(["production-check", "--path", str(scaffold), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exc_info.value.code == 0
    assert payload["passed"] is True
    assert payload["blockers"] == []


def test_production_check_requires_monitoring_credentials_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaffold = _generate_production_scaffold(tmp_path, monkeypatch)
    prometheus = scaffold / "monitoring" / "prometheus.yml"
    prometheus.write_text(
        prometheus.read_text(encoding="utf-8").replace(
            "    authorization:\n"
            "      credentials_file: /etc/prometheus/director-api-key\n",
            "",
        ),
        encoding="utf-8",
    )

    report = validate_production_scaffold(scaffold)

    assert report.passed is False
    assert any(check.code == "prometheus:auth_file" for check in report.blockers)


def test_production_check_require_secrets_fails_on_blank_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaffold = _generate_production_scaffold(tmp_path, monkeypatch)

    report = validate_production_scaffold(scaffold, require_secrets=True)

    assert report.passed is False
    assert any(
        check.code == "secret:DIRECTOR_API_KEY_TENANT_MAP" for check in report.blockers
    )


def test_production_check_require_secrets_accepts_filled_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaffold = _generate_production_scaffold(tmp_path, monkeypatch)
    env_path = scaffold / ".env"
    env_path.write_text(
        "\n".join(
            [
                'DIRECTOR_API_KEY_TENANT_MAP={"director-service-key":"tenant-a"}',
                "DIRECTOR_PROXY_API_KEYS=director-service-key",
                "DIRECTOR_LLM_API_URL=https://llm.internal.example/v1",
                "DIRECTOR_UPSTREAM_URL=https://llm.internal.example",
                "DIRECTOR_KB_HMAC_KEYS=key-id:secret",
                "DIRECTOR_CORS_ORIGINS=https://console.example.com",
                "DIRECTOR_TENANT_ID=tenant-a",
                "DIRECTOR_COHERENCE_THRESHOLD=0.6",
                "DIRECTOR_SERVER_PORT=8000",
                "",
            ],
        ),
        encoding="utf-8",
    )

    report = validate_production_scaffold(scaffold, require_secrets=True)

    assert report.passed is True


def test_production_check_rejects_research_surface_in_default_compose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaffold = _generate_production_scaffold(tmp_path, monkeypatch)
    compose = scaffold / "docker-compose.yml"
    compose.write_text(
        compose.read_text(encoding="utf-8")
        + "\n# accidental default exposure: meta_guard\n",
        encoding="utf-8",
    )

    report = validate_production_scaffold(scaffold)

    assert report.passed is False
    assert any(check.code == "compose:no_meta_guard" for check in report.blockers)


def test_production_check_is_listed_in_help(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--help"])

    assert "production-check" in capsys.readouterr().out
