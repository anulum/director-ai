# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - cross-language contract validator real-surface tests

"""Real subprocess coverage for the cross-language contract validator."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_cross_language_contracts.py"


def _run_validator(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cross_language_contract_validator_accepts_repo_manifest() -> None:
    result = _run_validator("--root", str(ROOT), "--json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    assert payload["summary"] == {
        "boundaries": 3,
        "gates": 4,
        "required_languages": ["go", "python", "rust"],
    }
    assert {boundary["id"] for boundary in payload["boundaries"]} == {
        "go-proto-v1",
        "python-proto-v1",
        "rust-python-score",
    }
    assert all(boundary["tests"] for boundary in payload["boundaries"])
    assert {gate["id"] for gate in payload["gates"]} == {
        "go-contracts",
        "manifest-contract",
        "python-contracts",
        "rust-contracts",
    }


def test_cross_language_contract_validator_rejects_missing_paths(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "requirements" / "cross_language_contracts.toml"
    manifest.parent.mkdir()
    (tmp_path / "schemas/proto/director/v1").mkdir(parents=True)
    (tmp_path / "src/director_ai/proto").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "schemas/proto/director/v1/director.proto").write_text(
        'syntax = "proto3";\n',
        encoding="utf-8",
    )
    (tmp_path / "src/director_ai/proto/converters.py").write_text(
        "def convert() -> None:\n    return None\n",
        encoding="utf-8",
    )
    (tmp_path / "tests/test_proto_serialization.py").write_text(
        "def test_contract() -> None:\n    assert True\n",
        encoding="utf-8",
    )
    manifest.write_text(
        textwrap.dedent(
            """
            id = "cross-language-contracts"
            status = "active"
            roadmap_item = "fake fixture"

            [[boundaries]]
            id = "python-proto-v1"
            schema = "schemas/proto/director/v1/director.proto"
            implementation = "src/director_ai/proto/converters.py"
            generated = "src/director_ai/proto/director/v1/director_pb2.py"
            tests = [
              "tests/test_cross_language_contracts.py",
              "tests/test_proto_serialization.py",
            ]

            [[gates]]
            id = "python-contracts"
            command = "pytest tests/test_cross_language_contracts.py tests/test_proto_serialization.py"
            """
        ).lstrip(),
        encoding="utf-8",
    )

    result = _run_validator("--root", str(tmp_path), "--json")

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["summary"] == {
        "boundaries": 1,
        "gates": 1,
        "required_languages": ["python"],
    }
    assert {
        "requirements/cross_language_contracts.toml: boundary python-proto-v1 generated path missing: src/director_ai/proto/director/v1/director_pb2.py",
        "requirements/cross_language_contracts.toml: boundary python-proto-v1 test path missing: tests/test_cross_language_contracts.py",
    }.issubset(set(payload["errors"]))
