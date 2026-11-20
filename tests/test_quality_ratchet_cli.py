# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — real Git and native Ruff quality CLI acceptance
"""Exercise the quality ratchet through actual subprocesses and immutable Git state."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_quality_ratchet import write_original_json
from tools.ci_workflow_inventory import object_mapping

BASELINE = "tools/quality_ratchet_baseline.json"


def commit_repository(root: Path) -> None:
    """Create an actual local immutable commit without user signing configuration.

    Parameters
    ----------
    root : Path
        Test repository with the exact candidate bytes to commit.
    """
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Quality Test",
            "-c",
            "user.email=quality@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "Record actual test state",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )


def run_cli(
    root: Path, command: str, *arguments: str
) -> subprocess.CompletedProcess[str]:
    """Run the public module CLI with the actual current Python and native Ruff.

    Parameters
    ----------
    root : Path
        Actual selected test Git repository.
    command : str
        Public capture, record or check operation.
    arguments : str
        Additional explicit CLI arguments.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Original process status, standard output and standard error.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.check_quality_ratchet",
            command,
            "--root",
            str(root),
            *arguments,
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


@pytest.fixture
def native_repository(tmp_path: Path) -> Path:
    """Create committed legacy bytes for actual native baseline adoption.

    Parameters
    ----------
    tmp_path : Path
        Native test runner's isolated temporary directory.

    Returns
    -------
    Path
        Actual Git repository containing original config and legacy documentation debt.
    """
    root = tmp_path / "native-quality"
    root.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    for directory in ("src", "tools", "requirements"):
        (root / directory).mkdir()
    (root / "src/legacy.py").write_text("def legacy(value):\n    return value\n")
    (root / "pyproject.toml").write_text(
        '[project]\nname="quality-fixture"\nversion="0.0.0"\n'
    )
    (root / "requirements/ci-lint.txt").write_text("ruff==0.16.10\n")
    commit_repository(root)
    return root


def adopt_repository(root: Path) -> None:
    """Adopt actual original debt and commit the native baseline carrier.

    Parameters
    ----------
    root : Path
        Committed native fixture before initial adoption.
    """
    result = run_cli(root, "record", "--output", str(root / BASELINE))
    assert result.returncode == 0, result.stderr
    commit_repository(root)


def test_cli_initial_adoption_and_committed_check(native_repository: Path) -> None:
    """Accept actual unchanged legacy findings before and after baseline commitment.

    Parameters
    ----------
    native_repository : Path
        Actual committed legacy fixture.
    """
    result = run_cli(
        native_repository, "record", "--output", str(native_repository / BASELINE)
    )
    assert result.returncode == 0, result.stderr
    before = run_cli(native_repository, "check")
    assert before.returncode == 0, before.stderr
    summary = object_mapping(json.loads(before.stdout))
    assert summary["findings"] == 2
    assert summary["complete"] is True
    commit_repository(native_repository)
    after = run_cli(native_repository, "check")
    assert after.returncode == 0, after.stderr


@pytest.mark.parametrize("inline_ignore", (False, True))
def test_cli_new_debt_and_inline_ignore_refuse(
    native_repository: Path, inline_ignore: bool
) -> None:
    """Refuse genuine new source debt even when it has an inline noqa suppression.

    Parameters
    ----------
    native_repository : Path
        Actual fixture with a committed original baseline.
    inline_ignore : bool
        Whether the new source attempts a file-level documentation suppression.
    """
    adopt_repository(native_repository)
    prefix = "# ruff: noqa: D\n" if inline_ignore else ""
    (native_repository / "src/new.py").write_text(
        prefix + "def introduced():\n    return 1\n"
    )
    result = run_cli(native_repository, "check")
    assert result.returncode == 1, result.stderr
    assert "New or modified source has native findings: src/new.py" in result.stdout


def test_cli_changed_legacy_cannot_keep_debt(native_repository: Path) -> None:
    """Require a changed legacy file to be fully documentation clean.

    Parameters
    ----------
    native_repository : Path
        Actual fixture with committed source and baseline bytes.
    """
    adopt_repository(native_repository)
    (native_repository / "src/legacy.py").write_text(
        "def renamed(value):\n    return value\n"
    )
    result = run_cli(native_repository, "check")
    assert result.returncode == 1, result.stderr
    assert "New or modified source has native findings: src/legacy.py" in result.stdout


def test_cli_initial_adoption_cannot_launder_changed_source(
    native_repository: Path,
) -> None:
    """Refuse first-baseline adoption of findings from bytes absent at the origin.

    Parameters
    ----------
    native_repository : Path
        Original committed fixture before any baseline exists.
    """
    (native_repository / "src/legacy.py").write_text(
        "def introduced(value):\n    return value + 1\n"
    )
    result = run_cli(
        native_repository, "record", "--output", str(native_repository / BASELINE)
    )
    assert result.returncode == 2
    assert "new or modified source bytes" in result.stderr
    assert not (native_repository / BASELINE).exists()


def test_cli_initial_adoption_cannot_restore_debt_from_an_older_origin(
    native_repository: Path,
) -> None:
    """Refuse restored historical debt when the actual base had already repaired it.

    Parameters
    ----------
    native_repository : Path
        Real original repository before its first quality baseline.
    """
    origin = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=native_repository,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    source = native_repository / "src/legacy.py"
    old_bytes = source.read_bytes()
    source.write_text('"""Document the actually repaired source."""\nVALUE = 1\n')
    commit_repository(native_repository)
    source.write_bytes(old_bytes)
    snapshot = native_repository / "snapshot.json"
    captured = run_cli(native_repository, "capture", "--output", str(snapshot))
    assert captured.returncode == 0, captured.stderr
    forged = {
        "schema": "quality-ratchet.baseline.v1",
        "origin_commit": origin,
        "profiles": {"python-docs": json.loads(snapshot.read_bytes())},
    }
    (native_repository / BASELINE).write_text(json.dumps(forged))
    checked = run_cli(native_repository, "check")
    assert checked.returncode == 2
    assert "exact original base commit" in checked.stderr


def test_cli_baseline_shrinks_and_removed_debt_cannot_return(
    native_repository: Path,
) -> None:
    """Accept clean changed/new files and permanently remove their old allowances.

    Parameters
    ----------
    native_repository : Path
        Actual fixture with committed original debt.
    """
    adopt_repository(native_repository)
    (native_repository / "src/legacy.py").write_text(
        '"""Publish a documented constant."""\nVALUE = 1\n'
    )
    (native_repository / "src/new.py").write_text(
        '"""Publish a new documented constant."""\nVALUE = 2\n'
    )
    output = native_repository / "candidate.json"
    result = run_cli(native_repository, "record", "--output", str(output))
    assert result.returncode == 0, result.stderr
    (native_repository / BASELINE).write_bytes(output.read_bytes())
    output.unlink()
    checked = run_cli(native_repository, "check")
    assert checked.returncode == 0, checked.stderr
    assert object_mapping(json.loads(checked.stdout))["findings"] == 0
    commit_repository(native_repository)
    (native_repository / "src/legacy.py").write_text(
        "def legacy(value):\n    return value\n"
    )
    result = run_cli(native_repository, "check")
    assert result.returncode == 1, result.stderr


@pytest.mark.parametrize("damage", ("syntax", "scope", "configuration", "duplicate"))
def test_cli_incomplete_or_changed_authority_refuses(
    native_repository: Path, damage: str
) -> None:
    """Refuse invalid native parsing, omitted scope, profile drift or duplicate JSON.

    Parameters
    ----------
    native_repository : Path
        Actual fixture with an independently committed baseline.
    damage : str
        Genuine input or source mutation applied before the public check.
    """
    adopt_repository(native_repository)
    if damage == "syntax":
        (native_repository / "src/legacy.py").write_text("def broken(:\n")
    elif damage == "scope":
        (native_repository / "outside.py").write_text(
            '"""Outside the declared scope."""\n'
        )
    elif damage == "configuration":
        with (native_repository / "pyproject.toml").open("a") as stream:
            stream.write("\n[tool.ruff.lint]\nignore = ['D']\n")
    else:
        path = native_repository / BASELINE
        path.write_text(
            path.read_text().replace('"schema":', '"schema":"forged","schema":', 1)
        )
    result = run_cli(native_repository, "check")
    assert result.returncode != 0
    assert "refus" in (result.stdout + result.stderr).lower()


def test_cli_candidate_cannot_replace_a_finding_at_same_count(
    native_repository: Path,
) -> None:
    """Refuse a forged allowance without relying on aggregate diagnostic counts.

    Parameters
    ----------
    native_repository : Path
        Actual fixture with unchanged original native diagnostics.
    """
    adopt_repository(native_repository)
    path = native_repository / BASELINE
    original = path.read_text()
    assert "Missing docstring" in original
    path.write_text(original.replace("Missing docstring", "Forged allowance", 1))
    result = run_cli(native_repository, "check")
    assert result.returncode == 1, result.stderr
    assert "gained a native finding" in result.stdout


def test_cli_capture_preserves_existing_output(native_repository: Path) -> None:
    """Preserve original artifacts and remove temporary publication files on refusal.

    Parameters
    ----------
    native_repository : Path
        Actual native fixture before baseline adoption.
    """
    output = native_repository / "original.json"
    output.write_bytes(b"original artifact\n")
    before = set(native_repository.iterdir())
    result = run_cli(native_repository, "capture", "--output", str(output))
    assert result.returncode == 2
    assert output.read_bytes() == b"original artifact\n"
    assert set(native_repository.iterdir()) == before


@pytest.mark.parametrize("deadline", ("nan", "inf", "0", "-1"))
def test_cli_invalid_deadline_refuses(native_repository: Path, deadline: str) -> None:
    """Refuse nonfinite or nonpositive deadlines before running native analysis.

    Parameters
    ----------
    native_repository : Path
        Actual selected native fixture.
    deadline : str
        Explicit invalid public CLI deadline.
    """
    result = run_cli(
        native_repository,
        "capture",
        "--timeout",
        deadline,
        "--output",
        str(native_repository / "snapshot.json"),
    )
    assert result.returncode == 2
    assert "positive and finite" in result.stderr
    assert not (native_repository / "snapshot.json").exists()


@pytest.mark.parametrize(
    "damage",
    (
        "first-profiles",
        "first-snapshot",
        "first-config",
        "original-profiles",
        "candidate-profiles",
    ),
)
def test_cli_original_and_candidate_native_authority_cannot_be_substituted(
    native_repository: Path, damage: str
) -> None:
    """Refuse omitted findings, changed original config and invented native profiles.

    Parameters
    ----------
    native_repository : Path
        Real Git/Ruff fixture with actual original documentation findings.
    damage : str
        Explicit invalid carrier or configuration mutation before public checking.
    """
    root = native_repository
    path = root / BASELINE
    output = root / "candidate.json"
    if damage == "first-config":
        with (root / "pyproject.toml").open("a") as stream:
            stream.write('description="Changed original configuration"\n')
        result = run_cli(root, "record", "--output", str(path))
        assert result.returncode == 2
        assert "configuration differs from its origin" in result.stderr
        assert not path.exists()
        return
    recorded = run_cli(root, "record", "--output", str(path))
    assert recorded.returncode == 0, recorded.stderr
    payload = object_mapping(json.loads(path.read_bytes()))
    profiles = object_mapping(payload["profiles"])
    snapshot = object_mapping(profiles["python-docs"])
    if damage in ("original-profiles", "candidate-profiles"):
        commit_repository(root)
    if damage == "first-snapshot":
        snapshot["findings"] = []
        profiles["python-docs"] = snapshot
    else:
        extra = dict(snapshot)
        extra["profile"] = "python-types"
        profiles["python-types"] = extra
    payload["profiles"] = profiles
    path.write_text(json.dumps(payload))
    if damage == "original-profiles":
        commit_repository(root)
    result = run_cli(root, "check")
    assert result.returncode == (1 if damage == "candidate-profiles" else 2)
    assert "profile" in (result.stdout + result.stderr) or "snapshot" in result.stderr
    replacement = run_cli(root, "record", "--output", str(output))
    if damage in ("first-snapshot", "first-profiles", "candidate-profiles"):
        assert replacement.returncode == 0, replacement.stderr
    else:
        assert replacement.returncode == 2
        assert not output.exists()


def test_cli_missing_output_and_public_artifact_failure_preserve_inputs(
    native_repository: Path,
) -> None:
    """Require explicit output and refuse absent output directories without artifacts.

    Parameters
    ----------
    native_repository : Path
        Actual native fixture and isolated output parent.
    """
    result = run_cli(native_repository, "capture")
    assert result.returncode == 2
    assert "require an explicit --output" in result.stderr
    output = native_repository / "absent-directory/artifact.json"
    with pytest.raises(FileNotFoundError):
        write_original_json(output, {"original": True})
    assert not output.parent.exists()
