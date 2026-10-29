# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — real native Go documentation ratchet contracts
"""Exercise actual Go and Revive processes through source-bound documentation capture."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.quality_ratchet_comparison import compare_snapshots
from tools.quality_ratchet_golang import capture_go_documentation, go_sources


@pytest.fixture(scope="module")
def go_tools() -> tuple[str, str]:
    """Resolve required real native tools without skipping missing prerequisites.

    Returns
    -------
    tuple[str, str]
        Actual Go and Revive executables, verified by each native capture.
    """
    go = shutil.which("go")
    revive = os.environ.get("DIRECTOR_QUALITY_REVIVE") or shutil.which("revive")
    assert go is not None and revive is not None, "Go and Revive are required"
    return go, revive


@pytest.fixture
def go_repository(tmp_path: Path) -> Path:
    """Create a real Git repository with a documented private Go declaration.

    Parameters
    ----------
    tmp_path : Path
        Actual native runner's isolated Samsung workspace.

    Returns
    -------
    Path
        Real maintained source, native module metadata and original Git commit.
    """
    root = tmp_path / "go-quality"
    root.mkdir()
    subprocess.run(
        ["git", "init"], cwd=root, check=True, capture_output=True, timeout=10
    )
    module = root / "gateway/go"
    module.mkdir(parents=True)
    (module / "go.mod").write_text("module native-fixture\n\ngo 1.27.1\n")
    (module / "go.sum").write_text("")
    (module / "value.go").write_text(
        "// Package fixture provides an actual value.\npackage fixture\n"
        "// value returns the original count.\nfunc value() int {return 1}\n"
    )
    subprocess.run(
        ["git", "add", "."], cwd=root, check=True, capture_output=True, timeout=10
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Native fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-m",
            "original native sources",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=10,
    )
    return root


def test_native_private_and_test_declarations_cannot_trade_legacy_findings(
    go_repository: Path,
    go_tools: tuple[str, str],
) -> None:
    """Refuse new whole-file findings while preserving an unchanged legacy source.

    Parameters
    ----------
    go_repository : Path
        Actual maintained Git source repository.
    go_tools : tuple[str, str]
        Original native compiler and convention checker.
    """
    root = go_repository
    original = capture_go_documentation(root, *go_tools)
    assert original.complete and original.findings == ()
    source = root / "gateway/go/value.go"
    source.write_text(
        "// Package fixture provides a value.\npackage fixture\nfunc value() int {return 1}\n"
    )
    dirty = capture_go_documentation(root, *go_tools)
    assert (
        dirty.sources[source.relative_to(root).as_posix()]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert len(dirty.findings) == 1
    assert dirty.findings[0].rule == "declaration-documentation"
    assert compare_snapshots(original, dirty)
    test = root / "gateway/go/value_test.go"
    test.write_text("package fixture\nfunc TestValue() {}\n")
    added = capture_go_documentation(root, *go_tools)
    assert any(finding.path.endswith("_test.go") for finding in added.findings)
    test.write_text(
        "package fixture\n// TestValue checks the count.\nfunc TestValue() {}\n"
    )
    corrected = capture_go_documentation(root, *go_tools)
    assert not any(finding.path.endswith("_test.go") for finding in corrected.findings)
    assert compare_snapshots(dirty, corrected) == ()


def test_native_source_directives_do_not_hide_original_physical_locations(
    go_repository: Path,
    go_tools: tuple[str, str],
) -> None:
    """Keep real physical diagnostics despite native lint and line masking directives.

    Parameters
    ----------
    go_repository : Path
        Actual maintained source repository.
    go_tools : tuple[str, str]
        Actual Go and Revive tools.
    """
    source = go_repository / "gateway/go/value.go"
    source.write_text(
        "// Package fixture holds values.\npackage fixture\n"
        "//revive:disable:exported\n//line invented.go:300\nfunc Hidden() {}\n"
    )
    actual = capture_go_documentation(go_repository, *go_tools)
    assert {(f.rule, f.line) for f in actual.findings} == {
        ("source-directive", 3),
        ("source-directive", 4),
        ("declaration-documentation", 5),
    }
    assert all(f.path == "gateway/go/value.go" for f in actual.findings)


def test_native_revive_conventions_survive_complete_declaration_capture(
    go_repository: Path,
    go_tools: tuple[str, str],
) -> None:
    """Retain actual original convention diagnostics alongside complete native parsing.

    Parameters
    ----------
    go_repository : Path
        Real Git-visible Go source.
    go_tools : tuple[str, str]
        Required native tools.
    """
    source = go_repository / "gateway/go/value.go"
    source.write_text(
        "// Package fixture holds values.\npackage fixture\n"
        "// WrongPrefix returns one.\nfunc Value() int {return 1}\n"
    )
    actual = capture_go_documentation(go_repository, *go_tools)
    assert len(actual.findings) == 1
    assert actual.findings[0].rule == "revive-exported"
    assert "Value" in actual.findings[0].message


def test_native_syntax_and_missing_tools_refuse_complete_analysis(
    go_repository: Path,
    go_tools: tuple[str, str],
) -> None:
    """Refuse actual failed native execution rather than silently reducing its scope.

    Parameters
    ----------
    go_repository : Path
        Actual source repository.
    go_tools : tuple[str, str]
        Actual compiler and Revive.
    """
    with pytest.raises(FileNotFoundError):
        capture_go_documentation(go_repository, "/absent/go", go_tools[1])
    with pytest.raises((subprocess.TimeoutExpired, ValueError)):
        capture_go_documentation(go_repository, *go_tools, timeout=0.0000001)
    (go_repository / "gateway/go/value.go").write_text(
        "package fixture\nfunc invalid( {\n"
    )
    with pytest.raises(ValueError, match="completely analyse"):
        capture_go_documentation(go_repository, *go_tools)


@pytest.mark.parametrize("timeout", (0.0, -1.0, float("nan"), float("inf")))
def test_invalid_whole_native_deadline_refuses(
    go_repository: Path,
    go_tools: tuple[str, str],
    timeout: float,
) -> None:
    """Require a positive finite whole-analysis deadline.

    Parameters
    ----------
    go_repository : Path
        Actual source root.
    go_tools : tuple[str, str]
        Required native executables.
    timeout : float
        Deliberately invalid whole deadline.
    """
    with pytest.raises(ValueError, match="positive and finite"):
        capture_go_documentation(go_repository, *go_tools, timeout=timeout)


def test_go_inventory_keeps_actual_generator_and_deleted_source_boundaries(
    go_repository: Path,
) -> None:
    """Exclude only exact generated outputs and refuse aliases or undeclared scopes.

    Parameters
    ----------
    go_repository : Path
        Actual selected Git source root.
    """
    root = go_repository
    generated = root / "gateway/go/proto/director/v1/director.pb.go"
    generated.parent.mkdir(parents=True)
    generated.write_text("generated original bytes\n")
    assert go_sources(root) == ("gateway/go/value.go",)
    alias = root / "gateway/go/alias.go"
    alias.symlink_to("value.go")
    with pytest.raises(ValueError, match="Symbolic"):
        go_sources(root)
    alias.unlink()
    outside = root / "outside.go"
    outside.write_text("package outside\n")
    with pytest.raises(ValueError, match="outside"):
        go_sources(root)
    outside.unlink()
    with pytest.raises(ValueError, match="actual Git root"):
        go_sources(root / "gateway")
    (root / "gateway/go/value.go").unlink()
    with pytest.raises(ValueError, match="inventory is empty"):
        go_sources(root)


def test_added_native_helper_metadata_preserves_original_profile_identity(
    go_repository: Path,
    go_tools: tuple[str, str],
) -> None:
    """Bind helper metadata once to its real engine without source-scope identity drift.

    Parameters
    ----------
    go_repository : Path
        Actual original source repository.
    go_tools : tuple[str, str]
        Original native executables.
    """
    original = capture_go_documentation(go_repository, *go_tools)
    helper = go_repository / "tools/quality_ratchet_go"
    helper.mkdir(parents=True)
    engine_module = (
        Path(__file__).resolve().parents[1] / "tools/quality_ratchet_go/go.mod"
    )
    (helper / "go.mod").write_bytes(engine_module.read_bytes())
    helper_source = helper / "extra.go"
    helper_source.write_text(
        "// Package extra documents a native helper.\npackage extra\n"
        "// value returns a count.\nfunc value() int {return 1}\n"
    )
    current = capture_go_documentation(go_repository, *go_tools)
    assert current.identity == original.identity
    assert "tools/quality_ratchet_go/extra.go" in current.sources
    assert compare_snapshots(original, current) == ()
    (go_repository / "gateway/go/go.sum").unlink()
    with pytest.raises(ValueError, match="not a regular file"):
        capture_go_documentation(go_repository, *go_tools)
