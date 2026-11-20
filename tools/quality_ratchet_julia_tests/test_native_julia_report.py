# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — real native Julia quality transport and CLI tests
"""Verify the native Julia bridge with the actual tuner and compiler report."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.test_quality_ratchet_typing_cli import commit_typing_sources
from tools.quality_ratchet_contracts import Finding, parse_snapshot
from tools.quality_ratchet_julia_report import parse_julia_report

ROOT = Path(__file__).resolve().parents[2]
TUNER = Path("tools/julia_tuner")
ANALYZER = Path("tools/quality_ratchet_julia")
HISTORICAL = "391228da74540de1f5edc948df83765c1619e42a"


@pytest.fixture(scope="module")
def julia_binary() -> str:
    """Resolve the actual native Julia binary for the scoped test execution."""
    binary = shutil.which(os.environ.get("DIRECTOR_QUALITY_JULIA", "julia"))
    assert binary is not None, "Install the qualified native Julia 1.13.4 binary"
    return str(Path(binary).resolve())


def _sha(path: Path) -> str:
    """Hash one actual file referenced by the Julia reader."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class NativeJuliaReport:
    """Retain one actual report and the independently measured original inputs."""

    output: bytes
    binary: str
    project: str
    manifest: str
    sources: dict[str, str]
    lines: dict[str, int]

    def decode(self, output: bytes) -> tuple[Finding, ...]:
        """Read the live or altered report through the production decoder.

        Parameters
        ----------
        output : bytes
            Native report stream supplied to the production decoder.

        Returns
        -------
        tuple of Finding
            Complete source-linked Julia documentation findings.
        """
        return parse_julia_report(
            output,
            versions=("1.13.4", "1.2.0", "0.39.10", "0.36.15"),
            binary=self.binary,
            project=self.project,
            manifest=self.manifest,
            sources=self.sources,
            line_counts=self.lines,
        )


@pytest.fixture(scope="module")
def native_report(julia_binary: str) -> NativeJuliaReport:
    """Run the native scanner and retain its original framing and source hashes."""
    result = subprocess.run(
        [
            julia_binary,
            "--startup-file=no",
            "--history-file=no",
            "--threads=1",
            "--compiled-modules=existing",
            "--pkgimages=existing",
            f"--project={ANALYZER}",
            "tools/quality_ratchet_julia/inspect.jl",
            str(ROOT),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    listing = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "tools/julia_tuner/*.jl"], cwd=ROOT
    ).decode()
    relative = tuple(filter(None, listing.split("\0")))
    return NativeJuliaReport(
        result.stdout,
        _sha(Path(julia_binary)),
        _sha(ROOT / ANALYZER / "Project.toml"),
        _sha(ROOT / ANALYZER / "Manifest.toml"),
        {path: _sha(ROOT / path) for path in relative},
        {path: len((ROOT / path).read_bytes().splitlines()) for path in relative},
    )


def _cli(root: Path, binary: str, output: Path) -> subprocess.CompletedProcess[str]:
    """Invoke the real singleton entry point with its selected Julia toolchain."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.check_quality_ratchet",
            "--profile",
            "julia-docs",
            "--julia",
            binary,
            "--root",
            str(root),
            "--timeout",
            "300",
            "capture",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=315,
    )


def test_native_report_carries_included_posterior_declarations(
    native_report: NativeJuliaReport,
) -> None:
    """Bind an exported method to its declaring file and the including module."""
    findings = native_report.decode(native_report.output)
    listing = str(TUNER / "src/threshold_posterior.jl")
    assert listing in native_report.sources
    rows = [
        row.split(b"\t")
        for row in native_report.output.splitlines()
        if row.startswith(b"declaration\t")
    ]
    qualified = "DirectorThresholdTuner.bayesian_threshold_posterior"
    assert any(
        (
            bytes.fromhex(row[1].decode()).decode(),
            row[3],
            bytes.fromhex(row[4].decode()).decode(),
            row[5],
        )
        == (listing, b"method", qualified, b"true")
        for row in rows
    )
    assert any(row[3] == b"argument" for row in rows)
    assert not findings


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        ("no-footer", "incomplete native Julia complete"),
        ("lost-row", "incomplete native Julia declaration"),
        ("duplicate-row", "duplicate native Julia declaration"),
        ("wrong-binary", "binary differs"),
        ("wrong-manifest", "manifest differs"),
        ("wrong-project", "project differs"),
        ("wrong-source", "source inventory or digest differs"),
        ("wrong-version", "runtime or generator versions differ"),
        ("out-of-bounds", "outside the qualified scope"),
        ("trailing", "did not finish exactly"),
        ("unsafe-name", "Julia name"),
        ("bad-hex", "Julia hex"),
        ("negative-count", "Julia counter"),
        ("bad-framing", "report framing"),
    ],
)
def test_native_julia_report_refuses_altered_real_output(
    native_report: NativeJuliaReport, operation: str, message: str
) -> None:
    """Refuse missing and unqualified records from an actual captured report."""
    rows = native_report.output.splitlines(keepends=True)
    source = next(
        index for index, row in enumerate(rows) if row.startswith(b"source\t")
    )
    declaration = next(
        index for index, row in enumerate(rows) if row.startswith(b"declaration\t")
    )
    match operation:
        case "no-footer":
            rows.pop()
        case "lost-row":
            rows.pop(declaration)
        case "duplicate-row":
            rows[declaration + 1] = rows[declaration]
        case "wrong-binary":
            rows[1] = b"binary\t" + b"0" * 64 + b"\n"
        case "wrong-project":
            rows[2] = b"project\t" + b"0" * 64 + b"\n"
        case "wrong-manifest":
            rows[3] = b"manifest\t" + b"0" * 64 + b"\n"
        case "wrong-source":
            row = rows[source].split(b"\t")
            row[2] = b"0" * 64
            rows[source] = b"\t".join(row)
        case "wrong-version":
            rows[0] = rows[0].replace(b"1.13.4", b"1.13.3", 1)
        case "trailing":
            rows.append(b"complete\t5\n")
        case "bad-framing":
            rows[-1] = rows[-1].removesuffix(b"\n")
        case _:
            index = source if operation == "negative-count" else declaration
            fields = rows[index].removesuffix(b"\n").split(b"\t")
            field, value = {
                "out-of-bounds": (2, b"999999999"),
                "unsafe-name": (4, b"09"),
                "bad-hex": (4, b"zz"),
                "negative-count": (3, b"-1"),
            }[operation]
            fields[field] = value
            rows[index] = b"\t".join(fields) + b"\n"
    with pytest.raises(ValueError, match=message):
        native_report.decode(b"".join(rows))


def test_native_julia_cli_exports_same_source_snapshot(
    native_report: NativeJuliaReport, julia_binary: str, tmp_path: Path
) -> None:
    """Reconcile the singleton CLI with the actual direct native report."""
    output = tmp_path / "julia-quality.json"
    result = _cli(ROOT, julia_binary, output)
    assert result.returncode == 0, result.stderr
    snapshot = parse_snapshot(json.loads(output.read_bytes()))
    assert snapshot.complete and snapshot.profile == "julia-docs"
    assert snapshot.sources == native_report.sources
    assert set(snapshot.findings) == set(native_report.decode(native_report.output))


@pytest.fixture
def tuner_repository(tmp_path: Path) -> Path:
    """Create a Git source view with the actual maintained Julia tuner files."""
    root = tmp_path / "julia-source"
    root.mkdir()
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    shutil.copytree(ROOT / TUNER, root / TUNER)
    return root


def test_native_julia_cli_refuses_historical_syntax(
    julia_binary: str, tuner_repository: Path, tmp_path: Path
) -> None:
    """Read the original invalid launcher and refuse before publishing a snapshot."""
    root = tuner_repository
    launcher = TUNER / "bin/tune_threshold.jl"
    (root / launcher).write_bytes(
        subprocess.check_output(["git", "show", f"{HISTORICAL}:{launcher}"], cwd=ROOT)
    )
    commit_typing_sources(root)
    output = tmp_path / "refused.json"
    result = _cli(root, julia_binary, output)
    assert result.returncode == 2 and "Native Julia parse failure" in result.stderr
    assert not output.exists()


def test_native_julia_cli_refuses_missing_executable(tmp_path: Path) -> None:
    """Refuse a missing compiler without producing a snapshot."""
    output = tmp_path / "refused.json"
    result = _cli(ROOT, str(tmp_path / "absent-julia"), output)
    assert result.returncode == 2 and "executable is unavailable" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("path", "text", "reason"),
    [
        (
            Path("outside.jl"),
            "unclassified_source() = 0\n",
            "outside the qualified package scope",
        ),
        (
            TUNER / "Manifest.toml",
            "[deps]\n",
            "Expected one pinned Julia generator ArgParse",
        ),
        (
            TUNER / "Manifest.toml",
            "[[deps.ArgParse]]\nuuid = 'c7e460c6-2fb9-53a9-8c5b-16f535851c63'\n",
            "Unversioned native Julia generator ArgParse",
        ),
    ],
)
def test_native_julia_cli_refuses_unqualified_source_and_lock(
    julia_binary: str,
    tuner_repository: Path,
    tmp_path: Path,
    path: Path,
    text: str,
    reason: str,
) -> None:
    """Refuse source or dependency scopes that differ from the native contract."""
    (tuner_repository / path).write_text(text)
    output = tmp_path / "refused.json"
    result = _cli(tuner_repository, julia_binary, output)
    assert result.returncode == 2 and reason in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("leaf", "text", "diagnostic"),
    [
        (
            "DirectorThresholdTuner.jl",
            'module BadModule\ninclude("absent.jl")\nend\n',
            "include target is absent",
        ),
        (
            "threshold_posterior.jl",
            'include("DirectorThresholdTuner.jl")\n',
            "Cyclic Julia include",
        ),
        (
            "second_owner.jl",
            'module SecondOwner\ninclude("threshold_posterior.jl")\nend\n',
            "Ambiguous Julia include owner",
        ),
        (
            "rogue.jl",
            "module Rogue\nexport local_prediction\nend\n",
            "public bindings without qualified declarations",
        ),
        (
            "DirectorThresholdTuner.jl",
            "module DirectorThresholdTuner\nexport bayesian_threshold_posterior\n"
            'if false\ninclude("threshold_posterior.jl")\nend\nend\n',
            "public bindings without qualified declarations",
        ),
        (
            "undeclared_generator.jl",
            "using Turing\n@model function observation(x)\n x ~ Normal()\nend\n",
            "Turing is not a direct dependency of the source project",
        ),
    ],
)
def test_native_julia_cli_refuses_unqualified_include_graph(
    julia_binary: str,
    tuner_repository: Path,
    tmp_path: Path,
    leaf: str,
    text: str,
    diagnostic: str,
) -> None:
    """Reject unsupported source ownership and generator graphs without a report."""
    (tuner_repository / TUNER / "src" / leaf).write_text(text)
    output = tmp_path / "refused.json"
    result = _cli(tuner_repository, julia_binary, output)
    assert result.returncode == 2 and diagnostic in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("relative", "original", "replacement", "diagnostic"),
    [
        (
            TUNER / "Manifest.toml",
            'git-tree-sha1 = "22cf435ac22956a7b45b0168abbc871176e7eecc"\n',
            "",
            "Unqualified native Julia generator source ArgParse",
        ),
        (
            TUNER / "Manifest.toml",
            "22cf435ac22956a7b45b0168abbc871176e7eecc",
            "13094058d44913122e5ab6b81c32d4da07d1f5ba",
            "ArgParse differs from the source lock",
        ),
        (
            TUNER / "Project.toml",
            "c7e460c6-2fb9-53a9-8c5b-16f535851c63",
            "c7e460c6-2fb9-53a9-8c5b-16f535851c64",
            "ArgParse differs from the source project",
        ),
    ],
)
def test_native_julia_cli_requires_matching_generator_identity(
    julia_binary: str,
    tuner_repository: Path,
    tmp_path: Path,
    relative: Path,
    original: str,
    replacement: str,
    diagnostic: str,
) -> None:
    """Reject real tuner manifest or project changes before publishing a report."""
    path = tuner_repository / relative
    text = path.read_text()
    assert text.count(original) == 1
    path.write_text(text.replace(original, replacement))
    output = tmp_path / "refused.json"
    result = _cli(tuner_repository, julia_binary, output)
    assert result.returncode == 2 and diagnostic in result.stderr
    assert not output.exists()
