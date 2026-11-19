# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — actual Lean declaration documentation admission tests
"""Exercise original compiler output and refuse corruption of native source identity."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from formal.HaltMonitor.tests.test_quality_ratchet_lean import (
    lean_binary as lean_binary,
)
from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_contracts import decode_quality_json
from tools.quality_ratchet_lean_declarations import lean_documentation_findings

SOURCE = """/-! Original documented native declaration controls. -/
namespace Admission
/-- Original value. -/
def documented : Nat := 1
/-- Original private value. -/
private def privateValue : Nat := 2
/-- Original proof. -/
theorem proof : True := True.intro
/-- Original type alias. -/
abbrev Alias := Nat
/-- Original record. -/
structure Record where
  /-- Original unicode π field. -/
  π : Nat
/-- Original choice. -/
inductive Choice where
  /-- Original constructor. -/
  | first
  deriving DecidableEq, Repr
/-- Original anonymous instance. -/
instance : Inhabited Record := ⟨⟨1⟩⟩
/-- Original named instance. -/
instance named : Inhabited Choice := ⟨.first⟩
/-- Original opaque value. -/
opaque hidden : Nat := 2
/-- Original assumption. -/
axiom assumed : Nat
@[inherit_doc documented]
def inherited : Nat := 3
/-- Original ephemeral proof. -/
example : True := True.intro
mutual
  /-- First mutual declaration. -/
  def firstMutual : Nat := 1
  /-- Second mutual declaration. -/
  def secondMutual : Nat := 2
end
end Admission
"""

NativeReader = Callable[[str], tuple[bytes, bytes, int]]


@pytest.fixture(scope="module")
def native_reader(tmp_path_factory: pytest.TempPathFactory) -> NativeReader:
    """Inspect real original source using the required pinned native Lean tool.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Actual isolated native runner workspace on the configured work disk.

    Returns
    -------
    NativeReader
        Original native stdout, unchanged source bytes and real process status.
    """
    binary = os.environ.get("DIRECTOR_QUALITY_LEAN") or shutil.which("lean")
    assert binary is not None, "Native Lean is required"
    executable = Path(binary).resolve(strict=True)
    helper = (
        Path(__file__).resolve().parents[3]
        / "tools/quality_ratchet_lean/QualityRatchet.lean"
    )
    root = tmp_path_factory.mktemp("lean-source-controls")
    sequence = 0

    def read(source: str) -> tuple[bytes, bytes, int]:
        """Preserve each actual process output before documentation admission.

        Parameters
        ----------
        source : str
            Complete unchanged original native source text.

        Returns
        -------
        tuple[bytes, bytes, int]
            Original compiler output, source bytes and native return code.
        """
        nonlocal sequence
        sequence += 1
        directory = root / str(sequence)
        directory.mkdir()
        path = directory / "Admission.lean"
        path.write_bytes(source.encode("utf-8"))
        process = subprocess.run(
            [
                str(executable),
                "--run",
                str(helper),
                "inspect",
                str(path),
                "Admission",
                str(executable.parent.parent),
            ],
            cwd=directory,
            capture_output=True,
            timeout=120,
        )
        (directory / "original.stdout").write_bytes(process.stdout)
        (directory / "original.stderr").write_bytes(process.stderr)
        return process.stdout, path.read_bytes(), process.returncode

    return read


@pytest.fixture(scope="module")
def documented_native(native_reader: NativeReader) -> tuple[bytes, bytes]:
    """Retain one complete original native documented source control.

    Parameters
    ----------
    native_reader : NativeReader
        Actual source-preserving compiler inspection recipe.

    Returns
    -------
    tuple[bytes, bytes]
        Native original stdout and exact original source bytes.
    """
    original, source, code = native_reader(SOURCE)
    assert code == 0
    return original, source


def test_documented_native_owners_and_generated_auxiliaries(
    documented_native: tuple[bytes, bytes],
) -> None:
    """Qualify private, unicode field, constructor, inherited and mutual docs.

    Parameters
    ----------
    documented_native : tuple[bytes, bytes]
        Complete actual native original output and source.
    """
    assert (
        lean_documentation_findings(
            *documented_native, "formal/Admission.lean", "Admission"
        )
        == []
    )


def test_undocumented_native_sources_and_linter_mask(
    native_reader: NativeReader,
) -> None:
    """Measure true missing docs despite a real compiler linter being disabled.

    Parameters
    ----------
    native_reader : NativeReader
        Actual native compiler recipe.
    """
    source = SOURCE
    for comment in (
        "/-! Original documented native declaration controls. -/",
        "/-- Original private value. -/",
        "/-- Original unicode π field. -/",
        "/-- Original constructor. -/",
        "/-- Original anonymous instance. -/",
        "/-- Original ephemeral proof. -/",
    ):
        source = source.replace(comment, "")
    source += "set_option linter.missingDocs false in\ndef masked : Nat := 1\n"
    original, encoded, code = native_reader(source)
    assert code == 0
    findings = lean_documentation_findings(
        original, encoded, "formal/Admission.lean", "Admission"
    )
    assert len(findings) == 8
    assert sum(item["rule"] == "source-directive" for item in findings) == 1
    assert sum(item["rule"] == "module-documentation" for item in findings) == 1
    messages = [str(item["message"]) for item in findings]
    assert any("privateValue" in message for message in messages)
    assert any("Record.π" in message for message in messages)
    assert any("Choice.first" in message for message in messages)
    assert any("instInhabitedRecord" in message for message in messages)
    assert any(message.endswith("example") for message in messages)
    assert any(message.endswith("masked") for message in messages)


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "module",
        "compiler",
        "compiler_commit",
        "mode",
        "schema",
        "has_errors",
        "extra",
    ],
)
def test_native_identity_cannot_be_projected(
    documented_native: tuple[bytes, bytes],
    change: str,
) -> None:
    """Refuse altered toolchain, source, completeness and unknown carrier fields.

    Parameters
    ----------
    documented_native : tuple[bytes, bytes]
        Actual native original output and source.
    change : str
        Specific identity or carrier member deliberately corrupted.
    """
    original, source = documented_native
    row = object_mapping(decode_quality_json(original))
    row[change] = True if change == "has_errors" else "corrupted"
    with pytest.raises(ValueError, match="identity or complete source"):
        lean_documentation_findings(
            json.dumps(row).encode(), source, "formal/Admission.lean", "Admission"
        )


def test_native_error_is_not_a_documentation_snapshot(
    native_reader: NativeReader,
) -> None:
    """Refuse actual elaboration failure rather than reporting a partial clean scope.

    Parameters
    ----------
    native_reader : NativeReader
        Actual native compiler recipe retaining original error output.
    """
    original, source, code = native_reader("def invalid : Nat := doesNotExist\n")
    assert code == 2
    with pytest.raises(ValueError, match="identity or complete source"):
        lean_documentation_findings(
            original, source, "formal/Admission.lean", "Admission"
        )


@pytest.mark.parametrize(
    "source",
    [
        "/-! Grouped fields. -/\n/-- Grouped record. -/\nstructure Grouped where\n  /-- Two fields. -/\n  (first second : Nat)\n",
        "/-! Local declarations. -/\n/-- Original value. -/\ndef value : Nat := localValue where\n  localValue : Nat := 1\n",
        '/-! New syntax. -/\nsyntax "custom_value" : term\n',
    ],
)
def test_unqualified_native_source_forms_refuse(
    native_reader: NativeReader, source: str
) -> None:
    """Refuse real native forms until their complete ownership route is qualified.

    Parameters
    ----------
    native_reader : NativeReader
        Actual native compiler inspection recipe.
    source : str
        Original valid source whose ownership route is not yet supported.
    """
    original, encoded, code = native_reader(source)
    assert code == 0
    with pytest.raises(ValueError, match="qualification|Unsupported"):
        lean_documentation_findings(
            original, encoded, "formal/Admission.lean", "Admission"
        )


@pytest.mark.parametrize(
    "attack",
    [
        "syntax-fields",
        "syntax-kind",
        "syntax-atom",
        "syntax-children",
        "position-fields",
        "position-integer",
        "position-overflow",
        "position-coordinate",
        "position-reversed",
        "position-utf8",
        "header-kind",
        "end-source",
        "imports-array",
        "import-empty",
        "message-error",
        "message-fields",
        "message-coordinate",
        "message-text",
        "constant-fields",
        "constant-doc",
        "constant-duplicate",
        "constant-range-fields",
        "constant-range-overflow",
        "constant-range-order",
        "authored-constant",
        "authored-command",
    ],
)
def test_original_native_transport_corruption_refuses(
    documented_native: tuple[bytes, bytes],
    attack: str,
) -> None:
    """Refuse lost declarations, changed coordinates and corrupted compiler data.

    Parameters
    ----------
    documented_native : tuple[bytes, bytes]
        Preserved actual native original output and source.
    attack : str
        Concrete deliberate corruption of that original transport.
    """
    original, source = documented_native
    row = object_mapping(decode_quality_json(original))
    header = object_mapping(row["header"])
    row["header"] = header
    if attack.startswith("syntax-"):
        changes: dict[str, tuple[str, object]] = {
            "syntax-fields": ("extra", 1),
            "syntax-kind": ("kind", None),
            "syntax-atom": ("atom", 1),
            "syntax-children": ("children", {}),
        }
        key, value = changes[attack]
        header[key] = value
    elif attack.startswith("position-"):
        position: dict[str, object] = {"line": 1, "column": 0, "byte_offset": 0}
        if attack == "position-fields":
            position.pop("line")
        elif attack == "position-integer":
            position["line"] = True
        elif attack == "position-overflow":
            position["byte_offset"] = len(source) + 1
        elif attack == "position-coordinate":
            position["column"] = 1
        elif attack == "position-reversed":
            position = {
                "line": source.count(b"\n") + 1,
                "column": 0,
                "byte_offset": len(source),
            }
            header["end"] = {"line": 1, "column": 0, "byte_offset": 0}
        else:
            position["byte_offset"] = source.index("π".encode()) + 1
        header["start"] = position
    elif attack == "header-kind":
        header["kind"] = "null"
    elif attack == "end-source":
        row["commands"] = []
    elif attack in ("imports-array", "import-empty"):
        row["imports"] = None if attack == "imports-array" else [""]
    elif attack.startswith("message-"):
        message: dict[str, object] = {
            "severity": "warning",
            "line": 1,
            "column": 0,
            "message": "Original warning",
        }
        if attack == "message-error":
            message["severity"] = "error"
        elif attack == "message-fields":
            message["extra"] = 1
        elif attack == "message-coordinate":
            message["line"] = 0
        else:
            message["message"] = "\x00"
        row["messages"] = [message]
    elif attack == "authored-command":
        commands = row["commands"]
        assert isinstance(commands, list)
        commands.pop(2)
    else:
        constants = row["constants"]
        assert isinstance(constants, list)
        values = [object_mapping(value) for value in constants]
        row["constants"] = values
        index = next(
            i
            for i, value in enumerate(values)
            if value["name"] == "Admission.documented"
        )
        constant = values[index]
        if attack == "constant-fields":
            constant.pop("name")
        elif attack == "constant-doc":
            constant["doc"] = True
        elif attack == "constant-duplicate":
            values.append(constant)
        elif attack == "authored-constant":
            values.pop(index)
        else:
            bounds = object_mapping(constant["range"])
            constant["range"] = bounds
            if attack == "constant-range-fields":
                bounds.pop("line")
            elif attack == "constant-range-overflow":
                bounds["end_line"] = source.count(b"\n") + 10
            else:
                bounds["selection_line"] = 1
    with pytest.raises(ValueError):
        lean_documentation_findings(
            json.dumps(row).encode(), source, "formal/Admission.lean", "Admission"
        )


@pytest.mark.parametrize(
    ("arguments", "diagnostic"),
    [
        ([], "Expected mode, source path, module identity and toolchain root"),
        (
            ["unknown", "Reading.lean", "Reading", "."],
            "Expected header or inspect and nonempty module and toolchain identities",
        ),
        (
            ["header", "Reading.lean", "", "."],
            "Expected header or inspect and nonempty module and toolchain identities",
        ),
        (
            ["inspect", "Reading.lean", "Reading", ""],
            "Expected header or inspect and nonempty module and toolchain identities",
        ),
    ],
)
def test_native_helper_rejects_unqualified_invocation(
    lean_binary: str, tmp_path: Path, arguments: list[str], diagnostic: str
) -> None:
    """Refuse an invalid native invocation before opening its source path."""
    helper = Path(__file__).resolve().parents[3] / (
        "tools/quality_ratchet_lean/QualityRatchet.lean"
    )
    result = subprocess.run(
        [lean_binary, "--run", str(helper), *arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == diagnostic


@pytest.mark.parametrize(
    ("mode", "source", "expected_exit"),
    [
        ("header", "import Lean\n/-! Original library. -/\ndef marker : Nat := 7\n", 0),
        ("header", "import\n", 2),
        ("inspect", "import\n", 2),
    ],
)
def test_native_helper_preserves_unelaborated_headers(
    lean_binary: str, tmp_path: Path, mode: str, source: str, expected_exit: int
) -> None:
    """Preserve the original header and parse errors before full elaboration."""
    executable = Path(lean_binary).resolve(strict=True)
    helper = Path(__file__).resolve().parents[3] / (
        "tools/quality_ratchet_lean/QualityRatchet.lean"
    )
    original = tmp_path / "Reading.lean"
    original.write_text(source)
    result = subprocess.run(
        [
            str(executable),
            "--run",
            str(helper),
            mode,
            str(original),
            "Reading",
            str(executable.parent.parent),
        ],
        cwd=tmp_path,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == expected_exit
    assert not result.stderr
    report = object_mapping(decode_quality_json(result.stdout))
    assert report["source"] == source
    assert report["mode"] == mode and report["module"] == "Reading"
    assert report["commands"] == report["constants"] == []
    assert report["has_errors"] is (expected_exit != 0)
    if expected_exit == 0:
        assert (
            report["imports"] == ["Init", "Init", "Lean"] and report["messages"] == []
        )
    else:
        messages = report["messages"]
        assert report["imports"] == []
        assert isinstance(messages, list)
        assert any(object_mapping(row)["severity"] == "error" for row in messages)
