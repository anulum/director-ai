# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — original native Protobuf documentation and public CLI tests
"""Exercise actual protoc source analysis, immutable allowances and refusal boundaries."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

from tools import quality_ratchet_protobuf as native_protobuf
from tools.quality_ratchet_comparison import compare_snapshots
from tools.quality_ratchet_protobuf import (
    capture_protobuf_documentation,
    protobuf_sources,
)

DOCUMENTED_SCHEMA = """// Admission wire contract.
syntax = "proto3";
package admission;
// Actual admission request.
message Request {
  // Original caller text.
  string text = 1;
}
"""


@pytest.fixture
def protobuf_repository(tmp_path: Path) -> Path:
    """Create a real committed native schema source repository.

    Parameters
    ----------
    tmp_path : Path
        Actual isolated Samsung runner directory.

    Returns
    -------
    Path
        Committed original maintained source root.
    """
    root = tmp_path / "protobuf-source"
    (root / "proto").mkdir(parents=True)
    (root / "proto/admission.proto").write_text(DOCUMENTED_SCHEMA)
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Native fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-m",
            "original schema",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return root


def test_actual_repository_protobuf_inventory_and_debt_are_original() -> None:
    """Measure both real maintained schemas and bind findings to their source bytes."""
    root = Path(__file__).resolve().parents[1]
    paths = protobuf_sources(root)
    assert paths == ("proto/director.proto", "schemas/proto/director/v1/director.proto")
    actual = capture_protobuf_documentation(root, "protoc")
    assert actual.complete and len(actual.findings) > 100
    assert {finding.path for finding in actual.findings} == set(paths)
    assert actual.sources == {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths
    }


def test_native_protobuf_changed_whole_file_requires_documentation(
    protobuf_repository: Path,
) -> None:
    """Refuse surviving debt in a changed file despite an unchanged finding count.

    Parameters
    ----------
    protobuf_repository : Path
        Original native schema repository.
    """
    root = protobuf_repository
    path = root / "proto/admission.proto"
    assert not capture_protobuf_documentation(root, "protoc").findings
    undocumented = DOCUMENTED_SCHEMA.replace("// Original caller text.\n", "")
    path.write_text(undocumented)
    original = capture_protobuf_documentation(root, "protoc")
    assert len(original.findings) == 1
    path.write_text(undocumented.replace("string text", "string prompt"))
    changed = capture_protobuf_documentation(root, "protoc")
    assert compare_snapshots(original, changed)
    finding = changed.findings[0]
    assert (
        finding.column
        == path.read_text().splitlines()[finding.line - 1].index("string") + 1
    )
    path.write_text(DOCUMENTED_SCHEMA.replace("string text", "string prompt"))
    assert not compare_snapshots(
        original, capture_protobuf_documentation(root, "protoc")
    )


def test_native_protobuf_inventory_refusals_and_native_syntax_failure(
    protobuf_repository: Path,
) -> None:
    """Refuse aliases, unknown scopes, empty inventories and actual compiler failures.

    Parameters
    ----------
    protobuf_repository : Path
        Actual Git source repository.
    """
    root = protobuf_repository
    alias = root / "proto/alias.proto"
    alias.symlink_to("admission.proto")
    with pytest.raises(ValueError, match="Symbolic"):
        protobuf_sources(root)
    alias.unlink()
    unknown = root / "unknown.proto"
    unknown.write_text(DOCUMENTED_SCHEMA)
    with pytest.raises(ValueError, match="outside"):
        protobuf_sources(root)
    unknown.unlink()
    with pytest.raises(ValueError, match="actual Git root"):
        protobuf_sources(root / "proto")
    with pytest.raises(FileNotFoundError):
        capture_protobuf_documentation(root, "/absent/protoc")
    path = root / "proto/admission.proto"
    path.write_text('syntax = "proto3"; message Invalid {\n')
    with pytest.raises(ValueError, match="completely analyse"):
        capture_protobuf_documentation(root, "protoc")
    path.unlink()
    with pytest.raises(ValueError, match="inventory is empty"):
        protobuf_sources(root)


@pytest.mark.parametrize("timeout", (0.0, -1.0, float("nan"), float("inf"), 1e-12))
def test_native_protobuf_deadline_cannot_qualify_late_work(
    protobuf_repository: Path,
    timeout: float,
) -> None:
    """Refuse invalid or already exhausted lifetimes before claiming completion.

    Parameters
    ----------
    protobuf_repository : Path
        Actual source root.
    timeout : float
        Invalid or deliberately exhausted native lifetime.
    """
    with pytest.raises((ValueError, subprocess.TimeoutExpired)):
        capture_protobuf_documentation(protobuf_repository, "protoc", timeout=timeout)


@pytest.mark.parametrize("fault", ("missing", "wrong-version"))
def test_native_protobuf_descriptor_dependency_is_required(
    protobuf_repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    """Retain actual installed dependency metadata before refusal-only corruption.

    Parameters
    ----------
    protobuf_repository : Path
        Actual maintained source repository.
    tmp_path : Path
        Original dependency metadata custody.
    monkeypatch : pytest.MonkeyPatch
        Negative-only metadata boundary replacement.
    fault : str
        Explicit missing-package or wrong-version refusal.
    """
    original = version("protobuf")
    (tmp_path / "original-reader-version.txt").write_text(original)
    assert original == "6.33.6"

    def metadata_after_original(distribution: str) -> str:
        """Read real metadata before producing the deliberate negative result."""
        assert version(distribution) == original
        if fault == "missing":
            raise PackageNotFoundError(distribution)
        return "0.0.0"

    monkeypatch.setattr(native_protobuf, "version", metadata_after_original)
    with pytest.raises(ValueError, match="reader"):
        capture_protobuf_documentation(protobuf_repository, "protoc")


@pytest.mark.parametrize(
    "fault",
    (
        "version",
        "version-status",
        "version-stderr",
        "status",
        "stderr",
        "bytes",
        "empty",
        "duplicate",
        "inventory",
        "no-location",
        "invalid-span",
        "missing-declaration",
        "ambiguous-declaration",
        "source-change",
    ),
)
def test_native_protobuf_carrier_refusals_keep_original_compiler_output(
    protobuf_repository: Path,
    tmp_path: Path,
    fault: str,
) -> None:
    """Retain original real compiler output before refusal-only transport corruption.

    Parameters
    ----------
    protobuf_repository : Path
        Actual compiler source repository.
    tmp_path : Path
        Original process and wrapper custody.
    fault : str
        Explicit negative fault, never a successful substituted native analysis.
    """
    protoc = shutil.which("protoc")
    assert protoc is not None
    wrapper = tmp_path / "protoc-original-before-fault"
    wrapper.write_text(
        "#!" + sys.executable + "\n"
        "import sys,subprocess,json,base64\nfrom pathlib import Path\n"
        "from google.protobuf import descriptor_pb2\n"
        "args=sys.argv[1:]; fault=" + repr(fault) + "\n"
        "result=subprocess.run(["
        + repr(protoc)
        + ",*args],capture_output=True,timeout=20)\n"
        "with Path('native-original.jsonl').open('a') as stream:\n"
        " stream.write(json.dumps({'argv':args,'status':result.returncode,"
        "'stdout_b64':base64.b64encode(result.stdout).decode(),"
        "'stderr_b64':base64.b64encode(result.stderr).decode()})+'\\n')\n"
        "output=result.stdout; error=result.stderr; status=result.returncode\n"
        "if args==['--version']:\n"
        " if fault=='version': output=b'libprotoc 0.0.0\\n'\n"
        " if fault=='version-status': status=2\n"
        " if fault=='version-stderr': error=b'negative version warning\\n'\n"
        "else:\n"
        " if fault=='status': status=2\n"
        " if fault=='stderr': error=b'negative analysis warning\\n'\n"
        " if fault=='bytes': output=b'\\xff'\n"
        " if fault in ('empty','duplicate','inventory','no-location','invalid-span',"
        "'missing-declaration','ambiguous-declaration'):\n"
        "  descriptor=descriptor_pb2.FileDescriptorSet.FromString(output)\n"
        "  if fault=='empty': descriptor.ClearField('file')\n"
        "  if fault=='duplicate': descriptor.file.add().CopyFrom(descriptor.file[0])\n"
        "  if fault=='inventory': descriptor.file[0].name='proto/foreign.proto'\n"
        "  locations=descriptor.file[0].source_code_info.location if descriptor.file else []\n"
        "  if fault=='no-location': descriptor.file[0].ClearField('source_code_info')\n"
        "  if fault=='invalid-span': locations[0].ClearField('span')\n"
        "  if fault in ('missing-declaration','ambiguous-declaration'):\n"
        "   index=next(i for i,l in enumerate(locations) if list(l.path)==[4,0])\n"
        "   if fault=='missing-declaration': del locations[index]\n"
        "   else: locations.add().CopyFrom(locations[index])\n"
        "  output=descriptor.SerializeToString()\n"
        " if fault=='source-change':\n"
        "  p=Path('proto/admission.proto'); p.write_bytes(p.read_bytes()+b'// actual change\\n')\n"
        "sys.stdout.buffer.write(output); sys.stderr.buffer.write(error); raise SystemExit(status)\n"
    )
    wrapper.chmod(0o700)
    with pytest.raises(ValueError):
        capture_protobuf_documentation(protobuf_repository, str(wrapper))
    originals = [
        json.loads(line)
        for line in (protobuf_repository / "native-original.jsonl")
        .read_text()
        .splitlines()
    ]
    assert originals and all(item["status"] == 0 for item in originals)
    assert all(not base64.b64decode(item["stderr_b64"]) for item in originals)
    assert base64.b64decode(originals[0]["stdout_b64"]).startswith(b"libprotoc 3.21.12")
    if len(originals) == 2:
        assert base64.b64decode(originals[1]["stdout_b64"])
