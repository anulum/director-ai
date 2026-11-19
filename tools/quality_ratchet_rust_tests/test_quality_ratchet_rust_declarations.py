# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — original native Rust evidence admission controls
"""Reject source and diagnostic transport corruption after real native execution."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_contracts import decode_quality_json
from tools.quality_ratchet_rust_declarations import rust_documentation_findings
from tools.quality_ratchet_rust_tests.test_quality_ratchet_rust import (
    rust_reader as rust_reader,
)
from tools.quality_ratchet_rust_tests.test_quality_ratchet_rust import (
    rust_tools as rust_tools,
)


@pytest.fixture(scope="module")
def native_original(
    rust_reader: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[bytes, dict[str, str], Path]:
    """Retain an actual native declaration report and its unchanged source.

    Parameters
    ----------
    rust_reader : Path
        Actual compiled production process entry.
    tmp_path_factory : pytest.TempPathFactory
        Owned native runner allocation.

    Returns
    -------
    tuple[bytes, dict[str, str], Path]
        Complete original stdout, actual source-byte digests and source root.
    """
    root = tmp_path_factory.mktemp("rust-original-evidence")
    relative = "backfire-kernel/crates/backfire-core/src/lib.rs"
    source = root / relative
    source.parent.mkdir(parents=True)
    source.write_text("//! Original source description.\nfn café() -> u8 { 1 }\n")
    result = subprocess.run(
        [str(rust_reader), "--source", relative],
        cwd=root,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 1 and not result.stderr
    (root / "native-original.stdout").write_bytes(result.stdout)
    (root / "native-original.stderr").write_bytes(result.stderr)
    return (
        result.stdout,
        {relative: hashlib.sha256(source.read_bytes()).hexdigest()},
        root,
    )


def test_complete_original_native_evidence_is_source_bound(
    native_original: tuple[bytes, dict[str, str], Path],
) -> None:
    """Admit the actual retained source, unchanged individual finding and Unicode prose."""
    original, sources, root = native_original
    findings = rust_documentation_findings(original, sources, root)
    assert len(findings) == 1
    assert findings[0]["path"] == next(iter(sources))
    assert findings[0]["message"] == "Function café requires a Rustdoc comment"
    assert findings[0]["line"] == 2 and findings[0]["column"] == 4


@pytest.mark.parametrize(
    "attack",
    (
        "schema-fields",
        "schema-version",
        "files-type",
        "files-missing",
        "row-fields",
        "path-type",
        "path-unknown",
        "path-duplicate",
        "source-type",
        "source-bytes",
        "refusals",
        "findings-type",
        "finding-fields",
        "rule",
        "line-type",
        "line-bool",
        "line-low",
        "line-high",
        "column-type",
        "column-bool",
        "column-low",
        "column-high",
        "message-type",
        "message-empty",
        "message-null",
    ),
)
def test_corrupted_original_native_transport_never_qualifies(
    native_original: tuple[bytes, dict[str, str], Path], attack: str
) -> None:
    """Refuse deliberate alterations to actual native source and original diagnostics."""
    original, sources, root = native_original
    report = object_mapping(decode_quality_json(original))
    values = report["files"]
    assert isinstance(values, list)
    rows = [object_mapping(value) for value in values]
    report["files"] = rows
    row = rows[0]
    diagnostics = row["findings"]
    assert isinstance(diagnostics, list)
    findings = [object_mapping(value) for value in diagnostics]
    row["findings"] = findings
    finding = findings[0]
    if attack == "schema-fields":
        report["unqualified"] = True
    elif attack == "schema-version":
        report["schema"] = "unqualified"
    elif attack == "files-type":
        report["files"] = {}
    elif attack == "files-missing":
        report["files"] = []
    elif attack == "row-fields":
        row["unqualified"] = True
    elif attack == "path-type":
        row["path"] = None
    elif attack == "path-unknown":
        row["path"] = "unbound.rs"
    elif attack == "path-duplicate":
        rows.append(dict(row))
    elif attack == "source-type":
        row["source"] = None
    elif attack == "source-bytes":
        row["source"] = "lost original source"
    elif attack == "refusals":
        row["refusals"] = ["original syntax is incomplete"]
    elif attack == "findings-type":
        row["findings"] = {}
    elif attack == "finding-fields":
        finding["unqualified"] = True
    elif attack == "rule":
        finding["rule"] = "unqualified"
    elif attack.startswith("line-"):
        finding["line"] = {
            "line-type": "2",
            "line-bool": True,
            "line-low": 0,
            "line-high": 200,
        }[attack]
    elif attack.startswith("column-"):
        finding["column"] = {
            "column-type": "4",
            "column-bool": True,
            "column-low": 0,
            "column-high": 200,
        }[attack]
    else:
        finding["message"] = {
            "message-type": None,
            "message-empty": " ",
            "message-null": "unbound\0text",
        }[attack]
    with pytest.raises(ValueError):
        rust_documentation_findings(json.dumps(report).encode(), sources, root)


@pytest.mark.parametrize("source_change", ("digest", "readback", "absent"))
def test_native_evidence_requires_actual_current_source_bytes(
    rust_reader: Path, tmp_path: Path, source_change: str
) -> None:
    """Bind actual native declarations to both retained digests and original readback."""
    relative = "original.rs"
    source = tmp_path / relative
    source.write_text("//! Original module.\nfn value() {}\n")
    native = subprocess.run(
        [str(rust_reader), "--source", relative],
        cwd=tmp_path,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert native.returncode == 1 and not native.stderr
    sources = {relative: hashlib.sha256(source.read_bytes()).hexdigest()}
    if source_change == "digest":
        sources[relative] = hashlib.sha256(b"unbound bytes").hexdigest()
    elif source_change == "readback":
        source.write_text(source.read_text() + "\n")
    else:
        source.unlink()
    with pytest.raises((ValueError, OSError)):
        rust_documentation_findings(native.stdout, sources, tmp_path)
