# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native Rust documentation evidence admission
"""Validate complete native Rust output against original source bytes."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_contracts import decode_quality_json
from tools.quality_ratchet_inventory import source_path


def rust_documentation_findings(
    original: bytes, sources: dict[str, str], root: Path
) -> list[dict[str, object]]:
    """Admit complete original native rows, bytes and actual Unicode source positions.

    Parameters
    ----------
    original : bytes
        Original complete native reader JSON, before any transport corruption.
    sources : dict[str, str]
        Complete original source inventory mapped to SHA-256 byte digests.
    root : Path
        Actual source directory, read independently to qualify every byte.

    Returns
    -------
    list[dict[str, object]]
        Individual complete native findings preserving original coordinates.

    Raises
    ------
    ValueError, OSError
        If source identity, native schema, grammar qualification or diagnostics
        do not match the actual complete original input.
    """
    report = object_mapping(decode_quality_json(original))
    rows = report.get("files")
    if (
        set(report) != {"schema", "files"}
        or report["schema"] != "quality-ratchet.rust-ast.v1"
        or not isinstance(rows, list)
    ):
        raise ValueError("Invalid original native Rust source report")
    actual_sources: dict[str, str] = {}
    findings: list[dict[str, object]] = []
    for raw in rows:
        row = object_mapping(raw)
        relative, text, refused, diagnostics = (
            row.get("path"),
            row.get("source"),
            row.get("refusals"),
            row.get("findings"),
        )
        if (
            set(row) != {"path", "source", "findings", "refusals"}
            or not isinstance(relative, str)
            or relative not in sources
            or relative in actual_sources
            or not isinstance(text, str)
            or refused != []
            or not isinstance(diagnostics, list)
        ):
            raise ValueError("Native Rust source or macro boundary is not qualified")
        data = text.encode("utf-8")
        if (
            hashlib.sha256(data).hexdigest() != sources[relative]
            or source_path(root, relative).read_bytes() != data
        ):
            raise ValueError("Native Rust report differs from its original source")
        actual_sources[relative] = sources[relative]
        lines = text.split("\n")
        for diagnostic in diagnostics:
            finding = object_mapping(diagnostic)
            line, column, rule, message = (
                finding.get("line"),
                finding.get("column"),
                finding.get("rule"),
                finding.get("message"),
            )
            if (
                set(finding) != {"rule", "line", "column", "message"}
                or rule
                not in {
                    "module-documentation",
                    "declaration-documentation",
                    "source-directive",
                }
                or type(line) is not int
                or type(column) is not int
                or not 1 <= line <= len(lines)
                or not 1 <= column <= len(lines[line - 1]) + 1
                or not isinstance(message, str)
                or not message.strip()
                or "\0" in message
            ):
                raise ValueError("Invalid original native Rust diagnostic location")
            findings.append({"path": relative, **finding, "cell": None})
    if actual_sources != sources:
        raise ValueError("Native Rust report omits original source inputs")
    return findings
