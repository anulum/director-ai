# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — strict native Julia quality report decoding
"""Validate complete native Julia framing and preserve source-linked findings."""

from __future__ import annotations

import re
from collections.abc import Iterator

from tools.quality_ratchet_contracts import Finding, repository_path, sha256_digest

JULIA_REPORT_VERSION = "julia-quality-v2"
JULIA_DECLARATIONS = frozenset(
    {
        "abstract",
        "argument",
        "constant",
        "field",
        "macro",
        "method",
        "model_evaluator",
        "model_factory",
        "module",
        "primitive",
        "struct",
    }
)


def _line(rows: Iterator[list[str]], width: int, label: str) -> list[str]:
    """Consume exactly one native row of the expected kind and width."""
    value = next(rows, [])
    if len(value) != width or value[0] != label:
        raise ValueError(f"Invalid or incomplete native Julia {label} row")
    return value


def _text(raw: str) -> str:
    """Decode canonical native hex and reject empty names or control characters."""
    if re.fullmatch(r"(?:[0-9a-f]{2})+", raw) is None:
        raise ValueError("Invalid native Julia hex text")
    value = bytes.fromhex(raw).decode("utf-8")
    if not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Invalid native Julia name")
    return value


def _count(raw: str, maximum: int) -> int:
    """Decode a bounded canonical integer from a native counter."""
    if re.fullmatch(r"0|[1-9][0-9]{0,8}", raw) is None:
        raise ValueError("Invalid native Julia counter")
    number = int(raw)
    if number > maximum:
        raise ValueError("Native Julia counter is outside the qualified scope")
    return number


def parse_julia_report(
    report: bytes,
    *,
    versions: tuple[str, ...],
    binary: str,
    project: str,
    manifest: str,
    sources: dict[str, str],
    line_counts: dict[str, int],
) -> tuple[Finding, ...]:
    """Qualify the native declaration report against independently measured inputs.

    Parameters
    ----------
    report : bytes
        Original native Julia standard output.
    versions : tuple of str
        Qualified Julia, ArgParse, Turing and DynamicPPL versions in that order.
    binary : str
        SHA-256 of the selected native Julia executable.
    project : str
        SHA-256 of the declared Julia project.
    manifest : str
        SHA-256 of the exact selected Julia manifest.
    sources : dict of str to str
        Complete sorted Git-visible source inventory mapped to byte digests.
    line_counts : dict of str to int
        Native source line bounds from the same source bytes.

    Returns
    -------
    tuple of Finding
        Every undocumented authored or generated declaration from the report.

    Raises
    ------
    ValueError
        If output is incomplete, a field is malformed or native identity differs.
    """
    if not report.endswith(b"\n") or b"\r" in report:
        raise ValueError("Invalid native Julia report framing")
    text = report.decode("utf-8").removesuffix("\n")
    row_count = text.count("\n") + 1
    rows = (line.split("\t") for line in text.split("\n"))
    header = _line(rows, 5, JULIA_REPORT_VERSION)
    if tuple(header[1:]) != versions:
        raise ValueError("Native Julia runtime or generator versions differ")
    if sha256_digest(_line(rows, 2, "binary")[1]) != binary:
        raise ValueError("Native Julia binary differs from the selected executable")
    if sha256_digest(_line(rows, 2, "project")[1]) != project:
        raise ValueError("Native Julia project differs from the qualified source")
    if sha256_digest(_line(rows, 2, "manifest")[1]) != manifest:
        raise ValueError("Native Julia manifest differs from the qualified source")

    findings: list[Finding] = []
    for path, digest in sorted(sources.items()):
        source = _line(rows, 4, "source")
        if (
            repository_path(_text(source[1])) != path
            or sha256_digest(source[2]) != digest
        ):
            raise ValueError("Native Julia source inventory or digest differs")
        count = _count(source[3], row_count)
        declarations: set[tuple[int, str, str]] = set()
        for _ in range(count):
            row = _line(rows, 6, "declaration")
            position = _count(row[2], line_counts[path])
            kind, name, documented = row[3], _text(row[4]), row[5]
            identity = (position, kind, name)
            if (
                row[1] != source[1]
                or position == 0
                or kind not in JULIA_DECLARATIONS
                or documented not in {"true", "false"}
                or identity in declarations
            ):
                raise ValueError("Invalid or duplicate native Julia declaration")
            declarations.add(identity)
            if documented == "false":
                findings.append(
                    Finding(
                        path,
                        f"julia-doc-{kind}",
                        position,
                        None,
                        f"Missing Julia documentation for {name}",
                    )
                )

    if (
        _line(rows, 2, "complete")[1] != str(len(sources))
        or next(rows, None) is not None
    ):
        raise ValueError(
            "Native Julia report did not finish exactly at the qualified scope"
        )
    return tuple(findings)
