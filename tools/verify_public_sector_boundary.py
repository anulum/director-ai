# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Public sector boundary verifier

"""Verify public files do not expose proprietary sector-pack identifiers."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PUBLIC_SCAN_ROOTS = (
    "README.md",
    "docs-site",
    "docs/_generated",
    "examples",
    "mkdocs.yml",
    "schemas",
    "src",
    "tests",
    "tools",
)

EXCLUDED_PATHS = frozenset(
    {
        "tests/test_precommit_exposure_guard.py",
        "tools/verify_public_sector_boundary.py",
    }
)

ALLOWED_PACK_MODULES = frozenset(
    {
        "src/director_ai/core/customer_model_factory/evidence_pack.py",
    }
)

ALLOWED_METADATA_SCHEMAS = frozenset(
    {
        "schemas/customer-model-factory-sector-metadata.schema.json",
    }
)

TEXT_SUFFIXES = frozenset(
    {
        "",
        ".cfg",
        ".ini",
        ".json",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)

BOUNDARY_TOKENS = (
    "bank" + "-alpha",
    "bank" + "_alpha",
    "customer-model-factory-" + "bank" + "ing",
    "bank" + "ing_pack",
    "Bank" + "ingRegulationMapping",
    "BANK" + "ING_",
    "build_" + "bank" + "ing",
    "validate_" + "bank" + "ing",
    "retail_" + "bank" + "ing",
    "private_" + "bank" + "ing",
    "corporate_" + "bank" + "ing",
    "financial_" + "advice_boundary",
    "kyc" + "_aml",
    "fees_" + "rates_terms",
)

BOUNDARY_PATTERN = re.compile("|".join(re.escape(token) for token in BOUNDARY_TOKENS))


@dataclass(frozen=True)
class BoundaryFinding:
    """One proprietary sector-boundary exposure finding."""

    path: str
    line_number: int
    token: str

    def format(self) -> str:
        """Return a stable human-readable finding line."""

        return f"{self.path}:{self.line_number}: {self.token}"


def main(argv: list[str] | None = None) -> int:
    """Run the public sector boundary verifier."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--staged",
        action="store_true",
        help="scan staged public additions instead of tracked public files",
    )
    args = parser.parse_args(argv)

    findings = (
        evaluate_staged_additions(args.root)
        if args.staged
        else evaluate_public_files(args.root)
    )
    for finding in findings:
        print(finding.format(), file=sys.stderr)
    return 0 if not findings else 1


def evaluate_public_files(root: Path) -> tuple[BoundaryFinding, ...]:
    """Return findings for all tracked public files under the scan roots."""

    root = root.resolve()
    findings: list[BoundaryFinding] = []
    for relative_path in _tracked_public_files(root):
        if relative_path in EXCLUDED_PATHS:
            continue
        path = root / relative_path
        if not path.is_file():
            continue
        findings.extend(_path_boundary_findings(relative_path))
        if path.suffix not in TEXT_SUFFIXES or not path.is_file():
            continue
        findings.extend(_scan_text(relative_path, path.read_text(encoding="utf-8")))
    return tuple(findings)


def evaluate_staged_additions(root: Path) -> tuple[BoundaryFinding, ...]:
    """Return findings for staged public addition lines."""

    root = root.resolve()
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--cached", "-U0", "--no-ext-diff", "--"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or "git diff --cached failed")

    findings: list[BoundaryFinding] = []
    current_path = ""
    current_line = 0
    reported_path_findings: set[str] = set()
    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current_path = line.removeprefix("+++ b/")
            current_line = 0
            continue
        if line.startswith("@@ "):
            current_line = _added_start_line(line)
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        if (
            current_path
            and _public_path(current_path)
            and current_path not in EXCLUDED_PATHS
        ):
            if current_path not in reported_path_findings:
                findings.extend(_path_boundary_findings(current_path))
                reported_path_findings.add(current_path)
            findings.extend(_scan_line(current_path, current_line, line[1:]))
        current_line += 1
    return tuple(findings)


def _tracked_public_files(root: Path) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--", *PUBLIC_SCAN_ROOTS],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    return tuple(
        line for line in result.stdout.splitlines() if line and _public_path(line)
    )


def _public_path(path: str) -> bool:
    return not (
        path.startswith("docs/internal/")
        or path.startswith(".coordination/")
        or path.startswith("04_ARCANE_SAPIENCE/")
    )


def _scan_text(path: str, text: str) -> tuple[BoundaryFinding, ...]:
    findings: list[BoundaryFinding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        findings.extend(_scan_line(path, line_number, line))
    return tuple(findings)


def _path_boundary_findings(path: str) -> tuple[BoundaryFinding, ...]:
    if (
        path.startswith("src/director_ai/core/customer_model_factory/")
        and path.endswith("_pack.py")
        and path not in ALLOWED_PACK_MODULES
    ):
        return (
            BoundaryFinding(
                path=path,
                line_number=0,
                token="proprietary sector pack module path",
            ),
        )
    if (
        path.startswith("schemas/customer-model-factory-")
        and path.endswith("-metadata.schema.json")
        and path not in ALLOWED_METADATA_SCHEMAS
    ):
        return (
            BoundaryFinding(
                path=path,
                line_number=0,
                token="proprietary sector metadata schema path",
            ),
        )
    return ()


def _scan_line(path: str, line_number: int, line: str) -> tuple[BoundaryFinding, ...]:
    return tuple(
        BoundaryFinding(path=path, line_number=line_number, token=match.group(0))
        for match in BOUNDARY_PATTERN.finditer(line)
    )


def _added_start_line(hunk_header: str) -> int:
    match = re.search(r"\+(\d+)", hunk_header)
    if match is None:
        return 0
    return int(match.group(1))


if __name__ == "__main__":
    raise SystemExit(main())
