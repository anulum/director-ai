#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Enforce module-specific test-surface policy.

The guard rejects generic bucket-style test files and unclassified mock-heavy
test surfaces. It intentionally checks path tokens, not substrings, so domain
terms such as ``grounding`` are not misclassified as the forbidden ``round``
token.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

FORBIDDEN_TEST_STRUCTURE_TOKENS = frozenset(
    {
        "batch",
        "coverage",
        "final",
        "misc",
        "push",
        "remaining",
        "round",
    }
)
FORBIDDEN_COMPOUND_TOKENS = frozenset({"new_modules"})
VALID_CLASSIFICATIONS = frozenset(
    {
        "approved-protocol-fake",
        "unit-guard-with-companion",
        "violation",
    }
)
SURFACE_MARKER_RE = re.compile(
    r"^\s*#\s*test-surface:\s*"
    r"(?P<classification>approved-protocol-fake|unit-guard-with-companion)\s*$",
    re.MULTILINE,
)
COMPANION_MARKER_RE = re.compile(
    r"^\s*#\s*real-surface-companion:\s*(?P<path>tests/[^\s#]+\.py)\s*$",
    re.MULTILINE,
)
MOCK_SURFACE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "unittest.mock",
        re.compile(
            r"\bfrom\s+unittest\s*\.\s*mock\b|\bimport\s+unittest\s*\.\s*mock\b"
        ),
    ),
    ("patch-call", re.compile(r"\bpatch(?:\s*\.\s*dict|\s*\.\s*object)?\s*\(")),
    ("sys.modules", re.compile(r"\bsys\s*\.\s*modules\b")),
    ("ModuleType", re.compile(r"\bModuleType\s*\(")),
    (
        "private-director-ai-import",
        re.compile(
            r"\bfrom\s+director_ai"
            r"(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*)*"
            r"\s+import\s+_"
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class SurfaceClassification:
    """Classified test-surface debt or approved fake metadata."""

    classification: str
    category: str

    def validate(self, path: str) -> str | None:
        """Return a validation error for invalid manifest metadata."""
        if self.classification not in VALID_CLASSIFICATIONS:
            return f"{path}: invalid classification '{self.classification}'"
        if not self.category.strip():
            return f"{path}: category must not be blank"
        return None


def _path_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for part in path.parts:
        stem = Path(part).stem.lower()
        tokens.update(token for token in re.split(r"[^a-z0-9]+", stem) if token)
    return tokens


def _normalised_path(path: Path) -> str:
    return "/".join(part.lower() for part in path.parts)


def _default_classifications() -> dict[str, SurfaceClassification]:
    return {
        path: SurfaceClassification(
            classification=classification,
            category=category,
        )
        for path, (
            classification,
            category,
        ) in KNOWN_TEST_SURFACE_CLASSIFICATIONS.items()
    }


def _relative_test_paths(root: Path) -> list[Path]:
    tests_root = root / "tests"
    if not tests_root.exists():
        return []
    return [path.relative_to(root) for path in sorted(tests_root.rglob("test*.py"))]


def _read_text(root: Path, relative: Path) -> str:
    return (root / relative).read_text(encoding="utf-8")


def _code_without_literals(text: str) -> str:
    tokens: list[str] = []
    stream = io.StringIO(text)
    try:
        for token in tokenize.generate_tokens(stream.readline):
            if token.type not in {tokenize.COMMENT, tokenize.STRING} and token.string:
                tokens.append(token.string)
    except tokenize.TokenError:
        return text
    return " ".join(tokens)


def _mock_surface_reason(text: str) -> str | None:
    for reason, pattern in MOCK_SURFACE_PATTERNS:
        if pattern.search(text):
            return reason
    return None


def _inline_surface_classification(root: Path, text: str) -> str | None:
    marker = SURFACE_MARKER_RE.search(text)
    if marker is None:
        return None
    companion = COMPANION_MARKER_RE.search(text)
    if companion is None:
        return "missing real-surface-companion marker"
    companion_path = root / companion.group("path")
    if not companion_path.is_file():
        return f"missing companion {companion.group('path')}"
    return None


def validate_classifications(
    classifications: dict[str, SurfaceClassification],
) -> list[str]:
    """Return invalid classification manifest entries."""
    errors: list[str] = []
    for path, classification in sorted(classifications.items()):
        error = classification.validate(path)
        if error is not None:
            errors.append(error)
    return errors


def find_forbidden_test_surfaces(root: Path) -> list[tuple[Path, str]]:
    """Return test paths whose names are structured as generic test buckets."""
    offenders: list[tuple[Path, str]] = []
    for relative in _relative_test_paths(root):
        tokens = _path_tokens(relative)
        for token in sorted(tokens & FORBIDDEN_TEST_STRUCTURE_TOKENS):
            offenders.append((relative, token))
        normalised = _normalised_path(relative)
        for token in sorted(FORBIDDEN_COMPOUND_TOKENS):
            if token in normalised:
                offenders.append((relative, token))
    return offenders


def find_unclassified_mock_surfaces(
    root: Path,
    classifications: dict[str, SurfaceClassification] | None = None,
) -> list[tuple[Path, str]]:
    """Return mock/sys.modules test files without a governed classification."""
    root = root.resolve()
    manifest = (
        _default_classifications() if classifications is None else classifications
    )
    offenders: list[tuple[Path, str]] = []
    for relative in _relative_test_paths(root):
        text = _read_text(root, relative)
        reason = _mock_surface_reason(_code_without_literals(text))
        if reason is None:
            continue
        normalised = _normalised_path(relative)
        if normalised in manifest:
            continue
        inline_error = _inline_surface_classification(root, text)
        if inline_error is None and SURFACE_MARKER_RE.search(text) is not None:
            continue
        offenders.append((relative, inline_error or reason))
    return offenders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reject bucket-style and unclassified mock test surfaces.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to scan.",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    classification_errors = validate_classifications(_default_classifications())
    if classification_errors:
        print("Invalid test-surface classification manifest:", file=sys.stderr)
        for error in classification_errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    forbidden = find_forbidden_test_surfaces(root)
    unclassified = find_unclassified_mock_surfaces(root)
    if forbidden:
        print(
            "Forbidden bucket-style test file names detected. "
            "Use module-specific test files instead:",
            file=sys.stderr,
        )
        for path, token in forbidden:
            print(f"  {path}: token '{token}'", file=sys.stderr)
    if unclassified:
        print(
            "Unclassified mock/sys.modules test surfaces detected. "
            "Add a manifest classification or an approved protocol-fake marker "
            "with a real-surface companion:",
            file=sys.stderr,
        )
        for path, reason in unclassified:
            print(f"  {path}: {reason}", file=sys.stderr)
    if forbidden or unclassified:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
