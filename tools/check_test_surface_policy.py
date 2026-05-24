#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Enforce module-specific test-surface naming policy.

The guard rejects generic bucket-style test files that obscure module ownership.
It intentionally checks path tokens, not substrings, so domain terms such as
``grounding`` are not misclassified as the forbidden ``round`` token.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

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


def _path_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for part in path.parts:
        stem = Path(part).stem.lower()
        tokens.update(token for token in re.split(r"[^a-z0-9]+", stem) if token)
    return tokens


def _normalised_path(path: Path) -> str:
    return "/".join(part.lower() for part in path.parts)


def find_forbidden_test_surfaces(root: Path) -> list[tuple[Path, str]]:
    """Return test paths whose names are structured as generic test buckets."""
    tests_root = root / "tests"
    if not tests_root.exists():
        return []

    offenders: list[tuple[Path, str]] = []
    for path in sorted(tests_root.rglob("test*.py")):
        relative = path.relative_to(root)
        tokens = _path_tokens(relative)
        for token in sorted(tokens & FORBIDDEN_TEST_STRUCTURE_TOKENS):
            offenders.append((relative, token))
        normalised = _normalised_path(relative)
        for token in sorted(FORBIDDEN_COMPOUND_TOKENS):
            if token in normalised:
                offenders.append((relative, token))
    return offenders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reject bucket-style test file names.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to scan.",
    )
    args = parser.parse_args(argv)

    offenders = find_forbidden_test_surfaces(args.root.resolve())
    if offenders:
        print(
            "Forbidden bucket-style test file names detected. "
            "Use module-specific test files instead:",
            file=sys.stderr,
        )
        for path, token in offenders:
            print(f"  {path}: token '{token}'", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
