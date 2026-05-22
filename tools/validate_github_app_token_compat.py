# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — GitHub App token compatibility validator

"""Validate repository compatibility with GitHub App stateless tokens.

The validator enforces that tracked source/workflow/docs text does not
contain brittle assumptions such as fixed 40-character installation
tokens or narrow ``ghs_`` regexes incompatible with JWT-format tokens.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCAN_DIRS = (
    "src",
    "tests",
    "tools",
    "docs",
    ".github/workflows",
)

EXCLUDE_PATHS = {
    Path("tools/validate_github_app_token_compat.py"),
    Path("tests/test_validate_github_app_token_compat.py"),
}

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".yml",
    ".yaml",
    ".toml",
    ".json",
    ".ini",
    ".cfg",
    ".sh",
}

# Patterns that indicate brittle assumptions for GitHub App installation
# tokens (ghs_...) and should be removed.
FAIL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"ghs_\[[^\]]+\]\{\s*40\s*\}"),
        "Fixed-length 40-char ghs token regex is incompatible with stateless format.",
    ),
    (
        re.compile(r"ghs_[A-Za-z0-9]{40}"),
        "Fixed-length 40-char ghs token regex is incompatible with stateless format.",
    ),
    (
        re.compile(r"ghs_[A-Za-z0-9_-]{40}"),
        "Fixed-length 40-char ghs token regex is incompatible with stateless format.",
    ),
    (
        re.compile(r"installation token.{0,40}40 characters", re.IGNORECASE),
        "Text states installation token length is fixed to 40 characters.",
    ),
    (
        re.compile(r"len\([^)]*token[^)]*\)\s*==\s*40"),
        "Token validation hardcodes exact length 40.",
    ),
)


def _iter_files() -> list[Path]:
    files: list[Path] = []
    for rel_dir in SCAN_DIRS:
        root = ROOT / rel_dir
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if rel in EXCLUDE_PATHS:
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            files.append(path)
    return files


def validate() -> list[str]:
    violations: list[str] = []
    for path in _iter_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for regex, reason in FAIL_PATTERNS:
            for match in regex.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                rel = path.relative_to(ROOT)
                violations.append(f"{rel}:{line}: {reason}")
    return violations


def main() -> int:
    violations = validate()
    if violations:
        print("GitHub App token compatibility check failed:")
        for item in violations:
            print(f"- {item}")
        return 1
    print("GitHub App token compatibility check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
