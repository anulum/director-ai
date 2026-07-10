#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Enforce justified-only ``pragma: no cover`` markers under ``src/``.

Every coverage exclusion must carry an inline justification on the same
line, separated by a dash (``# pragma: no cover — reason``), so the audit
trail travels with the marker. Bare markers hide untested code with no
recorded rationale; this gate rejects them. Redundant markers on lines the
global ``[tool.coverage.report] exclude_lines`` patterns already exclude
(``...`` stubs, ``if TYPE_CHECKING:``, ``__main__`` guards,
``except ImportError``) should simply be removed instead of annotated.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# A justification is any non-empty text after the marker, introduced by an
# em-dash, en-dash, or one/two ASCII hyphens.
_PRAGMA_RE = re.compile(r"#\s*pragma:\s*no\s*cover(?P<rest>[^\n]*)")
_JUSTIFIED_RE = re.compile(r"^\s*(?:—|–|--|-)\s*\S")


def find_bare_pragmas(root: Path) -> list[tuple[Path, int, str]]:
    """Return ``(path, line_number, line)`` for every unjustified marker."""
    offenders: list[tuple[Path, int, str]] = []
    for path in sorted((root / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = _PRAGMA_RE.search(line)
            if match is None:
                continue
            if not _JUSTIFIED_RE.match(match.group("rest")):
                offenders.append((path, line_number, line.strip()))
    return offenders


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: report bare markers and fail when any exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    offenders = find_bare_pragmas(args.root.resolve())
    if offenders:
        print("Unjustified 'pragma: no cover' markers (add '— reason'):")
        for path, line_number, line in offenders:
            print(f"  {path}:{line_number}: {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
