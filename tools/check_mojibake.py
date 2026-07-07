#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Reject multiply-encoded mojibake in the Python source tree.

An earlier editor/encoding mishap left cp1250/cp1252 mojibake -- corrupted em
dashes, box-drawing rules and accented characters -- across several modules
(fixed under WCH-5). This guard scans the tracked Python sources for the
specific artefact sequences that never occur in the project's legitimate
UTF-8 text (SPDX headers, box-drawing separators, arrows, en/de/sk prose), so
the corruption cannot silently regress.

The signatures are written as Unicode escapes rather than literal bytes so this
module -- and its tests -- stay pure ASCII and never flag themselves.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Sequences produced when UTF-8 text is misdecoded as cp1252/cp1250 and
# re-encoded (single- or multi-round). Each is validated to produce zero hits
# against the clean tree, so a match is always genuine corruption. Written as
# escapes to keep this module pure ASCII (see module docstring).
MOJIBAKE_SIGNATURES: tuple[str, ...] = (
    "\u00c3\u00a9",  # e-acute
    "\u00c3\u00a1",  # a-acute
    "\u00c3\u00a4",  # a-uml
    "\u00c3\u00b6",  # o-uml
    "\u00c3\u00bc",  # u-uml
    "\u00c3\u009f",  # sharp-s
    "\u00c3\u00a8",  # e-grave
    "\u00c3\u00af",  # i-uml
    "\u00c3\u00b1",  # n-tilde
    "\u00c3\u00a7",  # c-cedilla
    "\u00e2\u20ac",  # dash/quote prefix
    "\u00e2\u201a\u00ac",  # euro artefact
    "\u0102\u02d8",  # multi-round cp1250
    "\u00c5\u00a1",  # s-caron
    "\u00c5\u00be",  # z-caron
    "\u00c5\u0088",  # n-caron
    "\u00c4\u008d",  # c-caron
    "\u00c4\u009b",  # e-caron
    "\u00c5\u0099",  # r-caron
    "\u00c2\u00a0",  # nbsp artefact
    "\u00c2\u00b0",  # degree
    "\u00c2\u00b5",  # micro
    "\u00c3\u201a",  # double-encoded latin-1
)

SCAN_DIRS: tuple[str, ...] = ("src", "tools", "tests")


def find_mojibake(root: Path) -> list[tuple[Path, int, str]]:
    """Return ``(path, line-number, signature)`` for every mojibake hit under root.

    Only the first signature on a line is reported so one corrupted line yields
    one finding. Files that are not valid UTF-8 are skipped (a separate concern).
    """
    hits: list[tuple[Path, int, str]] = []
    for rel in SCAN_DIRS:
        base = root / rel
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):  # pragma: no cover - defensive
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                match = next((s for s in MOJIBAKE_SIGNATURES if s in line), None)
                if match is not None:
                    hits.append((path, lineno, match))
    return hits


def main(argv: list[str] | None = None) -> int:
    """CLI entry: print each finding and exit non-zero when mojibake is present."""
    parser = argparse.ArgumentParser(
        description="Reject cp1250/cp1252 mojibake in the Python source tree.",
    )
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    args = parser.parse_args(argv)

    hits = find_mojibake(Path(args.root))
    for path, lineno, sig in hits:
        print(f"{path}:{lineno}: mojibake artefact {sig!r}")
    if hits:
        print(
            f"\n{len(hits)} mojibake artefact(s) found; "
            "reconstruct to the intended UTF-8 characters (WCH-5).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
